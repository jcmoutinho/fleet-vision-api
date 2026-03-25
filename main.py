"""
Fleet Vision API — Análise de imagens de checklist de frotas.
Backend: Google Gemini Flash (primário) → LLaVA via Ollama (fallback local)
"""

import base64
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from database import (
    init_db, close_db, save_analysis, get_analysis,
    list_analyses, get_analysis_image, cleanup_expired, get_stats,
    find_duplicate,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fleet-vision-api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
OLLAMA_API_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434") + "/api/chat"
OLLAMA_MODEL = "llava:7b"

GEMINI_API_KEY: str = ""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_ANALYZE = """Você é um especialista em inspeção técnica de veículos de frota (caminhões, vans, carros utilitários e de passeio).

Sua tarefa: analisar a imagem enviada de um item do checklist de manutenção e comparar com a descrição fornecida pelo inspetor.

Regras:
1. Descreva objetivamente o que você vê na imagem relacionado ao item inspecionado.
2. Identifique detalhes específicos conforme o tipo de item:
   - **Placa do veículo:** se a placa estiver visível na imagem, leia e informe no campo `specific_data.plate`. Se não estiver visível, omita o campo.
   - **Pneus:** marca, modelo (se visível), tamanho/especificação, profundidade estimada da banda de rodagem, desgaste irregular, bolhas, cortes, calibragem visual.
   - **Lataria/carroceria:** amassados, arranhões, oxidação/ferrugem, pintura descascada, severidade e localização.
   - **Vidros:** trincas, lascas, adesivos obrigatórios, limpeza.
   - **Luzes/faróis:** funcionamento aparente, lentes trincadas ou embaçadas, cor correta.
   - **Fluidos:** nível visível, vazamentos, cor do fluido.
   - **Outros:** qualquer anomalia relevante para segurança ou manutenção.
3. **ANTES DE TUDO:** avalie a qualidade da imagem. Se a imagem estiver desfocada, muito escura, muito clara, com má resolução, cortada de forma que impeça a análise, ou de qualquer forma inadequada para inspeção técnica, retorne IMEDIATAMENTE o JSON abaixo e não tente analisar:
   ```json
   {
     "image_quality": "inadequada",
     "image_quality_reason": "descreva o problema: desfocada / escura / resolução baixa / cortada / etc.",
     "ai_analysis": "Imagem inadequada para análise. [motivo]",
     "description_matches": null,
     "confidence": "baixa",
     "additional_findings": [],
     "severity": "atenção",
     "specific_data": {},
     "recommendations": ["Refazer a foto com melhor iluminação e foco antes de prosseguir com a inspeção."]
   }
   ```
4. Se a imagem for adequada, prossiga com a análise normal e inclua `"image_quality": "adequada"` no JSON.
5. Compare sua análise com a descrição do inspetor e determine se a descrição confere com o que está na imagem.
6. Classifique a severidade: "ok" (sem problemas), "atenção" (requer monitoramento ou manutenção preventiva), "crítico" (risco à segurança, requer ação imediata).
7. Forneça recomendações práticas.

Responda EXCLUSIVAMENTE em JSON válido, sem markdown, sem blocos de código, com esta estrutura:
{
  "ai_analysis": "descrição detalhada do que você viu",
  "description_matches": true ou false,
  "confidence": "alta", "média" ou "baixa",
  "additional_findings": ["detalhe 1", "detalhe 2"],
  "severity": "ok", "atenção" ou "crítico",
  "specific_data": { campos relevantes ao tipo de item },
  "recommendations": ["recomendação 1", "recomendação 2"]
}

Linguagem: português do Brasil, técnica mas acessível."""

SYSTEM_PROMPT_EVOLUTION = """Você é um especialista em inspeção técnica de veículos de frota.

Sua tarefa: analisar uma sequência cronológica de imagens do MESMO item e identificar a evolução do estado ao longo do tempo.

Se a placa do veículo estiver visível em qualquer imagem, inclua-a no campo `specific_data.plate` da análise.

Responda EXCLUSIVAMENTE em JSON válido, sem markdown, sem blocos de código, com esta estrutura:
{
  "per_image_analysis": [
    { "image_index": 1, "date": "data ou null", "condition": "descrição", "severity": "ok/atenção/crítico" }
  ],
  "evolution_summary": "resumo da evolução",
  "trend": "melhorando/estável/deteriorando",
  "degradation_rate": "lenta/moderada/rápida ou null",
  "estimated_action_needed": "estimativa de quando ação será necessária",
  "recommendations": ["recomendação 1"]
}

Linguagem: português do Brasil, técnica mas acessível."""

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GEMINI_API_KEY
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY não configurada — usará apenas fallback LLaVA local.")
    await init_db()
    await cleanup_expired()
    logger.info("Fleet Vision API iniciada. Backend: Gemini Flash → LLaVA (fallback)")
    yield
    await close_db()
    logger.info("Fleet Vision API encerrada.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fleet Vision API",
    description="Análise de imagens de checklist de frotas — Gemini Flash + LLaVA fallback",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (UI de teste)
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

@app.get("/")
async def root():
    index = _os.path.join(_static_dir, "index.html")
    if _os.path.isfile(index):
        return FileResponse(index)
    return {"service": "fleet-vision-api", "docs": "/docs"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _read_and_validate_image(image: UploadFile) -> tuple[str, str]:
    content_type = image.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de imagem não suportado: {content_type}. Use jpg, png ou webp.",
        )
    data = await image.read()
    if len(data) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Imagem excede o limite de 10MB.")
    return base64.b64encode(data).decode("utf-8"), content_type


def _strip_json(content: str) -> dict:
    """Strip markdown fences and parse JSON. Handles truncated responses."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Tenta extrair o JSON entre { } mesmo que truncado
        start = content.find("{")
        if start == -1:
            raise
        # Conta chaves para encontrar onde o JSON termina ou tenta fechar
        depth = 0
        end = start
        for i, ch in enumerate(content[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if depth == 0:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        # Última tentativa: JSON truncado — retorna erro de qualidade
        logger.warning("JSON truncado da IA, retornando erro de qualidade de imagem.")
        return {
            "image_quality": "inadequada",
            "image_quality_reason": "A IA não conseguiu processar a imagem completamente. Tente novamente ou use uma imagem com melhor qualidade.",
            "ai_analysis": "Erro ao processar resposta da IA. Tente novamente.",
            "description_matches": None,
            "confidence": "baixa",
            "additional_findings": [],
            "severity": "atenção",
            "specific_data": {},
            "recommendations": ["Tente novamente. Se o erro persistir, refaça a foto com melhor qualidade."],
        }


GEMINI_PRICE_INPUT_PER_1M = 0.15   # USD por 1M tokens de entrada (gemini-2.5-flash)
GEMINI_PRICE_OUTPUT_PER_1M = 0.60  # USD por 1M tokens de saída

async def _call_gemini(prompt: str, images: list[tuple[str, str]], request_id: str) -> tuple[dict, dict]:
    """Call Gemini Flash with vision. Returns (parsed dict, usage_info) or raises."""
    parts = [{"text": prompt}]
    for b64, mime in images:
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})

    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}}
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, json=payload)

    elapsed = time.monotonic() - t0
    logger.info("Gemini response | request_id=%s | status=%s | time=%.2fs", request_id, resp.status_code, elapsed)

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")

    body = resp.json()
    content = body["candidates"][0]["content"]["parts"][0]["text"]

    # Extrai uso de tokens
    usage_meta = body.get("usageMetadata", {})
    input_tokens = usage_meta.get("promptTokenCount", 0)
    output_tokens = usage_meta.get("candidatesTokenCount", 0)
    total_tokens = usage_meta.get("totalTokenCount", input_tokens + output_tokens)
    cost_usd = (input_tokens / 1_000_000 * GEMINI_PRICE_INPUT_PER_1M) + (output_tokens / 1_000_000 * GEMINI_PRICE_OUTPUT_PER_1M)

    usage_info = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost_usd, 6),
        "estimated_cost_brl": round(cost_usd * 5.0, 5),  # cotação aproximada
    }
    logger.info("Gemini usage | request_id=%s | tokens=%d | cost_usd=%.6f", request_id, total_tokens, cost_usd)

    return _strip_json(content), usage_info


async def _call_ollama_llava(prompt: str, images: list[tuple[str, str]], request_id: str) -> tuple[dict, dict]:
    """Call LLaVA via Ollama as fallback. Returns parsed dict or raises."""
    images_b64 = [b64 for b64, _ in images]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Você é um especialista em inspeção de veículos. Responda em JSON válido, em português do Brasil."},
            {"role": "user", "content": prompt, "images": images_b64},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=300) as client:  # LLaVA pode demorar sem GPU
        resp = await client.post(OLLAMA_API_URL, json=payload)

    elapsed = time.monotonic() - t0
    logger.info("LLaVA response | request_id=%s | status=%s | time=%.2fs", request_id, resp.status_code, elapsed)

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")

    body = resp.json()
    content = body["message"]["content"]
    eval_count = body.get("eval_count", 0)
    prompt_eval_count = body.get("prompt_eval_count", 0)
    usage_info = {
        "input_tokens": prompt_eval_count,
        "output_tokens": eval_count,
        "total_tokens": prompt_eval_count + eval_count,
        "estimated_cost_usd": 0.0,
        "estimated_cost_brl": 0.0,
    }
    return _strip_json(content), usage_info


async def _analyze_with_fallback(prompt: str, images: list[tuple[str, str]], request_id: str) -> tuple[dict, str, dict]:
    """Try Gemini first, fall back to LLaVA. Returns (result, backend_used, usage_info)."""
    if GEMINI_API_KEY:
        try:
            result, usage = await _call_gemini(prompt, images, request_id)
            return result, "gemini-2.5-flash", usage
        except Exception as e:
            logger.warning("Gemini falhou, tentando LLaVA | request_id=%s | error=%s", request_id, str(e))

    try:
        result, usage = await _call_ollama_llava(prompt, images, request_id)
        return result, "llava-7b-local", usage
    except Exception as e:
        logger.error("LLaVA também falhou | request_id=%s | error=%s", request_id, str(e))
        raise HTTPException(status_code=502, detail="Nenhum backend de IA disponível no momento.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(OLLAMA_API_URL.replace("/api/chat", "/"))
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "service": "fleet-vision-api",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "backends": {
            "gemini_flash": bool(GEMINI_API_KEY),
            "llava_local": ollama_ok,
        },
    }


@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze(
    request: Request,
    item_name: str = Form(...),
    user_description: str = Form(...),
    image: UploadFile = File(...),
    force: bool = Form(False),
    vehicle_id: Optional[str] = Form(None),
):
    request_id = str(uuid.uuid4())[:8]
    logger.info("POST /analyze | request_id=%s | item=%s | force=%s", request_id, item_name, force)

    b64, media_type = await _read_and_validate_image(image)
    image_size_bytes = len(b64) * 3 // 4

    # Verifica duplicata (a menos que force=True)
    if not force:
        duplicate = await find_duplicate(b64, item_name)
        if duplicate:
            logger.info("Imagem duplicada encontrada | request_id_original=%s", duplicate["request_id"])
            # Retorna resultado cacheado
            cached = await get_analysis(duplicate["request_id"])
            return {
                **{k: v for k, v in (cached or {}).items() if k != "ai_result"},
                **(cached.get("ai_result", {}) if cached else {}),
                "item": item_name,
                "user_description": user_description,
                "cached": True,
                "original_request_id": duplicate["request_id"],
                "cache_note": "Esta imagem já foi analisada. Use force=true para forçar nova análise.",
                "request_id": request_id,
            }

    t0 = time.monotonic()
    vehicle_prefix = f"Veículo/Frota: {vehicle_id}\n" if vehicle_id else ""
    prompt = (
        f"{SYSTEM_PROMPT_ANALYZE}\n\n"
        f"{vehicle_prefix}"
        f"Item do checklist: {item_name}\n"
        f"Descrição do inspetor: {user_description}\n\n"
        "Analise a imagem e responda em JSON conforme instruído."
    )

    ai_result, backend, usage = await _analyze_with_fallback(prompt, [(b64, media_type)], request_id)
    processing_seconds = round(time.monotonic() - t0, 2)

    try:
        await save_analysis(
            request_id, item_name, user_description, b64, media_type,
            ai_result, backend, usage, processing_seconds, vehicle_id,
        )
    except Exception as e:
        logger.error("Falha ao salvar análise no banco | request_id=%s | error=%s", request_id, str(e))

    return {
        "item": item_name,
        "user_description": user_description,
        "description_matches": ai_result.get("description_matches"),
        "confidence": ai_result.get("confidence", "baixa"),
        "ai_analysis": ai_result.get("ai_analysis", ""),
        "additional_findings": ai_result.get("additional_findings", []),
        "severity": ai_result.get("severity", "atenção"),
        "specific_data": ai_result.get("specific_data", {}),
        "recommendations": ai_result.get("recommendations", []),
        "backend_used": backend,
        "usage": usage,
        "processing_seconds": processing_seconds,
        "image_size_bytes": image_size_bytes,
        "image_quality": ai_result.get("image_quality", "adequada"),
        "image_quality_reason": ai_result.get("image_quality_reason", ""),
        "vehicle_id": vehicle_id or "",
        "cached": False,
        "request_id": request_id,
    }


@app.post("/analyze/evolution")
@limiter.limit("10/minute")
async def analyze_evolution(
    request: Request,
    item_name: str = Form(...),
    images: list[UploadFile] = File(...),
    dates: Optional[list[str]] = Form(None),
):
    request_id = str(uuid.uuid4())[:8]
    logger.info("POST /analyze/evolution | request_id=%s | item=%s | images=%d", request_id, item_name, len(images))

    if len(images) < 2:
        raise HTTPException(status_code=400, detail="Envie pelo menos 2 imagens para análise de evolução.")
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Máximo de 10 imagens por requisição.")

    date_list = list(dates or [])
    while len(date_list) < len(images):
        date_list.append(None)

    imgs = []
    date_labels = []
    for idx, img in enumerate(images):
        b64, media_type = await _read_and_validate_image(img)
        imgs.append((b64, media_type))
        date_labels.append(date_list[idx] or f"Imagem {idx + 1}")

    dates_str = "\n".join([f"- {label}" for label in date_labels])
    prompt = (
        f"{SYSTEM_PROMPT_EVOLUTION}\n\n"
        f"Item do checklist: {item_name}\n"
        f"Sequência temporal ({len(images)} imagens):\n{dates_str}\n\n"
        "Analise a evolução e responda em JSON conforme instruído."
    )

    t0 = time.monotonic()
    ai_result, backend, usage = await _analyze_with_fallback(prompt, imgs, request_id)
    processing_seconds = round(time.monotonic() - t0, 2)

    try:
        first_b64, first_mime = imgs[0]
        await save_analysis(
            request_id, item_name, f"[Evolução] {len(images)} imagens",
            first_b64, first_mime, ai_result, backend, usage, processing_seconds,
        )
    except Exception as e:
        logger.error("Falha ao salvar evolução no banco | request_id=%s | error=%s", request_id, str(e))

    return {
        "item": item_name,
        "total_images": len(images),
        "per_image_analysis": ai_result.get("per_image_analysis", []),
        "evolution_summary": ai_result.get("evolution_summary", ""),
        "trend": ai_result.get("trend", ""),
        "degradation_rate": ai_result.get("degradation_rate"),
        "estimated_action_needed": ai_result.get("estimated_action_needed", ""),
        "recommendations": ai_result.get("recommendations", []),
        "backend_used": backend,
        "usage": usage,
        "processing_seconds": processing_seconds,
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# History & Stats endpoints
# ---------------------------------------------------------------------------
@app.get("/history")
async def history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    severity: Optional[str] = Query(None),
    item_name: Optional[str] = Query(None),
    vehicle_id: Optional[str] = Query(None),
):
    result = await list_analyses(page=page, per_page=per_page, severity=severity, item_name=item_name, vehicle_id=vehicle_id)
    return result


@app.get("/history/{request_id}")
async def history_detail(request_id: str):
    analysis = await get_analysis(request_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Análise não encontrada.")
    return analysis


@app.get("/history/{request_id}/image")
async def history_image(request_id: str):
    result = await get_analysis_image(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Imagem não encontrada.")
    image_b64, image_mime = result
    image_bytes = base64.b64decode(image_b64)
    return Response(content=image_bytes, media_type=image_mime)


@app.get("/stats")
async def stats():
    return await get_stats()


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", str(exc), exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Erro interno do servidor."})
