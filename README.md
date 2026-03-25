# Fleet Vision API 🚛🔍

API de análise de imagens para checklist de frotas, powered by GPT-4o.

Recebe fotos de itens do checklist (pneus, lataria, vidros, luzes, etc.) junto com a descrição do inspetor e retorna uma análise estruturada comparando o que foi descrito com o que está na imagem.

## Funcionalidades

- **Análise de item único** — compara descrição do inspetor com a imagem
- **Análise de evolução** — compara múltiplas fotos ao longo do tempo
- **Rate limiting** — 10 requisições/min por IP
- **Suporte a imagens** — JPG, PNG, WebP (max 10MB)

## Quick Start

### 1. Configurar

```bash
cp .env.example .env
# Edite .env e coloque sua OPENAI_API_KEY
```

### 2. Rodar com Docker

```bash
docker compose up -d --build
```

A API estará disponível em `http://localhost:8100`.

### 3. Rodar localmente (sem Docker)

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-proj-..."
uvicorn main:app --host 0.0.0.0 --port 8100
```

## Endpoints

### GET /health

```bash
curl http://localhost:8100/health
```

### POST /analyze

Análise de um item do checklist com uma foto.

```bash
curl -X POST http://localhost:8100/analyze \
  -F "item_name=Pneu dianteiro esquerdo" \
  -F "user_description=Pneu em bom estado, sem desgaste aparente" \
  -F "image=@foto_pneu.jpg"
```

**Resposta:**

```json
{
  "item": "Pneu dianteiro esquerdo",
  "user_description": "Pneu em bom estado, sem desgaste aparente",
  "description_matches": true,
  "confidence": "alta",
  "ai_analysis": "O pneu apresenta banda de rodagem com profundidade adequada...",
  "additional_findings": ["Marca Michelin visível", "Especificação 205/55 R16"],
  "severity": "ok",
  "specific_data": {
    "brand": "Michelin",
    "size": "205/55 R16",
    "tread_depth": "Aproximadamente 5mm",
    "wear_pattern": "Uniforme"
  },
  "recommendations": ["Manter calibragem conforme manual do veículo"],
  "request_id": "a1b2c3d4"
}
```

### POST /analyze/evolution

Análise de evolução temporal com múltiplas fotos.

```bash
curl -X POST http://localhost:8100/analyze/evolution \
  -F "item_name=Lateral direita - porta traseira" \
  -F "images=@foto_jan.jpg" \
  -F "images=@foto_mar.jpg" \
  -F "images=@foto_jun.jpg" \
  -F "dates=2025-01-15" \
  -F "dates=2025-03-15" \
  -F "dates=2025-06-15"
```

**Resposta:**

```json
{
  "item": "Lateral direita - porta traseira",
  "total_images": 3,
  "per_image_analysis": [
    {
      "image_index": 1,
      "date": "2025-01-15",
      "condition": "Arranhão superficial de ~15cm na porta traseira direita",
      "severity": "atenção"
    },
    {
      "image_index": 2,
      "date": "2025-03-15",
      "condition": "Arranhão com início de oxidação nas bordas",
      "severity": "atenção"
    },
    {
      "image_index": 3,
      "date": "2025-06-15",
      "condition": "Oxidação se expandiu, bolhas na pintura ao redor",
      "severity": "crítico"
    }
  ],
  "evolution_summary": "Arranhão inicial evoluiu para corrosão ativa em 6 meses",
  "trend": "deteriorando",
  "degradation_rate": "moderada",
  "estimated_action_needed": "Reparo imediato recomendado para evitar dano estrutural",
  "recommendations": [
    "Realizar reparo de funilaria e pintura o mais breve possível",
    "Aplicar tratamento anticorrosivo na área afetada"
  ],
  "request_id": "e5f6g7h8"
}
```

## Documentação interativa

Acesse `http://localhost:8100/docs` para a interface Swagger UI.

## Variáveis de ambiente

| Variável | Obrigatória | Padrão | Descrição |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | Chave da API OpenAI |
| `OPENAI_MODEL` | ❌ | `gpt-4o` | Modelo de visão a usar |

## Limites

- Tamanho máximo por imagem: **10 MB**
- Formatos aceitos: **JPG, PNG, WebP**
- Rate limit: **10 requisições/minuto** por IP
- Máximo de imagens no `/analyze/evolution`: **10**
