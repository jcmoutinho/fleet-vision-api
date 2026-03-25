"""
Fleet Vision API — Módulo de banco de dados SQLite (async).
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import aiosqlite

logger = logging.getLogger("fleet-vision-api.db")

DB_PATH = os.getenv("DB_PATH", "/data/fleet_vision.db")

_db: aiosqlite.Connection | None = None


async def init_db():
    """Cria a tabela se não existir e abre a conexão global."""
    global _db
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _db = await aiosqlite.connect(DB_PATH, check_same_thread=False)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id          TEXT UNIQUE NOT NULL,
            item_name           TEXT NOT NULL,
            user_description    TEXT,
            image_b64           TEXT,
            image_mime          TEXT,
            image_hash          TEXT,
            image_size_bytes    INTEGER DEFAULT 0,
            ai_result           TEXT,
            backend_used        TEXT,
            input_tokens        INTEGER DEFAULT 0,
            output_tokens       INTEGER DEFAULT 0,
            cost_usd            REAL DEFAULT 0.0,
            severity            TEXT,
            description_matches INTEGER DEFAULT 0,
            vehicle_id          TEXT,
            created_at          TEXT NOT NULL,
            completed_at        TEXT,
            processing_seconds  REAL DEFAULT 0.0,
            expires_at          TEXT NOT NULL
        )
    """)
    # Migração: adiciona colunas novas se não existirem (para DBs antigos)
    existing = set()
    cursor = await _db.execute("PRAGMA table_info(analyses)")
    async for row in cursor:
        existing.add(row["name"])
    for col, definition in [
        ("image_hash",         "TEXT"),
        ("image_size_bytes",   "INTEGER DEFAULT 0"),
        ("completed_at",       "TEXT"),
        ("processing_seconds", "REAL DEFAULT 0.0"),
        ("vehicle_id",         "TEXT"),
    ]:
        if col not in existing:
            await _db.execute(f"ALTER TABLE analyses ADD COLUMN {col} {definition}")
            logger.info("Migração: coluna '%s' adicionada.", col)

    await _db.commit()
    logger.info("Banco de dados inicializado: %s", DB_PATH)


async def close_db():
    """Fecha a conexão global."""
    global _db
    if _db:
        await _db.close()
        _db = None
        logger.info("Conexão com banco de dados fechada.")


def _image_hash(image_b64: str) -> str:
    """SHA-256 do conteúdo base64 da imagem."""
    return hashlib.sha256(image_b64.encode()).hexdigest()


async def find_duplicate(image_b64: str, item_name: str) -> dict | None:
    """Busca análise anterior com a mesma imagem e mesmo item (não expirada)."""
    h = _image_hash(image_b64)
    now = datetime.now(timezone.utc).isoformat()
    cursor = await _db.execute(
        """
        SELECT request_id, item_name, severity, description_matches,
               backend_used, cost_usd, created_at, completed_at, processing_seconds
        FROM analyses
        WHERE image_hash = ? AND item_name = ? AND expires_at > ?
        ORDER BY created_at DESC LIMIT 1
        """,
        (h, item_name, now),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "request_id": row["request_id"],
        "item_name": row["item_name"],
        "severity": row["severity"],
        "description_matches": bool(row["description_matches"]),
        "backend_used": row["backend_used"],
        "cost_usd": row["cost_usd"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "processing_seconds": row["processing_seconds"],
    }


async def save_analysis(
    request_id: str,
    item_name: str,
    user_description: str,
    image_b64: str,
    image_mime: str,
    ai_result_dict: dict,
    backend_used: str,
    usage_dict: dict,
    processing_seconds: float = 0.0,
    vehicle_id: str | None = None,
):
    """Insere um registro de análise no banco."""
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    completed_at = datetime.now(timezone.utc).isoformat()
    expires_at = (now + timedelta(days=90)).isoformat()

    severity = ai_result_dict.get("severity", "atenção")
    description_matches = 1 if ai_result_dict.get("description_matches") else 0
    input_tokens = usage_dict.get("input_tokens", 0)
    output_tokens = usage_dict.get("output_tokens", 0)
    cost_usd = usage_dict.get("estimated_cost_usd", 0.0)
    image_hash = _image_hash(image_b64)
    # Tamanho real em bytes (base64 → binário ≈ len * 3/4)
    image_size_bytes = len(image_b64) * 3 // 4

    await _db.execute(
        """
        INSERT INTO analyses
            (request_id, item_name, user_description, image_b64, image_mime,
             image_hash, image_size_bytes, ai_result, backend_used,
             input_tokens, output_tokens, cost_usd,
             severity, description_matches, vehicle_id, created_at, completed_at,
             processing_seconds, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_id, item_name, user_description, image_b64, image_mime,
            image_hash, image_size_bytes,
            json.dumps(ai_result_dict, ensure_ascii=False), backend_used,
            input_tokens, output_tokens, cost_usd,
            severity, description_matches, vehicle_id, created_at, completed_at,
            processing_seconds, expires_at,
        ),
    )
    await _db.commit()
    logger.info(
        "Análise salva | request_id=%s | severity=%s | %.2fs | %d tokens | $%.6f",
        request_id, severity, processing_seconds,
        input_tokens + output_tokens, cost_usd,
    )


async def get_analysis(request_id: str) -> dict | None:
    """Retorna análise completa por request_id (sem image_b64)."""
    cursor = await _db.execute(
        """
        SELECT request_id, item_name, user_description, ai_result, backend_used,
               input_tokens, output_tokens, cost_usd, severity, description_matches,
               image_mime, image_size_bytes, created_at, completed_at,
               processing_seconds, expires_at
        FROM analyses WHERE request_id = ?
        """,
        (request_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "request_id": row["request_id"],
        "item_name": row["item_name"],
        "user_description": row["user_description"],
        "ai_result": json.loads(row["ai_result"]) if row["ai_result"] else {},
        "backend_used": row["backend_used"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "cost_usd": row["cost_usd"],
        "severity": row["severity"],
        "description_matches": bool(row["description_matches"]),
        "image_mime": row["image_mime"],
        "image_size_bytes": row["image_size_bytes"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "processing_seconds": row["processing_seconds"],
        "expires_at": row["expires_at"],
    }


async def list_analyses(
    page: int = 1,
    per_page: int = 20,
    severity: str | None = None,
    item_name: str | None = None,
    vehicle_id: str | None = None,
) -> dict:
    """Lista análises com filtros e paginação (sem image_b64)."""
    conditions = []
    params: list = []

    if severity:
        conditions.append("severity = ?")
        params.append(severity)
    if item_name:
        conditions.append("item_name LIKE ?")
        params.append(f"%{item_name}%")
    if vehicle_id:
        conditions.append("vehicle_id LIKE ?")
        params.append(f"%{vehicle_id}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    cursor = await _db.execute(f"SELECT COUNT(*) as cnt FROM analyses {where}", params)
    row = await cursor.fetchone()
    total = row["cnt"]

    offset = (page - 1) * per_page
    cursor = await _db.execute(
        f"""
        SELECT request_id, item_name, user_description, severity, description_matches,
               backend_used, cost_usd, image_size_bytes, vehicle_id, created_at, completed_at,
               processing_seconds, expires_at
        FROM analyses {where}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        params + [per_page, offset],
    )
    rows = await cursor.fetchall()

    items = [
        {
            "request_id": r["request_id"],
            "item_name": r["item_name"],
            "user_description": r["user_description"],
            "severity": r["severity"],
            "description_matches": bool(r["description_matches"]),
            "backend_used": r["backend_used"],
            "cost_usd": r["cost_usd"],
            "image_size_bytes": r["image_size_bytes"],
            "vehicle_id": r["vehicle_id"] or "",
            "created_at": r["created_at"],
            "completed_at": r["completed_at"],
            "processing_seconds": r["processing_seconds"],
            "expires_at": r["expires_at"],
        }
        for r in rows
    ]

    return {"total": total, "page": page, "per_page": per_page, "items": items}


async def get_analysis_image(request_id: str) -> tuple[str, str] | None:
    """Retorna (image_b64, image_mime) para um request_id."""
    cursor = await _db.execute(
        "SELECT image_b64, image_mime FROM analyses WHERE request_id = ?",
        (request_id,),
    )
    row = await cursor.fetchone()
    if not row or not row["image_b64"]:
        return None
    return row["image_b64"], row["image_mime"]


async def cleanup_expired():
    """Deleta registros onde expires_at < agora."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = await _db.execute("DELETE FROM analyses WHERE expires_at < ?", (now,))
    await _db.commit()
    deleted = cursor.rowcount
    if deleted > 0:
        logger.info("Cleanup: %d registro(s) expirado(s) removido(s).", deleted)
    else:
        logger.info("Cleanup: nenhum registro expirado.")


async def get_stats() -> dict:
    """Retorna estatísticas gerais."""
    cursor = await _db.execute(
        "SELECT COUNT(*) as total, COALESCE(SUM(cost_usd),0) as total_cost, "
        "COALESCE(SUM(input_tokens+output_tokens),0) as total_tokens, "
        "COALESCE(AVG(processing_seconds),0) as avg_processing "
        "FROM analyses"
    )
    row = await cursor.fetchone()
    total = row["total"]
    total_cost_usd = row["total_cost"]
    total_tokens = row["total_tokens"]
    avg_processing = row["avg_processing"]

    cursor = await _db.execute(
        "SELECT severity, COUNT(*) as cnt FROM analyses GROUP BY severity"
    )
    by_severity: dict = {}
    async for r in cursor:
        by_severity[r["severity"]] = r["cnt"]

    return {
        "total_analyses": total,
        "by_severity": {
            "ok": by_severity.get("ok", 0),
            "atenção": by_severity.get("atenção", 0),
            "crítico": by_severity.get("crítico", 0),
        },
        "total_tokens": int(total_tokens),
        "avg_processing_seconds": round(avg_processing, 2),
        "total_cost_usd": round(total_cost_usd, 6),
        "total_cost_brl": round(total_cost_usd * 5.0, 4),
    }
