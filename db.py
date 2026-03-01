"""PostgreSQL connection and CRM helpers. Async (asyncpg) for non-blocking, low-latency voice agent."""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)

_pool: Any = None
_lock = asyncio.Lock()


def _get_uri() -> str | None:
    return os.environ.get("POSTGRESQL_URI")


async def _get_pool() -> Any:
    """Return the async connection pool; create on first use. Thread-safe via lock."""
    global _pool
    uri = _get_uri()
    if not uri:
        return None
    async with _lock:
        if _pool is None:
            try:
                import asyncpg
                _pool = await asyncpg.create_pool(
                    uri,
                    min_size=1,
                    max_size=4,
                    command_timeout=10,
                )
                logger.info("DB pool initialized for voice assistant.")
            except Exception as e:
                logger.warning("DB pool creation failed: %s", e)
                _pool = None
        return _pool


async def init_db_connection() -> bool:
    """Initialize DB pool at startup so first use has no connection latency. Call from entrypoint."""
    pool = await _get_pool()
    if pool is None:
        logger.info("POSTGRESQL_URI not set or connection failed; DB ops will no-op.")
        return False
    return True


async def mark_phone_wrong(
    phone_number: str | None,
    contact_id: str | None = None,
    reason: str = "wrong_number",
) -> None:
    """Mark contact phone as invalid and log attempt. No-op if no connection or missing ids."""
    if not phone_number and not contact_id:
        logger.debug("mark_phone_wrong: no phone_number or contact_id, skipping")
        return
    pool = await _get_pool()
    if pool is None:
        logger.warning("mark_phone_wrong: no DB pool; skipping")
        return
    try:
        async with pool.acquire() as conn:
            if contact_id:
                await conn.execute(
                    "UPDATE public.contacts SET phone_valid = false, updated_at = now() WHERE id = $1",
                    contact_id,
                )
            if phone_number:
                await conn.execute(
                    "UPDATE public.contacts SET phone_valid = false, updated_at = now() WHERE phone_number = $1",
                    phone_number,
                )
            cid = contact_id
            pn = phone_number or ""
            if not cid and phone_number:
                row = await conn.fetchrow(
                    "SELECT id FROM public.contacts WHERE phone_number = $1 LIMIT 1",
                    phone_number,
                )
                cid = str(row["id"]) if row else None
                pn = phone_number or ""
            if cid and not pn:
                row = await conn.fetchrow(
                    "SELECT phone_number FROM public.contacts WHERE id = $1 LIMIT 1",
                    cid,
                )
                pn = (row["phone_number"] or "") if row else ""
            if cid:
                await conn.execute(
                    "INSERT INTO public.call_attempts (contact_id, phone_number, outcome) VALUES ($1, $2, $3)",
                    cid,
                    pn,
                    reason,
                )
    except Exception as e:
        logger.warning("mark_phone_wrong failed: %s", e)


async def schedule_callback(
    callback_date: str,
    phone_number: str | None = None,
    contact_id: str | None = None,
    callback_time: str | None = None,
    preferred_raw: str | None = None,
) -> None:
    """Store a callback request. callback_date: YYYY-MM-DD. callback_time: HH:MM (24h) or None for default 10–12."""
    if not (callback_date or "").strip():
        logger.debug("schedule_callback: empty callback_date, skipping")
        return
    pool = await _get_pool()
    if pool is None:
        logger.warning("schedule_callback: no DB pool; skipping")
        return
    cid = contact_id
    pn = phone_number or ""
    try:
        async with pool.acquire() as conn:
            if not cid and phone_number:
                row = await conn.fetchrow(
                    "SELECT id FROM public.contacts WHERE phone_number = $1 LIMIT 1",
                    phone_number,
                )
                cid = str(row["id"]) if row else None
                pn = phone_number or ""
            if cid and not pn:
                row = await conn.fetchrow(
                    "SELECT phone_number FROM public.contacts WHERE id = $1 LIMIT 1",
                    cid,
                )
                pn = (row["phone_number"] or "") if row else ""
            if not cid:
                logger.warning("schedule_callback: could not resolve contact_id; skipping")
                return
            if not callback_time or not callback_time.strip():
                hour = 10
                minute = random.randint(0, 119)
                if minute >= 60:
                    hour, minute = 11, minute - 60
                callback_time = f"{hour:02d}:{minute:02d}"
            if len(callback_time) == 5 and callback_time[2] == ":":
                callback_time = callback_time + ":00"
            await conn.execute(
                """INSERT INTO public.scheduled_callbacks
                   (contact_id, phone_number, callback_date, callback_time, preferred_raw, status)
                   VALUES ($1, $2, $3::date, $4::time, $5, 'pending')""",
                cid,
                pn,
                callback_date.strip(),
                callback_time,
                preferred_raw or None,
            )
        logger.info("schedule_callback: saved for contact_id=%s on %s at %s", cid, callback_date, callback_time)
    except Exception as e:
        logger.warning("schedule_callback failed: %s", e)


async def add_contact_note(
    content: str,
    source: str = "assistant",
    *,
    contact_id: str | None = None,
    phone_number: str | None = None,
    note_type: str = "car_issue",
) -> None:
    """Append a note for a contact. source: soft_engagement | assistant | human."""
    content = (content or "").strip()
    if not content:
        logger.debug("add_contact_note: empty content, skipping")
        return
    pool = await _get_pool()
    if pool is None:
        logger.warning("add_contact_note: no DB pool; skipping")
        return
    try:
        async with pool.acquire() as conn:
            cid = contact_id
            if not cid and phone_number:
                row = await conn.fetchrow(
                    "SELECT id FROM public.contacts WHERE phone_number = $1 LIMIT 1",
                    phone_number,
                )
                cid = str(row["id"]) if row else None
            if cid is None:
                logger.warning("add_contact_note: could not resolve contact_id; skipping")
                return
            ntype = (note_type or "car_issue").strip().lower() or "car_issue"
            src = (source or "assistant").strip().lower() or "assistant"
            await conn.execute(
                """INSERT INTO public.contact_notes (contact_id, note_type, content, source)
                   VALUES ($1, $2, $3, $4)""",
                cid,
                ntype,
                content,
                src,
            )
        logger.info("add_contact_note: saved for contact_id=%s source=%s", cid, source)
    except Exception as e:
        logger.warning("add_contact_note failed: %s", e)
