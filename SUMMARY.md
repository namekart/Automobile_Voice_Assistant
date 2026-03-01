# Project Summary — Automobile Voice Assistant

## What this project is
An outbound automobile customer-support voice agent built on **LiveKit Agents** (Python), using an **STT → LLM → TTS** pipeline, with a task-based conversation flow and PostgreSQL persistence for CRM actions (callbacks, notes).

## Current runtime entrypoint
- **Main app**: `agent.py`
- **How it runs**: `python .\agent.py dev` then connect from LiveKit Playground to the registered agent (`agent_name="my-agent"`).

## Voice pipeline (current)
Configured in `agent.py`:
- **STT**: `sarvam.STT(model="saaras:v3", language="unknown", mode="translate")`
  - Automatic language detection enabled (`language="unknown"`).
  - Translation mode (`mode="translate"`) to normalize user speech into English text (useful later for RAG), while the agent can still speak in the user’s language.
- **LLM**: OpenRouter via `openai.LLM.with_openrouter(model="openai/gpt-4o-mini")`
- **TTS**: `sarvam.TTS(model="bulbul:v3", target_language_code="hi-IN")`
  - Uses `hi-IN` to produce natural Hinglish/Hindi speech from code-mixed text.
- **VAD**: Silero (`silero.VAD.load()`)
- **Turn detection / EOU**: `MultilingualModel()` + endpointing tuned:
  - `min_endpointing_delay=0.3`
  - `max_endpointing_delay=2.0`
- **Noise cancellation**: BVC / BVCTelephony selected based on participant kind.
- **Latency tuning**:
  - `preemptive_generation=False` to reduce contention and stabilize turn detection.

## Call context (MVP)
- Loaded from `data/call_context.json` with defaults in `agent.py`.
- Includes: customer name, car model, number ending, reason, last service date, dealership name, brand, phone, contact id.

## Conversation architecture (current)
The agent uses **LiveKit `AgentTask`** units for structured steps. The controlling Agent (`Assistant`) orchestrates tasks in order.

### Assistant orchestration (current order)
Defined in `Assistant.on_enter()` in `agent.py`:
1. **Verify customer**: `VerifyCustomerTask`
2. **Recording consent**: `RecordingConsentTask`
3. **Permission to talk / callback scheduling**: `PermissionToTalkTask`
4. **Soft engagement (issues)**: `SoftEngagementTask`
5. **Value add pitch**: a short `generate_reply()` (not a task)
6. **Open conversation**: greeting + normal assistant behavior/tools

### Implemented tasks
In `tasks/`:
- `verify_customer.py`: `VerifyCustomerTask`
  - If not verified, the agent says a strict goodbye and **shuts down** the session.
  - Also marks the phone wrong in DB (when configured).
- `recording_consent.py`: `RecordingConsentTask`
- `permission_to_talk.py`: `PermissionToTalkTask`
  - If user is busy, schedules a callback with **structured** date/time.
  - Uses IST (`UTC+5:30`) for “today” handling.
  - Prevents duplicate scheduling via `_callback_scheduled`.
- `soft_engagement.py`: `SoftEngagementTask`
  - Asks about performance and collects issues using the `note_issue` tool.
  - For lower perceived latency, the tool **responds first** (technician will check + asks if any other issue), then writes the note to DB.
  - Logs DB timing: `add_contact_note took Xms` to validate DB isn’t the bottleneck.

Exports: `tasks/__init__.py`

### Assistant-level tool
In `agent.py`:
- `note_car_issue`: tool to record an issue mentioned later in the call.

Note: if both the task and the assistant record the same issue, duplicates can happen. The intended production fix is to either:
- scope `note_car_issue` to “new issues after soft engagement”, and/or
- dynamically enable/disable tools via `agent.update_tools()` so `note_car_issue` isn’t available during `SoftEngagementTask`.

## Database layer (current)
File: `db.py`
- Uses **asyncpg** connection pooling for non-blocking DB operations.
- Initializes pool in `entrypoint()` via `await init_db_connection()`.

### Implemented DB operations
- `mark_phone_wrong(...)`: updates `public.contacts.phone_valid=false` and inserts into `public.call_attempts`
- `schedule_callback(...)`: inserts into `public.scheduled_callbacks` with default random time between 10:00–11:59 if time not provided
- `add_contact_note(...)`: inserts into `public.contact_notes`

### SQL schemas included in repo
In `sql/`:
- `scheduled_callbacks.sql`: creates `public.scheduled_callbacks`
- `contact_notes.sql`: creates `public.contact_notes`

Important: `contacts` and `call_attempts` tables are referenced by code but their CREATE TABLE SQL is not in this repo yet.

## Dependencies
File: `requirements.txt`
- LiveKit agents + plugins (sarvam, silero, turn-detector, noise-cancellation, openai)
- `python-dotenv`
- `asyncpg` (async PostgreSQL)

## Environment variables expected
Typical runtime requires:
- **LiveKit**: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (or LiveKit Cloud equivalents)
- **LLM via OpenRouter**: `OPENROUTER_API_KEY`
- **Database** (optional but recommended): `POSTGRESQL_URI`

## Key problems already solved
- **Call not ending on wrong number**: strict goodbye + session shutdown.
- **Windows timezone issue**: avoided `zoneinfo` dependency; IST is handled by fixed offset.
- **Callback time/date storage**: callback date saved as `YYYY-MM-DD` + time as `HH:MM(:SS)`; plus speech-friendly phrasing.
- **Non-blocking DB**: moved DB ops to asyncpg to avoid event-loop blocking.
- **EOU tuning**: reduced endpointing delays for faster turn-taking.

## Known gaps / next work
- **Booking appointment** workflow (likely TaskGroup or a dedicated booking task with availability tools).
- **Tool scoping** to prevent duplicate “issue note” behavior (dynamic tool enabling or stricter tool instructions).
- **DB schema completion** for `public.contacts` and `public.call_attempts` (and indexes/constraints).
- **Production hardening**: retries/backoff for DB, structured logging, metrics dashboards, telephony integration.

