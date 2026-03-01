# Automobile Voice Assistant

A voice agent for automobile customer support built with **LiveKit Agents** (Python). It runs a structured outbound flow: verify customer → recording consent → permission to talk / callback scheduling → soft engagement (performance & issues) → value-add pitch → open conversation. Supports Hinglish and persists callbacks and contact notes to PostgreSQL.

## Prerequisites

- **Python 3.10+**
- **LiveKit** project (e.g. [LiveKit Cloud](https://cloud.livekit.io)) for agent dispatch and room connectivity
- **OpenRouter** (or direct OpenAI) API key for the LLM
- **PostgreSQL** (e.g. Supabase) for CRM data — optional; the agent runs without DB but won’t save callbacks/notes

## Setup

1. **Clone and enter the project**
   ```bash
   cd Automobile_Voice_Assistant
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv voiceenv
   voiceenv\Scripts\activate   # Windows
   # source voiceenv/bin/activate   # Linux/macOS
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `.env.example` to `.env` (or create `.env`) and set:
   - **LiveKit**: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
   - **LLM**: `OPENROUTER_API_KEY` (for OpenRouter); or use `OPENAI_API_KEY` and switch the LLM in code to direct OpenAI
   - **Database** (optional): `POSTGRESQL_URI`

4. **Database tables** (if using PostgreSQL)
   - Ensure `public.contacts` and `public.call_attempts` exist (referenced by `db.py`).
   - Run the SQL scripts in `sql/` in your PostgreSQL client:
     - `sql/scheduled_callbacks.sql`
     - `sql/contact_notes.sql`

5. **Call context (MVP)**
   - Edit `data/call_context.json` with customer name, car model, number ending, dealership, brand, phone, contact_id, etc. The agent uses this for the current “call”.

## Run

```bash
python agent.py dev
```

Then open [LiveKit Playground](https://meet.livekit.io) (or your LiveKit project’s connect URL), join a room, and connect to the agent. The agent name is `my-agent`.

## Project structure

| Path | Purpose |
|------|--------|
| `agent.py` | Entrypoint, `AgentSession` config (STT/LLM/TTS/VAD/turn detection), `Assistant` agent and task orchestration |
| `db.py` | Async PostgreSQL (asyncpg) helpers: `mark_phone_wrong`, `schedule_callback`, `add_contact_note` |
| `tasks/` | LiveKit tasks: `verify_customer`, `recording_consent`, `permission_to_talk`, `soft_engagement` |
| `data/call_context.json` | Per-call context (customer, vehicle, dealership, phone, etc.) |
| `sql/` | Table definitions for `scheduled_callbacks`, `contact_notes` |

## Voice pipeline

- **STT**: Sarvam Saaras v3 (auto language, translate mode)
- **LLM**: OpenRouter → `openai/gpt-4o-mini` (configurable in `agent.py`)
- **TTS**: Sarvam Bulbul v3 (e.g. `hi-IN` for Hinglish)
- **VAD**: Silero  
- **Turn detection**: MultilingualModel + tuned endpointing delays

## Documentation

- **SUMMARY.md** — Detailed project summary: pipeline, task flow, DB layer, env vars, solved issues, and planned work.

## License

Use according to your organization’s policy.
