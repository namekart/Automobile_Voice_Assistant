# Automobile Voice Assistant

A production-grade Hinglish voice agent for automobile dealership outbound calls, built with **LiveKit Agents v1.4.5** (Python). Runs a structured task pipeline: verify customer → recording consent → permission to talk → soft engagement (car issues) → main conversation (service pitch, objection handling, booking). Supports Hinglish and persists callbacks and contact notes to PostgreSQL.

## Prerequisites

- **Python 3.10+**
- **LiveKit Cloud** project — for agent dispatch and room connectivity
- **OpenAI API key** — primary LLM (gpt-4.1-mini)
- **Anthropic API key** — fallback LLM (Claude Haiku)
- **Cartesia API key** — primary TTS (sonic-3, Hindi)
- **Sarvam API key** — fallback TTS (bulbul:v3-beta) and fallback STT (saaras:v3)
- **PostgreSQL** (e.g. Supabase) — optional; agent runs without it but won't persist callbacks or notes

## Setup

1. **Clone and enter the project**
   ```bash
   cd Automobile_Voice_Assistant
   ```

2. **Create a virtual environment and install dependencies**

   Using `uv` (recommended):
   ```bash
   uv sync
   ```

   Or with pip:
   ```bash
   python -m venv voiceenv
   voiceenv\Scripts\activate        # Windows
   # source voiceenv/bin/activate   # Linux/macOS
   pip install -r requirements.txt
   ```

3. **Configure environment**

   Create a `.env` file with:
   ```env
   # LiveKit
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=your_api_key
   LIVEKIT_API_SECRET=your_api_secret

   # LLM
   OPENAI_API_KEY=sk-...           # primary LLM (gpt-4.1-mini)
   ANTHROPIC_API_KEY=sk-ant-...    # fallback LLM (Claude Haiku)

   # TTS
   CARTESIA_API_KEY=...            # primary TTS (sonic-3)
   SARVAM_API_KEY=...              # fallback TTS + fallback STT

   # Database (optional)
   POSTGRESQL_URI=postgresql://user:password@host/db
   ```

4. **Database tables** (if using PostgreSQL)

   Run the SQL scripts in `sql/` in your PostgreSQL client:
   ```bash
   sql/scheduled_callbacks.sql
   sql/contact_notes.sql
   ```

5. **Call context**

   Edit `data/call_context.json` with customer details for the current call:
   ```json
   {
     "customer_name": "Mr. Rakesh Sharma",
     "car_model": "Swift",
     "number_ending": "1234",
     "dealership": "ABC Motors",
     "brand": "Maruti",
     "phone_number": "+91XXXXXXXXXX",
     "contact_id": "...",
     "reason": "periodic service due"
   }
   ```

## Run

**Development** (connects to LiveKit Cloud, join via [LiveKit Playground](https://agents-playground.livekit.io)):
```bash
python agent.py dev
```

**Local console** (microphone + speaker, no room needed — good for testing):
```bash
python agent.py console
```

**Production**:
```bash
python agent.py start
```

The agent name is `my-agent`.

## Project structure

| Path | Purpose |
|------|---------|
| `agent.py` | Entrypoint — `AgentSession` config (STT/LLM/TTS/VAD/turn detection), `Assistant` agent, task orchestration, system prompt |
| `db.py` | Async PostgreSQL (asyncpg) helpers: `mark_phone_wrong`, `schedule_callback`, `add_contact_note`, `init_db_connection` |
| `tasks/verify_customer.py` | Task 1 — confirm customer identity |
| `tasks/recording_consent.py` | Task 2 — get consent to record the call |
| `tasks/permission_to_talk.py` | Task 3 — check if now is convenient; schedule callback if not |
| `tasks/soft_engagement.py` | Task 4 — ask about car performance and issues; note complaints |
| `tasks/relative_choice.py` | Helper task — offer a date/slot choice relative to today |
| `tasks/__init__.py` | Shared `TASK_GUARDRAILS` string injected into all tasks |
| `data/call_context.json` | Per-call context (customer, vehicle, dealership, phone, etc.) |
| `sql/` | Table definitions for `scheduled_callbacks` and `contact_notes` |
| `livekit.toml` | LiveKit Cloud project config (used by `lk` CLI) |
| `pyproject.toml` | Project metadata and dependencies (uv/pip) |
| `Dockerfile` | Container image for cloud deployment |

## Voice pipeline

| Component | Primary | Fallback |
|-----------|---------|---------|
| **STT** | Deepgram nova-3 (via LiveKit Inference, Mumbai colocated) | Sarvam saaras:v3 |
| **LLM** | OpenAI gpt-4.1-mini (`attempt_timeout=2s`) | Anthropic Claude Haiku |
| **TTS** | Cartesia sonic-3, Hindi voice | Sarvam bulbul:v3-beta |
| **VAD** | Silero (`min_silence_duration=0.35s`) | — |
| **Turn detection** | LiveKit MultilingualModel | — |

**Latency optimizations:**
- LLM TCP warmup runs concurrently with room connect and DB init — eliminates ~2s cold-start on first user turn
- OpenAI prompt caching: stable sections of the system prompt are placed first; dynamic `## This Call` section is last — achieves ~1792–2048 cached tokens from turn 2 onward
- Preemptive LLM generation triggered by turn detector before user fully stops speaking

## Call flow

```
Call connects
    │
    ▼
VerifyCustomerTask      — "Kya main Mr. X se baat kar raha hoon?"
    │ verified / wrong_number / not_available
    ▼
RecordingConsentTask    — ask consent to record
    │ consent_given / consent_denied
    ▼
PermissionToTalkTask    — "Kya abhi ek minute baat karna convenient hoga?"
    │ user_has_time / callback_scheduled
    ▼
SoftEngagementTask      — ask about car performance and issues
    │ done_with_issues / done_no_issues
    ▼
Main conversation       — service pitch, objection handling, booking
    │ book_service_appointment / record_crm_correction / schedule_callback_tool / ...
    ▼
Call ends
```

## Key behaviors

- **Objection handling**: adaptive (not count-based) — agent addresses the specific new concern raised each time, using dealership USPs (pickup-drop, genuine parts, transparent billing, etc.). Only gracefully accepts when customer is clearly firm.
- **Human handoff**: if user asks for manager/person, agent says it will pass the message and offers a callback.
- **HOLD / driving / repeat / off-topic**: all handled via system prompt guardrails.
- **CRM corrections**: if user corrects car model, name, or says car is sold — immediately calls `record_crm_correction` or `record_car_sold` without re-asking.
- **Language**: Hinglish — Hindi words in Devanagari, English words in Latin script. Masculine verb forms throughout.

## License

Use according to your organization's policy.
