# How to run the LiveKit voice agent

## One-time setup

### Option A: Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is the package manager used in LiveKit’s official docs (faster installs, lockfile, no separate venv activation).

1. **Install uv** (if needed)
   ```powershell
   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   ```
   Or: `winget install astral.uv`

2. **Install dependencies and download plugin files**
   ```powershell
   cd c:\Users\champ\Desktop\AMP\Automobile_Voice_Assistant
   uv sync
   uv run agent.py download-files
   ```

3. **Environment**
   - Your `.env` already has `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `OPENROUTER_API_KEY`, `SARVAM_API_KEY`.

### Option B: Using pip + venv

1. **Virtual environment and dependencies**
   ```powershell
   cd c:\Users\champ\Desktop\AMP\Automobile_Voice_Assistant
   python -m venv voiceenv
   .\voiceenv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Download plugin model files**
   ```powershell
   python agent.py download-files
   ```

---

## Run the agent

**With uv (no activation needed):**
```powershell
uv run agent.py dev
```

**With pip/venv:**
```powershell
.\voiceenv\Scripts\Activate.ps1
python agent.py dev
```

Then open [Agents Playground](https://meet.livekit.io) (or your project’s connect URL), join a room, and connect to the agent **`my-agent`**.

**Other modes**
- **Console** (terminal only): `uv run agent.py console` or `python agent.py console`
- **Production**: `uv run agent.py start` or `python agent.py start`

## Reference

- Run steps from [LiveKit Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai-quickstart/) and [Server startup modes](https://docs.livekit.io/agents/server/startup-modes/), via LiveKit Docs MCP.
