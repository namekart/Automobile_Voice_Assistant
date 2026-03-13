"""
List ElevenLabs voices via v2 API and show which are usable on free plan.
Docs: https://elevenlabs.io/docs/api-reference/voices/search
"""
import json
import urllib.request
from pathlib import Path

# Load API key from project .env (avoid env var / quoting issues)
env_path = Path(__file__).resolve().parent / ".env"
raw = ""
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("ELEVEN_API_KEY="):
            raw = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
if not raw:
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    raw = os.getenv("ELEVEN_API_KEY") or ""
api_key = "".join(c for c in raw if c.isalnum() or c == "_")
if not api_key or not api_key.startswith("sk_"):
    print("ELEVEN_API_KEY not found or invalid in .env")
    exit(1)

# v2/voices with page_size to get more results (docs: List voices)
url = "https://api.elevenlabs.io/v2/voices?page_size=100&include_total_count=true"
req = urllib.request.Request(
    url,
    headers={"xi-api-key": api_key, "Content-Type": "application/json"},
    method="GET",
)
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"API error {e.code}: {e.reason}")
    if e.fp:
        body = e.fp.read().decode()
        print(body[:500])
    exit(1)

voices = data.get("voices", [])
total = data.get("total_count", len(voices))
has_more = data.get("has_more", False)

print(f"Total voices (this page): {len(voices)}")
if has_more:
    print("(More pages available; increase page_size or use next_page_token for full list)\n")

# Build list of voices usable on free plan:
# - sharing.free_users_allowed == True (shared voice allowed for free users)
# - No sharing block or category premade/default (typically available)
usable_free = []
for v in voices:
    vid = v.get("voice_id", "")
    name = v.get("name", "—")
    category = v.get("category", "—")
    sharing = v.get("sharing") or {}
    free_ok = sharing.get("free_users_allowed")
    if free_ok is True:
        usable_free.append((vid, name, category))
    elif free_ok is False:
        pass  # not usable on free
    else:
        # No sharing / N/A: default or personal voice, assume usable for owner
        usable_free.append((vid, name, category))

print("--- Voices you can use on free plan ---")
for vid, name, category in usable_free:
    print(f"  {vid}  |  {name}  |  category={category}")

print("\n--- All voices (id, name, category, free_users_allowed) ---")
for v in voices:
    vid = v.get("voice_id", "")
    name = v.get("name", "—")
    category = v.get("category", "—")
    sharing = v.get("sharing") or {}
    free_ok = sharing.get("free_users_allowed")
    free_str = "yes" if free_ok is True else ("no" if free_ok is False else "N/A (default/personal)")
    print(f"  {vid}  |  {name}  |  {category}  |  free_plan={free_str}")
