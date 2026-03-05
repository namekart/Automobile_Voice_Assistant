import asyncio
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    AutoSubscribe,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    function_tool,
    metrics,
    room_io,
    RunContext,
)
from livekit.agents.llm import ChatContext, ChatMessage

logger = logging.getLogger(__name__)

# Ensure app loggers are visible in dev (worker process may not inherit config)
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
# Reduce TTS/STT log noise so app logs stay visible
logging.getLogger("livekit.plugins.sarvam").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.sarvam.log").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.sarvam.log.SpeechStream").setLevel(logging.WARNING)

from livekit.plugins import deepgram, openai, silero, sarvam, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from tasks import (
    VerifyCustomerTask,
    VerifyResult,
    RecordingConsentTask,
    PermissionToTalkTask,
    PermissionResult,
    RelativeChoiceTask,
    SoftEngagementTask,
    SoftEngagementResult,
)
from db import init_db_connection, mark_phone_wrong, add_contact_note

load_dotenv()
# AGENT_NAME = random.choice(["Shubh", "Ritu", "Amit", "Sumit", "Pooja", "Manan", "Simran", "Rahul", "Kavya", "Ratan", "Priya", "Ishita", "Shreya", "Shruti"])
AGENT_NAME="Shubh"

# Call context: customer and vehicle info for this call. Load from JSON (MVP); later from DB or room metadata.
CALL_CONTEXT_PATH = Path(__file__).resolve().parent / "data" / "call_context.json"

DEFAULT_CALL_CONTEXT: dict[str, str | None] = {
    "customer_name": "the customer",
    "car_model": "their vehicle",
    "number_ending": "",
    "reason_for_call": "service reminder",
    "last_service_date": None,
    "dealership_name": "our dealership",
    "brand": "the brand",
    "phone_number": None,
    "contact_id": None,
    "language_preference": "en-IN",
}


# Sarvam Bulbul v3 has no separate "Hinglish" code; it handles code-mixed (Hinglish) text when
# target_language_code is hi-IN or en-IN. We use hi-IN so the assistant speaks in Hinglish/Hindi.
TTS_LANGUAGE = "hi-IN"

# STT validation: reject empty/garbage/inaudible before LLM; ask to repeat via LLM (user's language).
# Inaudible pattern: some STT providers (e.g. Deepgram, AssemblyAI) inject tokens like [inaudible],
# [unintelligible], etc. when speech is unclear. Sarvam may or may not; this is a safety net.
# If you see Sarvam output different placeholders in logs, add them to the pattern below.
INAUDIBLE_PATTERN = re.compile(
    r"^[\s\[\]\.\,\-\*]*(\[?(?:inaudible|unintelligible|silence|noise|unclear|cough|laughter)\]?[\s\[\]\.\,\-\*]*)+$",
    re.IGNORECASE,
)
# Message we inject so the LLM replies with "please repeat" in user's language (e.g. Hinglish).
INAUDIBLE_MARKER = (
    "[The user's speech was inaudible or unclear. "
    "Respond with a single short request in the user's language (e.g. Hinglish) asking them to repeat. Nothing else.]"
)


def _is_valid_user_transcript(text: str | None) -> bool:
    """Return False if transcript is empty, only whitespace, or only inaudible markers. Single words (e.g. yes/no) are valid."""
    if text is None:
        return False
    s = (text or "").strip()
    if not s:
        return False
    if INAUDIBLE_PATTERN.match(s):
        return False
    return True


def _callback_when_for_speech(permission: PermissionResult) -> str:
    """Phrase for when we'll call: use LLM's speech_phrase (user language) or fallback to natural date (no digits)."""
    if permission.speech_phrase and permission.speech_phrase.strip():
        return permission.speech_phrase.strip()
    if permission.callback_date:
        try:
            dt = datetime.strptime(permission.callback_date.strip()[:10], "%Y-%m-%d")
            date_part = f"{dt.day} {dt.strftime('%B')} ko"
            return f"{date_part} subah 10 aur 12 ke beech" if not permission.callback_time else f"{date_part} ke around"
        except (ValueError, TypeError):
            pass
    return "aapke bataye time pe"


def load_call_context() -> dict[str, str | None]:
    """Load call context from data/call_context.json. Falls back to DEFAULT_CALL_CONTEXT if missing or invalid."""
    if not CALL_CONTEXT_PATH.exists():
        logger.warning("Call context file not found at %s, using default", CALL_CONTEXT_PATH)
        return dict(DEFAULT_CALL_CONTEXT)
    try:
        data = json.loads(CALL_CONTEXT_PATH.read_text(encoding="utf-8"))
        # Merge with defaults so missing keys are filled
        out = dict(DEFAULT_CALL_CONTEXT)
        for key in DEFAULT_CALL_CONTEXT:
            if key in data and data[key] is not None:
                out[key] = str(data[key])
            elif key in data:
                out[key] = None
        return out
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load call context from %s: %s. Using default.", CALL_CONTEXT_PATH, e)
        return dict(DEFAULT_CALL_CONTEXT)


def _build_instructions(call_context: dict[str, str | None]) -> str:
    """Build assistant instructions from base persona and call context."""
    customer = call_context.get("customer_name") or "the customer"
    car = call_context.get("car_model") or "their vehicle"
    ending = call_context.get("number_ending") or ""
    reason = call_context.get("reason_for_call") or "service reminder"
    last_date = call_context.get("last_service_date")
    dealership = call_context.get("dealership_name") or "our dealership"
    brand = call_context.get("brand") or "the brand"
    number_line = f" (number ending {ending})" if ending else ""
    reason_line = reason.replace("_", " ")
    last_line = f" Last service was on {last_date}." if last_date else ""
    context_block = f"""## This call
You are calling {customer} about their {car}{number_line}. Reason: {reason_line}.{last_line} Dealership: {dealership} ({brand}). Use naturally in conversation."""
    behavior = """## Behavior
Stay calm and professional; never argue or be defensive. If the user is upset or sarcastic, acknowledge briefly and refocus on helping. If the user corrects any fact (e.g. wrong vehicle model, wrong service date), acknowledge and say you will get it updated, then call record_crm_correction. If the user says they sold the car, acknowledge, ask who has the car now, then call record_car_sold with any details they give."""
    base = f"""You are {AGENT_NAME}, voice assistant for {dealership} ({brand}). Help with service and bookings. Concise, user's language (e.g. Hinglish)."""
    return base + "\n\n" + context_block + "\n\n" + behavior


# Production pattern: STT converts speech → English (for RAG/LLM). We capture detected user
# language from STT and store it in session userdata so we can use it later for TTS (speak in
# user's language). Sarvam STT with language="unknown" returns language_code in the event.


class Assistant(Agent):
    def __init__(self, *, call_context: dict[str, str | None] | None = None) -> None:
        ctx = call_context if call_context is not None else load_call_context()
        self._call_context = ctx
        instructions = _build_instructions(ctx)
        super().__init__(instructions=instructions)

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Reject empty/garbage/inaudible transcripts; replace with marker so LLM asks to repeat in user's language."""
        raw = getattr(new_message, "text_content", None)
        text = (raw() if callable(raw) else raw) if raw is not None else ""
        if not _is_valid_user_transcript(text):
            logger.info("STT validation: rejecting transcript (empty/garbage/inaudible), LLM will ask to repeat")
            new_message.content = [INAUDIBLE_MARKER]
            # Do not raise StopResponse: let LLM generate one short "please repeat" in user's language (e.g. Hinglish).

    @function_tool
    async def record_crm_correction(
        self, context: RunContext, correction_type: str, correct_value: str
    ) -> None:
        """Call when the customer corrects a fact we have wrong (e.g. vehicle model, last service date). correction_type: e.g. 'car_model', 'last_service_date'. correct_value: what the customer said (e.g. 'Baleno', 'March')."""
        ctype = (correction_type or "").strip() or "unknown"
        val = (correct_value or "").strip() or ""
        if not val:
            return
        content = f"{ctype}: {val}"
        ctx = self._call_context
        pending = self.session.userdata.get("pending_contact_notes") or []
        if not isinstance(pending, list):
            pending = []
        pending.append({
            "content": content,
            "source": "assistant",
            "contact_id": ctx.get("contact_id"),
            "phone_number": ctx.get("phone_number"),
            "note_type": "crm_correction",
        })
        self.session.userdata["pending_contact_notes"] = pending
        logger.info("Assistant: record_crm_correction (deferred) %s=%s", ctype, val[:50])

    @function_tool
    async def record_car_sold(self, context: RunContext, new_owner_info: str = "") -> None:
        """Call when the customer says they sold the car. After asking who has the car now, pass whatever they said as new_owner_info (or leave empty if they don't know)."""
        info = (new_owner_info or "").strip()
        content = "Car sold." + (f" New owner / details: {info}" if info else " New owner details not provided.")
        ctx = self._call_context
        pending = self.session.userdata.get("pending_contact_notes") or []
        if not isinstance(pending, list):
            pending = []
        pending.append({
            "content": content,
            "source": "assistant",
            "contact_id": ctx.get("contact_id"),
            "phone_number": ctx.get("phone_number"),
            "note_type": "car_sold",
        })
        self.session.userdata["pending_contact_notes"] = pending
        logger.info("Assistant: record_car_sold (deferred) info=%s", info[:50] if info else "none")

    async def on_enter(self) -> None:
        call_context = self._call_context
        customer_name = call_context.get("customer_name") or "the customer"

        logger.info("Assistant on_enter: starting VerifyCustomerTask (customer_name=%s)", customer_name)
        # 1. Verify we're speaking with the right person (verified / wrong_number / not_available)
        result = await VerifyCustomerTask(
            chat_ctx=self.chat_ctx,
            customer_name=customer_name,
        )
        logger.info("Assistant on_enter: VerifyCustomerTask finished, verified=%s wrong_number=%s not_available=%s relation=%s",
                    result.verified, result.wrong_number, result.not_available, result.relation or "")
        if result.wrong_number:
            await self.session.generate_reply(
                instructions="One short line in user's language: apologise for the inconvenience and wish them a good day in user language. Nothing else.",
            )
            await mark_phone_wrong(
                call_context.get("phone_number"),
                call_context.get("contact_id"),
                reason="wrong_number",
            )
            self.session.shutdown()
            return
        if result.not_available:
            # Relative on line: offer speak-to-me or call-back-later (single question, single LLM+tool turn)
            continue_with_relative = await RelativeChoiceTask(
                chat_ctx=self.chat_ctx,
                customer_name=customer_name,
            )
            if not continue_with_relative:
                await self.session.generate_reply(
                    instructions="One short line in user's language: we will call back later. Thank and wish good day. Nothing else.",
                )
                self.session.shutdown()
                return
            # Continue conversation with relative (they will pass message); optional: set session.userdata["speaking_with_relative"] = result.relation
            self.session.userdata["speaking_with_relative"] = result.relation or "relative"

        logger.info("Assistant on_enter: starting RecordingConsentTask")
        # 2. Introduction + recording consent (task instructions active during task)
        consent = await RecordingConsentTask(
            chat_ctx=self.chat_ctx,
            agent_name=AGENT_NAME,
            dealership_name=call_context.get("dealership_name") or "our dealership",
        )
        logger.info("Assistant on_enter: RecordingConsentTask finished, consent=%s", consent)
        if consent:
            self.session.userdata["recording_consent"] = "true"
        else:
            self.session.userdata["recording_consent"] = "false"

        # 3. Permission to talk: state purpose, ask if 1 min convenient; if not, schedule callback
        # Merge parent CRM tools so corrections (e.g. vehicle model) can be recorded during the task.
        crm_tools = [t for t in self.tools if getattr(t, "id", None) in ("record_crm_correction", "record_car_sold")]
        logger.info("Assistant on_enter: starting PermissionToTalkTask")
        permission = await PermissionToTalkTask(
            chat_ctx=self.chat_ctx,
            dealership_name=call_context.get("dealership_name") or "our dealership",
            brand=call_context.get("brand") or "the brand",
            car_model=call_context.get("car_model") or "their vehicle",
            number_ending=call_context.get("number_ending") or "",
            reason_for_call=call_context.get("reason_for_call") or "service reminder",
            last_service_date=call_context.get("last_service_date"),
            phone_number=call_context.get("phone_number"),
            contact_id=call_context.get("contact_id"),
            extra_tools=crm_tools,
        )
        logger.info("Assistant on_enter: PermissionToTalkTask finished, convenient=%s", permission.convenient)
        if not permission.convenient:
            # Callback scheduled: close using phrase in user's language (LLM provides speech_phrase)
            when_phrase = _callback_when_for_speech(permission)
            await self.session.generate_reply(
                instructions=f"Brief closing in user's language: we will call at {when_phrase}. Thank and goodbye. Nothing else.",
            )
            self.session.shutdown()
            return

        # 4. Soft engagement: ask about performance and issues; note any for technician
        logger.info("Assistant on_enter: starting SoftEngagementTask")
        soft_result = await SoftEngagementTask(
            chat_ctx=self.chat_ctx,
            car_model=call_context.get("car_model") or "their vehicle",
            contact_id=call_context.get("contact_id"),
            phone_number=call_context.get("phone_number"),
            extra_tools=crm_tools,
        )
        logger.info("Assistant on_enter: SoftEngagementTask finished, issues=%s", soft_result.issues)

        # 5. Single transition: value-add + offer help (one reply for smooth handoff)
        await self.session.generate_reply(
            instructions="User's language: one short value-add (genuine parts, trained technicians, or pickup/drop) and one short line offering help with service or booking. Keep to two sentences total.",
        )
        logger.info("Assistant on_enter: value-add + greeting sent, main conversation ready")

    # Commented out: was running twice with SoftEngagement; issues are deferred to DB on disconnect via pending_contact_notes.
    # @function_tool
    # async def note_car_issue(self, context: RunContext, issue: str) -> None:
    #     """Record a new vehicle issue mentioned after the performance check. Call only for issues the user brings up now; do not re-record issues already captured in that check."""
    #     issue = (issue or "").strip()
    #     if not issue:
    #         return
    #     call_context = self._call_context
    #     await add_contact_note(
    #         content=issue,
    #         source="assistant",
    #         contact_id=call_context.get("contact_id"),
    #         phone_number=call_context.get("phone_number"),
    #     )
    #     logger.info("Assistant: noted car issue via tool: %s", issue[:80])


def _prewarm(proc: JobProcess) -> None:
    """Run once per process before any job. Preload VAD to avoid cold-start latency. DB pool is created async on first use in entrypoint."""
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.35)


server = AgentServer()
server.setup_fnc = _prewarm


@server.rtc_session(agent_name="my-agent")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Initialize DB pool so first DB op has no connection latency
    await init_db_connection()

    # TTS fixed to hi-IN: Sarvam Bulbul v3 handles Hinglish (code-mixed) text with this code.
    # pending_contact_notes: list of {content, source, contact_id, phone_number} flushed to DB on disconnect
    session_userdata: dict = {"detected_language": "en-IN", "pending_contact_notes": []}

    def on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final and ev.language:
            session_userdata["detected_language"] = ev.language

    # user_away_timeout: after this many seconds with no user speech, state becomes "away" (default ~15s).
    USER_AWAY_TIMEOUT_S = 18  # 15–20s to check if user is there
    STILL_THERE_CHECKS = 2  # number of "still there?" prompts before ending
    STILL_THERE_WAIT_S = 10  # seconds between checks

    session = AgentSession(
        stt=sarvam.STT(
            model="saaras:v3",
            language="unknown",
            mode="transcribe",
            high_vad_sensitivity=True,
            flush_signal=True,
        ),
        # OpenRouter: one extra hop, ~1s+ TTFT. For lower latency use direct OpenAI (set OPENAI_API_KEY) or local Ollama.
        # llm=openai.LLM.with_openrouter(model="openai/gpt-4o-mini"),
        llm=openai.LLM(model="gpt-4o-mini"),
        # llm=openai.LLM(model="gpt-4o-mini"),  # direct OpenAI, slightly lower TTFT
        # llm=openai.LLM.with_ollama(model="llama3.2"),  # local, no network latency; needs Ollama running
        tts=sarvam.TTS(
            model="bulbul:v3",
            target_language_code=TTS_LANGUAGE,
            speaker=AGENT_NAME.lower(),
            pace=1.1,
            speech_sample_rate=16000,
        ),
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        userdata=session_userdata,
        preemptive_generation=False,
        min_endpointing_delay=0.3,
        max_endpointing_delay=1.5,
        user_away_timeout=USER_AWAY_TIMEOUT_S,
    )
    session.on("user_input_transcribed", on_user_input_transcribed)

    inactivity_task: asyncio.Task | None = None

    async def _user_away_sequence() -> None:
        """After user went away: 2 'still there?' checks, then shutdown."""
        nonlocal inactivity_task
        try:
            for i in range(STILL_THERE_CHECKS):
                await session.generate_reply(
                    instructions="The user has been inactive. Politely ask once if they are still there, in the user's language. One short sentence.",
                )
                if i < STILL_THERE_CHECKS - 1:
                    await asyncio.sleep(STILL_THERE_WAIT_S)
            session.shutdown()
        except asyncio.CancelledError:
            pass
        finally:
            inactivity_task = None

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
        nonlocal inactivity_task
        if ev.new_state == "away":
            inactivity_task = asyncio.create_task(_user_away_sequence())
            return
        if inactivity_task is not None:
            inactivity_task.cancel()
            inactivity_task = None

    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)


    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    async def flush_pending_notes():
        pending = session.userdata.get("pending_contact_notes") or []
        for entry in pending:
            await add_contact_note(
                content=entry["content"],
                source=entry["source"],
                contact_id=entry.get("contact_id"),
                phone_number=entry.get("phone_number"),
                note_type=entry.get("note_type", "car_issue"),
            )
        if pending:
            logger.info("Flushed %d pending contact note(s) to DB on disconnect", len(pending))

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(flush_pending_notes)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if (
            ev.new_state == "speaking"
            and last_eou_metrics
            and session.current_speech
            and last_eou_metrics.speech_id == session.current_speech.id
        ):
            # EOUMetrics uses timestamp (float, epoch seconds); created_at is also float
            delta_s = ev.created_at - last_eou_metrics.timestamp
            logger.info("Time to first audio frame: %sms", round(delta_s * 1000))

    call_context = load_call_context()
    await session.start(
        room=ctx.room,
        agent=Assistant(call_context=call_context),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    # await session.generate_reply(
    #     instructions="Greet the user and offer your assistance with automobile support."
    # )


if __name__ == "__main__":
    agents.cli.run_app(server)
