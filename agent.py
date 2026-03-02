from __future__ import annotations

import json
import logging
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
    function_tool,
    metrics,
    room_io,
    RunContext,
)

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
    RecordingConsentTask,
    PermissionToTalkTask,
    PermissionResult,
    SoftEngagementTask,
    SoftEngagementResult,
)
from db import init_db_connection, mark_phone_wrong, add_contact_note

load_dotenv()
AGENT_NAME = "Shubh"

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
    base = f"""You are {AGENT_NAME}, automobile support voice assistant for {dealership} ({brand}). Help with service, bookings, escalate when needed. Concise, clear. Respond in user's language (e.g. Hinglish). When user mentions a vehicle issue (e.g. "one more issue", "AC noise") → call note_car_issue to record it."""
    number_line = f" (number ending {ending})" if ending else ""
    reason_line = reason.replace("_", " ")
    if last_date:
        context_block = f"""
## This call
You are calling {customer} about their {car}{number_line}. Reason for call: {reason_line}. Last service was on {last_date}. Dealership: {dealership}, authorized dealer for {brand}. Use this information naturally in your conversation."""
    else:
        context_block = f"""
## This call
You are calling {customer} about their {car}{number_line}. Reason for call: {reason_line}. Dealership: {dealership}, authorized dealer for {brand}. Use this information naturally in your conversation."""
    return base + context_block


# Production pattern: STT converts speech → English (for RAG/LLM). We capture detected user
# language from STT and store it in session userdata so we can use it later for TTS (speak in
# user's language). Sarvam STT with language="unknown" returns language_code in the event.


class Assistant(Agent):
    def __init__(self, *, call_context: dict[str, str | None] | None = None) -> None:
        ctx = call_context if call_context is not None else load_call_context()
        self._call_context = ctx
        instructions = _build_instructions(ctx)
        super().__init__(instructions=instructions)

    async def on_enter(self) -> None:
        call_context = self._call_context
        customer_name = call_context.get("customer_name") or "the customer"

        # Remove note_car_issue so only SoftEngagementTask records issues during performance check (no duplicate DB/messages)
        note_car_issue_tool = next((t for t in self.tools if getattr(t, "name", None) == "note_car_issue"), None)
        if note_car_issue_tool:
            self._note_car_issue_tool = note_car_issue_tool
            await self.update_tools([t for t in self.tools if getattr(t, "name", None) != "note_car_issue"])

        logger.info("Assistant on_enter: starting VerifyCustomerTask (customer_name=%s)", customer_name)
        # 1. Verify we're speaking with the right person (task instructions active during task)
        verified = await VerifyCustomerTask(
            chat_ctx=self.chat_ctx,
            customer_name=customer_name,
        )
        logger.info("Assistant on_enter: VerifyCustomerTask finished, verified=%s", verified)
        if not verified:
            await self.session.generate_reply(
                instructions="Say only: Koi baat nahi, dhanyavaad. Nothing else.",
            )
            await mark_phone_wrong(
                call_context.get("phone_number"),
                call_context.get("contact_id"),
                reason="wrong_number",
            )
            self.session.shutdown()
            return

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
        )
        logger.info("Assistant on_enter: SoftEngagementTask finished, issues=%s", soft_result.issues)

        # Re-remove note_car_issue before value-add/greeting (framework may restore default tools when task ends)
        await self.update_tools([t for t in self.tools if getattr(t, "name", None) != "note_car_issue"])
        # 5. Value add then greeting (no note_car_issue available)
        await self.session.generate_reply(
            instructions="Value-add, user's language: genuine parts, trained technicians, pickup & drop, same-day when possible, complimentary washing. Two short sentences.",
        )
        logger.info("Assistant on_enter: sending greeting (main conversation)")
        await self.session.generate_reply(
            instructions="Greet and offer help with service or booking. One short line.",
        )
        # Re-enable note_car_issue for rest of call (new issues only; performance-check issues already saved)
        if getattr(self, "_note_car_issue_tool", None):
            await self.update_tools(self.tools + [self._note_car_issue_tool])
            del self._note_car_issue_tool

    @function_tool
    async def note_car_issue(self, context: RunContext, issue: str) -> None:
        """Record a new vehicle issue mentioned after the performance check. Call only for issues the user brings up now; do not re-record issues already captured in that check."""
        issue = (issue or "").strip()      
        if not issue:
            return
        call_context = self._call_context
        await add_contact_note( 
            content=issue,   
            source="assistant",
            contact_id=call_context.get("contact_id"),
            phone_number=call_context.get("phone_number"),
        )
        logger.info("Assistant: noted car issue via tool: %s", issue[:80])


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
    session_userdata: dict[str, str] = {"detected_language": "en-IN"}

    def on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final and ev.language:
            session_userdata["detected_language"] = ev.language

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
            speaker="shubh",
            pace=1.1,
            speech_sample_rate=8000,
        ),
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        userdata=session_userdata,
        preemptive_generation=False,
        min_endpointing_delay=0.3,
        max_endpointing_delay=1.5,
    )
    session.on("user_input_transcribed", on_user_input_transcribed)

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


    ctx.add_shutdown_callback(log_usage)

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
