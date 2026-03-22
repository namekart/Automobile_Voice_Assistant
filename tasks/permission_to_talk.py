"""Permission-to-talk task: state purpose, ask if convenient now; if not, schedule callback with a specific date (and optional time)."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from livekit.agents import AgentTask, function_tool
from tasks import TASK_GUARDRAILS

logger = logging.getLogger(__name__)

# India Standard Time (UTC+5:30) so "today" is correct for the dealership
IST = timezone(timedelta(hours=5, minutes=30))


def _today_iso() -> str:
    d = datetime.now(IST)
    return f"{d.day} {d.strftime('%B %Y (%A)')}"


def _reason_script_static(reason: str, last_service_date: str | None) -> str:
    """Build reason line for permission intro. Shared so assistant can build same prompt."""
    r = (reason or "").strip().lower().replace("-", "_")
    if "overdue" in r and last_service_date:
        return f"Aapka last service {last_service_date} ko hua tha, isliye ab service due ho sakti hai."
    if "campaign" in r:
        return "Abhi hamare workshop mein complimentary health check-up camp chal raha hai."
    return "Hamare records ke according aapki gaadi ka periodic service due hai."


def build_permission_intro_instruction(
    car_model: str,
    number_ending: str,
    reason_for_call: str,
    last_service_date: str | None,
) -> str:
    """Build the exact intro instruction used by PermissionToTalkTask on_enter. Used by assistant for latency optimization."""
    reason_line = _reason_script_static(reason_for_call, last_service_date)
    car = (car_model or "their vehicle").strip()
    ending = (number_ending or "").strip()
    number_line = f" (number ending {ending})" if ending else ""
    return (
        "Exactly two sentences in user's language (Hinglish). Do NOT re-introduce dealership or your name — already done. "
        f"(1) Say: Main aapki {car}{number_line} ke baare mein call kar raha hoon; {reason_line} "
        "(2) Ask: Kya abhi 1 (ek) minute baat karna convenient hoga? Nothing else."
    )

@dataclass
class PermissionResult:
    """Result of PermissionToTalkTask: either user has time now, or callback was scheduled."""
    convenient: bool
    callback_date: str | None = None   # YYYY-MM-DD when callback_scheduled
    callback_time: str | None = None   # HH:MM when callback_scheduled (or default applied)
    preferred_raw: str | None = None   # What user said (e.g. "tomorrow evening")
    speech_phrase: str | None = None   # Natural phrase in user's language for when we'll call (e.g. "kal subah", "agale hafte Monday")


class PermissionToTalkTask(AgentTask[PermissionResult]):
    """
    State why we're calling (car, number ending, reason), ask if 1 minute is convenient.
    If yes -> complete with convenient=True. If no -> ask when to call back, then call
    schedule_callback with a specific date (and optional time); default time 10-12 AM.
    """

    def __init__(
        self,
        *,
        chat_ctx=None,
        dealership_name: str = "our dealership",
        brand: str = "the brand",
        car_model: str = "their vehicle",
        number_ending: str = "",
        reason_for_call: str = "service reminder",
        last_service_date: str | None = None,
        phone_number: str | None = None,
        contact_id: str | None = None,
        extra_tools: list | None = None,
    ) -> None:
        today = _today_iso()
        reason_script = _reason_script_static(reason_for_call, last_service_date)
        car = (car_model or "their vehicle").strip()
        ending = (number_ending or "").strip()
        number_part = f", number ending {ending}," if ending else ""
        super().__init__(
            # Earlier instructions:
            # instructions=f"""User's language (Hinglish). Two sentences max per reply. No filler.
            # CRITICAL: Your FIRST action in this task must be to ask the convenience question ( state the reason for call + "kya abhi 1 minute..."). Do NOT call any tool until AFTER the user replies to that question in this task.
            # Do NOT re-introduce the dealership or agent name — already done (only clarify if the user is confused or explicitly asks).
            # If unclear → re-ask in one short line. Only call tools when you have a clear answer.
            # Today's date: {today}. Resolve relative phrases (tomorrow, next week) to the correct calendar date.
            # yes/time now → user_has_time(). Busy → ask when to call back, then schedule_callback once with callback_date (YYYY-MM-DD), optional callback_time (HH:MM 24h), preferred_raw, speech_phrase. Confirm in one sentence (e.g. "2 baje call karunga").""",
            instructions=f"""Task: Confirm if user has 1 minute now. Hinglish. Two sentences max. No re-introduction.
Today: {today}. Resolve relative dates (tomorrow, next week) against today.
Tools: user_has_time() ONLY when user clearly says yes/available/bolo. schedule_callback(callback_date, callback_time, preferred_raw, speech_phrase) ONLY when user says busy AND gives a preferred time — ask when first, then call once. callback_date: YYYY-MM-DD. callback_time: HH:MM 24h (optional).
When user gives a time: IMMEDIATELY call schedule_callback AND confirm naturally in the SAME reply (e.g. "ठीक है, आज शाम 6 बजे call करूँगा"). Do NOT announce the schedule and wait for "ok" — confirm and close in ONE turn. Always speak times naturally (शाम 6 बजे, not 18:00 baje). NEVER use 24h format in speech.
NEVER call tool without clear user answer. NEVER guess dates — ask if unclear. If unclear → re-ask. NEVER respond to off-topic — re-ask convenience question. If angry, acknowledge once, then re-ask.
If user asks which car, which vehicle, or konsi gaadi — confidently restate: "{car}{number_part}" — you already have this info, never say you don't know.\n""" + TASK_GUARDRAILS,
            chat_ctx=chat_ctx,
        )
        self._dealership_name = dealership_name
        self._brand = brand
        self._car_model = car_model
        self._number_ending = number_ending
        self._reason_script_text = reason_script
        self._phone_number = phone_number
        self._contact_id = contact_id
        self._callback_scheduled = False  # guard: only complete once
        self._extra_tools = list(extra_tools) if extra_tools else []

    async def on_enter(self) -> None:
        if self._extra_tools:
            await self.update_tools(list(self.tools) + self._extra_tools)
        car = self._car_model.strip() or "their vehicle"
        ending = (self._number_ending or "").strip()
        number_part = f", number ending {ending}," if ending else ""
        reason_line = self._reason_script_text
        logger.info(
            "PermissionToTalkTask on_enter: dealer=%s car=%s",
            self._dealership_name,
            self._car_model,
        )
        # Earlier: generate_reply (LLM + TTS ~500-800ms). Now session.say (TTS only ~200ms).
        # await self.session.generate_reply(
        #     instructions=(
        #         "Exactly two sentences in user's language (Hinglish). "
        #         "Do NOT call any tool in this turn. "
        #         "Do NOT re-introduce dealership or your name — already done. "
        #         f"(1) Say: Main aapki {car}{number_part} ke service ke baare mein call kar raha hoon; {reason_line} "
        #         "(2) Ask: Kya abhi 1 minute baat karna convenient hoga? Nothing else."
        #     )
        # )
        await self.session.say(
            f"Main aapki {car}{number_part} ke service ke baare mein call kar raha hoon. {reason_line} Kya abhi ek minute baat karna convenient hoga?"
        )
        logger.info("PermissionToTalkTask: purpose and convenience question sent, waiting for user response")

    @function_tool
    async def user_has_time(self, unused: str = "") -> None:
        """Call ONLY when user clearly says they have time now (yes, haan, bolo). NEVER call if user sounds hesitant or says 'jaldi bolo' — that may mean they're rushed, not consenting. Do not ask about appointments in this turn."""
        logger.info("PermissionToTalkTask: user_has_time -> completing with convenient=True")
        self.complete(PermissionResult(convenient=True))

    @function_tool
    async def schedule_callback(
        self,
        callback_date: str,
        callback_time: str | None = None,
        preferred_raw: str | None = None,
        speech_phrase: str | None = None,
    ) -> None:
        """Use when the user wants a callback later. Call only once; if they give a range (e.g. one or two hours), use the latest time. NEVER call with a guessed date — if user says 'later' without specifics, ask when first. callback_date: YYYY-MM-DD. callback_time: optional HH:MM 24h (for DB only). preferred_raw: what they said. speech_phrase: MUST be natural Hinglish (e.g. 'aaj shaam 6 baje', 'kal subah 10 baje') — NEVER use 24h format in speech_phrase."""
        if self._callback_scheduled:
            logger.info("PermissionToTalkTask: schedule_callback ignored (already scheduled)")
            return
        callback_date = (callback_date or "").strip()
        if not callback_date:
            logger.warning("PermissionToTalkTask: schedule_callback called with empty callback_date")
            return
        self._callback_scheduled = True
        # Defer DB write: store in session userdata, flush on disconnect (like pending_contact_notes)
        self.session.userdata["pending_callback"] = {
            "callback_date": callback_date,
            "phone_number": self._phone_number,
            "contact_id": self._contact_id,
            "callback_time": (callback_time or "").strip() or None,
            "preferred_raw": (preferred_raw or "").strip() or None,
        }
        phrase = (speech_phrase or "").strip() or None
        logger.info(
            "PermissionToTalkTask: schedule_callback saved date=%s time=%s speech_phrase=%s -> completing with convenient=False",
            callback_date,
            callback_time,
            phrase,
        )
        self.complete(
            PermissionResult(
                convenient=False,
                callback_date=callback_date,
                callback_time=callback_time,
                preferred_raw=preferred_raw,
                speech_phrase=phrase,
            )
        )
