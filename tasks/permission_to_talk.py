"""Permission-to-talk task: state purpose, ask if convenient now; if not, schedule callback with a specific date (and optional time)."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from livekit.agents import AgentTask, function_tool

from db import schedule_callback as db_schedule_callback

logger = logging.getLogger(__name__)

# India Standard Time (UTC+5:30) so "today" is correct for the dealership
IST = timezone(timedelta(hours=5, minutes=30))


def _today_iso() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


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
        reason_script = self._reason_script(reason_for_call, last_service_date)
        super().__init__(
            instructions=f"""State reason for call. Ask if they have 1(ek) minute. Get a clear answer. User's language (e.g. Hinglish).
If user says didn't hear or unclear → re-ask in one short line. Only call tools when you have a clear answer.
Today's date: {today}. Resolve relative phrases (tomorrow, next week) to the 
correct calendar date; never use today when they mean later.
If they have time now and want to continue the conversation → user_has_time(). If busy → ask when to call back. When 
they give a time (or range like "one or two hours"): use the **latest** time 
they said, call schedule_callback **once** with callback_date (YYYY-MM-DD), 
optional callback_time (HH:MM 24h), preferred_raw, and speech_phrase (natural 
phrase in their language, no raw digits). Then confirm declaratively (e.g. "2 
baje call karunga" or "ek ghante baad call karunga").""",
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

    @staticmethod
    def _reason_script(reason: str, last_service_date: str | None) -> str:
        r = (reason or "").strip().lower().replace("-", "_")
        if "overdue" in r and last_service_date:
            return f"Aapka last service {last_service_date} ko hua tha, ab gaadi service ke liye due ho sakti hai."
        if "campaign" in r:
            return "Abhi hamare workshop mein complimentary health check-up camp chal raha hai."
        # service_due or default
        return "Hamare records ke according aapki gaadi ka periodic service due hai."

    async def on_enter(self) -> None:
        if self._extra_tools:
            await self.update_tools(list(self.tools) + self._extra_tools)
        dealer = self._dealership_name.strip() or "our dealership"
        brand = self._brand.strip() or "the brand"
        car = self._car_model.strip() or "their vehicle"
        ending = (self._number_ending or "").strip()
        number_line = f" (number ending {ending})" if ending else ""
        reason_line = self._reason_script_text
        logger.info(
            "PermissionToTalkTask on_enter: purpose + convenience check dealer=%s car=%s",
            dealer,
            car,
        )
        await self.session.generate_reply(
            instructions=f"""You are from {dealer}, authorized dealer for {brand}. Say you are calling regarding their {car}{number_line}. Then reason to call: {reason_line} Then ask: Kya abhi 1 minute baat karna convenient hoga?"""
        )
        logger.info("PermissionToTalkTask: purpose and convenience question sent, waiting for user response")

    @function_tool
    async def user_has_time(self) -> None:
        """Call when user clearly says they have time to talk now. Call this as soon as you get a yes; do not ask about appointments or scheduling in the same turn. Do not call if they said they are busy or want a callback later."""
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
        """Use when the user wants a callback later. Call only once; if they give a range (e.g. one or two hours), use the latest time. callback_date: YYYY-MM-DD. callback_time: optional HH:MM 24h. preferred_raw: what they said. speech_phrase: how to say when we'll call (e.g. kal subah, ek ghante baad)."""
        if self._callback_scheduled:
            logger.info("PermissionToTalkTask: schedule_callback ignored (already scheduled)")
            return
        callback_date = (callback_date or "").strip()
        if not callback_date:
            logger.warning("PermissionToTalkTask: schedule_callback called with empty callback_date")
            return
        self._callback_scheduled = True
        await db_schedule_callback(
            callback_date=callback_date,
            phone_number=self._phone_number,
            contact_id=self._contact_id,
            callback_time=(callback_time or "").strip() or None,
            preferred_raw=(preferred_raw or "").strip() or None,
        )
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
