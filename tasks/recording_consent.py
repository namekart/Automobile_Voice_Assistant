import logging
from livekit.agents import AgentTask, function_tool

logger = logging.getLogger(__name__)


class RecordingConsentTask(AgentTask[bool]):
    """Ask for recording consent after a brief introduction."""

    def __init__(
        self,
        *,
        chat_ctx=None,
        agent_name: str = "Shubh",
        dealership_name: str = "our dealership",
    ):
        super().__init__(
            instructions="""Introduce yourself briefly and aAsk for recording consent. Get a clear yes or no.
If user says didn't hear, want repeat, or unclear → re-ask in one short line (user's language). Do not call tools until you have a clear answer.
Be polite, concise. Speak in user's language (e.g. Hinglish).""",
            chat_ctx=chat_ctx,
        )
        self._agent_name = agent_name
        self._dealership_name = dealership_name

    async def on_enter(self) -> None:
        agent = self._agent_name.strip() or "Shubh"
        dealer = self._dealership_name.strip() or "our dealership"
        logger.info("RecordingConsentTask on_enter: intro + recording consent for agent=%s, dealer=%s", agent, dealer)
        await self.session.generate_reply(
            instructions=f"Short intro: {agent} from {dealer}. Ask permission to record the call for quality and training; Make it clear that they can decline. Keep it brief.",
        )
        logger.info("RecordingConsentTask: intro + consent question sent, waiting for user response")

    @function_tool
    async def consent_given(self) -> None:
        """Call only when user clearly gives consent to record (yes, haan, theek hai, etc.)."""
        logger.info("RecordingConsentTask: consent_given called -> completing with True")
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """Call only when user clearly denies consent (no, nahi, etc.). Do not use for unclear or repeat requests."""
        logger.info("RecordingConsentTask: consent_denied called -> completing with False")
        self.complete(False)
