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
            instructions="""
            Introduce yourself briefly and ask for recording consent and get a clear yes or no answer.
            Be polite and professional. Speak in Hinglish.
            """,
            chat_ctx=chat_ctx,
        )
        self._agent_name = agent_name
        self._dealership_name = dealership_name

    async def on_enter(self) -> None:
        agent = self._agent_name.strip() or "Shubh"
        dealer = self._dealership_name.strip() or "our dealership"
        logger.info("RecordingConsentTask on_enter: intro + recording consent for agent=%s, dealer=%s", agent, dealer)
        await self.session.generate_reply(
            instructions=f"""
            Briefly introduce yourself as {agent} from {dealer}, then ask for permission to record the call for quality assurance and training purposes.
            Make it clear that they can decline.
            """
        )
        logger.info("RecordingConsentTask: intro + consent question sent, waiting for user response")

    @function_tool
    async def consent_given(self) -> None:
        """Use this when the user gives consent to record."""
        logger.info("RecordingConsentTask: consent_given called -> completing with True")
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """Use this when the user denies consent to record."""
        logger.info("RecordingConsentTask: consent_denied called -> completing with False")
        self.complete(False)
