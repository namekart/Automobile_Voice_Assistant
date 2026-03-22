import logging
from livekit.agents import AgentTask, function_tool
from tasks import TASK_GUARDRAILS

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
            # Earlier instructions:
            # instructions="""User's language (Hinglish/Hindi). Be polite and concise. Introduce yourself briefly and ask for permission to record for quality and training purposes. Make it explicit: if they say no, we will NOT record.
            # If unclear/not heard → repeat the question in one short line.
            # Your FIRST action in this task must be to introduce yourself and ask for recording consent. Do NOT call any tool until the user replies to that question in this task clearly.""",
            instructions="""Task: Get recording consent. Hinglish. Polite, concise.
If unclear/not heard → repeat consent question in one short line.
Tools: consent_given() ONLY on clear yes/haan/theek hai/chalo. consent_denied() ONLY on clear no/nahi/mat karo. NEVER call tool on ambiguous replies (hmm, what?).
NEVER respond to off-topic questions — re-ask consent question. If user is angry, acknowledge once, then re-ask.\n""" + TASK_GUARDRAILS,
            chat_ctx=chat_ctx,
        )
        self._agent_name = agent_name
        self._dealership_name = dealership_name

    async def on_enter(self) -> None:
        agent = self._agent_name.strip() or "Shubh"
        dealer = self._dealership_name.strip() or "our dealership"
        logger.info("RecordingConsentTask on_enter: agent=%s dealer=%s", agent, dealer)
        # Earlier: generate_reply (LLM + TTS ~500-800ms). Now session.say (TTS only ~200ms).
        # await self.session.generate_reply(
        #     instructions=(
        #         "Two short sentences max in user's language (Hinglish/Hindi). "
        #         f"(1) Say: Main {agent}, {dealer} se bol raha hoon. "
        #         "(2) Ask: Quality and training purposes ke liye kya main is call ko record kar skta hoon?"
        #     ),
        # )
        await self.session.say(
            f"Main {agent}, {dealer} se bol raha hoon. Quality and training purposes ke liye kya main is call ko record kar sakta hoon?"
        )
        logger.info("RecordingConsentTask: intro + consent question sent, waiting for user response")

    @function_tool
    async def consent_given(self, unused: str = "") -> None:
        """Call ONLY when user clearly gives consent to record (yes, haan, theek hai, chalo). NEVER call on ambiguous replies."""
        logger.info("RecordingConsentTask: consent_given called -> completing with True")
        self.complete(True)

    @function_tool
    async def consent_denied(self, unused: str = "") -> None:
        """Call ONLY when user clearly denies consent (no, nahi, mat karo). NEVER call on 'what?', 'repeat?', or unclear audio."""
        logger.info("RecordingConsentTask: consent_denied called -> completing with False")
        self.complete(False)
