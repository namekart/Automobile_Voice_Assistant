"""When customer is not on the line but a relative is: offer speak-to-me or call-back-later. Single question, single LLM+tool turn."""
import logging
from livekit.agents import AgentTask, function_tool

logger = logging.getLogger(__name__)


class RelativeChoiceTask(AgentTask[bool]):
    """Ask relative: hear reason and pass to customer, or we call back later. Returns True to continue, False to call back."""

    def __init__(self, *, chat_ctx=None, customer_name: str = "the customer") -> None:
        super().__init__(
            instructions="""Get a clear choice. One question only.
If user says they didn't hear or unclear → re-ask in one short line. Do not call a tool until clear.
If user wants to hear the reason and pass it to the customer (e.g. "bol do", "tell me", "speak to me") → continue_with_relative().
If user wants us to call back later (e.g. "baad mein call karo", "call later") → call_back_later().
Speak in user's language. One reply then one tool call.""",
            chat_ctx=chat_ctx,
        )
        self._customer_name = customer_name

    async def on_enter(self) -> None:
        name = (self._customer_name or "the customer").strip()
        logger.info("RelativeChoiceTask on_enter: asking speak-to-me or call-back for customer=%s", name)
        await self.session.generate_reply(
            instructions=f"One short question in user's language: We were calling for {name}. Would you like to hear the reason and pass it to him, or should we call back when he's available? Nothing else.",
        )
        logger.info("RelativeChoiceTask: question sent, waiting for user response")

    @function_tool
    async def continue_with_relative(self) -> None:
        """Call when user wants to hear the reason for the call and pass it to the customer (e.g. speak to me, tell me, bol do)."""
        logger.info("RelativeChoiceTask: continue_with_relative called -> completing with True")
        self.complete(True)

    @function_tool
    async def call_back_later(self) -> None:
        """Call when user wants us to call back when the customer is available (e.g. call later, baad mein karo)."""
        logger.info("RelativeChoiceTask: call_back_later called -> completing with False")
        self.complete(False)
