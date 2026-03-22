"""When customer is not on the line but a relative is: offer speak-to-me or call-back-later. Single question, single LLM+tool turn."""
import logging
from livekit.agents import AgentTask, function_tool
from tasks import TASK_GUARDRAILS

logger = logging.getLogger(__name__)


class RelativeChoiceTask(AgentTask[bool]):
    """Ask relative: hear reason and pass to customer, or we call back later. Returns True to continue, False to call back."""

    def __init__(self, *, chat_ctx=None, customer_name: str = "the customer") -> None:
        super().__init__(
            # Earlier instructions:
            # instructions="""Get a clear choice. One question only.
            # If user says they didn't hear or unclear → re-ask in one short line. Do not call a tool until clear.
            # If user wants to hear the reason and pass it to the customer (e.g. "bol do", "tell me", "speak to me") → continue_with_relative().
            # If user wants us to call back later (e.g. "baad mein call karo", "call later") → call_back_later().
            # Speak in user's language. One reply then one tool call.""",
            instructions="""Task: Get clear choice — hear reason or call back later. Hinglish. One question only.
If unclear → re-ask in one short line.
Tools: continue_with_relative() ONLY when user wants to hear reason (e.g. "bol do", "tell me"). call_back_later() ONLY when user wants callback (e.g. "baad mein call karo").
NEVER call tool without clear answer. NEVER respond to off-topic — re-ask the choice. If angry, acknowledge once, then re-ask.\n""" + TASK_GUARDRAILS,
            chat_ctx=chat_ctx,
        )
        self._customer_name = customer_name

    async def on_enter(self) -> None:
        name = (self._customer_name or "the customer").strip()
        logger.info("RelativeChoiceTask on_enter: asking speak-to-me or call-back for customer=%s", name)
        # Earlier: generate_reply (LLM + TTS ~500-800ms). Now session.say (TTS only ~200ms).
        # await self.session.generate_reply(
        #     instructions=f"One short question in user's language: We were calling for {name}. Would you like to hear the reason and pass it to him, or should we call back when he's available? Nothing else.",
        # )
        await self.session.say(
            f"Hum {name} ji ke liye call kar rahe the. Kya main aapko reason bata doon taaki aap unhe bata sakein, ya phir hum baad mein call kar lein?"
        )
        logger.info("RelativeChoiceTask: question sent, waiting for user response")

    @function_tool
    async def continue_with_relative(self, unused: str = "") -> None:
        """Call ONLY when user clearly wants to hear the reason and pass it to the customer (e.g. speak to me, tell me, bol do). NEVER call on ambiguous replies."""
        logger.info("RelativeChoiceTask: continue_with_relative called -> completing with True")
        self.complete(True)

    @function_tool
    async def call_back_later(self, unused: str = "") -> None:
        """Call ONLY when user clearly wants us to call back later (e.g. call later, baad mein karo). NEVER call on ambiguous replies."""
        logger.info("RelativeChoiceTask: call_back_later called -> completing with False")
        self.complete(False)
