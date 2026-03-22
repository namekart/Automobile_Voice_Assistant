"""Verify we're speaking with the intended customer. Three outcomes: verified, wrong_number, not_available (relative)."""
import logging
from dataclasses import dataclass

from livekit.agents import AgentTask, function_tool
from tasks import TASK_GUARDRAILS

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    """Result of verification: exactly one of verified, wrong_number, or not_available is True."""
    verified: bool = False
    wrong_number: bool = False
    not_available: bool = False
    relation: str = ""  # e.g. "wife" when not_available

    @classmethod
    def verified_result(cls) -> "VerifyResult":
        return cls(verified=True)

    @classmethod
    def wrong_number_result(cls) -> "VerifyResult":
        return cls(wrong_number=True)

    @classmethod
    def not_available_result(cls, relation: str = "") -> "VerifyResult":
        return cls(not_available=True, relation=(relation or "").strip())


class VerifyCustomerTask(AgentTask[VerifyResult]):
    """Verify we're speaking with the intended customer. Returns VerifyResult (verified / wrong_number / not_available)."""
 
    def __init__(self, *, chat_ctx=None, customer_name: str = "the customer") -> None:
        super().__init__(
            # Earlier instructions:
            # instructions="""Single yes/no question: are you speaking with the right customer?
            # One sentence max per reply. No pleasantries, no filler. User's language (Hinglish).
            # If unclear or not heard → re-ask in one short line.
            # verified → customer_verified(). Wrong person → customer_not_verified(). Right person not available → customer_not_available(relation=...).""",
            instructions="""Task: Confirm you are speaking with the right customer. One sentence max per reply. Hinglish.
If unclear/not heard → re-ask in one short line.
Tools: customer_verified() ONLY when user clearly confirms identity. customer_not_verified() ONLY when user clearly says wrong number/wrong person. customer_not_available(relation=...) ONLY when user says the named person is unavailable and a relative is speaking.
NEVER call any tool on ambiguous replies (hmm, what, huh). NEVER respond to off-topic — re-ask the verification question. If user is angry, acknowledge once briefly ("Main samajh sakta hoon"), then re-ask.\n""" + TASK_GUARDRAILS,
            chat_ctx=chat_ctx,
        )
        self._customer_name = customer_name
 
    async def on_enter(self) -> None:
        name = self._customer_name.strip() or "the customer"
        logger.info("VerifyCustomerTask on_enter: asking for verification for customer_name=%s", name)
        # Earlier: generate_reply (LLM + TTS ~500-800ms). Now session.say (TTS only ~200ms).
        # await self.session.generate_reply(
        #     instructions=f"Exactly one sentence: ask if I am speaking with {name} ji. (For eg. kya main {name} ji se baat kar raha hu?) No greeting, no filler. User's language (Hinglish).",
        # )
        await self.session.say(
            f"Kya main {name} ji se baat kar raha hoon?"
        )
        logger.info("VerifyCustomerTask: verification question sent, waiting for user response")
 
    @function_tool 
    async def customer_verified(self, unused: str = "") -> None:
        """Call ONLY when user clearly confirms they are the named customer. NEVER call on ambiguous replies like 'hmm' or 'what?'."""
        logger.info("VerifyCustomerTask: customer_verified called -> completing with verified=True")
        self.complete(VerifyResult.verified_result())
 
    @function_tool
    async def customer_not_verified(self, unused: str = "") -> None:
        """Call ONLY when user explicitly says wrong number or wrong person. NEVER call if user is confused or asking to repeat."""
        logger.info("VerifyCustomerTask: customer_not_verified called -> completing with wrong_number=True")
        self.complete(VerifyResult.wrong_number_result())
 
    @function_tool 
    async def customer_not_available(self, relation: str = "") -> None:
        """Call when user says the customer is not on the line but they are a relative/family (e.g. wife, husband, son). Pass who is speaking (relation). NEVER call if the named person is present but just busy — that is not 'unavailable'."""
        relation = (relation or "").strip() or "relative"
        logger.info("VerifyCustomerTask: customer_not_available called -> completing with not_available=True relation=%s", relation)
        self.complete(VerifyResult.not_available_result(relation=relation))
