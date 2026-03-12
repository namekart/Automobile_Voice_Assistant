"""Verify we're speaking with the intended customer. Three outcomes: verified, wrong_number, not_available (relative)."""
import logging
from dataclasses import dataclass

from livekit.agents import AgentTask, function_tool

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
            instructions="""Single yes/no question: are you speaking with the right customer?
One sentence max per reply. No pleasantries, no filler. User's language (Hinglish).
If unclear or not heard → re-ask in one short line.
verified → customer_verified(). Wrong person → customer_not_verified(). Right person not available → customer_not_available(relation=...).""",
            chat_ctx=chat_ctx,
        )
        self._customer_name = customer_name

    async def on_enter(self) -> None:
        name = self._customer_name.strip() or "the customer"
        logger.info("VerifyCustomerTask on_enter: asking for verification for customer_name=%s", name)
        await self.session.generate_reply(
            instructions=f"Exactly one sentence: ask if you are speaking with {name} ji. No greeting, no filler. User's language (Hinglish).",
        )
        logger.info("VerifyCustomerTask: verification question sent, waiting for user response")

    @function_tool
    async def customer_verified(self) -> None:
        """Call only when user clearly confirms they are the right person (the customer)."""
        logger.info("VerifyCustomerTask: customer_verified called -> completing with verified=True")
        self.complete(VerifyResult.verified_result())

    @function_tool
    async def customer_not_verified(self) -> None:
        """Call only when user clearly says wrong number or wrong person (not the right contact)."""
        logger.info("VerifyCustomerTask: customer_not_verified called -> completing with wrong_number=True")
        self.complete(VerifyResult.wrong_number_result())

    @function_tool
    async def customer_not_available(self, relation: str = "") -> None:
        """Call when user says the customer is not on the line but they are a relative/family (e.g. wife, husband, son). Pass who is speaking (relation)."""
        relation = (relation or "").strip() or "relative"
        logger.info("VerifyCustomerTask: customer_not_available called -> completing with not_available=True relation=%s", relation)
        self.complete(VerifyResult.not_available_result(relation=relation))
