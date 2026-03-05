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
            instructions="""Get a clear yes or no: are you speaking with the right customer?
If user says they didn't hear, want repeat, or unclear → re-ask in one short line (user's language). Do NOT call any tool until you have a clear answer.
If user confirms they are the customer → customer_verified().
If user says wrong number or wrong person (not this contact) → customer_not_verified().
If user says the customer is not on the line but they are a relative (e.g. wife, husband, son) → customer_not_available(relation="wife").
Be polite, concise. Speak in user's language (e.g. Hinglish).""",
            chat_ctx=chat_ctx,
        )
        self._customer_name = customer_name

    async def on_enter(self) -> None:
        name = self._customer_name.strip() or "the customer"
        logger.info("VerifyCustomerTask on_enter: asking for verification for customer_name=%s", name)
        await self.session.generate_reply(
            instructions=f"One short greeting, then ask: kya main {name} ji se baat kar raha hoon? (or same in user's language). Nothing else.",
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
