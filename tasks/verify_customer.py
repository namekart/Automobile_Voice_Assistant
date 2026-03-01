import logging
from livekit.agents import AgentTask, function_tool

logger = logging.getLogger(__name__)


class VerifyCustomerTask(AgentTask[bool]):
    """Verify we are speaking with the intended customer. Returns True if confirmed, False if wrong person/wrong number."""
    def __init__(self, *, chat_ctx=None, customer_name: str = "the customer") -> None:
        super().__init__(
            instructions="""
            Verify you are speaking with the right customer. Get a clear yes or no.
            Be polite, concise, and professional. Speak in Hinglish.
            """,
            chat_ctx=chat_ctx,
        )
        self._customer_name = customer_name

    async def on_enter(self) -> None:
        name = self._customer_name.strip() or "the customer"
        logger.info("VerifyCustomerTask on_enter: asking for verification for customer_name=%s", name)
        await self.session.generate_reply(
            instructions=f"""
            Briefly greet, then ask if you are speaking with {name} ji.
            """
        )
        logger.info("VerifyCustomerTask: verification question sent, waiting for user response")

    @function_tool
    async def customer_verified(self) -> None:
        """Use this when the user confirms they are the right person."""
        logger.info("VerifyCustomerTask: customer_verified called -> completing with True")
        self.complete(True)

    @function_tool
    async def customer_not_verified(self) -> None:
        """Use this when the user says wrong person or wrong number."""
        logger.info("VerifyCustomerTask: customer_not_verified called -> completing with False")
        self.complete(False)  