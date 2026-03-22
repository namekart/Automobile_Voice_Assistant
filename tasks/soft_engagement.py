"""Soft engagement: ask about car performance and issues; defer DB write to call end."""
import logging
from dataclasses import dataclass

from livekit.agents import AgentTask, function_tool
from tasks import TASK_GUARDRAILS

logger = logging.getLogger(__name__)

 
@dataclass
class SoftEngagementResult:
    """Result: list of issues reported (may be empty)."""
    issues: list[str]


class SoftEngagementTask(AgentTask[SoftEngagementResult]):
    """Ask how the car is performing and if they have any issues. No thank you or goodbye; conversation continues after."""

    def __init__(
        self,
        *,
        chat_ctx=None,
        car_model: str = "their vehicle",
        contact_id: str | None = None,
        phone_number: str | None = None,
        extra_tools: list | None = None,
    ) -> None:
        super().__init__(
            # Earlier instructions:
            # instructions="Ask about car performance and any issues (noise, mileage, brake, AC). User's language (Hinglish). One question, one sentence.\n"
            # "If they report issues: in one short reply, acknowledge as the agent that you have noted the issues and use them as a reason to gently suggest service visit (e.g. इन्हें ठीक करवाने का यह सही समय है, service में सब check हो जाएगा). Do NOT promise any outcome or anything that guarantees a fix. Then call done_with_issues.\n"
            # "If no issues: in one short reply acknowledge briefly and add one natural line that regular servicing keeps the car in top shape and maintains resale value, then call done_no_issues.\n"
            # "Call exactly one of done_with_issues or done_no_issues. Never say you are calling a tool. No thank you or goodbye.",
            instructions="""Task: Learn about car performance and issues, then gently suggest service. Hinglish. Under 40 words per reply.
If issues reported: acknowledge the issue, highlight ONE dealership benefit (trained technicians, genuine parts, or pickup-drop), and ask if they would like you to check for a convenient slot. Silently invoke done_with_issues with the issues list IN THE SAME reply — do NOT wait for user to respond to the slot question.
If no issues: acknowledge briefly, mention regular servicing keeps car in top shape, and ask if they would like a routine checkup. Silently invoke done_no_issues IN THE SAME reply.
CRITICAL: You MUST invoke exactly ONE tool in your FIRST reply after user answers. NEVER write function names, parentheses, or code syntax in spoken text — tool calls are silent API actions. No thank you or goodbye.
NEVER invoke a tool without a clear user answer.\n""" + TASK_GUARDRAILS,
            chat_ctx=chat_ctx,
        )
        self._car_model = car_model
        self._contact_id = contact_id
        self._phone_number = phone_number
        self._completed = False
        self._extra_tools = list(extra_tools) if extra_tools else []

    async def on_enter(self) -> None:
        if self._extra_tools:
            await self.update_tools(list(self.tools) + self._extra_tools)
        car = (self._car_model or "their vehicle").strip()
        logger.info("SoftEngagementTask on_enter: car=%s", car)
        # Earlier: generate_reply (LLM + TTS ~500-800ms). Now session.say (TTS only ~200ms).
        # await self.session.generate_reply(
        #     instructions="Ask one short, natural question in the user's language: how the car is performing and whether they have any issues (e.g. noise, mileage, brakes, AC).",
        # )
        await self.session.say(
            f"Aapki {car} kaise perform kar rahi hai? Koi problem toh nahi aa rahi, jaise noise, mileage drop, brakes, ya AC mein?"
        )
        logger.info("SoftEngagementTask: performance question sent, waiting for user response")

    @function_tool
    async def done_with_issues(self, issues: list[str]) -> None:
        """Invoke ONLY after user has finished listing issues. Pass the list of issues mentioned (e.g. AC problem, brake noise). NEVER invoke while user is still mid-sentence. NEVER write this tool's name or syntax in your spoken text."""
        raw = [s.strip() for s in (issues or []) if isinstance(s, str) and s.strip()]
        if not raw:
            logger.debug("SoftEngagementTask: done_with_issues with no issues, completing empty")
            self._completed = True
            self.complete(SoftEngagementResult(issues=[]))
            return
        combined = "; ".join(raw)
        logger.info("SoftEngagementTask: done_with_issues storing %d issue(s) for later DB write: %s", len(raw), combined[:80])
        # Defer DB write to end of call (flush on disconnect)
        pending = self.session.userdata.get("pending_contact_notes", [])
        if not isinstance(pending, list):
            pending = []
        pending.append({
            "content": combined,
            "source": "soft_engagement",
            "contact_id": self._contact_id,
            "phone_number": self._phone_number,
            "note_type": "car_issue",
        })
        self.session.userdata["pending_contact_notes"] = pending
        self._completed = True
        self.complete(SoftEngagementResult(issues=raw))

    @function_tool
    async def done_no_issues(self, unused: str = "") -> None:
        """Invoke ONLY when user clearly says no issues (sab theek hai, koi problem nahi). NEVER invoke if user mentioned even one issue. NEVER write this tool's name or syntax in your spoken text."""
        if getattr(self, "_completed", False):
            logger.debug("SoftEngagementTask: done_no_issues skipped, task already complete")
            return
        logger.info("SoftEngagementTask: done_no_issues -> completing with empty list")
        self._completed = True
        self.complete(SoftEngagementResult(issues=[]))
