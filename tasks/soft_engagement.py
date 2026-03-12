"""Soft engagement: ask about car performance and issues; defer DB write to call end."""
import logging
from dataclasses import dataclass

from livekit.agents import AgentTask, function_tool

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
            instructions="Ask about performance and issues (noise, mileage, brake, AC). User's language. If they list issues: in one short reply say you have noted them and the technician will check; mention each issue naturally once, do not read back a long list; then call done_with_issues with the issue list. If no or no more issues: wrap up briefly and call done_no_issues. Call exactly one of done_with_issues or done_no_issues. Never say you are calling a tool or completing a step. No thank you or goodbye.",
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
        logger.info("SoftEngagementTask on_enter: asking about performance and issues for car=%s", car)
        await self.session.generate_reply(
            instructions="Ask one short, natural question in the user's language: how the car is performing and whether they have any issues (e.g. noise, mileage, brakes, AC).",
        )
        logger.info("SoftEngagementTask: performance question sent, waiting for user response")

    @function_tool
    async def done_with_issues(self, issues: list[str]) -> None:
        """Call when user listed one or more issues and is done. Same turn: say once (user's language) that you noted the issues and technician will check; then call this with the list. Do not announce any tool or step to the user. Do not call if user has no issues."""
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
    async def done_no_issues(self) -> None:
        """Call only when user has zero issues (said no issues at all). Do not call if they listed any issues."""
        if getattr(self, "_completed", False):
            logger.debug("SoftEngagementTask: done_no_issues skipped, task already complete")
            return
        await self.session.generate_reply(
            instructions="One line, user's language: regular servicing keeps vehicle life and resale value. Nothing else.",
        )
        logger.info("SoftEngagementTask: done_no_issues -> completing with empty list")
        self._completed = True
        self.complete(SoftEngagementResult(issues=[]))
