"""Soft engagement task: ask about car performance and any issues; note issues for technician."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from livekit.agents import AgentTask, function_tool

from db import add_contact_note as db_add_contact_note

logger = logging.getLogger(__name__)


@dataclass
class SoftEngagementResult:
    """Result of SoftEngagementTask: list of issues reported (may be empty)."""
    issues: list[str]


class SoftEngagementTask(AgentTask[SoftEngagementResult]):
    """
    Ask how the car is performing and if they have any issues (noise, mileage, brake, AC).
    If they mention issues, note each with note_issue and say technician will specially check.
    If no issues, say regular servicing keeps life and resale value. Then complete.
    """

    def __init__(
        self,
        *,
        chat_ctx=None,
        car_model: str = "their vehicle",
        contact_id: str | None = None,
        phone_number: str | None = None,
    ) -> None:
        # Task instructions: flow and behavior only. When to call which tool is in each tool's docstring (LLM sees those).
        super().__init__(
            instructions="Ask about car performance and any issues (noise, mileage, brake, AC). Speak in the user's language (e.g. Hinglish). If they mention an issue, say the technician will specially check that point. If no issues, say regular servicing keeps vehicle life and resale value. When the exchange is complete, finish the task. Do not repeat the question.",
            chat_ctx=chat_ctx,
        )
        self._car_model = car_model
        self._contact_id = contact_id
        self._phone_number = phone_number
        self._issues: list[str] = []

    async def on_enter(self) -> None:
        car = (self._car_model or "their vehicle").strip()
        logger.info("SoftEngagementTask on_enter: asking about performance and issues for car=%s", car)
        await self.session.generate_reply(
            instructions=f"""Ask in the user's language (e.g. Hinglish): How is the car performing? Any issue — noise, mileage, brake, AC? Keep it to one short question."""
        )
        logger.info("SoftEngagementTask: performance question sent, waiting for user response")

    @function_tool
    async def note_issue(self, description: str) -> None:
        """Call when the user mentions a vehicle issue (noise, mileage, brake, AC, etc.). Use a short description."""
        desc = (description or "").strip()
        if not desc:
            logger.debug("SoftEngagementTask: note_issue called with empty description, skipping")
            return
        self._issues.append(desc)
        # Reply first so user hears immediately; DB write runs after (non-blocking for UX)
        await self.session.generate_reply(
            instructions="Say one short line in the user's language (e.g. Hinglish): technician will specially check that point. Nothing else. Then ask if any other issue."
        )
        logger.info("SoftEngagementTask: noted issue %s (total %d)", desc, len(self._issues))
        t0 = time.perf_counter()
        await db_add_contact_note(
            content=desc,
            source="soft_engagement",
            contact_id=self._contact_id,
            phone_number=self._phone_number,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("SoftEngagementTask: add_contact_note took %.0fms", elapsed_ms)

    @function_tool
    async def done_no_issues(self) -> None:
        """Call when the user has no issues and you have said the regular servicing line. Completes the task."""
        logger.info("SoftEngagementTask: done_no_issues -> completing with empty list")
        self.complete(SoftEngagementResult(issues=[]))

    @function_tool
    async def done_with_issues(self) -> None:
        """Call when you have noted all issues the user mentioned and acknowledged. Completes the task."""
        logger.info("SoftEngagementTask: done_with_issues -> completing with issues=%s", self._issues)
        self.complete(SoftEngagementResult(issues=list(self._issues)))
