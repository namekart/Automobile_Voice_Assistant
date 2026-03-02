"""Soft engagement: ask about car performance and issues; save combined note for technician."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from livekit.agents import AgentTask, function_tool

from db import add_contact_note as db_add_contact_note

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
    ) -> None:
        super().__init__(
            instructions="Ask about performance and issues (noise, mileage, brake, AC). User's language. If they list issues: acknowledge briefly, ask any other. If no or no more: wrap up. When calling the completion tool for issues, say in the same turn (user's language) that technician will see these. Call exactly one completion tool. No thank you or goodbye.",
            chat_ctx=chat_ctx,
        )
        self._car_model = car_model
        self._contact_id = contact_id
        self._phone_number = phone_number
        self._completed = False

    async def on_enter(self) -> None:
        car = (self._car_model or "their vehicle").strip()
        logger.info("SoftEngagementTask on_enter: asking about performance and issues for car=%s", car)
        await self.session.generate_reply(
            instructions="One short question in user's language: Car performance kaisa hai? Koi issue — noise, mileage, brake, AC?",
        )
        logger.info("SoftEngagementTask: performance question sent, waiting for user response")

    @function_tool
    async def done_with_issues(self, issues: list[str]) -> None:
        """Call when user listed one or more issues and is done. Say in the same turn (user's language) that technician will see these; then call this. Pass list of short issue strings. Do not call if user has no issues."""
        raw = [s.strip() for s in (issues or []) if isinstance(s, str) and s.strip()]
        if not raw:
            logger.debug("SoftEngagementTask: done_with_issues with no issues, completing empty")
            self._completed = True
            self.complete(SoftEngagementResult(issues=[]))
            return
        combined = "; ".join(raw)
        logger.info("SoftEngagementTask: done_with_issues saving %d issue(s): %s", len(raw), combined[:80])
        t0 = time.perf_counter()
        await db_add_contact_note(
            content=combined,
            source="soft_engagement",
            contact_id=self._contact_id,
            phone_number=self._phone_number,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("SoftEngagementTask: add_contact_note took %.0fms", elapsed_ms)
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
