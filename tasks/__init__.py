# Shared guardrail for all tasks: handle service-related but out-of-scope questions naturally.
TASK_GUARDRAILS = (
    "If user says something related to service but outside this task — acknowledge briefly, then continue your task.\n"
    "If user says hold on or wait — wait silently. If angry — acknowledge once, continue.\n"
)

from tasks.verify_customer import VerifyCustomerTask, VerifyResult
from tasks.recording_consent import RecordingConsentTask
from tasks.permission_to_talk import PermissionToTalkTask, PermissionResult
from tasks.relative_choice import RelativeChoiceTask
from tasks.soft_engagement import SoftEngagementTask, SoftEngagementResult

__all__ = [
    "VerifyCustomerTask",
    "VerifyResult",
    "RecordingConsentTask",
    "PermissionToTalkTask",
    "PermissionResult",
    "RelativeChoiceTask",
    "SoftEngagementTask",
    "SoftEngagementResult",
]