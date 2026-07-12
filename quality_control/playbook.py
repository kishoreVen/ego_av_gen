"""Base class for control playbooks."""

from abc import ABC, abstractmethod
from typing import Callable

from model_router.router import ModelRouter
from quality_control.types import (
    CritiqueRequest,
    PlaybookConfig,
    QCFeedbackWithChecklist,
    QCResult,
    QCState,
)


class Playbook(ABC):
    """Base class for control playbooks.

    A playbook defines the strategy for how the quality control loop
    evaluates and resolves checklist items.
    """

    def __init__(self, config: PlaybookConfig, router: ModelRouter | None = None):
        """Initialize the playbook.

        Args:
            config: Playbook configuration.
            router: Optional ModelRouter instance to use for API calls.
                Pass a router with default_service_tier="flex" for batch jobs.
        """
        self.config = config
        self.router = router or ModelRouter()

    @abstractmethod
    def run(
        self,
        request: CritiqueRequest,
        revise_fn: Callable[[str, QCFeedbackWithChecklist], str],
        state: QCState | None = None,
        on_feedback: Callable[[QCFeedbackWithChecklist, int], None] | None = None,
    ) -> QCResult:
        """Run the quality control loop.

        Args:
            request: The critique request containing content, context, and control_guide.
            revise_fn: Function to revise content based on feedback.
                       Receives the current content string and feedback,
                       returns the revised content string.
            state: Optional state for restart recovery.
            on_feedback: Optional callback invoked after each feedback is parsed.

        Returns:
            QCResult with final content (just the content, not context)
            and feedback history.
        """
        pass
