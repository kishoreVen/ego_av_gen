"""Quality control pipeline implementation."""

import logging
from typing import Callable

from model_router.router import ModelRouter
from quality_control.playbook import Playbook
from quality_control.control_playbook import PLAYBOOK_REGISTRY
from quality_control.types import (
    CritiqueRequest,
    QCFeedbackWithChecklist,
    QCResult,
    QCState,
    QualityControlConfig,
)

logger = logging.getLogger(__name__)


class QualityControlPipeline:
    """Pipeline for iterative critique-and-revise loops."""

    def __init__(
        self, config: QualityControlConfig, router: ModelRouter | None = None
    ):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration.
            router: Optional ModelRouter instance to use for API calls.
                Pass a router with default_service_tier="flex" for batch jobs.
        """
        self.config = config
        self.router = router
        self.playbook = self._create_playbook()

    def _create_playbook(self) -> Playbook:
        """Create the playbook instance based on config.

        Returns:
            Playbook instance.

        Raises:
            ValueError: If playbook name is not in registry.
        """
        playbook_name = self.config.playbook
        if playbook_name not in PLAYBOOK_REGISTRY:
            available = ", ".join(PLAYBOOK_REGISTRY.keys())
            raise ValueError(
                f"Unknown playbook '{playbook_name}'. Available: {available}"
            )
        playbook_class = PLAYBOOK_REGISTRY[playbook_name]
        return playbook_class(self.config.playbook_config, router=self.router)

    def run(
        self,
        request: CritiqueRequest,
        revise_fn: Callable[[str, QCFeedbackWithChecklist], str],
        state: QCState | None = None,
        on_feedback: Callable[[QCFeedbackWithChecklist, int], None] | None = None,
    ) -> QCResult:
        """Run quality control loop until approved or max iterations.

        Args:
            request: The critique request containing content, context, and control_guide.
            revise_fn: Function to revise content based on feedback.
                       Receives just the content string and feedback,
                       returns the revised content string.
            state: Optional state for restart recovery.
            on_feedback: Optional callback invoked after each feedback is parsed.

        Returns:
            QCResult with final content (just the content, not context)
            and feedback history.
        """
        return self.playbook.run(
            request=request,
            revise_fn=revise_fn,
            state=state,
            on_feedback=on_feedback,
        )
