"""MultiStageSolve playbook - sequential stages with focused evaluation."""

import logging
from dataclasses import dataclass, field
from typing import Callable, List

from model_router.router import ModelRouter
from quality_control.playbook import Playbook
from quality_control.types import (
    CritiqueRequest,
    MultiStageState,
    PlaybookConfig,
    QCChecklistItem,
    QCFeedbackWithChecklist,
    QCResult,
    QCState,
    StageDefinition,
)
from quality_control.control_playbook.global_solve import (
    GlobalSolveConfig,
    GlobalSolvePlaybook,
)
from quality_control.control_playbook.swarm_solve import (
    SwarmSolveConfig,
    SwarmSolvePlaybook,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MultiStageConfig(PlaybookConfig):
    """Configuration for MultiStageSolve playbook.

    Attributes:
        stages: List of stage definitions to execute sequentially.
        feedback_decoder: Function to parse LLM responses to QCFeedbackWithChecklist.
        on_stage_change: Optional callback invoked when moving to a new stage.
                        Receives (stage_name, stage_index).
    """

    stages: List[StageDefinition] = field(default_factory=list)
    # Decoder function for parsing LLM text to QCFeedbackWithChecklist
    # Signature: (text: str, model: str) -> QCFeedbackWithChecklist
    feedback_decoder: Callable[[str, str], QCFeedbackWithChecklist] | None = None
    # Callback when stage changes (MultiStageSolve-specific)
    on_stage_change: Callable[[str, int], None] | None = None


class MultiStageSolvePlaybook(Playbook):
    """Playbook that executes QC in sequential stages.

    This playbook:
    1. Receives a list of stage definitions
    2. Executes each stage sequentially
    3. Each stage can use GlobalSolve or SwarmSolve internally
    4. Content flows from stage to stage
    5. Feedback is accumulated across all stages

    Example stages: Structural -> Vocabulary -> Formatting
    """

    def __init__(
        self, config: MultiStageConfig, router: ModelRouter | None = None
    ):
        """Initialize the MultiStageSolve playbook.

        Args:
            config: MultiStageSolve configuration with stage definitions.
            router: Optional ModelRouter instance to use for API calls.
        """
        super().__init__(config, router=router)
        self.config: MultiStageConfig = config

    def run(
        self,
        request: CritiqueRequest,
        revise_fn: Callable[[str, QCFeedbackWithChecklist], str],
        state: QCState | None = None,
        on_feedback: Callable[[QCFeedbackWithChecklist, int], None] | None = None,
    ) -> QCResult:
        """Run quality control through sequential stages.

        Args:
            request: The critique request containing content, context, and control_guide.
                     Note: The control_guide from request is used as a fallback if
                     a stage doesn't define its own control_guide.
            revise_fn: Function to revise content based on feedback.
            state: Optional MultiStageState for restart recovery.
            on_feedback: Optional callback invoked after each feedback is parsed.

        Returns:
            QCResult with final content and feedback history from all stages.

        Note:
            For stage change notifications, set on_stage_change in MultiStageConfig.
        """
        if not self.config.stages:
            raise ValueError(
                "MultiStageSolve requires at least one stage. "
                "Configure stages in MultiStageConfig."
            )

        # Initialize or restore state
        if state is None:
            multi_state = MultiStageState()
        elif isinstance(state, MultiStageState):
            multi_state = state
        else:
            # Convert basic QCState to MultiStageState
            multi_state = MultiStageState(
                feedback_history=state.feedback_history,
                content_history=state.content_history,
                iteration=state.iteration,
                accumulated_checklist=state.accumulated_checklist,
            )

        current_content = request.content
        default_context = request.context
        all_feedback: List[QCFeedbackWithChecklist] = list(multi_state.feedback_history)
        total_iterations = multi_state.iteration

        # Process each stage sequentially
        for stage_idx in range(multi_state.current_stage_index, len(self.config.stages)):
            stage = self.config.stages[stage_idx]
            multi_state.current_stage_index = stage_idx

            logger.info(f"Starting stage {stage_idx + 1}/{len(self.config.stages)}: {stage.name}")

            if self.config.on_stage_change:
                self.config.on_stage_change(stage.name, stage_idx)

            # Get or create stage-specific state
            stage_state = multi_state.stage_states.get(stage.name)
            if stage_state is None:
                stage_state = QCState()
                multi_state.stage_states[stage.name] = stage_state

            # Build context for this stage
            # Filter checklist to only include items from the current stage.
            # Prior stage items are locked decisions and must not be re-evaluated,
            # otherwise later stages can reopen issues they cannot satisfy
            # (e.g., simplification stage reopening additive foundation items).
            stage_title = stage.name.title()
            stage_checklist = [
                item
                for item in multi_state.accumulated_checklist
                if item.focus_area == stage_title
            ]
            if stage.context_builder:
                context = stage.context_builder(
                    current_content,
                    stage_checklist,
                    total_iterations,
                )
            else:
                context = default_context

            # Create stage-specific request
            stage_request = CritiqueRequest(
                content=current_content,
                context=context,
                control_guide=stage.control_guide,
            )

            # Create inner playbook based on stage configuration
            inner_playbook = self._create_inner_playbook(stage)

            # Create stage-aware feedback callback
            def stage_on_feedback(
                feedback: QCFeedbackWithChecklist, iteration: int
            ) -> None:
                # Tag feedback with stage info
                feedback.feedback.feedback = f"[{stage.name.upper()}] {feedback.feedback.feedback}"

                # Tag all checklist items with stage name (for UI display)
                for item in feedback.checklist:
                    if item.focus_area is None:
                        item.focus_area = stage.name.title()  # "Foundation", "Location", etc.

                all_feedback.append(feedback)
                multi_state.feedback_history = all_feedback

                # Track in stage history
                if stage.name not in multi_state.stage_feedback_history:
                    multi_state.stage_feedback_history[stage.name] = []
                multi_state.stage_feedback_history[stage.name].append(feedback)

                # Call original callback
                if on_feedback:
                    on_feedback(feedback, total_iterations + iteration)

            # Override max_iterations based on stage config
            original_max_iter = inner_playbook.config.max_iterations
            if stage.iterate_until_proceed:
                inner_playbook.config.max_iterations = stage.max_stage_iterations
            else:
                inner_playbook.config.max_iterations = 1

            # Run the stage
            stage_result = inner_playbook.run(
                request=stage_request,
                revise_fn=revise_fn,
                state=stage_state,
                on_feedback=stage_on_feedback,
            )

            # Restore original max_iterations
            inner_playbook.config.max_iterations = original_max_iter

            # Update content and state
            current_content = stage_result.content
            total_iterations += stage_result.iterations
            multi_state.iteration = total_iterations
            multi_state.accumulated_checklist = stage_result.feedback_history[-1].checklist if stage_result.feedback_history else []

            # Log stage result
            if stage_result.approved:
                logger.info(f"Stage '{stage.name}' approved after {stage_result.iterations} iterations")
            else:
                logger.warning(
                    f"Stage '{stage.name}' reached max iterations ({stage.max_stage_iterations}) "
                    "without approval, continuing to next stage"
                )

        # Final result - approved only if last stage approved
        last_stage_result_approved = True
        if self.config.stages:
            last_stage = self.config.stages[-1]
            last_stage_history = multi_state.stage_feedback_history.get(last_stage.name, [])
            if last_stage_history:
                last_stage_result_approved = last_stage_history[-1].action == "proceed"

        return QCResult(
            content=current_content,
            approved=last_stage_result_approved,
            iterations=total_iterations,
            feedback_history=all_feedback,
        )

    def _create_inner_playbook(self, stage: StageDefinition) -> Playbook:
        """Create the inner playbook for a stage.

        Args:
            stage: The stage definition.

        Returns:
            A Playbook instance (GlobalSolve or SwarmSolve).

        Raises:
            ValueError: If playbook_type is not supported.
        """
        if stage.playbook_type == "GlobalSolve":
            # Ensure config is GlobalSolveConfig
            if isinstance(stage.playbook_config, GlobalSolveConfig):
                config = stage.playbook_config
            else:
                # Create GlobalSolveConfig with defaults
                config = GlobalSolveConfig(
                    max_iterations=stage.max_stage_iterations,
                    feedback_decoder=self.config.feedback_decoder,
                )
            return GlobalSolvePlaybook(config, router=self.router)

        elif stage.playbook_type == "SwarmSolve":
            # Ensure config is SwarmSolveConfig
            if isinstance(stage.playbook_config, SwarmSolveConfig):
                config = stage.playbook_config
            else:
                raise ValueError(
                    f"Stage '{stage.name}' uses SwarmSolve but playbook_config "
                    "is not SwarmSolveConfig"
                )
            return SwarmSolvePlaybook(config, router=self.router)

        else:
            raise ValueError(f"Unsupported playbook_type: {stage.playbook_type}")
