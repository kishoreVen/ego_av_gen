"""SwarmSolve playbook - parallel evaluation by focused models."""

import copy
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, NotRequired, TypedDict

from model_router.query import Query, StructuredPrompt
from model_router.router import Capability, ModelRouter
from quality_control.checklist import merge_checklists
from quality_control.playbook import Playbook
from quality_control.types import (
    CritiqueRequest,
    PlaybookConfig,
    QCChecklistItem,
    QCFeedback,
    QCFeedbackWithChecklist,
    QCResult,
    QCState,
)


class FocusedModelDict(TypedDict):
    """Dictionary format for focused model configuration."""

    interface: str  # Model interface name (e.g., "openai_gpt5")
    focus_area: str  # Focus area (e.g., "dialogue", "pacing", "character")
    # Optional context builder: (content, checklist, iteration) -> context string
    context_builder: NotRequired[Callable[[str, List["QCChecklistItem"], int], str]]
    # Optional per-model prompt (overrides default control_guide)
    control_guide: NotRequired[StructuredPrompt]


@dataclass(kw_only=True)
class SwarmSolveConfig(PlaybookConfig):
    """Configuration for SwarmSolve playbook."""

    # Decoder function for parsing LLM text to QCFeedbackWithChecklist
    # Signature: (text: str, model: str) -> QCFeedbackWithChecklist
    feedback_decoder: Callable[[str, str], QCFeedbackWithChecklist]

    focused_models: List[FocusedModelDict] = field(default_factory=list)

logger = logging.getLogger(__name__)


@dataclass
class _FocusedModel:
    """Internal representation of a focused model."""

    interface: str
    focus_area: str
    context_builder: Callable[[str, List[QCChecklistItem], int], str] | None = None
    control_guide: StructuredPrompt | None = None


@dataclass
class SwarmChecklistItem(QCChecklistItem):
    """Checklist item with focus area for SwarmSolve."""

    focus_area: str = ""  # Which focus area this item belongs to


@dataclass
class SwarmFeedback:
    """Feedback from a focused model evaluation with focus area."""

    feedback: QCFeedback
    checklist: List[SwarmChecklistItem] = field(default_factory=list)
    focus_area: str = ""  # Which focus area this feedback is for

    @property
    def action(self) -> str:
        """Delegate to inner feedback."""
        return self.feedback.action

    @property
    def model(self) -> str:
        """Delegate to inner feedback."""
        return self.feedback.model


class SwarmSolvePlaybook(Playbook):
    """Playbook that uses multiple focused models to evaluate in parallel.

    This playbook:
    1. Receives a collection of models with focus areas
    2. Assigns checklist items to focus areas
    3. Evaluates items per focus area in parallel
    4. Aggregates feedback and creates new checklist items per focus area
    """

    def __init__(self, config: SwarmSolveConfig, router: ModelRouter | None = None):
        """Initialize the SwarmSolve playbook.

        Args:
            config: SwarmSolve configuration.
            router: Optional ModelRouter instance to use for API calls.
        """
        super().__init__(config, router=router)
        self.config: SwarmSolveConfig = config  # Type hint for IDE
        # Convert config dicts to internal _FocusedModel objects
        self.focused_models = [
            _FocusedModel(
                interface=fm["interface"],
                focus_area=fm["focus_area"],
                context_builder=fm.get("context_builder"),
                control_guide=fm.get("control_guide"),
            )
            for fm in config.focused_models
        ]

    def run(
        self,
        request: CritiqueRequest,
        revise_fn: Callable[[str, QCFeedbackWithChecklist], str],
        state: QCState | None = None,
        on_feedback: Callable[[QCFeedbackWithChecklist, int], None] | None = None,
    ) -> QCResult:
        """Run quality control loop with parallel focused evaluation.

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

        Raises:
            ValueError: If no focused models configured.
        """
        if state is None:
            state = QCState()

        if not self.focused_models:
            raise ValueError(
                "SwarmSolve requires at least one focused model. "
                "Configure focused_models in SwarmSolveConfig."
            )

        # Lock models on first iteration for consistency tracking
        if not state.locked_model:
            models_str = ", ".join(
                f"{fm.focus_area}:{fm.interface}" for fm in self.focused_models
            )
            state.locked_model = models_str
            logger.info(f"Locked models for swarm critic loop: {models_str}")

        current_content = request.content
        context = request.context
        control_guide = request.control_guide
        approved = False

        # Restart recovery: check for pending revision
        if state.feedback_history:
            revise_feedbacks = [f for f in state.feedback_history if f.action == "revise"]
            if len(revise_feedbacks) > len(state.content_history):
                pending_feedback = revise_feedbacks[len(state.content_history)]
                state.content_history.append(current_content)
                current_content = revise_fn(current_content, pending_feedback)
                state.iteration += 1
                logger.info(
                    f"Applied pending revision from iteration {state.iteration}"
                )

        # Check if already approved
        if state.feedback_history and state.feedback_history[-1].action == "proceed":
            return QCResult(
                content=current_content,
                approved=True,
                iterations=state.iteration,
                feedback_history=state.feedback_history,
            )

        while state.iteration < self.config.max_iterations:
            logger.debug(
                f"Swarm QC iteration {state.iteration + 1}/{self.config.max_iterations}"
            )

            # Evaluate content in parallel across all focused models
            focus_feedbacks = self._evaluate_parallel(
                content=current_content,
                default_context=context,
                control_guide=control_guide,
                accumulated_checklist=state.accumulated_checklist,
                current_iteration=state.iteration,
            )

            # Aggregate feedback from all focus areas
            aggregated_feedback = self._aggregate_feedback(
                focus_feedbacks, state.iteration
            )

            # Invoke callback if provided
            if on_feedback:
                on_feedback(aggregated_feedback, state.iteration)

            # Update accumulated checklist
            state.accumulated_checklist = aggregated_feedback.checklist
            state.feedback_history.append(copy.deepcopy(aggregated_feedback))

            if aggregated_feedback.action == "proceed":
                logger.info("Swarm QC approved, proceeding")
                approved = True
                break

            # Apply revision
            logger.info(f"Applying revision (iteration {state.iteration + 1})")
            state.content_history.append(current_content)
            current_content = revise_fn(current_content, aggregated_feedback)
            state.iteration += 1

        if not approved:
            logger.warning(
                f"Swarm QC reached max iterations ({self.config.max_iterations}) "
                "without approval"
            )

        return QCResult(
            content=current_content,
            approved=approved,
            iterations=len(state.feedback_history),
            feedback_history=state.feedback_history,
        )

    def _evaluate_parallel(
        self,
        content: str,
        default_context: str,
        control_guide: StructuredPrompt,
        accumulated_checklist: List[QCChecklistItem],
        current_iteration: int,
    ) -> List[SwarmFeedback]:
        """Evaluate content in parallel across all focused models.

        Args:
            content: The content to evaluate.
            default_context: Default context used when model has no context_builder.
            control_guide: The control guide (system prompt) for evaluation.
            accumulated_checklist: Checklist from previous iterations.
            current_iteration: Current iteration number.

        Returns:
            List of SwarmFeedback from each focused model.
        """
        focus_feedbacks: List[SwarmFeedback] = []

        with ThreadPoolExecutor(max_workers=len(self.focused_models)) as executor:
            future_to_model = {
                executor.submit(
                    self._evaluate_single,
                    focused_model,
                    content,
                    default_context,
                    control_guide,
                    accumulated_checklist,
                    current_iteration,
                ): focused_model
                for focused_model in self.focused_models
            }

            for future in as_completed(future_to_model):
                focused_model = future_to_model[future]
                try:
                    feedback = future.result()
                    focus_feedbacks.append(feedback)
                except Exception as e:
                    logger.error(
                        f"Error evaluating with {focused_model.interface} "
                        f"({focused_model.focus_area}): {e}"
                    )

        return focus_feedbacks

    def _evaluate_single(
        self,
        focused_model: _FocusedModel,
        content: str,
        default_context: str,
        control_guide: StructuredPrompt,
        accumulated_checklist: List[QCChecklistItem],
        current_iteration: int,
    ) -> SwarmFeedback:
        """Evaluate content with a single focused model.

        Args:
            focused_model: The focused model to use.
            content: The content to evaluate.
            default_context: Default context used when model has no context_builder.
            control_guide: The control guide (system prompt) for evaluation.
            accumulated_checklist: Checklist from previous iterations.
            current_iteration: Current iteration number.

        Returns:
            SwarmFeedback from the focused model.
        """
        # Filter checklist items for this focus area
        focus_checklist = [
            item
            for item in accumulated_checklist
            if isinstance(item, SwarmChecklistItem)
            and item.focus_area == focused_model.focus_area
        ]

        # Use focus-specific context if context_builder is provided
        # Pass only this focus area's checklist items to prevent cross-contamination
        if focused_model.context_builder:
            context = focused_model.context_builder(
                content, focus_checklist, current_iteration
            )
        else:
            context = default_context

        # Use per-model control_guide if provided, otherwise use default
        effective_control_guide = focused_model.control_guide or control_guide

        # Build query text from context + content
        query_text = CritiqueRequest(
            content=content, context=context, control_guide=effective_control_guide
        ).query

        query = Query(
            structured_prompt=effective_control_guide,
            query_text=query_text,
            temperature=random.uniform(0.2, 0.4),
            top_p=random.uniform(0.6, 0.8),
            top_k=random.randint(20, 40),
        )

        response = self.router.get_response(
            query=query,
            capability=Capability.TEXT,
            interface_type=focused_model.interface,
        )

        # Parse the response using configured decoder
        feedback = self.config.feedback_decoder(response["text"], focused_model.interface)

        # Sanitize first iteration - reset any hallucinated completion data
        if current_iteration == 0:
            for item in feedback.checklist:
                item.completed = False
                item.completed_at_iteration = None

        # Convert checklist items to SwarmChecklistItem with focus area
        swarm_checklist = [
            SwarmChecklistItem(
                id=item.id,
                description=item.description,
                done_when=item.done_when,
                priority=item.priority,
                completed=item.completed,
                completed_at_iteration=item.completed_at_iteration,
                focus_area=focused_model.focus_area,
            )
            for item in feedback.checklist
        ]

        # Merge with previous checklist for this focus area
        # Pass router for LLM-based semantic deduplication
        if focus_checklist:
            swarm_checklist = merge_checklists(
                focus_checklist,
                swarm_checklist,
                current_iteration,
                router=self.router,
            )

        return SwarmFeedback(
            feedback=QCFeedback(
                action=feedback.action,
                feedback=feedback.feedback.feedback,
                model=focused_model.interface,
            ),
            checklist=swarm_checklist,
            focus_area=focused_model.focus_area,
        )

    def _aggregate_feedback(
        self,
        focus_feedbacks: List[SwarmFeedback],
        current_iteration: int,
    ) -> QCFeedbackWithChecklist:
        """Aggregate feedback from all focus areas.

        Args:
            focus_feedbacks: List of feedback from each focus area.
            current_iteration: Current iteration number.

        Returns:
            Aggregated QCFeedbackWithChecklist.
        """
        if not focus_feedbacks:
            return QCFeedbackWithChecklist(
                feedback=QCFeedback(action="proceed", feedback="No feedback received"),
                checklist=[],
            )

        # Aggregate checklists from all focus areas
        all_checklist_items: List[QCChecklistItem] = []
        all_feedback_texts: List[str] = []

        for fb in focus_feedbacks:
            all_checklist_items.extend(fb.checklist)
            if fb.feedback.feedback:
                all_feedback_texts.append(f"[{fb.focus_area}] {fb.feedback.feedback}")

        # Determine action: revise if ANY focus area says revise
        action = "proceed"
        for fb in focus_feedbacks:
            if fb.action == "revise":
                action = "revise"
                break

        # Enforce action based on P0/P1 items
        blocking = [
            i
            for i in all_checklist_items
            if not i.completed and i.priority in ("P0", "P1")
        ]
        if action == "proceed" and blocking:
            logger.warning(
                f"Overriding 'proceed' to 'revise': {len(blocking)} P0/P1 items remain"
            )
            action = "revise"
        elif action == "revise" and not blocking:
            logger.info("Overriding 'revise' to 'proceed': no P0/P1 items remain")
            action = "proceed"

        # Combine feedback text
        combined_feedback = "\n".join(all_feedback_texts)

        # Models used
        models_used = ", ".join(fb.model for fb in focus_feedbacks)

        return QCFeedbackWithChecklist(
            feedback=QCFeedback(
                action=action,
                feedback=combined_feedback,
                model=models_used,
            ),
            checklist=all_checklist_items,
        )
