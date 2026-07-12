"""GlobalSolve playbook - iterates until all checklist items are resolved."""

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, List

from model_router.query import Query, StructuredPrompt
from model_router.router import Capability, ModelRouter
from quality_control.checklist import merge_checklists
from quality_control.playbook import Playbook
from model_router.lib.cost_tracker import get_current_accumulator
from quality_control.types import (
    CritiqueRequest,
    PlaybookConfig,
    QCFeedbackWithChecklist,
    QCResult,
    QCState,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GlobalSolveConfig(PlaybookConfig):
    """Configuration for GlobalSolve playbook."""

    # Decoder function for parsing LLM text to QCFeedbackWithChecklist
    # Signature: (text: str, model: str) -> QCFeedbackWithChecklist
    feedback_decoder: Callable[[str, str], QCFeedbackWithChecklist]

    model_interfaces: List[str] = field(default_factory=lambda: ["openai_gpt5"])


class GlobalSolvePlaybook(Playbook):
    """Default playbook that critics and resolves checklist items until completion.

    This playbook:
    1. Evaluates content to get feedback and checklist items
    2. If action is "revise", calls revise_fn to update content
    3. Merges checklists across iterations to track progress
    4. Repeats until approved or max iterations reached
    """

    def __init__(self, config: GlobalSolveConfig, router: ModelRouter | None = None):
        """Initialize the GlobalSolve playbook.

        Args:
            config: GlobalSolve configuration.
            router: Optional ModelRouter instance to use for API calls.
        """
        super().__init__(config, router=router)
        self.config: GlobalSolveConfig = config  # Type hint for IDE

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
        if state is None:
            state = QCState()

        current_content = request.content
        context = request.context
        control_guide = request.control_guide
        approved = False

        # Restart recovery: check for pending revision
        if state.feedback_history:
            revise_feedbacks = [
                f for f in state.feedback_history if f.action == "revise"
            ]
            if len(revise_feedbacks) > len(state.content_history):
                # There's a pending revision to apply
                pending_feedback = revise_feedbacks[len(state.content_history)]
                state.content_history.append(current_content)
                # Track revision cost by measuring delta on current accumulator
                acc = get_current_accumulator()
                cost_before = acc.cost_usd if acc else 0.0
                input_before = acc.input_tokens if acc else 0
                output_before = acc.output_tokens if acc else 0
                current_content = revise_fn(current_content, pending_feedback)
                # Store revision cost on the feedback (delta from before)
                if acc:
                    pending_feedback.revision_cost_usd = acc.cost_usd - cost_before
                    pending_feedback.revision_input_tokens = acc.input_tokens - input_before
                    pending_feedback.revision_output_tokens = acc.output_tokens - output_before
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
            logger.info(
                f"QC iteration {state.iteration + 1}/{self.config.max_iterations}"
            )

            # Evaluate content (combine context + content for query)
            query_text = CritiqueRequest(
                content=current_content, context=context, control_guide=control_guide
            ).query
            feedback = self._evaluate(
                query_text=query_text,
                control_guide=control_guide,
                accumulated_checklist=state.accumulated_checklist,
                current_iteration=state.iteration,
                state=state,
            )

            # Invoke callback if provided
            if on_feedback:
                on_feedback(feedback, state.iteration)

            # Update accumulated checklist
            state.accumulated_checklist = feedback.checklist
            state.feedback_history.append(copy.deepcopy(feedback))

            if feedback.action == "proceed":
                logger.info("QC approved, proceeding")
                approved = True
                break

            # Apply revision with cost tracking
            logger.info(f"Applying revision (iteration {state.iteration + 1})")
            state.content_history.append(current_content)
            # Track revision cost by measuring delta on current accumulator
            acc = get_current_accumulator()
            cost_before = acc.cost_usd if acc else 0.0
            input_before = acc.input_tokens if acc else 0
            output_before = acc.output_tokens if acc else 0
            current_content = revise_fn(current_content, feedback)
            # Store revision cost on the feedback (delta from before)
            if acc:
                feedback.revision_cost_usd = acc.cost_usd - cost_before
                feedback.revision_input_tokens = acc.input_tokens - input_before
                feedback.revision_output_tokens = acc.output_tokens - output_before
            state.iteration += 1

        if not approved:
            logger.warning(
                f"QC reached max iterations ({self.config.max_iterations}) "
                "without approval"
            )

        return QCResult(
            content=current_content,
            approved=approved,
            iterations=len(state.feedback_history),
            feedback_history=state.feedback_history,
        )

    def _evaluate(
        self,
        query_text: str,
        control_guide: StructuredPrompt,
        accumulated_checklist: list,
        current_iteration: int,
        state: QCState,
    ) -> QCFeedbackWithChecklist:
        """Evaluate content and return feedback.

        Args:
            query_text: The combined context + content text for LLM query.
            control_guide: The control guide (system prompt) for evaluation.
            accumulated_checklist: Checklist from previous iterations.
            current_iteration: Current iteration number.
            state: QC state for model locking across iterations.

        Returns:
            QCFeedbackWithChecklist with action and checklist.
        """
        # Use locked model if set, otherwise select and lock
        if state.locked_model:
            interface = state.locked_model
        else:
            interface = random.choice(self.config.model_interfaces)
            state.locked_model = interface
            logger.info(f"Locked model for critic loop: {interface}")

        query = Query(
            structured_prompt=control_guide,
            query_text=query_text,
            temperature=random.uniform(0.2, 0.4),
            top_p=random.uniform(0.6, 0.8),
            top_k=random.randint(20, 40),
        )

        response = self.router.get_response(
            query=query,
            capability=Capability.TEXT,
            interface_type=interface,
        )

        # Parse the response using configured decoder
        feedback = self.config.feedback_decoder(response["text"], interface)

        # Capture critique cost from response
        feedback.critique_cost_usd = response.get("cost", 0.0)
        usage = response.get("usage", {})
        feedback.critique_input_tokens = usage.get("input_tokens", 0)
        feedback.critique_output_tokens = usage.get("output_tokens", 0)

        # Sanitize first iteration - reset any hallucinated completion data
        if current_iteration == 0:
            for item in feedback.checklist:
                item.completed = False
                item.completed_at_iteration = None

        # Merge checklists first (so blocking check uses full history)
        # Pass router for LLM-based semantic deduplication
        if accumulated_checklist:
            feedback.checklist = merge_checklists(
                accumulated_checklist,
                feedback.checklist,
                current_iteration,
                router=self.router,
            )

        # Enforce action based on P0/P1 items
        blocking_count = sum(
            1
            for i in feedback.checklist
            if not i.completed and i.priority in ("P0", "P1")
        )
        if feedback.action == "proceed" and blocking_count > 0:
            logger.warning(
                f"Overriding 'proceed' to 'revise': {blocking_count} P0/P1 items remain"
            )
            feedback.action = "revise"
        elif feedback.action == "revise" and blocking_count == 0:
            logger.info("Overriding 'revise' to 'proceed': no P0/P1 items remain")
            feedback.action = "proceed"

        # Ensure model is set
        feedback.model = interface

        return feedback
