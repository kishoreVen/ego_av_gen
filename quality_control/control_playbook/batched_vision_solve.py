"""BatchedVisionSolve playbook - parallel vision evaluation across multiple images."""

import copy
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from PIL import Image

from model_router.query import Query, StructuredPrompt
from model_router.router import Capability, ModelRouter
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

logger = logging.getLogger(__name__)


@dataclass
class VisionEvaluationItem:
    """A single image to evaluate with its reference mapping.

    Attributes:
        image: The image to evaluate (PIL Image or base64 string).
        reference_keys: Keys into reference_images dict for this image.
                       If None, all reference images are included.
        id: Optional identifier for this image (used in feedback tracking).
        context: Optional per-image context that gets appended to the global query context.
                Use this to provide image-specific information like page outlines or scripts.
    """

    image: Image.Image | str
    reference_keys: List[str] | None = None
    id: str | None = None
    context: str | None = None


@dataclass
class VisionCritiqueRequest(CritiqueRequest):
    """Critique request with vision-specific fields.

    Extends CritiqueRequest to include images for batched vision evaluation.

    Attributes:
        images: List of images to evaluate with their reference mappings.
        reference_images: Dictionary of reference images (name -> image).
    """

    images: List[VisionEvaluationItem] = field(default_factory=list)
    reference_images: Dict[str, Image.Image | str] = field(default_factory=dict)


@dataclass(kw_only=True)
class BatchedVisionSolveConfig(PlaybookConfig):
    """Configuration for BatchedVisionSolve playbook.

    Attributes:
        feedback_decoder: Function to parse LLM text response into QCFeedbackWithChecklist.
                         Signature: (text: str, model: str) -> QCFeedbackWithChecklist
        model_interface: Model interface to use for vision evaluation.
        max_workers: Maximum number of parallel workers for batch processing.
    """

    feedback_decoder: Callable[[str, str], QCFeedbackWithChecklist]
    model_interface: str = "anthropic_claude4"
    max_workers: int = 4


@dataclass
class VisionFeedbackItem:
    """Feedback for a single image evaluation.

    Attributes:
        image_id: Identifier for the image (index or provided id).
        feedback: The structured QC feedback for this image.
    """

    image_id: str
    feedback: QCFeedbackWithChecklist


class BatchedVisionSolvePlaybook(Playbook):
    """Playbook for batched vision evaluation across multiple images.

    This playbook:
    1. Receives a batch of images with optional per-image reference mappings
    2. Runs the same query on each image in parallel
    3. Returns structured QC feedback for each image
    4. Aggregates results into a single QCResult

    Unlike other playbooks, this doesn't iterate on revisions - it performs
    a single evaluation pass across all images. The revise_fn parameter is
    accepted for interface compatibility but not used.

    Example usage:
        config = BatchedVisionSolveConfig(
            feedback_decoder=my_decoder_fn,
            model_interface="anthropic_claude4",
            max_workers=4,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",  # Not used for batched vision
            context="",  # Additional context for evaluation
            control_guide=my_system_prompt,
            images=[
                VisionEvaluationItem(image=img1, reference_keys=["ref1"]),
                VisionEvaluationItem(image=img2, reference_keys=["ref1", "ref2"]),
            ],
            reference_images={"ref1": ref_img1, "ref2": ref_img2},
        )

        result = playbook.run(request, revise_fn=lambda x, y: x)
    """

    def __init__(
        self, config: BatchedVisionSolveConfig, router: ModelRouter | None = None
    ):
        """Initialize the BatchedVisionSolve playbook.

        Args:
            config: BatchedVisionSolve configuration.
            router: Optional ModelRouter instance to use for API calls.
        """
        super().__init__(config, router=router)
        self.config: BatchedVisionSolveConfig = config

    def run(
        self,
        request: CritiqueRequest,
        revise_fn: Callable[[str, QCFeedbackWithChecklist], str],
        state: QCState | None = None,
        on_feedback: Callable[[QCFeedbackWithChecklist, int], None] | None = None,
    ) -> QCResult:
        """Run batched vision evaluation.

        Args:
            request: The critique request. Must be a VisionCritiqueRequest with
                    images and reference_images populated.
            revise_fn: Function to revise content (not used but required by interface).
            state: Optional state for tracking (used for restart recovery).
            on_feedback: Optional callback invoked after each image is evaluated.

        Returns:
            QCResult with aggregated feedback from all images.
            - content: JSON-serialized list of per-image feedback
            - approved: True if all images passed, False otherwise
            - iterations: Always 1 (single evaluation pass)
            - feedback_history: Aggregated feedback from all images

        Raises:
            TypeError: If request is not a VisionCritiqueRequest.
            ValueError: If no images provided in request.
        """
        if not isinstance(request, VisionCritiqueRequest):
            raise TypeError(
                "BatchedVisionSolvePlaybook requires a VisionCritiqueRequest. "
                f"Got {type(request).__name__} instead."
            )

        if not request.images:
            raise ValueError(
                "VisionCritiqueRequest must contain at least one image to evaluate."
            )

        if state is None:
            state = QCState()

        logger.info(
            f"Starting batched vision evaluation of {len(request.images)} images"
        )

        # Evaluate all images in parallel
        image_feedbacks = self._evaluate_batch(
            images=request.images,
            reference_images=request.reference_images,
            query_context=request.context,
            control_guide=request.control_guide,
            on_feedback=on_feedback,
        )

        # Aggregate results
        aggregated_feedback = self._aggregate_results(image_feedbacks)
        state.feedback_history.append(copy.deepcopy(aggregated_feedback))

        # Determine overall approval (all images must pass)
        all_approved = all(
            item.feedback.action == "proceed" for item in image_feedbacks
        )

        # Serialize per-image feedback to content for downstream processing
        import json

        content_data = [
            {
                "image_id": item.image_id,
                "action": item.feedback.action,
                "feedback": item.feedback.feedback.feedback,
                "checklist": [c.to_dict() for c in item.feedback.checklist],
            }
            for item in image_feedbacks
        ]
        result_content = json.dumps(content_data, indent=2)

        logger.info(
            f"Batched vision evaluation complete: "
            f"{sum(1 for i in image_feedbacks if i.feedback.action == 'proceed')}/{len(image_feedbacks)} passed"
        )

        return QCResult(
            content=result_content,
            approved=all_approved,
            iterations=1,
            feedback_history=state.feedback_history,
        )

    def _evaluate_batch(
        self,
        images: List[VisionEvaluationItem],
        reference_images: Dict[str, Image.Image | str],
        query_context: str,
        control_guide: StructuredPrompt,
        on_feedback: Callable[[QCFeedbackWithChecklist, int], None] | None = None,
    ) -> List[VisionFeedbackItem]:
        """Evaluate all images in parallel.

        Args:
            images: List of images to evaluate.
            reference_images: Dictionary of reference images.
            query_context: Query context/prompt for evaluation.
            control_guide: System prompt for evaluation.
            on_feedback: Optional callback invoked after each image is evaluated.

        Returns:
            List of VisionFeedbackItem with feedback for each image.
        """
        results: List[VisionFeedbackItem] = []

        max_workers = min(self.config.max_workers, len(images))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(
                    self._evaluate_single_image,
                    image_item,
                    idx,
                    reference_images,
                    query_context,
                    control_guide,
                ): (image_item, idx)
                for idx, image_item in enumerate(images)
            }

            for future in as_completed(future_to_image):
                image_item, idx = future_to_image[future]
                image_id = image_item.id or f"image_{idx}"

                try:
                    feedback = future.result()
                    result_item = VisionFeedbackItem(
                        image_id=image_id,
                        feedback=feedback,
                    )
                    results.append(result_item)

                    if on_feedback:
                        on_feedback(feedback, idx)

                    logger.debug(
                        f"Evaluated {image_id}: action={feedback.action}, "
                        f"checklist_items={len(feedback.checklist)}"
                    )

                except Exception as e:
                    logger.error(f"Error evaluating {image_id}: {e}")
                    # Create error feedback
                    error_feedback = QCFeedbackWithChecklist(
                        feedback=QCFeedback(
                            action="revise",
                            feedback=f"Error during evaluation: {str(e)}",
                            model=self.config.model_interface,
                        ),
                        checklist=[
                            QCChecklistItem(
                                id=f"{image_id}_error",
                                description=f"Evaluation failed: {str(e)}",
                                done_when="Evaluation completes successfully",
                                priority="P0",
                            )
                        ],
                    )
                    results.append(
                        VisionFeedbackItem(image_id=image_id, feedback=error_feedback)
                    )

        # Sort by original index to maintain order
        results.sort(
            key=lambda x: int(x.image_id.split("_")[1])
            if x.image_id.startswith("image_")
            else 0
        )

        return results

    def _evaluate_single_image(
        self,
        image_item: VisionEvaluationItem,
        idx: int,
        reference_images: Dict[str, Image.Image | str],
        query_context: str,
        control_guide: StructuredPrompt,
    ) -> QCFeedbackWithChecklist:
        """Evaluate a single image with the vision model.

        Args:
            image_item: The image to evaluate with its reference mapping.
            idx: Index of this image in the batch.
            reference_images: Dictionary of all reference images.
            query_context: Query context/prompt for evaluation.
            control_guide: System prompt for evaluation.

        Returns:
            QCFeedbackWithChecklist for this image.
        """
        # Determine which reference images to include
        if image_item.reference_keys is not None:
            # Use only specified references
            selected_refs = {
                k: v for k, v in reference_images.items() if k in image_item.reference_keys
            }
        else:
            # Use all references
            selected_refs = reference_images

        # Build images list: target image + reference images
        all_images: Dict[str, Image.Image | str] = {}

        # Add the target image
        all_images["target"] = image_item.image

        # Add reference images with their keys
        for ref_key, ref_image in selected_refs.items():
            all_images[f"reference_{ref_key}"] = ref_image

        # Build query text with image context
        image_context_parts = [f"- Target image to evaluate"]
        for ref_key in selected_refs.keys():
            image_context_parts.append(f"- Reference image: {ref_key}")

        # Combine global context with per-image context
        combined_context = query_context
        if image_item.context:
            combined_context = f"{query_context}\n\n{image_item.context}"

        image_context = "\n".join(image_context_parts)
        query_text = f"""Images provided:
{image_context}

{combined_context}"""

        # Create query with images
        query = Query(
            structured_prompt=control_guide,
            query_text=query_text,
            images=all_images,
            temperature=random.uniform(0.2, 0.4),
            top_p=random.uniform(0.6, 0.8),
            top_k=random.randint(20, 40),
        )

        # Call the model
        response = self.router.get_response(
            query=query,
            capability=Capability.IMAGE_ENC,
            interface_type=self.config.model_interface,
        )

        # Parse response using configured decoder
        feedback = self.config.feedback_decoder(
            response["text"], self.config.model_interface
        )

        # Track costs
        feedback.critique_cost_usd = response.get("cost", 0.0)
        usage = response.get("usage", {})
        feedback.critique_input_tokens = usage.get("input_tokens", 0)
        feedback.critique_output_tokens = usage.get("output_tokens", 0)

        return feedback

    def _aggregate_results(
        self,
        image_feedbacks: List[VisionFeedbackItem],
    ) -> QCFeedbackWithChecklist:
        """Aggregate feedback from all images into a single result.

        Args:
            image_feedbacks: List of feedback from each image.

        Returns:
            Aggregated QCFeedbackWithChecklist summarizing all results.
        """
        if not image_feedbacks:
            return QCFeedbackWithChecklist(
                feedback=QCFeedback(
                    action="proceed",
                    feedback="No images to evaluate",
                    model=self.config.model_interface,
                ),
                checklist=[],
            )

        # Collect all checklist items with image prefixes
        all_checklist_items: List[QCChecklistItem] = []
        all_feedback_texts: List[str] = []
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        for item in image_feedbacks:
            # Prefix checklist items with image id
            for checklist_item in item.feedback.checklist:
                prefixed_item = QCChecklistItem(
                    id=f"{item.image_id}_{checklist_item.id}",
                    description=f"[{item.image_id}] {checklist_item.description}",
                    done_when=checklist_item.done_when,
                    priority=checklist_item.priority,
                    completed=checklist_item.completed,
                    completed_at_iteration=checklist_item.completed_at_iteration,
                    focus_area=item.image_id,
                )
                all_checklist_items.append(prefixed_item)

            # Collect feedback text
            if item.feedback.feedback.feedback:
                all_feedback_texts.append(
                    f"[{item.image_id}] {item.feedback.feedback.feedback}"
                )

            # Accumulate costs
            total_cost += item.feedback.critique_cost_usd
            total_input_tokens += item.feedback.critique_input_tokens
            total_output_tokens += item.feedback.critique_output_tokens

        # Determine overall action
        passed = sum(1 for i in image_feedbacks if i.feedback.action == "proceed")
        failed = len(image_feedbacks) - passed
        action = "proceed" if failed == 0 else "revise"

        # Combine feedback text
        summary = f"Evaluated {len(image_feedbacks)} images: {passed} passed, {failed} failed"
        combined_feedback = f"{summary}\n\n" + "\n\n".join(all_feedback_texts)

        return QCFeedbackWithChecklist(
            feedback=QCFeedback(
                action=action,
                feedback=combined_feedback,
                model=self.config.model_interface,
            ),
            checklist=all_checklist_items,
            critique_cost_usd=total_cost,
            critique_input_tokens=total_input_tokens,
            critique_output_tokens=total_output_tokens,
        )
