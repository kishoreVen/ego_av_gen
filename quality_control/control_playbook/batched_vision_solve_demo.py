"""Simple runnable demo for BatchedVisionSolve playbook.

Run with: python -m quality_control.control_playbook.batched_vision_solve_demo
"""

import json
import os
from pathlib import Path

from PIL import Image

from model_router.query import StructuredPrompt
from quality_control.control_playbook.batched_vision_solve import (
    BatchedVisionSolveConfig,
    BatchedVisionSolvePlaybook,
    VisionCritiqueRequest,
    VisionEvaluationItem,
)
from quality_control.types import (
    QCChecklistItem,
    QCFeedback,
    QCFeedbackWithChecklist,
)

# Path to test assets
TEST_ASSET_DIR = Path(__file__).parent.parent / "test_asset"


def decode_feedback(text: str, model: str) -> QCFeedbackWithChecklist:
    """Decode LLM response into QCFeedbackWithChecklist."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return QCFeedbackWithChecklist(
            feedback=QCFeedback(
                action="revise",
                feedback=f"Failed to parse: {text[:200]}",
                model=model,
            ),
            checklist=[],
        )

    checklist = [
        QCChecklistItem(
            id=item.get("id", f"item_{i}"),
            description=item.get("description", ""),
            done_when=item.get("done_when", ""),
            priority=item.get("priority", "P1"),
        )
        for i, item in enumerate(data.get("checklist", []))
    ]

    return QCFeedbackWithChecklist(
        feedback=QCFeedback(
            action=data.get("action", "revise"),
            feedback=data.get("feedback", ""),
            model=model,
        ),
        checklist=checklist,
    )


def main():
    print("=" * 60)
    print("BatchedVisionSolve Demo")
    print("=" * 60)

    # Load test images
    target_path = TEST_ASSET_DIR / "Target.png"
    car_path = TEST_ASSET_DIR / "Car.png"
    dino_path = TEST_ASSET_DIR / "Dino.png"

    print(f"\nLoading images from: {TEST_ASSET_DIR}")
    print(f"  - Target: {target_path.exists()}")
    print(f"  - Car: {car_path.exists()}")
    print(f"  - Dino: {dino_path.exists()}")

    # Load and ensure images are in PNG-compatible format
    def load_as_png(path):
        img = Image.open(path)
        # Convert to RGB if RGBA (strip alpha for compatibility)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        # Force PNG format attribute
        img.format = 'PNG'
        return img

    target_img = load_as_png(target_path)
    car_img = load_as_png(car_path)
    dino_img = load_as_png(dino_path)

    print(f"\nImage sizes:")
    print(f"  - Target: {target_img.size}")
    print(f"  - Car: {car_img.size}")
    print(f"  - Dino: {dino_img.size}")

    # Create system prompt
    control_guide = StructuredPrompt(
        base_instruction="""You are evaluating an illustration for a children's picture book.
Compare the target image against the reference character images provided.""",
        sections={
            "Evaluation Criteria": """
1. Character Consistency: Do the characters in the target match their reference images?
2. Style Consistency: Is the art style consistent across the image?
3. Quality: Is the image clear and well-rendered?
""",
            "Output Format": """
Respond with valid JSON in this exact format:
{
    "action": "proceed" or "revise",
    "feedback": "Brief overall assessment",
    "checklist": [
        {
            "id": "unique_id",
            "description": "Issue description",
            "done_when": "How to verify it's fixed",
            "priority": "P0" | "P1" | "P2"
        }
    ]
}
""",
        },
        requirements=[
            "Only output valid JSON, no other text",
            "Use 'proceed' if image is acceptable, 'revise' if issues found",
        ],
    )

    # Create config - using Claude Sonnet 4 for vision
    config = BatchedVisionSolveConfig(
        feedback_decoder=decode_feedback,
        model_interface="anthropic_sonnet4",
        max_workers=1,  # Single worker for demo
    )

    # Create playbook
    playbook = BatchedVisionSolvePlaybook(config)

    # Create request with single image
    request = VisionCritiqueRequest(
        content="",
        context="Evaluate this illustration page for character consistency.",
        control_guide=control_guide,
        images=[
            VisionEvaluationItem(
                image=target_img,
                id="page_1",
                reference_keys=["char_car", "char_dino"],
                context="Page 1: Car and Dino are playing together in the park.",
            ),
        ],
        reference_images={
            "char_car": car_img,
            "char_dino": dino_img,
        },
    )

    print("\n" + "-" * 60)
    print("Request Details:")
    print("-" * 60)
    print(f"  Images to evaluate: {len(request.images)}")
    for i, img_item in enumerate(request.images):
        print(f"    [{i}] id={img_item.id}, ref_keys={img_item.reference_keys}")
        if img_item.context:
            print(f"        context: {img_item.context[:60]}...")
    print(f"  Reference images: {list(request.reference_images.keys())}")
    print(f"  Global context: {request.context[:60]}...")

    print("\n" + "-" * 60)
    print("Running BatchedVisionSolve...")
    print("-" * 60)

    # Run the playbook
    result = playbook.run(request, revise_fn=lambda x, y: x)

    print(f"\nResult:")
    print(f"  - Approved: {result.approved}")
    print(f"  - Iterations: {result.iterations}")
    print(f"  - Feedback history length: {len(result.feedback_history)}")

    if result.feedback_history:
        feedback = result.feedback_history[0]
        print(f"\nAggregated Feedback:")
        print(f"  - Action: {feedback.action}")
        print(f"  - Feedback: {feedback.feedback.feedback[:200]}...")
        print(f"  - Checklist items: {len(feedback.checklist)}")
        for item in feedback.checklist[:5]:
            print(f"    - [{item.priority}] {item.id}: {item.description[:50]}...")

    print(f"\nRaw content:\n{result.content[:500]}...")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
