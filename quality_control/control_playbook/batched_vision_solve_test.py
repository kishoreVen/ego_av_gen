"""Tests for BatchedVisionSolve playbook."""

import json
import unittest
from typing import Any, Dict
from unittest.mock import Mock

from PIL import Image

from model_router.interfaces.dummy_interface import DummyInterface
from model_router.query import StructuredPrompt
from quality_control.control_playbook.batched_vision_solve import (
    BatchedVisionSolveConfig,
    BatchedVisionSolvePlaybook,
    VisionCritiqueRequest,
    VisionEvaluationItem,
)
from quality_control.types import (
    CritiqueRequest,
    QCChecklistItem,
    QCFeedback,
    QCFeedbackWithChecklist,
)


def decode_feedback(text: str, model: str) -> QCFeedbackWithChecklist:
    """Simple decoder for test responses."""
    data = json.loads(text)
    checklist = [
        QCChecklistItem(
            id=item["id"],
            description=item.get("description", ""),
            done_when=item.get("done_when", ""),
            priority=item.get("priority", "P1"),
            completed=item.get("completed", False),
        )
        for item in data.get("checklist", [])
    ]
    return QCFeedbackWithChecklist(
        feedback=QCFeedback(
            action=data["action"],
            feedback=data["feedback"],
            model=model,
        ),
        checklist=checklist,
    )


class TestBatchedVisionSolvePlaybook(unittest.TestCase):
    """Tests for BatchedVisionSolvePlaybook."""

    def setUp(self):
        """Set up test fixtures."""
        self.system_prompt = StructuredPrompt(
            base_instruction="You are a vision quality reviewer.",
            sections={"Criteria": "Check image quality against reference."},
            requirements=["Be thorough"],
        )
        # Create test images
        self.test_image1 = Image.new("RGB", (100, 100), color="red")
        self.test_image2 = Image.new("RGB", (100, 100), color="blue")
        self.ref_image1 = Image.new("RGB", (100, 100), color="green")
        self.ref_image2 = Image.new("RGB", (100, 100), color="yellow")
        DummyInterface.reset()

    def tearDown(self):
        """Clean up after tests."""
        DummyInterface.reset()

    def _make_response(
        self,
        action: str = "proceed",
        feedback: str = "Looks good",
        checklist: list | None = None,
    ) -> Dict[str, Any]:
        """Helper to create a response dict."""
        return {
            "text": json.dumps(
                {
                    "action": action,
                    "feedback": feedback,
                    "checklist": checklist or [],
                }
            )
        }

    def test_single_image_proceed(self):
        """Single image evaluation passes."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Image looks good"),
        ])

        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
            max_workers=1,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="Evaluate image quality",
            control_guide=self.system_prompt,
            images=[VisionEvaluationItem(image=self.test_image1, id="img1")],
            reference_images={"ref1": self.ref_image1},
        )

        result = playbook.run(request, revise_fn=Mock())

        self.assertTrue(result.approved)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(len(result.feedback_history), 1)

    def test_multiple_images_all_pass(self):
        """Multiple images all pass evaluation."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Image 1 good"),
            self._make_response(action="proceed", feedback="Image 2 good"),
        ])

        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
            max_workers=2,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="Evaluate image quality",
            control_guide=self.system_prompt,
            images=[
                VisionEvaluationItem(image=self.test_image1, id="img1"),
                VisionEvaluationItem(image=self.test_image2, id="img2"),
            ],
            reference_images={"ref1": self.ref_image1},
        )

        result = playbook.run(request, revise_fn=Mock())

        self.assertTrue(result.approved)
        self.assertEqual(result.iterations, 1)

    def test_multiple_images_some_fail(self):
        """Some images fail evaluation."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Image 1 good"),
            self._make_response(
                action="revise",
                feedback="Image 2 has issues",
                checklist=[{
                    "id": "issue1",
                    "description": "Quality problem",
                    "done_when": "Fixed",
                    "priority": "P1",
                }],
            ),
        ])

        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
            max_workers=2,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="Evaluate image quality",
            control_guide=self.system_prompt,
            images=[
                VisionEvaluationItem(image=self.test_image1, id="img1"),
                VisionEvaluationItem(image=self.test_image2, id="img2"),
            ],
            reference_images={"ref1": self.ref_image1},
        )

        result = playbook.run(request, revise_fn=Mock())

        self.assertFalse(result.approved)
        # Content should contain per-image results
        content_data = json.loads(result.content)
        self.assertEqual(len(content_data), 2)

    def test_reference_key_mapping(self):
        """Reference keys correctly select subset of references."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Image with ref1 only"),
            self._make_response(action="proceed", feedback="Image with both refs"),
        ])

        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
            max_workers=2,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="Evaluate image quality",
            control_guide=self.system_prompt,
            images=[
                VisionEvaluationItem(
                    image=self.test_image1,
                    id="img1",
                    reference_keys=["ref1"],  # Only ref1
                ),
                VisionEvaluationItem(
                    image=self.test_image2,
                    id="img2",
                    reference_keys=["ref1", "ref2"],  # Both refs
                ),
            ],
            reference_images={
                "ref1": self.ref_image1,
                "ref2": self.ref_image2,
            },
        )

        result = playbook.run(request, revise_fn=Mock())

        self.assertTrue(result.approved)

        # Verify the calls included the right images
        calls = DummyInterface.get_calls()
        self.assertEqual(len(calls), 2)

    def test_no_reference_keys_uses_all_references(self):
        """When reference_keys is None, all references are included."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Good with all refs"),
        ])

        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
            max_workers=1,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="Evaluate image quality",
            control_guide=self.system_prompt,
            images=[
                VisionEvaluationItem(
                    image=self.test_image1,
                    id="img1",
                    reference_keys=None,  # Use all
                ),
            ],
            reference_images={
                "ref1": self.ref_image1,
                "ref2": self.ref_image2,
            },
        )

        result = playbook.run(request, revise_fn=Mock())

        self.assertTrue(result.approved)

        # Verify the query included all reference images
        calls = DummyInterface.get_calls()
        self.assertEqual(len(calls), 1)
        query = calls[0]["query"]
        # Should have target + 2 references = 3 images
        self.assertEqual(len(query.images), 3)

    def test_invalid_request_type_raises_error(self):
        """TypeError raised for non-VisionCritiqueRequest."""
        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
        )
        playbook = BatchedVisionSolvePlaybook(config)

        # Use regular CritiqueRequest instead of VisionCritiqueRequest
        request = CritiqueRequest(
            content="test",
            context="test",
            control_guide=self.system_prompt,
        )

        with self.assertRaises(TypeError) as ctx:
            playbook.run(request, revise_fn=Mock())

        self.assertIn("VisionCritiqueRequest", str(ctx.exception))

    def test_empty_images_raises_error(self):
        """ValueError raised when no images provided."""
        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="test",
            control_guide=self.system_prompt,
            images=[],  # Empty
            reference_images={},
        )

        with self.assertRaises(ValueError) as ctx:
            playbook.run(request, revise_fn=Mock())

        self.assertIn("at least one image", str(ctx.exception))

    def test_checklist_items_prefixed_with_image_id(self):
        """Checklist items are prefixed with image ID in aggregated result."""
        DummyInterface.set_responses([
            self._make_response(
                action="revise",
                feedback="Issues found",
                checklist=[{
                    "id": "quality_001",
                    "description": "Low resolution",
                    "done_when": "Resolution increased",
                    "priority": "P1",
                }],
            ),
        ])

        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
            max_workers=1,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="Evaluate image quality",
            control_guide=self.system_prompt,
            images=[VisionEvaluationItem(image=self.test_image1, id="test_img")],
            reference_images={},
        )

        result = playbook.run(request, revise_fn=Mock())

        # Check aggregated feedback has prefixed checklist
        aggregated = result.feedback_history[0]
        self.assertEqual(len(aggregated.checklist), 1)
        self.assertTrue(aggregated.checklist[0].id.startswith("test_img_"))
        self.assertIn("[test_img]", aggregated.checklist[0].description)

    def test_on_feedback_callback_called(self):
        """on_feedback callback is called for each image."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Good 1"),
            self._make_response(action="proceed", feedback="Good 2"),
        ])

        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
            max_workers=2,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="Evaluate",
            control_guide=self.system_prompt,
            images=[
                VisionEvaluationItem(image=self.test_image1, id="img1"),
                VisionEvaluationItem(image=self.test_image2, id="img2"),
            ],
            reference_images={},
        )

        callback_results = []

        def on_feedback(feedback, idx):
            callback_results.append((feedback.action, idx))

        result = playbook.run(request, revise_fn=Mock(), on_feedback=on_feedback)

        self.assertTrue(result.approved)
        self.assertEqual(len(callback_results), 2)

    def test_token_tracking_accumulated(self):
        """Token tracking is accumulated across all images."""
        DummyInterface.set_responses([
            {
                "text": json.dumps({
                    "action": "proceed",
                    "feedback": "Good",
                    "checklist": [],
                }),
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            {
                "text": json.dumps({
                    "action": "proceed",
                    "feedback": "Good",
                    "checklist": [],
                }),
                "usage": {"input_tokens": 150, "output_tokens": 75},
            },
        ])

        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
            max_workers=2,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="Evaluate",
            control_guide=self.system_prompt,
            images=[
                VisionEvaluationItem(image=self.test_image1, id="img1"),
                VisionEvaluationItem(image=self.test_image2, id="img2"),
            ],
            reference_images={},
        )

        result = playbook.run(request, revise_fn=Mock())

        # Check aggregated tokens (cost is 0 for dummy interface)
        aggregated = result.feedback_history[0]
        self.assertEqual(aggregated.critique_input_tokens, 250)
        self.assertEqual(aggregated.critique_output_tokens, 125)

    def test_auto_generated_image_ids(self):
        """Image IDs are auto-generated when not provided."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Good 1"),
            self._make_response(action="proceed", feedback="Good 2"),
        ])

        config = BatchedVisionSolveConfig(
            feedback_decoder=decode_feedback,
            model_interface="dummy",
            max_workers=2,
        )
        playbook = BatchedVisionSolvePlaybook(config)

        request = VisionCritiqueRequest(
            content="",
            context="Evaluate",
            control_guide=self.system_prompt,
            images=[
                VisionEvaluationItem(image=self.test_image1),  # No id
                VisionEvaluationItem(image=self.test_image2),  # No id
            ],
            reference_images={},
        )

        result = playbook.run(request, revise_fn=Mock())

        # Check content has auto-generated IDs
        content_data = json.loads(result.content)
        image_ids = [item["image_id"] for item in content_data]
        self.assertIn("image_0", image_ids)
        self.assertIn("image_1", image_ids)


if __name__ == "__main__":
    unittest.main()
