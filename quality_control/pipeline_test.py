"""Tests for quality control pipeline."""

import json
import unittest
from typing import Any, Dict
from unittest.mock import Mock

from model_router.interfaces.dummy_interface import DummyInterface
from model_router.query import StructuredPrompt
from quality_control.pipeline import QualityControlPipeline
from quality_control.control_playbook.global_solve import GlobalSolveConfig
from quality_control.types import (
    CritiqueRequest,
    QCChecklistItem,
    QCFeedback,
    QCFeedbackWithChecklist,
    QCState,
    QualityControlConfig,
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


class TestQualityControlPipeline(unittest.TestCase):
    """Tests for QualityControlPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.system_prompt = StructuredPrompt(
            base_instruction="You are a reviewer.",
            sections={"Criteria": "Check quality."},
            requirements=["Be thorough"],
        )
        # Reset dummy interface between tests
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

    def _make_checklist_item_dict(self, id: str) -> dict:
        """Helper to create checklist item dict for responses."""
        return {
            "id": id,
            "description": f"Fix {id}",
            "done_when": f"{id} is fixed",
            "priority": "P1",
        }

    def test_immediate_proceed(self):
        """Loop exits on first 'proceed' response."""
        DummyInterface.set_responses(
            [self._make_response(action="proceed", feedback="Looks good!")]
        )
        config = QualityControlConfig(
            playbook_config=GlobalSolveConfig(
                max_iterations=5,
                model_interfaces=["dummy"],
                feedback_decoder=decode_feedback,
            )
        )
        qc = QualityControlPipeline(config)

        result = qc.run(
            request=CritiqueRequest(
                content="test content",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
        )

        self.assertTrue(result.approved)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(len(result.feedback_history), 1)
        # Content should be just the content, not context
        self.assertEqual(result.content, "test content")

    def test_revise_then_proceed(self):
        """Loop revises once, then proceeds."""
        DummyInterface.set_responses(
            [
                self._make_response(
                    action="revise",
                    feedback="Fix X",
                    checklist=[self._make_checklist_item_dict("x")],
                ),
                self._make_response(action="proceed", feedback="Fixed!"),
            ]
        )
        config = QualityControlConfig(
            playbook_config=GlobalSolveConfig(
                max_iterations=5,
                model_interfaces=["dummy"],
                feedback_decoder=decode_feedback,
            )
        )
        qc = QualityControlPipeline(config)

        revise_fn = Mock(return_value="revised content")

        result = qc.run(
            request=CritiqueRequest(
                content="original content",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=revise_fn,
        )

        self.assertTrue(result.approved)
        self.assertEqual(result.iterations, 2)
        revise_fn.assert_called_once()
        # Check revise_fn was called with original content (not context) and feedback
        call_args = revise_fn.call_args
        self.assertEqual(call_args[0][0], "original content")
        self.assertEqual(call_args[0][1].action, "revise")
        # Result content should be the revised content
        self.assertEqual(result.content, "revised content")

    def test_max_iterations_reached(self):
        """Loop stops at max_iterations even without 'proceed'."""
        blocking_item = {"id": "blocking", "description": "Issue", "done_when": "Fixed", "priority": "P0"}
        DummyInterface.set_responses(
            [
                self._make_response(action="revise", feedback="Still wrong 1", checklist=[blocking_item]),
                self._make_response(action="revise", feedback="Still wrong 2", checklist=[blocking_item]),
                self._make_response(action="revise", feedback="Still wrong 3", checklist=[blocking_item]),
            ]
        )
        config = QualityControlConfig(
            playbook_config=GlobalSolveConfig(
                max_iterations=3,
                model_interfaces=["dummy"],
                feedback_decoder=decode_feedback,
            )
        )
        qc = QualityControlPipeline(config)

        revise_fn = Mock(return_value="revised")

        result = qc.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=revise_fn,
        )

        self.assertFalse(result.approved)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(revise_fn.call_count, 3)

    def test_content_updated_each_iteration(self):
        """Content is updated after each revision."""
        DummyInterface.set_responses(
            [
                self._make_response(
                    action="revise",
                    feedback="Fix 1",
                    checklist=[self._make_checklist_item_dict("issue1")],
                ),
                self._make_response(
                    action="revise",
                    feedback="Fix 2",
                    checklist=[self._make_checklist_item_dict("issue2")],
                ),
                self._make_response(action="proceed", feedback="Done"),
            ]
        )
        config = QualityControlConfig(
            playbook_config=GlobalSolveConfig(
                max_iterations=5,
                model_interfaces=["dummy"],
                feedback_decoder=decode_feedback,
            )
        )
        qc = QualityControlPipeline(config)

        call_count = [0]

        def revise_fn(content: str, feedback: QCFeedbackWithChecklist) -> str:
            call_count[0] += 1
            return f"revision_{call_count[0]}"

        result = qc.run(
            request=CritiqueRequest(
                content="original",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=revise_fn,
        )

        self.assertEqual(result.content, "revision_2")
        self.assertTrue(result.approved)

    def test_restart_recovery_with_pending_revision(self):
        """Loop resumes and applies pending revision."""
        # State: one "revise" feedback recorded, but no revision applied yet
        state = QCState(
            feedback_history=[
                QCFeedbackWithChecklist(
                    feedback=QCFeedback(action="revise", feedback="Fix something"),
                    checklist=[],
                ),
            ],
            content_history=[],  # No content history = revision not applied
            iteration=0,
        )

        DummyInterface.set_responses(
            [
                self._make_response(action="proceed", feedback="Now it's good"),
            ]
        )
        config = QualityControlConfig(
            playbook_config=GlobalSolveConfig(
                max_iterations=5,
                model_interfaces=["dummy"],
                feedback_decoder=decode_feedback,
            )
        )
        qc = QualityControlPipeline(config)

        revise_fn = Mock(return_value="revised from recovery")

        result = qc.run(
            request=CritiqueRequest(
                content="original",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=revise_fn,
            state=state,
        )

        self.assertTrue(result.approved)
        revise_fn.assert_called_once()
        self.assertEqual(result.content, "revised from recovery")

    def test_restart_recovery_already_approved(self):
        """Loop exits immediately if state shows already approved."""
        state = QCState(
            feedback_history=[
                QCFeedbackWithChecklist(
                    feedback=QCFeedback(action="proceed", feedback="Already good"),
                    checklist=[],
                ),
            ],
            content_history=[],
            iteration=1,
        )

        DummyInterface.set_responses([])  # Should not be called
        config = QualityControlConfig(
            playbook_config=GlobalSolveConfig(
                max_iterations=5,
                model_interfaces=["dummy"],
                feedback_decoder=decode_feedback,
            )
        )
        qc = QualityControlPipeline(config)

        result = qc.run(
            request=CritiqueRequest(
                content="original",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(),
            state=state,
        )

        self.assertTrue(result.approved)
        # Verify no calls were made to DummyInterface
        self.assertEqual(len(DummyInterface.get_calls()), 0)

    def test_feedback_history_accumulated(self):
        """Feedback history contains all feedback from the loop."""
        DummyInterface.set_responses(
            [
                self._make_response(
                    action="revise",
                    feedback="Issue 1",
                    checklist=[self._make_checklist_item_dict("issue1")],
                ),
                self._make_response(
                    action="revise",
                    feedback="Issue 2",
                    checklist=[self._make_checklist_item_dict("issue2")],
                ),
                self._make_response(action="proceed", feedback="All good"),
            ]
        )
        config = QualityControlConfig(
            playbook_config=GlobalSolveConfig(
                max_iterations=5,
                model_interfaces=["dummy"],
                feedback_decoder=decode_feedback,
            )
        )
        qc = QualityControlPipeline(config)

        result = qc.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
        )

        self.assertEqual(len(result.feedback_history), 3)
        self.assertEqual(result.feedback_history[0].feedback.feedback, "Issue 1")
        self.assertEqual(result.feedback_history[1].feedback.feedback, "Issue 2")
        self.assertEqual(result.feedback_history[2].feedback.feedback, "All good")

    def test_model_interface_recorded(self):
        """Model interface is recorded in feedback."""
        DummyInterface.set_responses(
            [
                self._make_response(action="proceed", feedback="Good"),
            ]
        )
        config = QualityControlConfig(
            playbook_config=GlobalSolveConfig(
                max_iterations=5,
                model_interfaces=["dummy"],
                feedback_decoder=decode_feedback,
            )
        )
        qc = QualityControlPipeline(config)

        result = qc.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(),
        )

        self.assertEqual(result.feedback_history[0].model, "dummy")

    def test_checklist_merged_across_iterations(self):
        """Checklist items are merged across iterations."""
        DummyInterface.set_responses(
            [
                self._make_response(
                    action="revise",
                    feedback="Fix A",
                    checklist=[self._make_checklist_item_dict("a")],
                ),
                self._make_response(
                    action="proceed",
                    feedback="Done",
                    checklist=[],  # Item "a" not mentioned = completed
                ),
            ]
        )
        config = QualityControlConfig(
            playbook_config=GlobalSolveConfig(
                max_iterations=5,
                model_interfaces=["dummy"],
                feedback_decoder=decode_feedback,
            )
        )
        qc = QualityControlPipeline(config)

        result = qc.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
        )

        # Second feedback should have merged checklist with item "a" completed
        final_checklist = result.feedback_history[-1].checklist
        self.assertEqual(len(final_checklist), 1)
        self.assertTrue(final_checklist[0].completed)


if __name__ == "__main__":
    unittest.main()
