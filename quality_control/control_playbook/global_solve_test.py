"""Tests for GlobalSolve playbook."""

import json
import unittest
from typing import Any, Dict
from unittest.mock import Mock

from model_router.interfaces.dummy_interface import DummyInterface
from model_router.query import StructuredPrompt
from quality_control.control_playbook.global_solve import (
    GlobalSolveConfig,
    GlobalSolvePlaybook,
)
from quality_control.types import (
    CritiqueRequest,
    QCChecklistItem,
    QCFeedback,
    QCFeedbackWithChecklist,
    QCState,
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


class TestGlobalSolvePlaybook(unittest.TestCase):
    """Tests for GlobalSolvePlaybook."""

    def setUp(self):
        """Set up test fixtures."""
        self.system_prompt = StructuredPrompt(
            base_instruction="You are a reviewer.",
            sections={"Criteria": "Check quality."},
            requirements=["Be thorough"],
        )
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

    def test_single_iteration_proceed(self):
        """Approves on first iteration."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Content is great"),
        ])

        config = GlobalSolveConfig(
            max_iterations=5,
            model_interfaces=["dummy"],
            feedback_decoder=decode_feedback,
        )
        playbook = GlobalSolvePlaybook(config)

        result = playbook.run(
            request=CritiqueRequest(
                content="test content",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
        )

        self.assertTrue(result.approved)
        self.assertEqual(result.iterations, 1)

    def test_revise_then_proceed(self):
        """Requests revision then approves."""
        DummyInterface.set_responses([
            self._make_response(
                action="revise",
                feedback="Fix this",
                checklist=[{"id": "issue_001", "description": "Fix it", "done_when": "Fixed", "priority": "P1"}],
            ),
            self._make_response(action="proceed", feedback="Now it's good"),
        ])

        config = GlobalSolveConfig(
            max_iterations=5,
            model_interfaces=["dummy"],
            feedback_decoder=decode_feedback,
        )
        playbook = GlobalSolvePlaybook(config)

        revise_fn = Mock(return_value="revised content")

        result = playbook.run(
            request=CritiqueRequest(
                content="original",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=revise_fn,
        )

        self.assertTrue(result.approved)
        self.assertEqual(result.iterations, 2)
        revise_fn.assert_called_once()

    def test_max_iterations_reached(self):
        """Stops at max iterations."""
        blocking_item = {"id": "blocking_001", "description": "Issue", "done_when": "Fixed", "priority": "P0"}
        DummyInterface.set_responses([
            self._make_response(action="revise", feedback="Still wrong 1", checklist=[blocking_item]),
            self._make_response(action="revise", feedback="Still wrong 2", checklist=[blocking_item]),
            self._make_response(action="revise", feedback="Still wrong 3", checklist=[blocking_item]),
        ])

        config = GlobalSolveConfig(
            max_iterations=3,
            model_interfaces=["dummy"],
            feedback_decoder=decode_feedback,
        )
        playbook = GlobalSolvePlaybook(config)

        result = playbook.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
        )

        self.assertFalse(result.approved)
        self.assertEqual(result.iterations, 3)

    def test_model_locking_preserves_model_across_iterations(self):
        """Model selected on first iteration is preserved for all subsequent iterations."""
        DummyInterface.set_responses([
            self._make_response(action="revise", feedback="Fix 1"),
            self._make_response(action="revise", feedback="Fix 2"),
            self._make_response(action="proceed", feedback="Good"),
        ])

        config = GlobalSolveConfig(
            max_iterations=5,
            model_interfaces=["dummy"],
            feedback_decoder=decode_feedback,
        )
        playbook = GlobalSolvePlaybook(config)
        state = QCState()

        result = playbook.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
            state=state,
        )

        # Model should be locked in state
        self.assertIsNotNone(state.locked_model)
        self.assertEqual(state.locked_model, "dummy")

        # All feedback should use the same model
        for feedback in result.feedback_history:
            self.assertEqual(feedback.model, "dummy")

    def test_model_locking_uses_locked_model_on_resume(self):
        """When resuming with a locked model, that model is used."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Good after resume"),
        ])

        config = GlobalSolveConfig(
            max_iterations=5,
            model_interfaces=["dummy", "other_dummy"],
            feedback_decoder=decode_feedback,
        )
        playbook = GlobalSolvePlaybook(config)

        # Pre-set locked_model to simulate resume
        state = QCState(locked_model="dummy")

        result = playbook.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
            state=state,
        )

        # Should still be using the locked model
        self.assertEqual(state.locked_model, "dummy")
        self.assertEqual(result.feedback_history[0].model, "dummy")

    def test_p0_items_block_proceed(self):
        """P0 items force revise even if model says proceed."""
        DummyInterface.set_responses([
            self._make_response(
                action="proceed",
                feedback="Looks good",
                checklist=[{
                    "id": "blocking_001",
                    "description": "Critical issue",
                    "done_when": "Fixed",
                    "priority": "P0",
                    "completed": False,
                }],
            ),
            self._make_response(action="proceed", feedback="Now good"),
        ])

        config = GlobalSolveConfig(
            max_iterations=5,
            model_interfaces=["dummy"],
            feedback_decoder=decode_feedback,
        )
        playbook = GlobalSolvePlaybook(config)

        result = playbook.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
        )

        # First feedback should be overridden to revise
        self.assertEqual(result.feedback_history[0].action, "revise")
        self.assertEqual(result.iterations, 2)

    def test_p2_items_dont_block_proceed(self):
        """P2 items don't block proceeding."""
        DummyInterface.set_responses([
            self._make_response(
                action="proceed",
                feedback="Looks good",
                checklist=[{
                    "id": "polish_001",
                    "description": "Minor polish",
                    "done_when": "Polished",
                    "priority": "P2",
                    "completed": False,
                }],
            ),
        ])

        config = GlobalSolveConfig(
            max_iterations=5,
            model_interfaces=["dummy"],
            feedback_decoder=decode_feedback,
        )
        playbook = GlobalSolvePlaybook(config)

        result = playbook.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
        )

        # Should proceed despite incomplete P2 item
        self.assertTrue(result.approved)
        self.assertEqual(result.iterations, 1)


if __name__ == "__main__":
    unittest.main()
