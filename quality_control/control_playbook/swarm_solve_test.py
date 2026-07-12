"""Tests for SwarmSolve playbook."""

import json
import unittest
from typing import Any, Dict
from unittest.mock import Mock

from model_router.interfaces.dummy_interface import DummyInterface
from model_router.query import StructuredPrompt
from quality_control.control_playbook.swarm_solve import (
    SwarmChecklistItem,
    SwarmSolveConfig,
    SwarmSolvePlaybook,
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


class TestSwarmSolvePlaybook(unittest.TestCase):
    """Tests for SwarmSolvePlaybook."""

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

    def test_empty_focused_models_raises_error(self):
        """Raises error when no focused models configured."""
        config = SwarmSolveConfig(
            max_iterations=5,
            focused_models=[],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)

        with self.assertRaises(ValueError) as ctx:
            playbook.run(
                request=CritiqueRequest(
                    content="test content",
                    context="test context",
                    control_guide=self.system_prompt,
                ),
                revise_fn=Mock(return_value="revised"),
            )

        self.assertIn("focused model", str(ctx.exception))

    def test_single_focused_model_proceed(self):
        """Single focused model approves immediately."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Dialogue is great"),
        ])

        config = SwarmSolveConfig(
            max_iterations=5,
            focused_models=[{"interface": "dummy", "focus_area": "dialogue"}],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)

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
        self.assertIn("[dialogue]", result.feedback_history[0].feedback.feedback)

    def test_single_focused_model_revise_then_proceed(self):
        """Single focused model requests revision then approves."""
        DummyInterface.set_responses([
            self._make_response(
                action="revise",
                feedback="Fix dialogue",
                checklist=[{"id": "issue_001", "description": "Fix it", "done_when": "Fixed", "priority": "P1"}],
            ),
            self._make_response(action="proceed", feedback="Dialogue fixed"),
        ])

        config = SwarmSolveConfig(
            max_iterations=5,
            focused_models=[{"interface": "dummy", "focus_area": "dialogue"}],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)

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

    def test_multiple_focused_models_all_proceed(self):
        """Multiple focused models all approve."""
        # Each model will get called once
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Dialogue good"),
            self._make_response(action="proceed", feedback="Pacing good"),
        ])

        config = SwarmSolveConfig(
            max_iterations=5,
            focused_models=[
                {"interface": "dummy", "focus_area": "dialogue"},
                {"interface": "dummy", "focus_area": "pacing"},
            ],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)

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
        # Both feedback texts should be in aggregated feedback
        feedback_text = result.feedback_history[0].feedback.feedback
        self.assertTrue(
            "[dialogue]" in feedback_text or "[pacing]" in feedback_text
        )

    def test_multiple_focused_models_one_revises(self):
        """If any focused model says revise, action is revise."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Dialogue good"),
            self._make_response(
                action="revise",
                feedback="Pacing needs work",
                checklist=[{"id": "pacing_001", "description": "Fix pacing", "done_when": "Fixed", "priority": "P1"}],
            ),
            # Second iteration - both proceed
            self._make_response(action="proceed", feedback="Dialogue still good"),
            self._make_response(action="proceed", feedback="Pacing fixed"),
        ])

        config = SwarmSolveConfig(
            max_iterations=5,
            focused_models=[
                {"interface": "dummy", "focus_area": "dialogue"},
                {"interface": "dummy", "focus_area": "pacing"},
            ],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)

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
        # First iteration should have action="revise"
        self.assertEqual(result.feedback_history[0].action, "revise")

    def test_checklist_items_get_focus_area(self):
        """Checklist items are tagged with focus area."""
        DummyInterface.set_responses([
            self._make_response(
                action="proceed",
                feedback="Good",
                checklist=[{"id": "item1", "description": "Fix X", "done_when": "X fixed", "priority": "P1"}],
            ),
        ])

        config = SwarmSolveConfig(
            max_iterations=5,
            focused_models=[{"interface": "dummy", "focus_area": "dialogue"}],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)

        result = playbook.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
        )

        # Checklist item should have focus_area set
        checklist = result.feedback_history[0].checklist
        self.assertEqual(len(checklist), 1)
        self.assertIsInstance(checklist[0], SwarmChecklistItem)
        swarm_item = checklist[0]
        assert isinstance(swarm_item, SwarmChecklistItem)
        self.assertEqual(swarm_item.focus_area, "dialogue")

    def test_max_iterations_reached(self):
        """Swarm stops at max iterations."""
        blocking_item = {"id": "blocking_001", "description": "Issue", "done_when": "Fixed", "priority": "P0"}
        DummyInterface.set_responses([
            self._make_response(action="revise", feedback="Still wrong 1", checklist=[blocking_item]),
            self._make_response(action="revise", feedback="Still wrong 2", checklist=[blocking_item]),
            self._make_response(action="revise", feedback="Still wrong 3", checklist=[blocking_item]),
        ])

        config = SwarmSolveConfig(
            max_iterations=3,
            focused_models=[{"interface": "dummy", "focus_area": "dialogue"}],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)

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

    def test_aggregated_feedback_combines_models(self):
        """Aggregated feedback shows which models were used."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Good dialogue"),
            self._make_response(action="proceed", feedback="Good pacing"),
        ])

        config = SwarmSolveConfig(
            max_iterations=5,
            focused_models=[
                {"interface": "dummy", "focus_area": "dialogue"},
                {"interface": "dummy", "focus_area": "pacing"},
            ],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)

        result = playbook.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
        )

        # Model field should contain both interface names
        self.assertIn("dummy", result.feedback_history[0].model)

    def test_model_locking_stores_focused_models(self):
        """Model locking stores focused models on first iteration."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Good"),
            self._make_response(action="proceed", feedback="Good"),
        ])

        config = SwarmSolveConfig(
            max_iterations=5,
            focused_models=[
                {"interface": "dummy", "focus_area": "dialogue"},
                {"interface": "dummy", "focus_area": "pacing"},
            ],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)
        state = QCState()

        playbook.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
            state=state,
        )

        # locked_model should contain all focus_area:interface pairs
        self.assertIsNotNone(state.locked_model)
        self.assertIn("dialogue:dummy", state.locked_model)
        self.assertIn("pacing:dummy", state.locked_model)

    def test_model_locking_preserved_on_resume(self):
        """When resuming with locked_model set, it's preserved."""
        DummyInterface.set_responses([
            self._make_response(action="proceed", feedback="Good"),
        ])

        config = SwarmSolveConfig(
            max_iterations=5,
            focused_models=[{"interface": "dummy", "focus_area": "dialogue"}],
            feedback_decoder=decode_feedback,
        )
        playbook = SwarmSolvePlaybook(config)

        # Pre-set locked_model to simulate resume
        state = QCState(locked_model="dialogue:other_model")

        playbook.run(
            request=CritiqueRequest(
                content="test",
                context="test context",
                control_guide=self.system_prompt,
            ),
            revise_fn=Mock(return_value="revised"),
            state=state,
        )

        # Should preserve the original locked_model
        self.assertEqual(state.locked_model, "dialogue:other_model")


if __name__ == "__main__":
    unittest.main()
