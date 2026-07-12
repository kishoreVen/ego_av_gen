"""Tests for checklist merging logic."""

import unittest

from quality_control.checklist import merge_checklists
from quality_control.types import QCChecklistItem


class TestMergeChecklists(unittest.TestCase):
    """Tests for merge_checklists function."""

    def _make_item(
        self,
        id: str,
        completed: bool = False,
        completed_at_iteration: int | None = None,
    ) -> QCChecklistItem:
        """Helper to create a checklist item."""
        return QCChecklistItem(
            id=id,
            description=f"Fix {id}",
            done_when=f"{id} is fixed",
            priority="P1",
            completed=completed,
            completed_at_iteration=completed_at_iteration,
        )

    def test_empty_lists(self):
        """Merging empty lists returns empty list."""
        result = merge_checklists([], [], current_iteration=1)
        self.assertEqual(result, [])

    def test_new_items_added(self):
        """New items not in previous are added."""
        previous = []
        new_items = [self._make_item("a"), self._make_item("b")]

        result = merge_checklists(previous, new_items, current_iteration=1)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "a")
        self.assertEqual(result[1].id, "b")
        self.assertFalse(result[0].completed)
        self.assertFalse(result[1].completed)

    def test_missing_items_marked_complete(self):
        """Items from previous not in new are marked as completed."""
        previous = [self._make_item("a")]
        new_items = []  # Item "a" not mentioned

        result = merge_checklists(previous, new_items, current_iteration=2)

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].completed)
        self.assertEqual(result[0].completed_at_iteration, 2)

    def test_already_completed_items_preserved(self):
        """Already completed items are kept as-is."""
        previous = [self._make_item("a", completed=True, completed_at_iteration=1)]
        new_items = []

        result = merge_checklists(previous, new_items, current_iteration=3)

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].completed)
        self.assertEqual(result[0].completed_at_iteration, 1)  # Original iteration

    def test_items_in_both_lists_kept(self):
        """Items in both lists are kept (still incomplete)."""
        previous = [self._make_item("a")]
        new_items = [self._make_item("a")]

        result = merge_checklists(previous, new_items, current_iteration=2)

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].completed)

    def test_item_marked_complete_in_new(self):
        """If critic marks item complete in new, it's completed."""
        previous = [self._make_item("a")]
        new_items = [self._make_item("a", completed=True)]

        result = merge_checklists(previous, new_items, current_iteration=2)

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].completed)
        self.assertEqual(result[0].completed_at_iteration, 2)

    def test_mixed_scenario(self):
        """Test realistic scenario with multiple items."""
        previous = [
            self._make_item("a"),  # Will be completed (not in new)
            self._make_item("b"),  # Will stay (still in new)
            self._make_item("c", completed=True, completed_at_iteration=1),  # Already done
        ]
        new_items = [
            self._make_item("b"),  # Still needs work
            self._make_item("d"),  # New issue
        ]

        result = merge_checklists(previous, new_items, current_iteration=3)

        self.assertEqual(len(result), 4)

        # Check each item
        by_id = {item.id: item for item in result}

        self.assertTrue(by_id["a"].completed)
        self.assertEqual(by_id["a"].completed_at_iteration, 3)

        self.assertFalse(by_id["b"].completed)

        self.assertTrue(by_id["c"].completed)
        self.assertEqual(by_id["c"].completed_at_iteration, 1)

        self.assertFalse(by_id["d"].completed)

    def test_order_preserved(self):
        """Previous items come first, then new items."""
        previous = [self._make_item("a"), self._make_item("b")]
        new_items = [self._make_item("c"), self._make_item("d")]

        result = merge_checklists(previous, new_items, current_iteration=1)

        ids = [item.id for item in result]
        self.assertEqual(ids, ["a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
