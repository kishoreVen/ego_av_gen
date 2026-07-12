"""Checklist merging logic for quality control pipeline."""

from __future__ import annotations

import json
import logging
from typing import List, TypeVar

from model_router.query import Query
from model_router.router import Capability, ModelRouter
from quality_control.types import QCChecklistItem
from model_router.lib.output_formatting import safe_json_decode

T = TypeVar("T", bound=QCChecklistItem)

logger = logging.getLogger(__name__)

# LLM interface for semantic deduplication (fast, cheap model)
_DEDUP_MODEL_INTERFACE = "anthropic_haiku45"


def _dedupe_checklist_llm(
    items: List[T],
    router: ModelRouter,
) -> List[T]:
    """Use LLM to semantically deduplicate a checklist.

    - Removes incomplete items that semantically match completed items
    - Merges semantically similar incomplete items into a single item

    Args:
        items: Checklist items (mix of completed and incomplete).
        router: ModelRouter for LLM calls.

    Returns:
        Deduped checklist.
    """
    if not items:
        return items

    # Serialize items to JSON
    items_json = json.dumps([item.to_dict() for item in items], indent=2)

    prompt = f"""You are deduplicating a quality control checklist.

INPUT CHECKLIST:
{items_json}

TASK:
1. If an incomplete item (completed=false) semantically matches a completed item (completed=true), remove the incomplete item.
2. If multiple incomplete items are semantically similar, merge them into a single item.

Examples of duplicates:
- "Key props omit the socks" and "Add socks as a key prop" = SAME issue
- "Beat 2 uses 'notices'" and "Replace 'notices' with observable verb in Beat 2" = SAME issue

OUTPUT FORMAT:
Return a JSON array of checklist items:
[
    {QCChecklistItem.schema()}
]

Return ONLY the JSON array."""

    try:
        response = router.get_response(
            query=Query(query_text=prompt),
            capability=Capability.TEXT,
            interface_type=_DEDUP_MODEL_INTERFACE,
        )

        text = response["text"].strip()
        parsed = safe_json_decode(text)
        result = [QCChecklistItem.from_dict(item) for item in parsed]

        removed = len(items) - len(result)
        if removed > 0:
            logger.info(f"LLM deduped checklist: removed/merged {removed} items")

        return result

    except Exception as e:
        logger.warning(f"Semantic dedup LLM call failed, skipping dedup: {e}")
        return items


def merge_checklists(
    previous: List[T],
    new_items: List[T],
    current_iteration: int,
    router: ModelRouter | None = None,
) -> List[T]:
    """
    Merge previous checklist items with new critic output.

    - Items from previous that are not in new = marked as completed
    - Items from previous that are in new = kept with their status
    - New items not in previous = added as incomplete (unless semantic duplicate)

    Semantic deduplication (when router provided) uses an LLM to detect when
    critics re-raise completed issues under new IDs, preventing iteration loops.

    Args:
        previous: Checklist items from previous iterations.
        new_items: New checklist items from the critic.
        current_iteration: Current iteration number.
        router: Optional ModelRouter for LLM-based semantic deduplication.

    Returns:
        Merged checklist with all items (completed and incomplete).
    """
    result = []
    new_ids = {item.id for item in new_items}
    prev_ids = {item.id for item in previous}

    # Process previous items
    for prev_item in previous:
        if prev_item.completed:
            # Already completed - keep as is
            result.append(prev_item)
        elif prev_item.id not in new_ids:
            # Not in new feedback - mark as completed
            prev_item.completed = True
            prev_item.completed_at_iteration = current_iteration
            result.append(prev_item)
        else:
            # Still in new feedback - find the updated version
            new_item = next(i for i in new_items if i.id == prev_item.id)
            if new_item.completed:
                # Critic marked it complete
                prev_item.completed = True
                prev_item.completed_at_iteration = current_iteration
            result.append(prev_item)

    # Add new items that weren't in previous
    for new_item in new_items:
        if new_item.id not in prev_ids:
            result.append(new_item)

    # Use LLM to dedupe the merged list if router provided
    if router:
        result = _dedupe_checklist_llm(result, router)

    return result
