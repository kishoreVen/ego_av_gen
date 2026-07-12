"""SAM 3.1 local inference interface."""
from __future__ import annotations

from typing import Any, Dict, List

from model_router.model_interface import Capability, Query
from model_router.scaffold.local_interface import LocalModelInterface, extract_images

# SAM3 expects short object labels for text-prompted grounding,
# not long VLM-style questions. When the query text looks like a
# sentence/question rather than a label, fall back to "object".
_MAX_LABEL_LEN = 40


def _extract_prompt(query: Query) -> str:
    """Extract a SAM3-compatible text prompt from the query.

    SAM3 needs short object labels like "person", "shoe", "dog".
    If the query text is a long sentence or question, default to "object".
    """
    text = (query.query_text or "").strip()
    if not text or len(text) > _MAX_LABEL_LEN or "?" in text:
        return "person"
    return text


class SAMInterface(LocalModelInterface):
    """SAM 3.1 via local inference — supports IMAGE_ENC capability.

    Query contract:
        query.images[0]  — RGB input image (PIL Image or base64 str)
        query.query_text  — short object label to segment (e.g. "person", "shoe")
                            Long sentences/questions are ignored; defaults to "object".
    """

    def __init__(self, seed: int | None = None) -> None:
        from model_router.models.sam import SAMModel

        super().__init__(model=SAMModel(confidence_threshold=0.3), seed=seed)

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.IMAGE_ENC]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        images = extract_images(query)

        if not images:
            raise ValueError(
                "SAMInterface requires an image in Query.images"
            )

        prompt = _extract_prompt(query)

        # Process all images, merge results
        all_masks = []
        all_labels = []
        all_scores = []
        summaries = []

        for i, img in enumerate(images):
            result = self.model.generate(image=img, prompt=prompt)
            prefix = f"img{i}" if len(images) > 1 else ""
            for j, (mask, label, score) in enumerate(
                zip(result["masks"], result["labels"], result["scores"])
            ):
                all_masks.append(mask)
                all_labels.append(f"{prefix}_{label}" if prefix else label)
                all_scores.append(score)
            summaries.append(
                f"Image {i}: {result['usage']['output_masks']} object(s)"
                if len(images) > 1
                else result["text"]
            )

        text = "\n".join(summaries) if len(images) > 1 else summaries[0]

        return {
            "text": text,
            "masks": all_masks,
            "labels": all_labels,
            "scores": all_scores,
            "usage": {
                "input_images": len(images),
                "output_masks": len(all_masks),
            },
        }
