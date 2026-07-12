# Quality Control Pipeline

A standalone, reusable component for iterative critique-and-revise loops. Any pipeline can plug into this with minimal configuration.

## Installation

The module is part of the `algos` package. No additional installation required.

## Quick Start

```python
import json
from quality_control import QualityControlPipeline, QualityControlConfig, QCFeedback
from model_router.query import StructuredPrompt

# 1. Define system prompt for evaluation
system_prompt = StructuredPrompt(
    base_instruction="You are a code reviewer.",
    sections={"Criteria": "Check for bugs, style issues, and best practices."},
    critical_requirements=["No security vulnerabilities"],
    requirements=["Follow coding standards"],
)

# 2. Define response parser (converts LLM response to QCFeedback)
def parse_response(response: dict, model: str) -> QCFeedback:
    data = json.loads(response["text"])
    return QCFeedback(
        action=data["action"],
        feedback=data["feedback"],
        checklist=[...],  # Parse checklist items
    )

# 3. Define revision function (called when content needs fixing)
def revise_content(content: str, feedback: QCFeedback) -> str:
    # Your revision logic here - could call an LLM, apply rules, etc.
    return revised_content

# 4. Run quality control
qc = QualityControlPipeline(
    config=QualityControlConfig(max_iterations=5, model_interfaces=["openai_gpt5"]),
)

result = qc.run(
    content="def foo(): pass",
    system_prompt=system_prompt,
    response_parser=parse_response,
    revise_fn=revise_content,
)

if result.approved:
    print("Content approved!")
    print(result.content)
else:
    print(f"Max iterations reached. Final content: {result.content}")
```

## API Reference

### QualityControlPipeline

Main pipeline class.

```python
class QualityControlPipeline:
    def __init__(self, config: QualityControlConfig):
        ...

    def run(
        self,
        content: str,                                       # Content to evaluate
        system_prompt: StructuredPrompt,                    # Evaluation criteria
        response_parser: Callable[[dict, str], QCFeedback], # Parse LLM response
        revise_fn: Callable[[str, QCFeedback], str],        # Revision callback
        state: QCState | None = None,                       # For restart recovery
    ) -> QCResult:
        ...
```

### QualityControlConfig

```python
@dataclass
class QualityControlConfig:
    max_iterations: int = 5                         # Max revision attempts
    model_interfaces: List[str] = ["openai_gpt5"]   # Models to use for evaluation
```

### QCFeedback

Simple feedback from a single evaluation (no checklist). Used for concept review.

```python
@dataclass
class QCFeedback:
    action: Literal["proceed", "revise"]  # Whether to approve or request revision
    feedback: str                          # Summary of issues or approval
    model: str                             # Which model was used
```

### QCFeedbackWithChecklist

Feedback with a structured checklist for iterative resolution. Used for beats, prose, and script review.

```python
@dataclass
class QCFeedbackWithChecklist:
    feedback: QCFeedback                   # The inner feedback object
    checklist: List[QCChecklistItem]       # Specific issues to address
```

Access `action` and `model` via delegation properties (e.g., `feedback_with_checklist.action` works).
Access the feedback text via `feedback_with_checklist.feedback.feedback`.

Use `QCFeedbackWithChecklist.from_flat_dict(data, model)` to parse LLM output (flat JSON) into the nested structure.

### QCChecklistItem

Individual issue to address.

```python
@dataclass
class QCChecklistItem:
    id: str                                # Unique identifier
    description: str                       #
    done_when: str                         # Verification criteria
    priority: Literal["P0", "P1", "P2"]    # P0=blocking, P1=important, P2=minor
    completed: bool = False
    completed_at_iteration: int | None = None
```

**Note:** The `description` field is opaque to the QC library. Callers control its format via their prompts - it can be plain text, JSON, or any structure the caller's critic and revision prompts agree on. For example, localization info (e.g., "Beat 3: ..." or "Page 5: ...") should be embedded in the description text rather than as separate fields.

### QCResult

Result of the quality control loop.

```python
@dataclass
class QCResult:
    content: str                                    # Final content (approved or max iterations)
    approved: bool                                  # Whether content was approved
    iterations: int                                 # Number of evaluation passes
    feedback_history: List[QCFeedbackWithChecklist] # All feedback from the loop
```

### QCState

State for restart recovery. Pass this to resume an interrupted loop.

```python
@dataclass
class QCState:
    feedback_history: List[QCFeedbackWithChecklist] # Previous feedback
    content_history: List[str]                      # Previous content versions
    iteration: int                                  # Current iteration
    accumulated_checklist: List[QCChecklistItem]    # Merged checklist
```

## Checklist Merging

The pipeline tracks issues across iterations:

- Items from previous iteration not in new feedback = marked complete
- Items in both = kept with current status
- New items = added as incomplete

Use `merge_checklists()` directly if needed:

```python
from quality_control import merge_checklists

merged = merge_checklists(
    previous=old_checklist,
    new_items=new_checklist,
    current_iteration=2,
)
```

## Testing

Run tests:

```bash
python -m unittest quality_control.checklist_test quality_control.pipeline_test -v
```

### Using DummyInterface for Tests

Use `DummyInterface` from `model_router` for testing:

```python
from model_router.interfaces.dummy_interface import DummyInterface

# Configure responses before test
DummyInterface.set_responses([
    {"text": '{"action": "revise", "feedback": "Fix X"}'},
    {"text": '{"action": "proceed", "feedback": "Looks good"}'},
])

# Run pipeline with "dummy" interface
config = QualityControlConfig(model_interfaces=["dummy"])
qc = QualityControlPipeline(config)
result = qc.run(...)

# Clean up after test
DummyInterface.reset()
```

## File Structure

```
algos/quality_control/
├── __init__.py          # Public exports
├── pipeline.py          # QualityControlPipeline
├── types.py             # QualityControlConfig, QCState, QCFeedback, etc.
├── checklist.py         # merge_checklists()
├── checklist_test.py    # Tests for checklist
├── pipeline_test.py     # Tests for pipeline
└── README.md
```
