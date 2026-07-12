"""Data types for quality control pipeline."""

from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Dict, List, Literal

from model_router.query import StructuredPrompt


# --- Feedback Types ---


@dataclass
class QCChecklistItem:
    """Base checklist item for quality control feedback.

    The description field is opaque to the QC library - callers control its format
    via their prompts. It can be plain text, JSON, or any structure the caller's
    critic and revision prompts agree on.
    """

    id: str
    description: str  # Opaque to QC - caller controls format via prompt
    done_when: str
    priority: Literal["P0", "P1", "P2"]  # P0=blocking, P1=important, P2=minor
    completed: bool = False
    completed_at_iteration: int | None = None
    focus_area: str | None = None  # Stage/critic name that created this item

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "QCChecklistItem":
        """Create from dict, filtering out unknown fields (LLM may hallucinate extra fields)."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        # LLM may omit priority - default to P1
        if "priority" not in filtered:
            filtered["priority"] = "P1"
        return cls(**filtered)

    @staticmethod
    def schema(description_hint: str = "The specific issue to fix") -> str:
        """Return JSON schema for LLM prompts.

        Args:
            description_hint: Caller-provided hint for what description should contain.
                              e.g., "Start with beat reference (e.g., 'Beat 3: ...')"

        Returns:
            JSON schema string for use in prompts.
        """
        return f"""{{
    "id": string, // Unique identifier for tracking this issue
    "description": string, // {description_hint}
    "done_when": string, // Clear verification criteria
    "priority": string, // "P0" (blocking), "P1" (important), or "P2" (minor)
    "completed": boolean, // false for new items, true if previously flagged and now fixed
    "completed_at_iteration": int | null // null for new/incomplete items
}}"""


@dataclass
class QCFeedback:
    """Simple feedback from a single evaluation - no checklist."""

    action: Literal["proceed", "revise"]
    feedback: str
    model: str = ""  # Which model was used

    @staticmethod
    def schema() -> str:
        """Return JSON schema for LLM prompts."""
        return """{
    "action": string, // "proceed" if approved, "revise" if changes needed
    "feedback": string // Brief summary with specific suggestions for improvement
}"""


@dataclass
class QCFeedbackWithChecklist:
    """Feedback from a single evaluation with structured checklist.

    Contains a QCFeedback plus a list of checklist items.

    Note: The LLM returns a flat structure with action/feedback/checklist as siblings.
    Use from_flat_dict() to parse LLM output into this nested structure.
    """

    feedback: QCFeedback
    checklist: List[QCChecklistItem] = field(default_factory=list)
    # Triage selections: maps focusArea (e.g. "page_10") -> selected item IDs
    page_triage_selections: Dict[str, List[str]] = field(default_factory=dict)
    # Cost data for this iteration
    critique_cost_usd: float = 0.0
    critique_input_tokens: int = 0
    critique_output_tokens: int = 0
    revision_cost_usd: float = 0.0
    revision_input_tokens: int = 0
    revision_output_tokens: int = 0

    @property
    def action(self) -> Literal["proceed", "revise"]:
        """Delegate to inner feedback."""
        return self.feedback.action

    @action.setter
    def action(self, value: Literal["proceed", "revise"]) -> None:
        """Set action on inner feedback."""
        self.feedback.action = value

    @property
    def model(self) -> str:
        """Delegate to inner feedback."""
        return self.feedback.model

    @model.setter
    def model(self, value: str) -> None:
        """Set model on inner feedback."""
        self.feedback.model = value

    @classmethod
    def from_flat_dict(cls, data: Dict, model: str = "") -> "QCFeedbackWithChecklist":
        """Create from flat dict structure (as returned by LLM).

        LLM returns: {"action": "...", "feedback": "...", "checklist": [...]}
        This method converts to nested structure.

        Args:
            data: Flat dictionary with action, feedback, checklist keys.
            model: Which model was used (injected by caller).

        Returns:
            QCFeedbackWithChecklist with nested QCFeedback.
        """
        checklist = [
            QCChecklistItem.from_dict(item) if isinstance(item, dict) else item
            for item in data.get("checklist", [])
        ]
        return cls(
            feedback=QCFeedback(
                action=data["action"],
                feedback=data["feedback"],
                model=model,
            ),
            checklist=checklist,
        )

    @staticmethod
    def schema(description_hint: str = "The specific issue to fix") -> str:
        """Return JSON schema for LLM prompts.

        Args:
            description_hint: Caller-provided hint for checklist item descriptions.
                              e.g., "Start with page reference (e.g., 'Page 5: ...')"

        Returns:
            JSON schema string for use in prompts.
        """
        return f"""{{
    "action": string, // "proceed" if approved, "revise" if changes needed
    "feedback": string, // Brief summary of the overall feedback
    "checklist": [ // Array of specific issues to address
        {QCChecklistItem.schema(description_hint)}
    ]
}}"""


@dataclass
class CritiqueRequest:
    """Request for critiquing content.

    Attributes:
        content: The actual content to critique (e.g., beats JSON, prose text).
                 This is what QCResult.content will contain after revisions.
        context: Supporting information for the critique (concept, locations,
                 characters, iteration info). Provides background for the critic
                 but is not part of the revised output.
        control_guide: The system prompt that determines how to critique/revise.
                       Defines the critic's role and evaluation criteria.
    """

    content: str
    context: str
    control_guide: StructuredPrompt

    @property
    def query(self) -> str:
        """Combine context and content for LLM query.

        The context comes first to set the scene, followed by the content
        that needs to be evaluated.
        """
        return f"{self.context}\n\n---\n\nContent to Review:\n{self.content}"


@dataclass
class QCResult:
    """Result of a quality control loop."""

    content: str
    approved: bool
    iterations: int
    feedback_history: List[QCFeedbackWithChecklist] = field(default_factory=list)


# --- Config Types ---


@dataclass
class PlaybookConfig:
    """Base configuration for playbooks."""

    max_iterations: int = 5


@dataclass
class QualityControlConfig:
    """Configuration for the quality control pipeline."""

    playbook: str = "GlobalSolve"  # Playbook strategy to use
    playbook_config: PlaybookConfig = field(default_factory=PlaybookConfig)


@dataclass
class QCState:
    """State for restart recovery."""

    feedback_history: List[QCFeedbackWithChecklist] = field(default_factory=list)
    content_history: List[str] = field(default_factory=list)
    iteration: int = 0
    accumulated_checklist: List[QCChecklistItem] = field(default_factory=list)
    # Model locking: once a model is selected on first iteration, preserve it
    # throughout the critic loop to prevent interpretation drift between models
    locked_model: str | None = None


@dataclass
class StageDefinition:
    """Definition for a single QC stage in MultiStageSolve.

    Each stage can use a different playbook (SwarmSolve, GlobalSolve) internally.
    """

    name: str  # e.g., "structural", "vocabulary", "formatting"
    playbook_type: Literal["SwarmSolve", "GlobalSolve"]
    playbook_config: "PlaybookConfig"  # Stage-specific playbook config
    control_guide: StructuredPrompt  # System prompt for this stage
    context_builder: "Callable[[str, List[QCChecklistItem], int], str] | None" = (
        None  # Optional context builder
    )
    iterate_until_proceed: bool = True  # If False, only do one iteration
    max_stage_iterations: int = 6


@dataclass
class MultiStageState(QCState):
    """State tracking for multi-stage QC.

    Extends QCState with stage-level tracking for restart recovery.
    """

    current_stage_index: int = 0
    stage_states: Dict[str, QCState] = field(default_factory=dict)
    stage_feedback_history: Dict[str, List[QCFeedbackWithChecklist]] = field(
        default_factory=dict
    )
