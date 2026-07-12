
"""
Base prompt formatter for model-agnostic prompt construction.

This module provides the base PromptFormatter class that can be extended
by model-specific formatters in their respective interface files.

Model-specific formatters should be defined in:
- anthropic_interface.py (ClaudePromptFormatter)
- gemini_interface.py (GeminiPromptFormatter)
- openai_interface.py (OpenAIPromptFormatter)
"""

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Dict, List, TypeVar

from model_router.query import Query, StructuredPrompt

# TypeVar for Query and its subclasses
Q = TypeVar("Q", bound=Query)


class PromptFormatter(ABC):
    """Abstract base class for model-specific prompt formatting.

    Subclasses implement model-specific formatting for optimal instruction
    following. Each model family (Claude, Gemini, OpenAI) has different
    preferences for how prompts should be structured.
    """

    @abstractmethod
    def wrap_section(self, name: str, content: str) -> str:
        """Wrap a section with appropriate delimiters.

        Args:
            name: Section name/title
            content: Section content

        Returns:
            Formatted section string
        """
        ...

    @abstractmethod
    def format_requirements(
        self, requirements: List[str], critical: List[str]
    ) -> str:
        """Format requirements with critical items first.

        Args:
            requirements: List of standard requirements
            critical: List of critical/must-follow requirements

        Returns:
            Formatted requirements string
        """
        ...

    def format_system_prompt(self, prompt: str) -> str:
        """Format system prompt for the specific model.

        Override in subclasses for model-specific transformations.

        Args:
            prompt: Raw system prompt

        Returns:
            Formatted system prompt
        """
        return prompt

    def build_prompt(
        self,
        base_instruction: str,
        sections: Dict[str, str],
        critical_requirements: List[str] | None = None,
        requirements: List[str] | None = None,
    ) -> str:
        """Build a complete prompt from components.

        Args:
            base_instruction: The main instruction/role description
            sections: Named sections to include (format specs, examples, etc.)
            critical_requirements: Must-follow requirements (placed first)
            requirements: Standard requirements

        Returns:
            Complete formatted prompt
        """
        result = base_instruction

        for name, content in sections.items():
            result += "\n\n" + self.wrap_section(name, content)

        if critical_requirements or requirements:
            result += "\n\n" + self.format_requirements(
                requirements or [], critical_requirements or []
            )

        return self.format_system_prompt(result)

    def __call__(self, query: Q) -> Q:
        """Format a Query and return a new Query with formatted system_prompt.

        This makes the formatter callable: formatted_query = formatter(query)

        Args:
            query: The Query (or subclass) to format

        Returns:
            A new Query of the same type with formatted system_prompt.
            The structured_prompt field will be None (formatting has been applied).
        """
        formatted_prompt: str | None = None

        if query.structured_prompt is not None:
            # Format the structured prompt
            formatted_prompt = self.build_prompt(
                base_instruction=query.structured_prompt.base_instruction,
                sections=query.structured_prompt.sections,
                critical_requirements=query.structured_prompt.critical_requirements or None,
                requirements=query.structured_prompt.requirements or None,
            )
        elif query.system_prompt is not None:
            # Apply model-specific transformations to plain string prompt
            formatted_prompt = self.format_system_prompt(query.system_prompt)
        else:
            # No prompt to format, return unchanged
            return query

        # Build kwargs from all fields of the query
        kwargs: Dict[str, any] = {}
        for f in fields(query):
            if f.name == "system_prompt":
                kwargs["system_prompt"] = formatted_prompt
            elif f.name == "structured_prompt":
                kwargs["structured_prompt"] = None
            else:
                kwargs[f.name] = getattr(query, f.name)

        # Return new instance of the same type
        return type(query)(**kwargs)


class GenericPromptFormatter(PromptFormatter):
    """Generic formatter with no model-specific optimizations.

    Uses simple formatting that works reasonably across all models.
    This is the default formatter used when no model-specific formatter
    is available.
    """

    def wrap_section(self, name: str, content: str) -> str:
        """Wrap content with simple header."""
        return f"{name}:\n{content}"

    def format_requirements(
        self, requirements: List[str], critical: List[str]
    ) -> str:
        """Format requirements with simple headers."""
        result = ""

        if critical:
            result += "Critical Requirements:\n"
            result += "\n".join(f"* {r}" for r in critical)

        if requirements:
            if result:
                result += "\n\n"
            result += "Requirements:\n"
            result += "\n".join(f"* {r}" for r in requirements)

        return result


# Default formatter instance for models without specific formatting
DEFAULT_FORMATTER = GenericPromptFormatter()
