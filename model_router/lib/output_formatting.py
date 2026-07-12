
"""Utilities for parsing and decoding LLM text output into Python objects."""

import dataclasses
import json
import logging
import re
import types
from datetime import datetime
from typing import Any, Dict, List, Type, TypeVar, Union, get_args, get_origin, get_type_hints

from json_repair import repair_json

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _strip_xml_wrapper(json_string: str) -> str:
    """Strip outer XML tags that LLMs sometimes wrap around JSON output.

    e.g. <feedback_output_format>{"key": "value"}</feedback_output_format>
    """
    pattern = re.compile(r"^\s*<(\w+)>\s*(.*?)\s*</\1>\s*$", re.DOTALL)
    match = pattern.match(json_string)
    if match:
        return _strip_xml_wrapper(match.group(2))
    return json_string


def safe_json_decode(json_string: str) -> Any:
    """Parse a JSON string from LLM output, repairing common formatting errors.

    Handles markdown code fences, XML wrappers, trailing commas, unescaped
    quotes, and other issues common in LLM responses. Uses json-repair for
    the heavy lifting.

    Args:
        json_string: Raw string from an LLM response.

    Returns:
        Decoded Python object (dict, list, etc.).

    Raises:
        json.JSONDecodeError: If the string cannot be repaired and decoded.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        original = json_string

        json_string = _strip_xml_wrapper(json_string)

        if "```json" in json_string:
            json_string = json_string.replace("```json", "")
        if "json\n" in json_string:
            json_string = json_string.replace("json", "")
        if "```" in json_string:
            json_string = json_string.replace("```", "")

        json_string = json_string.strip()

        try:
            return json.loads(repair_json(json_string))
        except Exception as e:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            error_file = f"/tmp/json_decode_error_{timestamp}.json"
            with open(error_file, "w") as f:
                f.write("# Original string:\n")
                f.write(original)
                f.write("\n\n# After cleanup:\n")
                f.write(json_string)
                f.write(f"\n\n# Error: {e}\n")
            logger.error(f"JSON decode failed, saved debug output to {error_file}")
            raise


def _decode_value(field_type: Type, value: Any) -> Any:
    """Recursively decode a value into its expected Python type."""
    if value is None:
        return None

    if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
        return safe_dataclass_decode(field_type, value)

    origin = get_origin(field_type)
    if origin is list:
        args = get_args(field_type)
        if args and dataclasses.is_dataclass(args[0]):
            return [safe_dataclass_decode(args[0], item) for item in value]

    if origin is Union or isinstance(field_type, types.UnionType):
        args = get_args(field_type)
        for arg in args:
            if arg is type(None):
                continue
            if dataclasses.is_dataclass(arg) and isinstance(value, dict):
                return safe_dataclass_decode(arg, value)
            arg_origin = get_origin(arg)
            if arg_origin is list:
                arg_args = get_args(arg)
                if arg_args and dataclasses.is_dataclass(arg_args[0]) and isinstance(value, list):
                    return [safe_dataclass_decode(arg_args[0], item) for item in value]

    return value


def safe_dataclass_decode(
    dataclass_type: Type[T], data: Union[str, Dict[str, Any]], **extra_fields
) -> T:
    """Decode a JSON string or dict into a dataclass, filtering unknown keys.

    Recursively decodes nested dataclasses and lists of dataclasses. Unknown
    keys from LLM output are silently filtered and logged as warnings.

    Args:
        dataclass_type: Target dataclass type.
        data: JSON string or dict from LLM response.
        **extra_fields: Additional fields to inject (not from LLM output).

    Returns:
        Populated dataclass instance.

    Raises:
        json.JSONDecodeError: If a string input cannot be decoded.
        TypeError: If required fields are missing.
    """
    if isinstance(data, str):
        data = safe_json_decode(data)

    fields = dataclass_type.__dataclass_fields__
    valid_keys = {f.name for f in fields.values()}
    filtered_data: Dict[str, Any] = {}
    filtered_out: List[str] = []

    try:
        resolved_types = get_type_hints(dataclass_type)
    except Exception:
        resolved_types = {name: f.type for name, f in fields.items()}

    for key, value in data.items():
        if key in valid_keys:
            field_type = resolved_types.get(key, fields[key].type)
            filtered_data[key] = _decode_value(field_type, value)
        else:
            filtered_out.append(key)

    if filtered_out:
        logger.warning(f"Filtered unknown keys from {dataclass_type.__name__}: {filtered_out}")

    filtered_data.update(extra_fields)

    required = [
        name for name, f in fields.items()
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    missing = [f for f in required if f not in filtered_data]
    if missing:
        raise TypeError(
            f"{dataclass_type.__name__} missing required fields: {missing}. "
            f"Received: {list(data.keys())}"
        )

    return dataclass_type(**filtered_data)
