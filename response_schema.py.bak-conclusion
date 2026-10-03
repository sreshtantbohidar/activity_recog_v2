"""Machine-readable response schemas plus a dependency-free validator/repairer.

Every response field the service guarantees is described here as a JSON Schema
dict. Model-derived fields (per-frame classification, the whole-video reasoning
result, the audio stage) and the top-level analysis envelope are modelled
explicitly; anything passed through unchanged (object tracking, the PDF result,
cadence analysis) stays permissive via ``additionalProperties``.

Two entry points:

* ``validate_response(response, schema)`` returns the list of violations. It is
  non-mutating and is used by the tests.
* ``validate_and_repair(response, schema)`` logs the violations and returns a
  response coerced into the schema, so a client always receives a conforming
  body even when a model answers with the wrong type or a missing field. This is
  applied at the API boundary in app.py.

The engine intentionally implements only the small JSON Schema subset the
service uses (type, properties, required, additionalProperties, items, enum,
minimum, maximum, minItems, anyOf, default) so it needs no third-party package.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from media_utils import ACTIVITY_LABELS, LOCATION_TYPES

logger = logging.getLogger("activity-recognition.schema")

# Default value used when a field is missing or cannot be coerced.
_TYPE_DEFAULTS: Dict[str, Any] = {
    "object": {}, "array": [], "string": "", "number": 0.0, "integer": 0,
    "boolean": False, "null": None,
}
_NO_COERCION = object()

# --- Model-derived fields ---------------------------------------------------

# The six fields the vision model is asked to return for one frame (or image).
_MODEL_FIELDS: Dict[str, Any] = {
    "scene_summary": {"type": "string", "default": ""},
    "key_objects": {"type": "array", "items": {"type": "string"}, "default": []},
    "location_type": {"type": "string", "enum": sorted(LOCATION_TYPES), "default": "unknown"},
    "detected_activity": {"type": "array", "minItems": 1, "default": ["miscellaneous"],
        "items": {"type": "string", "enum": sorted(ACTIVITY_LABELS), "default": "miscellaneous"}},
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
    "reasoning": {"type": "string", "default": ""},
}

IMAGE_RESULT_SCHEMA = {
    "type": "object", "properties": dict(_MODEL_FIELDS),
    "required": ["detected_activity", "confidence"], "additionalProperties": True,
}

FRAME_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "frame_index": {"type": "integer", "minimum": 0, "default": 0},
        "timestamp_sec": {"type": "number", "minimum": 0, "default": 0.0},
        **_MODEL_FIELDS,
    },
    "required": ["detected_activity", "confidence"], "additionalProperties": True,
}

TRANSCRIPT_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "start_sec": {"type": "number", "minimum": 0, "default": 0.0},
        "end_sec": {"type": "number", "minimum": 0, "default": 0.0},
        "text": {"type": "string", "default": ""},
    },
    "required": ["start_sec", "end_sec", "text"], "additionalProperties": True,
}

SOUND_EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "start_sec": {"type": "number", "minimum": 0, "default": 0.0},
        "end_sec": {"type": "number", "minimum": 0, "default": 0.0},
        "label": {"type": "string", "default": ""},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
    },
    "required": ["label", "start_sec", "end_sec", "confidence"], "additionalProperties": True,
}

AUDIO_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "has_audio": {"type": "boolean", "default": False},
        "transcript": {"type": "array", "items": TRANSCRIPT_SEGMENT_SCHEMA, "default": []},
        "sound_events": {"type": "array", "items": SOUND_EVENT_SCHEMA, "default": []},
        "warnings": {"type": "array", "items": {"type": "string"}, "default": []},
    },
    "required": ["has_audio", "transcript", "sound_events"], "additionalProperties": True,
}

VIDEO_LEVEL_REASONING_SCHEMA = {
    "default": None,
    "anyOf": [
        {"type": "null"},
        {"type": "object",
            "properties": {
                "activity": {"type": "string", "enum": sorted(ACTIVITY_LABELS), "default": "miscellaneous"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
                "reasoning": {"type": "string", "default": ""},
            },
            "required": ["activity", "confidence"], "additionalProperties": True},
    ],
}

# --- Analysis envelopes -----------------------------------------------------

VIDEO_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "frame_analysis": {"type": "array", "items": FRAME_RESULT_SCHEMA, "default": []},
        "audio_analysis": AUDIO_ANALYSIS_SCHEMA,
        "backends_used": {"type": "array", "items": {"type": "string"}, "default": []},
        "overall_analysis": {
            "type": "object",
            "properties": {
                "dominant_activity": {"type": "string", "enum": sorted(ACTIVITY_LABELS), "default": "miscellaneous"},
                "video_level_reasoning": VIDEO_LEVEL_REASONING_SCHEMA,
            },
            "additionalProperties": True,
        },
    },
    "required": ["frame_analysis", "audio_analysis"], "additionalProperties": True,
}

DETAILED_VIDEO_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {"type": "object", "default": {}},
        "frame_analysis": {"type": "array", "items": FRAME_RESULT_SCHEMA, "default": []},
        "frame_warnings": {"type": "array", "items": {"type": "string"}, "default": []},
        "audio_analysis": AUDIO_ANALYSIS_SCHEMA,
        "backends_used": {"type": "array", "items": {"type": "string"}, "default": []},
        "detailed_report": {
            "type": "object",
            "properties": {"video_level_reasoning": VIDEO_LEVEL_REASONING_SCHEMA},
            "additionalProperties": True,
        },
    },
    "required": ["frame_analysis", "audio_analysis"], "additionalProperties": True,
}

_ENVELOPE_PROPERTIES = {
    "filename": {"type": "string", "default": ""},
    "media_type": {"type": "string", "enum": ["video", "image"], "default": "video"},
}


def _envelope(analysis_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object",
        "properties": {**_ENVELOPE_PROPERTIES, "analysis": analysis_schema},
        "required": ["filename", "media_type", "analysis"], "additionalProperties": True}


IMAGE_ANALYSIS_ENVELOPE_SCHEMA = _envelope(IMAGE_RESULT_SCHEMA)
VIDEO_ANALYSIS_ENVELOPE_SCHEMA = _envelope(VIDEO_ANALYSIS_SCHEMA)
DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA = _envelope(DETAILED_VIDEO_ANALYSIS_SCHEMA)

# Discoverable map of endpoint name -> response schema.
RESPONSE_SCHEMAS = {
    "analyze-image": IMAGE_ANALYSIS_ENVELOPE_SCHEMA,
    "analyze-video": VIDEO_ANALYSIS_ENVELOPE_SCHEMA,
    "analyze-video-detailed": DETAILED_VIDEO_ANALYSIS_ENVELOPE_SCHEMA,
}

# --- Engine -----------------------------------------------------------------


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return "unknown"


def _type_ok(value: Any, expected: str) -> bool:
    actual = _type_name(value)
    if expected == "number":
        return actual in {"number", "integer"}
    if expected == "integer":
        return actual == "integer"
    return actual == expected


def _default_for(schema: Dict[str, Any]) -> Any:
    if "default" in schema:
        return schema["default"]
    if "anyOf" in schema:
        return _default_for(schema["anyOf"][0])
    if "enum" in schema:
        return schema["enum"][0]
    types = schema.get("type")
    if isinstance(types, list):
        for name in types:
            if name != "null":
                return _TYPE_DEFAULTS.get(name)
        return None
    if isinstance(types, str):
        return _TYPE_DEFAULTS.get(types)
    return None


def _soft_coerce(value: Any, expected: List[str]) -> Any:
    """Best-effort conversion between scalar types; returns _NO_COERCION if none fits."""
    for target in expected:
        if target == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if target in ("number", "integer"):
            if isinstance(value, bool):
                return int(value) if target == "integer" else float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        if target == "array" and isinstance(value, str):
            return [value]
    return _NO_COERCION


def _matches_type(value: Any, schema: Dict[str, Any]) -> bool:
    types = schema.get("type")
    if types is None:
        return True
    expected = types if isinstance(types, list) else [types]
    return any(_type_ok(value, name) for name in expected)


def _normalize(value: Any, schema: Dict[str, Any], path: str) -> Tuple[Any, List[str]]:
    """Coerce ``value`` to ``schema``; return (normalized_value, violations)."""
    problems: List[str] = []

    if "anyOf" in schema:
        matching = [option for option in schema["anyOf"] if _matches_type(value, option)]
        if matching:
            return _normalize(value, matching[0], path)
        return _default_for(schema), [f"{path}: got {_type_name(value)}, no allowed variant matched"]

    types = schema.get("type")
    expected = types if isinstance(types, list) else ([types] if types else [])
    if expected and not any(_type_ok(value, name) for name in expected):
        problems.append(f"{path}: expected {'|'.join(expected)}, got {_type_name(value)}")
        coerced = _soft_coerce(value, expected)
        if coerced is _NO_COERCION:
            return _default_for(schema), problems
        value = coerced

    if "enum" in schema and not isinstance(value, (dict, list)) and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} is not one of {schema['enum']}")
        value = _default_for(schema)

    primary = (next((name for name in expected if _type_ok(value, name)), expected[0])
        if expected else _type_name(value))

    if primary == "object":
        value = dict(value) if isinstance(value, dict) else {}
        properties = schema.get("properties", {})
        for name, property_schema in properties.items():
            if name in value:
                value[name], child = _normalize(value[name], property_schema, f"{path}.{name}")
                problems += child
            else:
                if name in schema.get("required", []):
                    problems.append(f"{path}.{name}: missing required field")
                value[name] = _default_for(property_schema)
        if schema.get("additionalProperties", True) is False:
            for extra in [key for key in value if key not in properties]:
                problems.append(f"{path}.{extra}: unexpected field removed")
                del value[extra]
    elif primary == "array":
        if isinstance(value, (list, tuple)):
            value = list(value)
        elif value is None:
            value = []
        else:
            value = [value]
        item_schema = schema.get("items")
        if item_schema:
            normalized_items = []
            for index, item in enumerate(value):
                item, child = _normalize(item, item_schema, f"{path}[{index}]")
                problems += child
                normalized_items.append(item)
            value = normalized_items
        minimum_items = schema.get("minItems")
        if minimum_items and len(value) < minimum_items:
            problems.append(f"{path}: fewer than {minimum_items} item(s)")
            value = _default_for(schema)
    elif primary in ("number", "integer"):
        try:
            value = int(value) if primary == "integer" else float(value)
        except (TypeError, ValueError):
            value = _default_for(schema)
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: {value} below minimum {schema['minimum']}")
            value = schema["minimum"]
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: {value} above maximum {schema['maximum']}")
            value = schema["maximum"]
    elif primary == "boolean":
        if not isinstance(value, bool):
            value = _default_for(schema)
    elif primary == "string":
        if not isinstance(value, str):
            value = "" if value is None else str(value)

    return value, problems


def validate_response(response: Any, schema: Dict[str, Any]) -> List[str]:
    """Return the schema violations in ``response`` without modifying it."""
    _, problems = _normalize(response, schema, "$")
    return problems


def normalize_response(response: Any, schema: Dict[str, Any]) -> Any:
    """Return a copy of ``response`` coerced into ``schema`` (no logging)."""
    repaired, _ = _normalize(response, schema, "$")
    return repaired


def validate_and_repair(response: Any, schema: Dict[str, Any]) -> Any:
    """Log violations and return a response coerced into ``schema``."""
    repaired, problems = _normalize(response, schema, "$")
    if problems:
        logger.warning("Response schema violations (%d): %s", len(problems), "; ".join(problems[:10]))
    return repaired
