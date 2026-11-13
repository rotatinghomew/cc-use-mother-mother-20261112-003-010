"""Canonical JSON, hashing, time parsing, and deterministic helpers."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .errors import ContractError


UTC = timezone.utc


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_json(value: Any) -> str:
    return digest_text(canonical_json(value))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_time(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("timestamp must be a non-empty string", details={"value": value})
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContractError("invalid ISO-8601 timestamp", details={"value": value}) from exc
    if parsed.tzinfo is None:
        raise ContractError("timestamp must contain timezone", details={"value": value})
    return parsed.astimezone(UTC)


def format_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ContractError("cannot format naive datetime")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def micros(value: str | datetime) -> int:
    parsed = parse_time(value) if isinstance(value, str) else value.astimezone(UTC)
    return int(parsed.timestamp() * 1_000_000)


def from_micros(value: int) -> str:
    return format_time(datetime.fromtimestamp(value / 1_000_000, tz=UTC))


def require_string(mapping: dict, key: str, *, allow_empty: bool = False) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ContractError("field must be a string", details={"field": key})
    value = value.strip()
    if not allow_empty and not value:
        raise ContractError("field cannot be empty", details={"field": key})
    return value


def optional_string(mapping: dict, key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ContractError("optional field must be a non-empty string", details={"field": key})
    return value.strip()


def require_bool(mapping: dict, key: str, default: bool | None = None) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ContractError("field must be boolean", details={"field": key})
    return value


def require_int(mapping: dict, key: str, *, minimum: int | None = None, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError("field must be integer", details={"field": key})
    if minimum is not None and value < minimum:
        raise ContractError("integer below minimum", details={"field": key, "minimum": minimum})
    return value


def require_number(mapping: dict, key: str, *, minimum: float | None = None, default: float | None = None) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError("field must be finite number", details={"field": key})
    result = float(value)
    if minimum is not None and result < minimum:
        raise ContractError("number below minimum", details={"field": key, "minimum": minimum})
    return result


def require_list(mapping: dict, key: str, *, default: list | None = None) -> list:
    value = mapping.get(key, default)
    if not isinstance(value, list):
        raise ContractError("field must be an array", details={"field": key})
    return value


def require_mapping(mapping: dict, key: str, *, default: dict | None = None) -> dict:
    value = mapping.get(key, default)
    if not isinstance(value, dict):
        raise ContractError("field must be an object", details={"field": key})
    return value


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    fraction = clamp(fraction, 0.0, 1.0)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pairwise(values: Sequence[Any]) -> Iterator[tuple[Any, Any]]:
    for index in range(len(values) - 1):
        yield values[index], values[index + 1]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError("JSON file does not exist", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise ContractError("invalid JSON document", details={"path": str(path), "line": exc.lineno}) from exc


def deterministic_id(prefix: str, value: Any, length: int = 20) -> str:
    return f"{prefix}_{digest_json(value)[:length]}"


def normalize_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ContractError("tags must be an array")
    tags: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ContractError("tag must be a non-empty string")
        tags.append(item.strip())
    return tuple(sorted(set(tags)))


def json_clone(value: Any) -> Any:
    return json.loads(canonical_json(value))

