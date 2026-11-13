"""Content-addressed provenance envelopes for offline imports and exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, digest_bytes, digest_json, digest_text, read_json
from .errors import IntegrityError


@dataclass(frozen=True)
class ProvenanceRecord:
    record_id: str
    artifact_type: str
    source_name: str
    content_hash: str
    parent_hashes: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return {"record_id": self.record_id, "artifact_type": self.artifact_type,
                "source_name": self.source_name, "content_hash": self.content_hash,
                "parent_hashes": list(self.parent_hashes), "metadata": self.metadata}


class ProvenanceLedger:
    def __init__(self) -> None:
        self.records: dict[str, ProvenanceRecord] = {}

    def add_json(self, artifact_type: str, source_name: str, value: Any,
                 *, parent_hashes: Iterable[str] = (), metadata: dict | None = None) -> ProvenanceRecord:
        content_hash = digest_json(value)
        return self._add(artifact_type, source_name, content_hash, parent_hashes, metadata or {})

    def add_text(self, artifact_type: str, source_name: str, value: str,
                 *, parent_hashes: Iterable[str] = (), metadata: dict | None = None) -> ProvenanceRecord:
        content_hash = digest_text(value)
        return self._add(artifact_type, source_name, content_hash, parent_hashes, metadata or {})

    def add_file(self, artifact_type: str, path: str | Path,
                 *, parent_hashes: Iterable[str] = (), metadata: dict | None = None) -> ProvenanceRecord:
        file_path = Path(path)
        if not file_path.is_file():
            raise IntegrityError("provenance source file missing", details={"path": str(file_path)})
        file_metadata = {"bytes": file_path.stat().st_size, **(metadata or {})}
        return self._add(artifact_type, file_path.name, digest_bytes(file_path.read_bytes()),
                         parent_hashes, file_metadata)

    def verify_file(self, record_id: str, path: str | Path) -> dict:
        record = self._require(record_id)
        file_path = Path(path)
        if not file_path.is_file():
            return {"ok": False, "record_id": record_id, "error": "missing_file", "path": str(file_path)}
        actual = digest_bytes(file_path.read_bytes())
        return {"ok": actual == record.content_hash, "record_id": record_id,
                "expected": record.content_hash, "actual": actual, "path": str(file_path)}

    def lineage(self, record_id: str) -> list[dict]:
        self._require(record_id)
        result: list[dict] = []
        stack: list[tuple[str, int]] = [(record_id, 0)]
        visited: set[str] = set()
        while stack:
            current_id, depth = stack.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            record = self._require(current_id)
            result.append({"depth": depth, **record.to_dict()})
            parents = sorted(record.parent_hashes, reverse=True)
            for parent_hash in parents:
                parent_id = self._record_id_by_content_hash(parent_hash)
                if parent_id:
                    stack.append((parent_id, depth + 1))
                else:
                    result.append({"depth": depth + 1, "external_content_hash": parent_hash})
        return result

    def roots(self) -> list[ProvenanceRecord]:
        referenced = {parent for record in self.records.values() for parent in record.parent_hashes}
        return sorted((record for record in self.records.values() if record.content_hash not in referenced),
                      key=lambda item: item.record_id)

    def leaves(self) -> list[ProvenanceRecord]:
        parent_hashes = {parent for record in self.records.values() for parent in record.parent_hashes}
        return sorted((record for record in self.records.values() if record.content_hash not in parent_hashes),
                      key=lambda item: item.record_id)

    def export(self) -> dict:
        records = [self.records[key].to_dict() for key in sorted(self.records)]
        graph = {record.record_id: sorted(record.parent_hashes) for record in self.records.values()}
        return {"schema_version": 1, "records": records, "graph": graph,
                "ledger_hash": digest_json({"records": records, "graph": graph})}

    @classmethod
    def from_export(cls, raw: dict) -> "ProvenanceLedger":
        if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
            raise IntegrityError("invalid provenance export")
        expected = raw.get("ledger_hash")
        material = {"records": raw["records"], "graph": raw.get("graph", {})}
        if expected != digest_json(material):
            raise IntegrityError("provenance ledger hash mismatch")
        ledger = cls()
        for item in raw["records"]:
            record = ProvenanceRecord(str(item["record_id"]), str(item["artifact_type"]),
                                      str(item["source_name"]), str(item["content_hash"]),
                                      tuple(item.get("parent_hashes", [])), dict(item.get("metadata", {})))
            ledger._insert(record)
        ledger.validate_graph()
        return ledger

    def validate_graph(self) -> dict:
        errors: list[dict] = []
        content_hashes = {record.content_hash for record in self.records.values()}
        for record in self.records.values():
            if record.content_hash in record.parent_hashes:
                errors.append({"type": "self_parent", "record_id": record.record_id})
            duplicates = len(record.parent_hashes) - len(set(record.parent_hashes))
            if duplicates:
                errors.append({"type": "duplicate_parent", "record_id": record.record_id, "count": duplicates})
        cycles = self._cycles()
        errors.extend({"type": "cycle", "record_ids": cycle} for cycle in cycles)
        external = sorted({parent for record in self.records.values() for parent in record.parent_hashes
                           if parent not in content_hashes})
        return {"ok": not errors, "errors": errors, "external_parent_hashes": external,
                "record_count": len(self.records), "ledger_hash": self.export()["ledger_hash"]}

    def _add(self, artifact_type: str, source_name: str, content_hash: str,
             parent_hashes: Iterable[str], metadata: dict) -> ProvenanceRecord:
        parents = tuple(sorted(set(parent_hashes)))
        identity = {"artifact_type": artifact_type, "source_name": source_name,
                    "content_hash": content_hash, "parent_hashes": list(parents), "metadata": metadata}
        record = ProvenanceRecord("prov_" + digest_json(identity)[:24], artifact_type, source_name,
                                  content_hash, parents, metadata)
        self._insert(record)
        return record

    def _insert(self, record: ProvenanceRecord) -> None:
        existing = self.records.get(record.record_id)
        if existing and existing != record:
            raise IntegrityError("provenance record identifier conflict", details={"record_id": record.record_id})
        same_hash = [item for item in self.records.values() if item.content_hash == record.content_hash]
        if same_hash and any(item.artifact_type != record.artifact_type for item in same_hash):
            raise IntegrityError("content hash reused across incompatible artifact types",
                                 details={"record_id": record.record_id, "content_hash": record.content_hash})
        self.records[record.record_id] = record

    def _require(self, record_id: str) -> ProvenanceRecord:
        try:
            return self.records[record_id]
        except KeyError as exc:
            raise IntegrityError("provenance record not found", details={"record_id": record_id}) from exc

    def _record_id_by_content_hash(self, content_hash: str) -> str | None:
        matches = sorted(record.record_id for record in self.records.values() if record.content_hash == content_hash)
        return matches[0] if matches else None

    def _cycles(self) -> list[list[str]]:
        by_hash = {record.content_hash: record.record_id for record in self.records.values()}
        adjacency = {record.record_id: [by_hash[parent] for parent in record.parent_hashes if parent in by_hash]
                     for record in self.records.values()}
        visiting: set[str] = set()
        visited: set[str] = set()
        cycles: list[list[str]] = []

        def walk(node: str, path: list[str]) -> None:
            if node in visiting:
                start = path.index(node)
                cycles.append(path[start:] + [node])
                return
            if node in visited:
                return
            visiting.add(node)
            path.append(node)
            for parent in adjacency[node]:
                walk(parent, path)
            path.pop()
            visiting.remove(node)
            visited.add(node)

        for node in sorted(adjacency):
            walk(node, [])
        return cycles


def fingerprint_package(root: str | Path, *, include_suffixes: tuple[str, ...] = (".py", ".json", ".txt")) -> dict:
    root_path = Path(root)
    files: list[dict] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix not in include_suffixes:
            continue
        relative = path.relative_to(root_path).as_posix()
        content = path.read_bytes()
        files.append({"path": relative, "sha256": digest_bytes(content), "bytes": len(content)})
    return {"root": root_path.name, "files": files, "package_hash": digest_json(files)}
