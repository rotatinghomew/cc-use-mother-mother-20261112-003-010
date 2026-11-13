"""SQLite evidence journal preserving raw observations and derived artifacts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .canonical import canonical_json, digest_json, format_time, parse_time
from .errors import IntegrityError
from .models import Candidate, Checkpoint, Event, PolicySpec


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    metadata_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_events (
    event_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    source_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    event_time TEXT NOT NULL,
    observed_time TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    correlation_id TEXT,
    evidence_hash TEXT NOT NULL,
    event_json TEXT NOT NULL,
    first_seen_order INTEGER NOT NULL,
    UNIQUE(incident_id, source_id, sequence_no)
);
CREATE INDEX IF NOT EXISTS idx_events_incident_time
ON evidence_events(incident_id, event_time, observed_time, event_id);
CREATE INDEX IF NOT EXISTS idx_events_incident_agent
ON evidence_events(incident_id, agent_id, sequence_no);
CREATE TABLE IF NOT EXISTS duplicate_observations (
    duplicate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    incident_id TEXT NOT NULL,
    observed_time TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    exact_duplicate INTEGER NOT NULL,
    raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rejected_observations (
    rejection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT,
    event_id TEXT,
    reason TEXT NOT NULL,
    raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    event_time TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_incident_time
ON checkpoints(incident_id, event_time);
CREATE TABLE IF NOT EXISTS interpolation_candidates (
    candidate_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    gap_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    candidate_hash TEXT NOT NULL,
    candidate_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_specs (
    experiment_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    created_at TEXT NOT NULL,
    seed INTEGER NOT NULL,
    spec_hash TEXT NOT NULL,
    spec_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiment_specs(experiment_id),
    scenario_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
    run_id TEXT,
    event_count INTEGER NOT NULL,
    last_event_id TEXT,
    state_hash TEXT NOT NULL,
    snapshot_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_manifest (
    artifact_name TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def register_incident(self, incident_id: str, title: str, created_at: str, metadata: dict) -> None:
        parse_time(created_at)
        encoded = canonical_json(metadata)
        hashed = digest_json(metadata)
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT title, created_at, metadata_hash FROM incidents WHERE incident_id=?",
                (incident_id,),
            ).fetchone()
            if existing:
                if (existing["title"], existing["created_at"], existing["metadata_hash"]) != (title, created_at, hashed):
                    raise IntegrityError("incident contract conflicts with existing record", details={"incident_id": incident_id})
                return
            conn.execute(
                "INSERT INTO incidents VALUES(?,?,?,?,?)",
                (incident_id, title, created_at, encoded, hashed),
            )

    def incident(self, incident_id: str) -> dict:
        row = self.connection.execute("SELECT * FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()
        if not row:
            raise IntegrityError("incident not found", details={"incident_id": incident_id})
        return {
            "incident_id": row["incident_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata_json"]),
            "metadata_hash": row["metadata_hash"],
        }

    def _next_order(self, incident_id: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(first_seen_order), 0) AS n FROM evidence_events WHERE incident_id=?",
            (incident_id,),
        ).fetchone()
        return int(row["n"]) + 1

    def ingest_events(self, events: Iterable[Event]) -> dict:
        accepted = 0
        duplicates = 0
        conflicts = 0
        ordered = list(events)
        with self.transaction() as conn:
            next_order: dict[str, int] = {}
            for event in ordered:
                if event.incident_id not in next_order:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(first_seen_order),0) AS n FROM evidence_events WHERE incident_id=?",
                        (event.incident_id,),
                    ).fetchone()
                    next_order[event.incident_id] = int(row["n"]) + 1
                existing = conn.execute(
                    "SELECT evidence_hash FROM evidence_events WHERE event_id=?",
                    (event.event_id,),
                ).fetchone()
                if existing:
                    exact = existing["evidence_hash"] == event.evidence_hash
                    conn.execute(
                        "INSERT INTO duplicate_observations(event_id,incident_id,observed_time,evidence_hash,exact_duplicate,raw_json) VALUES(?,?,?,?,?,?)",
                        (event.event_id, event.incident_id, event.observed_time, event.evidence_hash, int(exact), canonical_json(event.to_dict())),
                    )
                    if exact:
                        duplicates += 1
                        continue
                    conflicts += 1
                    conn.execute(
                        "INSERT INTO rejected_observations(incident_id,event_id,reason,raw_json) VALUES(?,?,?,?)",
                        (event.incident_id, event.event_id, "event_id_hash_conflict", canonical_json(event.to_dict())),
                    )
                    continue
                sequence_owner = conn.execute(
                    "SELECT event_id,evidence_hash FROM evidence_events WHERE incident_id=? AND source_id=? AND sequence_no=?",
                    (event.incident_id, event.source_id, event.sequence),
                ).fetchone()
                if sequence_owner:
                    conflicts += 1
                    conn.execute(
                        "INSERT INTO rejected_observations(incident_id,event_id,reason,raw_json) VALUES(?,?,?,?)",
                        (event.incident_id, event.event_id, "source_sequence_conflict", canonical_json(event.to_dict())),
                    )
                    continue
                order = next_order[event.incident_id]
                next_order[event.incident_id] += 1
                conn.execute(
                    "INSERT INTO evidence_events(event_id,incident_id,source_id,agent_id,kind,event_time,observed_time,sequence_no,correlation_id,evidence_hash,event_json,first_seen_order) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (event.event_id, event.incident_id, event.source_id, event.agent_id, event.kind,
                     event.event_time, event.observed_time, event.sequence, event.correlation_id,
                     event.evidence_hash, canonical_json(event.to_dict()), order),
                )
                accepted += 1
        return {"accepted": accepted, "duplicates": duplicates, "conflicts": conflicts, "received": len(ordered)}

    def reject_raw(self, raw: dict, reason: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO rejected_observations(incident_id,event_id,reason,raw_json) VALUES(?,?,?,?)",
                (raw.get("incident_id"), raw.get("event_id"), reason, canonical_json(raw)),
            )

    def load_events(self, incident_id: str, *, order: str = "event") -> list[Event]:
        if order == "event":
            clause = "event_time, observed_time, source_id, sequence_no, event_id"
        elif order == "observed":
            clause = "observed_time, first_seen_order, event_id"
        elif order == "arrival":
            clause = "first_seen_order, event_id"
        else:
            raise IntegrityError("unknown journal order", details={"order": order})
        rows = self.connection.execute(
            f"SELECT event_json FROM evidence_events WHERE incident_id=? ORDER BY {clause}",
            (incident_id,),
        ).fetchall()
        return [Event.from_dict(json.loads(row["event_json"])) for row in rows]

    def load_event(self, event_id: str) -> Event | None:
        row = self.connection.execute("SELECT event_json FROM evidence_events WHERE event_id=?", (event_id,)).fetchone()
        return Event.from_dict(json.loads(row["event_json"])) if row else None

    def counts(self, incident_id: str) -> dict:
        query = lambda table, where="incident_id=?": int(self.connection.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", (incident_id,)
        ).fetchone()["n"])
        return {
            "events": query("evidence_events"),
            "duplicates": query("duplicate_observations"),
            "rejections": query("rejected_observations"),
            "checkpoints": query("checkpoints"),
            "candidates": query("interpolation_candidates"),
            "snapshots": query("snapshots"),
        }

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        encoded = canonical_json(checkpoint.to_dict())
        with self.transaction() as conn:
            existing = conn.execute("SELECT state_hash FROM checkpoints WHERE checkpoint_id=?", (checkpoint.checkpoint_id,)).fetchone()
            if existing and existing["state_hash"] != checkpoint.state_hash:
                raise IntegrityError("checkpoint identifier reused with different state", details={"checkpoint_id": checkpoint.checkpoint_id})
            conn.execute(
                "INSERT OR IGNORE INTO checkpoints(checkpoint_id,incident_id,event_time,state_hash,checkpoint_json) VALUES(?,?,?,?,?)",
                (checkpoint.checkpoint_id, checkpoint.incident_id, checkpoint.event_time, checkpoint.state_hash, encoded),
            )

    def load_checkpoints(self, incident_id: str) -> list[Checkpoint]:
        rows = self.connection.execute(
            "SELECT checkpoint_json FROM checkpoints WHERE incident_id=? ORDER BY event_time,checkpoint_id", (incident_id,)
        ).fetchall()
        return [Checkpoint.from_dict(json.loads(row["checkpoint_json"])) for row in rows]

    def latest_checkpoint(self, incident_id: str, at_or_before: str | None = None) -> Checkpoint | None:
        if at_or_before:
            row = self.connection.execute(
                "SELECT checkpoint_json FROM checkpoints WHERE incident_id=? AND event_time<=? ORDER BY event_time DESC,checkpoint_id DESC LIMIT 1",
                (incident_id, at_or_before),
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT checkpoint_json FROM checkpoints WHERE incident_id=? ORDER BY event_time DESC,checkpoint_id DESC LIMIT 1",
                (incident_id,),
            ).fetchone()
        return Checkpoint.from_dict(json.loads(row["checkpoint_json"])) if row else None

    def save_candidates(self, incident_id: str, candidates: Sequence[Candidate]) -> None:
        with self.transaction() as conn:
            for candidate in candidates:
                value = candidate.to_dict()
                hashed = digest_json(value)
                existing = conn.execute("SELECT candidate_hash FROM interpolation_candidates WHERE candidate_id=?", (candidate.candidate_id,)).fetchone()
                if existing and existing["candidate_hash"] != hashed:
                    raise IntegrityError("candidate identifier conflict", details={"candidate_id": candidate.candidate_id})
                conn.execute(
                    "INSERT OR IGNORE INTO interpolation_candidates VALUES(?,?,?,?,?,?)",
                    (candidate.candidate_id, incident_id, candidate.gap_id, candidate.confidence, hashed, canonical_json(value)),
                )

    def load_candidates(self, incident_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT candidate_json FROM interpolation_candidates WHERE incident_id=? ORDER BY confidence DESC,candidate_id",
            (incident_id,),
        ).fetchall()
        return [json.loads(row["candidate_json"]) for row in rows]

    def save_experiment(self, experiment_id: str, incident_id: str, created_at: str, seed: int, spec: dict) -> str:
        hashed = digest_json(spec)
        with self.transaction() as conn:
            existing = conn.execute("SELECT spec_hash FROM experiment_specs WHERE experiment_id=?", (experiment_id,)).fetchone()
            if existing and existing["spec_hash"] != hashed:
                raise IntegrityError("experiment identifier conflict", details={"experiment_id": experiment_id})
            conn.execute(
                "INSERT OR IGNORE INTO experiment_specs VALUES(?,?,?,?,?,?)",
                (experiment_id, incident_id, created_at, seed, hashed, canonical_json(spec)),
            )
        return hashed

    def save_run(self, run_id: str, experiment_id: str, scenario_id: str, started_at: str, result: dict) -> str:
        hashed = digest_json(result)
        with self.transaction() as conn:
            existing = conn.execute("SELECT result_hash FROM experiment_runs WHERE run_id=?", (run_id,)).fetchone()
            if existing and existing["result_hash"] != hashed:
                raise IntegrityError("experiment run identifier conflict", details={"run_id": run_id})
            conn.execute(
                "INSERT OR IGNORE INTO experiment_runs VALUES(?,?,?,?,?,?)",
                (run_id, experiment_id, scenario_id, started_at, hashed, canonical_json(result)),
            )
        return hashed

    def load_runs(self, experiment_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT result_json FROM experiment_runs WHERE experiment_id=? ORDER BY scenario_id,run_id", (experiment_id,)
        ).fetchall()
        return [json.loads(row["result_json"]) for row in rows]

    def save_snapshot(self, snapshot_id: str, incident_id: str, run_id: str | None, event_count: int,
                      last_event_id: str | None, state: dict) -> str:
        hashed = digest_json(state)
        with self.transaction() as conn:
            existing = conn.execute("SELECT state_hash FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
            if existing and existing["state_hash"] != hashed:
                raise IntegrityError("snapshot identifier conflict", details={"snapshot_id": snapshot_id})
            conn.execute(
                "INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?,?,?)",
                (snapshot_id, incident_id, run_id, event_count, last_event_id, hashed, canonical_json(state)),
            )
        return hashed

    def load_snapshot(self, snapshot_id: str) -> dict | None:
        row = self.connection.execute("SELECT snapshot_json,state_hash FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if not row:
            return None
        value = json.loads(row["snapshot_json"])
        if digest_json(value) != row["state_hash"]:
            raise IntegrityError("stored snapshot hash mismatch", details={"snapshot_id": snapshot_id})
        return value

    def record_artifact(self, name: str, relative_path: str, content_hash: str, size: int, created_at: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifact_manifest VALUES(?,?,?,?,?)",
                (name, relative_path, content_hash, size, created_at),
            )

    def artifact_manifest(self) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM artifact_manifest ORDER BY artifact_name").fetchall()
        return [dict(row) for row in rows]

    def verify_integrity(self, incident_id: str) -> dict:
        errors: list[dict] = []
        for event in self.load_events(incident_id, order="arrival"):
            if event.calculate_evidence_hash() != event.evidence_hash:
                errors.append({"type": "event_hash", "event_id": event.event_id})
        for checkpoint in self.load_checkpoints(incident_id):
            if digest_json(checkpoint.state) != checkpoint.state_hash:
                errors.append({"type": "checkpoint_hash", "checkpoint_id": checkpoint.checkpoint_id})
        for row in self.connection.execute("SELECT candidate_id,candidate_hash,candidate_json FROM interpolation_candidates WHERE incident_id=?", (incident_id,)):
            if digest_json(json.loads(row["candidate_json"])) != row["candidate_hash"]:
                errors.append({"type": "candidate_hash", "candidate_id": row["candidate_id"]})
        return {"ok": not errors, "errors": errors, "counts": self.counts(incident_id)}
