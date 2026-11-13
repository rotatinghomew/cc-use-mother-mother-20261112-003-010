"""Read-only audit queries over journaled evidence and experiments."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from .canonical import digest_json, micros
from .journal import Journal
from .models import Event


class AuditQueries:
    def __init__(self, journal: Journal) -> None:
        self.journal = journal

    def evidence_inventory(self, incident_id: str) -> dict:
        events = self.journal.load_events(incident_id, order="event")
        kinds = Counter(event.kind for event in events)
        sources = Counter(event.source_id for event in events)
        agents = Counter(event.agent_id for event in events)
        applications = Counter(event.application for event in events if event.application)
        tools = Counter(event.tool for event in events if event.tool)
        return {
            "incident_id": incident_id, "event_count": len(events),
            "event_time_start": events[0].event_time if events else None,
            "event_time_end": events[-1].event_time if events else None,
            "kinds": dict(sorted(kinds.items())), "sources": dict(sorted(sources.items())),
            "agents": dict(sorted(agents.items())), "applications": dict(sorted(applications.items())),
            "tools": dict(sorted(tools.items())), "counts": self.journal.counts(incident_id),
        }

    def duplicate_inventory(self, incident_id: str) -> dict:
        rows = self.journal.connection.execute(
            "SELECT event_id,exact_duplicate,COUNT(*) AS n,MIN(observed_time) AS first_observed,MAX(observed_time) AS last_observed "
            "FROM duplicate_observations WHERE incident_id=? GROUP BY event_id,exact_duplicate ORDER BY event_id",
            (incident_id,),
        ).fetchall()
        return {"incident_id": incident_id, "groups": [dict(row) for row in rows],
                "exact": sum(row["n"] for row in rows if row["exact_duplicate"]),
                "conflicting": sum(row["n"] for row in rows if not row["exact_duplicate"])}

    def rejection_inventory(self, incident_id: str) -> dict:
        rows = self.journal.connection.execute(
            "SELECT reason,COUNT(*) AS n FROM rejected_observations WHERE incident_id=? GROUP BY reason ORDER BY reason",
            (incident_id,),
        ).fetchall()
        samples = self.journal.connection.execute(
            "SELECT event_id,reason,raw_json FROM rejected_observations WHERE incident_id=? ORDER BY rejection_id LIMIT 20",
            (incident_id,),
        ).fetchall()
        return {"incident_id": incident_id, "reasons": [dict(row) for row in rows],
                "samples": [{"event_id": row["event_id"], "reason": row["reason"],
                             "raw_hash": digest_json(json.loads(row["raw_json"]))} for row in samples]}

    def causal_neighborhood(self, incident_id: str, event_id: str, depth: int = 2) -> dict:
        events = self.journal.load_events(incident_id, order="event")
        index = {event.event_id: event for event in events}
        children: dict[str, set[str]] = defaultdict(set)
        for event in events:
            for parent in event.parent_ids:
                children[parent].add(event.event_id)
        if event_id not in index:
            return {"incident_id": incident_id, "event_id": event_id, "found": False, "nodes": []}
        frontier = {event_id}
        visited: set[str] = set()
        nodes: list[dict] = []
        for distance in range(depth + 1):
            next_frontier: set[str] = set()
            for current in sorted(frontier):
                if current in visited or current not in index:
                    continue
                visited.add(current)
                event = index[current]
                nodes.append({"distance": distance, "direction": "center" if distance == 0 else "related",
                              "event": event.to_dict()})
                next_frontier.update(parent for parent in event.parent_ids if parent in index)
                next_frontier.update(children[current])
            frontier = next_frontier - visited
        return {"incident_id": incident_id, "event_id": event_id, "found": True, "nodes": nodes}

    def application_exposure(self, incident_id: str) -> list[dict]:
        groups: dict[str, list[Event]] = defaultdict(list)
        for event in self.journal.load_events(incident_id, order="event"):
            if event.application:
                groups[event.application].append(event)
        result: list[dict] = []
        for application, events in sorted(groups.items()):
            result.append({
                "application": application, "event_count": len(events),
                "agents": sorted({event.agent_id for event in events}),
                "tools": sorted({event.tool for event in events if event.tool}),
                "permissions": sorted({permission for event in events for permission in event.permissions}),
                "loss_proxy": sum(event.loss_proxy for event in events),
                "mutations": sum(1 for event in events if event.is_mutation),
                "start_time": min(events, key=lambda item: micros(item.event_time)).event_time,
                "end_time": max(events, key=lambda item: micros(item.event_time)).event_time,
            })
        return result

    def experiment_inventory(self, incident_id: str) -> list[dict]:
        rows = self.journal.connection.execute(
            "SELECT experiment_id,created_at,seed,spec_hash,spec_json FROM experiment_specs WHERE incident_id=? ORDER BY created_at,experiment_id",
            (incident_id,),
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            spec = json.loads(row["spec_json"])
            run_rows = self.journal.connection.execute(
                "SELECT run_id,scenario_id,result_hash FROM experiment_runs WHERE experiment_id=? ORDER BY scenario_id,run_id",
                (row["experiment_id"],),
            ).fetchall()
            result.append({"experiment_id": row["experiment_id"], "created_at": row["created_at"],
                           "seed": row["seed"], "spec_hash": row["spec_hash"],
                           "spec_hash_valid": digest_json(spec) == row["spec_hash"],
                           "runs": [dict(run) for run in run_rows]})
        return result

    def snapshot_inventory(self, incident_id: str) -> list[dict]:
        rows = self.journal.connection.execute(
            "SELECT snapshot_id,run_id,event_count,last_event_id,state_hash,snapshot_json FROM snapshots "
            "WHERE incident_id=? ORDER BY snapshot_id", (incident_id,),
        ).fetchall()
        return [{"snapshot_id": row["snapshot_id"], "run_id": row["run_id"],
                 "event_count": row["event_count"], "last_event_id": row["last_event_id"],
                 "state_hash": row["state_hash"],
                 "state_hash_valid": digest_json(json.loads(row["snapshot_json"])) == row["state_hash"]}
                for row in rows]
