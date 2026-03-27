import json
import os
import time
from dataclasses import dataclass, field

from graph.brain import Brain

NOTEBOOK_PATH = "data/notebook.json"

ENTRY_MORNING = "morning"
ENTRY_FIELD_NOTES = "field_notes"
ENTRY_EVENING = "evening"
ENTRY_HYPOTHESIS = "hypothesis"
ENTRY_BREAKTHROUGH = "breakthrough"
ENTRY_DECISION = "decision"
ENTRY_TEST = "test"
ENTRY_EVIDENCE = "evidence"


@dataclass
class NotebookEntry:
    entry_type: str
    content: str
    cycle: int
    timestamp: float = field(default_factory=time.time)
    tags: list = field(default_factory=list)

    def to_dict(self):
        return self.__dict__


class Notebook:
    def __init__(self, brain: Brain, observer=None, scientist_name: str = "THE SCIENTIST"):
        self.brain = brain
        self.observer = observer
        self.name = scientist_name
        self.entries: list[NotebookEntry] = []
        self.running_hypothesis = ""
        self._last_event_idx = 0
        self._load()

    def _add_entry(self, entry_type: str, content: str, cycle: int, tags: list = None):
        e = NotebookEntry(entry_type=entry_type, content=content, cycle=cycle, tags=tags or [])
        self.entries.append(e)
        self._save()
        return e

    def write_morning_entry(self, dream_log, cycle: int):
        content = f"Morning state: mode={self.brain.get_mode()}, steps={len(dream_log.steps)}, insights={len(dream_log.insights)}, mission_advances={len(dream_log.mission_advances)}."
        self._add_entry(ENTRY_MORNING, content, cycle, tags=["dream_state"])
        return content

    def write_field_notes(self, research_log, cycle: int):
        resolved = sum(1 for e in research_log.entries if e.resolved in ["partial", "strong"])
        content = f"Field notes: researched={len(research_log.entries)}, resolved={resolved}."
        self._add_entry(ENTRY_FIELD_NOTES, content, cycle, tags=["research", f"resolved:{resolved}"])
        return content

    def write_evening_entry(self, consolidation_report, cycle: int):
        content = f"Evening consolidation: delta_objective={getattr(consolidation_report, 'delta_objective', 0.0):.6f}, merges={consolidation_report.merges}, abstractions={consolidation_report.abstractions}."
        self._add_entry(ENTRY_EVENING, content, cycle, tags=["consolidation"])
        return content

    def update_running_hypothesis(self, cycle: int):
        mission = self.brain.get_mission()
        m = mission["question"] if mission else "No mission"
        unresolved = len([a for a in (self.observer.agenda if self.observer else []) if not a.resolved])
        self.running_hypothesis = f"Working hypothesis around '{m}': unresolved agenda={unresolved}, uncertainty_mass={self.brain.stats().get('uncertainty_mass', 0.0):.3f}."
        self._add_entry(ENTRY_HYPOTHESIS, self.running_hypothesis, cycle, tags=["working_hypothesis"])
        return self.running_hypothesis

    def write_breakthrough(self, detail: str, cycle: int):
        return self._add_entry(ENTRY_BREAKTHROUGH, f"Breakthrough: {detail}", cycle, tags=["breakthrough"]).content

    def sync_from_structured_events(self, cycle: int = 0):
        events = self.brain.event_log[self._last_event_idx :]
        for e in events:
            typ = e.get("type")
            payload = e.get("payload", {})
            if typ == "control_decision":
                self._add_entry(ENTRY_DECISION, f"Decision: mode={payload.get('chosen', {}).get('mode')} actions={payload.get('chosen', {}).get('action_set')}", cycle, tags=["control"])
            elif typ == "evidence_update":
                self._add_entry(ENTRY_EVIDENCE, f"Evidence update: nodes={len(payload.get('affected_nodes', []))}, reliability={payload.get('reliability_weight')}", cycle, tags=["evidence"])
            elif typ == "test_outcome":
                self._add_entry(ENTRY_TEST, f"Test outcome: verdict={payload.get('verdict')} confidence={payload.get('confidence')}", cycle, tags=["test"])
        self._last_event_idx = len(self.brain.event_log)

    def get_entries_by_type(self, entry_type: str):
        return [e for e in self.entries if e.entry_type == entry_type]

    def get_recent_entries(self, n: int = 10):
        return sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:n]

    def get_all_for_display(self):
        return [e.to_dict() for e in sorted(self.entries, key=lambda x: x.timestamp, reverse=True)]

    def _save(self):
        os.makedirs(os.path.dirname(NOTEBOOK_PATH) or ".", exist_ok=True)
        with open(NOTEBOOK_PATH, "w") as f:
            json.dump({"entries": [e.to_dict() for e in self.entries], "running_hypothesis": self.running_hypothesis, "scientist_name": self.name, "last_event_idx": self._last_event_idx}, f, indent=2)

    def _load(self):
        try:
            with open(NOTEBOOK_PATH, "r") as f:
                raw = json.load(f)
            self.entries = [NotebookEntry(**e) for e in raw.get("entries", [])]
            self.running_hypothesis = raw.get("running_hypothesis", "")
            self.name = raw.get("scientist_name", self.name)
            self._last_event_idx = raw.get("last_event_idx", 0)
        except FileNotFoundError:
            self.entries = []

    def save(self):
        self._save()
