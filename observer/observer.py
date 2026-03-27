import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

from cognition.runtime import ControlSignal
from graph.brain import BrainMode


@dataclass
class AgendaItem:
    text: str
    item_type: str = "question"
    source_step: int = 0
    dream_cycle: int = 0
    count: int = 1
    resolved: bool = False
    resolution_grade: str = ""
    priority: float = 0.5
    incubation_age: int = 0
    node_id: str = ""
    partial_leads: list = field(default_factory=list)
    answer_node_id: str = ""


@dataclass
class MissionAdvance:
    node_id: str
    explanation: str
    strength: float
    cycle: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return self.__dict__


@dataclass
class EmergenceSignal:
    signal: str
    type: str
    cycle: int
    node_ids: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return self.__dict__


class Observer:
    def __init__(self, memory):
        self.memory = memory
        self.agenda: List[AgendaItem] = []
        self.emergence_feed: List[EmergenceSignal] = []
        self.mission_advances: List[MissionAdvance] = []
        self.cycle_count = 0
        self.runtime_snapshots: List[Dict] = []
        self.mode_history: List[Dict] = []
        self.decision_records: List[Dict] = []

        # objective weights
        self.weights = {
            "novelty": 1.0,
            "tension": 1.2,
            "uncertainty": 0.9,
            "resolution": 0.8,
            "cost": 0.2,
        }

    def add_to_agenda(self, text: str, item_type: str = "question", cycle: int = 0, step: int = 0, node_id: str = "") -> AgendaItem:
        for item in self.agenda:
            if item.text.strip().lower() == text.strip().lower():
                item.count += 1
                item.priority = min(1.0, item.priority + 0.1)
                return item
        item = AgendaItem(text=text, item_type=item_type, source_step=step, dream_cycle=cycle, node_id=node_id)
        self.agenda.append(item)
        return item

    def get_prioritized_questions(self, n: int = 10) -> list:
        unresolved = [i for i in self.agenda if not i.resolved]
        return sorted(unresolved, key=lambda i: i.priority, reverse=True)[:n]

    def record_answer(self, question_text: str, answer_node_id: str, explanation: str, grade: str = "strong"):
        for item in self.agenda:
            if item.text != question_text:
                continue
            item.answer_node_id = answer_node_id
            item.resolution_grade = grade
            if grade == "strong":
                item.resolved = True
            elif grade == "partial":
                item.partial_leads.append(answer_node_id)
                item.priority = min(1.0, item.priority + 0.05)
            if hasattr(self.memory, "brain"):
                self.memory.brain.log_event("question_resolution", {"question": question_text, "grade": grade, "node": answer_node_id, "explanation": explanation})
            break

    def record_mission_advance(self, node_id: str, explanation: str, strength: float):
        self.mission_advances.append(MissionAdvance(node_id=node_id, explanation=explanation, strength=float(strength), cycle=self.cycle_count))

    def observe_runtime(self, step):
        self.cycle_count += 1
        snap = {
            "step_id": step.step_id,
            "novelty": step.novelty,
            "tension": step.tension,
            "entropy": step.entropy,
            "uncertainty_mass": step.uncertainty_mass,
            "contradiction_density": step.contradiction_density,
        }
        self.runtime_snapshots.append(snap)
        self.runtime_snapshots = self.runtime_snapshots[-5000:]

    def _score_action(self, action: str, context: Dict[str, float]) -> float:
        novelty = context.get("novelty", 0.0)
        tension = context.get("tension", 0.0)
        unc = context.get("uncertainty_mass", 0.0)
        unresolved_pressure = context.get("unresolved_pressure", 0.0)

        if action == "wander":
            benefit = self.weights["novelty"] * novelty + 0.3
            cost = self.weights["cost"] * 0.4
        elif action == "focus":
            benefit = self.weights["resolution"] * unresolved_pressure + 0.2
            cost = self.weights["cost"] * 0.5
        elif action == "consolidate":
            benefit = self.weights["tension"] * tension + self.weights["uncertainty"] * unc
            cost = self.weights["cost"] * 0.6
        elif action == "acquire":
            benefit = self.weights["uncertainty"] * unc + self.weights["resolution"] * unresolved_pressure
            cost = self.weights["cost"] * 0.7
        elif action == "test":
            benefit = self.weights["tension"] * tension + self.weights["resolution"] * unresolved_pressure
            cost = self.weights["cost"] * 0.8
        else:
            benefit, cost = 0.0, 1.0
        return benefit - cost

    def decide(self) -> ControlSignal:
        recent = self.runtime_snapshots[-5:]
        if not recent:
            control = ControlSignal(mode=BrainMode.WANDERING.value, temperature=0.9, action_set=["wander"], budgets={"steps": 10}, gains={"score": 0.0})
            self._record_decision({}, [], control)
            return control

        context = {
            "novelty": sum(r["novelty"] for r in recent) / len(recent),
            "tension": sum(r["tension"] for r in recent) / len(recent),
            "uncertainty_mass": sum(r["uncertainty_mass"] for r in recent) / len(recent),
            "entropy": sum(r["entropy"] for r in recent) / len(recent),
            "unresolved_pressure": sum(i.priority for i in self.agenda if not i.resolved),
        }

        candidates = ["wander", "focus", "consolidate", "acquire", "test"]
        scored = [{"action": a, "score": self._score_action(a, context)} for a in candidates]
        best = max(scored, key=lambda x: x["score"])

        action = best["action"]
        mode_map = {
            "wander": BrainMode.WANDERING.value,
            "focus": BrainMode.FOCUSED.value,
            "consolidate": BrainMode.CONSOLIDATION.value,
            "acquire": BrainMode.ACQUISITION.value,
            "test": BrainMode.TESTING.value,
        }
        temp_map = {"wander": 0.9, "focus": 0.45, "consolidate": 0.25, "acquire": 0.55, "test": 0.4}

        control = ControlSignal(
            mode=mode_map[action],
            temperature=temp_map[action],
            action_set=[action],
            budgets={"steps": 12, "external": 2, "tests": 1 if action == "test" else 0},
            gains={"score": best["score"]},
        )
        self._record_decision(context, scored, control)
        self.mode_history.append({"time": time.time(), "mode": control.mode})
        return control

    def _record_decision(self, context: Dict, candidates: List[Dict], control: ControlSignal):
        rec = {
            "time": time.time(),
            "step": self.cycle_count,
            "inputs": context,
            "candidates": candidates,
            "chosen": {
                "mode": control.mode,
                "temperature": control.temperature,
                "action_set": control.action_set,
                "budgets": control.budgets,
            },
        }
        self.decision_records.append(rec)
        self.decision_records = self.decision_records[-5000:]
        if hasattr(self.memory, "brain"):
            self.memory.brain.log_event("control_decision", rec)

    def increment_incubation(self):
        for item in self.agenda:
            if not item.resolved:
                item.incubation_age += 1

    def observe(self, log):
        for q in getattr(log, "questions", []):
            self.add_to_agenda(q, cycle=self.cycle_count)
        for a in getattr(log, "mission_advances", []):
            self.record_mission_advance(a.get("node", ""), a.get("explanation", ""), a.get("strength", 0.0))
        self.increment_incubation()

    def save(self, path: str = "data/observer.json"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "cycle_count": self.cycle_count,
                    "agenda": [a.__dict__ for a in self.agenda],
                    "emergences": [e.to_dict() for e in self.emergence_feed],
                    "mission_advances": [m.to_dict() for m in self.mission_advances],
                    "runtime_snapshots": self.runtime_snapshots,
                    "mode_history": self.mode_history,
                    "decision_records": self.decision_records,
                },
                f,
                indent=2,
            )

    def load(self, path: str = "data/observer.json"):
        with open(path, "r") as f:
            data = json.load(f)
        self.cycle_count = data.get("cycle_count", 0)
        self.agenda = [AgendaItem(**i) for i in data.get("agenda", [])]
        self.emergence_feed = [EmergenceSignal(**i) for i in data.get("emergences", [])]
        self.mission_advances = [MissionAdvance(**i) for i in data.get("mission_advances", [])]
        self.runtime_snapshots = data.get("runtime_snapshots", [])
        self.mode_history = data.get("mode_history", [])
        self.decision_records = data.get("decision_records", [])
