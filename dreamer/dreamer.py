import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum

from cognition.runtime import CognitiveRuntime, ControlSignal
from cortex.interpreter import ReflectiveInterpreter
from graph.brain import Brain, BrainMode
from memory.store import NetworkXMemoryStoreAdapter


class DreamMode(str, Enum):
    WANDERING = "wandering"
    PRESSURE = "pressure"
    SEEDED = "seeded"


@dataclass
class DreamStep:
    step: int
    from_id: str
    to_id: str
    edge_type: str
    edge_narration: str
    narration: str
    question: str = ""
    is_insight: bool = False
    insight_depth: str = ""
    new_edge: bool = False
    answer_match: str = "none"
    answer_detail: str = ""
    depth_triggered: bool = False
    mission_advance: bool = False


@dataclass
class DreamLog:
    mode: str
    brain_mode: str = "wandering"
    started_at: float = field(default_factory=time.time)
    steps: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    insights: list = field(default_factory=list)
    answers: list = field(default_factory=list)
    mission_advances: list = field(default_factory=list)
    summary: str = ""

    def to_dict(self):
        return {
            "mode": self.mode,
            "brain_mode": self.brain_mode,
            "started_at": self.started_at,
            "steps": [s.__dict__ for s in self.steps],
            "questions": self.questions,
            "insights": self.insights,
            "answers": self.answers,
            "mission_advances": self.mission_advances,
            "summary": self.summary,
        }


class Dreamer:
    def __init__(self, brain: Brain, research_agenda=None):
        self.brain = brain
        self.memory = NetworkXMemoryStoreAdapter(brain)
        self.research_agenda = research_agenda
        self.runtime = CognitiveRuntime(self.memory, observer=research_agenda, seed=brain.global_state.rng_seed)
        self.interpreter = ReflectiveInterpreter(enabled=True)
        self.rand = random.Random(7)

    def _pick_jump(self):
        nodes = [(nid, d) for nid, d in self.brain.all_nodes() if d.get("node_type") != "mission"]
        if not nodes:
            return None
        weights = [max(0.01, (d.get("state") or {}).get("activation", d.get("activation", 0.0)) + (d.get("state") or {}).get("attention", d.get("attention", 0.0))) for _, d in nodes]
        total = sum(weights)
        r = self.rand.random() * total
        acc = 0.0
        for (nid, _), w in zip(nodes, weights):
            acc += w
            if acc >= r:
                return nid
        return nodes[-1][0]

    def dream(self, mode=DreamMode.WANDERING, steps=20, temperature=0.7, seed_id=None, run_nrem=True, log_path="logs/dream_latest.json"):
        control_mode = BrainMode.WANDERING.value if mode == DreamMode.WANDERING else BrainMode.FOCUSED.value
        control = ControlSignal(mode=control_mode, temperature=temperature, action_set=["wander" if mode == DreamMode.WANDERING else "focus"], budgets={"steps": 1}, gains={})

        log = DreamLog(mode=mode.value, brain_mode=control_mode)
        current = seed_id or self._pick_jump()

        for i in range(steps):
            rs = self.runtime.step(control=control)
            nxt = self._pick_jump()
            if not current or not nxt:
                continue
            narration = f"Activation drift {current[:6]}->{nxt[:6]} | novelty={rs.novelty:.3f} tension={rs.tension:.3f} uncertainty={rs.uncertainty_mass:.3f}"
            insight = rs.novelty > 0.8 and rs.entropy > 0.3
            ds = DreamStep(
                step=i,
                from_id=current,
                to_id=nxt,
                edge_type="activation_jump",
                edge_narration="stochastic spread",
                narration=narration,
                is_insight=insight,
                insight_depth="structural" if insight else "",
                mission_advance=rs.novelty > 0.75 and self.brain.get_mission() is not None,
            )
            log.steps.append(ds)
            if insight:
                log.insights.append({"step": i, "narration": self.interpreter.summarize_insight({"default": narration}), "depth": "structural", "node": nxt})
            if ds.mission_advance:
                log.mission_advances.append({"step": i, "node": nxt, "explanation": f"novel mission-linked activation at {nxt[:8]}", "strength": min(1.0, rs.novelty)})
            current = nxt

        log.summary = f"Dream: steps={len(log.steps)} insights={len(log.insights)} advances={len(log.mission_advances)}"
        if self.brain.is_transitional():
            self.brain.complete_transition()

        import os
        os.makedirs("logs", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log.to_dict(), f, indent=2)
        return log
