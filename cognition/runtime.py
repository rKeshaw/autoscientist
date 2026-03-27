from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cognition import dynamics
from cognition.metrics import activation_entropy, contradiction_density
from memory.store import MemoryStore


@dataclass
class ControlSignal:
    mode: str
    temperature: float
    action_set: List[str] = field(default_factory=list)
    budgets: Dict[str, int] = field(default_factory=dict)
    gains: Dict[str, float] = field(default_factory=dict)


@dataclass
class RuntimeStep:
    step_id: int
    mode: str
    novelty: float
    tension: float
    entropy: float
    uncertainty_mass: float
    contradiction_density: float
    rng_cursor: int
    deltas: Dict[str, float] = field(default_factory=dict)


class CognitiveRuntime:
    def __init__(self, memory: MemoryStore, observer=None, seed: int = 42):
        self.memory = memory
        self.observer = observer
        self.rng_seed = seed
        self.rng = random.Random(seed)
        self.rng_cursor = 0
        self.step_count = 0
        self.last_activation: Dict[str, float] = {}
        self.history: List[RuntimeStep] = []

        self.params = {
            "lambda_a": 0.08,
            "beta": 1.1,
            "gamma": 0.25,
            "kappa": 0.02,
            "eta_a": 1.0,
            "eta_v": 0.8,
            "eta_u": 0.7,
            "lambda_v": 0.08,
            "beta_m": 0.4,
            "beta_s": 0.2,
            "rho_c": 0.08,
            "rho_e": 0.12,
            "rho_t": 0.15,
            "delta": 0.03,
            "rho_sigma": 0.08,
            "rho_u": 0.05,
            "alpha_plastic": 0.02,
            "mu_plastic": 0.002,
            "xi_s": 0.05,
            "xi_k": 0.06,
        }

    def _bump_rng_cursor(self):
        self.rng_cursor += 1

    def _control_defaults(self, mode: str) -> ControlSignal:
        temp = 0.7
        if mode == "wandering":
            temp = 0.9
        elif mode == "focused":
            temp = 0.45
        elif mode == "consolidation":
            temp = 0.25
        elif mode == "acquisition":
            temp = 0.55
        return ControlSignal(mode=mode, temperature=temp, action_set=[], budgets={"steps": 1}, gains={})

    def _extract_fields(self, state):
        nodes = state["nodes"]
        node_ids = [n["id"] for n in nodes]
        activation = {n["id"]: float((n.get("state") or {}).get("activation", n.get("activation", 0.0))) for n in nodes}
        attention = {n["id"]: float((n.get("state") or {}).get("attention", n.get("attention", 0.0))) for n in nodes}
        value = {n["id"]: float((n.get("state") or {}).get("value", n.get("value", 0.0))) for n in nodes}
        uncertainty = {n["id"]: float((n.get("state") or {}).get("uncertainty", n.get("uncertainty", 0.5))) for n in nodes}
        stability = {n["id"]: float((n.get("state") or {}).get("stability", n.get("stability", 0.0))) for n in nodes}
        mission_proj = {n["id"]: float(n.get("mission_alignment", 0.0)) for n in nodes}
        return node_ids, activation, attention, value, uncertainty, stability, mission_proj

    def step(self, control: Optional[ControlSignal] = None) -> RuntimeStep:
        state = self.memory.read_state()
        if not state["nodes"]:
            step = RuntimeStep(step_id=self.step_count, mode="idle", novelty=0.0, tension=0.0, entropy=0.0, uncertainty_mass=0.0, contradiction_density=0.0, rng_cursor=self.rng_cursor)
            self.history.append(step)
            self.step_count += 1
            return step

        mode = state["global"].get("control_mode", "wandering")
        u = control or self._control_defaults(mode)

        node_ids, act, att, val, unc, stab, mission_proj = self._extract_fields(state)
        incoming: Dict[str, List[tuple]] = {nid: [] for nid in node_ids}
        contradiction_pairs = []
        contradiction_signal = {nid: 0.0 for nid in node_ids}
        evidence_signal = {nid: 0.0 for nid in node_ids}
        test_signal = {nid: 0.0 for nid in node_ids}

        for e in state["edges"]:
            incoming.setdefault(e["dst"], []).append((e["src"], float(e.get("weight", 0.5))))
            if e.get("type") == "contradicts":
                conf = float(e.get("confidence", 0.5))
                contradiction_pairs.append((e["src"], e["dst"], conf))
                contradiction_signal[e["src"]] = contradiction_signal.get(e["src"], 0.0) + conf
                contradiction_signal[e["dst"]] = contradiction_signal.get(e["dst"], 0.0) + conf
            if e.get("type") in ("supports", "derived_from"):
                evidence_signal[e["dst"]] = evidence_signal.get(e["dst"], 0.0) + float(e.get("confidence", 0.5))
            if e.get("type") in ("tested_by", "produced_result", "empirically_tested"):
                test_signal[e["src"]] = test_signal.get(e["src"], 0.0) + float(e.get("confidence", 0.5))

        inhibition = dynamics.compute_inhibition(act)
        noise_std = max(0.01, 0.25 * u.temperature)

        nxt_act = dynamics.activation_propagation(
            node_ids,
            act,
            incoming,
            val,
            mission_proj,
            decay_lambda=self.params["lambda_a"],
            beta=self.params["beta"],
            noise_std=noise_std,
            gamma=self.params["gamma"],
            kappa=self.params["kappa"],
            inhibition=inhibition,
            rng=self.rng,
        )
        self._bump_rng_cursor()

        nxt_att = dynamics.attention_update(nxt_act, val, unc, self.params["eta_a"], self.params["eta_v"], self.params["eta_u"])
        nxt_val = dynamics.value_update(val, mission_proj, evidence_signal, self.params["lambda_v"], self.params["beta_m"], self.params["beta_s"])
        nxt_unc = dynamics.uncertainty_update(unc, contradiction_signal, evidence_signal, test_signal, self.params["rho_c"], self.params["rho_e"], self.params["rho_t"])
        nxt_stab = dynamics.stability_update(stab, nxt_att, nxt_act, nxt_unc, self.params["delta"], self.params["rho_sigma"], self.params["rho_u"])

        delta = {"node_updates": [], "edge_updates": []}
        for nid in node_ids:
            delta["node_updates"].append(
                {
                    "id": nid,
                    "set": {
                        "activation": nxt_act[nid],
                        "attention": nxt_att[nid],
                        "value": nxt_val[nid],
                        "uncertainty": nxt_unc[nid],
                        "stability": nxt_stab[nid],
                        "state": {
                            "activation": nxt_act[nid],
                            "attention": nxt_att[nid],
                            "value": nxt_val[nid],
                            "uncertainty": nxt_unc[nid],
                            "stability": nxt_stab[nid],
                        },
                        "last_activated": time.time(),
                    },
                }
            )

        for e in state["edges"]:
            w_new = dynamics.plasticity_update(
                float(e.get("weight", 0.5)),
                nxt_act.get(e["src"], 0.0),
                nxt_act.get(e["dst"], 0.0),
                self.params["alpha_plastic"] * (1.5 - u.temperature),
                self.params["mu_plastic"],
            )
            support_sig = 1.0 if e.get("type") in ("supports", "derived_from") else 0.0
            contrad_sig = 1.0 if e.get("type") == "contradicts" else 0.0
            c_new = dynamics.confidence_update(float(e.get("confidence", 0.5)), support_sig, contrad_sig, self.params["xi_s"], self.params["xi_k"])
            delta["edge_updates"].append({"src": e["src"], "dst": e["dst"], "key": e.get("key", 0), "set": {"weight": w_new, "transmissibility": w_new, "confidence": c_new, "updated_at": time.time()}})

        self.memory.apply_delta(delta)

        post = self.memory.read_state()
        a_vec = {n["id"]: float((n.get("state") or {}).get("activation", 0.0)) for n in post["nodes"]}
        novelty = dynamics.novelty_signal(a_vec, self.last_activation)
        self.last_activation = a_vec
        tension = dynamics.contradiction_tension(contradiction_pairs, a_vec)
        entropy = activation_entropy(a_vec)
        unc_mass = dynamics.uncertainty_mass({n["id"]: float((n.get("state") or {}).get("uncertainty", 0.5)) for n in post["nodes"]})
        c_density = contradiction_density(len(post["edges"]), sum(1 for e in post["edges"] if e.get("type") == "contradicts"))

        # keep global state updated
        global_update = {
            "node_updates": [],
            "edge_updates": [],
        }
        # use memory adapter to mutate via brain global state when available
        if hasattr(self.memory, "brain"):
            b = self.memory.brain
            b.global_state.step_id = self.step_count
            b.global_state.rng_cursor = self.rng_cursor
            b.global_state.temperature = u.temperature
            b.global_state.inhibition = inhibition
            b.global_state.control_mode = u.mode
            b.cognitive_temperature = u.temperature
            b.global_inhibition = inhibition
            b.last_novelty = novelty
            b.log_event("runtime_step", {"step_id": self.step_count, "mode": u.mode, "novelty": novelty, "tension": tension, "uncertainty_mass": unc_mass, "entropy": entropy, "control": {"mode": u.mode, "temperature": u.temperature, "action_set": u.action_set, "budgets": u.budgets}})

        step = RuntimeStep(
            step_id=self.step_count,
            mode=u.mode,
            novelty=novelty,
            tension=tension,
            entropy=entropy,
            uncertainty_mass=unc_mass,
            contradiction_density=c_density,
            rng_cursor=self.rng_cursor,
            deltas={"mean_activation": sum(a_vec.values()) / max(1, len(a_vec)), "inhibition": inhibition},
        )
        self.history.append(step)
        self.step_count += 1
        if self.observer and hasattr(self.observer, "observe_runtime"):
            self.observer.observe_runtime(step)
        return step

    def run_cycle(self, steps: int = 20, control: Optional[ControlSignal] = None) -> List[RuntimeStep]:
        out = []
        for _ in range(steps):
            out.append(self.step(control=control))
        return out
