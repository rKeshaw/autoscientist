from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

from graph.brain import Edge, EdgeSource, EdgeType, Node, NodeStatus, NodeType
from memory.store import MemoryStore


@dataclass
class TestOutcome:
    test_id: str
    hypothesis_id: str
    verdict: str
    effect_size: float
    confidence: float
    uncertainty_reduction: float
    artifacts_ref: str
    reproducibility: Dict[str, str]


class TestOperator:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def run(self, hypothesis_id: str, test_spec: Dict) -> TestOutcome:
        if not hasattr(self.memory, "brain"):
            return TestOutcome(test_id="na", hypothesis_id=hypothesis_id, verdict="invalid", effect_size=0.0, confidence=0.0, uncertainty_reduction=0.0, artifacts_ref="", reproducibility={})

        brain = self.memory.brain
        h = brain.get_node(hypothesis_id)
        if not h:
            return TestOutcome(test_id="na", hypothesis_id=hypothesis_id, verdict="invalid", effect_size=0.0, confidence=0.0, uncertainty_reduction=0.0, artifacts_ref="", reproducibility={})

        # deterministic pseudo-evaluation from spec parameters
        strength = float(test_spec.get("expected_signal", 0.5))
        falsifier = float(test_spec.get("falsifier_strength", 0.2))
        net = strength - falsifier
        if net > 0.15:
            verdict = "support"
        elif net < -0.15:
            verdict = "contradict"
        else:
            verdict = "inconclusive"

        effect_size = abs(net)
        confidence = min(1.0, 0.45 + effect_size)
        unc_reduction = min(0.5, 0.1 + 0.4 * confidence)

        exp = Node(
            statement=f"Experiment for hypothesis {hypothesis_id[:8]}: {test_spec.get('name', 'unnamed')}",
            node_type=NodeType.EXPERIMENT,
            status=NodeStatus.SETTLED,
            importance=0.6,
            uncertainty=0.2,
            provenance={"sources": ["test_operator"], "method": "deterministic_test", "created_at": time.time(), "updated_at": time.time()},
            value=0.3,
        )
        exp_id = brain.add_node(exp)

        result = Node(
            statement=f"Test outcome={verdict}, effect={effect_size:.3f}, confidence={confidence:.3f}",
            node_type=NodeType.RESULT,
            status=NodeStatus.SETTLED if verdict != "inconclusive" else NodeStatus.UNCERTAIN,
            importance=confidence,
            uncertainty=max(0.0, 1.0 - confidence),
            provenance={"sources": ["test_operator"], "method": "deterministic_test", "created_at": time.time(), "updated_at": time.time()},
            value=0.25,
        )
        result_id = brain.add_node(result)

        brain.add_edge(hypothesis_id, exp_id, Edge(type=EdgeType.TESTED_BY, narration="Hypothesis tested by experiment", weight=confidence, confidence=confidence, source=EdgeSource.TEST_OPERATOR, polarity=0))
        brain.add_edge(exp_id, result_id, Edge(type=EdgeType.PRODUCED_RESULT, narration="Experiment produced result", weight=confidence, confidence=confidence, source=EdgeSource.TEST_OPERATOR, polarity=0))
        if verdict == "support":
            brain.add_edge(result_id, hypothesis_id, Edge(type=EdgeType.SUPPORTS, narration="Result supports hypothesis", weight=confidence, confidence=confidence, source=EdgeSource.TEST_OPERATOR, polarity=1, uncertainty_impact=unc_reduction))
        elif verdict == "contradict":
            brain.add_edge(result_id, hypothesis_id, Edge(type=EdgeType.CONTRADICTS, narration="Result contradicts hypothesis", weight=confidence, confidence=confidence, source=EdgeSource.TEST_OPERATOR, polarity=-1, uncertainty_impact=unc_reduction))

        # update hypothesis uncertainty/confidence
        h_state = h.get("state", {})
        old_u = float(h_state.get("uncertainty", h.get("uncertainty", 0.5)))
        new_u = max(0.0, old_u - unc_reduction)
        h_state["uncertainty"] = new_u
        h["state"] = h_state
        h["uncertainty"] = new_u
        h["confidence"] = min(1.0, float(h.get("confidence", 0.5)) + 0.15 * confidence)

        outcome = TestOutcome(
            test_id=exp_id,
            hypothesis_id=hypothesis_id,
            verdict=verdict,
            effect_size=effect_size,
            confidence=confidence,
            uncertainty_reduction=unc_reduction,
            artifacts_ref=result_id,
            reproducibility={"seed": str(brain.global_state.rng_seed), "environment": "local", "hash": f"{exp_id[:8]}-{result_id[:8]}"},
        )
        brain.log_event("test_outcome", outcome.__dict__)
        return outcome
