import json
import os
import time
from dataclasses import dataclass, field

from memory.store import MemoryStore
from operators.test_operator import TestOperator, TestOutcome
from graph.brain import NodeType

SANDBOX_LOG_PATH = "logs/sandbox_log.json"


@dataclass
class SandboxResult:
    hypothesis_node_id: str
    hypothesis: str
    approach: str
    code: str
    stdout: str
    stderr: str
    verdict: str
    confidence: float
    interpretation: str
    implications: str
    plot_path: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    effect_size: float = 0.0
    uncertainty_reduction: float = 0.0
    reproducibility: dict = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__


class Sandbox:
    def __init__(self, memory_or_brain, observer=None):
        if isinstance(memory_or_brain, MemoryStore):
            self.memory = memory_or_brain
            self.brain = getattr(memory_or_brain, "brain", None)
        else:
            from memory.store import NetworkXMemoryStoreAdapter

            self.brain = memory_or_brain
            self.memory = NetworkXMemoryStoreAdapter(memory_or_brain)
        self.observer = observer
        self.test_operator = TestOperator(self.memory)
        self.results: list[SandboxResult] = []
        self._load()

    def _classify(self, hypothesis: str) -> str:
        h = hypothesis.lower()
        if any(k in h for k in ["simulate", "equation", "differential", "model"]):
            return "simulation-friendly"
        if any(k in h for k in ["proof", "logic", "consistency", "axiom"]):
            return "symbolic"
        if any(k in h for k in ["compute", "algorithm", "complexity"]):
            return "computational"
        return "conceptual"

    def test_hypothesis(self, hypothesis: str, node_id: str = "") -> SandboxResult:
        t0 = time.time()
        kind = self._classify(hypothesis)
        spec = {"name": f"{kind}_test", "expected_signal": 0.6 if kind != "conceptual" else 0.45, "falsifier_strength": 0.25}
        outcome: TestOutcome = self.test_operator.run(node_id, spec)

        result = SandboxResult(
            hypothesis_node_id=node_id,
            hypothesis=hypothesis,
            approach=kind,
            code=f"test_spec:{json.dumps(spec)}",
            stdout=f"Structured test executed with verdict={outcome.verdict}",
            stderr="",
            verdict=outcome.verdict,
            confidence=outcome.confidence,
            interpretation=f"effect={outcome.effect_size:.3f}, uncertainty_reduction={outcome.uncertainty_reduction:.3f}",
            implications="Outcome integrated via experiment/result nodes.",
            duration_seconds=time.time() - t0,
            effect_size=outcome.effect_size,
            uncertainty_reduction=outcome.uncertainty_reduction,
            reproducibility=outcome.reproducibility,
        )
        self.results.append(result)
        self._save()
        return result

    def scan_and_test(self, max_tests: int = 3):
        if not self.brain:
            return []
        candidates = list(self.brain.nodes_by_type(NodeType.HYPOTHESIS))[:max_tests]
        out = []
        for nid, nd in candidates:
            out.append(self.test_hypothesis(nd.get("statement", ""), node_id=nid))
        return out

    def _save(self):
        os.makedirs("logs", exist_ok=True)
        with open(SANDBOX_LOG_PATH, "w") as f:
            json.dump({"results": [r.to_dict() for r in self.results]}, f, indent=2)

    def _load(self):
        try:
            with open(SANDBOX_LOG_PATH, "r") as f:
                self.results = [SandboxResult(**r) for r in json.load(f).get("results", [])]
        except FileNotFoundError:
            self.results = []
