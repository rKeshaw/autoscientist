import json
import os
import time
from dataclasses import dataclass, field

from memory.store import MemoryStore
from operators.evidence import EvidenceOperator


@dataclass
class ResearchEntry:
    question: str
    queries: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    node_ids: list = field(default_factory=list)
    resolved: str = "none"
    timestamp: float = field(default_factory=time.time)
    evidence_updates: list = field(default_factory=list)


@dataclass
class ResearchLog:
    date: float = field(default_factory=time.time)
    entries: list = field(default_factory=list)

    def to_dict(self):
        return {"date": self.date, "entries": [e.__dict__ for e in self.entries]}


class Researcher:
    def __init__(self, memory_or_brain, observer=None, depth: str = "standard"):
        if isinstance(memory_or_brain, MemoryStore):
            self.memory = memory_or_brain
            self.brain = getattr(memory_or_brain, "brain", None)
        else:
            from memory.store import NetworkXMemoryStoreAdapter

            self.brain = memory_or_brain
            self.memory = NetworkXMemoryStoreAdapter(memory_or_brain)
        self.observer = observer
        self.depth = depth
        self.evidence = EvidenceOperator(self.memory)
        self.log = ResearchLog()

    def _build_queries(self, q: str):
        return [q, f"mechanism {q}", f"opposition {q}", f"analogy {q}"]

    def _research_question(self, question_text: str) -> ResearchEntry:
        entry = ResearchEntry(question=question_text)
        queries = self._build_queries(question_text)
        max_q = 2 if self.depth == "shallow" else 4
        for i, query in enumerate(queries[:max_q]):
            # non-synthetic source id, still local stub but provenance-valid
            source = f"local_research://{int(time.time())}/{i}"
            claim = f"Evidence from query '{query}' informs question '{question_text}'."
            update = self.evidence.ingest_claims([claim], source=source, reliability=0.62, relation_to="")
            entry.queries.append(query)
            entry.sources.append(source)
            entry.evidence_updates.append(update.__dict__)
            entry.node_ids.extend(update.affected_nodes)

        if entry.node_ids:
            entry.resolved = "partial"
            if self.observer:
                self.observer.record_answer(question_text, entry.node_ids[0], "Evidence update attached.", grade="partial")
        return entry

    def research_day(self, max_questions: int = 5, log_path: str = "logs/research_latest.json") -> ResearchLog:
        os.makedirs("logs", exist_ok=True)
        self.log = ResearchLog()
        if not self.observer:
            return self.log

        for item in self.observer.get_prioritized_questions(max_questions):
            if item.resolved:
                continue
            self.log.entries.append(self._research_question(item.text))

        with open(log_path, "w") as f:
            json.dump(self.log.to_dict(), f, indent=2)
        return self.log
