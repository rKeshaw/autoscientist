import json
import os
import time
import uuid
from dataclasses import dataclass, field

import requests

from memory.store import MemoryStore
from operators.evidence import EvidenceOperator

READING_LIST_PATH = "data/reading_list.json"


@dataclass
class ReadingEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    title: str = ""
    source_type: str = "web"
    priority: float = 0.5
    status: str = "unread"
    added_by: str = "user"
    added_reason: str = ""
    absorbed_at: float = 0.0
    node_count: int = 0

    def to_dict(self):
        return self.__dict__


@dataclass
class AbsorptionResult:
    entry: ReadingEntry
    title: str
    text_length: int
    node_count: int
    summary: str
    success: bool
    evidence_updates: list = field(default_factory=list)
    error: str = ""


class Reader:
    def __init__(self, memory_or_brain, observer=None, notebook=None):
        if isinstance(memory_or_brain, MemoryStore):
            self.memory = memory_or_brain
            self.brain = getattr(memory_or_brain, "brain", None)
        else:
            from memory.store import NetworkXMemoryStoreAdapter

            self.brain = memory_or_brain
            self.memory = NetworkXMemoryStoreAdapter(memory_or_brain)
        self.observer = observer
        self.notebook = notebook
        self.evidence = EvidenceOperator(self.memory)
        self.reading_list: list[ReadingEntry] = []
        self._load_list()

    def _load_list(self):
        try:
            with open(READING_LIST_PATH, "r") as f:
                self.reading_list = [ReadingEntry(**x) for x in json.load(f)]
        except FileNotFoundError:
            self.reading_list = []

    def _save_list(self):
        os.makedirs(os.path.dirname(READING_LIST_PATH) or ".", exist_ok=True)
        with open(READING_LIST_PATH, "w") as f:
            json.dump([x.to_dict() for x in self.reading_list], f, indent=2)

    def add_to_list(self, url: str, title: str = "", source_type: str = "web", priority: float = 0.5, added_by: str = "user", reason: str = ""):
        for e in self.reading_list:
            if e.url == url:
                return e
        e = ReadingEntry(url=url, title=title or url, source_type=source_type, priority=priority, added_by=added_by, added_reason=reason)
        self.reading_list.append(e)
        self._save_list()
        return e

    def list_all(self):
        return [e.to_dict() for e in self.reading_list]

    def get_unread(self, n: int = 5):
        unread = [x for x in self.reading_list if x.status == "unread"]
        return sorted(unread, key=lambda x: x.priority, reverse=True)[:n]

    def _chunk(self, text: str, size: int = 1200):
        return [text[i : i + size] for i in range(0, len(text), size)]

    def _canonical(self, chunk: str):
        return " ".join(chunk.lower().split())[:240]

    def _fetch(self, entry: ReadingEntry):
        if entry.source_type == "text":
            return entry.title, entry.added_reason
        try:
            resp = requests.get(entry.url, timeout=15, headers={"User-Agent": "autodreamer/3.0"})
            resp.raise_for_status()
            return entry.title or entry.url, resp.text
        except Exception as e:
            return entry.title or entry.url, f"ERROR: {e}"

    def absorb_entry(self, entry: ReadingEntry):
        title, text = self._fetch(entry)
        if text.startswith("ERROR:"):
            entry.status = "failed"
            self._save_list()
            return AbsorptionResult(entry, title, 0, 0, "Fetch failed", False, error=text)

        seen = set()
        updates = []
        for chunk in self._chunk(text[:12000]):
            canon = self._canonical(chunk)
            if canon in seen:
                continue
            seen.add(canon)
            claims = [f"Reading extract from {title}: {chunk[:220]}"]
            upd = self.evidence.ingest_claims(claims, source=entry.url or f"text://{entry.id}", reliability=0.58)
            updates.append(upd)

        entry.status = "read"
        entry.absorbed_at = time.time()
        entry.node_count = sum(len(u.affected_nodes) for u in updates)
        self._save_list()

        summary = f"Absorbed {entry.node_count} evidence nodes from {title}."
        if self.notebook:
            self.notebook._add_entry("field_notes", summary, 0, tags=["reading", title])

        return AbsorptionResult(entry, title, len(text), entry.node_count, summary, True, evidence_updates=[u.__dict__ for u in updates])

    def absorb_url(self, url: str, title: str = "", source_type: str = "web"):
        entry = self.add_to_list(url=url, title=title or url, source_type=source_type, priority=0.7, added_by="user")
        return self.absorb_entry(entry)

    def add_text(self, text: str, title: str = "Manual text", priority: float = 0.7):
        entry = ReadingEntry(url=f"text://{uuid.uuid4()}", title=title, source_type="text", priority=priority, added_by="user", added_reason=text)
        self.reading_list.append(entry)
        self._save_list()
        return self.absorb_entry(entry)

    def generate_reading_list(self):
        # policy placeholder: mission/agenda-driven candidate slots, not synthetic example domains
        additions = []
        seed_topics = ["network science", "scientific method", "knowledge graph"]
        for topic in seed_topics:
            additions.append(self.add_to_list(url=f"topic://{topic.replace(' ', '_')}", title=topic.title(), source_type="topic", priority=0.4, added_by="auto", reason="agenda-driven"))
        return additions

    def reading_day(self, max_items: int = 2):
        return [self.absorb_entry(e) for e in self.get_unread(max_items)]

    def stats(self):
        return {
            "total": len(self.reading_list),
            "unread": sum(1 for e in self.reading_list if e.status == "unread"),
            "read": sum(1 for e in self.reading_list if e.status == "read"),
            "failed": sum(1 for e in self.reading_list if e.status == "failed"),
        }
