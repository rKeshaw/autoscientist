from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from graph.brain import Brain


class MemoryStore(ABC):
    @abstractmethod
    def read_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def apply_delta(self, delta: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, tag: str = "") -> str:
        raise NotImplementedError

    @abstractmethod
    def load_snapshot(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError


class NetworkXMemoryStoreAdapter(MemoryStore):
    def __init__(self, brain: Brain):
        self.brain = brain

    def read_state(self) -> Dict[str, Any]:
        return self.brain.read_state()

    def apply_delta(self, delta: Dict[str, Any]) -> None:
        self.brain.apply_delta(delta)

    def snapshot(self, tag: str = "") -> str:
        return self.brain.snapshot(label=tag)

    def load_snapshot(self, path: str) -> Dict[str, Any]:
        return self.brain.load_snapshot(path)
