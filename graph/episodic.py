import time
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from persistence import atomic_write_json

@dataclass
class EpisodicEvent:
    event_type: str  # e.g., 'think', 'read', 'critic_reject', 'salience_interrupt', 'sandbox'
    description: str
    nodes_involved: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return self.__dict__

class EpisodicStrip:
    """
    Hippocampal Episodic Memory.
    Stores a chronological ribbon of events and actions to allow for trajectory replay during Dreaming.
    """
    STRIP_PATH = "data/episodic.json"

    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: List[EpisodicEvent] = []
        self._load()

    def record(self, event_type: str, description: str, nodes_involved: List[str] = None):
        if nodes_involved is None:
            nodes_involved = []
        
        event = EpisodicEvent(
            event_type=event_type,
            description=description,
            nodes_involved=nodes_involved
        )
        self.events.append(event)
        
        # Trim
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
            
        print(f"  [Episodic] Recorded {event_type}: {description[:60]}...")
        self._save()
        
    def get_recent(self, n: int = 10) -> List[EpisodicEvent]:
        return self.events[-n:]

    def get_sequence(self, sequence_length: int = 5) -> List[EpisodicEvent]:
        """Returns a random contiguous sequence of events for dreaming/replay."""
        if len(self.events) < sequence_length:
            return self.events
        import random
        start = random.randint(0, len(self.events) - sequence_length)
        return self.events[start:start+sequence_length]

    def _load(self):
        if os.path.exists(self.STRIP_PATH):
            try:
                with open(self.STRIP_PATH, 'r') as f:
                    data = json.load(f)
                    self.events = [EpisodicEvent(**e) for e in data]
                print(f"Loaded EpisodicStrip with {len(self.events)} events.")
            except Exception as e:
                print(f"Failed to load EpisodicStrip: {e}")
                
    def _save(self):
        atomic_write_json(self.STRIP_PATH, [e.to_dict() for e in self.events])
