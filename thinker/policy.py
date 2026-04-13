import json
import os
import random
import numpy as np
from typing import List, Dict, Any
from persistence import atomic_write_json

# ── Procedural Memory Policy ──────────────────────────────────────────────────

class CognitivePolicy:
    """
    Contextual Bandit for deciding which cognitive pattern to use.
    State: (node_type, cluster)
    Actions: cognitive_patterns
    """
    
    POLICY_PATH = "data/policy.json"
    
    DEFAULT_ACTIONS = [
        "analogical",
        "dialectical",
        "reductive",
        "experimental",
        "integrative",
    ]

    def __init__(self, epsilon: float = 0.2, learning_rate: float = 0.1):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.q_table: Dict[str, Dict[str, float]] = {}  # { "node_type|cluster": { "action": q_value } }
        self._load()
    
    def _state_key(self, node_type: str, cluster: str) -> str:
        return f"{node_type}|{cluster}"
        
    def _init_state(self, state_key: str):
        if state_key not in self.q_table:
            # Initialize with slight optimism to encourage early exploration
            self.q_table[state_key] = {action: 0.5 for action in self.DEFAULT_ACTIONS}

    def _sanitize_state(self, state_key: str):
        self._init_state(state_key)
        prior = self.q_table.get(state_key, {})
        self.q_table[state_key] = {
            action: float(prior.get(action, 0.5))
            for action in self.DEFAULT_ACTIONS
        }

    def choose_pattern(self, node_type: str, cluster: str,
                       preferred_action: str = "") -> str:
        """Epsilon-greedy action selection with a semantic preference tie-break."""
        state_key = self._state_key(node_type, cluster)
        self._sanitize_state(state_key)
        actions = list(self.q_table[state_key].keys())
        preferred_action = (
            preferred_action if preferred_action in self.q_table[state_key] else ""
        )
        
        # Exploration
        if random.random() < self.epsilon:
            print(f"  [Procedural] Exploring random pattern for {state_key}")
            return random.choice(actions)
            
        # Exploitation
        best_val = -float('inf')
        best_action = preferred_action or random.choice(actions)
        for action, val in self.q_table[state_key].items():
            effective_val = val + (0.12 if action == preferred_action else 0.0)
            if effective_val > best_val:
                best_val = effective_val
                best_action = action
                
        pref_note = f", preferred={preferred_action}" if preferred_action else ""
        print(
            f"  [Procedural] Exploiting best pattern '{best_action}' "
            f"(val={best_val:.2f}{pref_note}) for {state_key}"
        )
        return best_action

    def update(self, node_type: str, cluster: str, action: str, reward: float, dopamine: float = 0.5):
        """
        Update the expected value of an action in a state.
        Dopamine level modulates the learning rate.
        """
        state_key = self._state_key(node_type, cluster)
        self._sanitize_state(state_key)
        if action not in self.DEFAULT_ACTIONS:
            print(f"  [Procedural] Skipping unsupported pattern '{action}' for {state_key}")
            return
        
        old_val = self.q_table[state_key][action]
        
        # If dopamine is high, we learn faster from positive rewards.
        # If dopamine is low, we learn slower.
        adjusted_lr = self.learning_rate * (1.0 + dopamine)
        
        new_val = old_val + adjusted_lr * (reward - old_val)
        self.q_table[state_key][action] = new_val
        
        print(f"  [Procedural] Policy update: {state_key} + {action} -> rew={reward:.2f}, val: {old_val:.2f}->{new_val:.2f}")
        self._save()

    def _load(self):
        if os.path.exists(self.POLICY_PATH):
            try:
                with open(self.POLICY_PATH, 'r') as f:
                    self.q_table = json.load(f)
                for state_key in list(self.q_table.keys()):
                    self._sanitize_state(state_key)
                print(f"Loaded procedural policy with {len(self.q_table)} states.")
            except Exception as e:
                print(f"Failed to load policy: {e}")
                self.q_table = {}
                
    def _save(self):
        atomic_write_json(self.POLICY_PATH, self.q_table)
