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
        "emergence",
        "first_principles",
        "lateral",
        "empirical"
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

    def choose_pattern(self, node_type: str, cluster: str) -> str:
        """Epsilon-greedy action selection."""
        state_key = self._state_key(node_type, cluster)
        self._init_state(state_key)
        
        # Exploration
        if random.random() < self.epsilon:
            print(f"  [Procedural] Exploring random pattern for {state_key}")
            return random.choice(list(self.q_table[state_key].keys()))
            
        # Exploitation
        best_val = -float('inf')
        best_action = random.choice(self.DEFAULT_ACTIONS)
        for action, val in self.q_table[state_key].items():
            if val > best_val:
                best_val = val
                best_action = action
                
        print(f"  [Procedural] Exploiting best pattern '{best_action}' (val={best_val:.2f}) for {state_key}")
        return best_action

    def update(self, node_type: str, cluster: str, action: str, reward: float, dopamine: float = 0.5):
        """
        Update the expected value of an action in a state.
        Dopamine level modulates the learning rate.
        """
        state_key = self._state_key(node_type, cluster)
        self._init_state(state_key)
        
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.5
            
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
                print(f"Loaded procedural policy with {len(self.q_table)} states.")
            except Exception as e:
                print(f"Failed to load policy: {e}")
                self.q_table = {}
                
    def _save(self):
        atomic_write_json(self.POLICY_PATH, self.q_table)

