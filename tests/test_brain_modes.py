"""Unit tests for Brain mode transitions and neuromodulator auto-resumption."""
import pytest
from graph.brain import Brain, BrainMode

def test_auto_resumption_from_wandering():
    brain = Brain()
    # Set a mission and suspend it to enter wandering mode
    brain.set_mission("Solve problem X")
    brain.suspend_mission()
    assert brain.is_wandering()
    assert brain._suspended_mission is not None

    # Set frustration to 0.45
    brain.frustration = 0.45
    # Decay: frustration drops by 0.20 to 0.25
    brain.apply_neuromodulator_decay()
    assert brain.frustration == pytest.approx(0.25, abs=1e-3)
    # Since frustration <= 0.40, mission should be automatically resumed
    assert brain.is_focused()
    assert brain.mission is not None
    assert brain.mission["question"] == "Solve problem X"
