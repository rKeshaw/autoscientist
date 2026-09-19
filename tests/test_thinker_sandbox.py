"""Unit tests for Thinker sandbox redirection and Observer vectorized agenda."""
from unittest.mock import MagicMock
import numpy as np
import pytest
from graph.brain import Brain, Node, NodeType, NodeStatus, Edge, EdgeType, EdgeSource
from thinker.thinker import Thinker, ThinkingLog
from observer.observer import Observer, AgendaItem
from sandbox.sandbox import SandboxResult

def test_thinker_sandbox_redirection_on_untested_hypothesis():
    brain = Brain()
    # Add an untested hypothesis
    h_node = Node(
        statement="Inhibitory STDP drives entropy production to a non-equilibrium steady state.",
        node_type=NodeType.HYPOTHESIS,
        cluster="thermodynamics",
        status=NodeStatus.UNCERTAIN
    )
    h_id = brain.add_node(h_node)

    mock_sandbox = MagicMock()
    mock_result = SandboxResult(
        hypothesis_node_id=h_id,
        hypothesis=h_node.statement,
        approach="Stochastic simulation of inhibitory network",
        code="print('done')",
        stdout="Entropy production rate: 4.22 nats/s",
        stderr="",
        verdict="supports",
        confidence=0.85,
        interpretation="Simulation showed robust positive entropy production matching theoretical bound.",
        implications="Confirms non-equilibrium steady state convergence."
    )
    mock_sandbox.test_hypothesis.return_value = mock_result

    thinker = Thinker(brain=brain, sandbox=mock_sandbox)
    res = thinker._recognize_and_run_experiment(
        question=h_node.statement,
        question_node_id=h_id,
        pattern="dialectical"
    )

    assert res is not None
    assert res.verdict == "supports"
    assert res.confidence == 0.85
    mock_sandbox.test_hypothesis.assert_called_once_with(h_node.statement, node_id=h_id)

def test_thinker_sandbox_skips_already_tested_hypothesis():
    brain = Brain()
    h_node = Node(statement="Test hypothesis", node_type=NodeType.HYPOTHESIS)
    h_id = brain.add_node(h_node)

    e_node = Node(statement="Empirical test", node_type=NodeType.EMPIRICAL)
    e_id = brain.add_node(e_node)

    # Link with EMPIRICALLY_TESTED edge
    edge = Edge(type=EdgeType.EMPIRICALLY_TESTED, narration="Tested")
    brain.add_edge(e_id, h_id, edge)

    mock_sandbox = MagicMock()
    thinker = Thinker(brain=brain, sandbox=mock_sandbox)
    res = thinker._recognize_and_run_experiment(
        question=h_node.statement,
        question_node_id=h_id,
        pattern="dialectical"
    )

    assert res is None
    mock_sandbox.test_hypothesis.assert_not_called()

def test_observer_vectorized_agenda_dedup():
    brain = Brain()
    observer = Observer(brain)
    observer._embed = MagicMock(side_effect=lambda text: np.array([1.0, 0.0] if "A" in text else [0.0, 1.0]))

    item1 = observer.add_to_agenda("Question A", item_type="question", cycle=1)
    assert len(observer.agenda) == 1
    assert item1.count == 1

    # Add duplicate Question A
    item2 = observer.add_to_agenda("Question A duplicate", item_type="question", cycle=1)
    assert len(observer.agenda) == 1
    assert item2.count == 2
    assert item2.priority > 0.5

    # Add distinct Question B
    item3 = observer.add_to_agenda("Question B", item_type="question", cycle=1)
    assert len(observer.agenda) == 2
    assert item3.count == 1
