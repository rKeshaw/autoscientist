"""Unit tests for InsightBuffer analogy fallthrough prevention and routing."""
from unittest.mock import MagicMock
import pytest
from graph.brain import Brain, Node, NodeType
from insight_buffer import InsightBuffer, PendingInsight

@pytest.fixture
def mock_brain():
    brain = Brain()
    # Add two dummy nodes
    node_a = Node(id="node_a", statement="Statement A", node_type=NodeType.CONCEPT)
    node_b = Node(id="node_b", statement="Statement B", node_type=NodeType.CONCEPT)
    brain.add_node(node_a)
    brain.add_node(node_b)
    return brain

def test_analogy_rejected_does_not_fallthrough_to_associative_gate(mock_brain):
    buffer = InsightBuffer(brain=mock_brain)
    buffer.save = MagicMock()
    mock_critic = MagicMock()
    buffer.critic = mock_critic

    # Mock critic re-evaluate returning "rejected"
    buffer._critic_reevaluate = MagicMock(return_value="rejected")
    buffer._llm_evaluate = MagicMock()

    # Add analogy pending insight with high original similarity (e.g. 0.75, which would satisfy context_score >= 0.58)
    pair = PendingInsight(
        node_a_id="node_a",
        node_b_id="node_b",
        original_similarity=0.75,
        claim="Analogy claim",
        proposed_type="structural_analogy",
        edge_type="structural_analogy"
    )
    buffer.pending = [pair]

    res = buffer.evaluate_all()

    # Should NOT call associative _llm_evaluate
    buffer._llm_evaluate.assert_not_called()
    # Rejected pair should be pruned, not promoted
    assert res["promoted"] == 0
    assert res["pruned"] == 1
    assert len(buffer.pending) == 0

def test_analogy_deferred_remains_in_buffer_without_promotion(mock_brain):
    buffer = InsightBuffer(brain=mock_brain)
    buffer.save = MagicMock()
    mock_critic = MagicMock()
    buffer.critic = mock_critic

    # Mock critic re-evaluate returning "deferred"
    buffer._critic_reevaluate = MagicMock(return_value="deferred")
    buffer._llm_evaluate = MagicMock()

    pair = PendingInsight(
        node_a_id="node_a",
        node_b_id="node_b",
        original_similarity=0.75,
        claim="Analogy claim",
        proposed_type="structural_analogy",
        edge_type="structural_analogy"
    )
    buffer.pending = [pair]

    res = buffer.evaluate_all()

    # Should NOT call associative _llm_evaluate
    buffer._llm_evaluate.assert_not_called()
    assert res["promoted"] == 0
    assert res["pruned"] == 0
    assert len(buffer.pending) == 1