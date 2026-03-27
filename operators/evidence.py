from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from graph.brain import Edge, EdgeSource, EdgeType, Node, NodeStatus, NodeType
from memory.store import MemoryStore


@dataclass
class EvidenceUpdate:
    affected_nodes: List[str] = field(default_factory=list)
    affected_relations: List[str] = field(default_factory=list)
    posterior_delta: Dict[str, float] = field(default_factory=dict)
    uncertainty_delta: Dict[str, float] = field(default_factory=dict)
    reliability_weight: float = 0.5
    provenance_ref: str = ""


class EvidenceOperator:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def ingest_claims(self, claims: List[str], source: str, reliability: float = 0.6, relation_to: str = "") -> EvidenceUpdate:
        update = EvidenceUpdate(reliability_weight=reliability, provenance_ref=source)
        if not hasattr(self.memory, "brain"):
            return update
        brain = self.memory.brain

        for claim in claims:
            node = Node(
                statement=claim,
                node_type=NodeType.EVIDENCE,
                status=NodeStatus.UNCERTAIN,
                importance=max(0.2, reliability),
                uncertainty=max(0.0, 1.0 - reliability),
                provenance={"sources": [source], "method": "evidence_operator", "created_at": time.time(), "updated_at": time.time()},
                value=0.2,
            )
            nid = brain.add_node(node)
            update.affected_nodes.append(nid)
            update.uncertainty_delta[nid] = -0.1 * reliability

            if relation_to and brain.get_node(relation_to):
                et = EdgeType.SUPPORTS
                if any(k in claim.lower() for k in ["not", "contradict", "fails", "false"]):
                    et = EdgeType.CONTRADICTS
                eid = str(uuid.uuid4())
                edge = Edge(
                    type=et,
                    narration=f"Evidence from {source}",
                    weight=reliability,
                    confidence=reliability,
                    source=EdgeSource.EVIDENCE_OPERATOR,
                    polarity=1 if et == EdgeType.SUPPORTS else -1,
                    provenance={"evidence_ids": [nid], "source_reliability": reliability, "created_at": time.time(), "updated_at": time.time()},
                    uncertainty_impact=0.1 * reliability,
                )
                brain.add_edge(nid, relation_to, edge)
                update.affected_relations.append(eid)
                update.posterior_delta[relation_to] = update.posterior_delta.get(relation_to, 0.0) + (0.2 * reliability if et == EdgeType.SUPPORTS else -0.2 * reliability)

        brain.log_event("evidence_update", update.__dict__)
        return update
