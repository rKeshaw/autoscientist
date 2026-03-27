import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field

from graph.brain import Brain, Edge, EdgeSource, EdgeType, Node, NodeStatus, NodeType


@dataclass
class ConsolidationReport:
    started_at: float = field(default_factory=time.time)
    new_nodes: int = 0
    merges: int = 0
    syntheses: int = 0
    abstractions: int = 0
    gaps: int = 0
    contradictions_updated: int = 0
    edges_decayed: int = 0
    objective_before: float = 0.0
    objective_after: float = 0.0
    delta_objective: float = 0.0
    summary: str = ""
    new_node_ids: list = field(default_factory=list)
    synthesis_ids: list = field(default_factory=list)
    gap_ids: list = field(default_factory=list)

    def to_dict(self):
        return self.__dict__


class Consolidator:
    def __init__(self, memory_or_brain, observer=None, epsilon: float = 1e-3):
        self.brain: Brain = memory_or_brain.brain if hasattr(memory_or_brain, "brain") else memory_or_brain
        self.observer = observer
        self.epsilon = epsilon

    def _objective(self):
        n = len(self.brain.graph.nodes)
        e = len(self.brain.graph.edges)
        uncertainty = sum(float((d.get("state") or {}).get("uncertainty", d.get("uncertainty", 0.5))) for _, d in self.brain.graph.nodes(data=True))
        contradictions = sum(float(ed.get("confidence", 0.5)) for _, _, _, ed in self.brain.graph.edges(keys=True, data=True) if ed.get("type") == EdgeType.CONTRADICTS.value)
        # proxy reconstruction term from average edge weight (higher is better => lower loss)
        avg_w = sum(float(ed.get("weight", 0.5)) for _, _, _, ed in self.brain.graph.edges(keys=True, data=True)) / max(1, e)
        recon = 1.0 - avg_w
        complexity = 0.001 * (n + e)
        return recon + complexity + 0.01 * uncertainty + 0.05 * contradictions

    def _near_duplicates(self):
        ids = list(self.brain.graph.nodes)
        pairs = []
        for i, a in enumerate(ids):
            sa = self.brain.graph.nodes[a].get("content", {}).get("text", self.brain.graph.nodes[a].get("statement", "")).strip().lower()
            for b in ids[i + 1 :]:
                sb = self.brain.graph.nodes[b].get("content", {}).get("text", self.brain.graph.nodes[b].get("statement", "")).strip().lower()
                if sa and sb and (sa == sb or sa in sb or sb in sa):
                    pairs.append((a, b))
        return pairs

    def _merge(self, keep: str, drop: str):
        for u, _, k, d in list(self.brain.graph.in_edges(drop, keys=True, data=True)):
            if u != keep:
                self.brain.graph.add_edge(u, keep, **d)
        for _, v, k, d in list(self.brain.graph.out_edges(drop, keys=True, data=True)):
            if v != keep:
                self.brain.graph.add_edge(keep, v, **d)
        prov = self.brain.graph.nodes[keep].get("provenance", {})
        prov.setdefault("parent_ids", []).append(drop)
        self.brain.graph.nodes[keep]["provenance"] = prov
        self.brain.graph.remove_node(drop)

    def _propose_ops(self):
        proposals = []
        for a, b in self._near_duplicates()[:20]:
            proposals.append(("merge", {"keep": a, "drop": b}))

        clusters = {}
        for nid, nd in self.brain.all_nodes():
            clusters.setdefault(nd.get("meta", {}).get("cluster", nd.get("cluster", "unclustered")), []).append((nid, nd))
        for cluster, items in clusters.items():
            stable = [nid for nid, nd in items if float((nd.get("state") or {}).get("stability", nd.get("stability", 0.0))) > 0.4]
            if len(stable) >= 3:
                proposals.append(("abstraction", {"cluster": cluster, "members": stable[:6]}))
        return proposals

    def _apply_op(self, op, report):
        typ, payload = op
        if typ == "merge":
            if payload["keep"] in self.brain.graph and payload["drop"] in self.brain.graph:
                self._merge(payload["keep"], payload["drop"])
                report.merges += 1
        elif typ == "abstraction":
            aid = self.brain.create_abstraction(payload["members"], statement=f"Abstraction:{payload['cluster']}", cluster=payload["cluster"])
            if aid:
                report.abstractions += 1
                report.new_node_ids.append(aid)

    def consolidate(self, new_node_ids: list = None, save_path: str = "logs/consolidation_latest.json"):
        report = ConsolidationReport()
        report.objective_before = self._objective()

        proposals = self._propose_ops()
        for op in proposals:
            snapshot_nodes = deepcopy(list(self.brain.graph.nodes(data=True)))
            snapshot_edges = deepcopy(list(self.brain.graph.edges(keys=True, data=True)))
            before = self._objective()
            self._apply_op(op, report)
            after = self._objective()
            if after - before > -self.epsilon:
                # revert if objective does not improve
                self.brain.graph.clear()
                self.brain.graph.add_nodes_from(snapshot_nodes)
                for u, v, k, d in snapshot_edges:
                    self.brain.graph.add_edge(u, v, key=k, **d)

        contradictions = 0
        for u, v, k, d in self.brain.graph.edges(keys=True, data=True):
            if d.get("type") == EdgeType.CONTRADICTS.value:
                contradictions += 1
                d["decay_exempt"] = True
        report.contradictions_updated = contradictions

        before_e = len(self.brain.graph.edges)
        self.brain.apply_decay()
        to_remove = [(u, v, k) for u, v, k, d in self.brain.graph.edges(keys=True, data=True) if d.get("weight", 0.0) < 0.02 and not d.get("decay_exempt")]
        for u, v, k in to_remove:
            self.brain.graph.remove_edge(u, v, key=k)
        report.edges_decayed = max(0, before_e - len(self.brain.graph.edges))

        report.objective_after = self._objective()
        report.delta_objective = report.objective_after - report.objective_before
        report.summary = f"Consolidation objective delta={report.delta_objective:.6f}, merges={report.merges}, abstractions={report.abstractions}."

        os.makedirs("logs", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        self.brain.log_event("consolidation", report.to_dict())
        return report
