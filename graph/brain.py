import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import networkx as nx

SCHEMA_VERSION = 3


class BrainMode(str, Enum):
    FOCUSED = "focused"
    WANDERING = "wandering"
    CONSOLIDATION = "consolidation"
    ACQUISITION = "acquisition"
    TESTING = "testing"
    OBSERVER_TRIGGERED = "observer_triggered"
    TRANSITIONAL = "transitional"


class NodeType(str, Enum):
    CONCEPT = "concept"
    CLAIM = "claim"
    HYPOTHESIS = "hypothesis"
    QUESTION = "question"
    ABSTRACTION = "abstraction"
    EVIDENCE = "evidence"
    EXPERIMENT = "experiment"
    RESULT = "result"
    MISSION = "mission"
    RELATION_INSTANCE = "relation_instance"
    # legacy compatibility
    ANSWER = "answer"
    SYNTHESIS = "synthesis"
    GAP = "gap"
    EMPIRICAL = "empirical"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"
    ARCHIVED = "archived"
    CONFIRMED = "confirmed"
    HYPOTHETICAL = "hypothetical"
    SETTLED = "settled"


class EdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    EXPLAINS = "explains"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    ANSWERS = "answers"
    DERIVED_FROM = "derived_from"
    TESTED_BY = "tested_by"
    PRODUCED_RESULT = "produced_result"
    ABOUT_MISSION = "about_mission"
    ASSOCIATED = "associated"
    PARTIAL = "partial"
    GAP_BRIDGE = "gap_bridge"
    EMPIRICALLY_TESTED = "empirically_tested"
    TOWARD_MISSION = "toward_mission"
    ABSTRACTION_OF = "abstraction_of"


class EdgeSource(str, Enum):
    CONVERSATION = "conversation"
    RESEARCH = "research"
    READING = "reading"
    DREAM = "dream"
    CONSOLIDATION = "consolidation"
    SANDBOX = "sandbox"
    RUNTIME = "runtime"
    EVIDENCE_OPERATOR = "evidence_operator"
    TEST_OPERATOR = "test_operator"


@dataclass
class NodeStateV3:
    activation: float = 0.0
    attention: float = 0.0
    value: float = 0.0
    uncertainty: float = 0.5
    stability: float = 0.0


@dataclass
class NodeV3:
    id: str
    type: str
    content: Dict[str, Any]
    state: Dict[str, float]
    confidence: float
    status: str
    provenance: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class RelationEdgeV3:
    id: str
    type: str
    src: str
    dst: str
    weight: float
    confidence: float
    polarity: int
    uncertainty_impact: float
    provenance: Dict[str, Any]


@dataclass
class RelationNodeV3:
    id: str
    relation_kind: str
    participants: List[str]
    qualifiers: Dict[str, Any]
    confidence: float
    provenance: Dict[str, Any]


@dataclass
class GlobalStateV3:
    temperature: float = 0.7
    inhibition: float = 0.0
    exploration_bias: float = 0.5
    mission_vector_ref: Optional[str] = None
    control_mode: str = BrainMode.WANDERING.value
    rng_seed: int = 42
    rng_cursor: int = 0
    step_id: int = 0
    schema_version: int = SCHEMA_VERSION


@dataclass
class Node:
    statement: str
    node_type: NodeType = NodeType.CONCEPT
    cluster: str = "unclustered"
    status: NodeStatus = NodeStatus.UNCERTAIN
    importance: float = 0.5
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    activated_at: float = field(default_factory=time.time)

    activation: float = 0.0
    attention: float = 0.0
    value: float = 0.0
    uncertainty: float = 0.5
    stability: float = 0.0

    resting_activation: float = 0.0
    salience: float = 0.5
    last_activated: float = field(default_factory=time.time)
    activation_trace: List[float] = field(default_factory=list)
    representation_modes: List[str] = field(default_factory=lambda: ["language"])
    provenance: Dict = field(default_factory=dict)
    mission_alignment: float = 0.0
    novelty_flag: bool = False

    predicted_answer: str = ""
    testable_by: str = ""
    incubation_age: int = 0
    first_queued_cycle: int = 0
    empirical_result: str = ""
    empirical_code: str = ""

    def to_dict(self):
        d = asdict(self)
        d["node_type"] = self.node_type.value
        d["status"] = self.status.value
        d["type"] = d["node_type"]
        d["content"] = {"text": self.statement, "embedding_ref": None, "symbolic_repr": None, "structural_signature": None}
        d["state"] = {
            "activation": self.activation,
            "attention": self.attention,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "stability": self.stability,
        }
        d["confidence"] = d.get("confidence", max(0.0, min(1.0, self.importance)))
        d["meta"] = {"cluster": self.cluster, "tags": [], "version": SCHEMA_VERSION}
        return d


@dataclass
class Edge:
    type: EdgeType
    narration: str
    weight: float = 0.5
    confidence: float = 0.5
    source: EdgeSource = EdgeSource.CONVERSATION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    transmissibility: float = 0.5
    plasticity_rate: float = 0.02
    decay_rate: float = 0.01
    edge_type: str = "semantic"
    recency: float = field(default_factory=time.time)
    reinforcement_count: int = 0
    provenance: Dict = field(default_factory=dict)
    decay_exempt: bool = False
    polarity: int = 0
    uncertainty_impact: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["type"] = self.type.value
        d["source"] = self.source.value
        d["id"] = d.get("id", str(uuid.uuid4()))
        return d


class Brain:
    def __init__(self, decay_rate: float = 0.01, scientificness: float = 0.7):
        self.graph = nx.MultiDiGraph()
        self.decay_rate = decay_rate
        self.scientificness = scientificness
        self.mission: Optional[dict] = None
        self._suspended_mission: Optional[dict] = None
        self._mode: BrainMode = BrainMode.WANDERING

        self.global_state = GlobalStateV3()
        self.cognitive_temperature = self.global_state.temperature
        self.global_inhibition = self.global_state.inhibition
        self.exploration_bias = self.global_state.exploration_bias
        self.mission_focus: Dict[str, float] = {}
        self.last_novelty = 0.0
        self.version = "3.0-schema"
        self.event_log: List[Dict] = []

    # MemoryStore boundary
    def read_state(self) -> Dict[str, Any]:
        return {
            "nodes": [dict(id=nid, **data) for nid, data in self.graph.nodes(data=True)],
            "edges": [dict(src=u, dst=v, key=k, **d) for u, v, k, d in self.graph.edges(keys=True, data=True)],
            "global": asdict(self.global_state),
            "mission": self.mission,
        }

    def apply_delta(self, delta: Dict[str, Any]) -> None:
        for upd in delta.get("node_updates", []):
            nid = upd["id"]
            if nid in self.graph:
                self.graph.nodes[nid].update(upd.get("set", {}))
        for add in delta.get("nodes_add", []):
            nid = add.get("id", str(uuid.uuid4()))
            payload = dict(add)
            payload.pop("id", None)
            self.graph.add_node(nid, **payload)
        for upd in delta.get("edge_updates", []):
            u, v, key = upd["src"], upd["dst"], upd.get("key", 0)
            if self.graph.has_edge(u, v, key=key):
                self.graph.edges[u, v, key].update(upd.get("set", {}))
        for add in delta.get("edges_add", []):
            self.graph.add_edge(add["src"], add["dst"], **{k: v for k, v in add.items() if k not in ("src", "dst")})

    def snapshot(self, directory: str = "data/snapshots", label: str = "") -> str:
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(directory, f"brain-{stamp}{('-' + label) if label else ''}.json")
        self.save(path)
        return path

    def load_snapshot(self, path: str) -> Dict[str, Any]:
        self.load(path)
        return self.read_state()

    @property
    def mode(self) -> BrainMode:
        return self._mode

    def get_mode(self) -> str:
        return self._mode.value

    def set_mode(self, mode: BrainMode):
        self._mode = mode
        self.global_state.control_mode = mode.value
        self.log_event("mode_change", {"mode": mode.value})

    def is_wandering(self):
        return self._mode == BrainMode.WANDERING

    def is_transitional(self):
        return self._mode == BrainMode.TRANSITIONAL

    def set_mission(self, question: str, context: str = "") -> str:
        if self.mission and self.mission.get("id") in self.graph:
            self.graph.remove_node(self.mission["id"])
        node = Node(statement=question, node_type=NodeType.MISSION, cluster="mission", status=NodeStatus.ACTIVE, importance=1.0, mission_alignment=1.0, value=1.0, predicted_answer=context)
        nid = self.add_node(node)
        self.mission = {"id": nid, "question": question, "context": context}
        self.set_mode(BrainMode.TRANSITIONAL)
        self.log_event("mission_set", self.mission)
        return nid

    def suspend_mission(self):
        if self.mission:
            self._suspended_mission = self.mission
            self.set_mode(BrainMode.WANDERING)

    def resume_mission(self):
        if self._suspended_mission:
            self.mission = self._suspended_mission
            self._suspended_mission = None
        if self.mission:
            self.set_mode(BrainMode.FOCUSED)

    def complete_transition(self):
        if self._mode == BrainMode.TRANSITIONAL:
            self.set_mode(BrainMode.FOCUSED)

    def get_mission(self):
        return self.mission

    def add_node(self, node: Node) -> str:
        self.graph.add_node(node.id, **node.to_dict())
        return node.id

    def add_relation_node(self, relation: RelationNodeV3) -> str:
        nid = relation.id or str(uuid.uuid4())
        self.graph.add_node(
            nid,
            id=nid,
            node_type=NodeType.RELATION_INSTANCE.value,
            type=NodeType.RELATION_INSTANCE.value,
            relation_kind=relation.relation_kind,
            participants=relation.participants,
            qualifiers=relation.qualifiers,
            confidence=relation.confidence,
            provenance=relation.provenance,
            content={"text": f"relation:{relation.relation_kind}"},
            state={"activation": 0.0, "attention": 0.0, "value": 0.0, "uncertainty": 0.5, "stability": 0.0},
            status=NodeStatus.UNCERTAIN.value,
            meta={"version": SCHEMA_VERSION},
        )
        return nid

    def get_node(self, node_id: str):
        return self.graph.nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs):
        if node_id in self.graph:
            self.graph.nodes[node_id].update(kwargs)

    def all_nodes(self):
        return list(self.graph.nodes(data=True))

    def nodes_by_type(self, node_type: NodeType):
        t = node_type.value
        return [(n, d) for n, d in self.graph.nodes(data=True) if d.get("node_type") == t or d.get("type") == t]

    def add_edge(self, from_id: str, to_id: str, edge: Edge):
        self.graph.add_edge(from_id, to_id, **edge.to_dict())

    def add_relation_edge_v3(self, rel: RelationEdgeV3):
        self.graph.add_edge(rel.src, rel.dst, id=rel.id, type=rel.type, weight=rel.weight, confidence=rel.confidence, polarity=rel.polarity, uncertainty_impact=rel.uncertainty_impact, provenance=rel.provenance)

    def get_edge(self, from_id: str, to_id: str) -> Optional[dict]:
        if not self.graph.has_edge(from_id, to_id):
            return None
        first_key = next(iter(self.graph[from_id][to_id].keys()))
        return self.graph[from_id][to_id][first_key]

    def update_edge(self, from_id: str, to_id: str, **kwargs):
        if not self.graph.has_edge(from_id, to_id):
            return
        for key in self.graph[from_id][to_id]:
            self.graph[from_id][to_id][key].update(kwargs)
            self.graph[from_id][to_id][key]["updated_at"] = time.time()

    def neighbors(self, node_id: str):
        return list(self.graph.successors(node_id))

    def link_to_mission(self, node_id: str, narration: str, strength: float = 0.6):
        if not self.mission or self.is_wandering() or node_id == self.mission["id"]:
            return
        self.add_edge(
            node_id,
            self.mission["id"],
            Edge(type=EdgeType.ABOUT_MISSION, narration=narration, weight=min(1.0, strength), confidence=max(0.4, strength), source=EdgeSource.RUNTIME, decay_exempt=True, polarity=1),
        )

    def apply_activation_updates(self, activation_map: Dict[str, float], ts: Optional[float] = None):
        now = ts or time.time()
        for nid, val in activation_map.items():
            if nid not in self.graph:
                continue
            nd = self.graph.nodes[nid]
            nd["activation"] = float(val)
            nd.setdefault("state", {})["activation"] = float(val)
            nd["last_activated"] = now
            nd["activated_at"] = now
            trace = nd.get("activation_trace", [])[-99:]
            trace.append(float(val))
            nd["activation_trace"] = trace

    def strengthen_edge(self, u: str, v: str, amount: float):
        if not self.graph.has_edge(u, v):
            return
        for key in self.graph[u][v]:
            data = self.graph[u][v][key]
            data["weight"] = min(1.5, data.get("weight", 0.5) + amount)
            data["reinforcement_count"] = data.get("reinforcement_count", 0) + 1
            data["recency"] = time.time()

    def mark_contradiction(self, u: str, v: str, narration: str = ""):
        self.add_edge(u, v, Edge(type=EdgeType.CONTRADICTS, narration=narration or "Contradiction marked", weight=0.7, confidence=0.7, source=EdgeSource.CONSOLIDATION, decay_exempt=True, polarity=-1))

    def create_abstraction(self, node_ids: List[str], statement: str, cluster: str = "abstraction") -> Optional[str]:
        if not node_ids:
            return None
        node = Node(statement=statement, node_type=NodeType.ABSTRACTION, cluster=cluster, status=NodeStatus.HYPOTHETICAL, importance=0.8, provenance={"compressed_from": node_ids}, value=0.4)
        aid = self.add_node(node)
        for nid in node_ids:
            if nid in self.graph:
                self.add_edge(aid, nid, Edge(type=EdgeType.ABSTRACTION_OF, narration="Compressed representation", weight=0.7, confidence=0.7, source=EdgeSource.CONSOLIDATION, polarity=1))
        self.log_event("abstraction_created", {"id": aid, "members": node_ids})
        return aid

    def apply_decay(self):
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if data.get("decay_exempt"):
                continue
            decay = data.get("decay_rate", self.decay_rate)
            data["weight"] = max(0.0, data.get("weight", 0.5) - decay)

    def log_event(self, event_type: str, payload: Dict[str, Any]):
        self.event_log.append({"time": time.time(), "step_id": self.global_state.step_id, "rng_cursor": self.global_state.rng_cursor, "type": event_type, "payload": payload})
        self.event_log = self.event_log[-5000:]

    def stats(self):
        nodes = len(self.graph.nodes)
        edges = len(self.graph.edges)
        contradictions = sum(1 for _, _, _, d in self.graph.edges(keys=True, data=True) if d.get("type") == EdgeType.CONTRADICTS.value)
        abstractions = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == NodeType.ABSTRACTION.value)
        uncertainty_mass = sum(float((d.get("state") or {}).get("uncertainty", d.get("uncertainty", 0.5))) for _, d in self.graph.nodes(data=True))
        return {
            "nodes": nodes,
            "edges": edges,
            "mode": self.get_mode(),
            "mission": self.mission["question"] if self.mission else None,
            "temperature": self.cognitive_temperature,
            "global_inhibition": self.global_inhibition,
            "contradictions": contradictions,
            "abstractions": abstractions,
            "uncertainty_mass": uncertainty_mass,
            "schema_version": SCHEMA_VERSION,
        }

    def _migrate_node_to_v3(self, nid: str, nd: Dict[str, Any]):
        content = nd.get("content") or {"text": nd.get("statement", ""), "embedding_ref": None, "symbolic_repr": None, "structural_signature": None}
        state = nd.get("state") or {
            "activation": float(nd.get("activation", 0.0)),
            "attention": float(nd.get("attention", nd.get("salience", 0.0))),
            "value": float(nd.get("value", nd.get("mission_alignment", 0.0))),
            "uncertainty": float(nd.get("uncertainty", 0.5)),
            "stability": float(nd.get("stability", 0.0)),
        }
        prov = nd.get("provenance") or {"sources": [], "method": "migration", "created_at": nd.get("created_at", time.time()), "updated_at": time.time()}
        if "sources" not in prov:
            prov["sources"] = []
        nd["id"] = nid
        nd["type"] = nd.get("type") or nd.get("node_type", NodeType.CONCEPT.value)
        nd["content"] = content
        nd["state"] = state
        nd["confidence"] = float(nd.get("confidence", max(0.0, min(1.0, nd.get("importance", 0.5)))))
        nd["meta"] = nd.get("meta") or {"cluster": nd.get("cluster", "unclustered"), "tags": [], "version": SCHEMA_VERSION}
        nd["uncertainty"] = state["uncertainty"]
        nd["attention"] = state["attention"]
        nd["value"] = state["value"]
        nd["activation"] = state["activation"]
        nd["stability"] = state["stability"]
        nd["provenance"] = prov
        nd["node_type"] = nd.get("node_type") or nd.get("type")

    def _migrate_edge_to_v3(self, d: Dict[str, Any]):
        d.setdefault("id", str(uuid.uuid4()))
        d.setdefault("confidence", 0.5)
        d.setdefault("weight", 0.5)
        t = d.get("type", EdgeType.ASSOCIATED.value)
        if t in (EdgeType.SUPPORTS.value, EdgeType.ABOUT_MISSION.value, EdgeType.ANSWERS.value, EdgeType.ABSTRACTION_OF.value):
            pol = 1
        elif t == EdgeType.CONTRADICTS.value:
            pol = -1
        else:
            pol = 0
        d.setdefault("polarity", pol)
        d.setdefault("uncertainty_impact", 0.0)
        d.setdefault("provenance", {"evidence_ids": [], "created_at": d.get("created_at", time.time()), "updated_at": time.time()})

    def _serialize(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "saved_at": time.time(),
            "version": self.version,
            "graph": nx.node_link_data(self.graph),
            "mission": self.mission,
            "suspended_mission": self._suspended_mission,
            "mode": self._mode.value,
            "config": {"decay_rate": self.decay_rate, "scientificness": self.scientificness},
            "global_state": asdict(self.global_state),
            "state": {
                "cognitive_temperature": self.cognitive_temperature,
                "global_inhibition": self.global_inhibition,
                "exploration_bias": self.exploration_bias,
                "mission_focus": self.mission_focus,
                "last_novelty": self.last_novelty,
                "event_log": self.event_log,
            },
        }

    def save(self, path: str = "data/brain.json"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = self._serialize()
        with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path) or ".") as tf:
            json.dump(payload, tf, indent=2)
            tmp = tf.name
        os.replace(tmp, path)

    def load(self, path: str = "data/brain.json"):
        with open(path, "r") as f:
            raw = json.load(f)

        if "graph" in raw:
            try:
                self.graph = nx.node_link_graph(raw["graph"], multigraph=True)
            except TypeError:
                self.graph = nx.node_link_graph(raw["graph"])
                if not isinstance(self.graph, nx.MultiDiGraph):
                    mg = nx.MultiDiGraph()
                    mg.add_nodes_from(self.graph.nodes(data=True))
                    for u, v, d in self.graph.edges(data=True):
                        mg.add_edge(u, v, **d)
                    self.graph = mg
            self.mission = raw.get("mission")
            self._suspended_mission = raw.get("suspended_mission")
            cfg = raw.get("config", {})
            self.decay_rate = cfg.get("decay_rate", self.decay_rate)
            self.scientificness = cfg.get("scientificness", self.scientificness)
            try:
                self._mode = BrainMode(raw.get("mode", BrainMode.WANDERING.value))
            except ValueError:
                self._mode = BrainMode.WANDERING
            gs = raw.get("global_state") or {}
            self.global_state = GlobalStateV3(
                temperature=gs.get("temperature", raw.get("state", {}).get("cognitive_temperature", 0.7)),
                inhibition=gs.get("inhibition", raw.get("state", {}).get("global_inhibition", 0.0)),
                exploration_bias=gs.get("exploration_bias", raw.get("state", {}).get("exploration_bias", 0.5)),
                mission_vector_ref=gs.get("mission_vector_ref"),
                control_mode=gs.get("control_mode", self._mode.value),
                rng_seed=gs.get("rng_seed", 42),
                rng_cursor=gs.get("rng_cursor", 0),
                step_id=gs.get("step_id", 0),
                schema_version=gs.get("schema_version", raw.get("schema_version", SCHEMA_VERSION)),
            )
            state = raw.get("state", {})
            self.cognitive_temperature = state.get("cognitive_temperature", self.global_state.temperature)
            self.global_inhibition = state.get("global_inhibition", self.global_state.inhibition)
            self.exploration_bias = state.get("exploration_bias", self.global_state.exploration_bias)
            self.mission_focus = state.get("mission_focus", {})
            self.last_novelty = state.get("last_novelty", 0.0)
            self.event_log = state.get("event_log", [])
        else:
            self.graph = nx.MultiDiGraph(nx.node_link_graph(raw))

        for nid, nd in self.graph.nodes(data=True):
            self._migrate_node_to_v3(nid, nd)
        for _, _, _, d in self.graph.edges(keys=True, data=True):
            self._migrate_edge_to_v3(d)

        # normalize attention
        self.normalize_attention()

    def normalize_attention(self):
        vals = []
        for _, nd in self.graph.nodes(data=True):
            st = nd.setdefault("state", {})
            vals.append(max(0.0, st.get("attention", nd.get("attention", 0.0))))
        total = sum(vals)
        if total <= 1e-12 and len(vals) > 0:
            default = 1.0 / len(vals)
            for _, nd in self.graph.nodes(data=True):
                nd["state"]["attention"] = default
                nd["attention"] = default
        elif total > 0:
            i = 0
            for _, nd in self.graph.nodes(data=True):
                a = vals[i] / total
                nd["state"]["attention"] = a
                nd["attention"] = a
                i += 1
