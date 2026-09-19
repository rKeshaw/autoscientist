#!/usr/bin/env python3
"""
scripts/run_track4_mission.py

Track 4: Multi-Domain Discovery Generalization Runner
=====================================================
Executes autonomous multi-cycle discovery for diverse scientific domains:
- Mission 2: Systems & Synthetic Biology (Gene Regulatory Networks & Gillespie SSA)
- Mission 3: Discrete Information Theory (Non-Binary LDPC & Belief Propagation)

Runs in complete isolation under runs/track4_{mission_id}/ without touching
primary Mission 1 data in data/ and logs/.
"""

import os
import sys
import time
import json
import argparse
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from graph.brain import Brain, Node, Edge, EdgeType, EdgeSource, NodeType, NodeStatus, BrainMode
from observer.observer import Observer
from embedding_index import EmbeddingIndex
from embedding import embed as shared_embed
from insight_buffer import InsightBuffer
from critic.critic import Critic
from ingestion.ingestor import Ingestor
from dreamer.dreamer import Dreamer, DreamMode
from consolidator.consolidator import Consolidator
from notebook.notebook import Notebook
from reader.reader import Reader
from sandbox.sandbox import Sandbox
from thinker.thinker import Thinker
from llm_utils import unload_all_models


# ── Mission Specifications ───────────────────────────────────────────────────

MISSION_SPECS = {
    "mission2": {
        "title": "Systems & Synthetic Biology (Gene Regulatory Motifs)",
        "question": (
            "What topological network motifs in gene regulatory circuits minimize expression "
            "variance in the presence of extrinsic molecular noise while maintaining rapid switching times?"
        ),
        "domain": "systems_biology",
        "foundational_concepts": [
            {
                "statement": "Negative autoregulation (NAR) in gene transcription accelerates response time to steady state while shifting intrinsic transcriptional noise to higher frequencies.",
                "cluster": "gene_regulation",
                "importance": 0.85
            },
            {
                "statement": "Incoherent feedforward loops (type-1 IFFL) decouple steady-state protein expression from upstream extrinsic noise through non-monotonic pulses.",
                "cluster": "gene_regulation",
                "importance": 0.85
            },
            {
                "statement": "The Chemical Master Equation and Gillespie Stochastic Simulation Algorithm (SSA) govern discrete copy-number fluctuations in low-abundance transcription factors.",
                "cluster": "stochastic_kinetics",
                "importance": 0.80
            },
            {
                "statement": "Mutual repression circuits (toggle switches) exhibit bistable potential landscapes where switching latency scales exponentially with effective barrier height.",
                "cluster": "synthetic_biology",
                "importance": 0.80
            },
            {
                "statement": "The Fano factor (variance-to-mean ratio) of protein expression is bounded by metabolic ATP dissipation and transcriptional bursting frequency.",
                "cluster": "bioenergetics",
                "importance": 0.75
            }
        ],
        "foundational_hypotheses": [
            {
                "statement": "Negative autoregulation (NAR) suppresses low-frequency extrinsic transcriptional fluctuations more effectively than constitutive expression, lowering steady-state Fano factor at the cost of metabolic dissipation.",
                "predicted_answer": "NAR achieves lower protein copy-number variance under copy number noise while maintaining faster recovery to steady state.",
                "testable_by": "Gillespie Stochastic Simulation Algorithm (SSA) comparing constitutive vs negative autoregulatory transcription.",
                "importance": 0.85
            },
            {
                "statement": "A Type-1 Incoherent Feedforward Loop (I1-FFL) provides near-perfect adaptation to upstream gene dosage variations by generating a non-monotonic pulse that resets downstream expression.",
                "predicted_answer": "Steady-state downstream protein levels remain invariant across a 4-fold variation in upstream promoter activity in I1-FFL compared to simple cascade.",
                "testable_by": "Simulate I1-FFL chemical kinetics across varying input transcription rates using Gillespie SSA.",
                "importance": 0.85
            },
            {
                "statement": "In a genetic toggle switch with mutual repression, switching latency between stable states scales exponentially with the effective barrier height determined by Hill coefficient and repression cooperativity.",
                "predicted_answer": "Mean first-passage switching time increases exponentially as Hill coefficient n increases from 1 to 4 under intrinsic molecular noise.",
                "testable_by": "Stochastic simulation of bistable mutual repression toggle switch measuring mean first passage time vs Hill coefficient.",
                "importance": 0.80
            }
        ],
        "domain_articles": [
            {"title": "Gene regulatory network", "url": "https://en.wikipedia.org/wiki/Gene_regulatory_network"},
            {"title": "Gillespie algorithm", "url": "https://en.wikipedia.org/wiki/Gillespie_algorithm"},
            {"title": "Network motif", "url": "https://en.wikipedia.org/wiki/Network_motif"},
            {"title": "Cellular noise", "url": "https://en.wikipedia.org/wiki/Cellular_noise"},
            {"title": "Genetic toggle switch", "url": "https://en.wikipedia.org/wiki/Genetic_toggle_switch"},
            {"title": "Chemical master equation", "url": "https://en.wikipedia.org/wiki/Master_equation#Chemical_master_equation"},
            {"title": "Operon", "url": "https://en.wikipedia.org/wiki/Operon"}
        ],
        "seed_questions": [
            "How does transcriptional bursting frequency affect the Fano factor in negative autoregulatory loops?",
            "Can an incoherent feedforward loop achieve near-zero noise amplification under extrinsic plasmid copy number fluctuations?",
            "What is the mathematical Pareto frontier between response time tau and expression variance in bistable switches?"
        ]
    },
    "mission3": {
        "title": "Discrete Information Theory & Complexity (Non-Binary LDPC)",
        "question": (
            "Can non-binary LDPC (low-density parity-check) codes maintain channel capacity "
            "under asymmetric burst-noise channels without exponential decoding latency?"
        ),
        "domain": "information_theory",
        "foundational_concepts": [
            {
                "statement": "Non-binary LDPC codes over Galois fields GF(q) approach Shannon channel capacity significantly closer than binary LDPC for moderate block lengths.",
                "cluster": "error_correction",
                "importance": 0.85
            },
            {
                "statement": "Belief Propagation (sum-product algorithm) on non-binary Tanner graphs achieves O(q log q) per-check complexity via Fast Fourier Transforms (FFT-BP).",
                "cluster": "decoding_algorithms",
                "importance": 0.85
            },
            {
                "statement": "The Gilbert-Elliott two-state Markov chain models asymmetric burst-noise channels with memory between good and bad transmission states.",
                "cluster": "channel_models",
                "importance": 0.80
            },
            {
                "statement": "Tanner graph girth and cycle trapping sets determine the error flare floor in message-passing decoding iterations.",
                "cluster": "graph_codes",
                "importance": 0.80
            },
            {
                "statement": "Decoding energy dissipation and iterative convergence time represent fundamental thermodynamic constraints on high-throughput communications.",
                "cluster": "algorithmic_physics",
                "importance": 0.75
            }
        ],
        "foundational_hypotheses": [
            {
                "statement": "Non-binary LDPC decoding over Galois fields GF(q) with q >= 4 reduces the number of belief propagation iterations required to achieve zero syndrome compared to binary LDPC (q=2) on correlated burst-noise Markov channels.",
                "predicted_answer": "Average BP decoding iterations decrease monotonically as field order q increases from 2 to 8 on Gilbert-Elliott burst channels.",
                "testable_by": "Simulate belief propagation decoding across field sizes q in [2, 4, 8] over a 2-state Gilbert-Elliott channel.",
                "importance": 0.85
            },
            {
                "statement": "Fast Fourier Transform Belief Propagation (FFT-BP) over Galois fields GF(q) bounds check-node update complexity to O(q log q), preventing exponential decoding latency growth under high-order field decoding.",
                "predicted_answer": "Execution time per check-node update scales quasi-linearly with q log2(q) rather than quadratic O(q^2).",
                "testable_by": "Benchmark execution runtime of FFT-BP check node convolution across increasing Galois field orders q.",
                "importance": 0.85
            },
            {
                "statement": "Tanner graphs designed with girth g >= 6 eliminate dominant 4-cycle trapping sets, significantly lowering the error floor under asymmetric burst-noise conditions.",
                "predicted_answer": "Bit-error rate in the high-SNR / moderate burst regime exhibits a steeper waterfall slope when 4-cycles are removed from the parity-check matrix H.",
                "testable_by": "Compare Tanner graph message passing decoding error rates between girth-4 and girth-6 parity check matrices under burst errors.",
                "importance": 0.80
            }
        ],
        "domain_articles": [
            {"title": "Low-density parity-check code", "url": "https://en.wikipedia.org/wiki/Low-density_parity-check_code"},
            {"title": "Belief propagation", "url": "https://en.wikipedia.org/wiki/Belief_propagation"},
            {"title": "Tanner graph", "url": "https://en.wikipedia.org/wiki/Tanner_graph"},
            {"title": "Gilbert–Elliott model", "url": "https://en.wikipedia.org/wiki/Gilbert%E2%80%93Elliott_model"},
            {"title": "Finite field", "url": "https://en.wikipedia.org/wiki/Finite_field"},
            {"title": "Error floor", "url": "https://en.wikipedia.org/wiki/Error_floor"},
            {"title": "Channel capacity", "url": "https://en.wikipedia.org/wiki/Channel_capacity"}
        ],
        "seed_questions": [
            "How does field order q affect the decoding convergence rate of belief propagation under bursty channels?",
            "Can non-binary check matrix structures eliminate small trapping sets in finite-length LDPC codes?",
            "What is the optimal trade-off between parity-check density and decoding latency under asymmetric Gilbert-Elliott noise?"
        ]
    }
}


# ── Mission Runner Class ──────────────────────────────────────────────────────

class Track4MissionRunner:
    def __init__(self, mission_id: str, run_dir: str = None, loops: int = 2):
        if mission_id not in MISSION_SPECS:
            raise ValueError(f"Unknown mission_id '{mission_id}'. Expected one of {list(MISSION_SPECS.keys())}")
        
        self.spec = MISSION_SPECS[mission_id]
        self.mission_id = mission_id
        self.loops = loops
        self.run_dir = run_dir or os.path.join(REPO_ROOT, f"runs/track4_{mission_id}")
        self.data_dir = os.path.join(self.run_dir, "data")
        self.logs_dir = os.path.join(self.run_dir, "logs")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.brain_path = os.path.join(self.data_dir, "brain.json")
        self.obs_path = os.path.join(self.data_dir, "observer.json")
        self.idx_path = os.path.join(self.data_dir, "embedding_index")
        self.buf_path = os.path.join(self.data_dir, "insight_buffer.json")
        self.note_path = os.path.join(self.data_dir, "notebook.json")
        self.rl_path = os.path.join(self.data_dir, "reading_list.json")
        self.sb_log_path = os.path.join(self.logs_dir, "sandbox_log.json")
        self.cycle_log_path = os.path.join(self.logs_dir, "cycle_log.json")

        self._init_components()

    def _init_components(self):
        print(f"[{self.mission_id.upper()}] Initializing isolated components in: {self.run_dir}")
        self.brain = Brain()
        if os.path.exists(self.brain_path):
            self.brain.load(self.brain_path)

        self.observer = Observer(self.brain)
        if os.path.exists(self.obs_path):
            self.observer.load(self.obs_path)

        if os.path.exists(self.idx_path + ".faiss"):
            self.emb_index = EmbeddingIndex.load(self.idx_path)
        else:
            self.emb_index = EmbeddingIndex.build_from_brain(self.brain, shared_embed)

        self.insight_buffer = InsightBuffer(self.brain, embedding_index=self.emb_index, path=self.buf_path)
        self.critic = Critic(self.brain, embedding_index=self.emb_index, insight_buffer=self.insight_buffer)
        self.insight_buffer.critic = self.critic

        self.ingestor = Ingestor(self.brain, research_agenda=self.observer,
                                 embedding_index=self.emb_index, insight_buffer=self.insight_buffer)
        self.dreamer = Dreamer(self.brain, research_agenda=self.observer, critic=self.critic)
        self.consolidator = Consolidator(self.brain, observer=self.observer,
                                         embedding_index=self.emb_index, insight_buffer=self.insight_buffer)
        self.notebook = Notebook(self.brain, observer=self.observer, path=self.note_path)
        self.reader = Reader(self.brain, observer=self.observer, notebook=self.notebook,
                             ingestor=self.ingestor, reading_list_path=self.rl_path)
        self.sandbox = Sandbox(self.brain, observer=self.observer, embedding_index=self.emb_index,
                               log_path=self.sb_log_path)
        self.thinker = Thinker(self.brain, observer=self.observer, embedding_index=self.emb_index,
                               critic=self.critic, sandbox=self.sandbox)

    def bootstrap(self):
        """Seed foundational domain concepts, hypotheses, reading list, and absorb initial literature."""
        if len(self.brain.graph.nodes) > 1:
            print(f"[{self.mission_id.upper()}] Brain already bootstrapped with {len(self.brain.graph.nodes)} nodes.")
            return

        print(f"\n" + "=" * 70)
        print(f"[{self.mission_id.upper()}] BOOTSTRAPPING MISSION: {self.spec['title']}")
        print("=" * 70)

        # 1. Set central mission
        mission_nid = self.brain.set_mission(self.spec["question"], f"Domain: {self.spec['domain']}")
        self.emb_index.add(mission_nid, shared_embed(self.spec["question"]))

        # 2. Add foundational concepts
        created_nodes = []
        for c in self.spec["foundational_concepts"]:
            node = Node(
                statement=c["statement"],
                node_type=NodeType.CONCEPT,
                cluster=c["cluster"],
                status=NodeStatus.SETTLED,
                importance=c["importance"],
                source_quality=0.9
            )
            nid = self.brain.add_node(node)
            self.emb_index.add(nid, shared_embed(c["statement"]))
            self.brain.link_to_mission(nid, f"Foundational axiom for {self.spec['domain']}", strength=0.8)
            created_nodes.append(nid)

        # Connect foundational concepts into initial backbone
        for i in range(len(created_nodes) - 1):
            u, v = created_nodes[i], created_nodes[i+1]
            e = Edge(
                type=EdgeType.ASSOCIATED,
                narration=f"Foundational inter-relation in {self.spec['domain']}",
                weight=0.75,
                confidence=0.85,
                source=EdgeSource.CONSOLIDATION
            )
            self.brain.add_edge(u, v, e)

        # 3. Add foundational testable hypotheses directly into graph
        for h in self.spec.get("foundational_hypotheses", []):
            hnode = Node(
                statement=h["statement"],
                node_type=NodeType.HYPOTHESIS,
                cluster=self.spec["domain"],
                status=NodeStatus.UNCERTAIN,
                importance=h["importance"],
                predicted_answer=h.get("predicted_answer", ""),
                testable_by=h.get("testable_by", ""),
                source_quality=0.85
            )
            hnid = self.brain.add_node(hnode)
            self.emb_index.add(hnid, shared_embed(h["statement"]))
            self.brain.link_to_mission(hnid, f"Core testable hypothesis for {self.spec['domain']}", strength=0.85)
            # connect hypothesis to related foundational concepts
            if created_nodes:
                self.brain.add_edge(hnid, created_nodes[0], Edge(
                    type=EdgeType.ASSOCIATED,
                    narration=f"Hypothesis addresses mechanism in {self.spec['domain']}",
                    weight=0.8,
                    confidence=0.85,
                    source=EdgeSource.CONSOLIDATION
                ))

        # 4. Seed observer agenda
        for q in self.spec["seed_questions"]:
            item = self.observer.add_to_agenda(text=q, item_type="question")
            if item:
                item.priority = 0.85

        # 5. Seed domain literature reading list
        for art in self.spec.get("domain_articles", []):
            self.reader.add_to_list(
                url=art["url"],
                title=art["title"],
                source_type="wikipedia",
                priority=0.85,
                added_by="bootstrap",
                reason=f"Foundational literature for {self.spec['domain']}"
            )

        # 6. Absorb initial batch of 2 foundational articles
        print(f"\n[{self.mission_id.upper()}] Absorbing foundational domain literature (first 2 articles)...")
        for entry in self.reader.get_unread(2):
            try:
                res = self.reader.absorb_entry(entry)
                if res.success:
                    print(f"  ✓ Absorbed '{res.title}': {res.node_count} concepts extracted.")
                else:
                    print(f"  ✗ Failed absorbing '{entry.title}': {res.error}")
            except Exception as e:
                print(f"  ✗ Error absorbing '{entry.title}': {e}")
            time.sleep(1)

        self._save_state()
        print(f"[{self.mission_id.upper()}] Bootstrap complete: {len(self.brain.graph.nodes)} nodes, {len(self.brain.graph.edges)} edges.")

    def run_cycle(self, cycle_num: int):
        print("\n" + "#" * 80)
        print(f"[{self.mission_id.upper()}] EXECUTING DIURNAL CYCLE {cycle_num} / {self.loops}")
        print("#" * 80)

        # Phase 1: Morning Dream Cycle + NREM
        print(f"\n--- [Cycle {cycle_num}: Phase 1/6] Associative Dream & NREM ---")
        try:
            dream_log = self.dreamer.dream(
                mode=DreamMode.WANDERING,
                steps=15,
                run_nrem=True,
                log_path=os.path.join(self.logs_dir, f"dream_cycle_{cycle_num}.json")
            )
            self.observer.observe(dream_log)
            self.notebook.write_morning_entry(dream_log, cycle=cycle_num)
        except Exception as e:
            print(f"  [Dream Error]: {e}")
            traceback.print_exc()

        # Phase 2: Domain Literature Reading & Ingestion
        print(f"\n--- [Cycle {cycle_num}: Phase 2/6] Domain Literature Reading & Ingestion ---")
        try:
            reading_results = self.reader.reading_day(max_items=1)
            absorbed = sum(1 for r in reading_results if r.success)
            print(f"  Reader absorbed {absorbed}/{len(reading_results)} texts from reading list.")
        except Exception as e:
            print(f"  [Reading Error]: {e}")
            traceback.print_exc()

        # Phase 3: Deliberate Thinking
        print(f"\n--- [Cycle {cycle_num}: Phase 3/6] Deliberate Thinking & Deconstruction ---")
        try:
            think_log = self.thinker.think()
            print(f"  Thinking pattern: {think_log.pattern} | Insight: {think_log.insight[:80]}...")
        except Exception as e:
            print(f"  [Thinker Error]: {e}")
            traceback.print_exc()

        # Phase 4: Empirical Sandbox Experimentation
        print(f"\n--- [Cycle {cycle_num}: Phase 4/6] Empirical Hypothesis Testing & Simulation ---")
        try:
            sb_results = self.sandbox.scan_and_test(max_tests=2)
            print(f"  Sandbox executed {len(sb_results)} empirical simulations.")
            for r in sb_results:
                print(f"    * Verdict: {r.verdict.upper()} (conf={r.confidence:.2f}) -> {r.hypothesis[:60]}...")
        except Exception as e:
            print(f"  [Sandbox Error]: {e}")
            traceback.print_exc()

        # Phase 5: Evening Sleep Consolidation (6 Passes)
        print(f"\n--- [Cycle {cycle_num}: Phase 5/6] 6-Pass Sleep Consolidation ---")
        try:
            cons_report = self.consolidator.consolidate(
                save_path=os.path.join(self.logs_dir, f"consolidation_cycle_{cycle_num}.json")
            )
            self.notebook.write_evening_entry(cons_report, cycle=cycle_num)
            print(f"  Consolidation complete: {cons_report.merges} merges, {cons_report.syntheses} syntheses, {cons_report.gaps} gaps.")
        except Exception as e:
            print(f"  [Consolidation Error]: {e}")
            traceback.print_exc()

        # Phase 6: Scientific Reflection & Hypothesis Synthesis
        print(f"\n--- [Cycle {cycle_num}: Phase 6/6] Notebook Running Hypothesis Update ---")
        try:
            hyp = self.notebook.update_running_hypothesis(cycle=cycle_num)
            print(f"  Running hypothesis updated ({len(hyp)} chars).")
        except Exception as e:
            print(f"  [Notebook Error]: {e}")
            traceback.print_exc()

        self._record_cycle_log(cycle_num)
        self._save_state()

        stats = self.brain.stats()
        print(f"\n[{self.mission_id.upper()}] End of Cycle {cycle_num}: {stats['nodes']} nodes, {stats['edges']} edges, {stats['clusters']} clusters, {stats['empirical']} empirical nodes.")

    def _record_cycle_log(self, cycle_num: int):
        try:
            entries = []
            if os.path.exists(self.cycle_log_path):
                with open(self.cycle_log_path) as f:
                    entries = json.load(f).get("entries", [])
            entries.append({
                "cycle": cycle_num,
                "timestamp": time.time(),
                "nodes": len(self.brain.graph.nodes),
                "edges": len(self.brain.graph.edges),
                "clusters": len(set(d.get("cluster", "") for _, d in self.brain.graph.nodes(data=True))),
                "empirical_nodes": len(self.brain.nodes_by_type(NodeType.EMPIRICAL)),
                "agenda_items": len(self.observer.agenda)
            })
            with open(self.cycle_log_path, "w") as f:
                json.dump({"entries": entries}, f, indent=2)
        except Exception as e:
            print(f"  [CycleLog Warning]: {e}")

    def _save_state(self):
        self.brain.save(self.brain_path)
        self.observer.save(self.obs_path)
        self.emb_index.sync_with_brain(self.brain, shared_embed)
        self.emb_index.save(self.idx_path)
        self.insight_buffer.save()
        self.notebook.save()

    def run(self):
        print("=" * 80)
        print(f"STARTING TRACK 4 AUTONOMOUS MISSION: {self.mission_id.upper()}")
        print(f"Question: {self.spec['question']}")
        print(f"Loops to run: {self.loops}")
        print(f"Run Directory: {self.run_dir}")
        print("=" * 80)

        self.bootstrap()
        for c in range(1, self.loops + 1):
            self.run_cycle(c)

        print("\n" + "=" * 80)
        print(f"[{self.mission_id.upper()}] ALL {self.loops} CYCLES COMPLETE.")
        print("=" * 80)
        unload_all_models()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track 4 Mission Runner")
    parser.add_argument("--mission-id", type=str, required=True, choices=["mission2", "mission3"],
                        help="Mission identifier (mission2=Systems Bio, mission3=Information Theory)")
    parser.add_argument("--loops", type=int, default=2, help="Number of diurnal cycles to execute")
    parser.add_argument("--run-dir", type=str, default=None, help="Custom isolated run directory")
    args = parser.parse_args()

    runner = Track4MissionRunner(mission_id=args.mission_id, run_dir=args.run_dir, loops=args.loops)
    runner.run()
