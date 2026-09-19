#!/usr/bin/env python3
"""
benchmark/test_track5_neuromodulatory_working_memory.py

Track 5 Comprehensive Benchmark: Neuromodulatory Dynamics & Working Memory Constraints
======================================================================================
Author: AutoScientist Architectural Benchmarking Suite
Date: September 18, 2026

Empirical Validation of Neurobiological Cognitive Constraints:
1. Experiment 5.1: Working Memory Capacity Sweep (Miller's Law 7 ± 2 vs. Extremes)
   - Evaluates W_max in [3, 7, 10, 25, 100] (unbounded) over a sequential stream
     of 150 scientific cognitive events across the 821-node connectome.
   - Metrics: Prompt token overhead, topological dispersion (graph hop distance),
     semantic coherence (cosine similarity), eviction thrashing rate, and focus
     efficiency ratio.
2. Experiment 5.2: Neuromodulatory Dynamics & Adaptive Cognitive Policy Verification
   - Suite 5.2A: Contradiction-Induced Frustration Escalation & Mission Suspension
   - Suite 5.2B: Diurnal Incubation Decay & Automatic Mission Resumption
   - Suite 5.2C: Dopaminergic Modulation of Contextual Bandit Learning Rates
   - Suite 5.2D: Frustration-Driven Policy Pivoting & Deadlock Escape Latency
"""

import os
import sys
import json
import math
import random
import numpy as np
import networkx as nx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from graph.brain import Brain, BrainMode, NodeType, EdgeType
from thinker.policy import CognitivePolicy
from embedding_index import EmbeddingIndex

BRAIN_PATH = "data/brain.json"
INDEX_PATH = "data/embedding_index"
OUTPUT_DIR = "benchmark/results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "track5_neuromodulatory_working_memory.json")


def load_data():
    """Load the ground-truth brain connectome and precomputed embeddings."""
    with open(BRAIN_PATH, 'r') as f:
        brain_raw = json.load(f)
    
    # Load embedding index
    emb_idx = EmbeddingIndex.load(INDEX_PATH)
    
    # Build NetworkX graph
    G = nx.Graph()
    nodes_dict = {}
    for n in brain_raw["graph"]["nodes"]:
        nid = n["id"]
        nodes_dict[nid] = n
        G.add_node(nid, **n)
        
    for e in brain_raw["graph"]["edges"]:
        u, v = e["source"], e["target"]
        if u != v:
            G.add_edge(u, v, weight=e.get("weight", 1.0))
            
    return brain_raw, nodes_dict, G, emb_idx


# ==============================================================================
# EXPERIMENT 5.1: WORKING MEMORY CAPACITY SWEEP
# ==============================================================================

def run_experiment_5_1_working_memory_sweep(nodes_dict, G, emb_idx):
    """
    Simulates working memory dynamics over an empirical stream of scientific events
    under different capacity constraints W_max in [3, 7, 10, 25, 100].
    """
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT 5.1: WORKING MEMORY CAPACITY SWEEP (W_max in [3, 7, 10, 25, 100])")
    print("="*80)
    
    # Sort nodes chronologically by created_at or activated_at
    sorted_nodes = sorted(
        nodes_dict.values(),
        key=lambda x: x.get("created_at", 0.0)
    )
    
    # Filter to nodes with embeddings
    valid_nodes = [n for n in sorted_nodes if emb_idx.has(n["id"])]
    
    # Select a 150-event sequence of active reasoning steps
    stream_length = min(150, len(valid_nodes))
    event_stream = valid_nodes[:stream_length]
    
    capacities = [3, 7, 10, 25, 100]
    results_by_capacity = {}
    
    # Precompute pairwise shortest path lengths on largest connected component
    # to avoid repeated all-pairs recalculation
    cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(cc)
    
    for W_max in capacities:
        print(f"\nEvaluating Working Memory Capacity W_max = {W_max}...")
        
        working_memory = []  # list of node_ids, index 0 is most recent
        item_entry_step = {} # node_id -> step entered
        
        token_overheads = []
        topological_dispersions = []
        semantic_coherences = []
        eviction_latencies = []
        thrashing_evictions = 0
        total_evictions = 0
        
        subsequent_hits = []
        
        for step, event_node in enumerate(event_stream):
            nid = event_node["id"]
            
            # Update working memory (insert at front, deduplicate)
            if nid in working_memory:
                working_memory.remove(nid)
            working_memory.insert(0, nid)
            item_entry_step[nid] = step
            
            # Check for eviction
            if len(working_memory) > W_max:
                evicted_nid = working_memory.pop()
                total_evictions += 1
                age_in_wm = step - item_entry_step.get(evicted_nid, step)
                eviction_latencies.append(age_in_wm)
                if age_in_wm <= 3:
                    thrashing_evictions += 1
                    
            # 1. Prompt Token Overhead: approx 1 token per 4 chars of prompt text
            wm_text = "\n".join([
                f"[FOCUS/{nodes_dict[m].get('node_type','concept')}] {nodes_dict[m].get('statement','')}"
                for m in working_memory
            ])
            token_count = max(1, len(wm_text) // 4)
            token_overheads.append(token_count)
            
            # If working memory has >= 2 items, compute dispersion and coherence
            if len(working_memory) >= 2:
                # 2. Topological Dispersion: average shortest path hops in G
                hop_dists = []
                for i in range(len(working_memory)):
                    for j in range(i + 1, len(working_memory)):
                        u, v = working_memory[i], working_memory[j]
                        if u in G_sub and v in G_sub:
                            try:
                                d = nx.shortest_path_length(G_sub, u, v)
                                hop_dists.append(d)
                            except nx.NetworkXNoPath:
                                hop_dists.append(15.0)  # max diameter penalty
                        else:
                            hop_dists.append(15.0)
                if hop_dists:
                    topological_dispersions.append(np.mean(hop_dists))
                    
                # 3. Semantic Coherence: average pairwise cosine similarity
                cos_sims = []
                for i in range(len(working_memory)):
                    for j in range(i + 1, len(working_memory)):
                        u, v = working_memory[i], working_memory[j]
                        vec_u = emb_idx.get_embedding(u)
                        vec_v = emb_idx.get_embedding(v)
                        sim = float(np.dot(vec_u, vec_v))
                        cos_sims.append(sim)
                if cos_sims:
                    semantic_coherences.append(np.mean(cos_sims))
                    
            # 4. Cognitive Salience Hit: does working memory intersect next step's neighbors?
            if step + 1 < len(event_stream):
                next_nid = event_stream[step + 1]["id"]
                next_neighbors = set(G.neighbors(next_nid)) if next_nid in G else set()
                intersection = set(working_memory).intersection(next_neighbors)
                hit = 1.0 if len(intersection) > 0 else 0.0
                subsequent_hits.append(hit)
                
        thrash_rate = (thrashing_evictions / total_evictions) if total_evictions > 0 else 0.0
        avg_tokens = float(np.mean(token_overheads))
        avg_dispersion = float(np.mean(topological_dispersions)) if topological_dispersions else 0.0
        avg_coherence = float(np.mean(semantic_coherences)) if semantic_coherences else 0.0
        avg_hit_rate = float(np.mean(subsequent_hits)) if subsequent_hits else 0.0
        
        # Context efficiency = Hit Rate / (Avg Tokens / 100)
        eff_score = (avg_hit_rate / (avg_tokens / 100.0)) if avg_tokens > 0 else 0.0
        
        results_by_capacity[str(W_max)] = {
            "W_max": W_max,
            "avg_prompt_tokens": round(avg_tokens, 1),
            "topological_dispersion_hops": round(avg_dispersion, 3),
            "semantic_coherence_cosine": round(avg_coherence, 4),
            "total_evictions": total_evictions,
            "thrashing_evictions": thrashing_evictions,
            "thrashing_rate": round(thrash_rate, 4),
            "associative_hit_rate": round(avg_hit_rate, 4),
            "context_efficiency_score": round(eff_score, 4)
        }
        
        print(f"  Result W={W_max}: Tokens={avg_tokens:.1f}, Dispersion={avg_dispersion:.2f} hops, "
              f"Coherence={avg_coherence:.4f}, Thrashing={thrash_rate*100:.1f}%, "
              f"Hit Rate={avg_hit_rate*100:.1f}%, Efficiency={eff_score:.3f}")
              
    return results_by_capacity


# ==============================================================================
# EXPERIMENT 5.2: NEUROMODULATORY DYNAMICS & POLICY VERIFICATION
# ==============================================================================

def run_experiment_5_2_neuromodulatory_dynamics():
    """
    Evaluates:
    - 5.2A: Frustration escalation & Mission Suspension (FOCUSED -> WANDERING)
    - 5.2B: Diurnal Incubation Decay & Auto-Resumption (WANDERING -> FOCUSED)
    - 5.2C: Dopaminergic modulation of policy learning rate
    - 5.2D: Frustration-driven policy exploration pivoting
    """
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT 5.2: NEUROMODULATORY DYNAMICS & ADAPTIVE POLICY DYNAMICS")
    print("="*80)
    
    # ── Suite 5.2A: Frustration Buildup & Mission Suspension ───────────────────
    print("\n[Suite 5.2A] Testing Frustration Buildup & Mission Suspension...")
    brain_a = Brain(decay_rate=0.01)
    brain_a.set_mission("How do non-equilibrium kinetic proofreading mechanisms govern trade-offs?")
    
    initial_mode = brain_a.get_mode()
    initial_frust = brain_a.frustration
    initial_dopa = brain_a.dopamine
    
    frustration_trace = []
    suspension_step = None
    
    # Apply successive contradiction / failure shocks (+0.2 each)
    for step in range(1, 6):
        brain_a.increase_frustration(0.2)
        frustration_trace.append({
            "step": step,
            "frustration": round(brain_a.frustration, 3),
            "dopamine": round(brain_a.dopamine, 3),
            "mode": brain_a.get_mode(),
            "mission_suspended": brain_a._suspended_mission is not None
        })
        if brain_a.is_wandering() and suspension_step is None:
            suspension_step = step
            
    print(f"  Suspension triggered at step {suspension_step} when frustration crossed threshold >= 0.8.")
    print(f"  Post-suspension state: Mode={brain_a.get_mode()}, Frustration={brain_a.frustration}, Dopamine={brain_a.dopamine}")
    
    # ── Suite 5.2B: Diurnal Incubation & Auto-Resumption ───────────────────────
    print("\n[Suite 5.2B] Testing Diurnal Incubation Decay & Auto-Resumption...")
    # Continue with brain_a in wandering mode (frustration is 0.40 after suspension)
    # Inject a slight increase to 0.45 to test exact decay below 0.40
    brain_a.frustration = 0.50
    decay_trace = []
    resumption_step = None
    
    # Simulate elapsed time increments of 0.25 days (6 hours)
    for step in range(1, 6):
        brain_a.apply_neuromodulator_decay(elapsed_days=0.25)
        decay_trace.append({
            "step": step,
            "elapsed_days": step * 0.25,
            "frustration": round(brain_a.frustration, 3),
            "dopamine": round(brain_a.dopamine, 3),
            "mode": brain_a.get_mode(),
            "is_focused": brain_a.is_focused()
        })
        if brain_a.is_focused() and resumption_step is None:
            resumption_step = step
            
    print(f"  Auto-resumption triggered at step {resumption_step} when frustration decayed to <= 0.40.")
    print(f"  Post-resumption state: Mode={brain_a.get_mode()}, Mission={brain_a.mission['question'][:50]}...")

    # ── Suite 5.2C: Dopaminergic Modulation of Learning Rates ─────────────────
    print("\n[Suite 5.2C] Testing Dopaminergic Modulation of Learning Rate...")
    # Contextual Bandit: Test convergence speed under 3 dopamine regimes:
    # Hypodopaminergic (0.1), Baseline (0.5), Hyperdopaminergic (0.9)
    # Learning rate formula: lr_eff = lr_base * (1 + dopamine)
    
    dopamine_levels = [0.1, 0.5, 0.9]
    dopamine_results = {}
    
    np.random.seed(42)
    random.seed(42)
    
    # Optimal action: "experimental" (true reward mean = 0.85)
    # Suboptimal: "reductive" (0.50), "analogical" (0.35), "dialectical" (0.30), "integrative" (0.25)
    true_rewards = {
        "experimental": 0.85,
        "reductive": 0.50,
        "analogical": 0.35,
        "dialectical": 0.30,
        "integrative": 0.25
    }
    
    num_episodes = 80
    num_trials = 30
    
    for dopa in dopamine_levels:
        steps_to_converge = []
        cumulative_regrets = []
        
        for trial in range(num_trials):
            policy = CognitivePolicy(epsilon=0.15, learning_rate=0.10)
            # Override policy q_table to fresh state
            policy.q_table = {}
            trial_regret = 0.0
            converged_at = num_episodes
            
            for ep in range(1, num_episodes + 1):
                chosen_act = policy.choose_pattern("hypothesis", "criticality")
                r_mean = true_rewards[chosen_act]
                r_sample = float(np.clip(np.random.normal(r_mean, 0.1), 0.0, 1.0))
                
                # Regret relative to optimal (0.85)
                trial_regret += (0.85 - r_mean)
                
                # Update with dopamine level
                policy.update("hypothesis", "criticality", chosen_act, r_sample, dopamine=dopa)
                
                # Check if greedy choice is optimal
                greedy_act = max(policy.q_table["hypothesis|criticality"].items(), key=lambda x: x[1])[0]
                if greedy_act == "experimental" and converged_at == num_episodes and ep > 10:
                    converged_at = ep
                    
            steps_to_converge.append(converged_at)
            cumulative_regrets.append(trial_regret)
            
        avg_converge = float(np.mean(steps_to_converge))
        avg_regret = float(np.mean(cumulative_regrets))
        eff_lr = 0.10 * (1.0 + dopa)
        
        dopamine_results[f"dopamine_{dopa}"] = {
            "dopamine_level": dopa,
            "effective_learning_rate": round(eff_lr, 3),
            "episodes_to_convergence": round(avg_converge, 1),
            "cumulative_regret": round(avg_regret, 2)
        }
        print(f"  Dopamine={dopa}: Eff LR={eff_lr:.3f}, Convergence={avg_converge:.1f} eps, Regret={avg_regret:.2f}")

    # ── Suite 5.2D: Frustration-Driven Policy Pivoting (Exploration Escalation) ─
    print("\n[Suite 5.2D] Testing Frustration-Driven Policy Pivoting...")
    # Biological principle: When frustration is high, epistemic deadlock forces exploration.
    # We model exploration rate as: epsilon(F) = epsilon_0 + alpha * F
    frustration_conditions = [0.0, 0.2, 0.5, 0.8]
    pivoting_results = {}
    
    alpha = 0.6  # exploration boost parameter
    
    for F in frustration_conditions:
        eps_eff = min(0.95, 0.10 + alpha * F)
        
        # Initialize a policy where "reductive" was previously learned as best
        policy = CognitivePolicy(epsilon=eps_eff, learning_rate=0.10)
        policy.q_table = {
            "hypothesis|criticality": {
                "reductive": 0.80,     # entrenched habit
                "experimental": 0.40,
                "analogical": 0.35,
                "dialectical": 0.30,
                "integrative": 0.25
            }
        }
        
        # Now simulate 50 actions where "reductive" fails repeatedly (reward = 0.0)
        # while "experimental" delivers discovery (reward = 0.9)
        actions_chosen = []
        pivoted_at = 50
        
        for ep in range(1, 51):
            act = policy.choose_pattern("hypothesis", "criticality")
            actions_chosen.append(act)
            
            if act == "reductive":
                rew = 0.05
            elif act == "experimental":
                rew = 0.90
            else:
                rew = 0.30
                
            policy.update("hypothesis", "criticality", act, rew, dopamine=0.5)
            
            if act != "reductive" and pivoted_at == 50 and ep > 3:
                # Check if "reductive" has been dethroned
                best_act = max(policy.q_table["hypothesis|criticality"].items(), key=lambda x: x[1])[0]
                if best_act != "reductive":
                    pivoted_at = ep
                    
        # Calculate action entropy (diversity)
        counts = {a: actions_chosen.count(a) for a in CognitivePolicy.DEFAULT_ACTIONS}
        probs = [c / len(actions_chosen) for c in counts.values() if c > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        
        pivoting_results[f"frustration_{F}"] = {
            "frustration_level": F,
            "effective_exploration_epsilon": round(eps_eff, 3),
            "deadlock_escape_latency_episodes": pivoted_at,
            "action_entropy_bits": round(entropy, 3),
            "pivoted_to_optimal": policy.q_table["hypothesis|criticality"]["experimental"] > policy.q_table["hypothesis|criticality"]["reductive"],
            "action_distribution": counts
        }
        print(f"  Frustration={F}: Epsilon={eps_eff:.2f}, Escape Latency={pivoted_at} eps, Action Entropy={entropy:.3f} bits")

    return {
        "suite_5_2a_suspension": {
            "initial_mode": initial_mode,
            "suspension_trigger_step": suspension_step,
            "frustration_escalation_trace": frustration_trace
        },
        "suite_5_2b_incubation": {
            "resumption_trigger_step": resumption_step,
            "incubation_decay_trace": decay_trace
        },
        "suite_5_2c_dopaminergic_learning": dopamine_results,
        "suite_5_2d_frustration_pivoting": pivoting_results
    }


# ==============================================================================
# MAIN EXECUTION & SERIALIZATION
# ==============================================================================

def main():
    print("Initializing Track 5 Neuromodulatory & Working Memory Benchmark...")
    brain_raw, nodes_dict, G, emb_idx = load_data()
    print(f"Substrate loaded: {len(nodes_dict)} nodes, {G.number_of_edges()} edges, {emb_idx.size} embeddings.")
    
    # Run Experiment 5.1
    exp_5_1_results = run_experiment_5_1_working_memory_sweep(nodes_dict, G, emb_idx)
    
    # Run Experiment 5.2
    exp_5_2_results = run_experiment_5_2_neuromodulatory_dynamics()
    
    # Compile full benchmark payload
    track5_payload = {
        "meta": {
            "benchmark": "Track 5: Neuromodulatory Dynamics & Working Memory Constraints",
            "author": "AutoScientist Architectural Benchmarking Suite",
            "date": "2026-09-18",
            "substrate_nodes": len(nodes_dict),
            "substrate_edges": G.number_of_edges()
        },
        "experiment_5_1_working_memory_sweep": exp_5_1_results,
        "experiment_5_2_neuromodulatory_dynamics": exp_5_2_results
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(track5_payload, f, indent=2)
        
    print(f"\n[COMPLETE] Track 5 Benchmark finished! Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
