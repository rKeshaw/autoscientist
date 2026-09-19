"""
Thinker — Deliberate, goal-directed reasoning for THE SCIENTIST.

Unlike the Dreamer (random walks, serendipity), the Thinker does structured,
convergent reasoning — working toward *answers*, not just associations.

Thinking patterns:
  1. Dialectical   — evidence for/against, then synthesis
  2. Analogical    — transfer solution from analogous domain
  3. Reductive     — break question into simpler sub-questions
  4. Experimental  — thought experiments: "If X, then we'd expect Y"
  5. Integrative   — combine ideas into a unifying principle

Usage:
    thinker = Thinker(brain, observer, embedding_index)
    log = thinker.think()              # auto-picks best question
    log = thinker.think(question="...") # think about specific topic
"""

import json
import time
from dataclasses import dataclass, field
from graph.brain import (Brain, Node, Edge, EdgeType, EdgeSource,
                         NodeType, NodeStatus)
from llm_utils import llm_call, llm_json, llm_chat
from embedding import embed as shared_embed
from thinker.policy import CognitivePolicy

# ── Thinking patterns ─────────────────────────────────────────────────────────

DIALECTICAL_PROMPT = """You are a rigorous theoretical scientist reasoning dialectically about a fundamental research question.

QUESTION: {question}

RELEVANT KNOWLEDGE & EVIDENCE:
{context}

Reason dialectically through a structured scientific analysis:
1. Thesis: What mechanistic, mathematical, or empirical evidence supports an affirmative resolution?
2. Antithesis: What fundamental physical limits, counter-arguments, or conflicting observations challenge it?
3. Synthesis: What higher-order model or reconciling principle best explains both perspectives?
4. Crucial Experiment: What decisive observable, parameter regime, or simulation would falsify or substantiate this synthesis?

Write a dense, structured, and authoritative scientific argument (2-4 paragraphs). Cite specific principles and relationships from the provided knowledge.
"""

ANALOGICAL_PROMPT = """You are a scientist searching for cross-domain structural isomorphisms to resolve a research question.

QUESTION: {question}

CROSS-DOMAIN KNOWLEDGE BASE:
{context}

Search for a deep structural or mathematical isomorphism from a different scientific domain:
1. What analogous phenomenon in another field shares the same underlying dynamics, mathematical formulation, or topological structure?
2. What principles, equations, or mechanisms resolved that problem in the source domain?
3. How can that formal solution be mapped onto our target domain? Identify exact variable-to-variable and mechanism-to-mechanism correspondences.
4. What novel, testable predictions does this cross-domain isomorphism generate that direct domain-specific thinking would miss?

If no valid structural analogy exists, state so clearly and explain the unique physical or biological constraints of this system.
"""

REDUCTIVE_PROMPT = """You are a scientist breaking down a complex research question into foundational sub-questions.

MAIN QUESTION: {question}

RELEVANT KNOWLEDGE:
{context}

Break this question down into 2-4 simpler, highly-focused SUB-QUESTIONS that directly constrain or answer the main question.

CRITICAL REQUIREMENT:
Every sub-question MUST be an actual interrogative inquiry ending with a question mark ('?'), asking about an unknown mechanism, relationship, or quantity (e.g., beginning with "What mechanism...", "How does X affect Y...", "Under what conditions...").
Do NOT output declarative assertions, background explanations, or topic headings.

Respond with a JSON object:
{{
  "sub_questions": [
    {{
      "question": "An interrogative inquiry ending with '?'",
      "existing_evidence": "what we already know, or 'none'",
      "tractability": "high/medium/low",
      "leverage": "high/medium/low"
    }}
  ],
  "recommended_focus": "which sub-question to pursue first and why"
}}
"""

EXPERIMENTAL_PROMPT = """You are a theoretical scientist designing a rigorous thought experiment or simulation protocol.

QUESTION: {question}

RELEVANT KNOWLEDGE:
{context}

Design a decisive thought experiment or computational simulation:
1. HYPOTHESIS: What specific, directional hypothesis does this question imply?
2. INDEPENDENT & DEPENDENT VARIABLES: What concrete physical or dynamical variables must be modulated and measured?
3. PREDICTED DIVERGENCE: If the hypothesis is TRUE, what exact quantitative signature, scaling law, or phase transition occurs? What happens if it is FALSE?
4. THEORETICAL EVALUATION: How do existing physical principles or empirical data inform this prediction?
5. FALSIFICATION CRITERION: Under what precise conditions would this hypothesis be decisively refuted?

Formulate the thought experiment as a clear, step-by-step scientific argument.
"""

INTEGRATIVE_PROMPT = """You are a theoretical scientist seeking a unifying principle across disparate ideas.

KNOWLEDGE BASE:
{context}

QUESTION: {question}

Formulate a unifying principle that bridges these ideas:
1. UNIFYING PRINCIPLE: What fundamental invariant, non-equilibrium thermodynamic constraint, or dynamical law explains why these disparate phenomena co-exist?
2. EMERGENT PREDICTION: What non-obvious, quantitative prediction arises from this unification that none of the individual ideas predict alone?
3. FALSIFIABILITY: What specific empirical observation or mathematical contradiction would prove this unifying principle wrong?

State the principle with mathematical and mechanistic precision.
"""

PICK_PATTERN_PROMPT = """You are selecting a reasoning strategy for a scientific question.

QUESTION: {question}

Available strategies:
- dialectical: weigh evidence for and against (best for contested claims)
- analogical: find parallels in other domains (best for novel problems)
- reductive: break into sub-questions (best for complex, multi-part problems)
- experimental: design thought experiments (best for testable hypotheses)
- integrative: find unifying principles (best when many related facts exist)

Which strategy is BEST for this question? Respond with ONLY the strategy name.
"""

THINKING_SUMMARY_PROMPT = """You are extracting the core scientific insight from a thinking session to be stored as a standalone node in a scientific knowledge graph.

Thinking session:
{reasoning}

Extract the single most significant, defensible scientific proposition or mechanism discovered in this session.
Write it as 1-2 standalone sentences. Do NOT include meta-labels, conversational preambles, or prefixes (such as 'Summary:' or '1-2 sentence summary:'). State the scientific claim directly.

Respond with a JSON object:
{{
  "insight_statement": "The direct conceptual claim or mechanism."
}}
"""

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ThinkingLog:
    question: str          = ""
    pattern: str           = ""
    node_type: str         = "question"
    cluster: str           = "unclustered"
    reasoning: str         = ""
    insight: str           = ""
    sub_questions: list    = field(default_factory=list)
    node_id: str           = ""
    experiment_run: bool   = False
    sandbox_verdict: str   = ""
    started_at: float      = field(default_factory=time.time)
    duration: float        = 0.0

    def to_dict(self):
        return self.__dict__


# ── Thinker ───────────────────────────────────────────────────────────────────

class Thinker:
    def __init__(self, brain: Brain, observer=None, embedding_index=None,
                 critic=None, sandbox=None):
        self.brain    = brain
        self.observer = observer
        self.index    = embedding_index
        self.critic   = critic   # System 2 gating (optional)
        self.sandbox  = sandbox  # Computational experimentation engine
        self.policy   = CognitivePolicy()

    def _build_context(self, question: str, max_nodes: int = 8) -> str:
        """Build relevant context from the graph for a given question."""
        lines = []

        # From embedding index
        if self.index and self.index.size > 0:
            q_emb = shared_embed(question)
            matches = self.index.query(q_emb, threshold=0.25, top_k=max_nodes)
            for nid, score in matches:
                node = self.brain.get_node(nid)
                if node:
                    ntype = node.get('node_type', 'concept')
                    status = node.get('status', 'uncertain')
                    lines.append(
                        f"[{ntype}/{status}] {node['statement']}"
                    )

        # Working memory items (always included)
        for nid, data in self.brain.get_working_memory():
            line = f"[FOCUS/{data.get('node_type','concept')}] {data['statement']}"
            if line not in lines:
                lines.append(line)

        # Mission context
        mission = self.brain.get_mission()
        if mission:
            lines.insert(0, f"[MISSION] {mission['question']}")

        return "\n\n".join(lines) if lines else "No relevant knowledge found."

    def _pick_question(self) -> tuple[str, str]:
        """Pick the best question to think about from the agenda/graph, returning (statement, node_id)."""
        # Priority: working memory hypotheses > agenda questions > high-importance gaps
        for nid, data in self.brain.get_working_memory():
            if data.get('node_type') in [NodeType.HYPOTHESIS.value,
                                          NodeType.QUESTION.value,
                                          NodeType.GAP.value]:
                return data['statement'], nid

        # From observer agenda
        if self.observer and hasattr(self.observer, 'agenda'):
            open_items = [
                item for item in self.observer.agenda
                if not item.resolved
            ]
            if open_items:
                # Pick highest priority
                best = max(open_items, key=lambda x: x.priority)
                return best.text, getattr(best, 'node_id', '')

        # From graph — highest importance unresolved question
        questions = self.brain.nodes_by_type(NodeType.QUESTION)
        gaps      = self.brain.nodes_by_type(NodeType.GAP)
        hyps      = self.brain.nodes_by_type(NodeType.HYPOTHESIS)

        candidates = questions + gaps + hyps
        if candidates:
            best = max(candidates,
                       key=lambda x: x[1].get('importance', 0.5))
            return best[1]['statement'], best[0]

        # Fallback: think about the mission
        mission = self.brain.get_mission()
        if mission:
            return mission['question'], mission.get('id', '')

        return "What is the most important open question in our knowledge?", ""

    def _pick_pattern(self, question: str) -> tuple[str, str, str]:
        """Let the RL policy choose the best reasoning pattern."""
        node_type = "question"
        cluster = "unclustered"
        
        q_emb = shared_embed(question)
        if self.index and self.index.size > 0:
            matches = self.index.query(q_emb, threshold=0.8, top_k=1)
            if matches:
                nid, _ = matches[0]
                node = self.brain.get_node(nid)
                if node:
                    node_type = node.get('node_type', 'question')
                    cluster = node.get('cluster', 'unclustered')
                    
        pattern = self.policy.choose_pattern(node_type, cluster)
        return node_type, cluster, pattern

    def _recognize_and_run_experiment(self, question: str, question_node_id: str, pattern: str):
        """
        Recognize if the current inquiry or hypothesis requires computational
        experimentation, and if so, redirect to Sandbox directly before pondering further.
        """
        if not self.sandbox:
            return None

        target_statement = None
        target_node_id = None

        # Case 1: Target node is an untested hypothesis in the graph
        if question_node_id and self.brain.graph.has_node(question_node_id):
            node = self.brain.get_node(question_node_id)
            if node and node.get('node_type') == NodeType.HYPOTHESIS.value:
                already_tested = any(
                    edata.get('type') == EdgeType.EMPIRICALLY_TESTED.value
                    for _, _, edata in self.brain.graph.out_edges(question_node_id, data=True)
                ) or any(
                    edata.get('type') == EdgeType.EMPIRICALLY_TESTED.value
                    for _, _, edata in self.brain.graph.in_edges(question_node_id, data=True)
                )
                if not already_tested:
                    target_statement = node.get('statement', question)
                    target_node_id = question_node_id

        # Case 2: Reasoning pattern is explicitly experimental or statement is an empirical inquiry
        if not target_statement:
            is_empirical_inquiry = (
                pattern == "experimental" or
                any(kw in question.lower() for kw in [
                    "simulate", "parameter sweep", "numerical", "scaling law",
                    "phase transition", "threshold", "rate of dissipation",
                    "test whether", "computational model", "lyapunov"
                ])
            )
            if is_empirical_inquiry:
                testable, reason, approach = self.sandbox.is_testable(question)
                if testable:
                    target_statement = question
                    target_node_id = question_node_id

        # Case 3: Immediate graph neighborhood has an untested hypothesis directly linked
        if not target_statement and question_node_id and self.brain.graph.has_node(question_node_id):
            for nbr_id in self.brain.graph.neighbors(question_node_id):
                nbr = self.brain.get_node(nbr_id)
                if nbr and nbr.get('node_type') == NodeType.HYPOTHESIS.value:
                    tested = any(
                        edata.get('type') == EdgeType.EMPIRICALLY_TESTED.value
                        for _, _, edata in self.brain.graph.out_edges(nbr_id, data=True)
                    ) or any(
                        edata.get('type') == EdgeType.EMPIRICALLY_TESTED.value
                        for _, _, edata in self.brain.graph.in_edges(nbr_id, data=True)
                    )
                    if not tested:
                        testable, _, _ = self.sandbox.is_testable(nbr['statement'])
                        if testable:
                            target_statement = nbr['statement']
                            target_node_id = nbr_id
                            break

        if target_statement:
            print(f"\n  ⚡ [Empirical Need Recognized] Prioritizing computational experiment before pondering:")
            print(f"     Target: '{target_statement[:80]}...'")
            try:
                result = self.sandbox.test_hypothesis(target_statement, node_id=target_node_id or "")
                return result
            except Exception as e:
                print(f"     [Sandbox Redirection Error]: {e}")
                return None

        return None

    def think(self, question: str = None, pattern: str = None,
              max_depth: int = 2) -> ThinkingLog:
        """
        Run a deliberate thinking session.

        Args:
            question: Topic to think about (auto-picks if None)
            pattern: Reasoning pattern to use (auto-picks if None)
            max_depth: For reductive thinking, how many levels deep

        Returns:
            ThinkingLog with the reasoning and any insights produced
        """
        start = time.time()
        log = ThinkingLog()

        # Pick question
        question_node_id = ""
        if not question:
            question, question_node_id = self._pick_question()
        else:
            if self.index and self.index.size > 0:
                matches = self.index.query(shared_embed(question), threshold=0.85, top_k=1)
                if matches:
                    question_node_id = matches[0][0]
        log.question = question
        print(f"\n── Thinking: {question[:80]}... ──")

        # Pick pattern
        if not pattern:
            log.node_type, log.cluster, pattern = self._pick_pattern(question)
        else:
            log.node_type, log.cluster = "question", "unclustered"
            
        log.pattern = pattern
        print(f"  Pattern: {pattern}")

        # Build context
        context = self._build_context(question)

        # ── Empirical Need Recognition & Sandbox Redirection ─────────────────
        # Evaluate if empirical simulations are required before theoretical pondering
        sandbox_res = self._recognize_and_run_experiment(question, question_node_id, pattern)
        if sandbox_res and sandbox_res.verdict != "error":
            log.experiment_run = True
            log.sandbox_verdict = sandbox_res.verdict
            context += (
                f"\n\n[EMPIRICAL EXPERIMENT RESULTS (from Sandbox)]\n"
                f"Tested Hypothesis / Question: {sandbox_res.hypothesis}\n"
                f"Simulation Verdict: {sandbox_res.verdict.upper()} (confidence={sandbox_res.confidence:.2f})\n"
                f"Simulation Approach: {sandbox_res.approach}\n"
                f"Simulation Output:\n{sandbox_res.stdout.strip()[:1000]}\n"
                f"Interpretation: {sandbox_res.interpretation}\n"
                f"Implications: {sandbox_res.implications}\n"
            )

        # Run the appropriate reasoning pattern.
        # Appendix C.1 describes five patterns, all five implemented as distinct
        # prompts below. CognitivePolicy's action space (thinker/policy.py) was
        # previously misaligned with this (7 actions, 4 of which silently fell
        # back to dialectical, while experimental/integrative weren't reachable
        # at all) -- fixed to match these five exactly.
        prompts = {
            "dialectical":  DIALECTICAL_PROMPT,
            "analogical":   ANALOGICAL_PROMPT,
            "reductive":    REDUCTIVE_PROMPT,
            "experimental": EXPERIMENTAL_PROMPT,
            "integrative":  INTEGRATIVE_PROMPT,
        }

        prompt = prompts.get(pattern, DIALECTICAL_PROMPT)

        if pattern == "reductive":
            # Reductive returns structured JSON
            result = llm_json(
                prompt.format(question=question, context=context),
                temperature=0.4,
                default={"sub_questions": [], "recommended_focus": ""}
            )
            log.sub_questions = result.get("sub_questions", [])
            log.reasoning = json.dumps(result, indent=2)

            # Add sub-questions to the graph and agenda
            for sq in log.sub_questions[:4]:
                q_text = sq.get("question", "").strip()
                if not q_text or len(q_text) < 10:
                    continue
                # Enforce semantic type invariant: true interrogatives -> QUESTION; declarative assertions -> CONCEPT
                is_inquiry = q_text.endswith("?") or any(
                    q_text.lower().startswith(w) for w in
                    ["what", "how", "why", "which", "can", "does", "is", "under what", "to what", "where", "when"]
                )
                node_type = NodeType.QUESTION if is_inquiry else NodeType.CONCEPT
                node = Node(
                    statement    = q_text,
                    node_type    = node_type,
                    cluster      = "thinking",
                    status       = NodeStatus.UNCERTAIN,
                    importance   = 0.65,
                    source_quality = 0.6
                )
                nid = self.brain.add_node(node)
                if self.index:
                    self.index.add(nid, shared_embed(q_text))

                # Connect sub-question to parent question in graph
                if question_node_id and self.brain.get_node(question_node_id):
                    sub_edge = Edge(
                        type       = EdgeType.ASSOCIATED,
                        narration  = f"Reductive sub-question constraining: {question[:80]}",
                        weight     = 0.75,
                        confidence = 0.80,
                        source     = EdgeSource.CONSOLIDATION
                    )
                    self.brain.add_edge(nid, question_node_id, sub_edge)

                # Add to agenda only if it is a genuine inquiry
                if is_inquiry and self.observer and hasattr(self.observer, 'add_to_agenda'):
                    item = self.observer.add_to_agenda(
                        text      = q_text,
                        item_type = "question",
                        cycle     = getattr(self.observer, 'cycle_count', 0),
                        node_id   = nid
                    )
                    # Higher leverage = higher priority
                    leverage = sq.get("leverage", "medium")
                    if leverage == "high":
                        item.priority = 0.8

                print(f"  Sub-question [{node_type.value}]: {q_text[:60]}...")

            # Summarize the recommended focus
            focus = result.get("recommended_focus", "")
            if focus:
                log.insight = focus
        else:
            # Other patterns return free-form reasoning
            log.reasoning = llm_call(
                prompt.format(question=question, context=context),
                temperature=0.5,
                role="reasoning"
            )

            # Extract insight via structured JSON contract
            summary_res = llm_json(
                THINKING_SUMMARY_PROMPT.format(reasoning=log.reasoning),
                temperature=0.2,
                role="precise",
                default={"insight_statement": ""}
            )
            if isinstance(summary_res, dict):
                log.insight = summary_res.get("insight_statement", "").strip()
            elif isinstance(summary_res, str):
                log.insight = summary_res.strip()
            else:
                log.insight = ""

        # ── System 2 gating ──
        # Route insight through Critic before graph insertion
        if log.insight and len(log.insight) > 15:
            if self.critic:
                from critic.critic import CandidateThought, Verdict
                candidate = CandidateThought(
                    claim         = log.insight,
                    source_module = "thinker",
                    proposed_type = "synthesis",
                    importance    = 0.7,
                    context       = log.reasoning,
                    node_a_id     = question_node_id,
                )
                critic_log = self.critic.evaluate_with_refinement(candidate)

                if critic_log.verdict == Verdict.ACCEPT:
                    reward = 1.0 * critic_log.confidence
                    # Use critic-assigned confidence instead of default
                    final_claim = log.insight
                    confidence  = critic_log.confidence
                    node = Node(
                        statement      = final_claim,
                        node_type      = NodeType.SYNTHESIS,
                        cluster        = "thinking",
                        status         = NodeStatus.UNCERTAIN,
                        importance     = confidence,
                        source_quality = confidence
                    )
                    nid = self.brain.add_node(node)
                    log.node_id = nid
                    if self.index:
                        self.index.add(nid, shared_embed(final_claim))
                    self.brain.focus_on(nid)

                    # Connect insight to question node in graph
                    if question_node_id and self.brain.get_node(question_node_id):
                        edge_type = EdgeType.ANSWERS if confidence >= 0.6 else EdgeType.SUPPORTS
                        ans_edge = Edge(
                            type       = edge_type,
                            narration  = f"Thinker [{pattern}] conclusion: {final_claim[:80]}",
                            weight     = confidence,
                            confidence = confidence,
                            source     = EdgeSource.CONSOLIDATION
                        )
                        self.brain.add_edge(nid, question_node_id, ans_edge)

                    # Propagate resolution to observer agenda
                    if self.observer and hasattr(self.observer, 'record_answer'):
                        grade = 'strong' if confidence >= 0.7 else 'partial'
                        self.observer.record_answer(
                            question_text  = question,
                            answer_node_id = nid,
                            explanation    = final_claim,
                            grade          = grade
                        )

                    # Propagate to mission if relevant
                    if confidence >= 0.65:
                        self.brain.link_to_mission(nid, final_claim, strength=confidence * 0.8)
                    if confidence >= 0.75 and self.observer and hasattr(self.observer, 'record_mission_advance'):
                        self.observer.record_mission_advance(
                            nid, f"Deliberate reasoning advance: {final_claim[:80]}", confidence * 0.85
                        )

                    print(f"  ✓ Insight accepted (conf={confidence:.2f}): "
                          f"{final_claim[:80]}...")

                elif critic_log.verdict == Verdict.REJECT:
                    reward = -1.0
                    print(f"  ✗ Insight rejected: {critic_log.rejection_reason}")
                    log.insight = ""  # clear so callers know it was rejected

                elif critic_log.verdict == Verdict.DEFER:
                    reward = 0.0
                    if "Failed to parse verdict" not in (critic_log.refinement_note or ""):
                        print(f"  ◇ Insight deferred to insight buffer")
                        self.critic.route_deferred(candidate)
                    else:
                        print(f"  ◇ Critic verdict parse error — skipping buffer routing")
                    log.insight = ""  # clear — not in graph yet

                else:  # REFINE exhausted → treated as DEFER by evaluate_with_refinement
                    reward = 0.0
                    print(f"  ◇ Insight deferred after refinement")
                    log.insight = ""

                # Train RL Policy
                self.policy.update(log.node_type, log.cluster, log.pattern, reward, self.brain.dopamine)

            else:
                self.policy.update(log.node_type, log.cluster, log.pattern, 0.5, self.brain.dopamine)
                # No critic — original behavior (direct insertion)
                node = Node(
                    statement      = log.insight,
                    node_type      = NodeType.SYNTHESIS,
                    cluster        = "thinking",
                    status         = NodeStatus.UNCERTAIN,
                    importance     = 0.7,
                    source_quality = 0.6
                )
                nid = self.brain.add_node(node)
                log.node_id = nid
                if self.index:
                    self.index.add(nid, shared_embed(log.insight))
                self.brain.focus_on(nid)

                if question_node_id and self.brain.get_node(question_node_id):
                    ans_edge = Edge(
                        type       = EdgeType.ANSWERS,
                        narration  = f"Thinker [{pattern}] conclusion: {log.insight[:80]}",
                        weight     = 0.7,
                        confidence = 0.7,
                        source     = EdgeSource.CONSOLIDATION
                    )
                    self.brain.add_edge(nid, question_node_id, ans_edge)

                if self.observer and hasattr(self.observer, 'record_answer'):
                    self.observer.record_answer(
                        question_text  = question,
                        answer_node_id = nid,
                        explanation    = log.insight,
                        grade          = 'strong'
                    )

                self.brain.link_to_mission(nid, log.insight, strength=0.6)
                print(f"  Insight: {log.insight[:80]}...")

        log.duration = time.time() - start
        print(f"  Thinking complete ({log.duration:.1f}s)")

        return log

    def think_session(self, num_rounds: int = 3) -> list[ThinkingLog]:
        """
        Run multiple rounds of thinking, each building on the last.

        Round 1: Pick a question and think about it
        Round 2+: Either refine the previous insight or pivot to a related question
        """
        logs = []
        print(f"\n══ Thinking session — {num_rounds} rounds ══")

        for i in range(num_rounds):
            print(f"\n── Round {i+1}/{num_rounds} ──")

            if i == 0:
                log = self.think()
            else:
                # Use previous insight as seed for next round
                prev = logs[-1]
                if prev.sub_questions:
                    # Follow up on a sub-question
                    best_sq = prev.sub_questions[0].get("question", "")
                    if best_sq:
                        log = self.think(question=best_sq)
                    else:
                        log = self.think()
                elif prev.insight:
                    # Refine the previous insight
                    follow_up = f"Given that '{prev.insight}', " \
                                f"what does this imply for our central mission?"
                    log = self.think(question=follow_up)
                else:
                    log = self.think()

            logs.append(log)

        print(f"\n══ Thinking session complete — {len(logs)} rounds ══")
        return logs
