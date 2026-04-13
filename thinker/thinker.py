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
import re
import time
from dataclasses import dataclass, field
from graph.brain import (Brain, Node, Edge, EdgeType, EdgeSource,
                         NodeType, NodeStatus)
from llm_utils import llm_call, llm_json, llm_chat
from embedding import embed as shared_embed
from thinker.policy import CognitivePolicy

# ── Thinking patterns ─────────────────────────────────────────────────────────

DIALECTICAL_PROMPT = """You are a scientist reasoning carefully about a question.

QUESTION: {question}

RELEVANT KNOWLEDGE:
{context}

Think dialectically:
1. What evidence or reasoning SUPPORTS a positive answer?
2. What evidence or reasoning ARGUES AGAINST it?
3. Given both sides, what is the most defensible conclusion right now?
4. What SPECIFIC piece of evidence or experiment would resolve the tension?

Write your reasoning as a structured argument (2-4 paragraphs).
Be precise. Cite specific ideas from the knowledge provided.
"""

ANALOGICAL_PROMPT = """You are a scientist looking for analogies that could solve a problem.

QUESTION: {question}

KNOWLEDGE FROM VARIOUS DOMAINS:
{context}

Is there a problem in a DIFFERENT domain that has a similar structure to this one?
If so:
1. What is the analogous problem?
2. How was it solved there?
3. Can that solution transfer to our domain? What would need to change?
4. What does this analogy reveal that direct reasoning might miss?

If no useful analogy exists, say so honestly and explain what makes this problem unique.
"""

REDUCTIVE_PROMPT = """You are a scientist trying to break down a hard question.

QUESTION: {question}

RELEVANT KNOWLEDGE:
{context}

This question may be too complex to answer directly. Break it down:
1. What are the 2-4 simpler SUB-QUESTIONS that, if answered, would answer the main question?
2. For each sub-question, do we already have evidence in our knowledge?
3. Which sub-question is the MOST tractable right now?
4. Which sub-question, if answered, would have the HIGHEST leverage?

Respond with a JSON object:
{{
  "sub_questions": [
    {{
      "question": "the sub-question",
      "existing_evidence": "what we already know, or 'none'",
      "tractability": "high/medium/low",
      "leverage": "high/medium/low"
    }}
  ],
  "recommended_focus": "which sub-question to pursue first and why"
}}
"""

EXPERIMENTAL_PROMPT = """You are a scientist designing a thought experiment.

QUESTION: {question}

RELEVANT KNOWLEDGE:
{context}

Design a thought experiment to test this:
1. What HYPOTHESIS does this question imply?
2. IF the hypothesis is true, what SPECIFIC OBSERVABLE consequence would we expect?
3. IF the hypothesis is false, what would we expect instead?
4. Can we check either prediction against what we already know?
5. What is the verdict so far?

Write your thought experiment as a clear, step-by-step argument.
"""

INTEGRATIVE_PROMPT = """You are a scientist looking for a unifying principle.

These ideas all seem related but no one has articulated WHY:
{context}

QUESTION: {question}

Can you find a UNIFYING PRINCIPLE that:
1. Explains why all these ideas are connected?
2. Predicts something NEW that none of them state individually?
3. Is FALSIFIABLE — what would prove it wrong?

If you find one, state it precisely. If not, explain what's missing.
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

THINKING_SUMMARY_PROMPT = """Summarize the key insight from this thinking session in 1-2 sentences.
This will be stored as a new node in the knowledge graph.

Thinking session:
{reasoning}

Respond with ONLY the insight statement. No preamble.
"""

NEXT_ROUND_PROMPT = """You are planning the next step in a scientific reasoning session.

MISSION:
{mission}

SESSION ANCHOR QUESTION:
{anchor_question}

PREVIOUS ROUND QUESTION:
{previous_question}

PREVIOUS ROUND PATTERN:
{previous_pattern}

PREVIOUS ROUND INSIGHT:
{previous_insight}

PREVIOUS ROUND SUB-QUESTIONS:
{previous_sub_questions}

EARLIER INSIGHTS:
{history}

Choose the SINGLE best next question to improve the session.

Requirements:
1. Stay tightly tied to the mission and anchor question.
2. Increase specificity, testability, or decision value.
3. Prefer a mechanism, observable, boundary condition, counterexample, or decisive comparison.
4. Do NOT merely restate the mission or previous insight at a higher level of abstraction.
5. If the previous round was integrative or analogical, prefer grounding or falsification over more synthesis.

Respond EXACTLY in JSON:
{{
  "next_question": "one precise question",
  "preferred_pattern": "dialectical|analogical|reductive|experimental|integrative",
  "goal": "one sentence describing how this question advances the session"
}}
"""

SUPPORTED_THINKING_PATTERNS = {
    "dialectical",
    "analogical",
    "reductive",
    "experimental",
    "integrative",
}

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
    started_at: float      = field(default_factory=time.time)
    duration: float        = 0.0

    def to_dict(self):
        return self.__dict__


# ── Thinker ───────────────────────────────────────────────────────────────────

class Thinker:
    def __init__(self, brain: Brain, observer=None, embedding_index=None,
                 critic=None):
        self.brain    = brain
        self.observer = observer
        self.index    = embedding_index
        self.critic   = critic   # System 2 gating (optional)
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

    def _pick_question(self) -> str:
        """Pick the best question to think about from the agenda/graph."""
        # Priority: working memory hypotheses > agenda questions > high-importance gaps
        for nid, data in self.brain.get_working_memory():
            if data.get('node_type') in [NodeType.HYPOTHESIS.value,
                                          NodeType.QUESTION.value,
                                          NodeType.GAP.value]:
                return data['statement']

        # From observer agenda
        if self.observer and hasattr(self.observer, 'agenda'):
            open_items = [
                item for item in self.observer.agenda
                if not item.resolved
            ]
            if open_items:
                # Pick highest priority
                best = max(open_items, key=lambda x: x.priority)
                return best.text

        # From graph — highest importance unresolved question
        questions = self.brain.nodes_by_type(NodeType.QUESTION)
        gaps      = self.brain.nodes_by_type(NodeType.GAP)
        hyps      = self.brain.nodes_by_type(NodeType.HYPOTHESIS)

        candidates = questions + gaps + hyps
        if candidates:
            best = max(candidates,
                       key=lambda x: x[1].get('importance', 0.5))
            return best[1]['statement']

        # Fallback: think about the mission
        mission = self.brain.get_mission()
        if mission:
            return mission['question']

        return "What is the most important open question in our knowledge?"

    def _pick_pattern(self, question: str) -> tuple[str, str, str]:
        """Choose a pattern using question semantics plus procedural memory."""
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

        preferred_pattern = self._semantic_pattern_hint(question)
        pattern = self.policy.choose_pattern(
            node_type, cluster, preferred_action=preferred_pattern
        )
        if pattern not in SUPPORTED_THINKING_PATTERNS:
            pattern = preferred_pattern or "dialectical"
        return node_type, cluster, pattern

    def _normalize_pattern_name(self, raw: str) -> str:
        text = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        alias_map = {
            "empirical": "experimental",
            "thought_experiment": "experimental",
            "first_principles": "reductive",
            "first_principle": "reductive",
            "synthesis": "integrative",
        }
        if text in alias_map:
            text = alias_map[text]
        return text if text in SUPPORTED_THINKING_PATTERNS else ""

    def _heuristic_pattern_hint(self, question: str) -> str:
        q = (question or "").lower()
        if any(phrase in q for phrase in (
            "sub-question",
            "sub question",
            "decompose",
            "break down",
            "simpler unknown",
            "tractable sub",
            "sub-problem",
            "sub problem",
        )):
            return "reductive"
        if any(phrase in q for phrase in (
            "unifying principle",
            "what explains why",
            "why all",
            "common principle",
            "unify",
        )):
            return "integrative"
        if any(phrase in q for phrase in (
            "what observable",
            "what would we expect",
            "what would distinguish",
            "if ",
            "prediction",
            "consequence",
            "test this",
        )):
            return "experimental"
        if any(phrase in q for phrase in (
            "supports versus",
            "supports and",
            "argues against",
            "evidence for and against",
            "weighed for and against",
            "does the evidence cut",
            "defensible, or",
            "versus weakens",
        )):
            return "dialectical"
        if any(phrase in q for phrase in (
            "analogy",
            "analogous",
            "parallel",
            "transfer",
            "different domain",
            "compare to",
            "offer an analogy",
        )):
            return "analogical"
        return ""

    def _semantic_pattern_hint(self, question: str) -> str:
        heuristic = self._heuristic_pattern_hint(question)
        if heuristic:
            return heuristic
        raw = llm_call(
            PICK_PATTERN_PROMPT.format(question=question),
            temperature=0.1,
            role="precise"
        )
        return self._normalize_pattern_name(raw)

    def _mission_text(self) -> str:
        mission = self.brain.get_mission()
        if mission and mission.get("question"):
            return mission["question"]
        return ""

    def _score_sub_question(self, sub_question: dict) -> tuple[float, float]:
        leverage_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
        tractability_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
        leverage = leverage_map.get(
            str(sub_question.get("leverage", "medium")).lower(),
            2.0,
        )
        tractability = tractability_map.get(
            str(sub_question.get("tractability", "medium")).lower(),
            2.0,
        )
        return leverage, tractability

    def _best_follow_up_subquestion(self, sub_questions: list[dict]) -> str:
        ranked = []
        for sq in sub_questions or []:
            question = (sq.get("question") or "").strip()
            if not question:
                continue
            leverage, tractability = self._score_sub_question(sq)
            ranked.append(((leverage, tractability, -len(question)), question))
        if not ranked:
            return ""
        ranked.sort(reverse=True)
        return ranked[0][1]

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _content_tokens(self, text: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
            "does", "for", "from", "how", "if", "in", "is", "it", "of",
            "on", "or", "the", "to", "what", "when", "which", "while",
            "with", "would",
        }
        tokens = re.findall(r"[a-z0-9]+", self._normalize_text(text))
        return {token for token in tokens if token not in stopwords}

    def _question_overlap_score(self, main_question: str, sub_question: str) -> float:
        main_tokens = self._content_tokens(main_question)
        sub_tokens = self._content_tokens(sub_question)
        if not main_tokens or not sub_tokens:
            return 0.0
        return len(main_tokens & sub_tokens) / max(len(sub_tokens), 1)

    def _focus_cue_bonus(self, question: str) -> float:
        q = self._normalize_text(question)
        bonus = 0.0
        cue_weights = {
            "mechanism": 1.2,
            "metric": 1.2,
            "quantif": 1.0,
            "measure": 1.0,
            "distinguish": 1.0,
            "compare": 0.8,
            "boundary": 0.8,
            "condition": 0.8,
            "transition": 1.0,
            "balance": 0.8,
            "mapping": 0.8,
            "framework": 0.6,
            "rate": 0.6,
            "dynamics": 0.8,
        }
        for cue, weight in cue_weights.items():
            if cue in q:
                bonus += weight
        return bonus

    def _question_like(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        if stripped.endswith("?"):
            return True
        return bool(re.match(r"^(What|Which|How|Why|When|Where|Can|Could|Is|Are|Under what)\b", stripped))

    def _select_focus_subquestion(
        self,
        main_question: str,
        sub_questions: list[dict],
        recommended_focus: str,
    ) -> str:
        focus_text = self._normalize_text(recommended_focus)
        candidates = []
        for sq in sub_questions or []:
            q_text = (sq.get("question") or "").strip()
            if not q_text:
                continue
            if self._normalize_text(q_text) in focus_text:
                return q_text
            leverage, tractability = self._score_sub_question(sq)
            overlap = self._question_overlap_score(main_question, q_text)
            cue_bonus = self._focus_cue_bonus(q_text)
            score = (leverage * 3.0) + (tractability * 0.8) + (overlap * 4.0) + cue_bonus
            candidates.append((score, len(q_text), q_text))

        if not candidates:
            return ""
        candidates.sort(reverse=True)
        return candidates[0][2]

    def _format_focus_insight(self, focus_question: str, recommended_focus: str) -> str:
        focus_question = (focus_question or "").strip()
        if not focus_question:
            return (recommended_focus or "").strip()
        if recommended_focus and not self._question_like(recommended_focus):
            return f"Priority focus: {focus_question} Reason: {recommended_focus.strip()}"
        return f"Priority focus: {focus_question}"

    def _fallback_next_round(self, anchor_question: str, previous_log: ThinkingLog) -> tuple[str, str]:
        mission = self._mission_text() or anchor_question or previous_log.question
        previous_insight = previous_log.insight or previous_log.question

        if previous_log.pattern in {"integrative", "analogical"}:
            return (
                f"What concrete observation, intervention, or failure case would discriminate whether '{previous_insight}' is the right explanation for '{mission}'?",
                "experimental",
            )
        if previous_log.pattern == "dialectical":
            return (
                f"What specific experiment, dataset, or comparison would resolve the strongest remaining uncertainty in '{anchor_question or mission}'?",
                "experimental",
            )
        if previous_log.pattern == "experimental":
            return (
                f"What boundary condition or competing explanation would most strongly challenge the current prediction about '{anchor_question or mission}'?",
                "dialectical",
            )
        if previous_log.pattern == "reductive" and previous_log.sub_questions:
            best_sq = self._best_follow_up_subquestion(previous_log.sub_questions)
            if best_sq:
                return best_sq, ""
        return (
            f"What specific mechanism or decision-relevant comparison would most directly answer '{anchor_question or mission}'?",
            "dialectical",
        )

    def _plan_next_round(self, previous_log: ThinkingLog, history: list[ThinkingLog]) -> tuple[str, str]:
        if previous_log.sub_questions:
            best_sq = self._best_follow_up_subquestion(previous_log.sub_questions)
            if best_sq:
                return best_sq, ""

        anchor_question = history[0].question if history else previous_log.question
        mission = self._mission_text() or anchor_question or previous_log.question
        recent_insights = [
            log.insight.strip()
            for log in history[-3:]
            if log.insight and log.insight.strip()
        ]
        history_text = "\n".join(f"- {item}" for item in recent_insights) or "- none"
        previous_sub_questions = json.dumps(previous_log.sub_questions[:4], indent=2)

        plan = llm_json(
            NEXT_ROUND_PROMPT.format(
                mission=mission,
                anchor_question=anchor_question,
                previous_question=previous_log.question,
                previous_pattern=previous_log.pattern or "unknown",
                previous_insight=previous_log.insight or "none",
                previous_sub_questions=previous_sub_questions,
                history=history_text,
            ),
            temperature=0.2,
            default={
                "next_question": "",
                "preferred_pattern": "",
                "goal": "",
            },
        )

        next_question = (plan.get("next_question") or "").strip()
        preferred_pattern = self._normalize_pattern_name(
            plan.get("preferred_pattern", "")
        )
        if len(next_question) < 15:
            return self._fallback_next_round(anchor_question, previous_log)
        if next_question.strip().lower() == previous_log.question.strip().lower():
            return self._fallback_next_round(anchor_question, previous_log)
        return next_question, preferred_pattern

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
        if not question:
            question = self._pick_question()
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

        # Run the appropriate reasoning pattern
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
                q_text = sq.get("question", "")
                if not q_text:
                    continue
                node = Node(
                    statement    = q_text,
                    node_type    = NodeType.QUESTION,
                    cluster      = "thinking",
                    status       = NodeStatus.UNCERTAIN,
                    importance   = 0.65,
                    source_quality = 0.6
                )
                nid = self.brain.add_node(node)
                if self.index:
                    self.index.add(nid, shared_embed(q_text))

                # Add to agenda
                if self.observer and hasattr(self.observer, 'add_to_agenda'):
                    item = self.observer.add_to_agenda(
                        text=q_text,
                        item_type="question",
                        cycle=getattr(self.observer, 'cycle_count', 0),
                        node_id=nid
                    )
                    # Higher leverage = higher priority
                    leverage = sq.get("leverage", "medium")
                    if leverage == "high":
                        item.priority = 0.8

                print(f"  Sub-question: {q_text[:60]}...")

            # Convert the model's recommended focus into a stable chosen focus.
            focus = result.get("recommended_focus", "")
            focus_question = self._select_focus_subquestion(
                question,
                log.sub_questions,
                focus,
            )
            if focus_question or focus:
                log.insight = self._format_focus_insight(focus_question, focus)
        else:
            # Other patterns return free-form reasoning
            log.reasoning = llm_call(
                prompt.format(question=question, context=context),
                temperature=0.5,
                role="reasoning"
            )

            # Extract insight as a storable statement
            log.insight = llm_call(
                THINKING_SUMMARY_PROMPT.format(reasoning=log.reasoning),
                temperature=0.2,
                role="precise"
            )

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
                )
                critic_log = self.critic.evaluate_with_refinement(candidate)
                final_claim = critic_log.final_claim or candidate.claim

                if critic_log.verdict == Verdict.ACCEPT:
                    reward = 1.0 * critic_log.confidence
                    # Use critic-assigned confidence instead of default
                    confidence  = critic_log.confidence
                    log.insight = final_claim
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
                    print(f"  ✓ Insight accepted (conf={confidence:.2f}): "
                          f"{final_claim[:80]}...")

                elif critic_log.verdict == Verdict.REJECT:
                    reward = -1.0
                    print(f"  ✗ Insight rejected: {critic_log.rejection_reason}")
                    log.insight = ""  # clear so callers know it was rejected

                elif critic_log.verdict == Verdict.DEFER:
                    reward = 0.0
                    print(f"  ◇ Insight deferred to insight buffer")
                    deferred_candidate = CandidateThought(
                        claim=final_claim,
                        source_module=candidate.source_module,
                        proposed_type=candidate.proposed_type,
                        importance=candidate.importance,
                        context=candidate.context,
                        edge_type=candidate.edge_type,
                        node_a_id=candidate.node_a_id,
                        node_b_id=candidate.node_b_id,
                        crosses_domains=candidate.crosses_domains,
                        contradicts_existing=candidate.contradicts_existing,
                    )
                    self.critic.route_deferred(deferred_candidate)
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
                prev = logs[-1]
                next_question, preferred_pattern = self._plan_next_round(prev, logs)
                log = self.think(
                    question=next_question or None,
                    pattern=preferred_pattern or None,
                )

            logs.append(log)

        print(f"\n══ Thinking session complete — {len(logs)} rounds ══")
        return logs
