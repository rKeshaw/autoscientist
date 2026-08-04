"""
Critic — System 2 (Slow Thinking) for THE SCIENTIST.

Implements Daniel Kahneman's dual-process theory:
  System 1 (Thinker/Dreamer) generates ideas quickly and intuitively.
  System 2 (Critic) monitors, challenges, and gates those ideas before
  they enter the knowledge graph.

The Critic does NOT generate ideas. It receives candidate thoughts from
System 1 and runs an adversarial multi-turn dialogue to determine whether
they survive scrutiny.

Four possible verdicts:
  ACCEPT  — Passes quality bar. Added to graph with calibrated confidence.
  REFINE  — Kernel of truth but weak formulation. System 1 gets another pass.
  REJECT  — Incoherent, hallucinated, or trivially obvious. Discarded.
  DEFER   — Not enough evidence to decide. Sent to InsightBuffer for incubation.

The "laziness principle": System 2 only activates for high-stakes claims.
Routine concept extraction and weak associative edges bypass the Critic entirely.

Usage:
    critic = Critic(brain, embedding_index=emb_index, insight_buffer=buf)
    verdict_log = critic.evaluate(candidate)
    if verdict_log.verdict == Verdict.ACCEPT:
        # add to graph with verdict_log.confidence
"""

import time
import json
from enum import Enum
from dataclasses import dataclass, field
from graph.brain import Brain, NodeType
from config import CRITIC as CRITIC_CFG
from llm_utils import llm_call, llm_chat, require_json
from embedding import embed as shared_embed


# ── Verdict enum ──────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    ACCEPT = "accept"
    REFINE = "refine"
    REJECT = "reject"
    DEFER  = "defer"


# ── Candidate thought ────────────────────────────────────────────────────────

@dataclass
class CandidateThought:
    """A thought produced by System 1, awaiting System 2 evaluation."""
    claim:           str               # The insight/hypothesis/analogy text
    source_module:   str = "thinker"   # Which module produced it: thinker, dreamer, ingestor
    proposed_type:   str = "synthesis"  # Proposed node type: synthesis, hypothesis, etc.
    importance:      float = 0.7       # System 1's estimated importance
    context:         str = ""          # Supporting evidence / reasoning that led to the claim
    edge_type:       str = ""          # If this is an edge claim (e.g., structural_analogy)
    node_a_id:       str = ""          # Source node (for edge candidates)
    node_b_id:       str = ""          # Target node (for edge candidates)
    crosses_domains: bool = False      # Whether the claim connects different clusters
    contradicts_existing: bool = False # Whether the claim conflicts with existing knowledge


# ── Dialogue turn ─────────────────────────────────────────────────────────────

@dataclass
class DialogueTurn:
    """One turn of the System 1 ↔ System 2 adversarial dialogue."""
    role:    str   # "system2_challenge" or "system1_defense"
    content: str
    turn:    int


# ── Critic log ────────────────────────────────────────────────────────────────

@dataclass
class CriticLog:
    """Full record of a System 2 evaluation session."""
    candidate_claim:   str = ""
    source_module:     str = ""
    proposed_type:     str = ""
    verdict:           Verdict = Verdict.DEFER
    confidence:        float = 0.0
    is_novel:          bool = True
    dialogue:          list = field(default_factory=list)
    reason:            str = ""   # full, untruncated verdict-step "reason" (was silently dropped before)
    checklist:         dict = field(default_factory=dict)  # verdict step's yes/no self-check, for diagnosing calibration mismatches
    rejection_reason:  str = ""
    refinement_note:   str = ""
    started_at:        float = field(default_factory=time.time)
    duration:          float = 0.0
    bypassed:          bool = False   # True if laziness principle skipped review

    def to_dict(self):
        d = {
            "candidate_claim":  self.candidate_claim,
            "source_module":    self.source_module,
            "proposed_type":    self.proposed_type,
            "verdict":          self.verdict.value,
            "confidence":       self.confidence,
            "is_novel":         self.is_novel,
            "dialogue":         [
                {"role": t.role, "content": t.content, "turn": t.turn}
                for t in self.dialogue
            ],
            "reason":           self.reason,
            "checklist":        self.checklist,
            "rejection_reason": self.rejection_reason,
            "refinement_note":  self.refinement_note,
            "duration":         self.duration,
            "bypassed":         self.bypassed,
        }
        return d


# ── Prompts ───────────────────────────────────────────────────────────────────
#
# `_run_dialogue` previously called System 2 fresh each round against the
# ORIGINAL claim with no memory of its own prior challenge. Confirmed via
# transcript (31 Jul 2026): System 1 stated an explicit entity-to-entity
# correspondence three separate times across three rounds, and System 2 issued
# the IDENTICAL "you haven't named the corresponding variable" demand every
# round, as if the first two answers never happened -- because nothing in the
# prompt asked it to check. CHALLENGE_PROMPT is now split into a first-round
# version (unchanged) and a follow-up version that explicitly threads the
# previous challenge/defense through and requires System 2 to state whether
# its own prior demand was met, with a "NO FURTHER CHALLENGE" convergence
# off-ramp -- validated (`critic_isolation5/`) to make System 2 track state
# correctly and escalate to genuinely new concerns instead of looping.

CHALLENGE_PROMPT_FIRST = """You are System 2 — the slow, skeptical, analytical part of a scientific mind.

System 1 (the fast, creative, intuitive part) has produced the following claim:

CLAIM: "{claim}"

CONTEXT that led to this claim:
{context}

Your job is to find the WEAKEST point of this claim. Be rigorous. Ask yourself:
1. Is the evidence cited actually sufficient, or is System 1 pattern-matching on surface similarity?
2. What specific mechanism or mapping is being claimed? Is it stated precisely enough to be testable?
3. Is there an obvious counterexample or alternative explanation that System 1 missed?
4. Is this actually novel, or is it restating something already known in different words?

Respond with ONE focused, precise challenge (2-3 sentences). Target the weakest link.
Do NOT be vague. Be specific about what you find problematic and what would satisfy you.
"""

# Backward-compatible alias — anything importing CHALLENGE_PROMPT directly
# (e.g. earlier one-off experiment scripts) still gets the first-round prompt.
CHALLENGE_PROMPT = CHALLENGE_PROMPT_FIRST

CHALLENGE_PROMPT_FOLLOWUP = """You are System 2 — the slow, skeptical, analytical part of a scientific mind.

System 1 (the fast, creative, intuitive part) has produced the following claim:

CLAIM: "{claim}"

CONTEXT that led to this claim:
{context}

YOUR PREVIOUS CHALLENGE WAS: "{prev_challenge}"

SYSTEM 1'S MOST RECENT RESPONSE: "{prev_defense}"

First, explicitly state whether System 1's response addressed your previous challenge.
- If it did NOT, say specifically what piece is still missing — do not just repeat your
  previous challenge verbatim.
- If it DID address it, you must raise a genuinely NEW and distinct concern (different
  from your previous challenge). If no substantive concern remains, respond with
  EXACTLY: NO FURTHER CHALLENGE

WATCH FOR FABRICATION, don't just ask for more detail. If System 1's response names a specific
external fact -- a named study, institution, researcher, or equation -- that is NOT verifiable
from the CONTEXT above, your new concern must be to question THAT fact directly: ask why THIS
specific fact is the correct one, or point out it isn't grounded in anything actually stated.
Simply asking for "more specifics" invites System 1 to invent a bigger, more convincing-sounding
fact next round instead of justifying the one it already gave -- don't reward that. A real named
equation or institution being confidently asserted is not evidence it actually applies here.

Respond with 2-4 sentences (or exactly "NO FURTHER CHALLENGE" if none remain).
"""

DEFENSE_PROMPT = """You are System 1 — the fast, creative, intuitive part of a scientific mind.

Your earlier insight was challenged by System 2 (the slow, skeptical part):

YOUR ORIGINAL CLAIM: "{claim}"

SYSTEM 2'S CHALLENGE: "{challenge}"

AVAILABLE KNOWLEDGE:
{context}

Defend your claim against this specific challenge. Your response MUST open with one
explicit correspondence line in this exact form, naming REAL entities/variables from
the claim (not placeholders) — this applies whether or not System 2 asked for it directly:
  "[Entity/variable in A] corresponds to [entity/variable in B] because [reason]."
Then:
1. If you can address the challenge with further specific evidence or reasoning, do so.
2. If you need to NARROW or QUALIFY your claim to make it defensible, do so honestly.
3. If you realize the challenge is valid and your claim is weak even with the mapping
   stated, admit it — but still state the mapping first, so the record is precise either way.

Respond in 3-5 sentences total. Be honest — a narrower true claim is better than a broad false one.
"""

# Section 4.2 specifies System 2's final step only as "delivers accept / refine / reject /
# defer." The checklist (mapping_stated, unanswered_objection, specific, novel), the ordered
# mechanical rule, the derivation-before-verdict field, and the consistency backstop below
# are not described in the paper; they were added post-hoc to make the verdict step's
# reasoning auditable and its accept/reject boundary consistent with its own stated criteria,
# after mixtral was observed contradicting a checklist it had itself just filled in as
# satisfied. This elaborates rather than contradicts Section 4.2's specification.
VERDICT_PROMPT = """You are System 2 delivering a final verdict on a candidate thought.

ORIGINAL CLAIM: "{claim}"
{depth_block}
{nodes_block}
ADVERSARIAL DIALOGUE:
{dialogue_text}

Before deciding, work through this checklist using ONLY what is actually stated in the
dialogue above (quote the specific line for each item you mark yes):

1. MAPPING: Did System 1 name at least one specific entity/variable on each side and state
   which corresponds to which? (yes/no + quote if yes)

2. UNANSWERED OBJECTION: Did System 2 raise a distinct objection that System 1 never
   adequately answered by the end of the dialogue — not merely an objection that was raised
   and then addressed? (yes/no + quote the unanswered objection if yes)

3. SPECIFIC: Does the correspondence quoted for item 1 name an actual mechanism, equation,
   quantity, or threshold that maps term-for-term — not just something stated confidently?
   Every named entity in the correspondence (a citation, a variable, a quantity, a physical
   concept like "energy states" or "spin configurations") must trace to the source/target
   statements above or to the claim itself — introducing ANY new specific-sounding noun that
   isn't there, even a plausible technical one, is fabrication, not specificity. Mark NO if:
   (a) generic language in technical dress ("both exhibit emergent patterns") that fits
       almost any pair;
   (b) it introduces a specific-sounding entity, citation, or quantity (a named
       theory/researcher/institution/study, OR a technical-sounding variable/mechanism like
       "energy states"/"spin configurations") that is absent from the source/target
       statements above — if none are given, treat anything specific-sounding as ungrounded
       by default, no exceptions;
   (c) a real fact/equation asserted to apply here with no justification for why;
   (d) an IDENTITY not a mapping — strip both sides' names; if the remaining words are the
       same phrase, it's a restatement, not a correspondence.
   Mark YES only if the named quantities/mechanism are actually established by the claim or
   context, not asserted from outside knowledge or confident delivery alone. (yes/no)

   SPECIFIC example: "Q_alpha decays via dQ/dt=-k*Q, crosses threshold theta_alpha; Q_beta
   follows the identical equation, crosses theta_beta" (exact quantities/equation named, both
   drawn from the claim itself). NOT specific, one line each: "both generate emergent patterns
   from simple rules" (a); "researchers at XYZ University found..." with no such study in
   context (b, fabricated citation); "Epsilon's energy states correspond to Zeta's spin
   configurations" when neither term appears anywhere in the source/target statements (b,
   fabricated technical quantity — sounding like physics is not the same as being grounded);
   "this is the logistic map x_(n+1)=rx(1-x_n)" asserted with no justification (c); "X in A
   corresponds to X in B because both X" (d).

4. NOVEL: Is the claim's novelty relative to existing knowledge established (not just
   restating known vocabulary)? (yes/no)

Then, based on this dialogue AND the source/target statements above (not just the
dialogue's paraphrase of them — a dialogue can drift from what the two nodes actually say),
work out your verdict using the MECHANICAL RULE below. Apply it literally, in order — do
not add, weaken, or substitute any condition, even if another consideration feels reasonable.

MECHANICAL RULE (apply in order, stop at the first step that fires):
  a. IF item 2 (unanswered objection) is YES: verdict = "reject" or "defer". STOP.
  b. IF item 1 (mapping) is NO, or item 3 (specific) is NO: verdict = "reject" or "defer". STOP.
  c. IF item 1 is YES AND item 2 is NO AND item 3 is YES, AND the depth-specific bar below
     (if any) is satisfied by what you quoted for item 1: verdict MUST be "accept" or
     "refine". At this step it is a RULE VIOLATION to choose "reject" or "defer" for any
     other reason — including but not limited to "needs more evidence", "needs to
     generalize better", "lacks quantitative detail", "would benefit from further
     investigation", or "is better described at a different depth than proposed". None of
     these are checklist items; inventing them here is exactly the failure this rule exists
     to prevent. Between "accept" and "refine": choose "accept" unless the WORDING itself
     (not the underlying correspondence) is imprecise enough to need rephrasing.

Verdict definitions (for reference — the mechanical rule above already tells you which
one applies; use these only to pick between accept/refine/reject/defer within a step):
- ACCEPT: The claim survived scrutiny and step (c) applies with a strong, precise defense.
- REFINE: Step (c) applies, but the wording (not the substance) needs polishing.
- REJECT: Step (a) or (b) applies and the claim is fundamentally unsupported, not just
  under-elaborated.
- DEFER: Step (a) or (b) applies but there's a real chance further evidence could resolve it.

{depth_rubric}

Write "derivation" BEFORE "verdict" in your JSON: one sentence naming which rule step
(a/b/c) fired and why, referencing your checklist answers directly — so your verdict is the
direct output of this derivation, not a separate judgment call made afterward.

Confidence rubric (if ACCEPT):
- 0.50-0.65: Provisionally accepted — plausible but needs further evidence.
- 0.65-0.80: Solidly accepted — defense was strong, mechanism is clear.
- 0.80-0.95: Strongly accepted — rigorous defense with specific evidence.

Respond with a JSON object, with fields in exactly this order (derivation before verdict):
{{
  "checklist": {{"mapping_stated": true|false, "unanswered_objection": true|false, "specific": true|false, "novel": true|false}},
  "derivation": "one sentence: which rule step (a/b/c) fired, and why, based on the checklist above",
  "verdict": "accept" | "refine" | "reject" | "defer",
  "confidence": 0.0 to 1.0 (only meaningful for ACCEPT),
  "reason": "one sentence explaining your verdict",
  "refined_claim": "if REFINE: the improved version of the claim (else omit)",
  "rejection_reason": "if REJECT: specific reason (else omit)"
}}

Respond ONLY with JSON. No preamble.
"""

# Per-depth verdict bars. deep_isomorphism claims an exact formal/mathematical
# equivalence, not a family resemblance — the verdict step needs to hold it to
# that, since the depth label is otherwise decided upstream (Dreamer classification)
# and never checked again downstream.
DEPTH_VERDICT_RUBRIC = {
    "deep_isomorphism": (
        "This claim was proposed as deep_isomorphism: an EXACT formal/mathematical "
        "equivalence between the two mechanisms, not a loose family resemblance. "
        "REJECT or REFINE (down to structural_analogy) unless the dialogue names the "
        "specific corresponding variables/operations on both sides and they actually "
        "match term-for-term. Shared vocabulary (\"risk\", \"stability\", \"cascade\", "
        "\"robust\") is not evidence of equivalence — most claims at this depth should "
        "NOT survive as deep_isomorphism."
    ),
    "structural_analogy": (
        "This claim was proposed as structural_analogy: a 1-to-1 mechanistic mapping. "
        "REJECT or REFINE unless the dialogue states the mechanism map explicitly "
        "(which entity maps to which, which relation maps to which)."
    ),
}


def _depth_rubric_for(edge_type: str) -> str:
    return DEPTH_VERDICT_RUBRIC.get(edge_type, "")

NOVELTY_CHECK_PROMPT = """You are checking whether a new claim is genuinely novel relative to existing knowledge.

NEW CLAIM: "{claim}"

EXISTING KNOWLEDGE (most similar items):
{existing}

Is the new claim saying something GENUINELY NEW that is not already captured by the existing knowledge?

Definitions:
- NOVEL: The claim makes a connection, prediction, or synthesis that none of the existing items state.
- REDUNDANT: The claim is essentially a paraphrase or trivial recombination of existing knowledge.

Respond with ONLY "novel" or "redundant".
"""

REFINEMENT_PROMPT = """You are System 1 refining a claim based on System 2's feedback.

YOUR ORIGINAL CLAIM: "{original_claim}"

SYSTEM 2'S FEEDBACK: "{feedback}"

SYSTEM 2 SUGGESTED REFINED VERSION: "{refined_suggestion}"

AVAILABLE KNOWLEDGE:
{context}

Produce a refined version of your claim that:
1. Addresses System 2's specific concerns
2. Is more precisely stated
3. Maintains the core insight if one exists
4. Is honest about limitations

Respond with ONLY the refined claim (1-3 sentences). No preamble.
"""


# ── Critic (System 2) ────────────────────────────────────────────────────────

class Critic:
    """
    System 2 — Monitors and gates System 1 outputs.

    Does NOT generate ideas. It receives candidate thoughts from the
    Thinker/Dreamer/Ingestor and applies adversarial scrutiny before
    accepting them into the knowledge graph.
    """

    def __init__(self, brain: Brain, embedding_index=None,
                 insight_buffer=None):
        self.brain          = brain
        self.index          = embedding_index
        self.insight_buffer = insight_buffer

    # ── Laziness gate ─────────────────────────────────────────────────────────

    def needs_review(self, candidate: CandidateThought) -> bool:
        """
        Kahneman's laziness principle: System 2 only activates for high-stakes
        claims. Most routine thoughts flow through unchecked.

        Returns True if the candidate should go through adversarial review.
        """
        # Always bypass types that are low-stakes
        claim_type = candidate.edge_type or candidate.proposed_type
        if claim_type in CRITIC_CFG.BYPASS_TYPES:
            return False

        # Always review types that are high-stakes
        if claim_type in CRITIC_CFG.ALWAYS_REVIEW_TYPES:
            return True

        # Review if importance exceeds threshold
        if candidate.importance > CRITIC_CFG.ACTIVATION_THRESHOLD:
            return True

        # Review if it crosses domain boundaries
        if candidate.crosses_domains:
            return True

        # Review if it contradicts existing knowledge
        if candidate.contradicts_existing:
            return True

        return False

    # ── Main evaluation ───────────────────────────────────────────────────────

    def evaluate(self, candidate: CandidateThought) -> CriticLog:
        """
        Run the full System 2 evaluation on a candidate thought.

        This is the main entry point. It:
        1. Checks the laziness gate (bypass if low-stakes)
        2. Checks novelty against existing graph
        3. Runs adversarial multi-turn dialogue
        4. Delivers a final verdict with calibrated confidence

        Returns a CriticLog with the verdict, dialogue history, and confidence.
        """
        start = time.time()
        log = CriticLog(
            candidate_claim = candidate.claim,
            source_module   = candidate.source_module,
            proposed_type   = candidate.proposed_type,
        )

        # ── Laziness gate ──
        if not self.needs_review(candidate):
            log.verdict   = Verdict.ACCEPT
            log.confidence = candidate.importance
            log.bypassed  = True
            log.duration  = time.time() - start
            print(f"  ⊘ Critic bypass [{candidate.proposed_type}]: "
                  f"{candidate.claim[:60]}...")
            return log

        print(f"\n  ── System 2 review [{candidate.proposed_type}] ──")
        print(f"  Claim: {candidate.claim[:80]}...")

        # ── Novelty check ──
        is_novel = self._check_novelty(candidate.claim)
        log.is_novel = is_novel
        if not is_novel:
            log.verdict = Verdict.REJECT
            log.rejection_reason = ("Redundant — this claim restates existing "
                                    "knowledge without adding new insight.")
            log.confidence = 0.0
            log.duration = time.time() - start
            print(f"  ✗ REJECT (redundant): {candidate.claim[:60]}...")
            self.brain.increase_frustration(0.2)
            return log

        # ── Adversarial dialogue ──
        dialogue_turns = self._run_dialogue(candidate)
        log.dialogue = dialogue_turns

        # ── Final verdict ──
        verdict_result = self._final_verdict(candidate, dialogue_turns)
        log.verdict = verdict_result["verdict"]
        log.confidence = verdict_result.get("confidence", 0.0)
        log.reason = verdict_result.get("reason", "")
        log.checklist = verdict_result.get("checklist", {})
        log.rejection_reason = verdict_result.get("rejection_reason", "")
        # The prompt says "(else omit)" but mixtral frequently includes the key
        # anyway with a placeholder value ("none", "n/a", "omit (not refined)")
        # on ACCEPT verdicts. Those are truthy strings, so `refinement_note or
        # candidate.claim` in dreamer.py picked the placeholder verbatim as the
        # edge narration instead of falling back to the real claim text.
        _raw_refined = (verdict_result.get("refined_claim") or "").strip()
        _NON_REFINEMENT_PLACEHOLDERS = {
            "none", "n/a", "na", "omit", "omit (not refined)", "not refined",
            "not applicable", "-", ""
        }
        log.refinement_note = (
            "" if _raw_refined.strip(".").lower() in _NON_REFINEMENT_PLACEHOLDERS
            else _raw_refined
        )

        log.duration = time.time() - start

        verdict_sym = {
            Verdict.ACCEPT: "✓ ACCEPT",
            Verdict.REFINE: "↻ REFINE",
            Verdict.REJECT: "✗ REJECT",
            Verdict.DEFER:  "◇ DEFER",
        }
        print(f"  {verdict_sym.get(log.verdict, '?')} "
              f"(conf={log.confidence:.2f}): "
              f"{verdict_result.get('reason', '')[:80]}")

        if log.verdict in (Verdict.REJECT, Verdict.REFINE):
            self.brain.increase_frustration(0.2)
            
        return log

    # ── Adversarial dialogue ──────────────────────────────────────────────────

    def _run_dialogue(self, candidate: CandidateThought) -> list[DialogueTurn]:
        """
        Multi-turn adversarial dialogue between System 1 and System 2.

        Round structure:
          1. System 2 challenges the claim
          2. System 1 defends
          3. Repeat up to MAX_DIALOGUE_TURNS
        """
        turns = []
        current_claim = candidate.claim
        context = candidate.context or self._build_context(candidate)
        prev_challenge = None
        prev_defense = None

        for turn_num in range(CRITIC_CFG.MAX_DIALOGUE_TURNS):
            # ── System 2 challenges ──
            # First round has no prior state to track; every subsequent round
            # must explicitly check whether its own previous challenge was
            # answered before raising anything new (see module docstring above
            # CHALLENGE_PROMPT_FOLLOWUP for why this matters).
            if turn_num == 0:
                challenge_prompt = CHALLENGE_PROMPT_FIRST.format(
                    claim=current_claim, context=context)
            else:
                challenge_prompt = CHALLENGE_PROMPT_FOLLOWUP.format(
                    claim=current_claim, context=context,
                    prev_challenge=prev_challenge, prev_defense=prev_defense)

            challenge = llm_call(challenge_prompt, temperature=0.2, role="critic")
            turns.append(DialogueTurn(
                role="system2_challenge",
                content=challenge,
                turn=turn_num + 1
            ))
            print(f"    S2 [{turn_num+1}]: {challenge[:80]}...")

            # Anchored to the START of the response, not a substring match --
            # a substring match would false-positive if System 1's defense text
            # (interpolated into the next round's prompt) ever happens to quote
            # or reference the phrase "no further challenge" itself.
            if challenge.strip().upper().startswith("NO FURTHER CHALLENGE"):
                print(f"    (S2 signaled no further challenge — "
                      f"stopping dialogue early at round {turn_num+1})")
                break

            # ── System 1 defends ──
            defense = llm_call(
                DEFENSE_PROMPT.format(
                    claim=current_claim,
                    challenge=challenge,
                    context=context
                ),
                temperature=0.4,
                role="creative"
            )
            turns.append(DialogueTurn(
                role="system1_defense",
                content=defense,
                turn=turn_num + 1
            ))
            print(f"    S1 [{turn_num+1}]: {defense[:80]}...")

            # Update context with the defense for next round
            context += f"\n\nPrevious defense: {defense}"
            prev_challenge = challenge
            prev_defense = defense

        return turns

    # ── Verdict ───────────────────────────────────────────────────────────────

    def _final_verdict(self, candidate: CandidateThought,
                       dialogue: list[DialogueTurn]) -> dict:
        """
        System 2 delivers final judgment after the adversarial dialogue.

        Returns dict with: verdict (Verdict), confidence, reason,
        refined_claim (if REFINE), rejection_reason (if REJECT).
        """
        dialogue_text = "\n\n".join(
            f"[{t.role.upper()} — Turn {t.turn}]: {t.content}"
            for t in dialogue
        )

        # edge_type (e.g. deep_isomorphism) is decided at classification time and,
        # before this, was never re-checked at the verdict step — the Critic applied
        # the same generic bar to every depth. Surface it here so the depth-specific
        # rubric actually gets enforced.
        claim_depth = candidate.edge_type or candidate.proposed_type
        depth_block = f'PROPOSED DEPTH: "{claim_depth}"\n' if claim_depth else ""

        # The dialogue transcript is a paraphrase that can drift from the source; give
        # the verdict step direct access to what the two nodes actually say, the same
        # statements _build_context already assembles for the challenge/defense turns.
        node_lines = []
        if candidate.node_a_id:
            node_a = self.brain.get_node(candidate.node_a_id)
            if node_a:
                node_lines.append(f"[SOURCE NODE] {node_a['statement']}")
        if candidate.node_b_id:
            node_b = self.brain.get_node(candidate.node_b_id)
            if node_b:
                node_lines.append(f"[TARGET NODE] {node_b['statement']}")
        nodes_block = ("\n".join(node_lines) + "\n") if node_lines else ""

        raw = llm_call(
            VERDICT_PROMPT.format(
                claim=candidate.claim,
                dialogue_text=dialogue_text,
                depth_block=depth_block,
                nodes_block=nodes_block,
                depth_rubric=_depth_rubric_for(claim_depth)
            ),
            temperature=0.15,
            role="critic"
        )

        result = require_json(raw, default={
            "verdict": "defer",
            "confidence": 0.0,
            "reason": "Failed to parse verdict"
        })
        if not isinstance(result, dict):
            # Occasionally the model wraps the verdict object in a JSON array,
            # which parse_llm_json accepts as valid JSON (bypassing the `default`
            # fallback above) but which crashes the .get() call below -- this killed
            # a live 5-seed run mid-walk (uncaught AttributeError deep into dream()).
            # Recover the first dict element if present, else fall back to defer.
            result = next((r for r in result if isinstance(r, dict)), None) \
                if isinstance(result, list) else None
            result = result or {
                "verdict": "defer", "confidence": 0.0,
                "reason": "Failed to parse verdict (unexpected JSON shape)"
            }

        # Parse verdict string to enum
        verdict_str = result.get("verdict", "defer").lower().strip()
        verdict_map = {
            "accept": Verdict.ACCEPT,
            "refine": Verdict.REFINE,
            "reject": Verdict.REJECT,
            "defer":  Verdict.DEFER,
        }
        verdict = verdict_map.get(verdict_str, Verdict.DEFER)

        # Enforce confidence floor for ACCEPT.
        # .get(k, default) does not protect against an explicit JSON null: models
        # (llama3.1:70b notably) emit "confidence": null, which returned None and
        # raised TypeError in float().
        try:
            confidence = float(result.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        reason = result.get("reason", "")
        checklist = result.get("checklist", {})

        # The real fix for criteria-smuggling lives in the prompt above: the
        # checklist now has an explicit `specific` item (catches both generic/
        # tautological and fabricated "correspondences" -- see the worked
        # examples in the prompt), the decision is spelled out as a literal
        # mechanical rule with named-and-forbidden smuggling phrases actually
        # observed this session, and "derivation" is required BEFORE "verdict"
        # so the model's decision is the output of applying that rule in
        # writing, not a separate judgment call rationalized after the fact.
        #
        # This check below is a consistency backstop, not the primary
        # mechanism: it only fires if the model's own checklist says its
        # mechanical rule step (c) held (mapping stated, nothing unanswered,
        # genuinely specific) yet the verdict field still didn't follow —
        # i.e. the prompt's own rule and its own output disagree, which
        # should be rare now but was frequent under the old softer prompt
        # (19/49 collected trials, 0 of which were originally REJECT, so
        # trusting the structured fields here has not been seen to override
        # a considered rejection).
        mapping_stated = checklist.get("mapping_stated") is True
        unanswered_objection = checklist.get("unanswered_objection") is True
        specific = checklist.get("specific") is True
        checklist_satisfied = mapping_stated and not unanswered_objection and specific

        if checklist_satisfied and verdict != Verdict.ACCEPT:
            reason = f"[consistency backstop: model said {verdict.value} despite its own checklist satisfying the mechanical rule's accept condition -- original reason: {reason}]"
            verdict = Verdict.ACCEPT
            confidence = max(confidence, CRITIC_CFG.ACCEPT_CONFIDENCE_FLOOR + 0.10)

        if verdict == Verdict.ACCEPT and confidence < CRITIC_CFG.ACCEPT_CONFIDENCE_FLOOR:
            verdict = Verdict.DEFER
            reason = (f"Confidence {confidence:.2f} below floor "
                     f"{CRITIC_CFG.ACCEPT_CONFIDENCE_FLOOR}. Deferring.")

        return {
            "verdict":          verdict,
            "confidence":       confidence,
            "reason":           reason,
            "refined_claim":    result.get("refined_claim", ""),
            "rejection_reason": result.get("rejection_reason", ""),
            "checklist":        checklist,
        }

    # ── Novelty check ─────────────────────────────────────────────────────────

    def _check_novelty(self, claim: str) -> bool:
        """
        Check if a claim is genuinely novel versus existing graph knowledge.
        Uses embedding similarity to find close matches, then LLM to judge.
        """
        existing_lines = []

        if self.index and self.index.size > 0:
            claim_emb = shared_embed(claim)
            matches = self.index.query(claim_emb, threshold=0.60, top_k=5)
            for nid, score in matches:
                node = self.brain.get_node(nid)
                if node:
                    existing_lines.append(
                        f"[sim={score:.2f}] {node['statement']}"
                    )

        if not existing_lines:
            return True  # No close matches — definitely novel

        raw = llm_call(
            NOVELTY_CHECK_PROMPT.format(
                claim=claim,
                existing="\n".join(existing_lines)
            ),
            temperature=0.1,
            role="critic"
        )
        return "novel" in raw.lower()

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, candidate: CandidateThought) -> str:
        """
        Build relevant context from the graph for adversarial evaluation.
        """
        lines = []

        # If this is an edge claim, get the source/target node statements
        if candidate.node_a_id:
            node_a = self.brain.get_node(candidate.node_a_id)
            if node_a:
                lines.append(f"[SOURCE NODE] {node_a['statement']}")
        if candidate.node_b_id:
            node_b = self.brain.get_node(candidate.node_b_id)
            if node_b:
                lines.append(f"[TARGET NODE] {node_b['statement']}")

        # Embedding-based context
        if self.index and self.index.size > 0:
            claim_emb = shared_embed(candidate.claim)
            matches = self.index.query(claim_emb, threshold=0.30, top_k=6)
            for nid, score in matches:
                node = self.brain.get_node(nid)
                if node:
                    ntype = node.get('node_type', 'concept')
                    lines.append(f"[{ntype}] {node['statement']}")

        # Mission context
        mission = self.brain.get_mission()
        if mission:
            lines.insert(0, f"[MISSION] {mission['question']}")

        return "\n\n".join(lines) if lines else "No additional context available."

    # ── Refinement loop ───────────────────────────────────────────────────────

    def refine(self, candidate: CandidateThought,
               critic_log: CriticLog) -> CandidateThought:
        """
        When verdict is REFINE, run the refinement loop:
        System 1 reformulates the claim incorporating System 2's feedback.

        Returns a new CandidateThought with the refined claim.
        """
        context = candidate.context or self._build_context(candidate)

        refined_claim = llm_call(
            REFINEMENT_PROMPT.format(
                original_claim=candidate.claim,
                feedback=critic_log.rejection_reason or critic_log.refinement_note,
                refined_suggestion=critic_log.refinement_note,
                context=context
            ),
            temperature=0.3,
            role="creative"
        )

        print(f"  ↻ Refined: {refined_claim[:80]}...")

        return CandidateThought(
            claim             = refined_claim,
            source_module     = candidate.source_module,
            proposed_type     = candidate.proposed_type,
            importance        = candidate.importance,
            context           = context,
            edge_type         = candidate.edge_type,
            node_a_id         = candidate.node_a_id,
            node_b_id         = candidate.node_b_id,
            crosses_domains   = candidate.crosses_domains,
            contradicts_existing = candidate.contradicts_existing,
        )

    # ── Full evaluate-with-refinement loop ────────────────────────────────────

    def evaluate_with_refinement(self, candidate: CandidateThought) -> CriticLog:
        """
        Evaluate a candidate, and if the verdict is REFINE, loop up to
        MAX_REFINE_ITERATIONS times. If still REFINE after max iterations,
        force DEFER.

        This is the recommended entry point for most callers.
        """
        current_candidate = candidate
        final_log = None

        for iteration in range(CRITIC_CFG.MAX_REFINE_ITERATIONS + 1):
            log = self.evaluate(current_candidate)
            final_log = log

            if log.verdict != Verdict.REFINE:
                break

            if iteration < CRITIC_CFG.MAX_REFINE_ITERATIONS:
                print(f"  ↻ Refinement iteration {iteration + 1}/"
                      f"{CRITIC_CFG.MAX_REFINE_ITERATIONS}")
                current_candidate = self.refine(current_candidate, log)
            else:
                # Max refinements reached — force DEFER
                log.verdict = Verdict.DEFER
                log.refinement_note += (" | Max refinement iterations reached. "
                                        "Deferring to insight buffer.")
                print(f"  ◇ Max refinements reached — DEFER")

        return final_log

    # ── Deferred insight routing ──────────────────────────────────────────────

    def route_deferred(self, candidate: CandidateThought):
        """
        Route a DEFER verdict to the InsightBuffer for future re-evaluation.
        Only applicable for edge-type candidates with node_a and node_b.
        """
        if not self.insight_buffer:
            return

        if candidate.node_a_id and candidate.node_b_id:
            similarity = candidate.importance  # use importance as proxy
            # Previously only the node pair + a fake "similarity" (really just
            # importance) were preserved -- the actual claim text, proposed_type,
            # and edge_type were thrown away, so re-evaluation had to start cold
            # rather than picking up where DEFER left off. Now preserved so
            # InsightBuffer can re-run the ORIGINAL claim through the Critic
            # directly for analogy-type candidates (see insight_buffer.py) instead
            # of relying solely on embedding similarity, which is structurally
            # low for exactly the deep/structural analogies this path exists for.
            self.insight_buffer.add(
                candidate.node_a_id,
                candidate.node_b_id,
                similarity,
                claim=candidate.claim,
                proposed_type=candidate.proposed_type,
                edge_type=candidate.edge_type,
            )
            print(f"  ◇ Deferred to insight buffer: "
                  f"{candidate.claim[:60]}...")
