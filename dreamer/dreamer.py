import random
import time
import json
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from ollama import Client
from graph.brain import (Brain, Edge, EdgeType, EdgeSource,
                         NodeStatus, NodeType, ANALOGY_WEIGHTS)

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL        = "llama3.1:8b"
DEFAULT_STEPS       = 20
DEFAULT_TEMP        = 0.7
DEPTH_STEPS         = 3
VISITED_PENALTY     = 0.25
QUESTION_DEDUP_HIGH = 0.90
QUESTION_DEDUP_LOW  = 0.70

# ── Dream Mode ────────────────────────────────────────────────────────────────

class DreamMode(str, Enum):
    WANDERING = "wandering"
    PRESSURE  = "pressure"
    SEEDED    = "seeded"

# ── Dream Step ────────────────────────────────────────────────────────────────

@dataclass
class DreamStep:
    step:            int
    from_id:         str
    to_id:           str
    edge_type:       str
    edge_narration:  str
    narration:       str
    question:        str  = ""
    is_insight:      bool = False
    insight_depth:   str  = ""    # "surface"|"structural"|"isomorphism"
    new_edge:        bool = False
    answer_match:    str  = "none"
    answer_detail:   str  = ""
    depth_triggered: bool = False
    mission_advance: bool = False  # did this step advance the central question?

# ── Dream Log ─────────────────────────────────────────────────────────────────

@dataclass
class DreamLog:
    mode:            str
    started_at:      float = field(default_factory=time.time)
    steps:           list  = field(default_factory=list)
    questions:       list  = field(default_factory=list)
    insights:        list  = field(default_factory=list)
    answers:         list  = field(default_factory=list)
    mission_advances: list = field(default_factory=list)
    summary:         str   = ""

    def to_dict(self):
        return {
            "mode":             self.mode,
            "started_at":       self.started_at,
            "steps":            [s.__dict__ for s in self.steps],
            "questions":        self.questions,
            "insights":         self.insights,
            "answers":          self.answers,
            "mission_advances": self.mission_advances,
            "summary":          self.summary
        }

# ── Prompts ───────────────────────────────────────────────────────────────────

NARRATION_PROMPT = """
You are a dreaming scientific mind, moving between ideas.

Central research question you are working on:
"{mission}"

You just moved from this idea:
"{from_node}"

Through a connection of type [{edge_type}]:
"{edge_narration}"

To this idea:
"{to_node}"

In 2-4 sentences, narrate this mental journey. Think like a scientist half-asleep —
associative, curious, sometimes surprising. Let the central question color your thinking
without forcing it.

Then, if something feels unresolved, ask ONE question starting with "Q:"

Classify any insight using these precise levels:
- INSIGHT: surface — shared vocabulary or theme only
- INSIGHT: structural — same relational pattern between different domains
- INSIGHT: isomorphism — formal mathematical or logical equivalence

If no insight, write: INSIGHT: none
"""

MISSION_ADVANCE_PROMPT = """
Central research question: "{mission}"

During a dream, the mind just encountered this idea:
"{node}"

And made this connection:
"{narration}"

Does this meaningfully advance the central research question — bringing it
closer to resolution, revealing a new angle, or deepening the problem?

Respond with a JSON object:
{{
  "advances": true or false,
  "explanation": "one sentence",
  "strength": a float 0.0 to 1.0
}}

Respond ONLY with JSON.
"""

ANSWER_CHECK_PROMPT = """
Current idea: {current_node}
Open question: {question}

Does the current idea answer or significantly advance this question?

Respond with JSON:
{{
  "match": one of ["none", "partial", "strong"],
  "explanation": "one sentence"
}}

Respond ONLY with JSON.
"""

QUESTION_SIMILAR_PROMPT = """
Are these two questions essentially the same, even if worded differently?
Q1: {q1}
Q2: {q2}
Respond ONLY "yes" or "no".
"""

DEPTH_NARRATION_PROMPT = """
You are a scientific mind that found something significant while dreaming.

Central research question: "{mission}"

You landed on: "{node}"
It connects to an open question: "{question}"
Connection: {explanation}

In 2-3 sentences, explore this deeply. What does it mean for the central question?
End with one follow-up question starting with "Q:"
"""

SUMMARY_PROMPT = """
You are summarizing a dream cycle of a scientific mind.

Central research question: "{mission}"

Dream content:
{steps}

Answer matches found:
{answers}

Mission advances:
{mission_advances}

Write a 4-6 sentence morning notebook entry. Address:
1. What the dream explored
2. Key connections made
3. Whether the central question was advanced — and how
4. What questions remain most pressing

Write it as a scientist's notebook entry — first person, specific, honest about uncertainty.
"""

START_NODE_PROMPT = """
Central research question: "{mission}"

Select a starting node for a dream that would most fruitfully explore the central question
through unexpected associations — not the most obvious node, but the most generative.

Nodes:
{nodes}

Respond with ONLY the node ID.
"""

# ── Dreamer ───────────────────────────────────────────────────────────────────

class Dreamer:
    def __init__(self, brain: Brain, research_agenda=None):
        self.brain           = brain
        self.llm             = Client()
        self.research_agenda = research_agenda
        self.embedder        = SentenceTransformer('all-MiniLM-L6-v2')

    def _llm(self, prompt: str) -> str:
        response = self.llm.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, normalize_embeddings=True)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _mission_text(self) -> str:
        m = self.brain.get_mission()
        return m['question'] if m else "No central question set."

    # ── Question deduplication ────────────────────────────────────────────────

    def _is_duplicate_question(self, new_q: str,
                                existing: list,
                                embeddings: list) -> bool:
        if not existing:
            return False
        new_emb = self._embed(new_q)
        for eq, emb in zip(existing, embeddings):
            sim = self._cosine(new_emb, emb)
            if sim > QUESTION_DEDUP_HIGH:
                return True
            if sim > QUESTION_DEDUP_LOW:
                raw = self._llm(QUESTION_SIMILAR_PROMPT.format(
                    q1=new_q, q2=eq))
                if raw.lower().startswith('yes'):
                    return True
        return False

    def _add_question(self, q: str, questions: list,
                      q_embeddings: list) -> bool:
        if not q:
            return False
        if self._is_duplicate_question(q, questions, q_embeddings):
            return False
        questions.append(q)
        q_embeddings.append(self._embed(q))
        return True

    # ── Node selection ────────────────────────────────────────────────────────

    def _select_start_node(self, mode: DreamMode,
                           seed_id: str = None) -> str:
        nodes = self.brain.all_nodes()
        if not nodes:
            raise ValueError("Brain is empty")

        if mode == DreamMode.SEEDED and seed_id:
            return seed_id

        if mode == DreamMode.PRESSURE:
            # prefer nodes linked to mission + high incubation or contradiction
            mission = self.brain.get_mission()
            mission_id = mission['id'] if mission else None

            candidates = [
                (nid, data) for nid, data in nodes
                if data.get('status') in [
                    NodeStatus.CONTRADICTED.value,
                    NodeStatus.UNCERTAIN.value
                ] and data.get('node_type') != NodeType.MISSION.value
            ]
            if candidates:
                def pressure_score(x):
                    nid, data = x
                    score = (data.get('incubation_age', 0) * 2 +
                             self.brain.graph.degree(nid))
                    # bonus for nodes linked to mission
                    if mission_id and self.brain.graph.has_edge(nid, mission_id):
                        score += 5
                    return score
                return max(candidates, key=pressure_score)[0]

        if mode == DreamMode.WANDERING:
            # exclude mission node from random sample
            sample_pool = [(nid, d) for nid, d in nodes
                           if d.get('node_type') != NodeType.MISSION.value]
            sample = random.sample(sample_pool, min(8, len(sample_pool)))
            node_list = "\n".join(
                f"{nid}: {data['statement'][:100]}"
                for nid, data in sample
            )
            chosen = self._llm(START_NODE_PROMPT.format(
                mission=self._mission_text(),
                nodes=node_list
            )).strip()
            if chosen in dict(nodes):
                return chosen

        non_mission = [(nid, d) for nid, d in nodes
                       if d.get('node_type') != NodeType.MISSION.value]
        return random.choice(non_mission)[0] if non_mission else nodes[0][0]

    # ── Edge scoring ──────────────────────────────────────────────────────────

    def _score_edge(self, edge_data: dict, temperature: float,
                    scientificness: float, visited: set,
                    target_id: str) -> float:
        weight = edge_data.get('weight', 0.5)
        etype  = edge_data.get('type', '')

        # analogy type weighting — isomorphisms score higher
        if etype in ANALOGY_WEIGHTS:
            weight = max(weight, ANALOGY_WEIGHTS.get(
                EdgeType(etype) if etype in [e.value for e in EdgeType]
                else EdgeType.ANALOGOUS_TO, 0.4))

        logical_types = {EdgeType.SUPPORTS.value, EdgeType.CAUSES.value,
                         EdgeType.CONTRADICTS.value}
        if etype in logical_types:
            weight += scientificness * 0.3
        if etype == EdgeType.ASSOCIATED.value:
            weight += (1 - scientificness) * 0.3

        # mission edges get a bonus — dreaming toward the question
        if etype == EdgeType.TOWARD_MISSION.value:
            weight += 0.3

        if target_id in visited:
            weight -= VISITED_PENALTY

        noise = random.gauss(0, temperature * 0.3)
        return max(0.001, weight + noise)

    # ── Single hop ────────────────────────────────────────────────────────────

    def _hop(self, current_id: str, temperature: float,
             scientificness: float, visited: set) -> tuple:
        neighbors = self.brain.neighbors(current_id)
        if not neighbors:
            all_ids   = [nid for nid, _ in self.brain.all_nodes()]
            unvisited = [n for n in all_ids if n not in visited]
            return random.choice(unvisited if unvisited else all_ids), None

        scored = []
        for nid in neighbors:
            edge = self.brain.get_edge(current_id, nid)
            if edge:
                score = self._score_edge(
                    edge, temperature, scientificness, visited, nid)
                scored.append((nid, edge, score))

        if not scored:
            return random.choice(neighbors), None

        total = sum(s for _, _, s in scored)
        roll  = random.uniform(0, total)
        cumulative = 0
        for nid, edge, score in scored:
            cumulative += score
            if cumulative >= roll:
                return nid, edge

        return scored[-1][0], scored[-1][1]

    # ── Parse narration ───────────────────────────────────────────────────────

    def _parse_narration(self, raw: str) -> tuple:
        question    = ""
        is_insight  = False
        insight_depth = ""
        lines       = raw.strip().split('\n')
        clean       = []

        for line in lines:
            if line.startswith("Q:"):
                question = line[2:].strip()
            elif line.upper().startswith("INSIGHT:"):
                rest = line.split(":", 1)[1].strip().lower()
                if rest == "none":
                    is_insight = False
                else:
                    is_insight    = True
                    insight_depth = rest  # "surface"|"structural"|"isomorphism"
            else:
                clean.append(line)

        return " ".join(clean).strip(), question, is_insight, insight_depth

    # ── Answer detection ──────────────────────────────────────────────────────

    def _check_answers(self, node_id: str, node_data: dict) -> tuple:
        if not self.research_agenda:
            return "none", "", ""
        open_items = self.research_agenda.get_prioritized_questions(15)
        for item in open_items:
            raw = self._llm(ANSWER_CHECK_PROMPT.format(
                current_node=node_data['statement'],
                question=item.text
            ))
            try:
                result = json.loads(raw)
                match  = result.get('match', 'none')
                expl   = result.get('explanation', '')
                if match in ['partial', 'strong']:
                    return match, expl, item.text
            except (json.JSONDecodeError, ValueError):
                continue
        return "none", "", ""

    # ── Mission advance check ─────────────────────────────────────────────────

    def _check_mission_advance(self, node_data: dict,
                               narration: str) -> tuple:
        mission = self.brain.get_mission()
        if not mission:
            return False, "", 0.0
        raw = self._llm(MISSION_ADVANCE_PROMPT.format(
            mission  = mission['question'],
            node     = node_data['statement'],
            narration= narration
        ))
        try:
            result = json.loads(raw)
            if result.get('advances') and result.get('strength', 0) > 0.5:
                return True, result.get('explanation', ''), result.get('strength', 0.5)
        except (json.JSONDecodeError, ValueError):
            pass
        return False, "", 0.0

    # ── Depth exploration ─────────────────────────────────────────────────────

    def _depth_explore(self, node_id: str, node_data: dict,
                       question: str, explanation: str,
                       temperature: float, scientificness: float,
                       visited: set, log: DreamLog,
                       questions: list, q_embeddings: list,
                       step_offset: int) -> str:
        print(f"      ↳ Depth [{DEPTH_STEPS} steps]")

        raw = self._llm(DEPTH_NARRATION_PROMPT.format(
            mission    = self._mission_text(),
            node       = node_data['statement'],
            question   = question,
            explanation= explanation
        ))
        lines    = raw.strip().split('\n')
        q_line   = next((l for l in lines if l.startswith('Q:')), "")
        followup = q_line[2:].strip() if q_line else ""
        self._add_question(followup, questions, q_embeddings)

        current_id   = node_id
        current_data = node_data

        for d in range(DEPTH_STEPS):
            next_id, edge = self._hop(
                current_id, temperature * 0.5, scientificness, visited)
            next_data = self.brain.get_node(next_id)
            if not next_data:
                continue

            edge_type      = edge.get('type', 'associated') if edge else 'associated'
            edge_narration = edge.get('narration', '') if edge else ''

            raw = self._llm(NARRATION_PROMPT.format(
                mission        = self._mission_text(),
                from_node      = current_data['statement'],
                edge_type      = edge_type,
                edge_narration = edge_narration,
                to_node        = next_data['statement']
            ))
            narration, _, is_insight, depth = self._parse_narration(raw)

            ds = DreamStep(
                step           = step_offset + d,
                from_id        = current_id,
                to_id          = next_id,
                edge_type      = edge_type,
                edge_narration = edge_narration,
                narration      = narration,
                is_insight     = is_insight,
                insight_depth  = depth
            )
            log.steps.append(ds)
            visited.add(next_id)
            self.brain.update_node(next_id, activated_at=time.time())
            current_id   = next_id
            current_data = next_data
            print(f"      depth {d+1}: {next_data['statement'][:60]}...")

        return current_id

    # ── NREM ─────────────────────────────────────────────────────────────────

    def nrem_pass(self):
        print("\n── NREM pass ──")
        self.brain.proximal_reinforce()
        print("── NREM complete ──\n")

    # ── Main dream loop ───────────────────────────────────────────────────────

    def dream(self,
              mode:        DreamMode = DreamMode.WANDERING,
              steps:       int       = DEFAULT_STEPS,
              temperature: float     = DEFAULT_TEMP,
              seed_id:     str       = None,
              run_nrem:    bool      = True,
              log_path:    str       = "logs/dream_latest.json") -> DreamLog:

        scientificness = self.brain.scientificness
        log            = DreamLog(mode=mode.value)
        visited        = set()
        questions      = []
        q_embeddings   = []
        mission        = self._mission_text()

        if run_nrem:
            self.nrem_pass()

        print(f"\n── REM [{mode.value}] steps={steps} temp={temperature} ──")
        print(f"   Mission: {mission[:70]}...\n")

        current_id = self._select_start_node(mode, seed_id)
        current    = self.brain.get_node(current_id)
        visited.add(current_id)
        print(f"   Start: {current['statement'][:80]}...\n")

        step = 0
        while step < steps:
            next_id, edge = self._hop(
                current_id, temperature, scientificness, visited)
            next_node = self.brain.get_node(next_id)
            if not next_node:
                step += 1
                continue

            # skip mission node — don't narrate it, just pass through
            if next_node.get('node_type') == NodeType.MISSION.value:
                visited.add(next_id)
                step += 1
                continue

            edge_type      = edge.get('type', 'associated') if edge else 'associated'
            edge_narration = edge.get('narration', '') if edge else ''

            raw = self._llm(NARRATION_PROMPT.format(
                mission        = mission,
                from_node      = current['statement'],
                edge_type      = edge_type,
                edge_narration = edge_narration,
                to_node        = next_node['statement']
            ))
            narration, question, is_insight, insight_depth = \
                self._parse_narration(raw)

            self._add_question(question, questions, q_embeddings)

            # answer detection
            match_grade, match_explanation, matched_q = \
                self._check_answers(next_id, next_node)

            # mission advance check
            mission_advance = False
            mission_explanation = ""
            mission_strength = 0.0
            if is_insight or match_grade == 'strong':
                mission_advance, mission_explanation, mission_strength = \
                    self._check_mission_advance(next_node, narration)

            # salience — depth on answer match
            depth_triggered = False
            if match_grade in ['partial', 'strong'] and matched_q:
                depth_triggered = True
                current_id = self._depth_explore(
                    next_id, next_node, matched_q, match_explanation,
                    temperature, scientificness, visited, log,
                    questions, q_embeddings, step_offset=step + 1000)
                current = self.brain.get_node(current_id)
                visited.add(current_id)

            elif (is_insight or
                  next_node.get('status') == NodeStatus.CONTRADICTED.value or
                  next_node.get('incubation_age', 0) > 3 or
                  mission_advance):
                depth_triggered = True
                current_id = self._depth_explore(
                    next_id, next_node,
                    "An interesting connection worth deeper exploration.",
                    mission_explanation or "Insight or tension detected.",
                    temperature, scientificness, visited, log,
                    questions, q_embeddings, step_offset=step + 1000)
                current = self.brain.get_node(current_id)
                visited.add(current_id)

            # new edge on insight
            new_edge = False
            if is_insight and insight_depth:
                type_map = {
                    "surface":     EdgeType.SURFACE_ANALOGY,
                    "structural":  EdgeType.STRUCTURAL_ANALOGY,
                    "isomorphism": EdgeType.DEEP_ISOMORPHISM,
                }
                etype = type_map.get(insight_depth, EdgeType.STRUCTURAL_ANALOGY)

                if not (self.brain.graph.has_edge(current_id, next_id) or
                        self.brain.graph.has_edge(next_id, current_id)):
                    dream_edge = Edge(
                        type          = etype,
                        narration     = narration,
                        weight        = ANALOGY_WEIGHTS.get(etype, 0.4),
                        confidence    = 0.45,
                        source        = EdgeSource.DREAM,
                        analogy_depth = insight_depth
                    )
                    self.brain.add_edge(current_id, next_id, dream_edge)
                    new_edge = True

                self.brain.restructure_around_insight(
                    current_id, next_id, narration, edge_type=etype.value)
                log.insights.append({
                    "step":          step,
                    "from":          current['statement'][:80],
                    "to":            next_node['statement'][:80],
                    "narration":     narration,
                    "depth":         insight_depth,
                    "mission_linked": mission_advance
                })

            if match_grade != 'none':
                log.answers.append({
                    "step":        step,
                    "node":        next_node['statement'][:80],
                    "question":    matched_q[:80],
                    "grade":       match_grade,
                    "explanation": match_explanation
                })
                if self.research_agenda:
                    self.research_agenda.record_answer(
                        matched_q, next_id,
                        match_explanation, grade=match_grade)

            if mission_advance:
                self.brain.link_to_mission(
                    next_id,
                    f"Dream insight: {mission_explanation[:80]}",
                    strength=mission_strength)
                log.mission_advances.append({
                    "step":        step,
                    "node":        next_node['statement'][:80],
                    "explanation": mission_explanation,
                    "strength":    mission_strength
                })
                if self.research_agenda:
                    self.research_agenda.record_mission_advance(
                        next_id, mission_explanation, mission_strength)

            ds = DreamStep(
                step           = step,
                from_id        = current_id,
                to_id          = next_id,
                edge_type      = edge_type,
                edge_narration = edge_narration,
                narration      = narration,
                question       = question,
                is_insight     = is_insight,
                insight_depth  = insight_depth,
                new_edge       = new_edge,
                answer_match   = match_grade,
                answer_detail  = match_explanation,
                depth_triggered= depth_triggered,
                mission_advance= mission_advance
            )
            log.steps.append(ds)
            self.brain.update_node(next_id, activated_at=time.time())
            visited.add(next_id)

            # print
            indicator = f"[{insight_depth[0].upper() if insight_depth else ''}]" if is_insight else ""
            print(f"   Step {step+1:02d} [{edge_type[:8]}]: "
                  f"{next_node['statement'][:55]}...")
            if question:
                print(f"            Q: {question[:65]}")
            if is_insight:
                print(f"            ✦ INSIGHT:{insight_depth} {indicator}")
            if match_grade != 'none':
                print(f"            ◎ [{match_grade}]: {match_explanation[:55]}")
            if mission_advance:
                print(f"            ★ MISSION ({mission_strength:.2f}): "
                      f"{mission_explanation[:55]}")
            if depth_triggered:
                print(f"            ↳ Depth")

            if not depth_triggered:
                current_id = next_id
                current    = next_node

            step += 1

        log.questions = questions

        # summarize with mission awareness
        step_text = "\n".join(
            f"- {s.narration[:120]}" for s in log.steps if s.narration)
        answer_text = "\n".join(
            f"- [{a['grade']}] {a['explanation']}"
            for a in log.answers) or "none"
        mission_text = "\n".join(
            f"- ({m['strength']:.2f}) {m['explanation']}"
            for m in log.mission_advances) or "none"

        log.summary = self._llm(SUMMARY_PROMPT.format(
            mission         = mission,
            steps           = step_text,
            answers         = answer_text,
            mission_advances= mission_text
        ))

        import os
        os.makedirs("logs", exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(log.to_dict(), f, indent=2)

        print(f"\n── Dream complete ──")
        print(f"   Steps:{len(log.steps)} Qs:{len(log.questions)} "
              f"Insights:{len(log.insights)} "
              f"Answers:{len(log.answers)} "
              f"Mission advances:{len(log.mission_advances)}")
        print(f"\n── Summary ──\n{log.summary}\n")

        return log