import json
import time
import os
from dataclasses import dataclass, field
from graph.brain import Brain, NodeType, EdgeType
from persistence import atomic_write_json
from llm_utils import llm_call

# ── Config ────────────────────────────────────────────────────────────────────

NOTEBOOK_PATH  = "data/notebook.json"
SCIENTIST_NAME = "THE SCIENTIST"

# ── Entry types ───────────────────────────────────────────────────────────────

ENTRY_MORNING       = "morning"      # after dream cycle
ENTRY_FIELD_NOTES   = "field_notes"  # after research day
ENTRY_EVENING       = "evening"      # after consolidation
ENTRY_HYPOTHESIS    = "hypothesis"   # running best answer to mission
ENTRY_BREAKTHROUGH  = "breakthrough" # flagged manually or by strong emergence
ENTRY_SYNTHESIS     = "synthesis"    # writing phase — forces clarity

# ── Prompts ───────────────────────────────────────────────────────────────────

MORNING_ENTRY_PROMPT = """
You are {name}, a research scientist keeping a personal scientific journal.

Your central research mission:
"{mission}"

Last night's associative dream walk summary:
{dream_summary}

Mission advances identified during dreaming:
{mission_advances}

Key structural/isomorphic insights (with depth):
{insights}

Emergent questions generated:
{questions}

Write a morning notebook entry. Be intellectually rigorous, direct, and insightful:
1. Articulate what fundamental mechanisms or connections the dream exploration revealed.
2. Examine whether any associative links represent genuine structural isomorphisms or physical principles rather than surface analogies.
3. Define the most critical open theoretical or empirical question that demands priority investigation today.
4. Reflect on the evolving conceptual framework of your research.

Write 4-6 dense, substantive sentences. Sign off as: — {name}
"""

FIELD_NOTES_PROMPT = """
You are {name}, an active researcher recording field notes after a focused research investigation.

Your central research question:
"{mission}"

Today's research findings & literature synthesis:
{findings}

Questions resolved or advanced:
{resolved}

New conceptual nodes and avenues opened:
{new_questions}

Write an incisive field notes entry. Be precise and scientifically grounded:
- Note the key mechanistic findings, unexpected relationships, or empirical constraints discovered.
- Assess how these findings alter or sharpen your mathematical, physical, or biological understanding of the central question.
- Explicitly highlight unresolved theoretical gaps or boundary conditions revealed by the literature.

Write 4-5 focused sentences. Sign off as: — {name}
"""

EVENING_ENTRY_PROMPT = """
You are {name}, a scientist synthesizing the day's consolidation results in your research journal.

Your central research question:
"{mission}"

Consolidation results:
- Near-duplicate concepts merged: {merges}
- Novel cross-domain syntheses created: {syntheses}
- Higher-order domain abstractions identified: {abstractions}
- Theoretical and causal gaps inferred: {gaps}
- Contradictions currently active in graph: {contradictions}

Active brain state: {brain_stats}

Write a thoughtful evening reflection on the architecture's evolving knowledge base:
- What unifying principles or structural invariants emerged today?
- What genuine contradictions or trade-offs remain unresolved, and what mechanistic explanations could reconcile them?
- What overarching theory or paradigm is the knowledge network converging toward?

Write 4-5 substantive sentences. Sign off as: — {name}
"""

RUNNING_HYPOTHESIS_PROMPT = """
You are {name}, an advanced scientist formulating the running hypothesis for the research mission:

"{mission}"

Current synthesis of knowledge and evidence accumulated across research cycles:

Mission progress synthesis:
{progress_summary}

Most significant mission advances:
{advances}

Strongest structural/isomorphic insights found:
{insights}

Active hypotheses in the knowledge graph:
{hypotheses}

Key unresolved contradictions and tensions:
{contradictions}

Synthesize the most rigorous, state-of-the-art answer to the central research question:

Scientific Guidelines:
1. Ground your reasoning in established physical, mathematical, and biological principles as well as the accumulated graph insights and empirical simulations.
2. Clearly delineate between:
   - What is firmly established by scientific laws, theorems, and evidence (do not feign ignorance about established facts).
   - What has been quantitatively supported or tested in this investigation.
   - What remains a genuine, unsolved scientific frontier.
3. Formulate a precise, testable, and falsifiable working hypothesis addressing the core open problem, specifying the exact physical mechanisms, mathematical dependencies, or observable conditions required for it to hold.

Write 5-7 clear, authoritative, and scientifically rigorous sentences.
Sign off as: — {name}
"""

BREAKTHROUGH_PROMPT = """
You are {name}, a scientist recording a pivotal breakthrough in your research journal.

Your central research question:
"{mission}"

Breakthrough event details:
{detail}

Write an incisive breakthrough entry:
- What fundamental insight, structural isomorphism, or empirical confirmation occurred?
- Why does this pivotally advance or reframe the central research question?
- What immediate theoretical formalization or experimental validation must be executed next?

Write 3-5 sharp, technically precise sentences. Sign off as: — {name}
"""

# ── Notebook entry ────────────────────────────────────────────────────────────

@dataclass
class NotebookEntry:
    entry_type:  str
    content:     str
    cycle:       int
    timestamp:   float = field(default_factory=time.time)
    tags:        list  = field(default_factory=list)

    def to_dict(self):
        return self.__dict__

# ── Notebook ──────────────────────────────────────────────────────────────────

class Notebook:
    def __init__(self, brain: Brain, observer=None,
                 scientist_name: str = SCIENTIST_NAME,
                 path: str = NOTEBOOK_PATH):
        self.brain          = brain
        self.observer       = observer
        self.name           = scientist_name
        self.path           = path
        self.entries: list[NotebookEntry] = []
        self.running_hypothesis: str = ""
        self._load()

    def _llm(self, prompt: str, temperature: float = 0.6) -> str:
        return llm_call(prompt, temperature=temperature, role="creative")

    def _mission(self) -> str:
        m = self.brain.get_mission()
        return m['question'] if m else "No central question set."

    def _add_entry(self, entry_type: str, content: str,
                   cycle: int, tags: list = None) -> NotebookEntry:
        entry = NotebookEntry(
            entry_type = entry_type,
            content    = content,
            cycle      = cycle,
            tags       = tags or []
        )
        self.entries.append(entry)
        self._save()
        return entry

    # ── Entry writers ─────────────────────────────────────────────────────────

    def write_morning_entry(self, dream_log, cycle: int) -> str:
        """Write a morning entry after a dream cycle."""
        mission_advances = "\n".join(
            f"- ({a['strength']:.2f}) {a['explanation']}"
            for a in dream_log.mission_advances
        ) or "none"

        insights = "\n".join(
            f"- [{i['depth']}] {i['narration']}"
            for i in dream_log.insights
        ) or "none"

        questions = "\n".join(
            f"- {q}" for q in dream_log.questions[:8]
        ) or "none"

        content = self._llm(MORNING_ENTRY_PROMPT.format(
            name             = self.name,
            mission          = self._mission(),
            dream_summary    = dream_log.summary,
            mission_advances = mission_advances,
            insights         = insights,
            questions        = questions
        ))

        entry = self._add_entry(
            ENTRY_MORNING, content, cycle,
            tags=["dream", f"insights:{len(dream_log.insights)}",
                  f"advances:{len(dream_log.mission_advances)}"]
        )
        print(f"\n── Notebook: morning entry written ──")
        return content

    def write_field_notes(self, research_log, cycle: int) -> str:
        """Write field notes after a research day."""
        findings = "\n".join(
            f"- Q: {e.question}\n  Found: {', '.join(e.sources[:2])}"
            for e in research_log.entries
        ) or "none"

        resolved = "\n".join(
            f"- [{e.resolved}] {e.question}"
            for e in research_log.entries
            if e.resolved in ['partial', 'strong']
        ) or "none"

        new_qs = sum(len(getattr(e, 'node_ids', [])) for e in research_log.entries)

        content = self._llm(FIELD_NOTES_PROMPT.format(
            name        = self.name,
            mission     = self._mission(),
            findings    = findings,
            resolved    = resolved,
            new_questions = f"{new_qs} new nodes added to graph"
        ))

        entry = self._add_entry(
            ENTRY_FIELD_NOTES, content, cycle,
            tags=["research",
                  f"resolved:{sum(1 for e in research_log.entries if e.resolved in ['partial','strong'])}"]
        )
        print(f"\n── Notebook: field notes written ──")
        return content

    def write_evening_entry(self, consolidation_report, cycle: int) -> str:
        """Write an evening reflection after consolidation."""
        content = self._llm(EVENING_ENTRY_PROMPT.format(
            name         = self.name,
            mission      = self._mission(),
            merges       = consolidation_report.merges,
            syntheses    = consolidation_report.syntheses,
            abstractions = consolidation_report.abstractions,
            gaps         = consolidation_report.gaps,
            contradictions = self.brain.stats().get('contradictions', 0),
            brain_stats  = (f"{self.brain.stats()['nodes']} nodes, "
                            f"{self.brain.stats()['edges']} edges")
        ))

        entry = self._add_entry(
            ENTRY_EVENING, content, cycle,
            tags=["consolidation",
                  f"syntheses:{consolidation_report.syntheses}",
                  f"gaps:{consolidation_report.gaps}"]
        )
        print(f"\n── Notebook: evening entry written ──")
        return content

    def update_running_hypothesis(self, cycle: int) -> str:
        """
        Update the running best answer to the central question.
        Called after consolidation — when the graph is freshest.
        """
        if not self.observer:
            return ""

        # top mission advances
        progress_summary = "none yet"
        if self.observer:
            try:
                progress_summary = self.observer.get_mission_progress_summary()
            except Exception:
                progress_summary = "none yet"

        # top mission advances
        advances = sorted(
            self.observer.mission_advances,
            key=lambda a: a.strength, reverse=True
        )[:5]
        advances_text = "\n".join(
            f"- ({a.strength:.2f}) {a.explanation}"
            for a in advances
        ) or "none yet"

        # strongest structural/isomorphic insights from recent logs
        insights = []
        try:
            dream_files = [
                fname for fname in sorted(os.listdir("logs"), reverse=True)
                if fname.startswith("dream_cycle") and fname.endswith(".json")
            ]
            if not dream_files:
                print("Notebook: no dream_cycle*.json files found for insight synthesis.")
            for fname in dream_files[:10]:
                with open(f"logs/{fname}") as f:
                    d = json.load(f)
                for ins in d.get("insights", []):
                    if ins.get("depth") in ["structural", "isomorphism"]:
                        insights.append(
                            f"[{ins['depth']}] {ins['narration']}")
                if len(insights) >= 5:
                    break
        except Exception:
            pass
        insights_text = "\n".join(f"- {i}" for i in insights) or "none yet"

        # active hypotheses from graph - prioritized by mission relevance, importance, and recency
        mission_id = (self.brain.get_mission() or {}).get("id")
        all_hyps = list(self.brain.nodes_by_type(NodeType.HYPOTHESIS))
        def _hyp_score(item):
            nid, data = item
            is_linked = 1.0 if (mission_id and (
                self.brain.graph.has_edge(nid, mission_id) or
                self.brain.graph.has_edge(mission_id, nid)
            )) else 0.0
            imp = data.get('importance', 0.5)
            created = data.get('created_at', 0)
            return (is_linked, imp, created)

        all_hyps.sort(key=_hyp_score, reverse=True)
        hypotheses_text = "\n".join(
            f"- {data['statement']}"
            for _, data in all_hyps[:5]
        ) or "none yet"

        # active contradictions - sorted by recency across the entire graph
        contradictions = []
        contra_edges = [
            (u, v, data) for u, v, data in self.brain.graph.edges(data=True)
            if data.get('type') == EdgeType.CONTRADICTS.value
        ]
        contra_edges.sort(
            key=lambda e: (e[2].get('created_at', 0) or e[2].get('updated_at', 0) or e[2].get('weight', 0)),
            reverse=True
        )
        for u, v, data in contra_edges:
            nu = self.brain.get_node(u)
            nv = self.brain.get_node(v)
            if nu and nv:
                contradictions.append(
                    f"{nu['statement']} ↔ {nv['statement']}")
            if len(contradictions) >= 5:
                break
        contradictions_text = "\n".join(
            f"- {c}" for c in contradictions) or "none"

        self.running_hypothesis = self._llm(RUNNING_HYPOTHESIS_PROMPT.format(
            name           = self.name,
            mission        = self._mission(),
            progress_summary = progress_summary,
            advances       = advances_text,
            insights       = insights_text,
            hypotheses     = hypotheses_text,
            contradictions = contradictions_text
        ))

        self._add_entry(
            ENTRY_HYPOTHESIS, self.running_hypothesis, cycle,
            tags=["running_hypothesis", f"cycle:{cycle}"]
        )
        print(f"\n── Notebook: running hypothesis updated ──")
        return self.running_hypothesis

    def write_breakthrough(self, detail: str, cycle: int) -> str:
        """Write a breakthrough note — called when observer flags mission_advance."""
        content = self._llm(BREAKTHROUGH_PROMPT.format(
            name    = self.name,
            mission = self._mission(),
            detail  = detail
        ))
        entry = self._add_entry(
            ENTRY_BREAKTHROUGH, content, cycle,
            tags=["breakthrough"]
        )
        print(f"\n── Notebook: BREAKTHROUGH entry written ──")
        return content

    def write_synthesis_essay(self, cycle: int) -> dict:
        """
        Writing phase — the scientist writes a structured essay.

        Writing forces clarity. The LLM is asked to *write* about the research,
        and the act of writing produces side-effect insights that get returned
        for graph ingestion.

        Returns dict with 'essay', 'insights' (list of strings), 'questions' (list).
        """
        SYNTHESIS_ESSAY_PROMPT = """You are {name}, a scientist writing an incisive, rigorous research synthesis essay.

Central research question:
"{mission}"

Current working hypothesis:
{hypothesis}

Key ideas in knowledge graph:
{key_ideas}

Current working memory focus:
{working_memory}

Write a structured, rigorous 3-5 paragraph essay that:
1. Synthesizes the deepest mechanistic understanding of the central question achieved so far.
2. Evaluates the strongest theoretical/empirical evidence and critiques the weakest assumptions or model limitations.
3. Analyzes 1-2 non-obvious insights or counterintuitive trade-offs discovered across domains.
4. Identifies the pivotal next theoretical or empirical breakthrough required to resolve the central tension.

Then separately extract:
- NEW INSIGHTS: Non-trivial conceptual, physical, or mechanistic realizations that crystallized through this synthesis.
- NEW QUESTIONS: Specific, high-leverage research questions exposed by this analysis.

Respond with a JSON object:
{{
  "essay": "the full essay text",
  "insights": ["insight 1", "insight 2"],
  "questions": ["question 1", "question 2"]
}}"""

        # Build key ideas from graph
        key_nodes = sorted(
            self.brain.all_nodes(),
            key=lambda x: x[1].get('importance', 0.5),
            reverse=True
        )[:12]
        key_ideas = "\n".join(
            f"- [{d.get('node_type','concept')}] {d.get('statement','')}"
            for _, d in key_nodes
        )

        # Working memory
        wm = self.brain.get_working_memory()
        wm_text = "\n".join(
            f"- [{d.get('node_type','concept')}] {d.get('statement','')}"
            for _, d in wm
        ) if wm else "Nothing currently in focus."

        raw = self._llm(SYNTHESIS_ESSAY_PROMPT.format(
            name           = self.name,
            mission        = self._mission(),
            hypothesis     = self.running_hypothesis or "None formulated yet.",
            key_ideas      = key_ideas,
            working_memory = wm_text
        ), temperature=0.5)

        # Try to parse as JSON
        from llm_utils import require_json
        result = require_json(raw, default=None)

        if result and isinstance(result, dict):
            essay    = result.get("essay", raw)
            insights = result.get("insights", [])
            questions = result.get("questions", [])
        else:
            essay = raw
            insights = []
            questions = []

        # Store the essay
        self._add_entry(
            ENTRY_SYNTHESIS, essay, cycle,
            tags=["synthesis_essay", f"cycle:{cycle}"]
        )

        print(f"\n── Notebook: synthesis essay written "
              f"({len(insights)} insights, {len(questions)} questions) ──")

        return {
            "essay":     essay,
            "insights":  insights if isinstance(insights, list) else [],
            "questions": questions if isinstance(questions, list) else []
        }

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_entries_by_type(self, entry_type: str) -> list:
        return [e for e in self.entries if e.entry_type == entry_type]

    def get_recent_entries(self, n: int = 10) -> list:
        return sorted(self.entries, key=lambda e: e.timestamp, reverse=True)[:n]

    def get_all_for_display(self) -> list:
        """Returns entries formatted for GUI display, newest first."""
        result = []
        for e in sorted(self.entries,
                        key=lambda x: x.timestamp, reverse=True):
            result.append({
                "type":      e.entry_type,
                "content":   e.content,
                "cycle":     e.cycle,
                "timestamp": e.timestamp,
                "tags":      e.tags
            })
        return result

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.path)
                    if os.path.dirname(self.path) else ".",
                    exist_ok=True)
        data = {
            "entries":            [e.to_dict() for e in self.entries],
            "running_hypothesis": self.running_hypothesis,
            "scientist_name":     self.name
        }
        atomic_write_json(self.path, data)

    def _load(self):
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
            self.entries = [
                NotebookEntry(**e) for e in data.get('entries', [])
            ]
            self.running_hypothesis = data.get('running_hypothesis', '')
            self.name = data.get('scientist_name', self.name)
            print(f"Notebook loaded — {len(self.entries)} entries")
        except FileNotFoundError:
            print("Notebook: starting fresh")

    def save(self):
        self._save()
        print(f"Notebook saved — {len(self.entries)} entries")
