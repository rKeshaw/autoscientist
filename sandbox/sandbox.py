import json
import time
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field
from graph.brain import Brain, Node, Edge, EdgeType, EdgeSource, NodeType, NodeStatus
from persistence import atomic_write_json
from llm_utils import llm_call, require_json

# ── Config ────────────────────────────────────────────────────────────────────

SANDBOX_TIMEOUT   = 30
MAX_OUTPUT_CHARS  = 2000
SANDBOX_LOG_PATH  = "logs/sandbox_log.json"

# ── Prompts ───────────────────────────────────────────────────────────────────

TESTABILITY_PROMPT = """
You are evaluating whether a scientific hypothesis can be tested computationally.

Hypothesis: {hypothesis}

A hypothesis is computationally testable if it:
- Makes a quantitative or directional prediction that can be modeled numerically
- Proposes a relationship between variables that can be simulated via differential equations, discrete dynamical systems, stochastic processes, or network models
- Suggests a formal mathematical, information-theoretic, or logical structure that can be verified
- Can be tested through comparative simulation (e.g. comparing a baseline system against the hypothesized mechanism)

A hypothesis is NOT computationally testable if it:
- Is purely definitional or semantic without any functional consequence
- Relies entirely on unspecifiable metaphysical concepts that cannot be mapped to variables
- Strictly requires physical wet-lab assay data that cannot be approximated or simulated numerically

Respond with a JSON object:
{{
  "testable": true or false,
  "reason": "1-2 sentences explaining why or why not",
  "approach": "if testable: describe the concrete simulation or calculation to test it"
}}

Respond ONLY with JSON.
"""

CODE_GENERATION_PROMPT = """
You are an expert computational scientist. Write clean, robust, standalone Python code to computationally test this hypothesis:

Hypothesis: {hypothesis}
Testing approach: {approach}
Central research question context: {mission}

Scientific & Formalism Requirements (No Hardcoding — Select the Natural Formalism):
- Adapt the mathematical and computational formalism to the natural scientific domain of the hypothesis:
  * Dynamical systems & non-equilibrium physics: Coupled ODEs/SDEs, Langevin dynamics, Master equations, Fokker-Planck equations, Lyapunov exponents, phase portraits, bifurcation analysis.
  * Information theory & discrete computation / complexity: Channel capacity, Shannon entropy, mutual information, transfer entropy, error-correcting codes, Boolean circuits, algorithmic complexity simulations.
  * Statistical mechanics & thermodynamics: Monte Carlo sampling (Metropolis-Hastings), Ising/Potts models, percolation thresholds, partition functions, Crooks/Jarzynski fluctuation theorem dissipation tracking.
  * Quantum systems & molecular physics: Hamiltonian matrices, unitary evolution, tunneling transmission coefficients, density matrix dynamics, Arrhenius barrier hopping vs tunneling.
  * Molecular, systems & synthetic biology: Gillespie stochastic simulation algorithm (SSA), chemical reaction networks (CRNs), gene regulatory networks, Michaelis-Menten/Hill kinetics, sequence alignment distances.
  * Neuroscience & neuromorphic architectures: Leaky Integrate-and-Fire (LIF) networks, Spike-Timing-Dependent Plasticity (STDP), branching ratios (criticality), avalanche size/duration power-law distributions.
- Implement a clear comparison: Simulate a CONTROL/BASELINE condition and the HYPOTHESIZED condition, or perform a parameter sweep across regimes to demonstrate divergence or scaling.
- Available Libraries: You have full access to standard and domain-specific Python libraries: numpy, scipy, matplotlib, networkx, sympy, pandas, scikit-learn, torch, etc. If a domain-specific package is required, import it directly (the sandbox installs missing packages on the fly).
- Numerical & Array Robustness:
  * Prevent array broadcasting mismatches (e.g. ensure 1D vs 2D arrays match when performing operations; use .ravel() or .squeeze() where appropriate).
  * Ensure valid function signatures: in solve_ivp(fun, t_span, y0, args=tuple), fun(t, y, *args) must return a 1D sequence or array of scalars with exactly the same length as y0 (parameters in args must be scalars or scalar functions of t, not arrays causing inhomogeneous shapes).
  * Prevent numerical instability (check for division by zero, bound exponential arguments, maintain non-negative probabilities).
  * For bipartite coding / Tanner graphs, represent edges using parity-check matrices H (where H[j, i] != 0 indicates an edge between check j and variable i) or lists of neighboring indices (e.g. var_neighbors[j]), avoiding naive integer containment checks like 'if i in j'.
  * For Gillespie stochastic simulation, implement standard exact SSA: calculate propensity vector a, total hazard a0 = sum(a), draw tau = np.random.exponential(1.0 / a0), select reaction with probability a / a0, and update discrete molecular copy numbers.
- Measurement & Output:
  * Compute quantitative metrics (e.g. dissipation rate, error rate, convergence time, fidelity, mutual information, Lyapunov exponent, scaling exponent).
  * Print clear, interpretable quantitative outputs comparing baseline vs hypothesized conditions across stages/regimes.
  * If plotting, save the figure as 'sandbox_output.png' in the current working directory.
  * Include a final print statement summarizing whether the quantitative metrics support, contradict, or leave inconclusive the hypothesis.

Write ONLY the executable Python code. No preamble, no markdown fences, no explanatory chat.
"""

CODE_CORRECTION_PROMPT = """
You are an expert computational scientist. The simulation code you previously generated encountered an execution error or produced invalid output.
Diagnose the failure and write the corrected, robust Python code.

Hypothesis: {hypothesis}
Testing approach: {approach}
Central research question: {mission}

PREVIOUS CODE THAT FAILED:
```python
{previous_code}
```

EXECUTION STDOUT:
{stdout}

EXECUTION ERROR / TRACEBACK (STDERR):
{stderr}

DIAGNOSTIC CRITIQUE:
{diagnostic}

Correction Guidelines:
1. Root Cause Analysis: Inspect the exact line in PREVIOUS CODE that raised the exception in STDERR.
2. Syntax & Indentation: Fix any unindented function bodies, unexpected indents, or malformed strings.
3. Array Shapes & Broadcasting: Resolve NumPy dimension mismatches (e.g., adding (N,) array to (N, 1) array; flatten or reshape appropriately).
4. API Signatures & Types: Ensure arguments passed to scipy/numpy functions match expected signatures (e.g., args must be tuples of scalars; in solve_ivp, the derivative function must return a 1D sequence of scalars matching y0; if using neural models, simulate directly via vector LIF/scipy rather than third-party wrappers that may conflict with NumPy 2.x).
5. Mathematical Stability: Add safeguards against division by zero, NaN propagation, or singular matrices.
6. Execution Completeness: Ensure the script runs standalone, prints clear comparative quantitative metrics, and saves 'sandbox_output.png' if generating a plot.

Write ONLY the complete, corrected executable Python code. No preamble, no markdown fences, no explanatory chat.
"""

RESULT_INTERPRETATION_PROMPT = """
You are a rigorous scientist analyzing computational simulation results to assess a hypothesis.

Hypothesis tested: {hypothesis}
Central research question: {mission}
Code executed:
{code}

Execution Output:
{output}

Execution Errors / Warnings:
{errors}

Evaluation Guidelines:
- "error": The simulation failed to execute or crashed due to syntax/runtime exceptions.
- "supports": The simulation executed successfully, and the quantitative metrics show a statistically or systematically significant effect aligning with the hypothesis compared to baseline.
- "contradicts": The simulation executed successfully, but the quantitative metrics show the opposite effect or fail to demonstrate the predicted divergence where it should have appeared.
- "inconclusive": The simulation ran, but the parameter regime, model fidelity, or metric variance was insufficient to definitively accept or refute the claim.

Confidence Rubric:
- 0.1-0.3: Low confidence — toy model with high parameter sensitivity or unclear signal.
- 0.4-0.6: Moderate confidence — sound dynamical model capturing key variables, with clear trend under the tested parameter regime.
- 0.7-0.85: High confidence — robust comparative simulation showing definitive separation between baseline and hypothesized mechanisms.
- 0.9-1.0: Definitive — rigorous mathematical proof, exhaustive parameter sweep, or exact analytical convergence.

Evaluate the results with scientific objectivity, avoiding reflexive skepticism or artificial uncertainty.

Respond with a JSON object:
{{
  "verdict": one of ["supports", "contradicts", "inconclusive", "error"],
  "confidence": a float 0.0 to 1.0 (use rubric above),
  "interpretation": "2-3 sentences analyzing the quantitative findings and whether the mechanism held",
  "implications": "1-2 sentences on what this result means for the central research mission"
}}

Respond ONLY with JSON.
"""

# ── Sandbox result ────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    hypothesis_node_id: str
    hypothesis:         str
    approach:           str
    code:               str
    stdout:             str
    stderr:             str
    verdict:            str
    confidence:         float
    interpretation:     str
    implications:       str
    plot_path:          str   = ""
    timestamp:          float = field(default_factory=time.time)
    duration_seconds:   float = 0.0

    def to_dict(self):
        return self.__dict__

# ── Sandbox ───────────────────────────────────────────────────────────────────

class Sandbox:
    def __init__(self, brain: Brain, observer=None, embedding_index=None,
                 log_path: str = SANDBOX_LOG_PATH):
        self.brain    = brain
        self.observer = observer
        self.embedding_index = embedding_index
        self.log_path = log_path
        self.results: list[SandboxResult] = []
        self._load()

    def _llm(self, prompt: str, temperature: float = 0.5) -> str:
        return llm_call(prompt, temperature=temperature, role="code")

    def _mission(self) -> str:
        m = self.brain.get_mission()
        return m['question'] if m else "No central question set."

    # ── Testability check ─────────────────────────────────────────────────────

    def is_testable(self, hypothesis: str) -> tuple:
        raw = self._llm(TESTABILITY_PROMPT.format(hypothesis=hypothesis), temperature=0.2)
        try:
            result = json.loads(raw)
            return (
                result.get('testable', False),
                result.get('reason', ''),
                result.get('approach', '')
            )
        except (json.JSONDecodeError, ValueError):
            return False, "Parse error", ""

    # ── Code execution ────────────────────────────────────────────────────────

    def _run_code(self, code: str, allow_install: bool = True) -> tuple:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT,
                cwd=os.getcwd()
            )
            stdout = proc.stdout[:MAX_OUTPUT_CHARS]
            stderr = proc.stderr[:MAX_OUTPUT_CHARS]
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = f"Timeout: execution exceeded {SANDBOX_TIMEOUT}s"
        except Exception as e:
            stdout = ""
            stderr = str(e)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # On-the-fly installation of missing domain-specific libraries
        if allow_install and stderr and ("ModuleNotFoundError: No module named" in stderr or "ImportError: No module named" in stderr):
            import re
            match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
            if match:
                pkg = match.group(1).split('.')[0]
                if pkg and pkg.isidentifier() and pkg not in sys.builtin_module_names:
                    print(f"   [Sandbox Library Manager] On-the-fly installing missing package: {pkg}...")
                    try:
                        pip_res = subprocess.run(
                            [sys.executable, "-m", "pip", "install", pkg],
                            capture_output=True, text=True, timeout=60
                        )
                        if pip_res.returncode == 0:
                            print(f"   [Sandbox Library Manager] Installed {pkg} successfully. Re-executing...")
                            return self._run_code(code, allow_install=False)
                        else:
                            print(f"   [Sandbox Library Manager] Failed to install {pkg}: {pip_res.stderr[:80]}")
                    except Exception as ex:
                        print(f"   [Sandbox Library Manager] Error during on-the-fly install: {ex}")

        return stdout, stderr, time.time() - start

    # ── Full test pipeline ────────────────────────────────────────────────────

    def test_hypothesis(self, hypothesis: str,
                        node_id: str = "") -> SandboxResult:
        print(f"\n── Sandbox: testing hypothesis ──")
        print(f"   {hypothesis}")

        # step 1: testability
        testable, reason, approach = self.is_testable(hypothesis)
        if not testable:
            print(f"   Not computationally testable: {reason}")
            result = SandboxResult(
                hypothesis_node_id = node_id,
                hypothesis         = hypothesis,
                approach           = "not testable",
                code               = "",
                stdout             = "",
                stderr             = reason,
                verdict            = "inconclusive",
                confidence         = 0.0,
                interpretation     = f"Not computationally testable: {reason}",
                implications       = "Further conceptual or empirical work needed."
            )
            self.results.append(result)
            self._save()
            return result

        print(f"   Approach: {approach}")

        # step 2: generate code and run (up to 3 tries on error, with coder LLM in the loop)
        max_tries = 3
        failures = 0
        last_code = ""
        last_stdout = ""
        last_stderr = ""
        last_interp = ""

        for attempt in range(max_tries):
            if attempt == 0:
                prompt = CODE_GENERATION_PROMPT.format(
                    hypothesis = hypothesis,
                    approach   = approach,
                    mission    = self._mission()
                )
            else:
                print(f"   [Coder LLM In The Loop] Passing failure context, code, and traceback to LLM for correction...")
                prompt = CODE_CORRECTION_PROMPT.format(
                    hypothesis    = hypothesis,
                    approach      = approach,
                    mission       = self._mission(),
                    previous_code = last_code,
                    stdout        = last_stdout or "no stdout produced",
                    stderr        = last_stderr or "none",
                    diagnostic    = last_interp or last_stderr
                )

            code = self._llm(prompt, temperature=0.2 + (0.1 * attempt))

            # strip markdown fences and extract code block cleanly
            if '```python' in code:
                code = code.split('```python', 1)[1].split('```', 1)[0]
            elif '```' in code:
                code = code.split('```', 1)[1].split('```', 1)[0]
            import textwrap
            code = textwrap.dedent(code).strip()

            print(f"   Running code attempt {attempt+1}/{max_tries} ({len(code)} chars)...")

            # step 3: run
            stdout, stderr, duration = self._run_code(code)
            print(f"   Completed in {duration:.1f}s")
            if stdout:
                print(f"   Output: {stdout[:200]}...")
            if stderr:
                print(f"   Errors: {stderr[:200]}...")

            # step 4: interpret
            raw = self._llm(RESULT_INTERPRETATION_PROMPT.format(
                hypothesis = hypothesis,
                mission    = self._mission(),
                code       = code,
                output     = stdout or "no output",
                errors     = stderr or "none"
            ), temperature=0.2)
            try:
                interp = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                interp = {
                    "verdict":        "error" if stderr else "inconclusive",
                    "confidence":     0.3,
                    "interpretation": raw,
                    "implications":   ""
                }

            verdict        = interp.get('verdict', 'inconclusive')
            confidence     = interp.get('confidence', 0.3)
            interpretation = interp.get('interpretation', '')
            implications   = interp.get('implications', '')

            # If no execution error, we are done testing
            if verdict != "error" and not stderr:
                break
                
            failures += 1
            last_code   = code
            last_stdout = stdout
            last_stderr = stderr
            last_interp = interpretation
            print(f"   Attempt {attempt+1} failed. Re-prompting coder LLM with error feedback...")

        if failures >= max_tries:
            print(f"   ✓ Sandbox failed after {max_tries} attempts. Increasing frustration.")
            self.brain.increase_frustration(0.3)

        print(f"   Verdict: {verdict} (confidence={confidence:.2f})")
        print(f"   {interpretation}")

        # check for plot
        plot_path = ""
        if os.path.exists("sandbox_output.png"):
            import shutil
            dest = f"logs/sandbox_{int(time.time())}.png"
            shutil.move("sandbox_output.png", dest)
            plot_path = dest
            print(f"   Plot saved: {dest}")

        result = SandboxResult(
            hypothesis_node_id = node_id,
            hypothesis         = hypothesis,
            approach           = approach,
            code               = code,
            stdout             = stdout,
            stderr             = stderr,
            verdict            = verdict,
            confidence         = confidence,
            interpretation     = interpretation,
            implications       = implications,
            plot_path          = plot_path,
            duration_seconds   = duration
        )
        self.results.append(result)
        self._integrate_result(result, node_id)
        self._save()
        return result

    # ── Graph integration ─────────────────────────────────────────────────────

    def _integrate_result(self, result: SandboxResult,
                          hypothesis_node_id: str):
        statement = (
            f"Computational test of: {result.hypothesis}. "
            f"Verdict: {result.verdict} (confidence={result.confidence:.2f}). "
            f"{result.interpretation}"
        )

        node = Node(
            statement        = statement,
            node_type        = NodeType.EMPIRICAL,
            cluster          = "empirical",
            status           = (NodeStatus.SETTLED
                                if result.confidence > 0.7
                                else NodeStatus.UNCERTAIN),
            importance       = result.confidence,
            empirical_result = result.interpretation,
            empirical_code   = result.code
        )
        nid = self.brain.add_node(node)

        # link to hypothesis node
        if hypothesis_node_id and self.brain.get_node(hypothesis_node_id):
            edge = Edge(
                type         = EdgeType.EMPIRICALLY_TESTED,
                narration    = (f"Computational test: {result.verdict} "
                                f"(confidence={result.confidence:.2f}). "
                                f"{result.implications}"),
                weight       = result.confidence,
                confidence   = result.confidence,
                source       = EdgeSource.SANDBOX,
                decay_exempt = result.confidence > 0.7
            )
            self.brain.add_edge(hypothesis_node_id, nid, edge)

            # add contradicts edge if result contradicts
            if result.verdict == "contradicts" and result.confidence > 0.6:
                contra = Edge(
                    type         = EdgeType.CONTRADICTS,
                    narration    = (f"Computational test contradicts: "
                                    f"{result.interpretation}"),
                    weight       = result.confidence,
                    confidence   = result.confidence,
                    source       = EdgeSource.SANDBOX,
                    decay_exempt = True
                )
                self.brain.add_edge(nid, hypothesis_node_id, contra)

        # only link to mission if test actually ran and is meaningful
        if result.verdict not in ("error",) and result.confidence > 0.6:
            self.brain.link_to_mission(
                nid,
                f"Empirical result: {result.implications}",
                strength=result.confidence * 0.7
            )

        # notify observer only on genuine supporting results
        if (self.observer and
                result.verdict == "supports" and
                result.confidence > 0.65):
            self.observer.record_mission_advance(
                nid,
                f"Computational test supports: {result.implications}",
                result.confidence * 0.8
            )

        print(f"   EMPIRICAL node created: {nid[:8]}")
        return nid

    # ── Scan and test ─────────────────────────────────────────────────────────

    def scan_and_test(self, max_tests: int = 3) -> list:
        print(f"\n── Sandbox scan: looking for testable hypotheses ──")

        mission_id = (self.brain.get_mission() or {}).get("id")
        tested_ids = set(r.hypothesis_node_id for r in self.results if getattr(r, 'hypothesis_node_id', None))

        candidates = []
        for nid, data in self.brain.nodes_by_type(NodeType.HYPOTHESIS):
            # skip already tested in session history
            if nid in tested_ids:
                continue

            # skip already tested in graph edges (out-edges or in-edges)
            already_tested = any(
                edata.get('type') == EdgeType.EMPIRICALLY_TESTED.value
                for _, _, edata in self.brain.graph.out_edges(nid, data=True)
            ) or any(
                edata.get('type') == EdgeType.EMPIRICALLY_TESTED.value
                for _, _, edata in self.brain.graph.in_edges(nid, data=True)
            )
            if already_tested:
                continue

            # Prioritize mission-connected and recently created hypotheses
            score = data.get('importance', 0.5)
            is_mission = 1.0 if (mission_id and (
                self.brain.graph.has_edge(nid, mission_id) or
                self.brain.graph.has_edge(mission_id, nid)
            )) else 0.0
            created = data.get('created_at', 0)
            score += (is_mission * 0.5) + ((created / 1e10) if created else 0.0)
            candidates.append((nid, data, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        results = []

        for nid, data, score in candidates[:max_tests]:
            stmt = data.get('statement', '')
            if not stmt:
                continue
            result = self.test_hypothesis(stmt, node_id=nid)
            results.append(result)
            time.sleep(1)

        print(f"\n── Sandbox complete: {len(results)} tests run ──")
        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        data = {"results": [r.to_dict() for r in self.results]}
        atomic_write_json(self.log_path, data)

    def _load(self):
        try:
            with open(self.log_path, 'r') as f:
                data = json.load(f)
            self.results = [
                SandboxResult(**r) for r in data.get('results', [])
            ]
            print(f"Sandbox loaded — {len(self.results)} prior results")
        except FileNotFoundError:
            print("Sandbox: starting fresh")
