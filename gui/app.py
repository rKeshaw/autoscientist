import json
import os
import sys
import time
import queue
import threading
from flask import Flask, jsonify, request, render_template, Response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from graph.brain import Brain, NodeStatus, NodeType, EdgeType
from observer.observer import Observer
from ingestion.ingestor import Ingestor, EdgeSource
from dreamer.dreamer import Dreamer, DreamMode
from consolidator.consolidator import Consolidator
from researcher.researcher import Researcher
from notebook.notebook import Notebook
from sandbox.sandbox import Sandbox

# ── Config ────────────────────────────────────────────────────────────────────

BRAIN_PATH    = "data/brain.json"
OBSERVER_PATH = "data/observer.json"

app   = Flask(__name__, template_folder='templates')
state = {
    "phase":   "idle",
    "message": "Standing by.",
    "cycle":   0,
    "thread":  None,
    "active_node_id": None,
}

# ── Event stream ──────────────────────────────────────────────────────────────

_event_queue = queue.Queue(maxsize=300)

def emit(event_type: str, data: dict):
    try:
        _event_queue.put_nowait({
            "type":      event_type,
            "data":      data,
            "timestamp": time.time()
        })
    except queue.Full:
        pass

@app.route("/api/stream")
def api_stream():
    def generate():
        while True:
            try:
                event = _event_queue.get(timeout=25)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type':'heartbeat','data':{}})}\n\n"
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )

# ── Shared state ──────────────────────────────────────────────────────────────

brain    = Brain()
observer = Observer(brain)

try:
    brain.load(BRAIN_PATH)
except FileNotFoundError:
    pass
try:
    observer.load(OBSERVER_PATH)
except Exception:
    pass

def _stream_cb(event_type: str, data: dict):
    """Callback passed to components so they can emit events."""
    if event_type == "dream_step":
        state["active_node_id"] = data.get("to_id")
    emit(event_type, data)

ingestor     = Ingestor(brain, research_agenda=observer)
dreamer      = Dreamer(brain, research_agenda=observer)
consolidator = Consolidator(brain, observer=observer)
researcher   = Researcher(brain, observer=observer, depth="standard")
notebook     = Notebook(brain, observer=observer)
sandbox      = Sandbox(brain, observer=observer)

def save_state():
    brain.save(BRAIN_PATH)
    observer.save(OBSERVER_PATH)

# ── API: Status ───────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({
        "phase":          state["phase"],
        "message":        state["message"],
        "cycle":          state["cycle"],
        "brain":          brain.stats(),
        "agenda":         len(observer.agenda),
        "resolved":       sum(1 for i in observer.agenda if i.resolved),
        "emergences":     len(observer.emergence_feed),
        "mission_advances": len(observer.mission_advances),
        "active_node_id": state["active_node_id"],
    })

# ── API: Graph ────────────────────────────────────────────────────────────────

@app.route("/api/graph")
def api_graph():
    nodes, edges = [], []
    for nid, data in brain.all_nodes():
        nodes.append({
            "id":         nid,
            "label":      data.get("statement", "")[:80],
            "statement":  data.get("statement", ""),
            "cluster":    data.get("cluster", "unclustered"),
            "node_type":  data.get("node_type", "concept"),
            "status":     data.get("status", "uncertain"),
            "importance": data.get("importance", 0.5),
            "incubation": data.get("incubation_age", 0),
        })
    for u, v, data in brain.graph.edges(data=True):
        edges.append({
            "source":    u,
            "target":    v,
            "type":      data.get("type", "associated"),
            "weight":    data.get("weight", 0.5),
            "narration": data.get("narration", ""),
            "analogy_depth": data.get("analogy_depth", ""),
        })
    return jsonify({"nodes": nodes, "edges": edges})

@app.route("/api/node/<node_id>")
def api_node(node_id):
    node = brain.get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    edges_out = []
    for neighbor in brain.neighbors(node_id):
        e  = brain.get_edge(node_id, neighbor)
        nb = brain.get_node(neighbor)
        if e and nb:
            edges_out.append({
                "target_id":    neighbor,
                "target_label": nb.get("statement", "")[:60],
                "type":         e.get("type", ""),
                "weight":       e.get("weight", 0),
                "narration":    e.get("narration", ""),
                "analogy_depth": e.get("analogy_depth", ""),
            })
    return jsonify({**node, "edges_out": edges_out})

# ── API: Agenda ───────────────────────────────────────────────────────────────

@app.route("/api/agenda")
def api_agenda():
    items = []
    for item in sorted(observer.agenda, key=lambda i: i.priority, reverse=True):
        items.append({
            "text":           item.text,
            "type":           item.item_type,
            "priority":       round(item.priority, 2),
            "incubation_age": item.incubation_age,
            "resolved":       item.resolved,
            "grade":          item.resolution_grade,
            "count":          item.count,
        })
    return jsonify(items)

# ── API: Emergences ───────────────────────────────────────────────────────────

@app.route("/api/emergences")
def api_emergences():
    signals = []
    for e in reversed(observer.emergence_feed[-60:]):
        signals.append({
            "signal":    e.signal,
            "type":      e.type,
            "cycle":     e.cycle,
            "timestamp": e.timestamp,
        })
    return jsonify(signals)

# ── API: Mission ──────────────────────────────────────────────────────────────

@app.route("/api/mission")
def api_mission():
    mission = brain.get_mission()
    if not mission:
        return jsonify({"set": False})
    advances = sorted(
        observer.mission_advances,
        key=lambda a: a.strength, reverse=True
    )[:8]
    return jsonify({
        "set":      True,
        "question": mission['question'],
        "context":  mission.get('context', ''),
        "id":       mission['id'],
        "advances": [
            {"explanation": a.explanation,
             "strength": a.strength,
             "cycle": a.cycle}
            for a in advances
        ]
    })

@app.route("/api/mission", methods=["POST"])
def api_set_mission():
    data    = request.json
    question = data.get("question", "").strip()
    context  = data.get("context", "").strip()
    if not question:
        return jsonify({"error": "no question"}), 400
    brain.set_mission(question, context)
    save_state()
    emit("mission_set", {"question": question})
    return jsonify({"status": "set", "question": question})

# ── API: Notebook ─────────────────────────────────────────────────────────────

@app.route("/api/notebook")
def api_notebook():
    return jsonify({
        "entries":            notebook.get_all_for_display(),
        "running_hypothesis": notebook.running_hypothesis
    })

@app.route("/api/notebook/hypothesis")
def api_notebook_hypothesis():
    return jsonify({"hypothesis": notebook.running_hypothesis})

# ── API: Sandbox ──────────────────────────────────────────────────────────────

@app.route("/api/sandbox")
def api_sandbox():
    results = []
    for r in reversed(sandbox.results[-20:]):
        results.append({
            "hypothesis":      r.hypothesis[:100],
            "verdict":         r.verdict,
            "confidence":      r.confidence,
            "interpretation":  r.interpretation,
            "implications":    r.implications,
            "plot_path":       r.plot_path,
            "timestamp":       r.timestamp,
            "approach":        r.approach,
        })
    return jsonify(results)

@app.route("/api/sandbox/run", methods=["POST"])
def api_sandbox_run():
    if state["phase"] != "idle":
        return jsonify({"error": "already running"}), 409
    data = request.json
    hypothesis = data.get("hypothesis", "").strip()
    if not hypothesis:
        return jsonify({"error": "no hypothesis"}), 400

    def run():
        state["phase"]   = "sandbox"
        state["message"] = "Running computation..."
        emit("phase_change", {"phase": "sandbox",
                               "message": "Sandbox computing..."})
        try:
            result = sandbox.test_hypothesis(hypothesis)
            emit("sandbox_result", {
                "verdict":        result.verdict,
                "confidence":     result.confidence,
                "interpretation": result.interpretation,
                "hypothesis":     result.hypothesis[:80],
            })
        except Exception as e:
            state["message"] = f"Sandbox error: {e}"
        finally:
            save_state()
            state["phase"]   = "idle"
            state["message"] = "Computation complete."
            emit("phase_change", {"phase": "idle", "message": "Computation complete."})

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})

# ── API: Dream log ────────────────────────────────────────────────────────────

@app.route("/api/dreamlog")
def api_dreamlog():
    logs = []
    if os.path.exists("logs"):
        for fname in sorted(os.listdir("logs"), reverse=True):
            if fname.startswith("dream_") and fname.endswith(".json"):
                try:
                    with open(os.path.join("logs", fname)) as f:
                        data = json.load(f)
                    logs.append({
                        "file":     fname,
                        "mode":     data.get("mode", ""),
                        "summary":  data.get("summary", ""),
                        "questions": len(data.get("questions", [])),
                        "insights":  len(data.get("insights", [])),
                        "answers":   len(data.get("answers", [])),
                        "steps":     len(data.get("steps", [])),
                        "mission_advances": len(data.get("mission_advances", [])),
                        "started":   data.get("started_at", 0),
                    })
                except Exception:
                    continue
    return jsonify(logs[:20])

@app.route("/api/dreamlog/<filename>")
def api_dreamlog_detail(filename):
    path = os.path.join("logs", filename)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))

# ── API: Controls ─────────────────────────────────────────────────────────────

@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "no text"}), 400

    def run():
        state["phase"]   = "ingesting"
        state["message"] = "Ingesting..."
        emit("phase_change", {"phase": "ingesting", "message": "Ingesting new input..."})
        ingestor.ingest(text, source=EdgeSource.CONVERSATION)
        save_state()
        state["phase"]   = "idle"
        state["message"] = "Ingestion complete."
        emit("phase_change", {"phase": "idle", "message": "Ingestion complete."})

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/seed", methods=["POST"])
def api_seed():
    data    = request.json
    concept = data.get("concept", "").strip()
    if not concept:
        return jsonify({"error": "no concept"}), 400
    item          = observer.add_to_agenda(concept, cycle=observer.cycle_count)
    item.priority = 0.95
    save_state()
    emit("seed_planted", {"concept": concept})
    return jsonify({"status": "seeded"})

@app.route("/api/run/<phase>", methods=["POST"])
def api_run_phase(phase):
    if state["phase"] != "idle":
        return jsonify({"error": "already running"}), 409
    phases = {
        "dream":         _run_dream,
        "research":      _run_research,
        "consolidation": _run_consolidation,
        "sandbox":       _run_sandbox,
    }
    if phase not in phases:
        return jsonify({"error": "unknown phase"}), 400
    threading.Thread(target=phases[phase], daemon=True).start()
    return jsonify({"status": "started", "phase": phase})

@app.route("/api/scientificness", methods=["POST"])
def api_scientificness():
    val = float(request.json.get("value", 0.7))
    brain.scientificness = max(0.0, min(1.0, val))
    save_state()
    return jsonify({"scientificness": brain.scientificness})

# ── Phase runners ─────────────────────────────────────────────────────────────

def _run_dream():
    state["cycle"]  += 1
    state["phase"]   = "dreaming"
    state["message"] = "NREM consolidation..."
    emit("phase_change", {"phase": "dreaming", "message": "Night cycle beginning..."})
    try:
        log1 = dreamer.dream(
            mode=DreamMode.WANDERING, steps=15, temperature=0.7,
            run_nrem=True,
            log_path=f"logs/dream_cycle{state['cycle']}_wandering.json"
        )
        # emit dream steps to stream
        for step in log1.steps:
            emit("dream_step", {
                "step":          step.step,
                "from_id":       step.from_id,
                "to_id":         step.to_id,
                "edge_type":     step.edge_type,
                "narration":     step.narration,
                "is_insight":    step.is_insight,
                "insight_depth": getattr(step, 'insight_depth', ''),
                "mission_advance": getattr(step, 'mission_advance', False),
                "question":      step.question,
            })
        observer.observe(log1)
        notebook.write_morning_entry(log1, state['cycle'])

        emit("phase_change", {"phase": "dreaming", "message": "Pressure dream..."})
        log2 = dreamer.dream(
            mode=DreamMode.PRESSURE, steps=8, temperature=0.6,
            run_nrem=False,
            log_path=f"logs/dream_cycle{state['cycle']}_pressure.json"
        )
        for step in log2.steps:
            emit("dream_step", {
                "step":      step.step,
                "from_id":   step.from_id,
                "to_id":     step.to_id,
                "edge_type": step.edge_type,
                "narration": step.narration,
                "is_insight": step.is_insight,
                "mission_advance": getattr(step, 'mission_advance', False),
            })
        observer.observe(log2)
        state["message"] = f"Dream cycle {state['cycle']} complete."
        emit("dream_complete", {
            "cycle":    state['cycle'],
            "insights": len(log1.insights) + len(log2.insights),
            "advances": len(log1.mission_advances) + len(log2.mission_advances),
            "summary":  log2.summary[:300],
        })
    except Exception as e:
        state["message"] = f"Dream error: {e}"
        emit("error", {"message": str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change", {"phase": "idle", "message": state["message"]})

def _run_research():
    state["phase"]   = "researching"
    state["message"] = "Research day..."
    emit("phase_change", {"phase": "researching", "message": "Research day beginning..."})
    try:
        log = researcher.research_day(
            max_questions=5,
            log_path=f"logs/research_cycle{state['cycle']}.json"
        )
        notebook.write_field_notes(log, state['cycle'])
        resolved = sum(1 for e in log.entries
                       if e.resolved in ['partial', 'strong'])
        state["message"] = f"Research complete. {resolved} questions advanced."
        emit("research_complete", {
            "questions_researched": len(log.entries),
            "resolved": resolved,
        })
    except Exception as e:
        state["message"] = f"Research error: {e}"
        emit("error", {"message": str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change", {"phase": "idle", "message": state["message"]})

def _run_consolidation():
    state["phase"]   = "consolidating"
    state["message"] = "Evening consolidation..."
    emit("phase_change", {"phase": "consolidating",
                           "message": "Evening consolidation..."})
    try:
        report = consolidator.consolidate(
            save_path=f"logs/consolidation_cycle{state['cycle']}.json")
        notebook.write_evening_entry(report, state['cycle'])
        notebook.update_running_hypothesis(state['cycle'])
        state["message"] = "Consolidation complete."
        emit("consolidation_complete", {
            "merges":        report.merges,
            "syntheses":     report.syntheses,
            "abstractions":  report.abstractions,
            "gaps":          report.gaps,
            "summary":       report.summary[:300],
        })
    except Exception as e:
        state["message"] = f"Consolidation error: {e}"
        emit("error", {"message": str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change", {"phase": "idle", "message": state["message"]})

def _run_sandbox():
    state["phase"]   = "sandbox"
    state["message"] = "Running computations..."
    emit("phase_change", {"phase": "sandbox", "message": "Running computations..."})
    try:
        results = sandbox.scan_and_test(max_tests=3)
        for r in results:
            emit("sandbox_result", {
                "verdict":       r.verdict,
                "confidence":    r.confidence,
                "interpretation": r.interpretation,
                "hypothesis":    r.hypothesis[:80],
                "plot_path":     r.plot_path,
            })
        state["message"] = f"Sandbox: {len(results)} tests complete."
    except Exception as e:
        state["message"] = f"Sandbox error: {e}"
        emit("error", {"message": str(e)})
    finally:
        save_state()
        state["phase"] = "idle"
        emit("phase_change", {"phase": "idle", "message": state["message"]})

# ── Serve frontend ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    print(f"DREAMER GUI starting on http://0.0.0.0:5000")
    print(f"Brain: {brain.stats()}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)