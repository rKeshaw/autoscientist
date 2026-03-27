import json
import os
import queue
import sys
import threading
import time

from flask import Flask, Response, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from cognition.metrics import activation_entropy, contradiction_density
from cognition.runtime import CognitiveRuntime
from consolidator.consolidator import Consolidator
from dreamer.dreamer import DreamMode, Dreamer
from graph.brain import Brain
from memory.store import NetworkXMemoryStoreAdapter
from notebook.notebook import Notebook
from observer.observer import Observer
from reader.reader import Reader
from researcher.researcher import Researcher
from sandbox.sandbox import Sandbox

BRAIN_PATH = "data/brain.json"
OBSERVER_PATH = "data/observer.json"

app = Flask(__name__, template_folder="templates")
_event_queue = queue.Queue(maxsize=500)


def emit(event_type, data):
    try:
        _event_queue.put_nowait({"type": event_type, "data": data, "timestamp": time.time()})
    except queue.Full:
        pass


@app.route("/api/stream")
def api_stream():
    def generate():
        while True:
            try:
                event = _event_queue.get(timeout=20)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat', 'data': {}})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


brain = Brain()
memory = NetworkXMemoryStoreAdapter(brain)
observer = Observer(memory)
try:
    brain.load(BRAIN_PATH)
except FileNotFoundError:
    pass
try:
    observer.load(OBSERVER_PATH)
except FileNotFoundError:
    pass

runtime = CognitiveRuntime(memory, observer=observer, seed=brain.global_state.rng_seed)
dreamer = Dreamer(brain, research_agenda=observer)
consolidator = Consolidator(memory, observer=observer)
researcher = Researcher(memory, observer=observer)
reader = Reader(memory, observer=observer)
notebook = Notebook(brain, observer=observer)
sandbox = Sandbox(memory, observer=observer)

state = {"phase": "idle", "message": "Standing by", "cycle": 0, "active_node_id": None}


def save_state():
    brain.save(BRAIN_PATH)
    observer.save(OBSERVER_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    activations = {nid: (d.get("state") or {}).get("activation", 0.0) for nid, d in brain.all_nodes()}
    contradiction_edges = sum(1 for _, _, _, d in brain.graph.edges(keys=True, data=True) if d.get("type") == "contradicts")
    return jsonify(
        {
            "phase": state["phase"],
            "message": state["message"],
            "cycle": state["cycle"],
            "brain": brain.stats(),
            "brain_mode": brain.get_mode(),
            "agenda": len(observer.agenda),
            "resolved": sum(1 for i in observer.agenda if i.resolved),
            "reading_list": reader.stats(),
            "metrics": {
                "activation_entropy": activation_entropy(activations),
                "contradiction_density": contradiction_density(len(brain.graph.edges), contradiction_edges),
                "recent_novelty": brain.last_novelty,
                "uncertainty_mass": brain.stats().get("uncertainty_mass", 0.0),
            },
            "mode_transitions": observer.mode_history[-10:],
            "decision_records": observer.decision_records[-20:],
            "replay": {"seed": brain.global_state.rng_seed, "rng_cursor": brain.global_state.rng_cursor, "step_id": brain.global_state.step_id},
        }
    )


@app.route("/api/graph")
def api_graph():
    nodes, edges = [], []
    for nid, data in brain.all_nodes():
        st = data.get("state", {})
        nodes.append(
            {
                "id": nid,
                "statement": data.get("content", {}).get("text", data.get("statement", "")),
                "cluster": data.get("meta", {}).get("cluster", data.get("cluster", "")),
                "node_type": data.get("node_type", data.get("type", "concept")),
                "status": data.get("status", "uncertain"),
                "confidence": data.get("confidence", 0.5),
                "state": {
                    "activation": st.get("activation", data.get("activation", 0.0)),
                    "attention": st.get("attention", data.get("attention", 0.0)),
                    "value": st.get("value", data.get("value", 0.0)),
                    "uncertainty": st.get("uncertainty", data.get("uncertainty", 0.5)),
                    "stability": st.get("stability", data.get("stability", 0.0)),
                },
            }
        )
    for u, v, k, data in brain.graph.edges(keys=True, data=True):
        edges.append({"source": u, "target": v, "key": k, "type": data.get("type", "associated"), "weight": data.get("weight", 0.0), "confidence": data.get("confidence", 0.5), "narration": data.get("narration", "")})
    return jsonify({"nodes": nodes, "edges": edges, "schema_version": 3})


@app.route("/api/node/<node_id>")
def api_node(node_id):
    node = brain.get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    edges_out = []
    for nb in brain.neighbors(node_id):
        e = brain.get_edge(node_id, nb)
        nd = brain.get_node(nb)
        if e and nd:
            edges_out.append({"target_id": nb, "target_label": nd.get("content", {}).get("text", nd.get("statement", "")), "type": e.get("type", ""), "weight": e.get("weight", 0.0), "confidence": e.get("confidence", 0.5), "narration": e.get("narration", "")})
    return jsonify({**node, "edges_out": edges_out})


@app.route("/api/agenda")
def api_agenda():
    return jsonify([a.__dict__ for a in observer.get_prioritized_questions(100)])


@app.route("/api/emergences")
def api_emergences():
    return jsonify([e.to_dict() for e in observer.emergence_feed[-120:]])


@app.route("/api/notebook")
def api_notebook():
    return jsonify({"entries": notebook.get_all_for_display(), "running_hypothesis": notebook.running_hypothesis})


@app.route("/api/sandbox")
def api_sandbox():
    return jsonify([r.to_dict() for r in sandbox.results[-30:]])


@app.route("/api/mission", methods=["GET", "POST"])
def api_mission():
    if request.method == "POST":
        data = request.json or {}
        q = data.get("question", "").strip()
        if not q:
            return jsonify({"error": "no question"}), 400
        brain.set_mission(q, data.get("context", ""))
        save_state()
        return jsonify({"status": "set", "question": q})
    m = brain.get_mission()
    return jsonify({"set": bool(m), "question": m["question"] if m else "", "mode": brain.get_mode()})


@app.route("/api/run/<phase>", methods=["POST"])
def api_run_phase(phase):
    if state["phase"] != "idle":
        return jsonify({"error": "already running"}), 409

    def run():
        try:
            state["phase"] = phase
            state["message"] = f"Running {phase}"
            if phase == "dream":
                state["cycle"] += 1
                log = dreamer.dream(mode=DreamMode.WANDERING, steps=10, temperature=brain.cognitive_temperature, run_nrem=True, log_path=f"logs/dream_cycle{state['cycle']}.json")
                observer.observe(log)
                notebook.write_morning_entry(log, state["cycle"])
            elif phase == "research":
                log = researcher.research_day(max_questions=4, log_path=f"logs/research_cycle{state['cycle']}.json")
                notebook.write_field_notes(log, state["cycle"])
            elif phase == "consolidation":
                report = consolidator.consolidate(save_path=f"logs/consolidation_cycle{state['cycle']}.json")
                notebook.write_evening_entry(report, state["cycle"])
            elif phase == "reading":
                reader.reading_day(max_items=2)
            elif phase == "sandbox":
                sandbox.scan_and_test(max_tests=2)
            elif phase == "runtime":
                control = observer.decide()
                for _ in range(10):
                    step = runtime.step(control=control)
                    emit("runtime_step", step.__dict__)
            save_state()
            emit("phase_complete", {"phase": phase})
        except Exception as e:
            emit("error", {"message": str(e)})
        finally:
            state["phase"] = "idle"
            state["message"] = "Standing by"

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "phase": phase})


@app.route("/api/reading-list")
def api_reading_list():
    return jsonify(reader.list_all())


@app.route("/api/reading-list/add", methods=["POST"])
def api_reading_add():
    d = request.json or {}
    e = reader.add_to_list(url=d.get("url", ""), title=d.get("title", ""), source_type=d.get("source_type", "web"), priority=float(d.get("priority", 0.5)), added_by="user")
    save_state()
    return jsonify({"status": "added", "entry": e.to_dict()})


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    text = (request.json or {}).get("text", "")
    if not text:
        return jsonify({"error": "no text"}), 400
    reader.add_text(text, title="GUI text")
    save_state()
    return jsonify({"status": "ok"})


@app.route("/api/control/decisions")
def api_control_decisions():
    return jsonify(observer.decision_records[-200:])


@app.route("/api/replay/meta")
def api_replay_meta():
    return jsonify({"seed": brain.global_state.rng_seed, "rng_cursor": brain.global_state.rng_cursor, "step_id": brain.global_state.step_id, "schema_version": brain.global_state.schema_version})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
