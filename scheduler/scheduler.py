import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from cognition.runtime import CognitiveRuntime
from consolidator.consolidator import Consolidator
from dreamer.dreamer import DreamMode, Dreamer
from graph.brain import Brain, BrainMode
from memory.store import NetworkXMemoryStoreAdapter
from notebook.notebook import Notebook
from observer.observer import Observer
from reader.reader import Reader
from researcher.researcher import Researcher
from sandbox.sandbox import Sandbox

SCHEDULE = {"cycle": "0 */6 * * *"}
BRAIN_PATH = "data/brain.json"
OBSERVER_PATH = "data/observer.json"


@dataclass
class CycleEntry:
    cycle: int
    phase: str
    brain_mode: str
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    summary: str = ""
    interrupted: bool = False
    manifest_path: str = ""

    def to_dict(self):
        return self.__dict__


class CycleLog:
    def __init__(self, path="logs/cycle_log.json"):
        self.path = path
        self.entries = []
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                self.entries = json.load(f).get("entries", [])
        except FileNotFoundError:
            self.entries = []

    def add(self, entry: CycleEntry):
        os.makedirs("logs", exist_ok=True)
        self.entries.append(entry.to_dict())
        with open(self.path, "w") as f:
            json.dump({"entries": self.entries}, f, indent=2)

    def current_cycle(self):
        return max((e.get("cycle", 0) for e in self.entries), default=0)


class DreamerScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.cycle_log = CycleLog()
        self._running = False

        self.brain = Brain()
        self.memory = NetworkXMemoryStoreAdapter(self.brain)
        self.observer = Observer(self.memory)
        self._load_state()

        self.runtime = CognitiveRuntime(self.memory, observer=self.observer, seed=self.brain.global_state.rng_seed)
        self.dreamer = Dreamer(self.brain, research_agenda=self.observer)  # compatibility mode
        self.consolidator = Consolidator(self.memory, observer=self.observer)
        self.researcher = Researcher(self.memory, observer=self.observer)
        self.reader = Reader(self.memory, observer=self.observer)
        self.sandbox = Sandbox(self.memory, observer=self.observer)
        self.notebook = Notebook(self.brain, observer=self.observer)

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _load_state(self):
        try:
            self.brain.load(BRAIN_PATH)
        except FileNotFoundError:
            pass
        try:
            self.observer.load(OBSERVER_PATH)
        except FileNotFoundError:
            pass

    def _save_state(self):
        self.brain.save(BRAIN_PATH)
        self.observer.save(OBSERVER_PATH)

    def _shutdown(self, signum, frame):
        self._save_state()
        self.stop()
        sys.exit(0)

    def _write_manifest(self, cycle: int, control, steps, action_records):
        os.makedirs("logs", exist_ok=True)
        path = f"logs/run_manifest_cycle{cycle}.json"
        payload = {
            "cycle": cycle,
            "timestamp": time.time(),
            "control": {
                "mode": control.mode,
                "temperature": control.temperature,
                "action_set": control.action_set,
                "budgets": control.budgets,
                "gains": control.gains,
            },
            "steps": [s.__dict__ for s in steps],
            "actions": action_records,
            "decision_record": self.observer.decision_records[-1] if self.observer.decision_records else {},
            "rng": {"seed": self.brain.global_state.rng_seed, "cursor": self.brain.global_state.rng_cursor},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def run_state_cycle(self):
        cycle = self.cycle_log.current_cycle() + 1
        control = self.observer.decide()
        self.brain.set_mode(BrainMode(control.mode))
        entry = CycleEntry(cycle=cycle, phase="state_cycle", brain_mode=control.mode)
        actions = []
        try:
            steps = self.runtime.run_cycle(steps=control.budgets.get("steps", 12), control=control)

            if "acquire" in control.action_set:
                rlog = self.researcher.research_day(max_questions=control.budgets.get("external", 2), log_path=f"logs/research_cycle{cycle}.json")
                self.notebook.write_field_notes(rlog, cycle)
                actions.append({"type": "research", "count": len(rlog.entries)})
                rr = self.reader.reading_day(max_items=1)
                actions.append({"type": "reading", "count": len(rr)})

            if "consolidate" in control.action_set:
                rep = self.consolidator.consolidate(save_path=f"logs/consolidation_cycle{cycle}.json")
                self.notebook.write_evening_entry(rep, cycle)
                self.notebook.update_running_hypothesis(cycle)
                actions.append({"type": "consolidation", "delta_objective": rep.delta_objective})

            if "test" in control.action_set:
                t = self.sandbox.scan_and_test(max_tests=control.budgets.get("tests", 1))
                actions.append({"type": "test", "count": len(t)})

            if "wander" in control.action_set or "focus" in control.action_set:
                dlog = self.dreamer.dream(mode=DreamMode.WANDERING if control.mode == "wandering" else DreamMode.PRESSURE, steps=8, temperature=control.temperature, run_nrem=False, log_path=f"logs/dream_cycle{cycle}.json")
                self.observer.observe(dlog)
                self.notebook.write_morning_entry(dlog, cycle)
                actions.append({"type": "dream", "steps": len(dlog.steps)})

            manifest = self._write_manifest(cycle, control, steps, actions)
            self.memory.snapshot(tag=f"cycle{cycle}")
            entry.manifest_path = manifest
            entry.summary = f"mode={control.mode}, steps={len(steps)}, actions={[a['type'] for a in actions]}"
        except Exception as e:
            entry.interrupted = True
            entry.summary = str(e)
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    def run_now(self, phase):
        if phase in ("cycle", "dream", "research", "reading", "consolidation", "sandbox"):
            self.run_state_cycle()

    def run_full_cycle_now(self):
        self.run_state_cycle()

    def set_mission(self, question, context=""):
        self.brain.set_mission(question, context)
        self._save_state()

    def suspend_mission(self):
        self.brain.suspend_mission()
        self._save_state()

    def resume_mission(self):
        self.brain.resume_mission()
        self._save_state()

    def seed_dream(self, concept):
        item = self.observer.add_to_agenda(concept, cycle=self.cycle_log.current_cycle())
        item.priority = 0.95
        self._save_state()

    def status(self):
        return {
            "running": self._running,
            "cycle": self.cycle_log.current_cycle(),
            "brain_mode": self.brain.get_mode(),
            "brain": self.brain.stats(),
            "agenda": len(self.observer.agenda),
            "resolved": sum(1 for i in self.observer.agenda if i.resolved),
            "decision_records": len(self.observer.decision_records),
            "reading_list": self.reader.stats(),
        }

    def start(self, schedule=None):
        sched = schedule or SCHEDULE
        self.scheduler.add_job(self.run_state_cycle, CronTrigger.from_crontab(sched["cycle"]), id="cycle", replace_existing=True)
        self.scheduler.start()
        self._running = True

    def stop(self):
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autodreamer scheduler")
    parser.add_argument("--mode", choices=["auto", "cycle", "status"], default="auto")
    args = parser.parse_args()

    ds = DreamerScheduler()
    if args.mode == "status":
        print(json.dumps(ds.status(), indent=2))
    elif args.mode == "cycle":
        ds.run_full_cycle_now()
    else:
        ds.start()
        print(f"Running state-driven scheduler at {datetime.now().isoformat()}")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            ds.stop()
