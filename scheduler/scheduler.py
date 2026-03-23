import time
import json
import signal
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
 
from graph.brain import Brain
from ingestion.ingestor import Ingestor, EdgeSource
from dreamer.dreamer import Dreamer, DreamMode
from observer.observer import Observer
from consolidator.consolidator import Consolidator
from researcher.researcher import Researcher

# ── Config ────────────────────────────────────────────────────────────────────
 
# Default schedule (24h cycle, all times are local)
SCHEDULE = {
    "nrem_rem":     "0 23 * * *",    # 11pm  — dream cycle begins
    "research":     "0 9  * * *",    # 9am   — research day begins
    "consolidate":  "0 20 * * *",    # 8pm   — evening consolidation
}

# Cycle parameters
DREAM_STEPS         = 20
DREAM_TEMPERATURE   = 0.7
RESEARCH_QUESTIONS  = 5
RESEARCH_DEPTH      = "standard"
 
# Data paths
BRAIN_PATH          = "data/brain.json"
OBSERVER_PATH       = "data/observer.json"
CYCLE_LOG_PATH      = "logs/cycle_log.json"

# ── Cycle Log ─────────────────────────────────────────────────────────────────
 
@dataclass
class CycleEntry:
    cycle:      int
    phase:      str
    started_at: float = field(default_factory=time.time)
    ended_at:   float = 0.0
    summary:    str   = ""
    stats:      dict  = field(default_factory=dict)
    interrupted: bool = False
 
    def to_dict(self):
        return self.__dict__
 
class CycleLog:
    def __init__(self, path: str = CYCLE_LOG_PATH):
        self.path    = path
        self.entries = []
        self._load()
 
    def _load(self):
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
            self.entries = data.get('entries', [])
        except FileNotFoundError:
            self.entries = []
 
    def add(self, entry: CycleEntry):
        self.entries.append(entry.to_dict())
        self._save()
 
    def _save(self):
        import os
        os.makedirs("logs", exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump({"entries": self.entries}, f, indent=2)
 
    def current_cycle(self) -> int:
        if not self.entries:
            return 0
        return max(e.get('cycle', 0) for e in self.entries)

# ── Scheduler ─────────────────────────────────────────────────────────────────
 
class DreamerScheduler:
    def __init__(self):
        self.scheduler   = BackgroundScheduler()
        self.cycle_log   = CycleLog()
        self._running    = False
        self._interrupted = False
 
        # load persistent state
        self.brain    = Brain()
        self.observer = Observer(self.brain)
        self._load_state()
 
        # wire components
        self.ingestor    = Ingestor(self.brain, research_agenda=self.observer)
        self.dreamer     = Dreamer(self.brain, research_agenda=self.observer)
        self.consolidator = Consolidator(self.brain, observer=self.observer)
        self.researcher  = Researcher(
            self.brain, observer=self.observer, depth=RESEARCH_DEPTH
        )
 
        # graceful shutdown
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
 
    def _load_state(self):
        try:
            self.brain.load(BRAIN_PATH)
        except FileNotFoundError:
            print("No brain state found — starting fresh")
 
        try:
            self.observer.load(OBSERVER_PATH)
        except Exception:
            print("No observer state found — starting fresh")
 
    def _save_state(self):
        self.brain.save(BRAIN_PATH)
        self.observer.save(OBSERVER_PATH)
 
    def _handle_shutdown(self, signum, frame):
        print("\n── Interrupt received — saving state and shutting down ──")
        self._interrupted = True
        self._save_state()
        self.stop()
        sys.exit(0)

    # ── Phase runners ─────────────────────────────────────────────────────────
 
    def run_dream_phase(self):
        """
        Night phase: NREM + REM dream cycles.
        Runs two dream cycles — wandering then pressure.
        """
        cycle = self.cycle_log.current_cycle() + 1
        entry = CycleEntry(cycle=cycle, phase="dream")
        print(f"\n{'='*60}")
        print(f"DREAMER — Night Cycle {cycle}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
 
        try:
            # wandering dream
            log1 = self.dreamer.dream(
                mode        = DreamMode.WANDERING,
                steps       = DREAM_STEPS,
                temperature = DREAM_TEMPERATURE,
                run_nrem    = True,
                log_path    = f"logs/dream_cycle{cycle}_wandering.json"
            )
            self.observer.observe(log1)
 
            # pressure dream — shorter, focused on tensions
            log2 = self.dreamer.dream(
                mode        = DreamMode.PRESSURE,
                steps       = DREAM_STEPS // 2,
                temperature = DREAM_TEMPERATURE - 0.1,
                run_nrem    = False,   # NREM already ran
                log_path    = f"logs/dream_cycle{cycle}_pressure.json"
            )
            self.observer.observe(log2)
 
            entry.summary = log2.summary
            entry.stats   = self.brain.stats()
 
        except Exception as e:
            print(f"Dream phase error: {e}")
            entry.interrupted = True
 
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    def run_research_phase(self):
        """
        Day phase: researcher pulls top questions and investigates.
        """
        cycle = self.cycle_log.current_cycle()
        entry = CycleEntry(cycle=cycle, phase="research")
        print(f"\n{'='*60}")
        print(f"DREAMER — Research Day {cycle}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
 
        try:
            log = self.researcher.research_day(
                max_questions = RESEARCH_QUESTIONS,
                log_path      = f"logs/research_cycle{cycle}.json"
            )
            resolved = sum(
                1 for e in log.entries
                if e.resolved in ['partial', 'strong']
            )
            entry.summary = (f"Researched {len(log.entries)} questions, "
                             f"{resolved} resolved/advanced.")
            entry.stats   = self.brain.stats()
 
        except Exception as e:
            print(f"Research phase error: {e}")
            entry.interrupted = True
 
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    def run_consolidation_phase(self):
        """
        Evening phase: consolidate the day's findings into the graph.
        """
        cycle = self.cycle_log.current_cycle()
        entry = CycleEntry(cycle=cycle, phase="consolidation")
        print(f"\n{'='*60}")
        print(f"DREAMER — Evening Consolidation {cycle}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
 
        try:
            report = self.consolidator.consolidate(
                save_path = f"logs/consolidation_cycle{cycle}.json"
            )
            entry.summary = report.summary
            entry.stats   = self.brain.stats()
 
        except Exception as e:
            print(f"Consolidation phase error: {e}")
            entry.interrupted = True
 
        finally:
            entry.ended_at = time.time()
            self.cycle_log.add(entry)
            self._save_state()

    # ── Ingestion interface ───────────────────────────────────────────────────
 
    def ingest(self, text: str,
               source: EdgeSource = EdgeSource.CONVERSATION):
        """
        Public method — feed new text into the brain at any time.
        Safe to call while scheduler is running.
        """
        print(f"\n── Manual ingestion ({len(text)} chars) ──")
        self.ingestor.ingest(text, source=source)
        self._save_state()
 
    def seed_dream(self, concept: str):
        """
        Inject a seed concept into the next dream cycle.
        Adds it to the agenda as a high-priority seeded question.
        """
        item = self.observer.add_to_agenda(
            text      = concept,
            item_type = "question",
            cycle     = self.cycle_log.current_cycle()
        )
        item.priority = 0.95
        print(f"── Seed planted: {concept[:80]} ──")
        self._save_state()
 
    # ── Manual phase triggers ─────────────────────────────────────────────────
 
    def run_now(self, phase: str):
        """
        Manually trigger a phase immediately.
        phase: "dream", "research", "consolidation"
        """
        phases = {
            "dream":         self.run_dream_phase,
            "research":      self.run_research_phase,
            "consolidation": self.run_consolidation_phase,
        }
        if phase not in phases:
            print(f"Unknown phase: {phase}. Choose from {list(phases.keys())}")
            return
        print(f"── Manual trigger: {phase} ──")
        phases[phase]()
 
    # ── Scheduler control ─────────────────────────────────────────────────────
 
    def start(self, schedule: dict = None):
        """
        Start the automated sleep-wake cycle.
        Uses default schedule or custom one provided.
        """
        s = schedule or SCHEDULE
 
        self.scheduler.add_job(
            self.run_dream_phase,
            CronTrigger.from_crontab(s["nrem_rem"]),
            id="dream",
            replace_existing=True
        )
        self.scheduler.add_job(
            self.run_research_phase,
            CronTrigger.from_crontab(s["research"]),
            id="research",
            replace_existing=True
        )
        self.scheduler.add_job(
            self.run_consolidation_phase,
            CronTrigger.from_crontab(s["consolidate"]),
            id="consolidation",
            replace_existing=True
        )
 
        self.scheduler.start()
        self._running = True
 
        print(f"\n{'='*60}")
        print("DREAMER is running.")
        print(f"Dream cycle:   {s['nrem_rem']}")
        print(f"Research:      {s['research']}")
        print(f"Consolidation: {s['consolidate']}")
        print(f"Brain:         {self.brain.stats()}")
        print(f"Agenda:        {len(self.observer.agenda)} items")
        print(f"Emergences:    {len(self.observer.emergence_feed)}")
        print(f"{'='*60}")
        print("Press Ctrl+C to stop.\n")

    def stop(self):
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            print("── Scheduler stopped ──")
 
    def status(self) -> dict:
        return {
            "running":      self._running,
            "cycle":        self.cycle_log.current_cycle(),
            "brain":        self.brain.stats(),
            "agenda":       len(self.observer.agenda),
            "resolved":     sum(1 for i in self.observer.agenda if i.resolved),
            "emergences":   len(self.observer.emergence_feed),
            "next_jobs":    [
                {
                    "id":       job.id,
                    "next_run": str(job.next_run_time)
                }
                for job in self.scheduler.get_jobs()
            ] if self._running else []
        }
 
    def run_full_cycle_now(self):
        """
        Run a complete cycle immediately — useful for testing.
        Dream → Research → Consolidate in sequence.
        """
        print("\n── Running full cycle now ──")
        self.run_dream_phase()
        self.run_research_phase()
        self.run_consolidation_phase()
        print("\n── Full cycle complete ──")
        print(json.dumps(self.status(), indent=2))

# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="DREAMER Scheduler")
    parser.add_argument(
        "--mode",
        choices=["auto", "dream", "research", "consolidation", "cycle", "status"],
        default="auto",
        help=(
            "auto     = start automated schedule\n"
            "dream    = run dream phase now\n"
            "research = run research phase now\n"
            "consolidation = run consolidation now\n"
            "cycle    = run full cycle now\n"
            "status   = print status and exit"
        )
    )
    parser.add_argument(
        "--ingest", type=str, default=None,
        help="Text to ingest before starting"
    )
    parser.add_argument(
        "--seed", type=str, default=None,
        help="Concept to seed into next dream"
    )
    parser.add_argument(
        "--depth", type=str, default="standard",
        choices=["shallow", "standard", "deep"],
        help="Research depth"
    )
    args = parser.parse_args()
 
    RESEARCH_DEPTH = args.depth
    ds = DreamerScheduler()
 
    if args.ingest:
        ds.ingest(args.ingest)
 
    if args.seed:
        ds.seed_dream(args.seed)
 
    if args.mode == "auto":
        ds.start()
        # keep alive
        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            pass
 
    elif args.mode == "status":
        print(json.dumps(ds.status(), indent=2))
 
    elif args.mode == "cycle":
        ds.run_full_cycle_now()
 
    else:
        ds.run_now(args.mode)