"""
Evaluation harness.

One seed -> one case: pick 6 signals, generate a healthy trace, maybe
inject a fault, keep the FaultLabel as the answer key. Every detector
(rules, LLM) sees the same trace and the same DBC context for its signals,
returns a Diagnosis, and is scored against the label.

Scoring is exact on structured fields, no judgement calls:

    type_ok    predicted fault_type == true fault_type ("none" included)
    signal_ok  type_ok and, for a real fault, the same signal
    onset_ok   signal_ok and, for a real fault, |onset error| <= 0.5 s
    exact      = onset_ok: the whole diagnosis is right

The headline number is exact-diagnosis accuracy. The other three show
where a detector loses points (wrong type? right type, wrong signal?).

Seeds 0-99 are the evaluation set. Seeds 1000+ are for development and
threshold tuning; nothing is ever tuned on the evaluation seeds.

Usage:
    python -m triage.harness                      # rules only, seeds 0-99
    python -m triage.harness --llm                # rules + LLM (needs ANTHROPIC_API_KEY)
    python -m triage.harness --llm --n 10         # cheap smoke test first
    python -m triage.harness --start 1000 --n 200 # development seeds
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .dbc import Message, Signal, measurement_signals, parse_dbc
from .faults import FAULT_TYPES, FaultLabel, make_case
from .rules import Diagnosis, diagnose
from .trace import TraceConfig, generate_trace

REPO = Path(__file__).resolve().parents[2]
DBC_PATH = REPO / "data" / "gen" / "hyundai_2015_ccan.dbc"
RESULTS_DIR = REPO / "results"

SIGNALS_PER_CASE = 6
ONSET_TOLERANCE_S = 0.5   # five samples at 10 Hz


@dataclass
class Case:
    seed: int
    rows: list[dict]
    signals: dict[str, tuple[str, Signal]]   # signal name -> (message name, Signal)
    label: FaultLabel


def build_case(messages: dict[str, Message], seed: int) -> Case:
    """Deterministic: the same seed always gives the same case.

    Signal choice uses its own random stream (seeded with a string) so it
    doesn't share draws with make_case, which seeds Random(seed) itself.
    """
    pool = measurement_signals(messages)
    selected = random.Random(f"signals:{seed}").sample(pool, SIGNALS_PER_CASE)
    healthy = generate_trace(selected, TraceConfig(seed=seed))
    rows, label = make_case(healthy, selected, seed)
    return Case(seed, rows, {s.name: (m, s) for m, s in selected}, label)


@dataclass
class Score:
    type_ok: bool
    signal_ok: bool
    onset_ok: bool

    @property
    def exact(self) -> bool:
        return self.onset_ok


def score(pred: Diagnosis, label: FaultLabel) -> Score:
    type_ok = pred.fault_type == label.fault_type
    if label.fault_type == "none":
        return Score(type_ok, type_ok, type_ok)
    signal_ok = type_ok and pred.signal == label.signal
    onset_ok = signal_ok and abs(pred.onset_t - label.onset_t) <= ONSET_TOLERANCE_S
    return Score(type_ok, signal_ok, onset_ok)


def run_rules(case: Case) -> tuple[Diagnosis, dict]:
    t0 = time.perf_counter()
    d = diagnose(case.rows, case.signals)
    return d, {"latency_s": time.perf_counter() - t0, "cost_usd": 0.0}


def run_llm(case: Case) -> tuple[Diagnosis, dict]:
    from .llm import diagnose_llm   # imported lazily: rules-only runs need no API key
    return diagnose_llm(case.rows, case.signals)


def evaluate(name: str, runner, cases: list[Case], out_dir: Path) -> dict:
    """Run one detector over all cases, write per-case records, return a summary."""
    records = []
    for case in cases:
        try:
            pred, meta = runner(case)
        except Exception as e:           # one failed call must not kill a 100-case run
            pred, meta = Diagnosis("error", evidence=f"{type(e).__name__}: {e}"), {"latency_s": 0.0, "cost_usd": 0.0}
        s = score(pred, case.label)
        records.append({
            "seed": case.seed,
            "label": case.label.to_dict(),
            "pred": asdict(pred),
            "score": {**asdict(s), "exact": s.exact},
            **meta,
        })
        print(f"  [{name}] seed {case.seed:>4}  true={case.label.fault_type:<12} pred={pred.fault_type:<12} "
              f"{'OK' if s.exact else '--'}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return summarise(name, records)


def summarise(name: str, records: list[dict]) -> dict:
    n = len(records)
    pct = lambda k: round(100 * sum(r["score"][k] for r in records) / n, 1)
    per_type = {}
    for ft in ("none",) + FAULT_TYPES:
        sub = [r for r in records if r["label"]["fault_type"] == ft]
        if sub:
            per_type[ft] = f'{sum(r["score"]["exact"] for r in sub)}/{len(sub)}'
    confusion = Counter((r["label"]["fault_type"], r["pred"]["fault_type"]) for r in records)
    lat = [r["latency_s"] for r in records]
    return {
        "detector": name,
        "cases": n,
        "exact_pct": pct("exact"),
        "type_pct": pct("type_ok"),
        "signal_pct": pct("signal_ok"),
        "per_type_exact": per_type,
        "errors": sum(r["pred"]["fault_type"] == "error" for r in records),
        "cost_usd_total": round(sum(r["cost_usd"] for r in records), 4),
        "cost_usd_per_case": round(sum(r["cost_usd"] for r in records) / n, 5),
        "latency_s_median": round(statistics.median(lat), 4),
        "latency_s_p95": round(sorted(lat)[max(0, int(0.95 * n) - 1)], 4),
        "confusion": {f"{a} -> {b}": c for (a, b), c in sorted(confusion.items()) if a != b},
    }


def print_table(summaries: list[dict]) -> None:
    cols = [("detector", 9), ("exact_pct", 8), ("type_pct", 8), ("signal_pct", 10),
            ("cost_usd_per_case", 17), ("latency_s_median", 16), ("errors", 6)]
    print("\n" + "  ".join(c.ljust(w) for c, w in cols))
    for s in summaries:
        print("  ".join(str(s[c]).ljust(w) for c, w in cols))
    for s in summaries:
        print(f"\n{s['detector']} exact by true type: {s['per_type_exact']}")
        print(f"{s['detector']} mistakes (true -> predicted): {s['confusion'] or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=0, help="first seed (default 0)")
    ap.add_argument("--n", type=int, default=100, help="number of cases (default 100)")
    ap.add_argument("--llm", action="store_true", help="also run the LLM detector")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    messages = parse_dbc(DBC_PATH)
    cases = [build_case(messages, s) for s in range(args.start, args.start + args.n)]
    print(f"{len(cases)} cases, seeds {args.start}-{args.start + args.n - 1}: "
          f"{dict(Counter(c.label.fault_type for c in cases))}")

    summaries = [evaluate("rules", run_rules, cases, args.out)]
    if args.llm:
        summaries.append(evaluate("llm", run_llm, cases, args.out))

    print_table(summaries)
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"seeds": [args.start, args.start + args.n - 1], "summaries": summaries}, f, indent=2)
    print(f"\nwrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
