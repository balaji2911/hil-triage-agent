"""
Print one case at every stage of the pipeline, for reading, not scoring.

    python -m triage.show 7          # case for seed 7
    python -m triage.show 7 --llm    # also call the LLM on it (needs a key)

Shows: the 6 signals picked and their DBC definitions, the healthy trace
around the fault, the same stretch after injection, the FaultLabel, and
what each detector concluded.
"""

from __future__ import annotations

import argparse

from .dbc import parse_dbc
from .harness import DBC_PATH, build_case, score
from .llm import format_trace
from .rules import diagnose
from .trace import TraceConfig, generate_trace


def _window(table: str, centre: float, half: float = 0.6) -> str:
    lines = table.splitlines()
    keep = [l for l in lines[1:] if abs(float(l.split(",")[0]) - centre) <= half + 1e-9]
    return "\n".join([lines[0], *keep])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    case = build_case(parse_dbc(DBC_PATH), args.seed)
    print(f"== 1. signals picked for seed {args.seed} (dbc.py -> harness.build_case)")
    for name, (msg, s) in case.signals.items():
        print(f"   {name:<16} {msg:<12} [{s.minimum:g}, {s.maximum:g}] {s.unit}")

    healthy = generate_trace([(m, s) for m, s in case.signals.values()], TraceConfig(seed=args.seed))
    centre = case.label.onset_t or 5.0
    print(f"\n== 2. healthy trace around t={centre:g} (trace.py)")
    print(_window(format_trace(healthy, case.signals), centre))

    print(f"\n== 3. after fault injection (faults.py)")
    print(_window(format_trace(case.rows, case.signals), centre))

    print(f"\n== 4. answer key: {case.label}")

    d = diagnose(case.rows, case.signals)
    print(f"\n== 5. rules.py: {d.fault_type} {d.signal} t={d.onset_t:g}  exact={score(d, case.label).exact}")
    print(f"      evidence: {d.evidence}")

    if args.llm:
        from .llm import diagnose_llm
        d, meta = diagnose_llm(case.rows, case.signals)
        print(f"\n== 6. llm.py: {d.fault_type} {d.signal} t={d.onset_t:g}  exact={score(d, case.label).exact}"
              f"  (${meta['cost_usd']:.4f}, {meta['latency_s']:.1f} s)")
        print(f"      explanation: {d.evidence}")


if __name__ == "__main__":
    main()
