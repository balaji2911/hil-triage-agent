"""
Fault injection.

Take a healthy trace, corrupt exactly one signal in one known way, and
return both the corrupted trace and the ground truth. That pairing is
the whole point: it's what lets the harness score a diagnosis as right
or wrong without a human in the loop.

Five fault types, all things a real HiL rig produces:

    stuck_at      signal freezes at its last good value
    dropout       the signal's message stops arriving for a window
    out_of_range  signal is pushed past its DBC maximum
    spike         a single sample jumps to an absurd value, then recovers
    drift         a linear offset grows from onset until the end of the run

Every fault has an onset time and (where meaningful) a duration, both
recorded in the label so a diagnosis can be checked for *when* as well
as *what*.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict

from .dbc import Signal

FAULT_TYPES = ("stuck_at", "dropout", "out_of_range", "spike", "drift")


@dataclass(frozen=True)
class FaultLabel:
    """Ground truth for one synthetic case."""
    fault_type: str
    signal: str
    message: str
    onset_t: float
    duration_s: float      # 0 for spike; run-to-end for drift

    def to_dict(self) -> dict:
        return asdict(self)


def inject_fault(
    rows: list[dict],
    target: tuple[str, Signal],
    fault_type: str,
    seed: int,
    duration_s: float = 3.0,
) -> tuple[list[dict], FaultLabel]:
    """Return (corrupted_rows, label). Never mutates the input trace.

    Onset is chosen in the middle 60% of the run so there's always a
    healthy stretch before it: a fault at t=0 has nothing to compare to.
    """
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"unknown fault type {fault_type!r}")

    rng = random.Random(seed)
    msg_name, sig = target
    t_end = rows[-1]["t"]
    onset = round(rng.uniform(0.2 * t_end, 0.8 * t_end), 1)
    span = sig.maximum - sig.minimum

    out: list[dict] = []
    last_good: float | None = None

    for r in rows:
        if r["sig"] != sig.name:
            out.append(r)
            continue

        t, v = r["t"], r["val"]
        in_window = onset <= t < onset + duration_s

        if t < onset:
            last_good = v
            out.append(r)
            continue

        if fault_type == "stuck_at" and in_window:
            out.append({**r, "val": last_good})

        elif fault_type == "dropout" and in_window:
            pass  # the row simply never arrives

        elif fault_type == "out_of_range" and in_window:
            out.append({**r, "val": round(sig.maximum + 0.2 * span, 4)})

        elif fault_type == "spike" and t == onset:
            out.append({**r, "val": round(sig.maximum + 2.0 * span, 4)})

        elif fault_type == "drift":
            # offset grows linearly from zero at onset to 50% of range at t_end
            frac = (t - onset) / (t_end - onset) if t_end > onset else 1.0
            out.append({**r, "val": round(v + 0.5 * span * frac, 4)})

        else:
            out.append(r)

    dur = {"spike": 0.0, "drift": round(t_end - onset, 1)}.get(fault_type, duration_s)
    label = FaultLabel(fault_type, sig.name, msg_name, onset, dur)
    return out, label


CLEAN_FRACTION = 0.2


def make_case(
    rows: list[dict],
    candidates: list[tuple[str, Signal]],
    seed: int,
) -> tuple[list[dict], FaultLabel]:
    """Pick a fault type and target signal from the seed, then inject.

    One seed -> one fully determined case. The harness iterates seeds.

    A fifth of cases are left clean and labelled "none". Without those,
    a detector that always reports a fault would score perfectly.
    """
    rng = random.Random(seed)
    if rng.random() < CLEAN_FRACTION:
        return list(rows), FaultLabel("none", "", "", 0.0, 0.0)
    fault_type = rng.choice(FAULT_TYPES)
    target = rng.choice(candidates)
    return inject_fault(rows, target, fault_type, seed=seed)
