"""
Clean trace generation.

A trace is what a HiL rig logs: a time-ordered list of signal readings.
We generate a *healthy* one first, with every signal behaving plausibly
and staying inside its DBC range. The fault injector then corrupts one
signal in a known way, which is what gives us labelled data.

Row format, kept deliberately flat so it's trivial to serialise:

    {"t": 0.10, "msg": "WHL_SPD11", "sig": "WHL_SPD_FL", "val": 12.4}

Waveform choice: a slow sinusoid plus small noise, scaled to sit in the
middle half of each signal's range. It isn't physically faithful, but it
gives every signal visible variation, which matters because "stuck-at"
faults are invisible on a signal that never moved.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .dbc import Message, Signal


@dataclass(frozen=True)
class TraceConfig:
    duration_s: float = 10.0     # length of the run
    period_s: float = 0.1        # sample interval; 10 Hz keeps traces readable
    noise_frac: float = 0.02     # noise amplitude as a fraction of range
    seed: int = 0


def _waveform(sig: Signal, t: float, phase: float, rng: random.Random, cfg: TraceConfig) -> float:
    """Plausible healthy value for `sig` at time `t`.

    Centred on the range midpoint, swinging over a quarter of the range,
    with a per-signal phase so signals don't all move in lockstep.
    """
    span = sig.maximum - sig.minimum
    mid = sig.minimum + span / 2
    amp = span / 4
    noise = rng.gauss(0, cfg.noise_frac * span)
    v = mid + amp * math.sin(2 * math.pi * t / cfg.duration_s + phase) + noise
    return max(sig.minimum, min(sig.maximum, v))


def generate_trace(
    selected: list[tuple[str, Signal]],
    cfg: TraceConfig = TraceConfig(),
) -> list[dict]:
    """Return a healthy trace for the given (message_name, Signal) pairs.

    Deterministic for a given seed, which the harness relies on: the same
    case number must always produce the same trace.
    """
    rng = random.Random(cfg.seed)
    phases = {s.name: rng.uniform(0, 2 * math.pi) for _, s in selected}

    rows = []
    n = int(cfg.duration_s / cfg.period_s)
    for i in range(n):
        t = round(i * cfg.period_s, 3)
        for msg_name, sig in selected:
            rows.append({
                "t": t,
                "msg": msg_name,
                "sig": sig.name,
                "val": round(_waveform(sig, t, phases[sig.name], rng, cfg), 4),
            })
    return rows


def select_signals(messages: dict[str, Message], names: list[str]) -> list[tuple[str, Signal]]:
    """Look up signals by name across all messages. Fails loudly on a typo."""
    index = {s.name: (m.name, s) for m in messages.values() for s in m.signals.values()}
    missing = [n for n in names if n not in index]
    if missing:
        raise KeyError(f"signals not in DBC: {missing}")
    return [index[n] for n in names]
