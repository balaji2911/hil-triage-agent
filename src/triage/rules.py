"""
Rule-based baseline detector.

This is the bar the LLM has to beat. Each fault type gets one hand-written
check that looks for the fingerprint that fault leaves in a trace:

    dropout       gap      a signal is missing at timestamps where the rest
                           of the bus is still logging
    spike         range    exactly one sample outside [min, max], and the
                           next sample is back inside
    out_of_range  range    a run of samples outside [min, max] that all sit
                           at the same value (a step to a fixed bad value)
    drift         range    a run outside [min, max] whose values keep moving
                  or trend a bend in an otherwise smooth in-range signal
    stuck_at      flat     many consecutive samples with exactly equal values

The detector sees only what the LLM sees: the trace rows plus the DBC
definition (message, range, unit) of each signal in it. It does carry
assumptions that happen to match the synthetic data: healthy signals are
smooth and noisy (so never exactly repeat), and a drift is a linear ramp.
Those are reasonable priors for real sensor data too, but they make this
baseline an upper bound, not a neutral one. The thresholds below were
tuned on development seeds 1000-1199; the harness scores on seeds 0-99,
which were never used for tuning.

Precedence: a trace has at most one fault, but a check can misfire on a
healthy signal. When several signals are flagged we keep the finding from
the most specific check (gap and range violations are hard evidence; a
trend is a statistical judgement), so a weak drift guess never overrides
a definite dropout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dbc import Signal

# A signal counts as stuck after this many identical consecutive samples.
# At 10 Hz, 8 samples is 0.8 s. Healthy signals carry 2% noise, so even
# two equal samples in a row are rare; 8 leaves a wide margin.
FLAT_MIN_SAMPLES = 8

# A spike must jump by more than this fraction of the signal's range, both
# into and out of the bad sample. Healthy samples move ~2% of range per step.
SPIKE_MIN_JUMP_FRAC = 0.5

# A gap shorter than this many missing samples is treated as logging jitter.
GAP_MIN_SAMPLES = 3

# Trend check: a signal is modelled as a smooth curve (degree-5 polynomial),
# optionally plus a bend at time tau that starts a linear ramp. If adding the
# bend improves the fit by more than this F-statistic, we call it drift.
# Swept 15-50 on dev seeds: 20-30 give identical results (no false drift
# calls, 17/39 drifts fully right); 15 starts calling healthy signals
# drift, 35+ starts missing drifts. 25 sits in the middle of the plateau.
TREND_POLY_DEGREE = 5
TREND_F_THRESHOLD = 25.0

# Lower number = more specific evidence = wins when several signals flag.
PRECEDENCE = {"dropout": 0, "spike": 1, "out_of_range": 2, "stuck_at": 3, "drift": 4}


@dataclass(frozen=True)
class Diagnosis:
    """What a detector claims happened. Same shape for rules and LLM."""
    fault_type: str          # one of FAULT_TYPES, or "none"
    signal: str = ""
    onset_t: float = 0.0
    evidence: str = ""       # one line a human can check against the trace


NO_FAULT = Diagnosis("none", evidence="no check fired")


def _series(rows: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    # Non-numeric or NaN readings are dropped, so they show up as gaps.
    pts = [(r["t"], r["val"]) for r in rows
           if r["sig"] == name and isinstance(r["val"], (int, float)) and np.isfinite(r["val"])]
    if not pts:
        return np.array([]), np.array([])
    t, v = zip(*pts)
    return np.array(t, dtype=float), np.array(v, dtype=float)


def check_gap(timeline: np.ndarray, t: np.ndarray) -> tuple[float, int] | None:
    """Longest run of timeline stamps where this signal has no sample.

    The timeline is every timestamp any signal was logged at, so "missing"
    means the rest of the bus was alive while this signal was silent.
    """
    present = set(np.round(t, 3))
    best: tuple[float, int] | None = None
    run_start, run_len = None, 0
    for ts in np.round(timeline, 3):
        if ts in present:
            run_start, run_len = None, 0
            continue
        if run_start is None:
            run_start = float(ts)
        run_len += 1
        if run_len >= GAP_MIN_SAMPLES and (best is None or run_len > best[1]):
            best = (run_start, run_len)
    return best


def check_range(t: np.ndarray, v: np.ndarray, sig: Signal) -> Diagnosis | None:
    """Classify samples outside the DBC range as spike, out_of_range or drift."""
    bad = np.flatnonzero((v > sig.maximum) | (v < sig.minimum))
    if bad.size == 0:
        return None

    first = int(bad[0])
    span = sig.maximum - sig.minimum

    # Spike: one bad sample that jumps far away and comes straight back.
    # "Far" matters: a slow drift can graze the limit for a single sample too.
    if bad.size == 1 and 0 < first < v.size - 1:
        jump_in = abs(v[first] - v[first - 1])
        jump_out = abs(v[first + 1] - v[first])
        if min(jump_in, jump_out) > SPIKE_MIN_JUMP_FRAC * span:
            return Diagnosis("spike", sig.name, float(t[first]),
                             f"single sample {v[first]:g} outside [{sig.minimum:g}, {sig.maximum:g}] at t={t[first]:g}")

    bad_vals = v[bad]
    if bad.size > 1 and np.ptp(bad_vals) <= 1e-9 * max(span, 1.0):
        return Diagnosis("out_of_range", sig.name, float(t[first]),
                         f"{bad.size} samples pinned at {bad_vals[0]:g}, outside [{sig.minimum:g}, {sig.maximum:g}]")

    # Values that leave the range while still changing: a drift that has
    # grown big enough to cross the limit. The crossing time is late; the
    # trend fit estimates when the drift actually started.
    fit = trend_fit(t, v)
    onset = fit[1] if fit else float(t[first])
    return Diagnosis("drift", sig.name, onset,
                     f"{bad.size} moving samples outside range from t={t[first]:g}; trend bend at t={onset:g}")


def check_flat(t: np.ndarray, v: np.ndarray, sig: Signal) -> Diagnosis | None:
    """Longest run of exactly repeated values."""
    best_start, best_len = 0, 1
    start = 0
    for i in range(1, v.size + 1):
        if i < v.size and v[i] == v[start]:
            continue
        if i - start > best_len:
            best_start, best_len = start, i - start
        start = i
    if best_len < FLAT_MIN_SAMPLES:
        return None
    # The run begins at the last good sample; the freeze starts one after it.
    onset_idx = min(best_start + 1, v.size - 1)
    return Diagnosis("stuck_at", sig.name, float(t[onset_idx]),
                     f"{best_len} identical samples at {v[best_start]:g} from t={t[best_start]:g}")


def trend_fit(t: np.ndarray, v: np.ndarray) -> tuple[float, float] | None:
    """Best (F-statistic, bend time) for 'smooth curve + ramp starting at tau'.

    Compares two models by least squares:
        smooth : degree-5 polynomial in t
        bent   : smooth + c * max(0, t - tau), for every candidate tau
    F measures how much the bend reduces the squared error, relative to the
    noise left over. A big F means a real change of slope, not noise.
    """
    n = v.size
    base = np.vander(t, TREND_POLY_DEGREE + 1)
    if n < base.shape[1] + 12:
        return None
    coef, *_ = np.linalg.lstsq(base, v, rcond=None)
    rss0 = float(((v - base @ coef) ** 2).sum())

    best: tuple[float, float] | None = None
    for tau in t[5:-5]:
        x = np.column_stack([base, np.maximum(0.0, t - tau)])
        c, *_ = np.linalg.lstsq(x, v, rcond=None)
        rss1 = float(((v - x @ c) ** 2).sum())
        if not np.isfinite(rss1) or rss1 <= 0:
            continue
        f = (rss0 - rss1) / (rss1 / (n - x.shape[1]))
        if best is None or f > best[0]:
            best = (f, float(tau))
    return best


def check_trend(t: np.ndarray, v: np.ndarray, sig: Signal) -> Diagnosis | None:
    fit = trend_fit(t, v)
    if fit is None or fit[0] < TREND_F_THRESHOLD:
        return None
    return Diagnosis("drift", sig.name, fit[1], f"slope change at t={fit[1]:g} (F={fit[0]:.0f})")


def diagnose(rows: list[dict], signals: dict[str, tuple[str, Signal]]) -> Diagnosis:
    """Run every check on every signal; return the most specific finding.

    `signals` maps signal name -> (message name, Signal) for the signals in
    the trace, i.e. the DBC context both detectors are given.
    """
    timeline = np.array(sorted({r["t"] for r in rows}), dtype=float)
    findings: list[Diagnosis] = []

    for name, (_msg, sig) in signals.items():
        t, v = _series(rows, name)
        if t.size == 0:
            findings.append(Diagnosis("dropout", name, float(timeline[0]) if timeline.size else 0.0,
                                      "signal never appears"))
            continue

        gap = check_gap(timeline, t)
        if gap:
            findings.append(Diagnosis("dropout", name, gap[0],
                                      f"{gap[1]} consecutive samples missing from t={gap[0]:g}"))
            continue

        for check in (check_range, check_flat, check_trend):
            d = check(t, v, sig)
            if d:
                findings.append(d)
                break

    if not findings:
        return NO_FAULT
    return min(findings, key=lambda d: PRECEDENCE[d.fault_type])
