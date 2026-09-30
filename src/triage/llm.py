"""
LLM detector: one API call per case, trace in, structured diagnosis out.

Same inputs as the rule baseline, same output type (rules.Diagnosis), so
the harness can score both identically:

    input   the trace as a table + each signal's DBC message, range and unit
    output  fault_type, signal, onset_t, and a short root-cause explanation

Design choices worth defending:

  * Forced tool call, not "please reply in JSON". The API is told the model
    must call `report_diagnosis`, whose schema restricts fault_type to the
    six labels and signal to the six names actually in the trace. The model
    cannot invent a seventh fault type or misspell a signal; the output is
    parsed by the API, not by a regex over prose.
  * Temperature 0, so reruns give (near-)identical answers and a change in
    score means a change in the system, not luck.
  * Wide table (one row per timestamp, one column per signal). A missing
    sample is an empty cell, which is exactly how a dropout looks to an
    engineer reading a CANoe trace. It is also ~3x fewer tokens than one
    row per reading.
  * The prompt defines the five fault types in words but gives no
    thresholds. The rules have thresholds tuned on dev seeds; the LLM gets
    the same definitions a new test engineer would.
  * Plain urllib instead of the anthropic SDK: one POST, no dependency,
    and the request body is visible in one place.

Needs ANTHROPIC_API_KEY in the environment (or in a git-ignored .env file
at the repo root). Model defaults to claude-sonnet-5-5; override with the
TRIAGE_MODEL environment variable.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .dbc import Signal
from .faults import FAULT_TYPES
from .rules import Diagnosis

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5-5"

# USD per million tokens (input, output), from platform.claude.com pricing,
# checked 2026-10-01. Cost is computed from the token counts the API returns.
PRICES = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}

SYSTEM = (
    "You are a hardware-in-the-loop (HiL) validation engineer triaging a CAN "
    "trace from an automated test run. Exactly one signal may be faulty, or "
    "the run may be healthy. Diagnose from the data only."
)

FAULT_DEFINITIONS = """\
Fault types:
- stuck_at: the signal freezes at a constant value (its last good value) for a period.
- dropout: the signal's samples stop arriving for a period while other signals keep logging (empty cells).
- out_of_range: the signal jumps to a constant value outside its DBC [min, max] range for a period.
- spike: a single sample jumps to an absurd value, then the signal immediately recovers.
- drift: an offset grows steadily from the onset until the end of the run; values may or may not leave the range.
- none: every signal behaves plausibly for the whole run.
onset_t is the timestamp of the first affected sample. For none, use signal "" and onset_t 0."""


def _tool(signal_names: list[str]) -> dict:
    return {
        "name": "report_diagnosis",
        "description": "Report the single most likely diagnosis for this trace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fault_type": {"type": "string", "enum": ["none", *FAULT_TYPES]},
                "signal": {"type": "string", "enum": ["", *signal_names]},
                "onset_t": {"type": "number", "description": "seconds"},
                "explanation": {
                    "type": "string",
                    "description": "2-3 sentences: what in the trace shows the fault, and a plausible root cause on a HiL rig.",
                },
            },
            "required": ["fault_type", "signal", "onset_t", "explanation"],
        },
    }


def format_trace(rows: list[dict], signals: dict[str, tuple[str, Signal]]) -> str:
    """Wide CSV: t, then one column per signal; empty cell = no sample logged."""
    names = list(signals)
    grid: dict[float, dict[str, float]] = {}
    for r in rows:
        grid.setdefault(r["t"], {})[r["sig"]] = r["val"]
    lines = ["t," + ",".join(names)]
    for t in sorted(grid):
        cells = [f"{grid[t][n]:.6g}" if n in grid[t] else "" for n in names]
        lines.append(f"{t:g}," + ",".join(cells))
    return "\n".join(lines)


def build_prompt(rows: list[dict], signals: dict[str, tuple[str, Signal]]) -> str:
    table = ["signal | message | min | max | unit"]
    for name, (msg, s) in signals.items():
        table.append(f"{name} | {msg} | {s.minimum:g} | {s.maximum:g} | {s.unit}")
    return (
        "Signals in this trace (from the DBC):\n" + "\n".join(table)
        + "\n\n" + FAULT_DEFINITIONS
        + "\n\nTrace (10 Hz):\n" + format_trace(rows, signals)
        + "\n\nCall report_diagnosis once with your diagnosis."
    )


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (environment or .env at repo root)")
    return key


def _http_post(body: dict) -> dict:
    """POST to the Messages API with simple retries on rate limits and overloads."""
    headers = {
        "x-api-key": _api_key(),
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }
    for attempt in range(5):
        req = urllib.request.Request(API_URL, json.dumps(body).encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2 ** attempt * 2)
                continue
            # Some newer models reject sampling parameters; retry once without.
            if e.code == 400 and "temperature" in detail and "temperature" in body:
                body = {k: v for k, v in body.items() if k != "temperature"}
                continue
            raise RuntimeError(f"API error {e.code}: {detail[:300]}") from None
    raise RuntimeError("API still failing after retries")


def diagnose_llm(
    rows: list[dict],
    signals: dict[str, tuple[str, Signal]],
    model: str | None = None,
    post: Callable[[dict], dict] = _http_post,
) -> tuple[Diagnosis, dict]:
    """Return (Diagnosis, meta) where meta has latency, tokens and cost.

    `post` is injectable so tests can run without a network or a key.
    """
    model = model or os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)
    body = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0,
        "system": SYSTEM,
        "tools": [_tool(list(signals))],
        "tool_choice": {"type": "tool", "name": "report_diagnosis"},
        "messages": [{"role": "user", "content": build_prompt(rows, signals)}],
    }

    t0 = time.perf_counter()
    resp = post(body)
    latency = time.perf_counter() - t0

    call = next((b for b in resp.get("content", []) if b.get("type") == "tool_use"), None)
    if call is None:
        raise RuntimeError(f"no tool call in response: {str(resp)[:300]}")
    out = call["input"]

    usage = resp.get("usage", {})
    tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    meta = {
        "latency_s": latency,
        "input_tokens": tin,
        "output_tokens": tout,
        "cost_usd": (tin * p_in + tout * p_out) / 1e6,
        "model": model,
    }
    diag = Diagnosis(
        fault_type=out["fault_type"],
        signal=out.get("signal", "") if out["fault_type"] != "none" else "",
        onset_t=float(out.get("onset_t", 0.0)),
        evidence=out.get("explanation", ""),
    )
    return diag, meta
