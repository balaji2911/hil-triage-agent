"""
LLM detector: one API call per case, trace in, structured diagnosis out.

Same inputs as the rule baseline, same output type (rules.Diagnosis), so
the harness can score both identically:

    input   the trace as a table + each signal's DBC message, range and unit
    output  fault_type, signal, onset_t, and a short root-cause explanation

Design choices worth defending:

  * Structured outputs (`output_config.format` with a JSON schema), not
    "please reply in JSON". The API constrains decoding so the reply is
    valid JSON in that shape, with fault_type limited to the six labels and
    signal limited to the names actually in the trace. The code still
    validates the result, because the docs note enum casing isn't
    guaranteed, and a truncated or refused reply can break the schema.
    (Forced tool calls, the older way to do this, return a 400 on
    Sonnet 5.5 / Opus 5.5.)
  * No temperature setting. Current models reject non-default sampling
    parameters, so run-to-run variation is handled the scientific way:
    rerun the harness and report the spread.
  * The model's default adaptive thinking is left on. Thinking tokens are
    billed as output and count toward max_tokens, hence the 8k budget.
  * Wide table (one row per timestamp, one column per signal). A missing
    sample is an empty cell, which is exactly how a dropout looks to an
    engineer reading a CANoe trace. It is also ~3x fewer tokens than one
    row per reading.
  * The prompt defines the five fault types in words but gives no
    thresholds. The rules have thresholds tuned on dev seeds; the LLM gets
    the same definitions a new test engineer would.
  * Plain urllib instead of the anthropic SDK: one POST, no dependency,
    and the whole request body is visible in one place.

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
MAX_TOKENS = 8000

# USD per million tokens (input, output), from platform.claude.com pricing,
# checked 2026-10-01. Cost is computed from the token counts the API returns.
PRICES = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}

LABELS = ("none", *FAULT_TYPES)

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


def output_schema(signal_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "fault_type": {"type": "string", "enum": list(LABELS)},
            "signal": {"type": "string", "enum": ["", *signal_names]},
            "onset_t": {"type": "number", "description": "seconds"},
            "explanation": {
                "type": "string",
                "description": "2-3 sentences: what in the trace shows the fault, and a plausible root cause on a HiL rig.",
            },
        },
        "required": ["fault_type", "signal", "onset_t", "explanation"],
        "additionalProperties": False,
    }


def format_trace(rows: list[dict], signals: dict[str, tuple[str, Signal]]) -> str:
    """Wide CSV: t, then one column per signal; empty cell = no sample logged."""
    names = list(signals)
    grid: dict[float, dict[str, float]] = {}
    for r in rows:
        grid.setdefault(r["t"], {})[r["sig"]] = r["val"]
    lines = ["t," + ",".join(names)]
    for t in sorted(grid):
        cells = [f"{grid[t][n]:.10g}" if n in grid[t] else "" for n in names]
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
        + "\n\nReport your diagnosis."
    )


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not key and env_file.exists():
        try:
            # utf-8-sig also accepts the byte-order mark Notepad adds.
            text = env_file.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raise RuntimeError(
                ".env is not UTF-8 (PowerShell 5 '>' writes UTF-16). Recreate it with: "
                'Set-Content .env "ANTHROPIC_API_KEY=sk-ant-..." -Encoding utf8'
            ) from None
        for line in text.splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (environment or .env at repo root)")
    return key


RETRY_STATUS = (429, 500, 502, 503, 529)


def _http_post(body: dict, attempts: int = 5) -> dict:
    """POST to the Messages API, retrying rate limits, overloads and network blips.

    A 400/401/403 is a bug in the request or the key: retrying can't fix
    it, so it is raised at once.
    """
    headers = {
        "x-api-key": _api_key(),
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }
    data = json.dumps(body).encode()
    for attempt in range(attempts):
        last = attempt == attempts - 1
        req = urllib.request.Request(API_URL, data, headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code not in RETRY_STATUS or last:
                raise RuntimeError(f"API error {e.code}: {detail[:300]}") from None
            wait = float(e.headers.get("retry-after") or 2 ** (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if last:
                raise RuntimeError(f"network error: {e}") from None
            wait = 2 ** (attempt + 1)
        time.sleep(min(wait, 60))
    raise AssertionError("unreachable")


_unpriced_warned: set[str] = set()


def diagnose_llm(
    rows: list[dict],
    signals: dict[str, tuple[str, Signal]],
    model: str | None = None,
    post: Callable[[dict], dict] = _http_post,
) -> tuple[Diagnosis, dict]:
    """Return (Diagnosis, meta) where meta has latency, tokens and cost.

    `post` is injectable so tests can run without a network or a key.
    Raises on a reply that is truncated, refused, or doesn't fit the schema;
    the harness records that case as an error.
    """
    model = model or os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)
    names = list(signals)
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": build_prompt(rows, signals)}],
        "output_config": {"format": {"type": "json_schema", "schema": output_schema(names)}},
    }

    t0 = time.perf_counter()
    resp = post(body)
    latency = time.perf_counter() - t0

    stop = resp.get("stop_reason")
    if stop in ("max_tokens", "refusal"):
        raise RuntimeError(f"reply ended with stop_reason={stop}")
    text = next((b["text"] for b in resp.get("content", []) if b.get("type") == "text"), None)
    if text is None:
        raise RuntimeError(f"no text block in response: {str(resp)[:300]}")
    out = json.loads(text)

    fault_type = str(out.get("fault_type", "")).strip().lower()
    if fault_type not in LABELS:
        raise RuntimeError(f"unknown fault_type {out.get('fault_type')!r}")
    by_lower = {n.lower(): n for n in names}
    signal = by_lower.get(str(out.get("signal", "")).strip().lower(), None)
    if fault_type != "none" and signal is None:
        raise RuntimeError(f"unknown signal {out.get('signal')!r}")

    usage = resp.get("usage", {})
    tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    if model not in PRICES and model not in _unpriced_warned:
        _unpriced_warned.add(model)
        print(f"warning: no price for {model}; cost reported as 0")
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    meta = {
        "latency_s": latency,
        "input_tokens": tin,
        "output_tokens": tout,      # includes thinking tokens
        "cost_usd": (tin * p_in + tout * p_out) / 1e6,
        "model": model,
    }
    diag = Diagnosis(
        fault_type=fault_type,
        signal=signal if fault_type != "none" else "",
        onset_t=float(out.get("onset_t", 0.0)),
        evidence=out.get("explanation", ""),
    )
    return diag, meta
