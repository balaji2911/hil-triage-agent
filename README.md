# HiL triage agent

An assistant that reads a failed CAN trace from a hardware-in-the-loop (HiL) rig, identifies the fault and drafts a root-cause explanation. It uses only public data: the DBC comes from [commaai/opendbc](https://github.com/commaai/opendbc), and every trace and fault is synthetic.

Stage 1 asks one question: **on 100 labelled cases, how does one LLM call compare with hand-written rules?**

## Results (stage 1, evaluation seeds 0–99)

| Detector | Exact diagnosis | Fault type right | Cost / case | Median latency |
|---|---|---|---|---|
| Rules | **90%** | 95% | $0 | 15 ms |
| LLM (claude-sonnet-5-5) | *not run yet* | | | |

An exact diagnosis means the fault type, the signal and the onset (within ±0.5 s) are all correct. Clean runs count as correct when the detector says `none`.

The rules get every stuck_at, dropout, out_of_range and spike case right. They lose points on drift (11/20): four slow drifts that never leave the DBC range go unnoticed, and five are detected but the onset estimate is off by more than 0.5 s. They also make one false alarm: a clean run (seed 32) scores just above the trend threshold, which was tuned on the dev seeds and deliberately not re-tuned after seeing this. For faults this well defined, rules are the right tool, and the LLM has to justify itself against that bar.

## Pipeline

```
DBC ──dbc.py──▶ messages ──measurement_signals──▶ 150 candidate signals
                                                        │ pick 6 per case (seeded)
                                             trace.py: healthy 10 s trace, 10 Hz
                                                        │
                                             faults.py: maybe inject one fault ──▶ FaultLabel (answer key)
                                                        │
                                 ┌──────────────────────┴───────────────────┐
                              rules.py                                   llm.py
                                 └──────────── Diagnosis ───────────────────┘
                                                        │
                                             harness.py: score vs FaultLabel ──▶ results/
```

| File | In | Out |
|---|---|---|
| `dbc.py` | DBC text file | `{name: Message}`; filtered list of measurement signals |
| `trace.py` | signals, seed | healthy rows `{t, msg, sig, val}` |
| `faults.py` | healthy rows, seed | corrupted rows + `FaultLabel` |
| `rules.py` | rows + signal definitions | `Diagnosis(fault_type, signal, onset_t, evidence)` |
| `llm.py` | rows + signal definitions | the same `Diagnosis`, plus tokens, cost and latency |
| `harness.py` | seed range | accuracy, cost and latency per detector; per-case JSONL |

## Running it

Everything below is PowerShell, run from the repo root. The code needs Python 3.10+ and numpy. The LLM call uses the standard library, so there's no SDK to install.

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "src"

python -m unittest discover -s tests -v     # 16 tests, no network needed
python -m triage.harness                    # rules on seeds 0-99  -> results/

# LLM: put your key in .env (it's git-ignored) as ANTHROPIC_API_KEY=sk-ant-...
python -m triage.harness --llm --n 5        # smoke test first: check it works and read the cost per case
python -m triage.harness --llm              # full 100 cases
```

The prompt is about 3k input tokens. Sonnet 5.5 thinks by default and thinking is billed as output, so expect roughly $1–4 for 100 cases; the smoke test gives the real figure. To try another model, set `$env:TRIAGE_MODEL = "claude-haiku-4-5-20251001"`. Seeds 1000 and up are the development set: all rule thresholds were tuned there, and seeds 0–99 were never used for tuning.

## Known limitations

- **Waveforms are synthetic.** Each signal is a sinusoid plus noise, centred in its DBC range. This isn't physically realistic: wheel speed swings around 337 km/h because that's the middle of its DBC range. Realistic operating ranges are deferred to stage 2.
- **One fault per case, on one signal, with no interactions between signals.** Relational faults, where signal A is inconsistent with signal B, are stage 2 work.
- **The rules and the injector were written by the same person**, so the rules know the fault catalogue, and their priors (smooth, noisy signals; linear drift) match the generator. Their thresholds were tuned on separate seeds, but 90% is still an upper bound on how rules would do on real rig data.
- **Signal names must be unique within a case.** Five names (TQI, N, TQFR, TQI_ACOR, PV_AV_CAN) appear in two messages each. The harness keeps the first message for each name, so there are 145 candidate signals, not 150.
- **`dbc.py` skips multiplexed signals.** Eight lines in the Hyundai DBC have a mux marker, plus one `SG_MUL_VAL_` line, and the regex doesn't match them. See `docs/STAGE1_GUIDE.md`.

See `NOTES.md` for the decision log and the stage plan.
