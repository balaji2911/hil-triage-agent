# HiL triage agent

An assistant that reads a failed CAN trace from a hardware-in-the-loop (HiL) rig, identifies the fault and drafts a root-cause explanation. It uses only public data: the DBC comes from [commaai/opendbc](https://github.com/commaai/opendbc), and every trace and fault is synthetic.

Stage 1 asks one question: **on 100 labelled cases, how does one LLM call compare with hand-written rules?**

## Results (stage 1, evaluation seeds 0–99)

| Detector | Exact diagnosis | Fault type right | Cost / case | Median latency |
|---|---|---|---|---|
| Rules | **93%** | 98% | $0 | 15 ms |
| LLM (claude-sonnet-5-5) | *not run yet* | | | |

An exact diagnosis means the fault type, the signal and the onset (within ±0.5 s) are all correct. Clean runs count as correct when the detector says `none`.

The rules get every stuck_at, dropout, out_of_range, spike and clean case right. They lose points only on drift (13/20): two slow drifts that never leave the DBC range go unnoticed, and five are detected but the onset estimate is off by more than 0.5 s. For faults this well defined, rules are the right tool, and the LLM has to justify itself against that bar.

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

python -m unittest discover -s tests -v     # 14 tests, no network needed
python -m triage.harness                    # rules on seeds 0-99  -> results/

# LLM: put your key in .env (it's git-ignored) as ANTHROPIC_API_KEY=sk-ant-...
python -m triage.harness --llm --n 5        # smoke test first, a few cents
python -m triage.harness --llm              # full 100 cases, roughly $1 on Sonnet
```

To try another model, set `$env:TRIAGE_MODEL = "claude-haiku-4-5-20251001"`. Seeds 1000 and up are the development set: all rule thresholds were tuned there, and seeds 0–99 were never used for tuning.

## Known limitations

- **Waveforms are synthetic.** Each signal is a sinusoid plus noise, centred in its DBC range. This isn't physically realistic: wheel speed swings around 337 km/h because that's the middle of its DBC range. Realistic operating ranges are deferred to stage 2.
- **One fault per case, on one signal, with no interactions between signals.** Relational faults, where signal A is inconsistent with signal B, are stage 2 work.
- **The rules and the injector were written by the same person**, so the rules know the fault catalogue. They don't know how the waveforms are generated, and their thresholds were tuned on separate seeds. Even so, 93% is an upper bound on how rules would do on real rig data.
- **`dbc.py` skips multiplexed signals.** Eight lines in the Hyundai DBC have a mux marker, plus one `SG_MUL_VAL_` line, and the regex doesn't match them. See `docs/STAGE1_GUIDE.md`.

See `NOTES.md` for the decision log and the stage plan.
