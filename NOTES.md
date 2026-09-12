# Project log — hil-triage-agent

## 2026-09-12 — Kickoff

### What I did
- Installed Python (python.org, "Add to PATH") and Git; verified with `python --version`, `git --version`
- Created `Documents\projects\hil-triage-agent`, `git init`, scaffolded `data/gen`, `baseline`, `evals`, empty `README.md`; first commit "scaffold"
- Created venv (`python -m venv .venv`, activate with `.venv\Scripts\Activate.ps1`), installed `cantools numpy pandas`, froze `requirements.txt`
- Cloned `commaai/opendbc` with `--depth 1` into `Documents\projects` (outside the repo)
- Copied `hyundai_2015_ccan.dbc` into `data/gen/`
- Installed VS Code, opened the repo with `code .`

### Things I learned
- `--depth 1` clones only the latest snapshot, no history — smaller and faster; can't view old commits in that clone
- `code .` opens the current directory; running it inside `opendbc` would have opened that folder instead — always check the terminal path
- If venv activation fails on PowerShell: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Editor choice doesn't matter; VS Code chosen for low friction only

### Concepts (three-question version, no mechanics yet)
- **RAG** — solves: the model doesn't know my private/changing docs and they don't all fit in a prompt. Use when the answer lives in text I own and needs citing. Don't use when the text fits in one prompt, the task needs computation not lookup, or the sources are unreliable. Correction to my first take: RAG does not learn over time; every query is a fresh lookup of a few relevant snippets.
- **Agent** — solves: tasks where the next step depends on the result of the last one. Use when the inspection path isn't fixed. Don't use when the steps are known (write a pipeline) or a wrong action is expensive. Stops when the model returns a final answer instead of a tool call; back it up with a step limit and an output checker.
- **Harness** — a fixed case set with known answers + a runner + metrics. Not a single metric, and it doesn't auto-tweak anything: I tweak, it measures. Evaluates retrieval too. Score structured fields exactly; use an LLM judge only for free text, and check the judge against my own labels.
- **Embeddings** — search by meaning instead of keywords. Use for natural-language docs. Don't use alone for exact identifiers (signal names, REQ IDs, arbitration IDs) — combine with keyword search.
- The model never learns from the project; everything it needs is in the prompt every call. Tokens ≈ 4 chars, paid per token. Temperature 0 for anything measured; rerun evals before trusting small differences.

### Decisions
- Decision: project = HiL test-failure triage agent (option 1), minimum version first, then layer up. Not re-brainstorming.
- Decision: rule-based checker is the baseline before any LLM. For plain range/stuck/dropout faults rules are the best tool; the LLM must justify itself against them.
- Decision: LLM stack earns its place only on (a) explaining root cause in words, (b) requirements I didn't hand-code, (c) ambiguous cases where the inspection path isn't fixed. Report each layer's marginal gain over the previous.
- Decision: no JLR data, code or requirements. Public DBC from opendbc + synthetic logs + mock requirements only.
- Decision: papers (CRAG, Wang et al. RAG best practices, Zheng et al. LLM-as-judge) are pulled in only when a weak number in the harness needs them — not reproduced up front.
- Decision: ground truth is decided at generation time from fault parameters, never by re-checking the trace with my own checker.
- Decision: start applying for jobs after stage 2 (single LLM call + number in README), not after stage 6.

### Plan
| Stage | Weeks | Deliverable | README number |
|---|---|---|---|
| 1. Data + baseline | 1–2 | Fault injector, 100 cases with truth, rule checker, scorer | Rule-checker accuracy |
| 2. Single LLM call | 3 | Log excerpt + requirement → structured verdict | LLM vs rules accuracy, cost/case |
| 3. Retrieval | 4–5 | Requirements corpus, hybrid search | Recall@k, end-to-end accuracy |
| 4. Agent | 6–8 | Tool loop for ambiguous subset | Agent vs pipeline on hard cases |
| 5. Harness hardening | 9–10 | LLM judge, judge-vs-human agreement, CI gate | Judge agreement, p95 latency |
| 6. Deploy + write-up | 11–12 | FastAPI + Docker on AWS, tracing, ablations, failure analysis | Live demo |

### Fault injector design (next)
- Pick 4–6 signals from `hyundai_2015_ccan.dbc` (wheel speed, RPM, accel pedal, brake flag) — note message, cycle time, min/max, units
- Write 10 mock requirements in `requirements.md`: range, rate/timing, relational (2–3 relational)
- Fault types v1: `stuck`, `out_of_range`, `dropout`; v2: `late_message`, `relational_violation`
- Case = folder `cases/case_NNNN/` with `trace.csv` (timestamp_s, arbitration_id, data_hex) and `truth.json` (case_id, fault_type, signal, start_s, end_s, violated_requirements, seed)
- 20% clean cases (`fault_type: "none"`)
- `data/gen/inject.py --n 100 --seed 0`, reproducible
- Done-for-day-1 = `--n 5` produces five folders and `cantools decode` shows the fault where truth.json says it is

### Habit notes
- Shipping rule before learning rule: no study until the week's artefact is in the repo
- "What didn't work this week" is a README section from month one
- Study a concept only via the three questions until the project forces the "how"
- Send the data scientist the repo link by a named date

### Next session
- Send chosen signals → write the ten requirements → write `inject.py`
