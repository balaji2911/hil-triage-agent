# Stage 1: top-down study guide

Work through this from top to bottom. Each step has a clear finish line, so it's obvious when to move on.

## Step 1: see the whole pipeline (15 min)

Read the Pipeline section and the file table in `README.md`. Then close them and draw the diagram from memory: the six files, what each one takes in, and what it hands to the next. You're done when you can draw it without looking.

## Step 2: look at real data (20 min)

```powershell
$env:PYTHONPATH = "src"
python -m triage.show 7     # dropout
python -m triage.show 2     # stuck_at
python -m triage.show 0     # spike
python -m triage.show 3     # drift
python -m triage.show 1     # clean run
```

Each command prints the signals picked for that seed, the healthy trace, the same stretch after the fault, the answer key and the rules' verdict. For every case, find the fault in section 3 yourself **before** you read section 4.

## Step 3: file by file, decisions only

For each file, write down the answer to each question in a sentence or two. The line references show where the answer lives in the code. Skip syntax: if an interviewer asks what a regex token means, you can say you'd check.

**`dbc.py`**
1. Why is there a hand-written parser instead of cantools? (docstring)
2. Why is `Signal` frozen but `Message` isn't? (lines 28–51)
3. Why filter out flags, zero-range signals and unitless signals? (`measurement_signals`)
4. Why is this the only file that knows the file format? What would supporting the xlsx view cost?
5. **Parked question:** the parser silently drops multiplexed signals. A reviewer says that's dangerous. What do you say?
   The fact to use: of the 9 `SG_` lines that fail the regex, 8 are multiplexed signals (`M` or `m0`–`m3` markers) and 1 is an `SG_MUL_VAL_` line, which isn't a signal at all.

**`trace.py`**
1. Why a sinusoid plus noise, and not constants or pure noise? (docstring: stuck_at)
2. Why seed everything? What breaks in the harness if you don't?
3. What's wrong with using the DBC range as the operating range? When does that matter?

**`faults.py`**
1. Why is the label fixed at generation time and never re-derived from the trace?
2. Why is the onset placed in the middle 60% of the run?
3. Why are 20% of cases clean?
4. Why does `inject_fault` never mutate its input?

**`rules.py`**
1. Why build a rules baseline before the LLM at all? (`NOTES.md`, Decisions)
2. What fingerprint does each check look for? Why does range run before flat-line? (Hint: an out_of_range fault is also flat.)
3. Why tune on seeds 1000+ and score on 0–99?
4. What is the trend check comparing, in one sentence? (`trend_fit` docstring)
5. Why does the most specific finding win when several signals flag?

**`llm.py`**
1. Why a forced tool call with enums, instead of asking for JSON?
2. Why temperature 0?
3. Why a wide table, and what does a dropout look like in it?
4. Why give the LLM the fault definitions but no thresholds?

**`harness.py`**
1. Why score exact fields, with no LLM judge?
2. Why report type, signal and exact separately, and not just one number?
3. What does ±0.5 s onset tolerance mean in samples, and why that value?
4. Why does one failed API call not stop the run?

## Step 4: the parked "why" questions from earlier

Answer these in the file they belong to: seeding goes with `trace.py`, the middle-60% onset with `faults.py`. The `[None]` base case of the permutation recursion was from Python practice, so answer it there.

## Step 5: run the LLM and read the number

```powershell
python -m triage.harness --llm --n 5     # check the key works, a few cents
python -m triage.harness --llm           # full run, roughly $1
```

Then open `results/llm_predictions.jsonl` and read the explanations on the cases the LLM got wrong. Where the LLM loses to the rules, and where it adds something the rules can't (the explanation text), is the material for the stage 2 README and for the interview.

## Where the rules lose points (seeds 0–99)

| Seed | True onset | Rules said | Why |
|---|---|---|---|
| 24, 92 | 6.2, 4.5 | none | the drift stays inside the DBC range, and the trend fit is below its threshold |
| 19 | 6.0 | drift @ 3.8 | trend found, but the bend was placed too early |
| 51, 61, 71, 73 | 3.4, 4.9, 3.9, 3.6 | drift, onset off by >0.5 s | range crossing caught, but the healthy sinusoid confuses where the ramp starts |
