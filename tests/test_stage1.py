"""
Stage 1 tests. Standard-library unittest, so they run with no extra install:

    python -m unittest discover -s tests -v

(pytest also picks them up if you have it.)
"""

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triage import harness, llm, rules                      # noqa: E402
from triage.dbc import measurement_signals, parse_dbc       # noqa: E402
from triage.faults import FAULT_TYPES, FaultLabel, inject_fault, make_case   # noqa: E402
from triage.trace import TraceConfig, generate_trace        # noqa: E402

MESSAGES = parse_dbc(harness.DBC_PATH)


class TestDbc(unittest.TestCase):
    def test_counts_on_hyundai_file(self):
        self.assertEqual(len(MESSAGES), 113)
        self.assertEqual(sum(len(m.signals) for m in MESSAGES.values()), 1146)
        self.assertEqual(len(measurement_signals(MESSAGES)), 150)

    def test_measurement_filter(self):
        for _, s in measurement_signals(MESSAGES):
            self.assertFalse(s.is_flag)
            self.assertGreater(s.maximum, s.minimum)
            self.assertTrue(s.unit)


class TestTraceAndFaults(unittest.TestCase):
    def setUp(self):
        self.case = harness.build_case(MESSAGES, 7)

    def test_case_is_deterministic(self):
        again = harness.build_case(MESSAGES, 7)
        self.assertEqual(self.case.rows, again.rows)
        self.assertEqual(self.case.label, again.label)

    def test_inject_never_mutates_input(self):
        selected = [(m, s) for m, s in self.case.signals.values()]
        healthy = generate_trace(selected, TraceConfig(seed=7))
        snapshot = [dict(r) for r in healthy]
        for ft in FAULT_TYPES:
            inject_fault(healthy, selected[0], ft, seed=7)
        self.assertEqual(healthy, snapshot)

    def test_label_mix_over_eval_seeds(self):
        # The distribution recorded in the handoff notes; guards against
        # accidental changes to make_case's random draws.
        mix = Counter(harness.build_case(MESSAGES, s).label.fault_type for s in range(100))
        self.assertEqual(mix, Counter(spike=16, none=21, stuck_at=16, drift=20, out_of_range=12, dropout=15))

    def test_clean_case_is_unchanged(self):
        selected = [(m, s) for m, s in self.case.signals.values()]
        healthy = generate_trace(selected, TraceConfig(seed=0))
        seed = next(s for s in range(100) if make_case(healthy, selected, s)[1].fault_type == "none")
        rows, label = make_case(healthy, selected, seed)
        self.assertEqual(rows, healthy)
        self.assertEqual(label.fault_type, "none")


class TestRules(unittest.TestCase):
    """Each fault type, injected on purpose, is found by the matching check."""

    def setUp(self):
        case = harness.build_case(MESSAGES, 11)
        self.signals = case.signals
        self.selected = [(m, s) for m, s in case.signals.values()]
        self.healthy = generate_trace(self.selected, TraceConfig(seed=11))

    def test_healthy_trace_is_clean(self):
        self.assertEqual(rules.diagnose(self.healthy, self.signals).fault_type, "none")

    def test_each_step_fault_found_with_correct_signal_and_onset(self):
        for ft in ("stuck_at", "dropout", "out_of_range", "spike"):
            with self.subTest(fault=ft):
                rows, label = inject_fault(self.healthy, self.selected[2], ft, seed=5)
                d = rules.diagnose(rows, self.signals)
                self.assertTrue(harness.score(d, label).exact, f"{ft}: got {d}")


class TestScoring(unittest.TestCase):
    label = FaultLabel("spike", "SIG_A", "MSG", 4.0, 0.0)

    def test_exact(self):
        s = harness.score(rules.Diagnosis("spike", "SIG_A", 4.3), self.label)
        self.assertTrue(s.exact)

    def test_onset_outside_tolerance(self):
        s = harness.score(rules.Diagnosis("spike", "SIG_A", 4.6), self.label)
        self.assertTrue(s.signal_ok)
        self.assertFalse(s.exact)

    def test_wrong_signal(self):
        s = harness.score(rules.Diagnosis("spike", "SIG_B", 4.0), self.label)
        self.assertTrue(s.type_ok)
        self.assertFalse(s.signal_ok)

    def test_none_needs_only_type(self):
        s = harness.score(rules.NO_FAULT, FaultLabel("none", "", "", 0.0, 0.0))
        self.assertTrue(s.exact)


class TestLlmWithoutNetwork(unittest.TestCase):
    """The LLM path, with the API replaced by a fake. No key, no cost."""

    def setUp(self):
        self.case = harness.build_case(MESSAGES, 3)
        self.sent = None

    def fake_post(self, body):
        self.sent = body
        lab = self.case.label
        return {
            "content": [{"type": "tool_use", "name": "report_diagnosis",
                         "input": {"fault_type": lab.fault_type, "signal": lab.signal,
                                   "onset_t": lab.onset_t, "explanation": "test"}}],
            "usage": {"input_tokens": 2500, "output_tokens": 150},
        }

    def test_request_shape_and_parsing(self):
        d, meta = llm.diagnose_llm(self.case.rows, self.case.signals,
                                   model="claude-sonnet-5-5", post=self.fake_post)
        self.assertEqual(self.sent["tool_choice"], {"type": "tool", "name": "report_diagnosis"})
        self.assertEqual(self.sent["temperature"], 0)
        enum = self.sent["tools"][0]["input_schema"]["properties"]["signal"]["enum"]
        self.assertEqual(set(enum), {""} | set(self.case.signals))
        self.assertTrue(harness.score(d, self.case.label).exact)
        self.assertAlmostEqual(meta["cost_usd"], (2500 * 2 + 150 * 10) / 1e6)

    def test_dropout_shows_as_empty_cells(self):
        selected = [(m, s) for m, s in self.case.signals.values()]
        healthy = generate_trace(selected, TraceConfig(seed=3))
        rows, label = inject_fault(healthy, selected[0], "dropout", seed=3)
        table = llm.format_trace(rows, self.case.signals).splitlines()
        onset_line = next(line for line in table[1:] if float(line.split(",")[0]) == label.onset_t)
        self.assertEqual(onset_line.split(",")[1], "")


if __name__ == "__main__":
    unittest.main()
