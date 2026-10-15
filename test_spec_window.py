#!/usr/bin/env python3
"""Replays recorded vLLM servers and published measurements against spec_window's model.

Three kinds of known answer, all in fixtures/ (fixtures/README.md says where each comes from):
  1. twenty recorded /metrics scrapes of one H200 serving Qwen3-14B, plain and with three
     drafters, with the throughput each run measured;
  2. the same H200's request-level ladder: five traces, eight concurrency levels, measured
     time per output token and acceptance for every drafter;
  3. the EAGLE-3 paper's vLLM batch table, a different GPU, model and engine version.
Run: python3 test_spec_window.py
"""
import csv
import io
import json
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

import spec_window as sw

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures")
SWEEP = os.path.join(FX, "h200-sweep")
LADDER_CS = (1, 5, 10, 25, 35, 50, 75, 100)
SWEEP_CS = (1, 5, 25, 50, 100)
ARM_NAMES = {"EAGLE3": "eagle3", "separate-draft": "draft-0.6B", "n-gram": "ngram", "baseline": "baseline"}
METHOD = {"eagle3": "eagle", "draft-0.6B": "draft", "ngram": "ngram"}
# prompt and output tokens per request in the ladder's traces (fixtures/README.md)
TRACES = {"sharegpt": (192, 256), "swebench": (349, 256), "cnn_short": (744, 64),
          "cnn_long": (745, 256), "edgar_json": (1720, 320)}

CARD = sw.Card("h200", *sw.GPUS["h200"])
CFG, TOTAL = sw.load_config(os.path.join(FX, "Qwen3-14B"))
DCFG, DTOTAL = sw.load_config(os.path.join(FX, "Qwen3-0.6B"))


def drafter(arm, cache_dtype="auto"):
    if arm == "eagle3":
        return sw.head_net(CFG, cache_dtype, "eagle")
    if arm == "draft-0.6B":
        return sw.target_net(DCFG, DTOTAL, cache_dtype, name="draft")
    return None


def measured_break_even(points):
    """Concurrency where a measured gain first drops below 1.0, interpolated."""
    (c0, g0) = points[0]
    if g0 < 1:
        return 0.0
    for c1, g1 in points[1:]:
        if g1 < 1:
            return c0 + (g0 - 1) / (g0 - g1) * (c1 - c0)
        c0, g0 = c1, g1
    return None


def sweep_measured():
    out = {}
    with open(os.path.join(SWEEP, "measured.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(ARM_NAMES[r["arm"]], int(r["concurrency"]))] = float(r["speedup_vs_baseline"])
    return out


def sweep_report(arm, c):
    """spec_window's analysis of one sweep run: that run's scrape less the one before it."""
    path = os.path.join(SWEEP, f"{arm}-2K-sharegpt-c{c}.metrics.txt")
    i = SWEEP_CS.index(c)
    since = os.path.join(SWEEP, f"{arm}-2K-sharegpt-c{SWEEP_CS[i - 1]}.metrics.txt") if i else None
    t = sw.traffic(sw.read_scrape(path, since))
    target = sw.target_net(CFG, TOTAL, t["cache_dtype"])
    return t, sw.analyse(t, target, drafter(arm, t["cache_dtype"]), CARD, METHOD.get(arm, "eagle"),
                         accept_len=None if arm != "baseline" else 1.0)


def recorded_sweep(fails, rows):
    """Each speculative run's scrape, read by the tool, against that run's measured gain."""
    meas = sweep_measured()
    for arm in ("eagle3", "draft-0.6B", "ngram"):
        be_meas = measured_break_even([(c, meas[(arm, c)]) for c in SWEEP_CS])
        for c in SWEEP_CS:
            t, r = sweep_report(arm, c)
            g = r["w"].gain(t["batch"])
            m = meas[(arm, c)]
            rows.append(f"  {arm:10s} c{c:<3d} gain now {g:.2f} measured {m:.2f}   "
                        f"break-even {r['break_even']:5.1f} measured {be_meas:5.1f}")
            if abs(g - m) > 0.20:
                fails.append(f"{arm} c{c}: gain at the scraped batch {g:.2f} vs measured {m:.2f}")
            if arm != "ngram" and abs(r["break_even"] - be_meas) > 10:
                fails.append(f"{arm} c{c}: break-even {r['break_even']:.1f} vs measured {be_meas:.1f}")
    # the plain model on the plain server's own scrapes
    for c in SWEEP_CS:
        t, r = sweep_report("baseline", c)
        model = r["w"].plain(t["batch"]) / r["w"].scale
        err = model / t["step_s"] - 1
        rows.append(f"  baseline   c{c:<3d} plain step model {model * 1e3:5.1f} ms measured {t['step_s'] * 1e3:5.1f} ms")
        if abs(err) > 0.10:
            fails.append(f"baseline c{c}: plain step off by {100 * err:.0f}%")


def ladder():
    with open(os.path.join(FX, "h200-ladder.jsonl"), encoding="utf-8") as f:
        return {(r["trace"], r["arm"], r["concurrency"]): r for r in map(json.loads, f)}


def recorded_ladder(fails, rows):
    """Anchor at one concurrency, the way a scrape would, and predict where the gain crosses 1.0."""
    d = ladder()
    target = sw.target_net(CFG, TOTAL, "auto")
    worst = 0.0
    for arm in ("eagle3", "draft-0.6B"):
        for trace, (prompt, output) in TRACES.items():
            be_meas = measured_break_even([(c, d[(trace, arm, c)]["output_tok_s"] /
                                            d[(trace, "baseline", c)]["output_tok_s"]) for c in LADDER_CS])
            preds = []
            for anchor in (5, 10, 25, 35, 50):
                row = d[(trace, arm, anchor)]
                t = {"context": prompt + output / 2, "prefill_per_token": prompt / output}
                w = sw.Window(target, drafter(arm), CARD, t, row["accept_len"], 3)
                residual = row["mean_tpot_ms"] / 1e3 * row["accept_len"] - w.speculative(anchor)
                w.overhead = min(max(residual, 0.0), sw.PRIOR_OVERHEAD[METHOD[arm]] * w.plain(1))
                w.per_seq = max(0.0, residual - w.overhead) / anchor
                be = w.break_even()
                preds.append(be)
                worst = max(worst, abs(be - be_meas))
                if abs(be - be_meas) > 12:
                    fails.append(f"ladder {arm} {trace} anchored at {anchor}: {be:.1f} vs measured {be_meas:.1f}")
            rows.append(f"  {arm:10s} {trace:10s} measured {be_meas:5.1f}   anchored at 5/10/25/35/50: "
                        + " ".join(f"{p:5.1f}" for p in preds))
    rows.append(f"  largest miss {worst:.1f} batch slots")


def published_table(fails, rows):
    """EAGLE-3 paper, Table 5: vLLM, LLaMA-Instruct 3.1 8B, MT-Bench, chain length 2, no scrape.

    The caption names an A100 and the text an RTX 3090; this replays the A100 reading.
    The model is given only the gain at batch 2 and must produce the decay after it."""
    llama = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
             "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 14336,
             "vocab_size": 128256, "tie_word_embeddings": False, "torch_dtype": "bfloat16"}
    batches = (2, 4, 8, 16, 24, 32, 48, 56)
    table = {"EAGLE-3": (1.75, 1.68, 1.58, 1.49, 1.42, 1.36, 1.21, 1.01),
             "EAGLE": (1.30, 1.25, 1.21, 1.10, 1.03, 0.93, 0.82, 0.71)}
    card = sw.Card("a100-80gb", *sw.GPUS["a100-80gb"])
    for name, gains in table.items():
        w = sw.Window(sw.target_net(llama, None, "auto"), sw.head_net(llama, "auto", "eagle"), card,
                      {"context": 600, "prefill_per_token": 0.3}, 1.0, 2)
        w.overhead = sw.PRIOR_OVERHEAD["eagle"] * w.plain(1)
        lo, hi = 1.0, 4.0
        for _ in range(60):
            w.tau = (lo + hi) / 2
            lo, hi = (w.tau, hi) if w.gain(2) < gains[0] else (lo, w.tau)
        pred = [w.gain(b) for b in batches]
        rows.append(f"  {name:8s} paper " + " ".join(f"{g:.2f}" for g in gains))
        rows.append(f"  {'':8s} model " + " ".join(f"{g:.2f}" for g in pred))
        if any(b > a for a, b in zip(pred, pred[1:])):
            fails.append(f"{name}: modelled gain does not fall with batch")
        if max(abs(p - g) for p, g in zip(pred, gains)) > 0.25:
            fails.append(f"{name}: modelled gains drift more than 0.25 from the table")


def unit_checks():
    fails = []
    target = sw.target_net(CFG, TOTAL, "auto")
    # Qwen3-14B by hand: 40 x (5120*5120*2 + 2*5120*1024 + 3*5120*17408) + 151936*5120
    if round(target.params) != 13_989_969_920:
        fails.append(f"Qwen3-14B multiplying parameters {target.params:,.0f}")
    if target.kv_bytes != 2 * 40 * 8 * 128 * 2:
        fails.append(f"Qwen3-14B KV bytes per position {target.kv_bytes}")
    if sw.target_net(CFG, TOTAL, "fp8").kv_bytes != 81_920:
        fails.append("an fp8 KV cache should halve bytes per position")
    # a later scrape less an earlier one keeps gauges and subtracts counters
    a = sw.parse_prometheus('x_total 10\ng 3\nh_bucket{le="8.0"} 4\nh_bucket{le="+Inf"} 9\n')
    b = sw.parse_prometheus('x_total 4\ng 7\nh_bucket{le="8.0"} 1\nh_bucket{le="+Inf"} 2\n')
    s = sw.Scrape(sw.minus(a, b))
    if s.total("x_total") != 6 or s.total("g") != 3 or s.buckets("h") != [(8.0, 3.0), (float("inf"), 7.0)]:
        fails.append("minus() on counters, gauges or buckets")
    # share of steps: 40 of 100 steps at <= 8 tokens, 60 at <= 16; 12 tokens sits halfway
    share = sw.share_at_or_below([(8.0, 40.0), (16.0, 60.0), (float("inf"), 100.0)], 12)
    if abs(share - 0.5) > 1e-9:
        fails.append(f"bucket interpolation gave {share}")
    # n-gram drafts on about a third of steps: acceptance per draft overstates tokens per step
    t = sw.traffic(sw.read_scrape(os.path.join(SWEEP, "ngram-2K-sharegpt-c1.metrics.txt")))
    per_draft = 1 + t["accepted"] / t["drafts"]
    if not (per_draft > 3.0 and 1.5 < t["committed"] < 1.7):
        fails.append(f"n-gram committed {t['committed']:.2f} vs per draft {per_draft:.2f}")
    # the command line: a plain scrape needs --accept-len, a MoE config is refused, a fixture reports
    base = [os.path.join(SWEEP, "baseline-2K-sharegpt-c25.metrics.txt"), "--model-dir",
            os.path.join(FX, "Qwen3-14B"), "--gpu", "h200"]
    err = io.StringIO()
    with redirect_stderr(err), redirect_stdout(io.StringIO()):
        code = sw.main(base)
    if code != 2 or "--accept-len" not in err.getvalue():
        fails.append("a plain scrape without --accept-len should exit 2 and name the flag")
    moe = dict(CFG, num_experts=128)
    try:
        sw.target_net(moe, None, "auto")
        fails.append("a mixture-of-experts config should be refused")
    except sw.SpecWindowError:
        pass
    out = io.StringIO()
    with redirect_stdout(out):
        code = sw.main([os.path.join(SWEEP, "draft-0.6B-2K-sharegpt-c25.metrics.txt"), "--since",
                        os.path.join(SWEEP, "draft-0.6B-2K-sharegpt-c5.metrics.txt"), "--model-dir",
                        os.path.join(FX, "Qwen3-14B"), "--gpu", "h200", "--method", "draft",
                        "--draft-dir", os.path.join(FX, "Qwen3-0.6B")])
    text = out.getvalue()
    if code != 0 or "window     speculation pays from batch 1 to 11, break-even at 11.8" not in text:
        fails.append("the README's recorded example no longer prints its window line")
    if any(tok in text for tok in ("TODO", "XXX", "None", "nan", "inf")):
        fails.append("placeholder or non-number text in the report")
    return fails


GROUPS = [
    ("recorded H200 sweep: each run's scrape against that run's measured gain", recorded_sweep),
    ("H200 request ladder: break-even anchored at one concurrency, against the measured one", recorded_ladder),
    ("EAGLE-3 paper, Table 5 (no scrape, prior drafter cost)", published_table),
]


def main():
    failed, passed = [], 0
    for title, group in GROUPS:
        rows, fails = [], []
        group(fails, rows)
        print(title)
        print("\n".join(rows) + "\n")
        failed += fails
        passed += not fails
    units = unit_checks()
    for f in failed + units:
        print("FAIL", f)
    print(f"{passed}/{len(GROUPS)} recorded case groups pass")
    print(f"{9 - len(units)}/9 unit checks pass")
    return 1 if failed or units else 0


if __name__ == "__main__":
    sys.exit(main())
