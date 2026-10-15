#!/usr/bin/env python3
"""spec_window: the batch sizes where speculative decoding still pays on your vLLM server.

    curl -s localhost:8000/metrics > metrics.txt
    python3 spec_window.py metrics.txt --model-dir <folder with config.json> --gpu h100-sxm

Inputs, all files you already have:
  metrics.txt    one scrape of the server's Prometheus endpoint (vLLM V1 metric names)
  --model-dir    the target model's folder: config.json, plus model.safetensors.index.json if present
  --gpu          the card the server runs on (see --list-gpus), or --tflops and --bandwidth
  --since        an earlier scrape of the same server, to read only the traffic in between

If the server already drafts, the scrape carries the acceptance length and the time of
a speculative step, and spec_window anchors its model on them. If it does not, pass
--accept-len from a short trial run and --method for the drafter you would add.

The model: a decode step reads every weight once plus each sequence's KV cache, and
does two FLOPs per parameter per token. Plain decoding runs one token per sequence;
a speculative step verifies the drafted tokens plus one, runs the drafter, and commits
the acceptance length on average. Speculation pays while
    acceptance length x plain step time  >  speculative step time.
The constants below were fitted on one recorded H200 sweep; README.md lists how far
the model landed from every measured break-even it has been checked against.

Standard library only. Read-only. Exit 0 on a report, 2 on unreadable input.
"""
import argparse
import json
import os
import re
import sys

# name: (dense BF16 TFLOPS, dense FP8 TFLOPS or None, memory bandwidth TB/s), NVIDIA datasheets
GPUS = {
    "a100-40gb": (312.0, None, 1.555),
    "a100-80gb": (312.0, None, 2.039),
    "a100-80gb-pcie": (312.0, None, 1.935),
    "l40s": (362.05, 733.0, 0.864),
    "h100-sxm": (989.5, 1979.0, 3.35),
    "h100-nvl": (835.5, 1670.5, 3.9),
    "h200": (989.5, 1979.0, 4.8),
    "h200-nvl": (835.5, 1670.5, 4.8),
    "b200": (2250.0, 4500.0, 8.0),
}

# Fitted on the capacity-tax H200 sweep (Qwen3-14B, five traces, EAGLE3 and a 0.6B draft).
EFF_MEM = 0.70      # share of peak bandwidth a decode step reaches
EFF_CMP = 0.86      # share of peak dense FLOPs a large step reaches
SMOOTH = 1.1        # how gently a step turns from memory-bound to compute-bound
# Drafter cost a plain scrape cannot show, as a share of a one-sequence plain step,
# measured on the same sweep at concurrency 1 (vLLM 0.27.1, three draft tokens).
PRIOR_OVERHEAD = {"eagle": 0.20, "mtp": 0.20, "draft": 0.88, "ngram": 0.88}
MIN_GAIN = 1.10     # the cutoff keeps a 10 percent margin over break-even
MAX_BATCH = 1024    # vLLM's default --max-num-seqs on 70 GB-class cards

KV_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
            "fp8_inc": 1, "nvfp4": 0.5}
TORCH_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
METHODS = {"eagle": "eagle", "eagle3": "eagle", "mtp": "mtp", "draft": "draft",
           "draft_model": "draft", "ngram": "ngram"}


class SpecWindowError(Exception):
    """Input spec_window cannot use; main() turns it into exit code 2."""


# ---------------------------------------------------------------- reading the scrape

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+\S+)?\s*$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text):
    """{metric name: {sorted label tuple: float}} from the text exposition format."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(4))
        except ValueError:
            continue
        labels = tuple(sorted(LABEL.findall(m.group(3) or "")))
        out.setdefault(m.group(1), {})[labels] = value
    return out


def minus(later, earlier):
    """Counters and histograms of `later` less `earlier`; gauges and info series from `later`."""
    out = {}
    for name, series in later.items():
        cumulative = name.endswith(("_total", "_sum", "_count", "_bucket"))
        before = earlier.get(name, {})
        out[name] = {lab: v - before.get(lab, 0.0) if cumulative else v for lab, v in series.items()}
    return out


class Scrape:
    def __init__(self, metrics):
        self.m = metrics

    def total(self, *names):
        for n in names:
            if n in self.m:
                return sum(self.m[n].values())
        return None

    def hist(self, name):
        """(count, sum) of a histogram, or (None, None) when absent."""
        return self.total(name + "_count"), self.total(name + "_sum")

    def buckets(self, name):
        """[(upper bound, cumulative count)] summed across label sets, sorted by bound."""
        acc = {}
        for labels, v in self.m.get(name + "_bucket", {}).items():
            le = dict(labels).get("le")
            if le is None:
                continue
            bound = float("inf") if le in ("+Inf", "Inf") else float(le)
            acc[bound] = acc.get(bound, 0.0) + v
        return sorted(acc.items())

    def label(self, name, key):
        for labels in self.m.get(name, {}):
            val = dict(labels).get(key)
            if val not in (None, ""):
                return val
        return None

    def positions(self):
        """Distinct draft positions the acceptance counter reports: the draft length."""
        pos = set()
        for labels in self.m.get("vllm:spec_decode_num_accepted_tokens_per_pos_total", {}):
            p = dict(labels).get("position")
            if p is not None:
                pos.add(p)
        return len(pos)


def read_scrape(path, since=None):
    with open(path, encoding="utf-8") as f:
        m = parse_prometheus(f.read())
    if not any(n.startswith("vllm:") for n in m):
        raise SpecWindowError(f"{path} holds no vllm: metrics; is it a vLLM /metrics page?")
    if since:
        with open(since, encoding="utf-8") as f:
            m = minus(m, parse_prometheus(f.read()))
    return Scrape(m)


def traffic(s):
    """What the scrape says about the load: steps, batch, step time, lengths, acceptance."""
    steps = s.total("vllm:iteration_tokens_total_count")
    itl_n, itl_sum = s.hist("vllm:inter_token_latency_seconds")
    if itl_n is None:
        itl_n, itl_sum = s.hist("vllm:time_per_output_token_seconds")
    p_n, p_sum = s.hist("vllm:request_prompt_tokens")
    g_n, g_sum = s.hist("vllm:request_generation_tokens")
    if not steps or not itl_n or not p_n or not g_n:
        raise SpecWindowError("this scrape shows no finished requests; scrape a server that has served traffic")
    generated = s.total("vllm:generation_tokens_total", "vllm:generation_tokens") or g_sum
    computed = s.hist("vllm:request_prefill_kv_computed_tokens")[1]
    if computed is None:
        prompt = s.total("vllm:prompt_tokens_total") or p_sum
        computed = prompt - (s.total("vllm:prompt_tokens_cached_total") or 0.0)
    t = {
        "steps": steps,
        "batch": itl_n / steps,                  # decoding sequences per engine step
        "step_s": itl_sum / itl_n,               # one latency per sequence per step
        "context": p_sum / p_n + 0.5 * g_sum / g_n,
        "prefill_per_token": computed / generated if generated else 0.0,
        "model": s.label("vllm:num_requests_running", "model_name")
        or s.label("vllm:iteration_tokens_total_count", "model_name"),
        "cache_dtype": s.label("vllm:cache_config_info", "cache_dtype") or "auto",
        "buckets": s.buckets("vllm:iteration_tokens_total"),
        "drafts": s.total("vllm:spec_decode_num_drafts_total"),
        "accepted": s.total("vllm:spec_decode_num_accepted_tokens_total"),
        "draft_tokens": s.total("vllm:spec_decode_num_draft_tokens_total"),
        "positions": s.positions(),
        "committed": (generated - g_n) / itl_n,   # tokens a sequence commits per step; first tokens excluded
    }
    t["speculating"] = bool(t["drafts"])
    return t


# ---------------------------------------------------------------- the networks

def load_config(model_dir):
    """config.json plus, when present, the checkpoint's exact weight bytes."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        raise SpecWindowError(f"no config.json in {model_dir}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "text_config" in cfg and "num_hidden_layers" not in cfg:
        cfg = dict(cfg["text_config"], quantization_config=cfg.get("quantization_config"))
    total = None
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            total = (json.load(f).get("metadata") or {}).get("total_size")
    return cfg, total


def weight_bits(cfg):
    q = cfg.get("quantization_config") or {}
    method = str(q.get("quant_method", "")).lower()
    if not method:
        return 8 * TORCH_BYTES.get(str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"), 2)
    if method in ("fp8", "fbgemm_fp8", "modelopt") or "fp8" in str(q.get("fmt", "")):
        return 8
    if q.get("bits") or q.get("w_bit"):
        return int(q.get("bits") or q.get("w_bit"))
    for group in (q.get("config_groups") or {}).values():
        w = (group or {}).get("weights") or {}
        if w.get("num_bits"):
            return int(w["num_bits"])
    return 16


def is_moe(cfg):
    return bool(cfg.get("num_experts") or cfg.get("num_local_experts") or cfg.get("n_routed_experts"))


def layer_params(cfg):
    h, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads") or heads
    hd = cfg.get("head_dim") or h // heads
    return h * heads * hd + 2 * h * kv * hd + heads * hd * h + 3 * h * cfg["intermediate_size"]


def kv_per_position(cfg, cache_dtype, layers=None):
    stored = cache_dtype
    if cache_dtype in (None, "", "auto"):
        stored = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16")
    per_value = KV_BYTES.get(stored, TORCH_BYTES.get(stored, 2))
    n = layers if layers is not None else cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads") or heads
    hd = cfg.get("head_dim") or cfg["hidden_size"] // heads
    return 2 * n * kv * hd * per_value


class Net:
    """One network's cost per forward pass: bytes it reads, parameters each token multiplies."""

    def __init__(self, name, weight_bytes, params, kv_bytes, fp8):
        self.name, self.weight_bytes, self.params = name, weight_bytes, params
        self.kv_bytes, self.fp8 = kv_bytes, fp8


def target_net(cfg, total_bytes, cache_dtype, name="target"):
    if is_moe(cfg):
        raise SpecWindowError("this config is a mixture of experts; spec_window models dense models only "
                              "(a MoE step reads a batch-dependent share of its experts)")
    bits = weight_bits(cfg)
    h, vocab = cfg["hidden_size"], cfg["vocab_size"]
    params = cfg["num_hidden_layers"] * layer_params(cfg) + vocab * h   # the LM head multiplies too
    if total_bytes:
        embed = 0 if cfg.get("tie_word_embeddings") else vocab * h * bits / 8
        weight_bytes = total_bytes - embed    # an embedding lookup reads a few rows, not the table
    else:
        weight_bytes = params * bits / 8
    return Net(name, weight_bytes, params, kv_per_position(cfg, cache_dtype), bits == 8)


def head_net(cfg, cache_dtype, method):
    """An EAGLE or MTP head sized from its target: one decoder layer, an input projection, a head."""
    h, vocab = cfg["hidden_size"], cfg["vocab_size"]
    head_vocab = min(vocab, 32000) if method == "eagle" else vocab
    params = layer_params(cfg) + 3 * h * h + head_vocab * h
    return Net(method + " head", params * 2, params, kv_per_position(cfg, cache_dtype, layers=1), False)


class Card:
    def __init__(self, name, bf16_tflops, fp8_tflops, tb_s, count=1):
        self.name, self.count = name, count
        self.bf16, self.fp8, self.bw = bf16_tflops * 1e12 * count, (fp8_tflops or 0) * 1e12 * count, tb_s * 1e12 * count

    def flops(self, net):
        return self.fp8 if net.fp8 and self.fp8 else self.bf16


def pass_seconds(net, card, tokens, batch, context, kv_reads=1.0):
    """One forward pass: the slower of reading bytes and doing FLOPs, blended smoothly."""
    mem = (net.weight_bytes + kv_reads * batch * context * net.kv_bytes) / (EFF_MEM * card.bw)
    cmp_ = tokens * 2 * net.params / (EFF_CMP * card.flops(net))
    return (mem ** SMOOTH + cmp_ ** SMOOTH) ** (1 / SMOOTH)


# ---------------------------------------------------------------- the window

class Window:
    """Plain and speculative step times as the batch grows, anchored on the scrape."""

    def __init__(self, target, drafter, card, t, tau, k):
        self.target, self.drafter, self.card = target, drafter, card
        self.ctx, self.prefill, self.tau, self.k = t["context"], t["prefill_per_token"], tau, k
        self.scale, self.overhead, self.per_seq = 1.0, 0.0, 0.0

    def plain(self, b):
        return self.scale * pass_seconds(self.target, self.card, b * (1 + self.prefill), b, self.ctx)

    def speculative(self, b):
        k, tokens = self.k, b * (self.k + 1) + b * self.tau * self.prefill
        # verification attends every drafted position over the whole context
        t = pass_seconds(self.target, self.card, tokens, b, self.ctx, kv_reads=k + 1)
        if self.drafter is not None:   # one pass over the committed tokens, then k - 1 single steps
            t += pass_seconds(self.drafter, self.card, tokens, b, self.ctx)
            t += (k - 1) * pass_seconds(self.drafter, self.card, b, b, self.ctx)
        return self.scale * t + self.overhead + self.per_seq * b

    def gain(self, b):
        return self.tau * self.plain(b) / self.speculative(b)

    def break_even(self):
        """Batch where the gain crosses 1.0, interpolated; 0 if it never clears it."""
        prev = self.gain(1)
        if prev < 1:
            return 0.0
        for b in range(2, MAX_BATCH + 1):
            g = self.gain(b)
            if g < 1:
                return (b - 1) + (prev - 1) / (prev - g)
            prev = g
        return float(MAX_BATCH)

    def cutoff(self):
        """Largest batch that still clears MIN_GAIN."""
        last = 0
        for b in range(1, MAX_BATCH + 1):
            if self.gain(b) >= MIN_GAIN:
                last = b
            else:
                break
        return last


def share_at_or_below(buckets, tokens):
    """Share of engine steps that carried at most `tokens`, interpolated inside a bucket."""
    if not buckets or buckets[-1][1] <= 0:
        return None
    total, lo_bound, lo_count = buckets[-1][1], 0.0, 0.0
    for bound, count in buckets:
        if tokens <= bound:
            if bound == float("inf"):
                return lo_count / total
            frac = (tokens - lo_bound) / (bound - lo_bound) if bound > lo_bound else 1.0
            return (lo_count + frac * (count - lo_count)) / total
        lo_bound, lo_count = bound, count
    return 1.0


def analyse(t, target, drafter, card, method, accept_len=None, draft_len=None):
    if t["speculating"]:
        tau = t["committed"]    # measured: n-gram skips drafting on steps with no match
        k = t["positions"] or round(t["draft_tokens"] / t["drafts"])
        w = Window(target, drafter, card, t, tau, k)
        # what the model misses at the scraped batch: a fixed drafter cost first,
        # the remainder as a cost per sequence, so the curve passes through the scrape
        residual = t["step_s"] - w.speculative(t["batch"])
        if t["batch"] < 2:   # one sequence cannot tell a fixed cost from a per-sequence one
            w.overhead = max(residual, 0.0)
        else:
            w.overhead = min(max(residual, 0.0), PRIOR_OVERHEAD[method] * w.plain(1))
            w.per_seq = max(0.0, residual - w.overhead) / t["batch"]
        anchor = "speculative"
    else:
        if not accept_len:
            raise SpecWindowError("this scrape has no speculative decoding counters; pass --accept-len "
                                  "(tokens a step commits, from a short trial run or the drafter's card)")
        tau, k = accept_len, draft_len or 3
        w = Window(target, drafter, card, t, tau, k)
        w.scale = t["step_s"] / w.plain(t["batch"])
        w.overhead = PRIOR_OVERHEAD[method] * w.plain(1)
        anchor = "plain"
    be, cut = w.break_even(), w.cutoff()
    per_seq = tau if t["speculating"] else 1.0
    share = share_at_or_below(t["buckets"], be * per_seq) if be > 0 else 0.0
    if share is None:
        verdict = "no step histogram in this scrape; compare the window with your busy-hour batch"
    elif share >= 0.8:
        verdict = "keep speculation on: most steps run inside the window"
    elif share <= 0.2 or cut == 0:
        verdict = "turn speculation off: continuous batching already fills this GPU past break-even"
    else:
        verdict = f"gate it by load: draft up to batch {cut}, stop above it"
    # The anchor pairs the mean batch with the mean step time, which only describes one load
    # level. Steps far below the mean batch mean the scrape spans several.
    low = share_at_or_below(t["buckets"], per_seq * t["batch"] / 4) if t["batch"] >= 4 else 0.0
    mixed = low is not None and low > 0.25
    return {"t": t, "w": w, "tau": tau, "k": k, "anchor": anchor, "break_even": be, "cutoff": cut,
            "share": share, "verdict": verdict, "method": method, "mixed": low if mixed else None}


# ---------------------------------------------------------------- the report

def config_line(method, k, cut):
    name = {"eagle": "eagle3", "mtp": "mtp", "draft": "draft_model", "ngram": "ngram"}[method]
    sched = [[1, cut, k], [cut + 1, MAX_BATCH, 0]]
    return json.dumps({"method": name, "num_speculative_tokens": k,
                       "num_speculative_tokens_per_batch_size": sched})


DRAFTER_NAMES = {"eagle": "an EAGLE head", "mtp": "an MTP head", "draft": "a draft model", "ngram": "n-gram lookup"}


def render(r, card_name):
    t, w = r["t"], r["w"]
    model = t["model"] or w.target.name
    lines = [f"spec_window · {model} · {card_name} · {DRAFTER_NAMES[r['method']]} proposing {r['k']}, "
             f"{r['tau']:.2f} tokens committed a step"]
    if r["anchor"] == "speculative":
        lines.append(f"now        batch {t['batch']:.1f} at {t['step_s'] * 1e3:.1f} ms a speculative step, "
                     f"{t['steps']:,.0f} steps in this scrape")
    else:
        lines.append(f"now        batch {t['batch']:.1f} at {t['step_s'] * 1e3:.1f} ms a plain step, "
                     f"{t['steps']:,.0f} steps; the drafter's cost is a prior, not a measurement")
    now = max(1, round(t["batch"]))
    probes = sorted({1, now, 8, 32, 128})
    lines.append("gain       " + " · ".join(
        f"{w.gain(b):.2f}x at {'batch ' if b == 1 else ''}{b}{' (now)' if b == now else ''}" for b in probes))
    be = r["break_even"]
    if be <= 0:
        lines.append("window     none: a speculative step loses even at batch 1")
    elif be >= MAX_BATCH:
        lines.append(f"window     speculation pays at every batch up to {MAX_BATCH}")
    else:
        lines.append(f"window     speculation pays from batch 1 to {int(be)}, break-even at {be:.1f}")
    if r["share"] is not None:
        lines.append(f"steps      {100 * r['share']:.0f}% of this scrape's engine steps ran inside that window")
    if r["cutoff"] > 0 and be < MAX_BATCH:
        lines.append(f"cutoff     draft up to batch {r['cutoff']} (gain {MIN_GAIN:.2f}x or better), none above:")
        lines.append(f"           --speculative-config '{config_line(r['method'], r['k'], r['cutoff'])}'")
    lines.append(f"verdict    {r['verdict']}")
    if r["mixed"] is not None:
        lines.append(f"note       {100 * r['mixed']:.0f}% of steps ran below a quarter of the mean batch, so this scrape "
                     "spans several load levels; for a sharper window, scrape twice a few minutes apart "
                     "at steady load and pass the first with --since")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scrape", nargs="?", help="a saved curl of the server's /metrics")
    ap.add_argument("--since", help="an earlier scrape of the same server; read only the traffic in between")
    ap.add_argument("--model-dir", help="target model folder with config.json")
    ap.add_argument("--gpu", help="card name, see --list-gpus")
    ap.add_argument("--gpus", type=int, default=1, help="tensor-parallel GPUs serving one replica")
    ap.add_argument("--tflops", type=float, help="dense BF16 TFLOPS, for a card not in the list")
    ap.add_argument("--bandwidth", type=float, help="memory bandwidth in TB/s, for a card not in the list")
    ap.add_argument("--method", default="eagle3", choices=sorted(METHODS), help="the drafter (default eagle3)")
    ap.add_argument("--draft-dir", help="draft model folder with config.json (for --method draft)")
    ap.add_argument("--accept-len", type=float, help="tokens a speculative step commits, for a plain scrape")
    ap.add_argument("--draft-len", type=int, help="draft tokens a step proposes, for a plain scrape (default 3)")
    ap.add_argument("--list-gpus", action="store_true")
    a = ap.parse_args(argv)
    if a.list_gpus:
        for n, (bf, f8, bw) in GPUS.items():
            fp8 = f"{f8:8.1f} FP8" if f8 else "  no FP8   "
            print(f"{n:15s} {bf:8.1f} BF16 TFLOPS  {fp8}  {bw:5.3f} TB/s")
        return 0
    try:
        if not a.scrape or not a.model_dir:
            raise SpecWindowError("pass a scrape and --model-dir")
        if a.gpu:
            if a.gpu not in GPUS:
                raise SpecWindowError(f"unknown --gpu {a.gpu}; see --list-gpus, or pass --tflops and --bandwidth")
            card = Card(a.gpu, *GPUS[a.gpu], count=a.gpus)
            card_name = a.gpu if a.gpus == 1 else f"{a.gpus}x {a.gpu}"
        elif a.tflops and a.bandwidth:
            card = Card("custom", a.tflops, None, a.bandwidth, count=a.gpus)
            card_name = f"{a.tflops:g} TFLOPS, {a.bandwidth:g} TB/s"
        else:
            raise SpecWindowError("pass --gpu, or --tflops and --bandwidth")
        method = METHODS[a.method]
        t = traffic(read_scrape(a.scrape, a.since))
        cfg, total = load_config(a.model_dir)
        target = target_net(cfg, total, t["cache_dtype"], name=os.path.basename(os.path.normpath(a.model_dir)))
        if method == "draft":
            if not a.draft_dir:
                raise SpecWindowError("--method draft needs --draft-dir, the draft model's folder")
            dcfg, dtotal = load_config(a.draft_dir)
            drafter = target_net(dcfg, dtotal, t["cache_dtype"], name="draft")
        elif method == "ngram":
            drafter = None
        elif a.draft_dir:
            dcfg, dtotal = load_config(a.draft_dir)
            drafter = target_net(dcfg, dtotal, t["cache_dtype"], name="head")
        else:
            drafter = head_net(cfg, t["cache_dtype"], method)
        r = analyse(t, target, drafter, card, method, a.accept_len, a.draft_len)
    except (SpecWindowError, OSError, KeyError, ValueError, ZeroDivisionError) as e:
        print(f"spec_window: {e}", file=sys.stderr)
        return 2
    print(render(r, card_name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
