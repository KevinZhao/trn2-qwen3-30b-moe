#!/usr/bin/env python3
"""P5 packed-prefill numerics check (run after the server is READY).

usage: P4_KNOBS=/tmp/ni2_knobs_<TAG>.json python3 p5_pack_check.py <port> <model_path> [--steplog PATH]

1. pack_max=1  -> POST /reset_prefix_cache -> 16 prompts one at a time (max_concurrency=1)
2. pack_max=16 -> POST /reset_prefix_cache -> the same 16 prompts sent at once (packed)
3. pack_max=16 -> reset -> packed again (packed-vs-packed determinism baseline)
Compares greedy completions (temperature 0, max_tokens 64, ignore_eos) prompt by prompt.
Prompts are sent as token-id lists so lengths are exact; lengths vary, two are equal-length,
the longest is first (the argmax(positions) padding-mask trap), and the sum fits pack_budget 4096.
The knob file is read-modify-written so dsbpc/dsbp/budget keep their values
(p4 resets missing dsbpc/dsbp keys to 0, so they default to 1/1 if the file does not exist yet).
"""
import argparse
import json
import os
import threading
import time
import urllib.request

LENGTHS = [480, 37, 64, 100, 128, 150, 180, 200, 230, 256, 256, 300, 333, 400, 450, 500]
# sum = 4064 <= 4096. 256 appears twice (equal length, different content).

TEXT = (
    "The history of navigation spans thousands of years, from early sailors who read the stars "
    "and the swell of the ocean to modern satellite systems that fix a position within meters. "
    "Polynesian wayfinders memorized rising points of stars and the behaviour of birds and clouds. "
    "Medieval mariners carried the magnetic compass, while the astrolabe and later the sextant "
    "allowed them to measure the altitude of the sun at noon. The longitude problem resisted "
    "solution until reliable marine chronometers made it possible to compare local noon with the "
    "time at a reference meridian. Today receivers combine signals from several constellations, "
    "correct for atmospheric delay, and blend inertial sensors to keep working in tunnels. "
)


def http(port, path, body=None, timeout=600):
    url = f"http://127.0.0.1:{port}{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


def build_prompts(model_path):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        enc = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
        dec = tok.decode
    except Exception as e:  # noqa: BLE001
        print(f"[warn] tokenizer unavailable ({e}); using synthetic token ids")
        tok = None
        enc = None
        dec = None
    prompts = []
    for i, n in enumerate(LENGTHS):
        if enc is not None:
            # distinct first block per prompt (otherwise the scheduler refuses to pack them)
            ids = enc(f"Report {i} (variant {i * 7919 % 1000}). " + TEXT * 8)
            ids = ids[: n - 1] + enc(" Summary:")[:1]
        else:
            ids = [1000 + (i * 131 + j * 17) % 30000 for j in range(n)]
        assert len(ids) == n, (i, len(ids), n)
        prompts.append(ids)
    return prompts, dec


def set_knob(path, pack_max):
    d = {"dsbpc": 1, "dsbp": 1}
    if os.path.exists(path):
        try:
            d = json.load(open(path))
        except ValueError:
            pass
    d["pack_max"] = pack_max
    d["tag"] = f"p5check_pack{pack_max}"
    tmp = path + ".tmp"
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, path)
    time.sleep(1.2)  # _p4_refresh polls at most every 0.5 s and keys on mtime


def complete(port, model, ids, max_tokens):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "return_token_ids": True}
    t0 = time.time()
    r = http(port, "/v1/completions", body)
    c = r["choices"][0]
    toks = c.get("token_ids")
    return {"text": c.get("text", ""), "tokens": toks, "n": r.get("usage", {}).get("completion_tokens"),
            "latency": time.time() - t0}


def run_sequential(port, model, prompts, max_tokens):
    return [complete(port, model, p, max_tokens) for p in prompts]


def run_concurrent(port, model, prompts, max_tokens):
    out = [None] * len(prompts)
    barrier = threading.Barrier(len(prompts))

    def w(i):
        barrier.wait()
        try:
            out[i] = complete(port, model, prompts[i], max_tokens)
        except Exception as e:  # noqa: BLE001
            out[i] = {"error": repr(e)}

    ts = [threading.Thread(target=w, args=(i,)) for i in range(len(prompts))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def compare(a, b, dec):
    same = 0
    rows = []
    for i, (x, y) in enumerate(zip(a, b)):
        if "error" in x or "error" in y:
            rows.append((i, "ERROR", x.get("error") or y.get("error")))
            continue
        tx, ty = x["tokens"], y["tokens"]
        if tx is not None and ty is not None:
            eq = tx == ty
            div = next((k for k, (p, q) in enumerate(zip(tx, ty)) if p != q), None)
        else:  # server ignored return_token_ids: fall back to text
            eq = x["text"] == y["text"]
            div = next((k for k, (p, q) in enumerate(zip(x["text"], y["text"])) if p != q), None)
        same += eq
        rows.append((i, "same" if eq else f"DIFF@{div}", ""))
    return same, rows


def steplog_packs(path, t0, t1):
    """Max requests in one prefill step and the number of multi-request prefill steps in [t0,t1]."""
    if not path or not os.path.exists(path):
        return None
    mx, multi = 0, 0
    for line in open(path):
        f = line.split()
        if len(f) >= 5 and f[0] == "S" and f[2] == "P" and t0 <= float(f[1]) <= t1:
            nr = int(f[4])
            mx = max(mx, nr)
            multi += nr > 1
    return mx, multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("model_path")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--steplog", default=os.environ.get("P4_STEPLOG", ""))
    ap.add_argument("--pack-max", type=int, default=16)
    a = ap.parse_args()
    knobs = os.environ.get("P4_KNOBS")
    assert knobs, "export P4_KNOBS=<knob json path used by the server>"

    model = http(a.port, "/v1/models")["data"][0]["id"]
    prompts, dec = build_prompts(a.model_path)
    print(f"model={model} prompts={len(prompts)} lengths={LENGTHS} total={sum(LENGTHS)}")

    results = {}
    for name, pm, fn in (("serial", 1, run_sequential),
                         ("packed", a.pack_max, run_concurrent),
                         ("packed2", a.pack_max, run_concurrent)):
        set_knob(knobs, pm)
        print(f"reset_prefix_cache: {http(a.port, '/reset_prefix_cache', {})!r}"[:120])
        t0 = time.time()
        results[name] = fn(a.port, model, prompts, a.max_tokens)
        t1 = time.time()
        sl = steplog_packs(a.steplog, t0, t1)
        lat = [r.get("latency", 0) for r in results[name] if "latency" in r]
        print(f"[{name}] pack_max={pm} wall={t1 - t0:.2f}s max_req_latency={max(lat):.2f}s "
              f"steplog(max_reqs_per_prefill_step, multi_req_prefill_steps)={sl}")
    set_knob(knobs, a.pack_max)

    for ref, cur in (("serial", "packed"), ("serial", "packed2"), ("packed", "packed2")):
        same, rows = compare(results[ref], results[cur], dec)
        print(f"\n{ref} vs {cur}: identical {same}/{len(prompts)} = {same / len(prompts):.1%}")
        for i, st, extra in rows:
            if st != "same":
                print(f"  prompt {i} len={LENGTHS[i]}: {st} {extra}")
    s0 = results["serial"][0]
    print("\nsample serial[0]:", repr(s0.get("text", "")[:160]))
    json.dump({k: v for k, v in results.items()}, open(f"/tmp/p5_pack_check_{a.port}.json", "w"))
    print(f"raw results -> /tmp/p5_pack_check_{a.port}.json")


if __name__ == "__main__":
    main()
