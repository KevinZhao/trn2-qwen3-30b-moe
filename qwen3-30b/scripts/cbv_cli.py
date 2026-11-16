# cbv verifier client (independent of the cb_* scripts). usage: cbv_cli.py CMD PORT ...
import sys, json, time, random, re, statistics as st, urllib.request, concurrent.futures as cf
CMD, PORT = sys.argv[1], sys.argv[2]; A = sys.argv[3:]
B = f"http://127.0.0.1:{PORT}"; MODEL = "qwen3-30b-a3b"

def post(path, body, timeout=1200):
    r = urllib.request.Request(B + path, json.dumps(body).encode() if body is not None else b"",
                               {"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(r, timeout=timeout)

def reset():
    return post("/reset_prefix_cache", None).status

def metrics():
    t = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode(); q = h = 0.0
    for l in t.splitlines():
        if l.startswith("vllm:prefix_cache_queries_total"): q += float(l.split()[-1])
        elif l.startswith("vllm:prefix_cache_hits_total"): h += float(l.split()[-1])
    return q, h

def stream(path, body):
    """returns dict(ttft ms, itl list ms, e2e ms, text, ntok, chunks)"""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    t0 = time.perf_counter(); ts = []; parts = []; ntok = None
    with post(path, body) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): ntok = d["usage"].get("completion_tokens")
            for c in d.get("choices") or []:
                t = c.get("text") if "text" in c else (c.get("delta") or {}).get("content")
                if t:
                    ts.append(time.perf_counter()); parts.append(t)
    t1 = time.perf_counter()
    if not ts: return {"ttft": None, "itl": [], "e2e": (t1 - t0) * 1e3, "text": "", "ntok": ntok, "chunks": 0}
    return {"ttft": (ts[0] - t0) * 1e3, "itl": [(b - a) * 1e3 for a, b in zip(ts, ts[1:])], "e2e": (t1 - t0) * 1e3,
            "text": "".join(parts), "ntok": ntok, "chunks": len(ts)}

def chat_body(content, mt, **kw):
    return dict({"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": mt, "temperature": 0.0,
                 "chat_template_kwargs": {"enable_thinking": False}}, **kw)

def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p / 100; f = int(k); c = min(f + 1, len(v) - 1); return v[f] + (v[c] - v[f]) * (k - f)

def summ(rs, wall):
    tt = [r["ttft"] for r in rs if r["ttft"] is not None]
    tp = [(r["e2e"] - r["ttft"]) / (r["ntok"] - 1) for r in rs if r["ttft"] is not None and (r["ntok"] or 0) > 1]
    itl = [x for r in rs for x in r["itl"]]; nt = sum(r["ntok"] or 0 for r in rs)
    multi = sum(1 for r in rs if r["ntok"] and r["chunks"] and r["ntok"] > r["chunks"] + 1)
    return {"n": len(rs), "fail": len(rs) - len(tt), "ttft_mean": st.mean(tt), "ttft_med": st.median(tt), "ttft_p99": pct(tt, 99),
            "tpot_mean": st.mean(tp), "itl_med": st.median(itl) if itl else None, "itl_max": max(itl) if itl else None,
            "out_tok_s": nt / wall, "out_tok": nt, "wall_s": wall, "req_multi_tok_chunks": multi}

def run_load(bodies, conc, path):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(conc) as ex: rs = list(ex.map(lambda b: stream(path, b), bodies))
    return rs, time.perf_counter() - t0

if CMD == "rst":
    print("reset", reset())

elif CMD == "sg":   # own ShareGPT loader: sg PORT SEED CONC N OUT
    seed, conc, n, out = int(A[0]), int(A[1]), int(A[2]), A[3]
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("/data/models/Qwen3-30B-A3B")
    ds = json.load(open("/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"))
    rng = random.Random(seed * 1000 + 17); idx = list(range(len(ds))); rng.shuffle(idx); ps = []
    for i in idx:
        c = ds[i].get("conversations") or []
        if len(c) < 2 or c[0].get("from") != "human": continue
        L = len(tk(c[0]["value"]).input_ids)
        if 8 <= L <= 1024: ps.append((c[0]["value"], L))
        if len(ps) == n: break
    reset()
    rs, wall = run_load([dict(chat_body(p, 512), chat_template_kwargs={}) for p, _ in ps], conc, "/v1/chat/completions")
    s = summ(rs, wall); s["in_tok"] = sum(L for _, L in ps)
    json.dump({"summary": s, "req": [{k: r[k] for k in ("ttft", "e2e", "ntok", "chunks")} | {"itl_max": max(r["itl"] or [0])} for r in rs]}, open(out, "w"))
    print("SG", seed, conc, n, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}))

elif CMD == "ttft":  # ttft PORT OUT LENS REPS
    out, lens, reps = A[0], [int(x) for x in A[1].split(",")], int(A[2])
    rng = random.Random(int(time.time()) ^ 0xC0FFEE)
    def one(n):
        return stream("/v1/completions", {"model": MODEL, "prompt": [rng.randrange(1000, 150000) for _ in range(n)],
                                          "max_tokens": 2, "temperature": 0.0, "ignore_eos": True})["ttft"]
    reset(); [one(n) for n in (64, 3000, 7000)]
    res = {}
    for n in lens:
        v = [one(n) for _ in range(reps)]; res[n] = v
        print(f"TTFT len={n:5d} med={st.median(v):7.1f} all={[round(x, 1) for x in v]}", flush=True)
    json.dump(res, open(out, "w"))

elif CMD == "pfx":   # retrieval-based prefix-cache test: pfx PORT OUT
    out = A[0]; rng = random.Random(4242)
    names = [f"{a}-{b:03d}" for a in ("Alpha", "Bravo", "Cedar", "Delta", "Ember", "Fjord", "Garnet", "Harbor") for b in range(1, 61)]
    rng.shuffle(names); codes = {nm: str(rng.randrange(100000, 999999)) for nm in names}
    lines = [f"Entry {i+1}: the access code for station {nm} is {codes[nm]}." for i, nm in enumerate(names)]
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("/data/models/Qwen3-30B-A3B")
    head = "The following is a station registry. Each entry lists a station and its six-digit access code.\n\n"
    def doc_upto(L):
        d = head; k = 0
        while k < len(lines) and len(tk(d + lines[k] + "\n").input_ids) <= L: d += lines[k] + "\n"; k += 1
        return d, k
    def q(nm): return f"\nQuestion: What is the access code for station {nm}? Reply with only the six-digit code."
    def ask(d, nm, mt=12):
        r = stream("/v1/chat/completions", chat_body(d + q(nm), mt)); r["ok"] = codes[nm] in r["text"]; r["nm"] = nm; return r
    def ask_m(d, nm):
        q0, h0 = metrics(); r = ask(d, nm); q1, h1 = metrics(); r["hit"] = h1 - h0; r["qry"] = q1 - q0; return r
    res = []; agg = {"cold_ok": 0, "cold_n": 0, "hit_ok": 0, "hit_n": 0, "same": 0, "same_n": 0, "det": 0, "det_n": 0}
    for L in (1000, 2100, 3000, 4100, 6100, 7900):
        d, k = doc_upto(L); keys = [names[0], names[k // 3], names[(2 * k) // 3], names[k - 1]]
        reset(); c0 = ask_m(d, keys[0])               # cold
        h_same = ask_m(d, keys[0])                    # hit, same prompt
        h_diff = [ask_m(d, nm) for nm in keys[1:]]    # hits, different question
        reset(); c0b = ask_m(d, keys[0])              # cold determinism
        colds = []
        for nm in keys[1:]: reset(); colds.append(ask_m(d, nm))
        reset(); ask(d, names[1])                     # warm the prefix
        with cf.ThreadPoolExecutor(4) as ex: hc = list(ex.map(lambda nm: ask(d, nm), keys))  # 4 concurrent hits
        for r in [c0, c0b] + colds: agg["cold_ok"] += r["ok"]; agg["cold_n"] += 1
        for r in [h_same] + h_diff + hc: agg["hit_ok"] += r["ok"]; agg["hit_n"] += 1
        pairs = [(c0, h_same)] + list(zip(colds, h_diff)) + list(zip([c0] + colds, hc))
        for a, b in pairs: agg["same"] += a["text"] == b["text"]; agg["same_n"] += 1
        agg["det"] += c0["text"] == c0b["text"]; agg["det_n"] += 1
        rec = {"L": L, "ntok": len(tk(d).input_ids), "entries": k, "keys": keys,
               "cold": [(r["nm"], r["text"], r["ok"], round(r["ttft"], 1), r["hit"], r["qry"]) for r in [c0, c0b] + colds],
               "hit": [(r["nm"], r["text"], r["ok"], round(r["ttft"], 1), r.get("hit"), r.get("qry")) for r in [h_same] + h_diff + hc]}
        res.append(rec)
        print(f"PFXL L={L} tok={rec['ntok']} entries={k} cold_ttft={c0['ttft']:.0f} hit_ttft={h_same['ttft']:.0f}/{st.mean(r['ttft'] for r in h_diff):.0f} "
              f"hit_tok={h_same['hit']:.0f}/{h_same['qry']:.0f} cold_ok={[r['ok'] for r in [c0, c0b] + colds]} hit_ok={[r['ok'] for r in [h_same] + h_diff + hc]} "
              f"cold0={c0['text']!r} hit0={h_same['text']!r}", flush=True)
    json.dump({"agg": agg, "res": res}, open(out, "w"), indent=1)
    print("PFX", json.dumps(agg))

elif CMD == "gsm":   # gsm PORT OUT N
    out, n = A[0], int(A[1])
    qs = [json.loads(l) for l in open("/data/datasets/gsm8k/test.jsonl")][:n]; reset()
    def solve(it):
        b = chat_body("Solve this grade-school math problem. Reason briefly, then give the final answer on its own line as '#### <number>'.\n\n"
                      + it["question"], 1024)
        txt = json.load(post("/v1/chat/completions", b))["choices"][0]["message"]["content"] or ""
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", txt) or re.findall(r"-?\d[\d,]*\.?\d*", txt)
        pred = m[-1].replace(",", "").rstrip(".") if m else ""
        try: ok = abs(float(pred) - float(gold)) < 1e-6
        except Exception: ok = False
        return {"ok": ok, "pred": pred, "gold": gold, "out": txt}
    t0 = time.time(); rs = [solve(it) for it in qs]
    json.dump(rs, open(out, "w"))
    print("GSM", json.dumps({"n": len(rs), "acc": sum(r["ok"] for r in rs) / len(rs), "wall_s": round(time.time() - t0)}))
