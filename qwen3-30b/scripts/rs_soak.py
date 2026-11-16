# nj_soak.py PORT OUT_JSONL DUR_S SEG SEED : one soak segment of mixed load (ShareGPT chat + random long prompts),
# rotating concurrency c1/c4/c16 sub-phases, greedy canaries every 45 s, per-request timeout (hang detection).
import sys, json, time, random, re, threading, urllib.request, concurrent.futures as cf
PORT, OUT, DUR, SEG, SEED = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
B = f"http://127.0.0.1:{PORT}"; MODEL = "qwen3-30b-a3b"; TMO = 600
rng0 = random.Random(SEED * 7919 + SEG)
ds = json.load(open("/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"))
SG = [c[0]["value"] for c in (d.get("conversations") or [] for d in ds) if len(c) >= 2 and c[0].get("from") == "human" and 20 <= len(c[0]["value"]) <= 12000]
lock = threading.Lock(); fo = open(OUT, "a", buffering=1)
def rec(d):
    with lock: fo.write(json.dumps(d) + "\n")
def stream(path, body):
    body = dict(body, stream=True, stream_options={"include_usage": True}); t0 = time.perf_counter(); first = None; parts = []; ntok = None; fr = None
    r = urllib.request.Request(B + path, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=TMO) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): ntok = d["usage"].get("completion_tokens")
            for c in d.get("choices") or []:
                t = c.get("text") if "text" in c else (c.get("delta") or {}).get("content")
                fr = c.get("finish_reason") or fr
                if t:
                    if first is None: first = time.perf_counter()
                    parts.append(t)
    t1 = time.perf_counter()
    return {"ttft": None if first is None else (first - t0) * 1e3, "e2e": (t1 - t0) * 1e3, "text": "".join(parts), "ntok": ntok, "fr": fr}
def sane(t):
    if not t or not t.strip(): return "empty"
    if re.search(r"(.)\1{40,}", t): return "char-run"
    if "�" in t: return "replacement-char"
    return None
def one(kind, conc, rng):
    t = time.time()
    try:
        if kind == "sg":
            p = rng.choice(SG); mt = rng.choice([128, 256, 512])
            r = stream("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0.0})
            L = len(p) // 4
        else:
            L = rng.choice([256, 1024, 2048, 3000, 4096, 4500, 5000, 6000, 6500, 7000, 7000, 7000]); mt = rng.choice([16, 64, 256])
            r = stream("/v1/completions", {"model": MODEL, "prompt": [rng.randrange(1000, 150000) for _ in range(L)], "max_tokens": mt,
                                           "temperature": 0.0, "ignore_eos": True})
        bad = sane(r["text"]) if kind == "sg" else (None if r["ntok"] == mt else f"ntok {r['ntok']}!={mt}")
        rec({"t": t, "seg": SEG, "kind": kind, "conc": conc, "L": L, "mt": mt, "ttft": r["ttft"], "e2e": r["e2e"], "ntok": r["ntok"], "fr": r["fr"],
             "ok": r["ttft"] is not None, "bad": bad, "snip": r["text"][:80] if bad else None})
    except Exception as e:
        rec({"t": t, "seg": SEG, "kind": kind, "conc": conc, "ok": False, "err": repr(e)[:300], "dt": time.time() - t})
CAN = [("What is 17 multiplied by 23? Reply with only the number.", "391"), ("What is the capital of France? Reply with one word.", "Paris"),
       ("Spell the word 'banana' backwards. Reply with only the result.", "ananab")]
stop = threading.Event()
def canary():
    i = 0
    while not stop.wait(45):
        q, a = CAN[i % len(CAN)]; i += 1; t = time.time()
        try:
            r = stream("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": q}], "max_tokens": 24, "temperature": 0.0,
                                                "chat_template_kwargs": {"enable_thinking": False}})
            rec({"t": t, "seg": SEG, "kind": "canary", "ok": True, "pass": a.lower() in r["text"].lower(), "text": r["text"][:60], "e2e": r["e2e"]})
        except Exception as e:
            rec({"t": t, "seg": SEG, "kind": "canary", "ok": False, "pass": False, "err": repr(e)[:300]})
# sub-phases: (conc, p_random, p_long-only)  rotating; each ~DUR/6
SUB = [(1, 0.5), (4, 0.5), (16, 0.35), (32, 0.4), (4, 0.8), (32, 0.6)] if __import__("os").environ.get("SOAK32") == "1" else [(1, 0.5), (4, 0.5), (16, 0.35), (16, 0.7), (4, 0.8), (16, 0.5)]
th = threading.Thread(target=canary, daemon=True); th.start()
t_end = time.time() + DUR; sub_d = DUR / len(SUB)
for si, (conc, prn) in enumerate(SUB):
    s_end = min(t_end, time.time() + sub_d)
    def worker(w):
        rng = random.Random(rng0.random() + w)
        while time.time() < s_end: one("rn" if rng.random() < prn else "sg", conc, rng)
    with cf.ThreadPoolExecutor(conc) as ex: list(ex.map(worker, range(conc)))
stop.set(); th.join(timeout=120); print("SEG_DONE", SEG)
