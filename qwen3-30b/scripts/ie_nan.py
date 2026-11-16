# NaN regression test: identical concurrent prompts + n=4 parallel sampling. usage: ie_nan.py PORT
import sys, json, time, re, urllib.request, urllib.error, concurrent.futures as cf
PORT = sys.argv[1]
URL = f"http://127.0.0.1:{PORT}/v1"
def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=900).read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:
        return None, repr(e)[:300]
def chat(q, n=1, temp=0.0, mt=160):
    d, err = post("/chat/completions", {"model": "qwen3-30b-a3b", "messages": [{"role": "user", "content": q}], "max_tokens": mt,
                                        "temperature": temp, "n": n, "chat_template_kwargs": {"enable_thinking": False}})
    if err: return None, err
    return [(c["message"]["content"] or "", c.get("finish_reason")) for c in d["choices"]], None
def sane(t):
    if not t or not t.strip(): return "empty"
    if re.search(r"(.)\1{15,}", t): return "char-run"
    if "�" in t: return "replacement-char"
    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t)) < 0.3: return "few-letters"
    w = t.split()
    if len(w) >= 20 and len(set(w)) / len(w) < 0.15: return "repetitive"
    return None
Q = ["What is the capital of France? Answer in one sentence, then name two famous landmarks there.",
     "Write a Python function that returns the n-th Fibonacci number iteratively.",
     "Explain in three sentences why the sky is blue.",
     "List the first ten prime numbers, separated by commas, then explain what a prime is."]
fails, report = [], []
def check(tag, outs):
    for i, (t, fr) in enumerate(outs):
        bad = sane(t)
        if bad: fails.append(f"{tag}[{i}] {bad}: {t[:120]!r}")
    uniq = len(set(t for t, _ in outs))
    report.append(f"{tag}: {len(outs)} outputs, {uniq} unique, lens={[len(t) for t, _ in outs]}, finish={sorted(set(fr or '?' for _, fr in outs))}")
    print(f"=== {tag} sample[0] ({len(outs[0][0])} chars)\n{outs[0][0][:400]}\n", flush=True)
t0 = time.time()
# A: 16 identical concurrent prompts (decode bucket 16, all rows identical -> union 8)
for n in (16, 4, 3, 2):
    with cf.ThreadPoolExecutor(n) as ex: rs = list(ex.map(lambda q: chat(q), [Q[0]] * n))
    errs = [e for _, e in rs if e]
    if errs: fails.append(f"dup{n} errors: {errs[:2]}"); continue
    check(f"dup{n}", [r[0][0] for r in rs])
# B: n=4 parallel sampling (greedy first; fallback to temperature>0 if greedy n>1 is rejected)
def psamp(tag, prompts, n=4):
    for temp in (0.0, 0.7):
        with cf.ThreadPoolExecutor(len(prompts)) as ex: rs = list(ex.map(lambda q: chat(q, n=n, temp=temp), prompts))
        errs = [e for _, e in rs if e]
        if errs:
            report.append(f"{tag} temp={temp}: rejected: {errs[0]}"); continue
        for j, (outs, _) in enumerate(rs):
            if len(outs) != n: fails.append(f"{tag} temp={temp} req{j}: got {len(outs)} choices")
            check(f"{tag}_t{temp}_req{j}", outs)
        return
    fails.append(f"{tag}: all temperatures rejected")
psamp("n4x1", [Q[1]])
psamp("n4x4", Q)          # 16 sequences: 4 groups of 4 identical rows
psamp("n4x4same", [Q[2]] * 4)  # 16 sequences of the same prompt
# C: mixed 8 identical + 8 distinct-ish
mix = [Q[3]] * 8 + [f"In one paragraph, describe the number {k} and one interesting fact about it." for k in range(8)]
with cf.ThreadPoolExecutor(16) as ex: rs = list(ex.map(lambda q: chat(q), mix))
errs = [e for _, e in rs if e]
if errs: fails.append(f"mix16 errors: {errs[:2]}")
else: check("mix16_dup8", [r[0][0] for r in rs[:8]]); check("mix16_distinct8", [r[0][0] for r in rs[8:]])
print("\n".join(report))
print("FAILS:", len(fails)); print("\n".join(fails[:40]))
print("RESULT", "PASS" if not fails else "FAIL", f"{time.time()-t0:.1f}s")
