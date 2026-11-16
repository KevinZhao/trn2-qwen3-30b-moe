# rerun the 2 RS7-soak char-run prompts (found by L=len//4 + keywords), greedy 512, save full text; c1 x2 and inside a c16 batch
import json, re, random, urllib.request, concurrent.futures as cf, sys
PORT = sys.argv[1]; OUT = sys.argv[2]; B = f"http://127.0.0.1:{PORT}"
ds = json.load(open("/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"))
SG = [c[0]["value"] for c in (d.get("conversations") or [] for d in ds) if len(c) >= 2 and c[0].get("from") == "human" and 20 <= len(c[0]["value"]) <= 12000]
c1 = sorted({p for p in SG if len(p) // 4 == 17 and re.search(r"brute", p, re.I) and re.search(r"dictio", p, re.I)})
c2 = sorted({p for p in SG if len(p) // 4 == 778 and re.search(r"css", p, re.I)})
print("cand1", len(c1), [p[:100] for p in c1]); print("cand2", len(c2), [p[:100] for p in c2[:5]])
def chat(p, mt=512):
    b = {"model": "qwen3-30b-a3b", "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0.0}
    r = urllib.request.Request(B + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(r, timeout=600)); return j["choices"][0]["message"].get("content") or "", j["choices"][0].get("finish_reason")
def runs(t):
    return [(m.start(), m.group(0)[:1], len(m.group(0))) for m in re.finditer(r"(.)\1{40,}", t, re.S)]
res = []
targets = c1 + c2
for rep in range(2):
    for p in targets:
        t, fr = chat(p); res.append({"mode": f"c1_r{rep}", "prompt": p, "text": t, "fr": fr, "runs": runs(t)})
rng = random.Random(5); fill = [p for p in rng.sample(SG, 40) if p not in targets][:16 - len(targets)]
with cf.ThreadPoolExecutor(16) as ex:
    outs = list(ex.map(chat, targets + fill))
for p, (t, fr) in zip(targets, outs[:len(targets)]):
    res.append({"mode": "c16", "prompt": p, "text": t, "fr": fr, "runs": runs(t)})
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
for r in res:
    print("==", r["mode"], "fr", r["fr"], "runs", r["runs"], "| prompt:", r["prompt"][:60].replace("\n", " "))
    for (s, ch, n) in r["runs"]:
        print("   ctx:", repr(r["text"][max(0, s - 150): s + n + 80]))
print("P1_DONE")
