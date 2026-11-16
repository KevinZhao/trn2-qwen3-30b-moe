import json, re, random, urllib.request, concurrent.futures as cf, sys
PORT=sys.argv[1]; OUT=sys.argv[2]; B=f"http://127.0.0.1:{PORT}"
P = json.load(open("/data/cache/rs_results/RS7/soak/charrun_rerun2.json"))[0]["prompt"]
print("prompt runs:", [(m.group(0)[:1], len(m.group(0))) for m in re.finditer(r"(.)\1{40,}", P, re.S)], "max dash run", max((len(m.group(0)) for m in re.finditer(r"-+", P)), default=0))
ds = json.load(open("/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"))
SG = [c[0]["value"] for c in (d.get("conversations") or [] for d in ds) if len(c) >= 2 and c[0].get("from") == "human" and 20 <= len(c[0]["value"]) <= 12000]
def chat(p, mt=512):
    b = {"model": "qwen3-30b-a3b", "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0.0}
    r = urllib.request.Request(B + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(r, timeout=600)); return j["choices"][0]["message"].get("content") or ""
res = []
for rep in range(3):
    fill = random.Random(rep).sample(SG, 31)
    with cf.ThreadPoolExecutor(32) as ex: outs = list(ex.map(chat, [P] + fill))
    t = outs[0]; runs = [(m.start(), m.group(0)[:1], len(m.group(0))) for m in re.finditer(r"(.)\1{40,}", t, re.S)]
    res.append({"rep": rep, "text": t, "runs": runs}); print("== c32 rep", rep, "runs", runs)
    for (s, ch, n) in runs: print("   ctx:", repr(t[max(0, s-200): s+n+100]))
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1); print("P1C_DONE")
