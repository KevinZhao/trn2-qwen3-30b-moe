import json, re, urllib.request, sys
PORT=sys.argv[1]; OUT=sys.argv[2]; B=f"http://127.0.0.1:{PORT}"
ds = json.load(open("/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"))
SG = [c[0]["value"] for c in (d.get("conversations") or [] for d in ds) if len(c) >= 2 and c[0].get("from") == "human" and 20 <= len(c[0]["value"]) <= 12000]
c = sorted({p for p in SG if len(p)//4 == 778 and re.search(r"\d+px|[{};]\s*\n\s*[a-z-]+\s*:", p)})
print("n778", len({p for p in SG if len(p)//4 == 778}), "cssish", len(c))
def chat(p, mt=512):
    b = {"model": "qwen3-30b-a3b", "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0.0}
    r = urllib.request.Request(B + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(r, timeout=600)); return j["choices"][0]["message"].get("content") or "", j["choices"][0].get("finish_reason")
res = []
for p in c:
    t, fr = chat(p); runs = [(m.start(), m.group(0)[:1], len(m.group(0))) for m in re.finditer(r"(.)\1{40,}", t, re.S)]
    res.append({"prompt": p, "text": t, "fr": fr, "runs": runs})
    print("== head", repr(t[:70]), "runs", runs, "| prompt:", repr(p[:70]))
    for (s, ch, n) in runs: print("   ctx:", repr(t[max(0, s-200): s+n+100]))
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1); print("P1B_DONE")
