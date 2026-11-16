# nm_gaps.py prof.json sg : PE idle-gap histogram and what other engines do during big PE gaps
import json, sys, collections, statistics as st
j = json.load(open(sys.argv[1])); sg = sys.argv[2]
f = lambda i: i.get("fully_qualified_subgraph", "").lstrip("/")
ins = [i for i in j["instruction"] if f(i) == sg]
tm = sorted([i for i in ins if i.get("label") == "TensorMatrix"], key=lambda i: i["timestamp"])
t0 = tm[0]["timestamp"]; t1 = tm[-1]["timestamp"] + tm[-1]["duration"]
gaps = []
for a, b in zip(tm, tm[1:]):
    g = b["timestamp"] - a["timestamp"]
    gaps.append((g, a, b))
tot = sum(g for g, _, _ in gaps)
big = [x for x in gaps if x[0] > 1000]
print(f"PROF PE span {(t1-t0)/1e3:.0f}us n={len(tm)} ; start-to-start sum {tot/1e3:.0f}us ; gaps>1us: n={len(big)} sum={sum(g for g,_,_ in big)/1e3:.0f}us ; >0.4us sum={sum(g for g,_,_ in gaps if g>400)/1e3:.0f}us")
src = collections.Counter(); srcn = collections.Counter()
for g, a, b in big:
    k = (str(a.get("nki_source_location")).split(":")[-1], str(b.get("nki_source_location")).split(":")[-1]); src[k] += g; srcn[k] += 1
for k, v in src.most_common(8): print(f"PROF big-gap after line {k[0]} -> before line {k[1]}: n={srcn[k]} sum {v/1e3:.0f}us")
# during big gaps: what is ACT/DVE/Pool/Sync doing (union busy, non-wait)
for lab in ("Scalar", "Vector", "GpSimd", "Sync"):
    ev = [i for i in ins if i.get("label") == lab and "SEMAPHORE" not in str(i.get("opcode"))]
    busy = 0
    for g, a, b in big:
        s0, s1 = a["timestamp"], b["timestamp"]
        busy += sum(max(0, min(i["timestamp"] + i["duration"], s1) - max(i["timestamp"], s0)) for i in ev if i["timestamp"] < s1 and i["timestamp"] + i["duration"] > s0)
    print(f"PROF during big PE gaps {lab} busy {busy/1e3:.0f}us")
dma = [d for d in j.get("dma", []) if d.get("fully_qualified_subgraph", d.get("subgraph", "")).lstrip("/") == sg]
print("PROF dma bytes", sum(d.get("transfer_size", 0) for d in dma) / 1e6, "MB")
