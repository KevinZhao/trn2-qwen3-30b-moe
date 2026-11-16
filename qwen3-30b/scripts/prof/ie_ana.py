# ie_ana.py prof.json : per-layer breakdown of a decode NEFF using all-reduce boundaries.
# Order on sg00: [embed] AR_emb | attn_0 AR | moe_0 AR | attn_1 AR | ... | moe_47 AR | head(lm_head+sampling, allgathers)
import json, sys, statistics as st, collections
j = json.load(open(sys.argv[1]))
tot = j["summary"][0].get("total_time")
cc = [o for o in j["cc_ops"] if o.get("subgraph") == "sg00"]
ar = sorted([o for o in cc if o["operation"] == "ALLREDUCE"], key=lambda o: o["timestamp"])
ins = [i for i in j["instruction"] if i.get("subgraph") == "sg00"]
t0 = min(i["timestamp"] for i in ins); t1 = max(i["timestamp"] + i["duration"] for i in ins)
nL = (len(ar) - 1) // 2
assert len(ar) == 2 * nL + 1, len(ar)
e = lambda o: o["timestamp"] + o["duration"]
embed = ar[0]["timestamp"] - t0
rows = []
for l in range(nL):
    a, m = ar[1 + 2 * l], ar[2 + 2 * l]
    prev = ar[2 * l]
    rows.append((a["timestamp"] - e(prev), a["duration"], m["timestamp"] - e(a), m["duration"]))
head = t1 - e(ar[-1])
names = ["attn", "AR_attn", "moe", "AR_moe"]
print(f"total_time_ms={tot*1e3:.3f} layers={nL} embed+AR_emb_us={(embed+ar[0]['duration'])/1e3:.1f} head(lm_head+sampling+allgather)_us={head/1e3:.1f}")
for k, n in enumerate(names):
    v = [r[k] / 1e3 for r in rows]
    print(f"  {n:8s} median={st.median(v):7.1f}us mean={st.mean(v):7.1f}us min={min(v):7.1f} max={max(v):7.1f} total_ms={sum(v)/1e3:.3f} share={sum(v)*1e3/(t1-t0)*100:5.1f}%")
print(f"  per-layer sum median={st.median(sum(r) for r in rows)/1e3:.1f}us  x{nL} = {sum(sum(r) for r in rows)/1e6:.3f} ms")
