# nm_util.py prof.json : per (subgraph,label) union-of-intervals busy time (excluding semaphore waits), + iteration stats
import json, sys, collections
j = json.load(open(sys.argv[1])); ins = j["instruction"]
grp = collections.defaultdict(list)
for i in ins:
    if "SEMAPHORE" in str(i.get("opcode")) or i.get("opcode") in ("DRAIN",): continue
    grp[(i.get("fully_qualified_subgraph","").lstrip("/"), i.get("label"))].append((i["timestamp"], i["timestamp"] + i["duration"]))
t0 = min(i["timestamp"] for i in ins); t1 = max(i["timestamp"] + i["duration"] for i in ins)
for k in sorted(grp, key=str):
    iv = sorted(grp[k]); u = 0; cs, ce = iv[0]
    for a, b in iv[1:]:
        if a > ce: u += ce - cs; cs, ce = a, b
        else: ce = max(ce, b)
    u += ce - cs
    print(f"PROF union {k}: {u/1e3:.0f}us of {(t1-t0)/1e3:.0f}us ({u/(t1-t0)*100:.0f}%) n={len(iv)}")
br = sorted(i["timestamp"] for i in ins if i.get("opcode") == "COMPARE_BRANCH" and i.get("label") == "Tensor" and i.get("fully_qualified_subgraph","").lstrip("/") == "sg00")
d = [(b - a) / 1e3 for a, b in zip(br, br[1:])]
if d:
    import statistics as st
    print(f"PROF Tensor-branch gaps n={len(d)} med={st.median(d):.1f}us p10={sorted(d)[len(d)//10]:.1f} p90={sorted(d)[9*len(d)//10]:.1f}")
