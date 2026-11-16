import json, sys, collections
j = json.load(open(sys.argv[1])); L = int(sys.argv[2]); NB = 14
ins = j["instruction"]; e = lambda o: o["timestamp"] + o["duration"]
cc = sorted([o for o in j["cc_ops"] if o.get("subgraph") == "sg00" and o["operation"] == "ALLREDUCE"], key=lambda o: o["timestamp"])
a, b = e(cc[2 * L]), cc[1 + 2 * L]["timestamp"]
W = [i for i in ins if a <= i["timestamp"] < b and "self_attn" in str(i.get("hlo_name", "")) or (a <= i["timestamp"] < b and i.get("subgraph") != "sg00")]
K = [i for i in ins if a <= i["timestamp"] < b]
span = (b - a) / NB
for q in range(NB):
    lo, hi = a + q * span, a + (q + 1) * span
    act = collections.defaultdict(float); top = collections.defaultdict(float)
    for i in K:
        s0, s1 = max(lo, i["timestamp"]), min(hi, e(i))
        if s1 > s0: act[i["label"]] += s1 - s0; top[(i["label"][:6], i.get("opcode"))] += s1 - s0
    t3 = sorted(top.items(), key=lambda x: -x[1])[:4]
    print(f"{(lo-a)/1e3:5.1f}-{(hi-a)/1e3:5.1f}us " + " ".join(f"{k[:6]}={v/span*100:.0f}" for k, v in sorted(act.items()) if v > 0.02*span) + " | " + ", ".join(f"{k[0]}:{k[1]}={v/span*100:.0f}" for k, v in t3))
ops = collections.defaultdict(lambda: [0, 0.0])
for i in K: k = (i["label"], i.get("opcode")); ops[k][0] += 1; ops[k][1] += i["duration"]
print("top:", "; ".join(f"{k[0]}/{k[1]} n={n} {d/1e3:.1f}us" for k, (n, d) in sorted(ops.items(), key=lambda x: -x[1][1])[:14]))
