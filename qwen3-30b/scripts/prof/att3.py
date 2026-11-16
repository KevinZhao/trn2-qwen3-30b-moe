import json, sys, collections, re
j = json.load(open(sys.argv[1])); L = int(sys.argv[2])
ins = j["instruction"]
nm = {}
for r in j["layer_summary"]:
    m = re.search(r"_((?:custom-call|add|all-reduce|[a-z\-]+)\.\d+)$", r["name"]); 
    if m: nm[m.group(1)] = r["name"].split("/")[-1][:90]
cc = sorted([o for o in j["cc_ops"] if o.get("subgraph") == "sg00" and o["operation"] == "ALLREDUCE"], key=lambda o: o["timestamp"])
e = lambda o: o["timestamp"] + o["duration"]
a, b = e(cc[2 * L]), cc[1 + 2 * L]["timestamp"]
W = [i for i in ins if a <= i["timestamp"] < b]
print(f"layer {L} attn {(b-a)/1e3:.1f} us, {len(W)} instr")
eng = collections.defaultdict(float)
for i in W: eng[i["label"]] += i["duration"]
print(" busy by engine:", {k: round(v / (b - a) * 100) for k, v in sorted(eng.items(), key=lambda x: -x[1])})
H = collections.defaultdict(lambda: [1e30, 0, 0, collections.defaultdict(float)])
for i in W:
    hn = (i.get("hlo_name") or "").split(" = ")[0].lstrip("%"); h = H[hn]
    h[0] = min(h[0], i["timestamp"]); h[1] = max(h[1], e(i)); h[2] += 1; h[3][i["label"]] += i["duration"]
for hn, (s0, s1, n, ed) in sorted(H.items(), key=lambda x: x[1][0]):
    print(f"  [{(s0-a)/1e3:6.1f} → {(s1-a)/1e3:6.1f} us] {hn:22s} n={n:5d} {nm.get(hn,'')[:70]}  " + " ".join(f"{k}={v/1e3:.1f}" for k, v in sorted(ed.items(), key=lambda x: -x[1])[:4]))
ops = collections.defaultdict(lambda: [0, 0.0])
for i in W: k = (i["label"], i.get("opcode")); ops[k][0] += 1; ops[k][1] += i["duration"]
print(" top opcodes:"); [print(f"   {k}: n={n} {d/1e3:.1f} us") for k, (n, d) in sorted(ops.items(), key=lambda x: -x[1][1])[:16]]
