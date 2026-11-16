import json, sys, collections, statistics as st
j = json.load(open(sys.argv[1])); L = int(sys.argv[2])
ins = j["instruction"]
cc = sorted([o for o in j["cc_ops"] if o.get("subgraph") == "sg00" and o["operation"] == "ALLREDUCE"], key=lambda o: o["timestamp"])
e = lambda o: o["timestamp"] + o["duration"]
a, b = e(cc[1 + 2 * L]), cc[2 + 2 * L]["timestamp"]
W = [i for i in ins if a <= i["timestamp"] < b]
print(f"layer {L} moe {(b-a)/1e3:.1f} us, {len(W)} instr")
eng = collections.defaultdict(float)
for i in W: eng[i["label"]] += i["duration"]
print(" busy by engine (sum of 2 cores, % of window):", {k: round(v / (b - a) * 100) for k, v in sorted(eng.items(), key=lambda x: -x[1])})
ops = collections.defaultdict(list)
for i in W: ops[(i["label"], i.get("opcode"))].append(i)
for k, v in sorted(ops.items(), key=lambda x: -sum(i["duration"] for i in x[1]))[:14]:
    d = [i["duration"] for i in v]
    print(f"  {k}: n={len(v)} total={sum(d)/1e3:.1f}us median={st.median(d):.0f}ns")
mm = [i for i in W if i.get("opcode") == "MATMUL"]
sh = collections.defaultdict(list)
for i in mm: sh[str(i.get("operands"))[:160]].append(i["duration"])
print(" MATMUL by operands (top):")
for k, v in sorted(sh.items(), key=lambda x: -sum(x[1]))[:8]: print(f"   n={len(v)} med={st.median(v):.0f}ns tot={sum(v)/1e3:.1f}us :: {k}")
lw = [i for i in W if i.get("opcode") == "LDWEIGHTS"]
sh = collections.defaultdict(list)
for i in lw: sh[str(i.get("operands"))[:160]].append(i["duration"])
print(" LDWEIGHTS by operands (top):")
for k, v in sorted(sh.items(), key=lambda x: -sum(x[1]))[:6]: print(f"   n={len(v)} med={st.median(v):.0f}ns tot={sum(v)/1e3:.1f}us :: {k}")
# one core timeline: PE busy fraction per core
for core in sorted({i.get("engine_instance_idx") for i in W}):
    pe = sum(i["duration"] for i in W if i["label"] in ("TensorMatrix",) and i.get("engine_instance_idx") == core)
    print(f" core {core}: TensorMatrix busy {pe/(b-a)*100:.0f}%")
