import json, sys, collections, re
j = json.load(open(sys.argv[1])); L = int(sys.argv[2])
ins = j["instruction"]
cc = sorted([o for o in j["cc_ops"] if o.get("subgraph") == "sg00" and o["operation"] == "ALLREDUCE"], key=lambda o: o["timestamp"])
e = lambda o: o["timestamp"] + o["duration"]
a, b = e(cc[1 + 2 * L]), cc[2 + 2 * L]["timestamp"]
W = [i for i in ins if a <= i["timestamp"] < b]
c = collections.Counter(); d = collections.defaultdict(float)
for i in W:
    if i.get("opcode") in ("MATMUL", "LDWEIGHTS", "CAST"):
        dt = tuple(sorted(set(re.findall(r"(float8\w*|bfloat16|float32|fp8\w*)", str(i.get("operands")))))); k = (i["opcode"], dt); c[k] += 1; d[k] += i["duration"]
for k in sorted(c, key=lambda k: -d[k]): print(k, c[k], f"{d[k]/1e3:.1f}us")
for i in W:
    if i.get("opcode") == "CAST": print("CAST eg:", str(i.get("operands"))[:300], i.get("hlo_name", "")[:80]); break
