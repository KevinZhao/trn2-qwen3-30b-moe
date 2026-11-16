# att_ana.py prof.json [layer] : what runs inside one decode layer's attention segment (between AR boundaries)
import json, sys, collections
j = json.load(open(sys.argv[1])); L = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ins = [i for i in j["instruction"] if i.get("subgraph") == "sg00"]
print("instr keys:", sorted(ins[0].keys()))
cc = sorted([o for o in j["cc_ops"] if o.get("subgraph") == "sg00" and o["operation"] == "ALLREDUCE"], key=lambda o: o["timestamp"])
e = lambda o: o["timestamp"] + o["duration"]
for seg, (a, b) in {"attn": (e(cc[2 * L]), cc[1 + 2 * L]["timestamp"]), "moe": (e(cc[1 + 2 * L]), cc[2 + 2 * L]["timestamp"])}.items():
    W = [i for i in ins if a <= i["timestamp"] < b]
    print(f"== layer {L} {seg}: {(b - a) / 1e3:.1f} us, {len(W)} instr")
    eng = collections.defaultdict(lambda: [0, 0.0])
    for i in W:
        k = i.get("engine") or i.get("queue") or i.get("hw_queue") or "?"; eng[k][0] += 1; eng[k][1] += i["duration"]
    for k, (n, d) in sorted(eng.items(), key=lambda x: -x[1][1]): print(f"   engine {k}: n={n} busy={d / 1e3:.1f} us ({d / (b - a) * 100:.0f}%)")
    nm = collections.defaultdict(lambda: [0, 0.0])
    for i in W:
        k = (i.get("engine") or "?", i.get("opcode") or i.get("name") or i.get("instruction") or "?"); nm[k][0] += 1; nm[k][1] += i["duration"]
    for k, (n, d) in sorted(nm.items(), key=lambda x: -x[1][1])[:22]: print(f"   {k}: n={n} {d / 1e3:.1f} us")
    # kernel/hlo name if present
    for key in ("hlo_name", "kernel_name", "op_name", "label"):
        if key in W[0]:
            c = collections.Counter(); dd = collections.defaultdict(float)
            for i in W: c[i.get(key)] += 1; dd[i.get(key)] += i["duration"]
            print(f"   by {key}:"); [print(f"     {k}: n={c[k]} {dd[k]/1e3:.1f} us") for k in sorted(dd, key=lambda k: -dd[k])[:15]]
    # timeline coarse: 10 buckets, which engines active
    span = (b - a) / 10
    for q in range(10):
        lo, hi = a + q * span, a + (q + 1) * span
        act = collections.defaultdict(float)
        for i in W:
            s0, s1 = max(lo, i["timestamp"]), min(hi, e(i))
            if s1 > s0: act[i.get("engine") or "?"] += s1 - s0
        print(f"   t{q}: " + " ".join(f"{k}={v / span * 100:.0f}%" for k, v in sorted(act.items())))
dm = j.get("dma") or j.get("dma_ops") or []
print("dma entries", len(dm), "keys", sorted(dm[0].keys()) if dm else None)
