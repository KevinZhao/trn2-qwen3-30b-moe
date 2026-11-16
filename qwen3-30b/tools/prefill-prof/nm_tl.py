# nm_tl.py prof.json [sg] [t_from_us] [t_len_us] : per-core engine busy + a timeline window + DMA activity summary
import json, sys, collections, statistics as st
j = json.load(open(sys.argv[1])); sg = sys.argv[2] if len(sys.argv) > 2 else "sg00"
ins = sorted([i for i in j["instruction"] if i.get("fully_qualified_subgraph","").lstrip("/") == sg], key=lambda i: i["timestamp"])
t0 = ins[0]["timestamp"]; t1 = max(i["timestamp"] + i["duration"] for i in ins)
print(f"PROF {sg} span {(t1-t0)/1e3:.1f}us n={len(ins)}")
b = collections.Counter(); w = collections.Counter()
for i in ins:
    (w if ("SEMAPHORE" in str(i.get("opcode")) or i.get("opcode") == "DRAIN") else b)[i["label"]] += i["duration"]
print("PROF busy(no-wait) " + " ".join(f"{k}={v/1e3:.0f}" for k, v in b.most_common()) + " | wait " + " ".join(f"{k}={v/1e3:.0f}" for k, v in w.most_common()))
dma = [d for d in j.get("dma", []) if d.get("subgraph") == sg]
if dma:
    q = collections.Counter(); qb = collections.Counter()
    for d in dma: q[d.get("dma_queue")] += d["duration"]; qb[d.get("dma_queue")] += d.get("transfer_size", 0)
    dt0 = min(d["timestamp"] for d in dma); dt1 = max(d["timestamp"] + d["duration"] for d in dma)
    print(f"PROF dma n={len(dma)} span {(dt1-dt0)/1e3:.0f}us bytes {sum(qb.values())/1e6:.1f}MB -> {sum(qb.values())/(dt1-dt0):.1f} GB/s")
    for k, v in q.most_common(12): print(f"PROF dmaq {k}: busy {v/1e3:.0f}us {qb[k]/1e6:.1f}MB")
    de = collections.Counter(); deb = collections.Counter(); sz = collections.Counter()
    for d in dma: de[d.get("dma_engine")] += d["duration"]; deb[d.get("dma_engine")] += d.get("transfer_size", 0); sz[d.get("transfer_size")] += 1
    print("PROF dma engines", sorted((k, round(deb[k] / 1e6, 1), round(v / 1e3)) for k, v in de.items()))
    print("PROF dma sizes", sz.most_common(8))
    # dma bytes over time (10 bins)
    nb = 10; bins = [0] * nb
    for d in dma: bins[min(nb - 1, int((d["timestamp"] - dt0) / (dt1 - dt0 + 1) * nb))] += d.get("transfer_size", 0)
    print("PROF dma GB/s per decile", [round(x / ((dt1 - dt0) / nb), 1) for x in bins])
a = float(sys.argv[3]) if len(sys.argv) > 3 else (t1 - t0) / 2e3; L = float(sys.argv[4]) if len(sys.argv) > 4 else 40
for i in ins:
    r = (i["timestamp"] - t0) / 1e3
    if a <= r < a + L and i["duration"] > 30:
        print(f"PROF tl {r:9.2f} {i['label'][:8]:8s} {str(i.get('opcode'))[:22]:22s} {i['duration']:6d} {str(i.get('free_dimension'))[:5]:5s} {str(i.get('operands'))[:70]}")
