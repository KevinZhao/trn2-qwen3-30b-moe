import json, sys
j = json.load(open(sys.argv[1])); s = j["summary"][0]
for k, v in sorted(s.items()):
    if any(w in k for w in ("throttle", "tensor_engine", "mfu", "flops", "active_time", "freq", "clock", "hbm", "dma_active")): print("PROF", k, v)
