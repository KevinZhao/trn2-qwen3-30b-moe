import json, sys, collections
j = json.load(open(sys.argv[1]))
ins = j["instruction"]; print("n instr", len(ins))
c = collections.Counter((i.get("label"), i.get("subgraph")) for i in ins); print(c.most_common(20))
print("layer_summary type", type(j["layer_summary"]), (list(j["layer_summary"])[:3] if isinstance(j["layer_summary"], dict) else j["layer_summary"][:2]))
print("opcode_summary", str(j["opcode_summary"])[:1500])
print("summary keys", list(j["summary"][0].keys())[:80])
