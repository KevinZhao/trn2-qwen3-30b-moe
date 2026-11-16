# nj_flip.py A/gsm.json B/gsm.json : per-question flips
import sys, json
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
n = min(len(a), len(b)); w2r = sum(1 for i in range(n) if not a[i]["ok"] and b[i]["ok"]); r2w = sum(1 for i in range(n) if a[i]["ok"] and not b[i]["ok"])
same = sum(1 for i in range(n) if a[i]["out"] == b[i]["out"])
print(f"FLIP n={n} accA={sum(x['ok'] for x in a[:n])/n:.3f} accB={sum(x['ok'] for x in b[:n])/n:.3f} wrong->right={w2r} right->wrong={r2w} identical_text={same}")
