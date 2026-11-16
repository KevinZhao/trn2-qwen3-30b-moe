# nj_cmp.py A_DIR B_DIR : same-session A/B table (A=anchor C, B=C+NM) with deltas; seed/run noise rows
import sys, json, os, statistics as st
A, B = sys.argv[1], sys.argv[2]
def ld(d, n):
    p = f"{d}/runs/{n}.json"
    return json.load(open(p)) if os.path.exists(p) else None
def g(d, n, k):
    j = ld(d, n); return None if j is None else j[k]
def f(x, nd=1): return "-" if x is None else f"{x:.{nd}f}"
def row(lab, n, k, nd=1):
    a, b = g(A, n, k), g(B, n, k)
    d = None if a is None or b is None else b - a; pc = None if d is None or a == 0 else 100 * d / a
    print(f"| {lab} | {f(a,nd)} | {f(b,nd)} | {f(d,nd)} | {f(pc,1)}% |")
print("| metric | C (anchor) | C+NM | delta | delta% |\n|---|---|---|---|---|")
for c, n in ((1, 12), (4, 32), (16, 64)):
    seeds = (42,) if c == 1 else (42, 123)
    for s in seeds:
        row(f"SG c{c} s{s} TTFT mean", f"sg_c{c}_s{s}", "mean_ttft_ms"); row(f"SG c{c} s{s} TTFT med", f"sg_c{c}_s{s}", "median_ttft_ms")
        row(f"SG c{c} s{s} TPOT", f"sg_c{c}_s{s}", "mean_tpot_ms", 2); row(f"SG c{c} s{s} tok/s", f"sg_c{c}_s{s}", "output_throughput")
    row(f"SG c{c} s42 rerun TTFT mean", f"sg_c{c}_s42_r2", "mean_ttft_ms"); row(f"SG c{c} s42 rerun TPOT", f"sg_c{c}_s42_r2", "mean_tpot_ms", 2)
    row(f"SG c{c} s42 rerun tok/s", f"sg_c{c}_s42_r2", "output_throughput")
for n in (256, 1024, 3000, 6000):
    for c in (1, 4):
        row(f"rn{n} c{c} TTFT mean", f"rn{n}_c{c}_s42", "mean_ttft_ms"); row(f"rn{n} c{c} TPOT", f"rn{n}_c{c}_s42", "mean_tpot_ms", 2)
        row(f"rn{n} c{c} tok/s", f"rn{n}_c{c}_s42", "output_throughput")
for n, c in ((1024, 1), (6000, 1), (3000, 4)):
    row(f"rn{n} c{c} rerun TTFT mean", f"rn{n}_c{c}_s42_r2", "mean_ttft_ms")
row("rn7000 c16 TTFT mean", "rn7000_c16_s42", "mean_ttft_ms"); row("rn7000 c16 TTFT med", "rn7000_c16_s42", "median_ttft_ms")
row("rn7000 c16 TPOT", "rn7000_c16_s42", "mean_tpot_ms", 2); row("rn7000 c16 tok/s", "rn7000_c16_s42", "output_throughput")
row("rn7000 c16 rerun TTFT mean", "rn7000_c16_s42_r2", "mean_ttft_ms"); row("rn7000 c16 rerun tok/s", "rn7000_c16_s42_r2", "output_throughput")
for tf in ("ttft", "ttft_r2"):
    ja = json.load(open(f"{A}/{tf}.json")) if os.path.exists(f"{A}/{tf}.json") else {}; jb = json.load(open(f"{B}/{tf}.json")) if os.path.exists(f"{B}/{tf}.json") else {}
    for L in ja:
        a = st.median(ja[L]); b = st.median(jb[L]) if L in jb else None
        print(f"| c1 TTFT sweep{'(rerun)' if tf=='ttft_r2' else ''} len {L} (median of 3) | {f(a)} | {f(b)} | {f(None if b is None else b-a)} | {f(None if b is None else 100*(b-a)/a)}% |")
fail = lambda d: sum(json.load(open(f"{d}/runs/{x}")).get("failed", 0) for x in os.listdir(f"{d}/runs") if x.endswith(".json"))
print(f"| failed requests (all bench runs) | {fail(A)} | {fail(B)} | | |")
