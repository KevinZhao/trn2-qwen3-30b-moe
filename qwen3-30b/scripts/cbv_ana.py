# cbv_ana.py DIR... : summarize vllm bench result jsons (own analyzer)
import sys, json, glob, os, statistics as st
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/runs/*.json")):
        n = os.path.basename(f)[:-5]
        j = json.load(open(f))
        if "summary" in j:  # own client
            s = j["summary"]; print(f"{d.split('_')[-1]:5s} OWN {n:18s} n={s['n']:3d} fail={s['fail']} | TTFT {s['ttft_mean']:7.1f} {s['ttft_med']:7.1f} {s['ttft_p99']:7.1f} | TPOT {s['tpot_mean']:6.2f} | ITL {s['itl_med']:6.2f} max {s['itl_max']:7.1f} | tok/s {s['out_tok_s']:6.1f} multi={s['req_multi_tok_chunks']}"); continue
        itl = [x * 1e3 for r in j.get("itls", []) for x in r]
        print(f"{d.split('_')[-1]:5s} VB  {n:18s} n={j['completed']:3d} fail={j.get('failed',0)} | TTFT {j['mean_ttft_ms']:7.1f} {j['median_ttft_ms']:7.1f} {j['p99_ttft_ms']:7.1f} | TPOT {j['mean_tpot_ms']:6.2f} p99 {j.get('p99_tpot_ms',-1):6.2f} | ITL {j['median_itl_ms']:6.2f} max {max(itl) if itl else -1:7.1f} | tok/s {j['output_throughput']:6.1f} in={j.get('total_input_tokens')}")
