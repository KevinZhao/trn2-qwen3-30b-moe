mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; sleep 15
: > /tmp/rs_steplog_RS8.txt
TAG=RS8 PKG=/data/cache/rs_pkg/pkg_rs3 NMV=1 SRC=/data/cache/rs_RS7 PORT=8802 COMM=62802 P5P=1 KEEP=1 NSB="[1,4,8,16,32]" MNS=32 NGB=8704 \
 NMX="Q3_FUSED_MOE_KMOD=p2_fused11r RS_FP8=1 NM_KMOD=nm_k9f Q3_FUSED_MOE_MAX_T=32 P5_PACK=1 P5_PACK_BUCKETS=512,1024,2048,4096,8192 P4_STEPLOG=/tmp/rs_steplog_RS8.txt" PHASES="nan" bash /tmp/ni2e/rs_main.sh
[ -f /tmp/rs_out_RS8/READY ] || { echo SRV_FAIL; grep -iE "error|exception" /tmp/rs_out_RS8/server.log | tail -20 | cut -c1-300; exit 1; }
grep -E "neffs_after" /tmp/rs_out_RS8/info
echo '{"dsbpc": 1, "dsbp": 1, "pack_max": 32, "pack_budget": 8192, "tag": "pm32b8k"}' > /tmp/ni2_knobs_RS8.json
export PORT=8802
PHASES="sgx perf ttft gsm" bash /tmp/ni2e/rs_bench.sh RS8
python3 /tmp/ni2e/cbv_ana.py /tmp/rs_out_RS8 > /tmp/rs_out_RS8/summary 2>&1
cp /tmp/rs_steplog_RS8.txt /tmp/rs_out_RS8/steplog.txt; grep K5 /tmp/rs_out_RS8/steplog.txt | tail -2
awk '$1=="S" && $3=="P" {n++; if ($5>1) m++; if ($5>mx) mx=$5; if ($5>1 && $4>4096) big++} END {print "psteps", n, m, mx, "packed>4096tok", big}' /tmp/rs_out_RS8/steplog.txt
rm -rf /data/cache/rs_results/RS8; cp -a /tmp/rs_out_RS8 /data/cache/rs_results/RS8
cat /tmp/rs_out_RS8/summary; tail -10 /tmp/rs_out_RS8/blog; echo RS8_DONE

echo JOBDONE
