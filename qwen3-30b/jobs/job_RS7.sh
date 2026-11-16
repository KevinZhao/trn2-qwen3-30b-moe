mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; sleep 15
: > /tmp/rs_steplog_RS7.txt
TAG=RS7 PKG=/data/cache/rs_pkg/pkg_rs3 NMV=1 SRC=/data/cache/rs_RS3 PORT=8802 COMM=62802 P5P=1 KEEP=1 NSB="[1,4,8,16,32]" MNS=32 NGB=8704 \
 NMX="Q3_FUSED_MOE_KMOD=p2_fused11r RS_FP8=1 NM_KMOD=nm_k9f Q3_FUSED_MOE_MAX_T=32 P5_PACK=1 P4_STEPLOG=/tmp/rs_steplog_RS7.txt" PHASES="nan" bash /tmp/ni2e/rs_main.sh
[ -f /tmp/rs_out_RS7/READY ] || { echo SRV_FAIL; tail -30 /tmp/rs_out_RS7/server.log | cut -c1-300; exit 1; }
export PORT=8802
P4_KNOBS=/tmp/ni2_knobs_RS7.json python3 /tmp/ni2e/p5_pack_check.py 8802 /data/models/Qwen3-30B-A3B --steplog /tmp/rs_steplog_RS7.txt > /tmp/rs_out_RS7/p5check.txt 2>&1
tail -22 /tmp/rs_out_RS7/p5check.txt
PHASES="sgx perf ttft gsm" bash /tmp/ni2e/rs_bench.sh RS7
python3 /tmp/ni2e/cbv_ana.py /tmp/rs_out_RS7 > /tmp/rs_out_RS7/summary 2>&1
cp /tmp/rs_steplog_RS7.txt /tmp/rs_out_RS7/steplog.txt
rm -rf /data/cache/rs_results/RS7; cp -a /tmp/rs_out_RS7 /data/cache/rs_results/RS7
tail -12 /tmp/rs_out_RS7/blog; echo RS7_DONE

echo JOBDONE
