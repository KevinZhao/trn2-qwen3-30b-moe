mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
export PORT=8802
P4_KNOBS=/tmp/ni2_knobs_RS4.json python3 /tmp/ni2e/p5_pack_check.py 8802 /data/models/Qwen3-30B-A3B --steplog /tmp/rs_steplog_RS4.txt > /tmp/rs_out_RS4/p5check.txt 2>&1
echo "p5check rc=$?"; tail -25 /tmp/rs_out_RS4/p5check.txt; cat /tmp/ni2_knobs_RS4.json
if grep -q "PASS" /tmp/rs_out_RS4/p5check.txt; then PHASES="perf ttft gsm" bash /tmp/ni2e/rs_bench.sh RS4; python3 /tmp/ni2e/cbv_ana.py /tmp/rs_out_RS4 > /tmp/rs_out_RS4/summary 2>&1; rm -rf /data/cache/rs_results/RS4; cp -a /tmp/rs_out_RS4 /data/cache/rs_results/RS4; cat /tmp/rs_out_RS4/blog | tail -20; fi

echo JOBDONE
