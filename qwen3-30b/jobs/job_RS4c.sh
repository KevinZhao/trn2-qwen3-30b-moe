mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
export PORT=8802
PHASES="perf ttft gsm" bash /tmp/ni2e/rs_bench.sh RS4
python3 /tmp/ni2e/cbv_ana.py /tmp/rs_out_RS4 > /tmp/rs_out_RS4/summary 2>&1
cp /tmp/rs_steplog_RS4.txt /tmp/rs_out_RS4/steplog.txt
rm -rf /data/cache/rs_results/RS4; cp -a /tmp/rs_out_RS4 /data/cache/rs_results/RS4
tail -15 /tmp/rs_out_RS4/blog; echo BENCH4_DONE

echo JOBDONE
