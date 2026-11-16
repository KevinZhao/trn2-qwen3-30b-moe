mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
for i in $(seq 1 400); do grep -q MB1_DONE /tmp/rs/log_MB1.txt && break; sleep 10; done
export PYTHONPATH=/tmp/rs/k; exec 9>/data/cache/ie_bench.lock
for T in 16 32; do for M in A B C; do for R in 64 1024; do flock -w 3600 9; (cd /tmp/rs/k && MODE=$M R=$R T=$T timeout 900 python3 pe_probe.py 2>&1 | grep -E "PEPROBE|Error" | head -2); flock -u 9; done; done; done
echo PE_DONE

echo JOBDONE
