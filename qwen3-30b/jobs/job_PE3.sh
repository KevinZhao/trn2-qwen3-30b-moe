mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
export PYTHONPATH=/tmp/rs/k; exec 9>/data/cache/ie_bench.lock
for M in A B C; do for R in 64 8192; do flock -w 3600 9; (cd /tmp/rs/k && MODE=$M R=$R T=16 timeout 1500 python3 pe_probe.py 2>&1 | grep -E "PEPROBE|got Assert|Error:" | head -2 | cut -c1-300); flock -u 9; done; done
echo PE_DONE

echo JOBDONE
