mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
# item 3 microbench: 11r vs 11s variants (single logical core, LAYOUT=fp8s, II_DGE=sw as production)
pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; sleep 20
pgrep -f "serve /data/models" > /dev/null && { echo "server running, abort"; exit 1; }
export PYTHONPATH=/tmp/rs/k II_DGE=sw LAYOUT=fp8s E=128 I=192 L=8
exec 9>/data/cache/ie_bench.lock
run() { T=$1 KM=$2 KWS=$3; flock -w 3600 9; o=$(cd /tmp/rs/k && T=$T KMOD=$KM KW="$KWS" timeout 1500 python3 wsB_t1.py 2>&1); flock -u 9
  echo "CASE T=$T $KM KW=$KWS | $(echo "$o" | grep -oE 'per-layer [0-9.]+ ms' | head -1) | $(echo "$o" | grep -oE 'kernel-vs-dequant-ref rel-fro [0-9.e+-]+') | $(echo "$o" | grep -oE 'rel-fro [0-9.e+-]+, ' | tail -1) | $(echo "$o" | grep -m1 -oE '(Error|error|ERROR)[^\n]{0,200}')"; }
for T in 16 32 4; do
  run $T p2_fused11r '{}'
  for kw in '{}' '{"ND":1}' '{"SPLIT":1}' '{"SC":1}' '{"ND":1,"SPLIT":1,"SC":1}' '{"HWW":1}'; do run $T p2_fused11s "$kw"; done
done
echo MB1_DONE

echo JOBDONE
