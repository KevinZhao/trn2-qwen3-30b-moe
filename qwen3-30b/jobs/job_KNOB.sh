mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
# knob sweep on live RSP server (NJ config). knob file read by patched scheduler every 0.5s
export TAG=RSP PORT=8802; O=/tmp/rs_out_RSP; R=$O/runs; K=/tmp/ni2_knobs_RSP.json
M=/data/models/Qwen3-30B-A3B; DS=/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
sg() { curl -s -o /dev/null -X POST http://127.0.0.1:$PORT/reset_prefix_cache; vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:$PORT \
  --model qwen3-30b-a3b --tokenizer $M --dataset-name sharegpt --dataset-path $DS --sharegpt-output-len 512 \
  --max-concurrency $1 --num-prompts $2 --seed $3 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
  --save-result --save-detailed --result-dir $R --result-filename kn_$4_c$1_s$3.json > $R/kn_$4_c$1_s$3.log 2>&1; }
exec 9>/data/cache/ie_bench.lock; flock -w 7200 9
for cfg in "1 1" "0 0" "0 1" "1 0" "2 1" "1 2"; do set -- $cfg; echo "{\"dsbpc\": $1, \"dsbp\": $2, \"tag\": \"c$1p$2\"}" > $K; sleep 2
  sg 16 64 42 c$1p$2; sg 16 64 123 c$1p$2; sg 4 32 42 c$1p$2; done
echo '{"dsbpc": 1, "dsbp": 1, "tag": "default"}' > $K
flock -u 9; python3 /tmp/ni2e/cbv_ana.py $O | grep kn_

echo JOBDONE
