#!/bin/bash
# nj_bench.sh TAG   (server up on $PORT)  phases: nan pfx perf ttft gsm perf2
TAG=$1; PORT=${PORT:?}; O=/tmp/ni2_out_$TAG; R=$O/runs; mkdir -p $R
M=/data/models/Qwen3-30B-A3B; DS=/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json; CLI="python3 /tmp/ni2e/cbv_cli.py"
log() { echo "$(date -u +%H:%M:%S) $*" >> $O/blog; }
RST() { c=$(curl -s -o /dev/null -w %{http_code} -X POST http://127.0.0.1:$PORT/reset_prefix_cache); echo "$1 $c" >> $O/resets; }
sg() { RST sg_c$1_s$3$4; vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:$PORT \
  --model qwen3-30b-a3b --tokenizer $M --dataset-name sharegpt --dataset-path $DS --sharegpt-output-len 512 \
  --max-concurrency $1 --num-prompts $2 --seed $3 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
  --save-result --save-detailed --result-dir $R --result-filename sg_c$1_s$3$4.json > $R/sg_c$1_s$3$4.log 2>&1; }
rn() { RST rn$1_c$2_s$4$5; vllm bench serve --backend vllm --endpoint /v1/completions --base-url http://127.0.0.1:$PORT \
  --model qwen3-30b-a3b --tokenizer $M --dataset-name random --random-input-len $1 --random-output-len 256 --ignore-eos \
  --max-concurrency $2 --num-prompts $3 --seed $4 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
  --save-result --save-detailed --result-dir $R --result-filename rn$1_c$2_s$4$5.json > $R/rn$1_c$2_s$4$5.log 2>&1; }
LOCK() { exec 9>/data/cache/ie_bench.lock; flock -w 3600 9 && log "lock got" || log "LOCK TIMEOUT"; }
UNLOCK() { flock -u 9; log unlock; }
for ph in ${PHASES:-nan pfx perf ttft gsm perf2}; do
  log "phase $ph"
  case $ph in
    nan) python3 /tmp/ni2e/ie_nan.py $PORT > $O/nan.txt 2>&1; grep RESULT $O/nan.txt >> $O/blog ;;
    pfx) $CLI pfx $PORT $O/pfx.json > $O/pfx.txt 2>&1; grep "^PFX " $O/pfx.txt >> $O/blog ;;
    perf) LOCK; sg 2 4 7; rn 3000 2 2 7
      sg 1 12 42; sg 4 32 42; sg 16 64 42; sg 4 32 123; sg 16 64 123
      for n in 256 1024 3000 6000; do rn $n 1 8 42; rn $n 4 16 42; done
      rn 7000 16 32 42; UNLOCK ;;
    ttft) LOCK; $CLI ttft $PORT $O/ttft.json 128,512,1024,2048,2049,3000,4096,4097,6000,8000 3 > $O/ttft.txt 2>&1; UNLOCK ;;
    gsm) $CLI gsm $PORT $O/gsm.json 500 > $O/gsm.txt 2>&1; grep GSM $O/gsm.txt >> $O/blog ;;
    perf2) LOCK; sg 2 4 7 _r2; sg 1 12 42 _r2; sg 4 32 42 _r2; sg 16 64 42 _r2
      rn 1024 1 8 42 _r2; rn 6000 1 8 42 _r2; rn 3000 4 16 42 _r2; rn 7000 16 32 42 _r2
      $CLI ttft $PORT $O/ttft_r2.json 128,512,1024,2048,2049,3000,4096,4097,6000,8000 3 > $O/ttft_r2.txt 2>&1; UNLOCK ;;
  esac
done
echo "resets $(wc -l < $O/resets) non200 $(grep -vc ' 200$' $O/resets)" >> $O/blog
echo "errlines $(grep -ciE 'BAD_OPCODE|SEQUENCER_ERROR|FATAL|Traceback| nan ' $O/server.log)" >> $O/blog
echo "errlines2 $(grep -ciE 'nrta|SEQUENCER|BAD_OPCODE|NRT_TIMEOUT|NRT_EXEC|NRT_FAIL|EngineDeadError|Traceback|FATAL| nan ' $O/server.log)" >> $O/blog
log "bench done"; touch $O/BENCHDONE
