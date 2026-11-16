#!/bin/bash
# cc_test.sh — rs-cache-test pod 内：W = 从 S3 拉 rs_RS0 的 NEFF 后启动 RS0 配置；C = 空缓存冷启动同一配置；对比耗时与 NEFF 集合
set -u
RUN=${RUN:-$(date -u +%Y%m%dT%H%M%SZ)}; S3R=s3://<NEFF_CACHE_BUCKET>/_meta/cache-test/$RUN
mkdir -p /tmp/ni2e; cp /tscripts/rs_srv.sh /tscripts/p4_patch_sched.py /tmp/ni2e/
export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp PYTHONUNBUFFERED=1
R=/tmp/cc_res; mkdir -p $R; up() { aws s3 sync $R $S3R/ --only-show-errors; }
srv() {  # $1=tag $2=cache root
  local T=$1 C=$2 t0 t1 rc O=/tmp/rs_out_$1
  t0=$(date +%s); PORT=8810 COMM=62810 NMV=1 bash /tmp/ni2e/rs_srv.sh $T /pkg/pkg_nj $C; rc=$?; t1=$(date +%s)
  curl -s -m 180 localhost:8810/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-30b-a3b","prompt":"The capital of France is","max_tokens":16,"temperature":0}' > $R/${T}_resp.json
  local nc; nc=$(grep -cE "neuronx-cc|Compil(ing|e) .*neff|NEFF compil" $O/server.log)
  echo "PHASE $T rc=$rc startup_sec=$((t1-t0)) compile_log_lines=$nc $(grep -E 'neffs_before|neffs_after' $O/info | tr '\n' ' ') resp=$(head -c 300 $R/${T}_resp.json)" | tee -a $R/summary.txt
  kill $(cat $O/pid) 2>/dev/null; sleep 20; pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; sleep 15
  cp $O/info $R/${T}_info.txt; cp $O/neffs_before $R/${T}_neffs_before.txt; cp $O/neffs_after $R/${T}_neffs_after.txt 2>/dev/null
  tail -300 $O/server.log > $R/${T}_server_tail.log; gzip -c $O/server.log > $R/${T}_server.log.gz; up
}
echo "RUN=$RUN node=$NODE_NAME start $(date -u +%FT%TZ)" | tee $R/summary.txt
# W: warm from S3
W=/work/W; rm -rf $W; t0=$(date +%s)
bash /scripts/neff_cache.sh pull rs_RS0 $W/vc 2>&1 | tee -a $R/summary.txt
echo "PULL_SEC $(( $(date +%s) - t0 ))" | tee -a $R/summary.txt; up
srv ccW $W
# C: cold
if [ "${SKIP_COLD:-0}" != 1 ]; then C=/work/C; rm -rf $C; srv ccC $C
  echo "NEFFSET cold_vs_s3: only_s3=$(comm -23 $R/ccW_neffs_before.txt $R/ccC_neffs_after.txt | wc -l) only_cold=$(comm -13 $R/ccW_neffs_before.txt $R/ccC_neffs_after.txt | wc -l) common=$(comm -12 $R/ccW_neffs_before.txt $R/ccC_neffs_after.txt | wc -l)" | tee -a $R/summary.txt
fi
echo "ALL_DONE $(date -u +%FT%TZ)" | tee -a $R/summary.txt; up
rm -rf /work/W /work/C
