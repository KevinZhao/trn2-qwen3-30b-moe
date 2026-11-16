#!/bin/bash
# neff_sync_host.sh — 把节点 hostPath 上已有的 /data/cache/<tag>/vc/neuron/native_neff_cache 只读地推到 S3。
# 在 neff-sync pod 里跑（hostPath 以 readOnly 挂到 /src/cache）。跳过 20 分钟内有写入的目录（可能仍在编译）和 $SKIP 里的 tag。
set -uo pipefail
S=/scripts/neff_cache.sh; LOG=/tmp/sync.log; : > $LOG
for d in /src/cache/*/; do
  t=$(basename "$d"); n="$d/vc/neuron/native_neff_cache"
  [ -d "$n" ] || continue
  case " ${SKIP:-} " in *" $t "*) echo "SKIP $t (listed active)" | tee -a $LOG; continue;; esac
  c=$(find "$n" -name '*.neff' | wc -l); [ "$c" -gt 0 ] || { echo "SKIP $t (0 neff)" | tee -a $LOG; continue; }
  NEFF_CACHE_NOTE="seeded from existing hostPath cache on $NODE_NAME by neff-sync" bash $S push "$t" "$d/vc" 2>&1 | tee -a $LOG
done
echo "SYNC_DONE $(date -u +%FT%TZ)" | tee -a $LOG
aws s3 cp $LOG "s3://${NEFF_CACHE_BUCKET:-<NEFF_CACHE_BUCKET>}/_meta/sync-logs/${NODE_NAME}-$(date -u +%Y%m%dT%H%M%SZ).log" --only-show-errors
sleep ${HOLD:-60}
