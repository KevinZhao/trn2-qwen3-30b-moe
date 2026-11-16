#!/bin/bash
# neff_cache.sh — 在 S3 与本地 vllm-neuron 2.33 native 缓存之间同步 NEFF（只同步可跨节点复用的 native_neff_cache）
#
#   neff_cache.sh ls   [TAG]                    列出当前 STACK 下的配置前缀（或某个 TAG 的 manifest）
#   neff_cache.sh pull TAG [VC_ROOT]            S3 → $VC_ROOT/neuron/native_neff_cache（只增不删，可多次 pull 不同 TAG 取并集）
#   neff_cache.sh push TAG [VC_ROOT]            本地 → S3（只增不删；有 20 分钟内写入的文件时拒绝，FORCE=1 跳过检查）
#   neff_cache.sh stack                         打印本镜像对应的 STACK 前缀
#
# VC_ROOT 默认 $VLLM_CACHE_ROOT（即 /data/cache/<tag>/vc）。S3 布局：
#   s3://$NEFF_CACHE_BUCKET/v1/<STACK>/<TAG>/native_neff_cache/<pk[0:2]>/<pk[2:4]>/<pk>.neff
#                                            /native_neff_cache/cache_key_index/...
#                                     <TAG>/_manifest/<host>-<utc>.txt
# STACK 自动取 vn<vllm-neuron 版本>_cc<neuronx-cc 版本>，如 vn0.29.0.1.2.0_cc2.28.7631.0（可用 NEFF_CACHE_STACK 覆盖）。
# 认证：Pod 用 ServiceAccount trn2-perf/neff-cache（EKS Pod Identity → IAM 角色 <NEFF_CACHE_ROLE>）。
set -uo pipefail
BUCKET=${NEFF_CACHE_BUCKET:?set NEFF_CACHE_BUCKET to your S3 bucket}
export AWS_REGION=${AWS_REGION:-ap-south-2} AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-ap-south-2}
aws configure set default.s3.max_concurrent_requests "${NEFF_CACHE_S3_CONC:-16}" >/dev/null 2>&1 || true

stack() {
  [ -n "${NEFF_CACHE_STACK:-}" ] && { echo "$NEFF_CACHE_STACK"; return; }
  python3 - <<'PY'
import importlib.metadata as m
def v(n):
    try: return m.version(n).split("+")[0]
    except Exception: return "none"
print(f"vn{v('vllm-neuron')}_cc{v('neuronx-cc')}")
PY
}

cmd=${1:-}; TAG=${2:-}; VC=${3:-${VLLM_CACHE_ROOT:-}}
ST=$(stack); BASE="s3://$BUCKET/v1/$ST"
case "$cmd" in
stack) echo "$ST" ;;
ls)
  if [ -z "$TAG" ]; then aws s3 ls "$BASE/"; else aws s3 ls "$BASE/$TAG/_manifest/"; fi ;;
pull|push)
  [ -n "$TAG" ] && [ -n "$VC" ] || { echo "usage: $0 $cmd TAG [VC_ROOT]（或设 VLLM_CACHE_ROOT）" >&2; exit 2; }
  L="$VC/neuron/native_neff_cache"; R="$BASE/$TAG/native_neff_cache/"
  t0=$(date +%s)
  if [ "$cmd" = pull ]; then
    mkdir -p "$L"
    aws s3 sync "$R" "$L/" --only-show-errors --exclude '*.tmp' || { echo "NEFF_CACHE pull FAILED $R" >&2; exit 1; }
    # 带过来的 NEFF 不会和本地 torch_compile/compile_cache 冲突；若 VC_ROOT 里有失败启动留下的这两个目录，见 README（NCC_EVRF059）
  else
    [ -d "$L" ] || { echo "no $L" >&2; exit 1; }
    if [ "${FORCE:-0}" != 1 ] && [ -n "$(find "$L" -type f -mmin -20 2>/dev/null | head -1)" ]; then
      echo "NEFF_CACHE push REFUSED: $L 有 20 分钟内写入的文件（可能仍在编译），等服务 READY 后再 push 或 FORCE=1" >&2; exit 3; fi
    aws s3 sync "$L/" "$R" --only-show-errors --exclude '*.tmp' --exclude '*.lock' || { echo "NEFF_CACHE push FAILED" >&2; exit 1; }
    M=$(mktemp)
    { echo "tag=$TAG stack=$ST host=${NODE_NAME:-$(hostname)} pod=${POD_NAME:-$(hostname)} utc=$(date -u +%FT%TZ)"
      echo "src=$L neff=$(find "$L" -name '*.neff' | wc -l) MB=$(du -sm "$L" | cut -f1)"
      env | grep -E '^(NEURON_CC_FLAGS|NEURON_LOGICAL_NC_CONFIG|NEURON_COMPILER_OPT_LEVEL|NEURON_PLATFORM_TARGET_OVERRIDE)=' | sort
      [ -n "${NEFF_CACHE_NOTE:-}" ] && echo "note=$NEFF_CACHE_NOTE"
      [ -n "${NEFF_CACHE_PKG:-}" ] && md5sum "$NEFF_CACHE_PKG"/*.py 2>/dev/null
    } > "$M"
    aws s3 cp "$M" "$BASE/$TAG/_manifest/${NODE_NAME:-$(hostname)}-$(date -u +%Y%m%dT%H%M%SZ).txt" --only-show-errors; rm -f "$M"
  fi
  echo "NEFF_CACHE $cmd $TAG stack=$ST neff=$(find "$L" -name '*.neff' 2>/dev/null | wc -l) MB=$(du -sm "$L" 2>/dev/null | cut -f1) sec=$(( $(date +%s) - t0 ))" ;;
*) sed -n '2,16p' "$0"; exit 2 ;;
esac
