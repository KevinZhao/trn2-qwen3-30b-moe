#!/bin/bash
# nj_srv.sh TAG PKG CACHE   env: PORT COMM NMV(0|1|unset) [NMX="K=V ..." extra env]
TAG=$1; PKG=$2; C=$3; PORT=${PORT:?}; COMM=${COMM:?}
VN=/opt/conda/lib/python3.14/site-packages/vllm_neuron; Q=$VN/model/qwen3_moe; S=$VN/vllm/core/scheduler.py
O=/tmp/rs_out_$TAG; rm -rf $O; mkdir -p $O/runs; L=$O/server.log
mkdir -p $Q; rm -rf $Q/*.py $Q/__pycache__; cp $PKG/*.py $Q/
grep -q Qwen3MoeForCausalLM $VN/model/registry.py || python3 - $VN/model/registry.py <<'PY'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("from .qwen3_asr import","from .qwen3_moe import Qwen3MoeForCausalLM\nfrom .qwen3_asr import",1)
s=s.replace('("Qwen3ForCausalLM", Qwen3ForCausalLM),','("Qwen3ForCausalLM", Qwen3ForCausalLM),\n        ("Qwen3MoeForCausalLM", Qwen3MoeForCausalLM),',1)
open(p,"w").write(s)
PY
[ -f $S.njorig ] || cp $S $S.njorig; cp $S.njorig $S
unset Q3_FUSED_MOE Q3_FUSED_MOE_KMOD WSA_FN II_DGE P3_KVIX P4_DSBPC P4_DSBP P4_KNOBS NM_MOE NM_KMOD NM_KW NM_COMB NM_LEAN NM_CAP NF_MAP NF_MAPMOD NF_COMBMOD NF_CMODE NF_CTB NF_O_MIN NF_SPLIT NF_O2 NI_NH NI_NH_MAX NI_NH_MIN NH_R NI_MAPMOD
python3 /tmp/ni2e/p4_patch_sched.py $S || exit 1
for f in $VN/vllm/worker/neuron_model_runner.py $VN/vllm/worker/neuron_worker.py; do [ -f $f.p5orig ] && cp $f.p5orig $f; done
[ "$P5P" = 1 ] && { python3 /tmp/ni2e/p5_patch_pack.py $VN >> $O/p5patch.log 2>&1 || { echo P5PATCH_FAIL >> $O/info; touch $O/DIED; exit 1; }; rm -rf $VN/vllm/worker/__pycache__; }
export Q3_FUSED_MOE=1 Q3_FUSED_MOE_KMOD=p2_fused11n WSA_FN=1 II_DGE=sw P3_KVIX=1 P4_DSBPC=1 P4_DSBP=1 P4_KNOBS=/tmp/ni2_knobs_$TAG.json
[ -n "$NMV" ] && export NM_MOE=$NMV
for kv in $NMX; do export "$kv"; done
rm -f /tmp/ni2_knobs_$TAG.json; rm -rf $VN/vllm/core/__pycache__
mkdir -p $C/home $C/vc
export VLLM_MQ_MAX_CHUNK_BYTES_MB=1 NEURON_SKIP_EFA_AFFINITY=1 TMPDIR=/tmp VLLM_CACHE_ROOT=$C/vc HOME=$C/home \
  NEURON_RT_ROOT_COMM_ID=127.0.0.1:$COMM VLLM_SERVER_DEV_MODE=1
rm -rf /var/tmp/nki-* $C/home/.cache/neuron_libtorch /root/.cache/neuron_libtorch
NSB=${NSB:-[1,4,8,16]}; NBT=${NBT:-[128,256,512,1024,2048,4096,8192]}; DCB=${DCB:-[1024,2048,4096]}; MNS=${MNS:-16}; MNBT=${MNBT:-8192}; NGB=${NGB:-4352}
PROFJ=""; [ "$PROF" = 1 ] && PROFJ=',"neuron_profiler":{"activities":["device_profile"],"output_dir":"/tmp/rs_prof_'$TAG'"}'
CFG='{"neuron_config":{"quantization":"bf16","on_device_sampling_config":{"all_greedy":true},"num_batched_tokens_buckets":'$NBT',"num_seqs_buckets":'$NSB',"decode_context_length_buckets":'$DCB',"kv_segment_size_buckets":[8192]'${NCX:-}'}'$PROFJ'}'
[ "$PROF" = 1 ] && SRVX="$SRVX --profiler-config {\"profiler\":\"cuda\",\"max_iterations\":${PMAX:-4}}"
echo "CFG=$CFG SRVX=$SRVX" > $O/cfg
find $C/vc -name '*.neff' | xargs -r -n1 basename | sort > $O/neffs_before
{ echo "TAG=$TAG MNS=$MNS NSB=$NSB PKG=$PKG CACHE=$C PORT=$PORT COMM=$COMM start $(date -u)"; env | grep -E "^(Q3_|WSA_FN|II_DGE|P3_|P4_|NM_|NF_|NI_|NH_|NEURON_RT_ROOT|VLLM_MQ|NEURON_SKIP)" | sort; md5sum $Q/*.py $S; echo "neffs_before $(wc -l < $O/neffs_before)"; } > $O/info
( mx=0; while [ ! -f $O/READY ] && [ ! -f $O/DIED ]; do m=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0); [ $m -gt $mx ] && mx=$m && echo $mx > $O/mem_peak; sleep 5; done ) &
cd /tmp; vllm serve /data/models/Qwen3-30B-A3B --served-model-name qwen3-30b-a3b --host 127.0.0.1 --port $PORT --tensor-parallel-size 4 \
  --max-num-batched-tokens $MNBT --max-num-seqs $MNS --max-model-len ${MML:-8192} --num-gpu-blocks-override $NGB --additional-config "$CFG" $SRVX > $L 2>&1 &
SP=$!; echo $SP > $O/pid
for i in $(seq 1 360); do grep -q "Application startup complete" $L && break; kill -0 $SP 2>/dev/null || { echo "DIED $(date -u)" >> $O/info; touch $O/DIED; exit 1; }; sleep 15; done
grep -q "Application startup complete" $L || { echo "TIMEOUT" >> $O/info; touch $O/DIED; exit 1; }
find $C/vc -name '*.neff' | xargs -r -n1 basename | sort > $O/neffs_after
{ echo "ready $(date -u)"; echo "neffs_after $(wc -l < $O/neffs_after) new=$(comm -13 $O/neffs_before $O/neffs_after | wc -l)"; echo "mem_peak_GiB $(( $(cat $O/mem_peak) / 1073741824 ))"
  for p in $(pgrep -f "VLLM|vllm" ); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -E "^(P3_KVIX|Q3_FUSED_MOE_KMOD|P4_DSBP|NM_MOE|NM_CAP|II_DGE|NF_MAP|NI_NH|NH_R)=" | sed "s/^/pid$p /"; done | sort -u -k2 | head; } >> $O/info
touch $O/READY
