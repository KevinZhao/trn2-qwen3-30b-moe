mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
grep -c "hit_len, n_uncached_common" /tmp/ni2e/p5_patch_pack.py
pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; sleep 15
C=/data/cache/rs_RS4; ts=$(date +%s); for d in compile_cache torch_compile; do [ -d $C/vc/neuron/$d ] && mv $C/vc/neuron/$d $C/vc/neuron/$d.stale_$ts; done; ls $C/vc/neuron/
: > /tmp/rs_steplog_RS4.txt
TAG=RS4 PKG=/data/cache/rs_pkg/pkg_rs3 NMV=1 SRC=/data/cache/nj_CAP PORT=8802 COMM=62802 P5P=1 KEEP=1 \
 NMX="Q3_FUSED_MOE_KMOD=p2_fused11r RS_FP8=1 NM_KMOD=nm_k9f P5_PACK=1 P4_STEPLOG=/tmp/rs_steplog_RS4.txt" PHASES="nan" bash /tmp/ni2e/rs_main.sh
[ -f /tmp/rs_out_RS4/READY ] || { echo SRV_FAIL; exit 1; }
grep RESULT /tmp/rs_out_RS4/nan.txt; grep -c "prefix-hit probe failed" /tmp/rs_out_RS4/server.log
export PORT=8802
P4_KNOBS=/tmp/ni2_knobs_RS4.json python3 /tmp/ni2e/p5_pack_check.py 8802 /data/models/Qwen3-30B-A3B --steplog /tmp/rs_steplog_RS4.txt > /tmp/rs_out_RS4/p5check.txt 2>&1
echo "p5check rc=$?"; tail -25 /tmp/rs_out_RS4/p5check.txt; grep -c "prefix-hit probe failed" /tmp/rs_out_RS4/server.log
echo P5CHECK_DONE

echo JOBDONE
