mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; sleep 10
C=/data/cache/rs_RS4; ts=$(date +%s); for d in compile_cache torch_compile; do [ -d $C/vc/neuron/$d ] && mv $C/vc/neuron/$d $C/vc/neuron/$d.stale_$ts; done; ls $C/vc/neuron/
TAG=RS4 PKG=/data/cache/rs_pkg/pkg_rs3 NMV=1 SRC=/data/cache/nj_CAP PORT=8802 COMM=62802 P5P=1 KEEP=1 \
 NMX="Q3_FUSED_MOE_KMOD=p2_fused11r RS_FP8=1 NM_KMOD=nm_k9f P5_PACK=1 P4_STEPLOG=/tmp/rs_steplog_RS4.txt" PHASES="nan" bash /tmp/ni2e/rs_main.sh

echo JOBDONE
