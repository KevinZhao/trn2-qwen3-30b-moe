mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
pgrep -f "serve /data/models" > /dev/null && { echo "server already running"; exit 0; }
C=/data/cache/rs_RS7; ts=$(date +%s); for d in compile_cache torch_compile; do [ -d $C/vc/neuron/$d ] && mv $C/vc/neuron/$d $C/vc/neuron/$d.stale_$ts; done
cp -a /tmp/rs_out_RS7 /data/cache/rs_results/RS7_prev_out 2>/dev/null
: > /tmp/rs_steplog_RS7.txt
TAG=RS7 PKG=/data/cache/rs_pkg/pkg_rs3 NMV=1 SRC=/data/cache/rs_RS3 PORT=8802 COMM=62802 P5P=1 KEEP=1 NSB="[1,4,8,16,32]" MNS=32 NGB=8704 \
 NMX="Q3_FUSED_MOE_KMOD=p2_fused11r RS_FP8=1 NM_KMOD=nm_k9f Q3_FUSED_MOE_MAX_T=32 P5_PACK=1 P4_STEPLOG=/tmp/rs_steplog_RS7.txt" PHASES="nan" bash /tmp/ni2e/rs_main.sh
echo '{"dsbpc": 1, "dsbp": 1, "pack_max": 16, "pack_budget": 4096, "tag": "rs7"}' > /tmp/ni2_knobs_RS7.json
grep -E "neffs_after" /tmp/rs_out_RS7/info; grep RESULT /tmp/rs_out_RS7/nan.txt; ls /tmp/rs_out_RS7/READY && echo RS7R_READY

echo JOBDONE
