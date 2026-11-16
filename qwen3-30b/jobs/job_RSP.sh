mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
TAG=RSP PKG=/data/cache/nj_results/pkg_nj NMV=1 SRC=/data/cache/nj_CAP PORT=8802 COMM=62802 PROF=1 KEEP=1 PHASES="prof" bash /tmp/ni2e/rs_main.sh

echo JOBDONE
