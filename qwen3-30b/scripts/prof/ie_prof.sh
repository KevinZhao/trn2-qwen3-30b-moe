#!/bin/bash
# in-pod: ie_prof.sh NEFF TAG  -> /tmp/ie_prof_TAG/{p*.ntff, cap.log, sum.txt, prof.json}
NEFF=$1; TAG=$2; D=/tmp/ie_prof_$TAG; rm -rf $D; mkdir -p $D; cd $D
export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62214
if pgrep -f "serve /data/models" > /dev/null; then echo "server running, abort"; exit 1; fi
cp $NEFF $D/m.neff
timeout 1200 neuron-explorer capture -n $D/m.neff -s $D/p.ntff -r 4 --collectives-worker-count 4 -i ${PID:-0} --num-exec ${NEX:-6} --profile-nth-exec ${NEX:-6} ${CAPX:-} > cap.log 2>&1; echo "cap_rc=$?" >> cap.log
ls -la $D >> cap.log
S=$(ls $D/*.ntff 2>/dev/null | head -1); echo "ntff=$S" >> cap.log
[ -n "$S" ] || exit 1
timeout 1200 neuron-explorer view -n $D/m.neff -s $S --output-format summary-text > sum.txt 2>&1
timeout 1800 neuron-explorer view -n $D/m.neff -s $S --output-format json --output-file $D/prof.json --ignore-dma-trace --ignore-nc-buf-usage > view.log 2>&1; echo "view_rc=$?" >> cap.log
ls -la $D >> cap.log; touch $D/DONE
