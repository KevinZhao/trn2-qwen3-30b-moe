#!/bin/bash
# rrun.sh <pod> <tag> <local job script> : detached in pod, cwd /tmp/rs, log /tmp/rs/log_<tag>.txt
POD=$1; TAG=$2; case $POD in rs-a) CP=62801;; rs-b) CP=62802;; rs-c) CP=62803;; rs-d) CP=62804;; esac
Y=$( (echo "mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:$CP PYTHONUNBUFFERED=1 RSPOD=$POD"; cat $3; echo; echo 'echo JOBDONE') | base64 | tr -d '\n')
F=/tmp/rs/_run_$$.sh
cat > $F <<EOS
kubectl -n trn2-perf exec $POD -- bash -c 'mkdir -p /tmp/rs; echo $Y | base64 -d > /tmp/rs/job_$TAG.sh; setsid nohup bash /tmp/rs/job_$TAG.sh > /tmp/rs/log_$TAG.txt 2>&1 < /dev/null & echo started $TAG'
EOS
/tmp/trn2-hyd/rk.sh $F 60; rm -f $F
