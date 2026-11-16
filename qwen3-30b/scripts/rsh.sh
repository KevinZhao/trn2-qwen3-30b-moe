#!/bin/bash
# rsh.sh <pod> <timeout> <local script> : run synchronously in pod
POD=$1; T=$2; Y=$(base64 < $3 | tr -d '\n'); F=/tmp/rs/_sh_$$.sh
echo "kubectl -n trn2-perf exec $POD -- bash -c 'echo $Y | base64 -d > /tmp/rs_sh_$$.sh; bash /tmp/rs_sh_$$.sh' 2>&1" > $F
/tmp/trn2-hyd/rk.sh $F $T; rm -f $F
