#!/bin/bash
# nj_main.sh  env: TAG PKG SRC NMV PORT COMM [PHASES] [KEEP=1] [SOAK=1] [NMX]
cd /tmp/ni2e; C=/data/cache/ni2_$TAG; O=/tmp/ni2_out_$TAG; ML=/tmp/ni2e/main_$TAG.log
echo "$(date -u +%T) $TAG copy $SRC -> $C" >> $ML
[ -d $C ] || { cp -a $SRC $C; ts=$(date +%s); for d in compile_cache torch_compile; do [ -d $C/vc/neuron/$d ] && mv $C/vc/neuron/$d $C/vc/neuron/$d.stale_$ts; done; }   # stale NKI .colz refs (EVRF059)
bash /tmp/ni2e/nj_srv.sh $TAG $PKG $C
echo "$(date -u +%T) $TAG srv ready=$([ -f $O/READY ] && echo 1) $(grep neffs_after $O/info)" >> $ML
save() { python3 /tmp/ni2e/cbv_ana.py $O > $O/summary 2>&1; mkdir -p /data/cache/ni2_results; rm -rf /data/cache/ni2_results/$TAG; cp -a $O /data/cache/ni2_results/$TAG; }
if [ -f $O/READY ]; then
  [ "$PHASES" != none ] && bash /tmp/ni2e/nj_bench.sh $TAG
  save; echo "$(date -u +%T) $TAG bench saved" >> $ML
  [ "$SOAK" = 1 ] && { bash /tmp/ni2e/nj_soak.sh $TAG; save; echo "$(date -u +%T) $TAG soak saved" >> $ML; }
fi
[ "$KEEP" = 1 ] || { kill $(cat $O/pid) 2>/dev/null; sleep 20; pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; sleep 15; }
save; echo "$(date -u +%T) $TAG done" >> $ML; echo MAIN_DONE >> $ML
