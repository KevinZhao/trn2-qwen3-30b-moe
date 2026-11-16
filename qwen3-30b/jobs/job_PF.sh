mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
for i in $(seq 1 600); do grep -q PE_DONE /tmp/rs/log_PE.txt && break; sleep 10; done
pgrep -f "serve /data/models" > /dev/null && { echo "server running"; exit 1; }
ls /tmp/rs/prof
python3 /tmp/rs/prof/ie_pick.py /data/cache/rs_RS7/vc | awk '$1>0' | sort -n | uniq | head -40
for BC in "16 1024" "16 2048" "1 1024"; do
  N=$(python3 /tmp/rs/prof/ie_pick.py /data/cache/rs_RS7/vc $BC | head -1); echo "NEFF $BC -> $N"
  [ -n "$N" ] || continue
  T=b$(echo $BC | tr ' ' _); flock -w 3600 /data/cache/ie_bench.lock bash /tmp/rs/prof/ie_prof.sh $N $T; tail -4 /tmp/ie_prof_$T/cap.log
  [ -f /tmp/ie_prof_$T/prof.json ] && python3 /tmp/rs/prof/ie_ana.py /tmp/ie_prof_$T/prof.json
  head -60 /tmp/ie_prof_$T/sum.txt | grep -iE "total|util|busy|dma|tensor|vector|scalar|gpsimd" | head -20
done
echo PF_DONE

echo JOBDONE
