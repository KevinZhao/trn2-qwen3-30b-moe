mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
pgrep -f "serve /data/models" > /dev/null && { echo "server running"; exit 1; }
for BC in "16 1024" "16 2048" "1 1024"; do
  set -- $BC; N=$(python3 /tmp/rs/pick2.py /data/cache/rs_RS7/vc | awk -v b=$1 -v c=$2 '$1==b && $2==c' | sort -k4,4n | tail -1 | awk '{print $5}'); echo "NEFF $BC -> $N"
  T=r7b$1_$2; flock -w 3600 /data/cache/ie_bench.lock bash /tmp/rs/prof/ie_prof.sh $N $T; tail -2 /tmp/ie_prof_$T/cap.log
  [ -f /tmp/ie_prof_$T/prof.json ] && { python3 /tmp/rs/prof/ie_ana.py /tmp/ie_prof_$T/prof.json; python3 /tmp/rs/moe_ana2.py /tmp/ie_prof_$T/prof.json 20; python3 /tmp/rs/moe_ana.py /tmp/ie_prof_$T/prof.json 20 2>&1 | head -18; python3 /tmp/rs/att3.py /tmp/ie_prof_$T/prof.json 20 | head -12; }
done
echo PF2_DONE

echo JOBDONE
