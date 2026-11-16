mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
export PORT=8802; T=RS7pm32; O=/tmp/rs_out_$T; rm -rf $O; mkdir -p $O/runs; ln -s /tmp/rs_out_RS7/server.log $O/server.log
cat /tmp/ni2_knobs_RS7.json; echo
echo '{"dsbpc": 1, "dsbp": 1, "pack_max": 32, "pack_budget": 4096, "tag": "pm32"}' > /tmp/ni2_knobs_RS7.json
sleep 20; grep "P5 knobs" /tmp/rs_out_RS7/server.log | tail -2 | cut -c1-200
S0=$(wc -l < /tmp/rs_steplog_RS7.txt)
PHASES="sgx perf gsm" bash /tmp/ni2e/rs_bench.sh $T
python3 /tmp/ni2e/cbv_ana.py $O > $O/summary 2>&1
tail -n +$((S0+1)) /tmp/rs_steplog_RS7.txt > $O/steplog.txt
awk '$1=="S" && $3=="P" {n++; if ($5>1) m++; if ($5>mx) mx=$5} END {print "psteps", n, m, mx}' $O/steplog.txt
rm -rf /data/cache/rs_results/$T; cp -a $O /data/cache/rs_results/$T; rm -f /data/cache/rs_results/$T/server.log
cat $O/summary; tail -8 $O/blog; echo PM32_DONE

echo JOBDONE
