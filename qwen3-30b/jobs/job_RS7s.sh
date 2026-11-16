mkdir -p /tmp/rs; cd /tmp/rs; export NEURON_SKIP_EFA_AFFINITY=1 NEURON_LOGICAL_NC_CONFIG=2 TMPDIR=/tmp NEURON_RT_ROOT_COMM_ID=127.0.0.1:62802 PYTHONUNBUFFERED=1 RSPOD=rs-b
export PORT=8802 SOAK32=1
SOAK_MIN=50 bash /tmp/ni2e/rs_soak.sh RS7
python3 /tmp/ni2e/cbv_ana.py /tmp/rs_out_RS7 > /tmp/rs_out_RS7/summary 2>&1
cp /tmp/rs_steplog_RS7.txt /tmp/rs_out_RS7/steplog.txt
rm -rf /data/cache/rs_results/RS7; cp -a /tmp/rs_out_RS7 /data/cache/rs_results/RS7
cat /tmp/rs_out_RS7/soak/summary.txt; cat /tmp/rs_out_RS7/soak/slog; awk '$1=="S" && $3=="P" {n++; if ($5>1) m++} END {print "psteps", n, m}' /tmp/rs_out_RS7/steplog.txt; echo SOAK7_DONE

echo JOBDONE
