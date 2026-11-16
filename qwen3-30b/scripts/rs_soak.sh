#!/bin/bash
# nj_soak.sh TAG : >= SOAK_MIN minutes of mixed load in lock-held segments of SEG_S seconds; monitors server log + health
TAG=$1; PORT=${PORT:?}; O=/tmp/rs_out_$TAG; S=$O/soak; mkdir -p $S; L=$O/server.log; SP=$(cat $O/pid)
SOAK_MIN=${SOAK_MIN:-66}; SEG_S=${SEG_S:-600}; NSEG=$(( (SOAK_MIN * 60 + SEG_S - 1) / SEG_S ))
log() { echo "$(date -u +%H:%M:%S) $*" >> $S/slog; }
PAT='nrta|SEQUENCER|BAD_OPCODE|NRT_TIMEOUT|NRT_EXEC|NRT_FAIL|EngineDeadError|Traceback|FATAL| nan |NaN|hang|timed out'
base=$(wc -l < $L); echo $base > $S/log_base
( while [ ! -f $S/DONE ]; do t0=$(date +%s.%N); c=$(curl -s -m 60 -o /dev/null -w %{http_code} http://127.0.0.1:$PORT/health); t1=$(date +%s.%N); echo "$(date -u +%T) $c $(awk "BEGIN{printf \"%.2f\", $t1-$t0}")" >> $S/health; sleep 30; done ) &
log "soak start NSEG=$NSEG SEG_S=$SEG_S base_log_lines=$base"; load=0
exec 9>/data/cache/ie_bench.lock
python3 /tmp/ni2e/ie_nan.py $PORT > $S/nan0.txt 2>&1; log "nan0 $(grep RESULT $S/nan0.txt)"
for seg in $(seq 1 $NSEG); do
  flock -w 3600 9 && log "seg $seg lock got" || log "seg $seg LOCK TIMEOUT"
  t0=$(date +%s); python3 /tmp/ni2e/rs_soak.py $PORT $S/req.jsonl $SEG_S $seg 1234 >> $S/py.log 2>&1; t1=$(date +%s); load=$((load + t1 - t0))
  flock -u 9
  e=$(tail -n +$((base+1)) $L | grep -ciE "$PAT"); alive=$(kill -0 $SP 2>/dev/null && echo 1 || echo 0)
  n=$(grep -c . $S/req.jsonl); f=$(grep -c '"ok": false' $S/req.jsonl)
  log "seg $seg done dur=$((t1-t0))s load_total=${load}s reqs=$n fails=$f errlines=$e alive=$alive"
  [ $seg = $(( (NSEG + 1) / 2 )) ] && { python3 /tmp/ni2e/ie_nan.py $PORT > $S/nan_mid.txt 2>&1; log "nan_mid $(grep RESULT $S/nan_mid.txt)"; }
  [ $alive = 0 ] && { log "SERVER DIED"; break; }
done
python3 /tmp/ni2e/ie_nan.py $PORT > $S/nan_end.txt 2>&1; log "nan_end $(grep RESULT $S/nan_end.txt)"
tail -n +$((base+1)) $L | grep -iE "$PAT" | head -50 > $S/errlines.txt
python3 - $S/req.jsonl > $S/summary.txt <<'PY'
import sys, json, collections, statistics as st
rs = [json.loads(l) for l in open(sys.argv[1])]
ld = [r for r in rs if r["kind"] != "canary"]; cn = [r for r in rs if r["kind"] == "canary"]
print("load requests", len(ld), "fail", sum(not r["ok"] for r in ld), "bad", sum(1 for r in ld if r.get("bad")))
for k in ("sg", "rn"):
    x = [r for r in ld if r["kind"] == k]; ok = [r for r in x if r["ok"]]
    print(k, "n", len(x), "fail", len(x) - len(ok), "bad", sum(1 for r in x if r.get("bad")), "tok", sum(r.get("ntok") or 0 for r in ok),
          "e2e_max_s", round(max([r["e2e"] for r in ok] or [0]) / 1e3, 1), "ttft_max_s", round(max([r["ttft"] or 0 for r in ok] or [0]) / 1e3, 1))
rn = [r for r in ld if r["kind"] == "rn" and r["ok"]]
print("rn prompts >=4096 tok:", sum(r["L"] >= 4096 for r in rn), " >=6000:", sum(r["L"] >= 6000 for r in rn), " prefill tokens total:", sum(r["L"] for r in rn))
for c in (1, 4, 16, 32): print("conc", c, "n", sum(r["conc"] == c for r in ld))
print("canary n", len(cn), "pass", sum(r.get("pass", False) for r in cn), "fail", sum(not r.get("pass", False) for r in cn))
errs = collections.Counter((r.get("err") or "")[:80] for r in ld if not r["ok"]); print("errors", dict(errs))
bad = [r for r in ld if r.get("bad")][:10]; print("bad samples", [(r["kind"], r["bad"], r.get("snip")) for r in bad])
if rs: print("span_min", round((max(r["t"] for r in rs) - min(r["t"] for r in rs)) / 60, 1))
PY
log "soak end load_total=${load}s"; touch $S/DONE; cat $S/summary.txt >> $S/slog
