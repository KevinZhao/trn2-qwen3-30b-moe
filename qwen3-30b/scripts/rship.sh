#!/bin/bash
# rship.sh <pod> <pod dir> <local dir> [names...]
POD=$1; PD=$2; LD=$3; shift 3; T=/tmp/rs/_s_$$.tgz; B=/root/work/trn2/rs/big_$$.b64
(cd $LD && COPYFILE_DISABLE=1 tar czf $T --no-xattrs --exclude __pycache__ "$@")
base64 < $T | tr -d '\n' > $T.b64; N=$(wc -c < $T.b64); off=0; CH=60000; first=1; echo "ship $N b64 bytes"
while [ $off -lt $N ]; do
  C=$(tail -c +$((off+1)) $T.b64 | head -c $CH); if [ $first = 1 ]; then OP=">"; first=0; else OP=">>"; fi
  printf 'mkdir -p /root/work/trn2/rs; echo -n %s %s %s\n' "$C" "$OP" "$B" > $T.c.sh; /tmp/trn2-hyd/rk.sh $T.c.sh 60 > /dev/null; off=$((off+CH))
done
cat > $T.c.sh <<EOS
base64 -d $B > $B.tgz && rm -f $B && kubectl -n trn2-perf cp $B.tgz $POD:/tmp/rs_ship.tgz && rm -f $B.tgz && kubectl -n trn2-perf exec $POD -- bash -c 'mkdir -p $PD && cd $PD && tar xzf /tmp/rs_ship.tgz && ls $PD | tr "\n" " "'
EOS
/tmp/trn2-hyd/rk.sh $T.c.sh 120; rm -f $T $T.b64 $T.c.sh
