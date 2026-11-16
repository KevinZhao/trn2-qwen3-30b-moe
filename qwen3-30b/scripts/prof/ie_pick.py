# ie_pick.py CACHE_VC_DIR [B CTX] -> list decode NEFFs as "B CTX path" (or the one matching)
import sys, os, io, tarfile, json, glob, re
out = []
for f in glob.glob(sys.argv[1] + "/neuron/native_neff_cache/*/*/*.neff"):
    b = open(f, "rb").read(); off = b.find(b"\x1f\x8b", 0, 4096)
    s = tarfile.open(fileobj=io.BytesIO(b[off:]), mode="r:gz").extractfile("neff.json").read().decode()
    m = re.search(r'"shape":\["list_shape",\[(\[[0-9,]+\](?:,\[[0-9,]+\]){5})', s)
    sh = json.loads("[" + m.group(1) + "]")
    if sh[0][1] == 2 and sh[1][1] == 1:
        out.append((sh[0][0], sh[5][1] * 32, f))
    else:
        out.append((0, 0, f))
out.sort()
if len(sys.argv) > 3:
    for B, C, f in out:
        if B == int(sys.argv[2]) and C == int(sys.argv[3]): print(f)
else:
    for B, C, f in out: print(B, C, f)
