import sys, io, tarfile, glob, re, json, os
for f in glob.glob(sys.argv[1] + "/neuron/native_neff_cache/*/*/*.neff"):
    b = open(f, "rb").read(); off = b.find(b"\x1f\x8b", 0, 4096)
    s = tarfile.open(fileobj=io.BytesIO(b[off:]), mode="r:gz").extractfile("neff.json").read().decode()
    m = re.search(r'"shape":\["list_shape",\[(\[[0-9,]+\](?:,\[[0-9,]+\]){5})', s)
    if not m: continue
    sh = json.loads("[" + m.group(1) + "]")
    if not (sh[0][1] == 2 and sh[1][1] == 1): continue
    print(sh[0][0], sh[5][1] * 32, "fp8" if re.search(r"float8|f8e4|fp8", s) else "bf16", os.path.getmtime(f), f)
