# nm_capt.py NEFF DUMPDIR OUTDIR : build ifmaps by matching NEFF input shapes with dumped tensors (DUMPDIR/*.npy, file name = any),
# zeros for the rest, then neuron-explorer capture + view json.
import sys, os, io, tarfile, json, re, glob, subprocess, numpy as np
neff, dd, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True)
def nj_of(f):
    b = open(f, "rb").read(); off = b.find(b"\x1f\x8b", 0, 4096)
    t = tarfile.open(fileobj=io.BytesIO(b[off:]), mode="r:gz"); return json.loads(t.extractfile("neff.json").read().decode())
NEED = int(os.environ.get("NIN", "0"))
if os.path.isdir(neff):   # pick newest neff whose input set contains all dumped (numel,dtype) keys, NIN inputs
    want = sorted((int(np.load(f).size)) for f in glob.glob(dd + "/*.npy"))
    cands = sorted(glob.glob(neff + "/**/*.neff", recursive=True), key=os.path.getmtime, reverse=True)
    for f in cands:
        try: sg = nj_of(f)["metadata"]["signatures"]["default"]["inputs"]
        except Exception: continue
        sizes = sorted(int(np.prod(v["shape"])) for v in sg.values())
        if all(w in sizes for w in want) and (NEED == 0 or len(sg) == NEED): neff = f; break
    print("PROF picked", neff, flush=True)
nj = nj_of(neff)
sig = nj["metadata"]["signatures"]["default"]["inputs"]
ins = [dict(name=k, shape=v["shape"], dtype=v["dtype"]) for k, v in sorted(sig.items(), key=lambda kv: kv[1]["id"])]
DT = {"bfloat16": "u2", "float32": "f4", "int32": "i4", "int64": "i8", "uint16": "u2", "bool": "u1", "uint8": "u1", "pred": "u1"}
dumps = {}
for f in glob.glob(dd + "/*.npy"):
    a = np.load(f); key = (int(a.size), a.dtype.kind + str(a.dtype.itemsize)); dumps.setdefault(key, []).append(f)
args = []
for o in ins:
    sh = tuple(int(x) for x in o["shape"]); dt = o["dtype"]; n = int(np.prod(sh))
    d = DT.get(dt, "u2"); key = (n, {"u2": "u2", "f4": "f4", "i4": "i4", "i8": "i8", "u1": "u1"}[d])
    if key in dumps and dumps[key]:
        f = dumps[key].pop(0)
    else:
        f = f"{od}/zero_{o['name']}.npy"; np.save(f, np.zeros(sh, dtype=np.dtype(d)))
    print("PROF ifmap", o["name"], sh, dt, os.path.basename(f), flush=True)
    args += [o["name"], f]
cmd = ["neuron-explorer", "capture", "-n", neff, "-s", f"{od}/p.ntff", "--num-exec", "4", "--profile-nth-exec", "4"] + args
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200); print("PROF capture rc", r.returncode, r.stdout[-1500:], r.stderr[-1500:])
nt = sorted(glob.glob(f"{od}/*.ntff"))
print("PROF ntffs", nt)
if nt:
    r = subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", nt[-1], "--output-format", "json", "--output-file", f"{od}/prof.json", "--ignore-nc-buf-usage"], capture_output=True, text=True, timeout=1800)
    print("PROF view rc", r.returncode, r.stderr[-800:])
