# PE cost probe: per-matmul time of (A) stationary x[128,T] bf16 / moving w[128,384] fp8  vs  (B) stationary w[128,128] fp8 / moving x[128,T]
# (C) = B with K=64 rows. env MODE=A|B|C, R repetitions (static), T
import os, time, torch
import nki, nki.isa as nisa, nki.language as nl
import libtorch_neuronx_lite
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.envs import get_compile_backend_name
MODE = os.environ.get("MODE", "A"); R = int(os.environ.get("R", "512")); T = int(os.environ.get("T", "16")); P = 128
@nki.jit
def probe(x, w):
    out = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.shared_hbm)
    xs = nl.ndarray((P, T), dtype=x.dtype, buffer=nl.sbuf); nisa.dma_copy(dst=xs[0:P, 0:T], src=x[0:P, 0:T])
    ws = nl.ndarray((P, 8, 384), dtype=w.dtype, buffer=nl.sbuf); nisa.dma_copy(dst=ws[0:P, 0:8, 0:384], src=w[0:P, 0:8, 0:384])
    ps = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum, name=f"ps{b}") for b in range(2)]
    for b in range(2): nisa.memset(ps[b][0:P, 0:512], 0.0)
    for i in range(R):
        k = i % 8; b = (i // 16) % 2
        if MODE == "A":
            nisa.nc_matmul(dst=ps[b][0:T, 0:384], stationary=xs[0:P, 0:T], moving=ws[0:P, k, 0:384], accumulate=True)
        elif MODE == "B":
            c = (i % 3) * 128
            nisa.nc_matmul(dst=ps[b][0:P, (i % 4) * T:(i % 4) * T + T], stationary=ws[0:P, k, c:c + 128], moving=xs[0:P, 0:T], accumulate=True)
        else:
            c = (i % 3) * 128
            nisa.nc_matmul(dst=ps[b][0:P, (i % 4) * T:(i % 4) * T + T], stationary=ws[0:64, k, c:c + 128], moving=xs[0:64, 0:T], accumulate=True)
    o = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.sbuf)
    o1 = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=o1[0:P, 0:512], src=ps[1][0:P, 0:512])
    nisa.tensor_tensor(o[0:P, 0:512], ps[0][0:P, 0:512], o1[0:P, 0:512], nl.add)
    nisa.dma_copy(dst=out[0:P, 0:512], src=o[0:P, 0:512])
    return out
dev = "neuron:0"
x = (torch.randn(P, T) * 0.1).to(torch.bfloat16).to(dev); w = (torch.randn(P, 8, 384) * 0.1).to(torch.float8_e4m3fn).to(dev)
f = torch.compile(lambda a, b: wrap_nki(probe)[2](x=a, w=b), backend=get_compile_backend_name(), fullgraph=True)
f(x, w).cpu()
for _ in range(3): f(x, w).cpu()
ts = []
for _ in range(5):
    n = 30; t0 = time.time()
    for _ in range(n): f(x, w).cpu()
    ts.append((time.time() - t0) / n)
print(f"PEPROBE MODE={MODE} T={T} R={R} call {min(ts)*1e6:.1f} us", flush=True)
