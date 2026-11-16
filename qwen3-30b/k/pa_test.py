# S0(a): packed causal flash_attention (bound_min/bound_max) vs per-sequence reference. env T K L
import os, time, torch
import libtorch_neuronx_lite
import vllm_neuron.functional as NF
from vllm_neuron.envs import get_compile_backend_name
T = int(os.environ.get("T", "4096")); K = int(os.environ.get("K", "16")); L = int(os.environ.get("L", "8")); Nh, D = 8, 128
torch.manual_seed(0)
cuts = sorted(torch.randperm(T - 200)[:K - 1].tolist()); lens = [b - a for a, b in zip([0] + cuts, cuts + [T - 200])]
lens = [max(1, x) for x in lens]; tot = sum(lens); pad = T - tot
if os.environ.get("EQ") == "1": lens = [T // K] * K; tot = T; pad = 0
starts = [sum(lens[:i]) for i in range(len(lens))]
bmin = torch.zeros(T, dtype=torch.int32); bmax = torch.zeros(T, dtype=torch.int32)
for s, l in zip(starts, lens): bmin[s:s + l] = s; bmax[s:s + l] = s + l
if pad: bmin[tot:] = tot; bmax[tot:] = T            # padding = its own sequence at the tail
q = torch.randn(Nh, T, D).to(torch.bfloat16); k = torch.randn(Nh, T, D).to(torch.bfloat16); v = torch.randn(Nh, T, D).to(torch.bfloat16)
sc = D ** -0.5; dev = "neuron:0"
def attn(qd, kd, vd, bm, bx, packed):
    kw = dict(bound_min=bm.view(1, T, 1).expand(Nh, T, 1).contiguous(), bound_max=bx.view(1, T, 1).expand(Nh, T, 1).contiguous()) if packed else {}
    return NF.flash_attention(qd.transpose(1, 2), kd.transpose(1, 2), vd, scale=sc, tp_q=False, tp_out=True, **kw)   # [Nh, D, T]
fp = torch.compile(lambda a, b, c, d, e: attn(a, b, c, d, e, True), backend=get_compile_backend_name(), fullgraph=True)
fu = torch.compile(lambda a, b, c, d, e: attn(a, b, c, d, e, False), backend=get_compile_backend_name(), fullgraph=True)
args = [t.to(dev) for t in (q, k, v, bmin, bmax)]
y = fp(*args).cpu().float().transpose(1, 2)   # [Nh, T, D]
err = 0.0; nrm = 0.0
for s, l in zip(starts, lens):
    qs, ks, vs = q[:, s:s + l].float(), k[:, s:s + l].float(), v[:, s:s + l].float()
    a = (qs @ ks.transpose(1, 2)) * sc + torch.triu(torch.full((l, l), float("-inf")), 1)
    r = torch.softmax(a, -1) @ vs
    err += (y[:, s:s + l] - r).pow(2).sum().item(); nrm += r.pow(2).sum().item()
print(f"PACKCHK T={T} K={len(lens)} lens={lens[:6]}.. pad={pad}: rel-fro {(err/nrm)**0.5:.3e} nan={torch.isnan(y).any().item()}", flush=True)
def tm(f):
    for _ in range(3): f(*args).cpu()
    ts = []
    for _ in range(5):
        t0 = time.time()
        for _ in range(L): o = f(*args)
        o.cpu(); ts.append((time.time() - t0) / L)
    return min(ts) * 1e3
print(f"PACKTIME T={T}: packed {tm(fp):.3f} ms/call, unpacked causal {tm(fu):.3f} ms/call (per layer per rank)", flush=True)
