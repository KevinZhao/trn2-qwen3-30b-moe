# test + bench harness for fused Qwen3 MoE kernels. env: KMOD, E, I, T, L, CHECK
import os, time, importlib, torch
import torch.nn.functional as F
E=int(os.environ.get("E","128")); I=int(os.environ.get("I","192")); T=int(os.environ.get("T","1")); L=int(os.environ.get("L","16"))
H=2048; P=128; H1=H//P; K=8; EPS=1e-6
ID = I if I <= 128 else (I//2 if I % 128 else 128); ND = I // ID
KMOD=os.environ.get("KMOD","wsB_fused1"); KFN=os.environ.get("KFN","q3moe_fused")
torch.manual_seed(0)
LAYOUT=os.environ.get("LAYOUT","nat")
FMAX = float(os.environ.get("FMAX", "240"))
def quant_fp8(gu, dn):
    # gu [E,H,2I]: per output column; dn [E,I,H]: per input row i
    sgu = gu.float().abs().amax(dim=1).clamp_min(1e-12) / FMAX           # [E, 2I]
    sdn = dn.float().abs().amax(dim=2).clamp_min(1e-12) / FMAX           # [E, I]
    guq = (gu.float() / sgu[:, None, :]).to(torch.float8_e4m3fn)
    dnq = (dn.float() / sdn[:, :, None]).to(torch.float8_e4m3fn)
    Iq = dn.shape[1]; NR = 2 * Iq // P
    ii = (64 * torch.arange(NR)[None, :] + torch.arange(P)[:, None] // 2)  # [P, NR] -> channel i
    sg = sgu[:, :Iq][:, ii]                                                # [E, P, NR]
    sud = (sgu[:, Iq:] * sdn)[:, ii]
    sc = torch.cat([sg, sud], dim=2).contiguous().float()                  # [E, P, 2NR]
    return guq.contiguous(), (dnq.contiguous(), sc)
def dequant_fp8(guq, dnsc):
    dnq, sc = dnsc
    raise NotImplementedError
def relayout(gu, dn, rwT):
    if LAYOUT == "fp8s":
        sg_ = gu[:, :, :gu.shape[2]//2].float().abs().amax(dim=(1, 2)).clamp_min(1e-12) / FMAX
        su_ = gu[:, :, gu.shape[2]//2:].float().abs().amax(dim=(1, 2)).clamp_min(1e-12) / FMAX
        sd_ = dn.float().abs().amax(dim=(1, 2)).clamp_min(1e-12) / FMAX
        sgu_ = torch.cat([sg_[:, None].expand(-1, gu.shape[2]//2), su_[:, None].expand(-1, gu.shape[2]//2)], dim=1)
        guq = (gu.float() / sgu_[:, None, :]).to(torch.float8_e4m3fn); dnq = (dn.float() / sd_[:, None, None]).to(torch.float8_e4m3fn)
        return guq.contiguous(), (dnq.contiguous(), sg_[:, None].contiguous(), (su_ * sd_)[:, None].contiguous()), rwT.contiguous()
    if LAYOUT == "fp8":
        return quant_fp8(gu, dn) + (rwT.contiguous(),)
    if LAYOUT == "nat":
        if os.environ.get('WFP8') == '1': return (gu.float()*32).to(torch.float8_e4m3fn).contiguous(), (dn.float()*32).to(torch.float8_e4m3fn).contiguous(), rwT.contiguous()   # timing probe only
        return gu.contiguous(), dn.contiguous(), rwT.contiguous()
    if LAYOUT == "blob":
        NR = 2 * I // P; HB = H // 2
        gub = gu.reshape(E, P, H1 * 2 * I)
        dnf = dn.reshape(E, 2 * I, HB).reshape(E, NR, P, HB).permute(0, 2, 1, 3).reshape(E, P, NR * HB)
        blob = torch.cat([gub, dnf], dim=2).contiguous()
        if os.environ.get('WFP8') == '1': blob = (blob.float() * 32).to(torch.float8_e4m3fn)   # timing probe: W8A16 (scale folded out)
        return blob, None, rwT.contiguous()
    gu_re = gu.reshape(E, H1, P, 2*I).permute(0, 2, 1, 3).contiguous()
    dn_re = dn.reshape(E, ND, ID, H).permute(0, 2, 1, 3).contiguous()
    rw_re = rwT.reshape(H1, P, E).permute(1, 0, 2).contiguous()
    return gu_re, dn_re, rw_re
def ref(x, lnw, rwT, gu, dn):
    xf = x.float(); var = xf.pow(2).mean(-1, keepdim=True)
    xn = (xf * torch.rsqrt(var + EPS)).to(torch.bfloat16) * lnw.to(torch.bfloat16)
    xn = xn.float()
    lg = xn @ rwT.float()
    tv, ti = torch.topk(lg, K, dim=-1)
    aff = torch.softmax(tv, dim=-1)
    out = torch.zeros(x.shape[0], H)
    for t in range(x.shape[0]):
        for k in range(K):
            e = ti[t, k].item()
            gup = xn[t] @ gu[e].float()          # [2I]
            h = F.silu(gup[:I]) * gup[I:]
            out[t] += aff[t, k] * (h @ dn[e].float())
    return out, ti
def mkw():
    gu = (torch.randn(E, H, 2*I) * 0.02).to(torch.bfloat16)
    dn = (torch.randn(E, I, H) * 0.02).to(torch.bfloat16)
    rwT = (torch.randn(H, E) * 0.05).to(torch.bfloat16)
    lnw = (1.0 + 0.1*torch.randn(1, H)).to(torch.bfloat16)
    return gu, dn, rwT, lnw
import libtorch_neuronx_lite
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.envs import get_compile_backend_name
kern = getattr(importlib.import_module(KMOD), KFN)
dev = "neuron:0"
x = (torch.randn(T, H) * 1.0).to(torch.bfloat16)
gu, dn, rwT, lnw = mkw()
gu_re, dn_re, rw_re = relayout(gu, dn, rwT)
KW = eval(os.environ.get('KW', '{}'))
def call(xd, p, rw=None, lnw=None):
    rw = p[2] if rw is None else rw; lnw = p[3] if lnw is None else lnw
    if LAYOUT == 'blob': return wrap_nki(kern)[2](hidden=xd, ln_w=lnw, rw=rw, wb=p[0], I=I, **KW)
    if LAYOUT == 'fp8s': return wrap_nki(kern)[2](hidden=xd, ln_w=lnw, rw=rw, gu=p[0], dn=p[1][0], sg=p[1][1], sud=p[1][2], **KW)
    if LAYOUT == 'fp8': return wrap_nki(kern)[2](hidden=xd, ln_w=lnw, rw=rw, gu=p[0], dn=p[1][0], sc=p[1][1], **KW)
    return wrap_nki(kern)[2](hidden=xd, ln_w=lnw, rw=rw, gu=p[0], dn=p[1], **KW)
todev = lambda b: None if b is None else (tuple(t.to(dev) for t in b) if isinstance(b, tuple) else b.to(dev))
pd0 = (gu_re.to(dev), todev(dn_re), rw_re.to(dev), lnw.to(dev))
f1 = torch.compile(lambda xd: call(xd, pd0), backend=get_compile_backend_name(), fullgraph=True)
t0 = time.time(); y = f1(x.to(dev)).cpu().float(); y = y.sum(0) if y.dim() == 3 else y; print("first call s", round(time.time()-t0, 1), flush=True)
r, ti = ref(x, lnw, rwT, gu, dn)
if LAYOUT == "fp8s":
    gq, (dq, sg_, sud_), _ = relayout(gu, dn, rwT); I2 = gu.shape[2]//2
    su_ = gu[:, :, I2:].float().abs().amax(dim=(1, 2)).clamp_min(1e-12) / FMAX; sd_ = sud_[:, 0] / su_
    sgu_ = torch.cat([sg_.expand(-1, I2), su_[:, None].expand(-1, I2)], dim=1)
    rq, _ = ref(x, lnw, rwT, (gq.float() * sgu_[:, None, :]).to(torch.bfloat16), (dq.float() * sd_[:, None, None]).to(torch.bfloat16))
    eq = (y - rq).abs(); print(f"FP8 kernel-vs-dequant-ref rel-fro {(eq.norm()/rq.norm()).item():.3e} ; quant-only (dequant-ref vs bf16-ref) rel-fro {((rq-r).norm()/r.norm()).item():.3e}", flush=True)
if LAYOUT == "fp8":
    sgu = gu.float().abs().amax(dim=1).clamp_min(1e-12) / FMAX; sdn = dn.float().abs().amax(dim=2).clamp_min(1e-12) / FMAX
    gdq = ((gu.float() / sgu[:, None, :]).to(torch.float8_e4m3fn).float() * sgu[:, None, :]).to(torch.bfloat16)
    ddq = ((dn.float() / sdn[:, :, None]).to(torch.float8_e4m3fn).float() * sdn[:, :, None]).to(torch.bfloat16)
    rq, _ = ref(x, lnw, rwT, gdq, ddq)
    eq = (y - rq).abs(); print(f"FP8 kernel-vs-dequant-ref rel-fro {(eq.norm()/rq.norm()).item():.3e} ; quant-only (dequant-ref vs bf16-ref) rel-fro {((rq-r).norm()/r.norm()).item():.3e}", flush=True)
err = (y - r).abs(); print(f"MAXERR T={T} E={E} I={I}: max abs {err.max().item():.3e}, ref max {r.abs().max().item():.3e}, rel(max) {err.max().item()/r.abs().max().item():.3e}, rel-fro {(err.norm()/r.norm()).item():.3e}", flush=True)
# timing: L layers chained, distinct weights
layers = [pd0]
for _ in range(L-1):
    g2, d2, r2, l2 = mkw(); a, b, c = relayout(g2, d2, r2)
    layers.append((a.to(dev), todev(b), c.to(dev), l2.to(dev)))
GEN = os.environ.get("GEN", "0") in ("1", "2"); GEN2 = os.environ.get("GEN", "0") == "2"   # synthesize inputs inside the graph (profiler capture feeds zeros)
def step(xd):
    for li, p in enumerate(layers):
        if GEN:
            xi = (0 if GEN2 else xd) + torch.sin(torch.arange(T * H, device=xd.device, dtype=torch.float32).reshape(T, H) * (0.37 + 0.01 * li)).to(xd.dtype)
            rwg = ((0 if GEN2 else p[2].float()) + torch.sin(torch.arange(H * E, device=xd.device, dtype=torch.float32).reshape(H, E) * (0.113 + 0.007 * li)) * 0.05).to(p[2].dtype)
            lnw_ = torch.ones(1, H, device=xd.device, dtype=p[3].dtype) if GEN2 else p[3]
            o = call(xi, p, rw=rwg, lnw=lnw_)
            xd = xd + ((o[0] if os.environ.get('NOSUM') == '1' else o.sum(0)) if o.dim() == 3 else o)
            continue
        o = call(xd, p)
        xd = xd + ((o[0] if os.environ.get('NOSUM') == '1' else o.sum(0)) if o.dim() == 3 else o)
    return xd
f = torch.compile(step, backend=get_compile_backend_name(), fullgraph=True)
xd = x.to(dev); f(xd).cpu()
for _ in range(3): f(xd).cpu()
dts = []
for _b in range(6):
    n = 20; t0 = time.time()
    for _ in range(n): f(xd).cpu()
    dts.append((time.time()-t0)/n)
dt = min(dts)
print(f"FUSEDRES {KMOD} E={E} I={I} T={T} L={L}: per-step {dt*1e3:.3f} ms, per-layer {dt*1e3/L:.4f} ms (minus 0.13ms launch: {(dt*1e3-0.13)/L:.4f})", flush=True)
