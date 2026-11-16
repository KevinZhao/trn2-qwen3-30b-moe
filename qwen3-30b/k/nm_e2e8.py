# per-layer prefill MoE (mapping + experts + combine) : stock (B=256 shard_on_block) vs nm. env T L ROUTE PADT IMPL CHECK
import os, time, math, torch
from nm_common import *
T = int(os.environ.get("T", "2048")); L = int(os.environ.get("L", "3")); ROUTE = os.environ.get("ROUTE", "real")
PADT = int(os.environ.get("PADT", str(T))); IMPL = os.environ.get("IMPL", "nm"); CHECK = os.environ.get("CHECK", "1") == "1"
import libtorch_neuronx_lite
import vllm_neuron.functional as NF
from vllm_neuron.envs import get_compile_backend_name
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
import nki.language as nl
import nm_glue
dev = "neuron:0"
G = nm_glue.OneRankGroup(dev)
def stock(x, ea, gu, dn, pm):
    eam, tp, b2e, cond = NF.build_blockwise_mapping(expert_affinities=ea, num_local_experts=E, num_experts_per_token=K, block_size=256,
                                                     moe_group=G, tp_degree=1, padding_mask=pm)
    return NF.moe_cte(implementation=MoECTEImplementation.shard_on_block, conditions=cond, hidden_states=x, expert_affinities_masked=eam,
        gate_up_proj_weight=gu.reshape(E, H, 2, I), down_proj_weight=dn, activation_function=ActFnType.SiLU, block_size=256,
        token_position_to_id=tp.to(torch.int32), block_to_expert=b2e.to(torch.int32), expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        skip_token=True, is_tensor_update_accumulating=True, compute_dtype=nl.bfloat16)
def nm(x, ea, gu, dn, pm):
    return nm_glue.nm_moe_prefill(x, ea, gu, dn, pm, G, 1)
FMAX = 240.0
def q8(gu, dn):
    I2 = gu.shape[2] // 2
    sg = gu[:, :, :I2].float().abs().amax(dim=(1, 2)).clamp_min(1e-12) / FMAX
    su = gu[:, :, I2:].float().abs().amax(dim=(1, 2)).clamp_min(1e-12) / FMAX
    sd = dn.float().abs().amax(dim=(1, 2)).clamp_min(1e-12) / FMAX
    s = torch.cat([sg[:, None].expand(-1, I2), su[:, None].expand(-1, I2)], dim=1)
    gq = (gu.float() / s[:, None, :]).to(torch.float8_e4m3fn); dq = (dn.float() / sd[:, None, None]).to(torch.float8_e4m3fn)
    gdq = (gq.float() * s[:, None, :]).to(torch.bfloat16); ddq = (dq.float() * sd[:, None, None]).to(torch.bfloat16)
    return gq, dq, sg[:, None].contiguous(), (su * sd)[:, None].contiguous(), gdq, ddq
def nm8(x, ea, gu, dn, sg, sud, pm):
    return nm_glue.nm_moe_prefill(x, ea, gu, dn, pm, G, 1, sg=sg, sud=sud)
def part(x, ea, gu, dn, pm):
    PT = os.environ.get("PART")
    z = torch.zeros_like(x)
    if PT == "map":
        eam, tp, b2e, cond = NF.build_blockwise_mapping(expert_affinities=ea, num_local_experts=E, num_experts_per_token=K, block_size=128,
                                                         moe_group=G, tp_degree=1, padding_mask=pm)
        return z + (tp.sum() + b2e.sum() + eam.sum()).to(x.dtype) * 0
    tok, meta, pos, wts, eid = nm_glue.nm_tables(ea, pm, G, 1)
    if PT == "k9":
        Yt = nm_glue.wrap_nki(nm_glue._KM.nm_experts)[2](x=x, gu=gu, dn=dn, tok=tok, ea=(ea * pm.float().unsqueeze(1)).contiguous(), jx=pos, eid=eid, meta=meta, R=4, **nm_glue._KW)
        return z + Yt[0:1, :].to(x.dtype) * 1e-30
    if PT == "tab9":
        return z + (tok.reshape(-1)[-1:].float() + meta.reshape(-1)[-1:].float() + pos.reshape(-1)[-1:] + eid.reshape(-1)[-1:]).to(x.dtype) * 1e-30
    if PT.startswith("t_"):
        v = {"t_tok": tok, "t_wts": wts, "t_meta": meta, "t_pos": pos}[PT]
        return z + v.reshape(-1)[-1:].to(torch.float32).to(x.dtype) * 1e-30
    Y = nm_glue.wrap_nki(nm_glue._KM.nm_experts)[2](x=x, gu=gu, dn=dn, tok=tok, ea=(ea * pm.float().unsqueeze(1)).contiguous(), eid=eid, meta=meta, R=4, **nm_glue._KW)
    if PT == "kernel":
        return z + Y[0, 0:1, :].to(x.dtype) * 0 + pos.sum().to(x.dtype) * 0
    raise ValueError(PT)
fn = part if IMPL == "part" else (nm if IMPL == "nm" else (nm8 if IMPL == "nm8" else stock))
pm_cpu = torch.arange(T) < PADT
layers = []
for li in range(L):
    lg = make_logits(T, ROUTE, layer=li * 5)
    p = torch.softmax(lg.float(), -1); tv, ti = p.topk(K, -1); tv = tv / tv.sum(-1, keepdim=True)
    ea = torch.zeros_like(p).scatter_(1, ti, tv)
    gu, dn = mkw(li)
    if IMPL == "nm8":
        gq, dq, sg, sud, gdq, ddq = q8(gu, dn); layers.append((ea, gq, dq, sg, sud, gdq, ddq))
    else:
        layers.append((ea, gu, dn))
x = torch.randn(T, H).to(torch.bfloat16)
pmd = pm_cpu.to(dev)
if os.environ.get("DUMP"):
    import numpy as np; D = os.environ["DUMP"]; os.makedirs(D, exist_ok=True)
    np.save(D + "/ea.npy", layers[0][0].numpy().astype(np.float32)); np.save(D + "/pm.npy", pm_cpu.numpy().astype(np.uint8))
dl = [tuple(t.to(dev) for t in (l[:5] if IMPL == "nm8" else l)) for l in layers]
if CHECK:
    f1 = torch.compile(lambda xd, *a: fn(xd, *a), backend=get_compile_backend_name(), fullgraph=True)
    t0 = time.time(); y = f1(x.to(dev), *dl[0], pmd).cpu().float(); print("first call s", round(time.time() - t0, 1), flush=True)
    if IMPL == "nm8":
        ea, gu, dn = layers[0][0], layers[0][5], layers[0][6]
    else:
        ea, gu, dn = layers[0]
    r = ref_moe(x, ea * pm_cpu[:, None].float(), gu, dn)
    e = y - r; ep = e[:PADT]
    print(f"MAXERR {IMPL} T={T} ROUTE={ROUTE} PADT={PADT}: rel-fro {(ep.norm()/r[:PADT].norm()).item():.3e} max abs {ep.abs().max().item():.3e} ref max {r.abs().max().item():.3e} nan {torch.isnan(y).any().item()} pad-rows absmax {y[PADT:].abs().max().item() if PADT < T else 0.0:.3e}", flush=True)
def step(xd, dls, pm):
    for l in dls:
        xd = xd + fn(xd, *l, pm) * 0.01
    return xd
f = torch.compile(step, backend=get_compile_backend_name(), fullgraph=True)
def tm(dls):
    xd = x.to(dev); f(xd, dls, pmd).cpu()
    for _ in range(3): f(xd, dls, pmd).cpu()
    dts = []
    for _ in range(5):
        n = 10; t0 = time.time()
        for _ in range(n): f(xd, dls, pmd).cpu()
        dts.append((time.time() - t0) / n)
    return min(dts)
dt = tm(dl); dth = tm(dl[:1])
print(f"NMRES e2e {IMPL} T={T} ROUTE={ROUTE} PADT={PADT} SLOPE per-layer {(dt-dth)/(L-1)*1e3:.4f} ms (L={L} {dt*1e3:.3f} ms, L=1 {dth*1e3:.3f} ms)", flush=True)
