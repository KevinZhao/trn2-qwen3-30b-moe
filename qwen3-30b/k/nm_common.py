# shared: routing generation, blockwise mapping (TP1 replica of vllm_neuron build_blockwise_mapping), fp32 reference
import os, math, glob, torch
import torch.nn.functional as F
E, H, K = 128, 2048, 8
I = int(os.environ.get("I", "192"))
_REAL = None
def real_logits(layer):
    global _REAL
    if _REAL is None:
        fs = sorted(glob.glob("/data/cache/nm_route/seq*.pt"))
        _REAL = [torch.load(f)["logits"] for f in fs]          # each [L, S, E] fp16
    return torch.cat([r[layer] for r in _REAL], 0).float()     # [sum S, E]
def make_logits(T, route, layer=0, seed=0):
    g = torch.Generator().manual_seed(seed * 1000 + layer)
    if route == "real":
        lg = real_logits(layer % 48)
        if lg.shape[0] < T: lg = lg.repeat(math.ceil(T / lg.shape[0]), 1)
        return lg[:T].clone()
    if route == "uni": return torch.randn(T, E, generator=g)
    if route == "skew":   # popularity skew ~ real-ish
        b = torch.randn(E, generator=g) * float(os.environ.get("SKEW", "1.0"))
        return torch.randn(T, E, generator=g) + b
    if route == "same":   # every token picks the same 8 experts
        lg = torch.randn(T, E, generator=g) * 0.01; lg[:, 5:13] += 10.0; return lg
    raise ValueError(route)
def route_aff(logits):
    """softmax over E -> top8 -> renorm (Qwen3 norm_topk_prob) -> dense [T,E] fp32, topk idx/val"""
    p = torch.softmax(logits.float(), -1); tv, ti = p.topk(K, -1); tv = tv / tv.sum(-1, keepdim=True)
    dense = torch.zeros_like(p).scatter_(1, ti, tv)
    return dense, ti, tv
def num_blocks(T, B):
    tl = T * min(K, E); return min(math.ceil((tl - (E - 1)) / B) + E - 1, tl)
def mapping(dense, B, padding_mask=None, mask_pad_routing=False):
    """returns eam [T*E,1] fp32 (padding tokens zeroed), tpid [N*B] int32 (-1 pad), b2e [N] int32, N, n_real_blocks"""
    T = dense.shape[0]
    mask = (dense != 0)
    if padding_mask is not None and mask_pad_routing: mask = mask & padding_mask[:, None]
    eam = dense.clone()
    if padding_mask is not None: eam = eam * padding_mask[:, None].float()
    cnt = mask.sum(0); bpe = (cnt + B - 1) // B; N = num_blocks(T, B)
    cum = torch.cumsum(bpe, 0); excl = cum - bpe
    assert cum[-1] <= N, (cum[-1], N)
    b2e = torch.searchsorted(cum, torch.arange(N), right=True).clamp(max=E - 1).to(torch.int32)
    tp = torch.full((N * B,), -1, dtype=torch.int32)
    for e in range(E):
        toks = torch.nonzero(mask[:, e]).flatten()
        if toks.numel(): tp[excl[e] * B: excl[e] * B + toks.numel()] = toks.to(torch.int32)
    return eam.reshape(-1, 1).contiguous(), tp, b2e, N, int(cum[-1])
def ref_moe(x, dense, gu, dn):
    """fp32 reference. x [T,H] bf16, dense [T,E] affinities (already padding-masked), gu [E,H,2I], dn [E,I,H]"""
    xf = x.float(); out = torch.zeros(x.shape[0], H)
    for e in range(E):
        toks = torch.nonzero(dense[:, e]).flatten()
        if toks.numel() == 0: continue
        gup = xf[toks] @ gu[e].float(); h = F.silu(gup[:, :I]) * gup[:, I:]
        out[toks] += dense[toks, e:e + 1] * (h @ dn[e].float())
    return out
def mkw(seed):
    g = torch.Generator().manual_seed(seed)
    gu = (torch.randn(E, H, 2 * I, generator=g) * 0.02).to(torch.bfloat16)
    dn = (torch.randn(E, I, H, generator=g) * 0.02).to(torch.bfloat16)
    return gu, dn
