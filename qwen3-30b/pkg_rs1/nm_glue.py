# nm_glue: on-device glue for the nm_k5 prefill-MoE kernel (pure torch around vllm_neuron's build_blockwise_mapping).
#   tables: stock blockwise mapping with block_size=128 (items = blocks, expert-major); the n_real real blocks are split
#   into 2R contiguous lanes of Lr = ceil(n_real / 2R) blocks; lane c*R + l -> core c, lane l, item slot k = step*R + l.
#   combine: out[t] = sum_j Y[pos[t, j]] (pos = Y row of token t for its j-th routed expert)
import math, os, importlib, torch
import vllm_neuron.functional as NF
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
def _imp(name):
    try:
        return importlib.import_module("." + name, __package__) if __package__ else importlib.import_module(name)
    except ImportError:
        return importlib.import_module(name)
_KM = _imp(os.environ.get("NM_KMOD", "nm_k9"))
_KW = eval(os.environ.get("NM_KW", "{'YQS': 1}"))
_COMB = os.environ.get("NM_COMB", "0")
_LEAN = os.environ.get("NM_LEAN", "1") == "1"; _CAP = os.environ.get("NM_CAP", "0") == "1"   # NM_CAP=1: item bound also <= E*T/128
_AFF_IN_KERNEL = os.environ.get("NM_KMOD", "nm_k9") in ("nm_k8", "nm_k9")
_CB = _imp("nm_comb")
class OneRankGroup:
    world_size = 1
    def __init__(self, dev): self.rank_in_group_tensor = torch.zeros(1, dtype=torch.int32, device=dev)
def nm_num_items(N, R):
    return R * ((((N + 1) // 2) + R - 1) // R)
def nm_tables(expert_affinities, padding_mask, group, tp_degree=1, B=128, R=4, K=8, need_aff=False):
    T, E = expert_affinities.shape
    dev = expert_affinities.device
    eam, tpid, b2e, cond = NF.build_blockwise_mapping(expert_affinities=expert_affinities, num_local_experts=E,
        num_experts_per_token=K, block_size=B, moe_group=group, tp_degree=tp_degree, padding_mask=padding_mask)
    N = b2e.shape[0]; NI = nm_num_items(N, R)
    n_real = cond.to(torch.int32).sum().reshape(1)
    Lr = torch.div(n_real + (2 * R - 1), 2 * R, rounding_mode="floor")
    k = torch.arange(NI, device=dev, dtype=torch.int32); lane = k % R; step = torch.div(k, R, rounding_mode="floor")
    c = torch.arange(2, device=dev, dtype=torch.int32).unsqueeze(1)
    g = (c * R + lane.unsqueeze(0)) * Lr + step.unsqueeze(0)                     # [2, NI]
    valid = (step.unsqueeze(0) < Lr) & (g < n_real)
    gi = torch.where(valid, g, torch.full_like(g, N)).to(torch.int64)
    tp_ext = torch.cat([tpid.view(N, B), torch.full((1, B), -1, dtype=tpid.dtype, device=dev)], 0)
    e_ext = torch.cat([b2e.view(N), torch.full((1,), E, dtype=b2e.dtype, device=dev)], 0)
    tokb = tp_ext.index_select(0, gi.reshape(-1)).view(2, NI, B)
    ek = e_ext.index_select(0, gi.reshape(-1)).view(2, NI)
    eprev = e_ext.index_select(0, torch.clamp(gi - 1, min=0).reshape(-1)).view(2, NI)
    same = (step.unsqueeze(0) > 0) & (eprev == ek)
    lid = torch.where(valid & ~same, ek, torch.full_like(ek, E))
    ydst = torch.where(valid, c * NI + k.unsqueeze(0), torch.full_like(g, 2 * NI))
    meta = torch.cat([lid.to(torch.int32), ydst.to(torch.int32)], 1).contiguous()
    tok = tokb.transpose(1, 2).contiguous().to(torch.int32)
    mask = (expert_affinities != 0).to(torch.float32)
    if _KM.__name__.endswith("nm_k9"):
        eid = ek.to(torch.float32).unsqueeze(1).expand(2, B, NI).contiguous()
        triu = torch.triu(torch.ones(E, E, device=dev, dtype=torch.float32))
        return tok, meta, (mask @ triu - 1.0).contiguous(), None, eid
    # combine positions without scatter: rank of token t in expert e's (ascending) token list via cumsum
    mask = (expert_affinities != 0).to(torch.float32)
    cnt = mask.sum(0)                                                            # [E]
    bpe = torch.ceil(cnt / B)
    bb = torch.cumsum(bpe, 0) - bpe                                              # blocks before expert e
    cum = NF_cumsum(mask)                                                        # [T, E] inclusive
    ti = torch.topk(expert_affinities, K, dim=-1).indices                        # [T, K]
    oh = (ti.unsqueeze(2) == torch.arange(E, device=dev).view(1, 1, E)).to(torch.float32)   # [T, K, E]
    r = (oh * (cum - 1.0).unsqueeze(1)).sum(-1)                                  # rank in expert list
    gb = (oh * bb.view(1, 1, E)).sum(-1) + torch.floor(r / B)                     # block index g
    p_in = r - torch.floor(r / B) * B
    gq = gb.to(torch.int32)
    ch = torch.div(gq, Lr, rounding_mode="floor"); st = gq - ch * Lr
    slot = torch.div(ch, R, rounding_mode="floor") * NI + st * R + (ch % R)
    pos = (slot * B + p_in.to(torch.int32)).to(torch.int32)
    wts = torch.gather(expert_affinities, 1, ti)
    if padding_mask is not None:
        wts = wts * padding_mask.to(wts.dtype).unsqueeze(1)
    eid = ek.to(torch.float32).unsqueeze(1).expand(2, B, NI).contiguous()
    if _KM.__name__.endswith("nm_k9"):   # j-index table: rank of expert e among token t's experts (ascending e)
        triu = torch.triu(torch.ones(E, E, device=dev, dtype=torch.float32))
        jx = (mask @ triu - 1.0).contiguous()
        return tok, meta, jx, None, eid
    return tok, meta, pos, wts.to(torch.float32).contiguous(), eid
def NF_cumsum(m):
    from vllm_neuron.functional.moe.moe_blockwise import _cumsum_matmul
    return _cumsum_matmul(m)
def nm_tables2(expert_affinities, padding_mask, B=128, R=4, K=8):
    """lean tables (no indexed_flatten / collectives): per-expert token lists from find_nonzero_indices over ALL experts,
    items = (expert, 128-token block) in expert order, lanes as in nm_tables. Requires T % 128 == 0."""
    from nkilib.core.subkernels.find_nonzero_indices import find_nonzero_indices
    T, E = expert_affinities.shape
    dev = expert_affinities.device
    mask = (expert_affinities != 0).to(torch.float32)
    indices, cnt = wrap_nki(find_nonzero_indices)[2](input_tensor=mask, col_start_id=torch.zeros(1, dtype=torch.int32, device=dev),
                                                    n_cols=E, chunk_size=min(T, 16384))
    NBE = T // B                                                                  # block capacity per expert
    N = min(math.ceil((T * K - (E - 1)) / B) + E - 1, T * K, E * NBE if _CAP else T * K); NI = nm_num_items(N, R)   # each expert <= T/B blocks
    bpe = torch.div(cnt.to(torch.int32) + (B - 1), B, rounding_mode="floor")      # [E]
    triu = torch.triu(torch.ones(E, E, device=dev, dtype=torch.float32))
    cum = (bpe.to(torch.float32).unsqueeze(0) @ triu).squeeze(0).to(torch.int32)  # inclusive cumsum [E]
    n_real = cum[E - 1:E]
    Lr = torch.div(n_real + (2 * R - 1), 2 * R, rounding_mode="floor")
    k = torch.arange(NI, device=dev, dtype=torch.int32); lane = k % R; step = torch.div(k, R, rounding_mode="floor")
    c = torch.arange(2, device=dev, dtype=torch.int32).unsqueeze(1)
    g = (c * R + lane.unsqueeze(0)) * Lr + step.unsqueeze(0)                     # [2, NI]
    valid = (step.unsqueeze(0) < Lr) & (g < n_real)
    def exp_of(gq):
        return (gq.unsqueeze(-1) >= cum.view(1, 1, E)).to(torch.int32).sum(-1)     # expert of global block
    ek = exp_of(g); ep = exp_of(g - 1)
    ekc = torch.clamp(ek, max=E - 1).to(torch.int64)
    start = (cum - bpe).index_select(0, ekc.reshape(-1)).view(2, NI)              # first global block of expert
    r = g - start
    row = torch.where(valid, ek * NBE + r, torch.full_like(g, E * NBE)).to(torch.int64)
    ind_ext = torch.cat([indices.view(E * NBE, B), torch.full((1, B), -1, dtype=indices.dtype, device=dev)], 0)
    tokb = ind_ext.index_select(0, row.reshape(-1)).view(2, NI, B)
    same = (step.unsqueeze(0) > 0) & (ep == ek)
    lid = torch.where(valid & ~same, ek, torch.full_like(ek, E))
    ydst = torch.where(valid, c * NI + k.unsqueeze(0), torch.full_like(g, 2 * NI))
    meta = torch.cat([lid.to(torch.int32), ydst.to(torch.int32)], 1).contiguous()
    tok = tokb.transpose(1, 2).contiguous().to(torch.int32)
    eid = torch.where(valid, ek, torch.full_like(ek, E)).to(torch.float32).unsqueeze(1).expand(2, B, NI).contiguous()
    jx = (mask @ triu - 1.0).contiguous()
    return tok, meta, jx, None, eid
def nm_moe_prefill(hidden, expert_affinities, gu, dn, padding_mask, group, tp_degree=1, R=4):
    """hidden [T,H] bf16 (normed), expert_affinities [T,E] fp32 (router output), gu [E,H,2I], dn [E,I,H] -> [T,H] bf16"""
    T, H = hidden.shape
    if _LEAN and hidden.shape[0] % 128 == 0 and _KM.__name__.endswith("nm_k9"):
        tok, meta, pos, wts, eid = nm_tables2(expert_affinities, padding_mask, R=R)
    else:
        tok, meta, pos, wts, eid = nm_tables(expert_affinities, padding_mask, group, tp_degree, R=R)
    if _KM.__name__.endswith("nm_k9"):
        ea = expert_affinities if padding_mask is None else expert_affinities * padding_mask.to(expert_affinities.dtype).unsqueeze(1)
        Yt = wrap_nki(_KM.nm_experts)[2](x=hidden, gu=gu, dn=dn, tok=tok, ea=ea.contiguous(), jx=pos, eid=eid, meta=meta, R=R, **_KW)
        if _COMB == "torch":
            return Yt.view(T, 8, H).to(torch.float32).sum(1).to(hidden.dtype)
        return wrap_nki(_CB.nm_combine)[2](Yf=Yt, pos=torch.empty(T, 8, 1, device=hidden.device), w=None, MODE=4)
    if _AFF_IN_KERNEL:
        ea = expert_affinities if padding_mask is None else expert_affinities * padding_mask.to(expert_affinities.dtype).unsqueeze(1)
        Y = wrap_nki(_KM.nm_experts)[2](x=hidden, gu=gu, dn=dn, tok=tok, ea=ea.contiguous(), eid=eid, meta=meta, R=R, **_KW)
        wts = None
    else:
        Y = wrap_nki(_KM.nm_experts)[2](x=hidden, gu=gu, dn=dn, tok=tok, meta=meta, R=R, **_KW)
    Yf = Y.reshape(-1, H)
    if _COMB == "torch":
        yy = Yf.index_select(0, pos.reshape(-1).to(torch.int64)).view(T, pos.shape[1], H).to(torch.float32)
        return (yy if wts is None else yy * wts.unsqueeze(2)).sum(1).to(hidden.dtype)
    return wrap_nki(_CB.nm_combine)[2](Yf=Yf, pos=pos, w=wts, MODE=int(os.environ.get('NM_CMODE', '3')), GS=int(os.environ.get('NM_CGS', '1')))
def nm_prefill_layer(m, hidden_states, expert_affinities, padding_mask):
    """NM_MOE=1 branch of Qwen3MoeExperts.forward_prefill (pure TP, no EP/cross-DP): nm_k9 experts + combine, then the
    same reduce_scatter back to SP layout as the stock path."""
    output = nm_moe_prefill(hidden_states, expert_affinities,
                            m._gate_up_w().reshape(m.num_local_experts, m.hidden_size, 2 * m.intermediate_size_per_rank),
                            m._down_w(), padding_mask, m.ep_tp_group, m.tp_degree)
    if m.moe_group.world_size > 1:
        output = m.moe_group.reduce_scatter(output, dim=0)
    return output
