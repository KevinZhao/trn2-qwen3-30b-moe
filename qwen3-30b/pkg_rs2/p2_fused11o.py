# p2_fused11o.py (rs research, 2026-11-08) = p2_fused11n + interleaved processing order of union slots (T > 1 path).
# In 11n the static loop processes slots 0..NSL-1 in order; padding slots (>= this core's union count n) are always the
# LAST positions, so their (useless) PE/ACT work forms a serial tail after the last weight DMA (~56 us/layer at T=16).
# 11o keeps steps 0..R-1 = slots 0..R-1 (v11 ring-buffer redirect semantics unchanged) and processes the remaining slots
# in bit-reversed order, so padding steps are spread among real ones and their compute hides under weight DMAs.
# Ring buffer / DMA lead are indexed by processing STEP; expert id / affinity column by SLOT.  env P2_ORD = vdc | nat
_P2_ORD = __import__("os").environ.get("P2_ORD", "vdc")
def _order(NSL, R):
    head = list(range(min(R, NSL)))
    rest = list(range(len(head), NSL))
    m = len(rest)
    if _P2_ORD == "nat" or m <= 2:
        return head + rest
    nb = max(1, (m - 1).bit_length())
    key = lambda i: int(format(i, "0%db" % nb)[::-1], 2)
    return head + [rest[i] for i in sorted(range(m), key=key)]
# p2_fused11n.py = ii_fused11a with NATURAL (stock PR56) expert weight layouts instead of the blob:
#   gu [E, H, 2I] (gate | up on the last dim, == gate_up_proj_weight), dn [E, I, H] (== down_proj_weight).
# The SBUF image of an expert is bit-identical to the blob kernel's (gate/up rows h = p*H1 + h1 per partition p;
# folded down rows f = r*128 + p of dn.reshape(2I, H/2)), it is just fetched with 2 indirect DMAs per expert
# (gu: 1 x 12 KiB contiguous per partition; dn: 3 x 2 KiB per partition) instead of 1 blob DMA.  Compute is unchanged.
# Prefill / T>16 decode then use the stored tensors directly (no in-graph unblob copies).
# env P2_Q = alt (default: both DMAs of position j on the queue of j's parity, as ii_fused11a) | split (gu->sync, dn->scalar)
_P2_Q = __import__("os").environ.get("P2_Q", "alt")
# ii_fused11a.py = wsB_fused11.py (v11) + integration workaround for a full-model-graph device fault
# (SEQUENCER_ERROR_SUBTYPE_BAD_OPCODE / exec hang seen with v9 inside the 48-layer vLLM decode NEFF):
#  * scalar_offset of the per-expert weight DMA is given as an explicit access pattern ldc.ap([[NS,1],[1,1]], offset=j)
#    instead of a tensor slice ldc[0:1, j:j+1] (nkilib selective_expert_impl does the same, citing NKI-333).
#  * env II_DGE = hw (default, as v11) | sw | default : DGE mode of all kernel DMAs (trace-time switch).
import os as _ii_os
_II_DGE = _ii_os.environ.get("II_DGE", "hw")
_II_SOFF = _ii_os.environ.get("II_SOFF", "ap")
# Workstream B: fused Qwen3-MoE decode layer (BF16), v11 = v9 + fix for the stale-SBUF NaN bug.
# v11 fix: in v9, padding union positions j (union entries beyond this core's count n) skipped their weight DMA
# and computed on whatever ring buffer wbs[j % R] held.  If n < R, buffers n..R-1 were never written in this
# invocation, so they held stale SBUF from earlier kernels (NaN/Inf allowed); NaN*0 then poisoned the shared PSUM
# accumulator -> whole [T,H] output NaN (dup tokens, parallel sampling, overlapping routing, union < 2R).
# Now positions 1..min(R,NSL)-1 that are padding DMA the core's first union expert (always valid: union >= 8)
# instead of skipping, so every ring buffer holds finite weights of a real expert before it is first read;
# padding positions >= R keep oob-skip (their buffer already holds finite weights from position j-R or earlier).
# Affinity of padding positions stays exactly 0 (taken from lsf = -1, not from the redirected DMA id), and with
# finite weights the padding contribution is exactly 0.  Extra cost only when n < R (<= R-n extra expert loads,
# bounded by the cost of an n = R invocation); zero extra DMA in the common case.
# --- v9 notes ---
# Workstream B: fused Qwen3-MoE decode layer (BF16), v9 (= v8 routing + blob weights + PE software pipeline):
# one DMA per expert from a re-laid-out blob [E, 128, H1*2I + NR*H/2] (gate/up rows h=p*H1+h1, then the down
# projection 'folded' to 128-row chunks: row f = 2i + b holds W_down[i, b*H/2:(b+1)*H/2]).
# v8 notes: any T <= 16. Each routed expert is computed ONCE for the
# union of the T tokens' top-8 sets (tokens that did not pick it get affinity 0). The compacted union list is
# processed by a STATIC unrolled loop of NSL = min(E/2, 4T) positions per core (no dynamic-loop barriers);
# padding positions use an out-of-range expert id so their weight DMAs are skipped (oob_mode=skip) and their
# affinity is 0 (v9 BUG: 'finite stale weights' is not guaranteed for ring buffers not loaded in this call; see v11 fix).
#   post-attn RMSNorm -> router (bf16 mm, fp32 acc) -> per-token top8 -> softmax over top8 (== softmax + renorm,
#   Qwen3 norm_topk_prob=True) -> union of selected experts (nonzero_with_count) -> SiLU-GLU experts -> weighted sum
# The union list is split between the 2 physical cores (slot parity); each core loops over groups of G slots
# (G static buffers -> DMA of the next group's slot u overlaps compute of slots > u).
# Output: [2, T, H] partials (one per physical core); caller sums dim 0 (fuses with the TP all-reduce/residual).
# Layout: hidden index h = p*H1 + h1; natural HBM layouts: rw [H, E], gu [E, H, 2I] (gate | up), dn [E, I, H].
import nki, nki.isa as nisa, nki.language as nl
P = 128
BANK = 512
CT = 32

def _hw(eng):
    if _II_DGE == "sw":
        return dict(dge_mode=nisa.dge_mode.swdge)
    if _II_DGE == "default":
        return {}
    return dict(dge_mode=nisa.dge_mode.hwdge, engine=eng)

def _ct(q):
    return dict(tile_position=(0, CT * q), tile_size=(128, CT))

@nki.jit
def q3moe_fused(hidden, ln_w, rw, gu, dn, eps=1e-6, R=6, L=4, nsl=0, I=192, lagd=2, debug=False, seq=False, redirect=True):
    T, H = hidden.shape
    E, _Hg, GU = gu.shape
    I = GU // 2
    H1 = H // P
    HB = H // 2
    NR = GU // P                      # folded down chunks (128 rows each)
    GBL = H1 * GU                     # gate/up part of the blob (per partition)
    NQ = 4
    QW = HB // NQ                     # 256 output columns per column tile
    assert _Hg == H and tuple(dn.shape) == (E, I, H) and T <= 16
    NS = E // 2                       # max union slots per core
    NSL = nsl if nsl else min(NS, 4 * T)
    pid = nl.program_id(axis=0)
    TQ = T + (T % 2)
    f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
    out = nl.ndarray((2, T, H), dtype=hidden.dtype, buffer=nl.shared_hbm)

    gps = [nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum, name=f"gps{b}") for b in range(2)]
    tps = [nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum, name=f"tps{b}") for b in range(2)]
    acc = nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum, name="acc")
    mps = nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum, name="mps")
    rps = nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum, name="rps")
    sps = nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum, name="sps")

    # ---- loads + constants ----
    xp = nl.ndarray((P, T, H1), dtype=hidden.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xp[0:P, 0:T, 0:H1], src=hidden.ap(pattern=[[H1, P], [H, T], [1, H1]], offset=0), **_hw(nisa.engine.sync))
    gam = nl.ndarray((P, H1), dtype=ln_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gam[0:P, 0:H1], src=ln_w.ap(pattern=[[H1, P], [1, H1]], offset=0), **_hw(nisa.engine.scalar))
    rw_sb = nl.ndarray((P, H1, E), dtype=rw.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw_sb[0:P, 0:H1, 0:E], src=rw.ap(pattern=[[H1 * E, P], [E, H1], [1, E]], offset=0), **_hw(nisa.engine.scalar))
    ones = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    zst = nl.ndarray((P, TQ), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(zst, 0.0)
    zmv = nl.ndarray((P, BANK), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(zmv, 0.0)
    # sel[k, t] = 1 if k % 32 == t ; rep[t, k] = sel[k, t]
    sio = nl.ndarray((P, TQ), dtype=i32, buffer=nl.sbuf)
    nisa.iota(sio[0:P, 0:TQ], pattern=[[-1, TQ]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=sio[0:P, 0:TQ], data=sio[0:P, 0:TQ], op0=nl.bitwise_and, operand0=CT - 1)
    sel = nl.ndarray((P, TQ), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sel[0:P, 0:TQ], data=sio[0:P, 0:TQ], op0=nl.equal, operand0=0.0)
    selb = nl.ndarray((P, TQ), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=selb[0:P, 0:TQ], src=sel[0:P, 0:TQ])
    rio = nl.ndarray((P, P), dtype=i32, buffer=nl.sbuf)
    nisa.iota(rio[0:T, 0:P], pattern=[[1, P]], offset=0, channel_multiplier=-1)
    nisa.tensor_scalar(dst=rio[0:T, 0:P], data=rio[0:T, 0:P], op0=nl.bitwise_and, operand0=CT - 1)
    rep = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rep[0:T, 0:P], data=rio[0:T, 0:P], op0=nl.equal, operand0=0.0)
    pio = nl.ndarray((P, NS), dtype=f32, buffer=nl.sbuf)       # pio[p, u] = p
    nisa.iota(pio[0:P, 0:NS], pattern=[[0, NS]], offset=0, channel_multiplier=1)
    nisa.memset(rps[0:P, 0:BANK], 0.0)
    mio = nl.ndarray((P, 2), dtype=i32, buffer=nl.sbuf)       # msk2[p, b] = (p % 2 == b)
    nisa.iota(mio[0:P, 0:2], pattern=[[-1, 2]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=mio[0:P, 0:2], data=mio[0:P, 0:2], op0=nl.bitwise_and, operand0=1)
    msk2 = nl.ndarray((P, 2), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=msk2[0:P, 0:2], data=mio[0:P, 0:2], op0=nl.equal, operand0=0.0)
    for b in range(2):
        nisa.memset(gps[b][0:P, 0:BANK], 0.0)
    pert = nl.ndarray((P, E), dtype=f32, buffer=nl.sbuf)     # tie-break (zero/padded tokens): +e * 1e-7
    nisa.iota(pert[0:P, 0:E], pattern=[[1, E]], offset=0, channel_multiplier=0)
    nisa.tensor_scalar(dst=pert[0:P, 0:E], data=pert[0:P, 0:E], op0=nl.multiply, operand0=1e-7)

    # ---- RMSNorm over 128 partitions ----
    sq = nl.ndarray((P, T, H1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(sq[0:P, 0:T, 0:H1], xp[0:P, 0:T, 0:H1], xp[0:P, 0:T, 0:H1], nl.multiply)
    ssp = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ssp[0:P, 0:T], op=nl.add, data=sq[0:P, 0:T, 0:H1], axis=2)
    nisa.nc_matmul(dst=mps[0:P, 0:T], stationary=ones[0:P, 0:P], moving=ssp[0:P, 0:T], accumulate=False)
    rr = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rr[0:P, 0:T], data=mps[0:P, 0:T], op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=eps)
    nisa.tensor_scalar(dst=rr[0:P, 0:T], data=rr[0:P, 0:T], op0=nl.rsqrt, operand0=0.0)
    xn = nl.ndarray((P, T, H1), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(xn[0:P, 0:T, 0:H1], xp[0:P, 0:T, 0:H1], rr[0:P, 0:T].expand_dim(2).broadcast(2, H1), nl.multiply)
    xT = nl.ndarray((P, H1, T), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(xT.permute((0, 2, 1))[0:P, 0:T, 0:H1], xn[0:P, 0:T, 0:H1],
                       gam[0:P, 0:H1].expand_dim(1).broadcast(1, T), nl.multiply)

    # ---- router logits [T, E] ----
    for h1 in range(H1):
        q = h1 % 4
        nisa.nc_matmul(dst=rps[CT * q:CT * q + T, 0:E], stationary=xT[0:P, h1, 0:T], moving=rw_sb[0:P, h1, 0:E],
                       accumulate=(h1 >= 4), **_ct(q))
    rsb = nl.ndarray((P, E), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rsb[0:P, 0:E], src=rps[0:P, 0:E])
    nisa.nc_matmul(dst=mps[0:T, 0:E], stationary=sel[0:P, 0:T], moving=rsb[0:P, 0:E], accumulate=False)
    lg = nl.ndarray((P, E), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(lg[0:T, 0:E], mps[0:T, 0:E], pert[0:T, 0:E], nl.add)

    if T == 1:
        # T == 1: the union is just this token's top-8 (distinct experts) -> slots pid, pid+2, pid+4, pid+6
        top8 = nl.ndarray((P, 8), dtype=f32, buffer=nl.sbuf)
        nisa.max8(dst=top8[0:1, 0:8], src=lg[0:1, 0:E])
        idx = nl.ndarray((P, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=idx[0:1, 0:8], data=lg[0:1, 0:E], vals=top8[0:1, 0:8])
        ldc = nl.ndarray((P, NS), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ldc[0:1, 0:NSL], src=idx[0:1, pid:8:2])
        nmx = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nmx[0:1, 0:1], data=top8[0:1, 0:1], op0=nl.multiply, operand0=-1.0)
        ex8 = nl.ndarray((P, 8), dtype=f32, buffer=nl.sbuf)
        es = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=ex8[0:1, 0:8], op=nl.exp, data=top8[0:1, 0:8], bias=nmx[0:1, 0:1], reduce_op=nl.add,
                        reduce_res=es[0:1, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        ers = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=ers[0:1, 0:1], data=es[0:1, 0:1])
        a8 = nl.ndarray((P, 8), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a8[0:1, 0:8], data=ex8[0:1, 0:8], op0=nl.multiply, operand0=ers[0:1, 0:1])
        nisa.nc_matmul(dst=sps[0:P, 0:8], stationary=ones[0:1, 0:P], moving=a8[0:1, 0:8], accumulate=False)
        sa = nl.ndarray((P, NS), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sa[0:P, 0:NSL], src=sps[0:P, pid:8:2])
    else:
        # ---- top8 mask + softmax over top8 -> dense affinities A[t, e] ----
        top8 = nl.ndarray((P, 8), dtype=f32, buffer=nl.sbuf)
        nisa.max8(dst=top8[0:T, 0:8], src=lg[0:T, 0:E])
        msk = nl.ndarray((P, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=msk[0:T, 0:E], data=lg[0:T, 0:E], op0=nl.greater_equal, operand0=top8[0:T, 7:8])
        # union of experts over tokens -> row on partition 0
        nisa.nc_matmul(dst=mps[0:1, 0:E], stationary=ones[0:T, 0:1], moving=msk[0:T, 0:E], accumulate=False)
        used = nl.ndarray((P, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=used[0:1, 0:E], src=mps[0:1, 0:E])
        lst = nl.ndarray((P, E + 1), dtype=i32, buffer=nl.sbuf)
        nisa.nonzero_with_count(dst=lst[0:1, 0:E + 1], src=used[0:1, 0:E], index_offset=0, padding_val=-1)
        # this core's slots: lst[pid::2][:NSL]; padding (-1) -> E (out of range -> DMA skipped)
        lsf = nl.ndarray((P, NS), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lsf[0:1, 0:NSL], src=lst[0:1, pid:pid + 2 * NSL - 1:2])
        pad = nl.ndarray((P, NS), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pad[0:1, 0:NSL], data=lsf[0:1, 0:NSL], op0=nl.less, operand0=0.0, op1=nl.multiply, operand1=float(E + 1))
        ldf = nl.ndarray((P, NS), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(ldf[0:1, 0:NSL], lsf[0:1, 0:NSL], pad[0:1, 0:NSL], nl.add)
        # v11: positions 1..NR0-1 never skip -> padding loads slot 0's expert (ldf = -1 + (lsf0 + 1) = lsf0)
        NR0 = min(R, NSL)
        if NR0 > 1 and redirect:
            l0p = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=l0p[0:1, 0:1], data=lsf[0:1, 0:1], op0=nl.add, operand0=1.0)
            nisa.tensor_scalar(dst=pad[0:1, 1:NR0], data=lsf[0:1, 1:NR0], op0=nl.less, operand0=0.0,
                               op1=nl.multiply, operand1=l0p[0:1, 0:1])
            nisa.tensor_tensor(ldf[0:1, 1:NR0], lsf[0:1, 1:NR0], pad[0:1, 1:NR0], nl.add)
        ldc = nl.ndarray((P, NS), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ldc[0:1, 0:NSL], src=ldf[0:1, 0:NSL])

        nmx = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nmx[0:T, 0:1], data=top8[0:T, 0:1], op0=nl.multiply, operand0=-1.0)
        ex8 = nl.ndarray((P, 8), dtype=f32, buffer=nl.sbuf)
        es = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=ex8[0:T, 0:8], op=nl.exp, data=top8[0:T, 0:8], bias=nmx[0:T, 0:1], reduce_op=nl.add,
                        reduce_res=es[0:T, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        ers = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=ers[0:T, 0:1], data=es[0:T, 0:1])
        exf = nl.ndarray((P, E), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=exf[0:T, 0:E], op=nl.exp, data=lg[0:T, 0:E], bias=nmx[0:T, 0:1])
        aff = nl.ndarray((P, E), dtype=f32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=aff[0:T, 0:E], data=exf[0:T, 0:E], op0=nl.multiply, operand0=ers[0:T, 0:1],
                                  op1=nl.multiply, operand1=msk[0:T, 0:E])
        # SA[32q + t, u] = A[t, slot_u]  (per-partition scalars used to build the per-slot affinity selectors)
        nisa.nc_matmul(dst=sps[0:E, 0:P], stationary=aff[0:T, 0:E], moving=rep[0:T, 0:P], accumulate=False)
        ats = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ats[0:E, 0:P], src=sps[0:E, 0:P])
        nisa.nc_matmul(dst=mps[0:P, 0:NSL], stationary=ones[0:1, 0:P], moving=lsf[0:1, 0:NSL], accumulate=False)
        oh = nl.ndarray((P, NS), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(oh[0:E, 0:NSL], mps[0:E, 0:NSL], pio[0:E, 0:NSL], nl.equal)
        nisa.nc_matmul(dst=sps[0:P, P:P + NSL], stationary=ats[0:E, 0:P], moving=oh[0:E, 0:NSL], accumulate=False)
        sa = nl.ndarray((P, NS), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sa[0:P, 0:NSL], src=sps[0:P, P:P + NSL])

    # ---- expert loop (static, software pipelined in program order) ----
    # position j: blob ring buffer j % R; PE order per step j: GU(j), SEL(j-1), DN(j-lagd); DMA lead L <= R - lagd
    wgs = [nl.ndarray((P, GBL), dtype=gu.dtype, buffer=nl.sbuf, name=f"wg{u}") for u in range(R)]
    wds = [nl.ndarray((P, NR, HB), dtype=dn.dtype, buffer=nl.sbuf, name=f"wd{u}") for u in range(R)]
    SKIP = dict(oob_mode=nisa.oob_mode.skip)
    def _soff(j):
        return ldc.ap(pattern=[[NS, 1], [1, 1]], offset=j) if _II_SOFF == "ap" else ldc[0:1, j:j + 1]
    seq = seq or T == 1          # T=1: only 4 positions per core, latency favours strict per-position order
    ORD = list(range(NSL)) if seq else _order(NSL, R)
    def _issue(j):
        sl = ORD[j]
        e0 = nisa.engine.sync if j % 2 == 0 else nisa.engine.scalar
        e1 = e0 if _P2_Q == "alt" else nisa.engine.scalar
        if _P2_Q != "alt":
            e0 = nisa.engine.sync
        nisa.dma_copy(dst=wgs[j % R][0:P, 0:GBL],
                      src=gu.ap(pattern=[[GBL, P], [1, GBL]], offset=0, scalar_offset=_soff(sl), indirect_dim=0),
                      **SKIP, **_hw(e0))
        nisa.dma_copy(dst=wds[j % R][0:P, 0:NR, 0:HB],
                      src=dn.ap(pattern=[[HB, P], [P * HB, NR], [1, HB]], offset=0, scalar_offset=_soff(sl), indirect_dim=0),
                      **SKIP, **_hw(e1))
    gsbs, Ss = {}, {}
    def _gu(j):
        w = wgs[j % R]
        g_ps = gps[j % 2]
        for h1 in range(H1):
            q = h1 % 4
            nisa.nc_matmul(dst=g_ps[CT * q:CT * q + T, 0:GU], stationary=xT[0:P, h1, 0:T], moving=w[0:P, h1 * GU:(h1 + 1) * GU],
                           accumulate=(h1 >= 4), **_ct(q))
        gsb = nl.ndarray((P, GU, 2), dtype=bf16, buffer=nl.sbuf)       # each column duplicated (folded-down rows)
        nisa.tensor_copy(dst=gsb[0:P, 0:GU, 0:2], src=g_ps[0:P, 0:GU].expand_dim(2).broadcast(2, 2), engine=nisa.engine.scalar)
        gsbs[j] = gsb
    def _sel(j):
        gsb = gsbs.pop(j)
        sela = nl.ndarray((P, TQ), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sela[0:P, 0:T], data=sel[0:P, 0:T], op0=nl.multiply, operand0=sa[0:P, ORD[j]:ORD[j] + 1])
        gT = tps[j % 2][0:P, 0:2 * NR * TQ].reshape((P, 2 * NR, TQ))
        for r in range(NR):
            for g in range(2):     # g=0 gate, g=1 up (affinity folded into the up selector)
                c0 = g * I + 64 * r
                nisa.nc_matmul(dst=gT[0:P, g * NR + r, 0:T], stationary=gsb[0:P, c0:c0 + 64, 0:2].flatten_dims(1, 2),
                               moving=(selb if g == 0 else sela)[0:P, 0:T], accumulate=False)
        sl = nl.ndarray((P, NR, T), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=sl[0:P, 0:NR, 0:T], op=nl.silu, data=gT[0:P, 0:NR, 0:T])
        hm = nl.ndarray((P, NR, T), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(hm[0:P, 0:NR, 0:T], sl[0:P, 0:NR, 0:T], gT[0:P, NR:2 * NR, 0:T], nl.multiply)
        S = nl.ndarray((P, NR, T, 2), dtype=bf16, buffer=nl.sbuf)
        for b in range(2):
            nisa.tensor_scalar(dst=S[0:P, 0:NR, 0:T, b], data=hm[0:P, 0:NR, 0:T], op0=nl.multiply, operand0=msk2[0:P, b:b + 1])
        Ss[j] = S
    def _dn(j):
        S = Ss.pop(j)
        w = wds[j % R]
        for r in range(NR):
            for q in range(NQ):
                nisa.nc_matmul(dst=acc[CT * q:CT * q + 2 * T, 0:QW], stationary=S[0:P, r, 0:T, 0:2].flatten_dims(1, 2),
                               moving=w[0:P, r, q * QW:(q + 1) * QW],
                               accumulate=(j > 0 or r > 0), **_ct(q))
    for j in range(min(L, NSL)):
        _issue(j)
    if seq:
        for j in range(NSL):
            if j + L < NSL:
                _issue(j + L)
            _gu(j); _sel(j); _dn(j)
    assert L <= R - lagd
    for j in range(0 if seq else NSL + lagd):
        if j < NSL:
            _gu(j)
        if 1 <= j <= NSL:
            _sel(j - 1)
        if j >= lagd and j - lagd < NSL:
            _dn(j - lagd)
        if j + L < NSL:          # after this step's DN: the buffer's last reader (DN(j+L-R)) is already emitted
            _issue(j + L)

    # ---- write this core's partial: row 32q + 2t + b holds out[t, b*HB + q*QW : +QW] ----
    o_sb = nl.ndarray((P, QW), dtype=hidden.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o_sb[0:P, 0:QW], src=acc[0:P, 0:QW])
    for q in range(NQ):
        nisa.dma_copy(dst=out.ap(pattern=[[H, T], [HB, 2], [1, QW]], offset=pid * T * H + q * QW),
                      src=o_sb[CT * q:CT * q + 2 * T, 0:QW], **_hw(nisa.engine.sync if q % 2 == 0 else nisa.engine.scalar))
    if debug:
        dbg = nl.ndarray((2, P, QW), dtype=f32, buffer=nl.shared_hbm)
        dsb = nl.ndarray((P, QW), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dsb[0:P, 0:QW], src=acc[0:P, 0:QW])
        nisa.dma_copy(dst=dbg[pid, 0:P, 0:QW], src=dsb[0:P, 0:QW])
        return out, dbg
    return out
