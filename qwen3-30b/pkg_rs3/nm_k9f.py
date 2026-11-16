# nm_k9f = nm_k9 with FP8 (e4m3) expert weights gu/dn (rs research 2026-11-08): per-expert scalar scales; the gate scale is
# applied as the per-item SiLU activation scale (sgt [2, P, NI] fp32 = s_gate[expert of item]); scale(up)*scale(down) is folded
# into the dense affinities ea by the caller.
# nm_k9 = nm_k8 + Y rows SCATTERED straight to token-major Yt[t*K + j] (j = rank of the expert among token t's experts, gathered
# in-kernel from jx [T*E]), so the combine is a dense, contiguous K-way sum (no gather).
# nm_k8 = nm_k5 with the per-row routing affinity gathered IN-KERNEL from the dense affinities ea [T*E] (fp32)
# (index tok*E + e built on DVE from the tok table and a per-item expert-id table eid [2, P, NI]); no host/torch gather.
# nm_k5: prefill MoE experts kernel v5 = nm_k4 (static item loop, R weight lanes, oob-skipped padding) + explicit
# cross-item software pipelining of the PE program order:  step k emits  S3(k) S2(k+1) S1(k+2) S4(k)
#   S1 = x gather-transposes (PE) + PSUM->SBUF copies   S2 = gate/up matmuls (PE) + SiLU*up*affinity (ACT/DVE)
#   S3 = h transposes (PE) + copies                       S4 = down matmuls (PE) + copies + Y write
# so the PE never waits on the ACT/DVE result it has just produced (v4: ~2us bubble after every S2 and S3).
# x [T,H] bf16, gu [E,H,2I] (gate|up), dn [E,I,H]
# tok [2, P, NI] int32 (-1 pad) ; aff [2, P, NI] fp32 ; meta [2, 2*NI] int32: lid(k) (E = no load) then ydst(k) (2*NI = skip)
# out Y [2*NI, P, H] bf16
import nki, nki.isa as nisa, nki.language as nl
P = 128
@nki.jit
def nm_experts(x, gu, dn, tok, ea, jx, eid, meta, sgt, R=4, LA=3, XB=4, ROWT=0, DPS=2, YQS=1, GS=1, KT=8, WSPL=0):
    T, H = x.shape
    E, _H, GU = gu.shape
    I = GU // 2
    NI = tok.shape[2]
    H1 = H // P
    QW = 512
    NQ = H // QW
    I2 = I - P
    pid = nl.program_id(axis=0)
    f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
    Y = nl.ndarray((T * KT, H), dtype=x.dtype, buffer=nl.shared_hbm)
    mt = nl.ndarray((1, 2 * NI), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mt[0:1, 0:2 * NI], src=meta.ap(pattern=[[2 * NI, 1], [1, 2 * NI]], offset=pid * 2 * NI))
    tk = nl.ndarray((P, NI), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tk[0:P, 0:NI], src=tok.ap(pattern=[[NI, P], [1, NI]], offset=pid * P * NI))
    EE = ea.shape[1]
    eidt = nl.ndarray((P, NI), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eidt[0:P, 0:NI], src=eid.ap(pattern=[[NI, P], [1, NI]], offset=pid * P * NI))
    sgtt = nl.ndarray((P, NI), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sgtt[0:P, 0:NI], src=sgt.ap(pattern=[[NI, P], [1, NI]], offset=pid * P * NI))
    tkf = nl.ndarray((P, NI), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tkf[0:P, 0:NI], src=tk[0:P, 0:NI])
    aidx = nl.ndarray((P, NI), dtype=i32, buffer=nl.sbuf)       # tok*E + e (negative for padding rows -> oob skip)
    nisa.scalar_tensor_tensor(dst=aidx[0:P, 0:NI], data=tkf[0:P, 0:NI], op0=nl.multiply, operand0=float(EE), op1=nl.add, operand1=eidt[0:P, 0:NI])
    af = nl.ndarray((P, NI), dtype=f32, buffer=nl.sbuf)
    nisa.memset(af[0:P, 0:NI], 0.0)
    eaf = ea.reshape((ea.shape[0] * EE, 1))
    jxf = jx.reshape((jx.shape[0] * EE, 1))
    jf = nl.ndarray((P, NI), dtype=f32, buffer=nl.sbuf)
    nisa.memset(jf[0:P, 0:NI], 0.0)
    ydx = nl.ndarray((P, NI), dtype=i32, buffer=nl.sbuf)          # tok*K + j  (negative for padding rows -> scatter skipped)
    def issue_a(k):
        nisa.dma_copy(dst=af[0:P, k:k + 1], src=eaf.ap(pattern=[[1, P], [1, 1]], offset=0, vector_offset=aidx[0:P, k:k + 1], indirect_dim=0),
                      oob_mode=nisa.oob_mode.skip)
        nisa.dma_copy(dst=jf[0:P, k:k + 1], src=jxf.ap(pattern=[[1, P], [1, 1]], offset=0, vector_offset=aidx[0:P, k:k + 1], indirect_dim=0),
                      oob_mode=nisa.oob_mode.skip)
        nisa.scalar_tensor_tensor(dst=ydx[0:P, k:k + 1], data=tkf[0:P, k:k + 1], op0=nl.multiply, operand0=float(KT), op1=nl.add, operand1=jf[0:P, k:k + 1])
    wgs = [nl.ndarray((P, H1 * GU), dtype=gu.dtype, buffer=nl.sbuf, name=f"wg{u}") for u in range(R)]
    wd1 = [nl.ndarray((P, H), dtype=dn.dtype, buffer=nl.sbuf, name=f"wd1_{u}") for u in range(R)]
    wd2 = [nl.ndarray((P, H), dtype=dn.dtype, buffer=nl.sbuf, name=f"wd2_{u}") for u in range(R)]
    xg = [nl.ndarray((P, H), dtype=x.dtype, buffer=nl.sbuf, name=f"xg{u}") for u in range(XB)]
    SKIP = dict(oob_mode=nisa.oob_mode.skip)
    WQ = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    YQ = [dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar), dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync),
          dict(dge_mode=nisa.dge_mode.swdge), None][YQS]
    def issue_w(k):
        b = k % R; so = mt.ap(pattern=[[2 * NI, 1], [1, 1]], offset=k)
        nisa.dma_copy(dst=wgs[b][0:P, 0:H1 * GU], src=gu.ap(pattern=[[H1 * GU, P], [1, H1 * GU]], offset=0, scalar_offset=so, indirect_dim=0), **SKIP, **WQ)
        WQ2 = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar) if WSPL else WQ
        nisa.dma_copy(dst=wd1[b][0:P, 0:H], src=dn.ap(pattern=[[H, P], [1, H]], offset=0, scalar_offset=so, indirect_dim=0), **SKIP, **WQ2)
        nisa.dma_copy(dst=wd2[b][0:I2, 0:H], src=dn.ap(pattern=[[H, I2], [1, H]], offset=P * H, scalar_offset=so, indirect_dim=0), **SKIP, **WQ2)
        if ROWT:
            nisa.dma_copy(dst=wd2[b][I2:2 * I2, 0:H], src=dn.ap(pattern=[[H, I2], [1, H]], offset=P * H, scalar_offset=so, indirect_dim=0), **SKIP, **WQ)
    def issue_x(k):
        issue_a(k)
        W = H // GS
        for c in range(GS):
            nisa.dma_copy(dst=xg[k % XB][0:P, c * W:(c + 1) * W], src=x.ap(pattern=[[H, P], [1, W]], offset=c * W, vector_offset=tk[0:P, k:k + 1], indirect_dim=0), **SKIP)
    st = {}
    def S1(k):
        xb = xg[k % XB]
        xT = nl.ndarray((P, H1, P), dtype=bf16, buffer=nl.sbuf)
        for g in range(H1 // 4):
            tp = nl.ndarray((P, 4 * P), dtype=bf16, buffer=nl.psum)
            for u in range(4):
                h1 = g * 4 + u
                nisa.nc_transpose(dst=tp[0:P, u * P:(u + 1) * P], data=xb[0:P, h1:H:H1])
            nisa.tensor_copy(dst=xT[0:P, g * 4:(g + 1) * 4, 0:P], src=tp[0:P, 0:4 * P].reshape((P, 4, P)),
                             engine=(nisa.engine.scalar if g % 2 == 0 else nisa.engine.vector))
        st[("xT", k)] = xT
    def S2(k):
        b = k % R; xT = st.pop(("xT", k))
        gps = nl.ndarray((P, GU), dtype=f32, buffer=nl.psum)
        for h1 in range(H1):
            nisa.nc_matmul(dst=gps[0:P, 0:GU], stationary=xT[0:P, h1, 0:P], moving=wgs[b][0:P, h1 * GU:(h1 + 1) * GU], accumulate=(h1 > 0))
        sg = nl.ndarray((P, I), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=sg[0:P, 0:I], op=nl.silu, data=gps[0:P, 0:I], scale=sgtt[0:P, k:k + 1])
        hb = nl.ndarray((P, I), dtype=bf16, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=hb[0:P, 0:I], data=sg[0:P, 0:I], op0=nl.multiply, operand0=af[0:P, k:k + 1],
                                  op1=nl.multiply, operand1=gps[0:P, I:GU])
        st[("hb", k)] = hb
    def S3(k):
        hb = st.pop(("hb", k))
        hps = nl.ndarray((P, 2 * P), dtype=bf16, buffer=nl.psum)
        nisa.nc_transpose(dst=hps[0:P, 0:P], data=hb[0:P, 0:P])
        nisa.nc_transpose(dst=hps[0:I2, P:2 * P], data=hb[0:P, P:I])
        hT = nl.ndarray((P, 2 * P), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hT[0:P, 0:P], src=hps[0:P, 0:P], engine=nisa.engine.vector)
        nisa.tensor_copy(dst=hT[0:I2, P:2 * P], src=hps[0:I2, P:2 * P], engine=nisa.engine.scalar)
        st[("hT", k)] = hT
    def S4(k):
        b = k % R; hT = st.pop(("hT", k))
        ysb = nl.ndarray((P, H), dtype=bf16, buffer=nl.sbuf)
        for q in range(NQ):
            dps = nl.ndarray((P, QW), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=dps[0:P, 0:QW], stationary=hT[0:P, 0:P], moving=wd1[b][0:P, q * QW:(q + 1) * QW], accumulate=False)
            nisa.nc_matmul(dst=dps[0:P, 0:QW], stationary=hT[0:I2, P:2 * P], moving=wd2[b][0:I2, q * QW:(q + 1) * QW], accumulate=True)
            nisa.tensor_copy(dst=ysb[0:P, q * QW:(q + 1) * QW], src=dps[0:P, 0:QW],
                             engine=(nisa.engine.scalar if q % 2 == 0 else nisa.engine.vector))
        nisa.dma_copy(dst=Y.ap(pattern=[[H, P], [1, H]], offset=0, vector_offset=ydx[0:P, k:k + 1], indirect_dim=0), src=ysb[0:P, 0:H], **SKIP)
    # prologue
    for k in range(min(R, NI)):
        issue_w(k)
    for k in range(min(LA, NI)):
        issue_x(k)
    S1(0)
    if NI > 1: S1(1)
    S2(0)
    for k in range(NI):
        if k + LA < NI:
            issue_x(k + LA)
        S3(k)
        if k + 1 < NI: S2(k + 1)
        if k + 2 < NI: S1(k + 2)
        S4(k)
        if k + R < NI:
            issue_w(k + R)      # lane buffer k % R: last reader S4(k) emitted
    return Y
