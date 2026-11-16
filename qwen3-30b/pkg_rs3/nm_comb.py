# nm_comb: combine kernel out[t] = sum_j Yf[pos[t, j]]   Yf [R, H] bf16, pos [T, K] int32 -> out [T, H] bf16 (fp32 accumulate)
# MODE 0: K gathers into bf16 tiles + fp32 adds on DVE/ACT/Pool ; MODE 1: DMA gather-RMW (dma_compute) into an fp32 SBUF tile
import nki, nki.isa as nisa, nki.language as nl
P = 128
@nki.jit
def nm_combine(Yf, pos, w=None, MODE=0, GB=3, GS=1):
    NR, H = Yf.shape
    T, K = pos.shape[0], pos.shape[1]
    NT = (T + P - 1) // P
    pid = nl.program_id(axis=0)
    f32, bf16 = nl.float32, nl.bfloat16
    out = nl.ndarray((T, H), dtype=Yf.dtype, buffer=nl.shared_hbm)
    SK = dict(oob_mode=nisa.oob_mode.skip)
    engs = [nisa.engine.vector, nisa.engine.gpsimd, nisa.engine.scalar]
    if MODE in (3, 4):
        ident = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        iot = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(iot[0:P, 0:P], pattern=[[1, P]], offset=0, channel_multiplier=-1)
        nisa.tensor_scalar(dst=ident[0:P, 0:P], data=iot[0:P, 0:P], op0=nl.equal, operand0=0)
    for t in range(pid, NT, 2):
        rows = min(P, T - t * P)
        pt = nl.ndarray((P, K), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pt[0:rows, 0:K], src=pos[t * P:t * P + rows, 0:K])
        if w is not None:
            wt = nl.ndarray((P, K), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt[0:rows, 0:K], src=w[t * P:t * P + rows, 0:K])
        acc = nl.ndarray((P, H), dtype=f32, buffer=nl.sbuf)
        if MODE == 1:
            nisa.memset(acc[0:rows, 0:H], 0.0)
            for j in range(K):
                nisa.dma_compute(dst=acc[0:rows, 0:H], srcs=[acc[0:rows, 0:H], Yf.ap(pattern=[[H, rows], [1, H]], offset=0, vector_offset=pt[0:rows, j:j + 1], indirect_dim=0)],
                                 reduce_op=nl.add)
        elif MODE == 4:   # dense: Yt [T*K, H] token-major -> out[t] = sum_j Yt[t*K + j]  (PE identity sum, fp32 PSUM)
            g = nl.ndarray((P, K, H), dtype=bf16, buffer=nl.sbuf)
            if rows < P:
                nisa.memset(g[0:P, 0:K, 0:H], 0.0)
            for j in range(K):
                nisa.dma_copy(dst=g[0:rows, j, 0:H], src=Yf.ap(pattern=[[K * H, rows], [1, H]], offset=t * P * K * H + j * H),
                              dge_mode=nisa.dge_mode.none)
            ib = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf); nisa.tensor_copy(dst=ib, src=ident)
            ob = nl.ndarray((P, H), dtype=bf16, buffer=nl.sbuf)
            for q in range(H // 512):
                ps = nl.ndarray((P, 512), dtype=f32, buffer=nl.psum)
                for j in range(K):
                    nisa.nc_matmul(dst=ps[0:P, 0:512], stationary=ib[0:P, 0:P], moving=g[0:P, j, q * 512:(q + 1) * 512], accumulate=(j > 0))
                nisa.tensor_copy(dst=ob[0:P, q * 512:(q + 1) * 512], src=ps[0:P, 0:512], engine=(nisa.engine.scalar if q % 2 == 0 else nisa.engine.vector))
            nisa.dma_copy(dst=out[t * P:t * P + rows, 0:H], src=ob[0:rows, 0:H])
            continue
        elif MODE == 3:   # PE: out = sum_j diag(w_j) @ g_j  (fp32 PSUM accumulate)
            gs = []
            W = H // GS
            for j in range(K):
                gt = nl.ndarray((P, H), dtype=bf16, buffer=nl.sbuf)
                if rows < P:
                    nisa.memset(gt[0:P, 0:H], 0.0)
                for c_ in range(GS):
                    nisa.dma_copy(dst=gt[0:rows, c_ * W:(c_ + 1) * W], src=Yf.ap(pattern=[[H, rows], [1, W]], offset=c_ * W, vector_offset=pt[0:rows, j:j + 1], indirect_dim=0), **SK)
                gs.append(gt)
            dg = nl.ndarray((P, K, P), dtype=bf16, buffer=nl.sbuf)
            if rows < P:
                nisa.memset(dg[0:P, 0:K, 0:P], 0.0)
            if w is None:
                for j in range(K):
                    nisa.tensor_copy(dst=dg[0:rows, j, 0:P], src=ident[0:rows, 0:P], engine=(nisa.engine.vector if j % 2 == 0 else nisa.engine.gpsimd))
            else:
              for j in range(K):
                nisa.tensor_scalar(dst=dg[0:rows, j, 0:P], data=ident[0:rows, 0:P], op0=nl.multiply, operand0=wt[0:rows, j:j + 1],
                                   engine=(nisa.engine.vector if j % 2 == 0 else nisa.engine.gpsimd))
            ob = nl.ndarray((P, H), dtype=bf16, buffer=nl.sbuf)
            for q in range(H // 512):
                ps = nl.ndarray((P, 512), dtype=f32, buffer=nl.psum)
                for j in range(K):
                    nisa.nc_matmul(dst=ps[0:P, 0:512], stationary=dg[0:P, j, 0:P], moving=gs[j][0:P, q * 512:(q + 1) * 512], accumulate=(j > 0))
                nisa.tensor_copy(dst=ob[0:P, q * 512:(q + 1) * 512], src=ps[0:P, 0:512], engine=(nisa.engine.scalar if q % 2 == 0 else nisa.engine.vector))
            nisa.dma_copy(dst=out[t * P:t * P + rows, 0:H], src=ob[0:rows, 0:H])
            continue
        elif MODE == 2:   # weighted: acc = sum_j w[:, j] * g_j  (fp32), multiply-add chains split over DVE / Pool
            gs = []
            for j in range(K):
                gt = nl.ndarray((P, H), dtype=bf16, buffer=nl.sbuf)
                nisa.dma_copy(dst=gt[0:rows, 0:H], src=Yf.ap(pattern=[[H, rows], [1, H]], offset=0, vector_offset=pt[0:rows, j:j + 1], indirect_dim=0), **SK)
                gs.append(gt)
            ch = []
            for u in range(2):
                a = nl.ndarray((P, H), dtype=f32, buffer=nl.sbuf)
                e_ = engs[u]
                nisa.tensor_scalar(dst=a[0:rows, 0:H], data=gs[u][0:rows, 0:H], op0=nl.multiply, operand0=wt[0:rows, u:u + 1], engine=e_)
                for j in range(u + 2, K, 2):
                    if u == 0:
                        nisa.scalar_tensor_tensor(dst=a[0:rows, 0:H], data=gs[j][0:rows, 0:H], op0=nl.multiply, operand0=wt[0:rows, j:j + 1],
                                                  op1=nl.add, operand1=a[0:rows, 0:H])
                    else:
                        tmp = nl.ndarray((P, H), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=tmp[0:rows, 0:H], data=gs[j][0:rows, 0:H], op0=nl.multiply, operand0=wt[0:rows, j:j + 1], engine=e_)
                        nisa.tensor_tensor(dst=a[0:rows, 0:H], data1=a[0:rows, 0:H], data2=tmp[0:rows, 0:H], op=nl.add, engine=e_)
                ch.append(a)
            acc = nl.ndarray((P, H), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=acc[0:rows, 0:H], data1=ch[0][0:rows, 0:H], data2=ch[1][0:rows, 0:H], op=nl.add, engine=nisa.engine.vector)
        else:
            gs = []
            for j in range(K):
                gt = nl.ndarray((P, H), dtype=bf16, buffer=nl.sbuf)
                nisa.dma_copy(dst=gt[0:rows, 0:H], src=Yf.ap(pattern=[[H, rows], [1, H]], offset=0, vector_offset=pt[0:rows, j:j + 1], indirect_dim=0), **SK)
                gs.append(gt)
            # pairwise tree in fp32: (0+1) (2+3) (4+5) (6+7) on 3 engines, then sums
            s = []
            for u in range(K // 2):
                a = nl.ndarray((P, H), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a[0:rows, 0:H], data1=gs[2 * u][0:rows, 0:H], data2=gs[2 * u + 1][0:rows, 0:H], op=nl.add, engine=engs[u % 2])
                s.append(a)
            while len(s) > 1:
                n2 = []
                for u in range(0, len(s), 2):
                    a = nl.ndarray((P, H), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=a[0:rows, 0:H], data1=s[u][0:rows, 0:H], data2=s[u + 1][0:rows, 0:H], op=nl.add, engine=engs[(u // 2) % 2])
                    n2.append(a)
                s = n2
            acc = s[0]
        ob = nl.ndarray((P, H), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ob[0:rows, 0:H], src=acc[0:rows, 0:H], engine=nisa.engine.scalar)
        nisa.dma_copy(dst=out[t * P:t * P + rows, 0:H], src=ob[0:rows, 0:H])
    return out
