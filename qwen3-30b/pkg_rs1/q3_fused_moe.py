# SPDX-License-Identifier: Apache-2.0
"""Fused Qwen3-MoE decode kernel integration helpers (Q3_FUSED_MOE=1).

Blob layout (per local expert e, per rank): wb[e] = [128, H1*2I + NR*H/2]  (bf16)
  * columns [0, H1*2I):  gate|up rows, row p holds gu[e, p*H1 + h1, :] for h1 = 0..H1-1
                          (== gu[e].reshape(128, H1*2I), gu = [E, H, 2I] with gate | up on the last dim)
  * columns [H1*2I, BL): folded down projection, row p, chunk r holds dn2[e, r*128 + p, :]
                          with dn2 = dn.reshape(E, 2I, H/2) (row f = 2i + b holds W_down[i, b*H/2:(b+1)*H/2])
Only the blob is kept on device; prefill / large-T fallback re-derive the natural layouts from it in-graph.

Kernel module: env Q3_FUSED_MOE_KMOD (default ii_fused11a; wsB_fused9 faults in the full graph). Accepts a module name importable as a sibling of
this package (e.g. wsB_fused11), an absolute module name, or a path to a .py file. Function name:
env Q3_FUSED_MOE_KFN (default q3moe_fused). Signature:
    q3moe_fused(hidden[T,H], ln_w[1,H] bf16, rw[H,E] bf16, wb[E,128,BL], eps=..., I=I_per_rank) -> [2, T, H]
(partial per physical core; caller sums dim 0 before the TP all-reduce).
"""

import importlib
import importlib.util
import logging
import os

import torch

logger = logging.getLogger(__name__)

P = 128
Q3_FUSED_MOE = os.environ.get("Q3_FUSED_MOE", "0") == "1"
# P2: Q3_FUSED_MOE_LAYOUT = nat (default) -> expert weights kept in the stock PR56 layouts gate_up_proj_weight [E,H,2I]
# and down_proj_weight [E,I,H]; the decode kernel (p2_fused11n) DMAs them directly and prefill / T>16 decode use them
# with no in-graph re-layout.  blob -> legacy [E,128,BL] blob (ii_fused11a) with in-graph unblob for prefill.
Q3_FUSED_MOE_LAYOUT = os.environ.get("Q3_FUSED_MOE_LAYOUT", "nat")
assert Q3_FUSED_MOE_LAYOUT in ("nat", "blob"), Q3_FUSED_MOE_LAYOUT
Q3_FUSED_MOE_NAT = Q3_FUSED_MOE_LAYOUT == "nat"
Q3_FUSED_MOE_KMOD = os.environ.get("Q3_FUSED_MOE_KMOD", "p2_fused11n" if Q3_FUSED_MOE_NAT else "ii_fused11a")
if Q3_FUSED_MOE_NAT and Q3_FUSED_MOE_KMOD == "ii_fused11a":
    Q3_FUSED_MOE_KMOD = "p2_fused11n"   # ii_fused11a needs the blob; same kernel body with natural-layout DMAs
Q3_FUSED_MOE_KFN = os.environ.get("Q3_FUSED_MOE_KFN", "q3moe_fused")
# decode token counts above this fall back to moe_block_tkg (kernel asserts T <= 16)
Q3_FUSED_MOE_MAX_T = int(os.environ.get("Q3_FUSED_MOE_MAX_T", "16"))

_KERNEL = None


def blob_width(hidden_size: int, i_per_rank: int) -> int:
    gu = 2 * i_per_rank
    assert hidden_size % P == 0 and gu % P == 0 and hidden_size % 2 == 0
    return (hidden_size // P) * gu + (gu // P) * (hidden_size // 2)


def build_blob(gu: torch.Tensor, dn: torch.Tensor) -> torch.Tensor:
    """gu [E, H, 2I] (gate | up), dn [E, I, H] -> blob [E, 128, BL] (contiguous)."""
    E, H, GU = gu.shape
    I = GU // 2
    assert dn.shape == (E, I, H), (dn.shape, gu.shape)
    H1, NR, HB = H // P, GU // P, H // 2
    a = gu.reshape(E, P, H1 * GU)
    b = dn.reshape(E, GU, HB).reshape(E, NR, P, HB).permute(0, 2, 1, 3).reshape(E, P, NR * HB)
    return torch.cat([a, b], dim=2).contiguous()


def unblob_gate_up(blob: torch.Tensor, hidden_size: int, i_per_rank: int) -> torch.Tensor:
    """blob -> gate_up [E, H, 2, I] (materialized copy; strided gather in-graph)."""
    E = blob.shape[0]
    GU = 2 * i_per_rank
    H1 = hidden_size // P
    return blob[:, :, : H1 * GU].reshape(E, hidden_size, 2, i_per_rank)


def unblob_down(blob: torch.Tensor, hidden_size: int, i_per_rank: int) -> torch.Tensor:
    """blob -> down [E, I, H] (materialized copy)."""
    E = blob.shape[0]
    GU = 2 * i_per_rank
    H1, NR, HB = hidden_size // P, GU // P, hidden_size // 2
    d = blob[:, :, H1 * GU :].reshape(E, P, NR, HB).permute(0, 2, 1, 3)
    return d.reshape(E, GU, HB).reshape(E, i_per_rank, hidden_size)


def _import_kernel_module(name: str):
    if name.endswith(".py"):
        spec = importlib.util.spec_from_file_location(os.path.basename(name)[:-3], name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    if __package__:
        try:
            return importlib.import_module(f".{name}", package=__package__)
        except ModuleNotFoundError:
            pass
    return importlib.import_module(name)


def get_kernel():
    global _KERNEL
    if _KERNEL is None:
        mod = _import_kernel_module(Q3_FUSED_MOE_KMOD)
        _KERNEL = getattr(mod, Q3_FUSED_MOE_KFN)
        logger.info("Q3_FUSED_MOE: using kernel %s.%s (%s)", Q3_FUSED_MOE_KMOD, Q3_FUSED_MOE_KFN,
                    getattr(mod, "__file__", "?"))
    return _KERNEL


if Q3_FUSED_MOE:
    get_kernel()  # import eagerly: importlib cannot be traced by dynamo


def fused_moe_decode(hidden: torch.Tensor, ln_w: torch.Tensor, rw_t: torch.Tensor, blob: torch.Tensor,
                     eps: float, i_per_rank: int) -> torch.Tensor:
    """hidden [T, H] bf16 -> rank-partial MoE output [T, H] (before the TP all-reduce)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    out = wrap_nki(get_kernel())[2](hidden=hidden, ln_w=ln_w.view(1, -1), rw=rw_t, wb=blob,
                                    eps=float(eps), I=int(i_per_rank))
    return out[0] + out[1]


def fused_moe_decode_nat(hidden: torch.Tensor, ln_w: torch.Tensor, rw_t: torch.Tensor, gu: torch.Tensor,
                         dn: torch.Tensor, eps: float, i_per_rank: int) -> torch.Tensor:
    """P2 natural layouts: gu [E, H, 2I] (gate | up), dn [E, I, H] -> rank-partial MoE output [T, H]."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    out = wrap_nki(get_kernel())[2](hidden=hidden, ln_w=ln_w.view(1, -1), rw=rw_t, gu=gu, dn=dn,
                                    eps=float(eps), I=int(i_per_rank))
    return out[0] + out[1]
