"""Generate the static SVG figures for the Trn2 MoE AWS-CN blog docx.

usage: make_charts.py <out_dir>
All numbers come from blog-trn2-draft.md / the raw summaries it cites.
Palette: validated categorical slots (blue / orange / aqua) on white; every bar is
direct-labelled (aqua is below 3:1 contrast, so labels are the required relief).
"""
import sys

OUT = sys.argv[1]
FONT = "-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei','Segoe UI',sans-serif"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e2"
NEUTRAL = "#c9c8c3"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_open(w, h):
    return [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" fill="white"/>']


def text(x, y, s, size=11, fill=INK2, anchor="start", weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" font-size="{size}"{w}>{esc(s)}</text>'


def hbar(x0, y, w, h, color, r=4):
    r = min(r, max(w, 0))
    return (f'<path d="M{x0:.1f},{y:.1f} H{x0+w-r:.1f} Q{x0+w:.1f},{y:.1f} {x0+w:.1f},{y+r:.1f} V{y+h-r:.1f} '
            f'Q{x0+w:.1f},{y+h:.1f} {x0+w-r:.1f},{y+h:.1f} H{x0:.1f} Z" fill="{color}"/>')


def vbar(x, base, w, h, color, r=4):
    r = min(r, max(h, 0))
    top = base - h
    return (f'<path d="M{x:.1f},{base:.1f} V{top+r:.1f} Q{x:.1f},{top:.1f} {x+r:.1f},{top:.1f} H{x+w-r:.1f} '
            f'Q{x+w:.1f},{top:.1f} {x+w:.1f},{top+r:.1f} V{base:.1f} Z" fill="{color}"/>')


def legend(items, x, y, gap=34):
    out, cx = [], x
    for label, color in items:
        out.append(f'<rect x="{cx}" y="{y-9}" width="12" height="12" rx="2" fill="{color}"/>')
        out.append(text(cx + 18, y + 1, label, 11, INK2))
        cx += gap + sum(2 if ord(c) > 0x2E7F else 1 for c in label) * 6
    return "\n".join(out)


def box(x, y, w, h, stroke, fill="white", r=10, dash=None, sw=2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def arrow(x1, y1, x2, y2, color="#4a5160"):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 9 * math.cos(a), y2 - 9 * math.sin(a)
    p1 = (hx + 5 * math.sin(a), hy - 5 * math.cos(a))
    p2 = (hx - 5 * math.sin(a), hy + 5 * math.cos(a))
    return (f'<line x1="{x1}" y1="{y1}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{color}" stroke-width="2"/>'
            f'<path d="M{x2},{y2} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{color}"/>')


def save(name, s):
    s.append("</svg>")
    open(f"{OUT}/{name}", "w", encoding="utf-8").write("\n".join(s))


# ---------- Figure 1: reference architecture ----------
W, H = 880, 540
s = svg_open(W, H)
s.append(text(W / 2, 30, "参考架构：私有 EKS 上的 Trainium2 MoE 推理", 17, INK, "middle", 700))
s.append(box(16, 46, 848, 476, "#a0a7b4", "white", 14, "7,5", 1.5))
s.append(text(32, 70, "AWS Region", 12.5, "#4a5160"))
s.append(box(32, 82, 650, 424, "#1f9d55", "#f3fbf6", 14))
s.append(text(48, 106, "私有 VPC（无公网入口）", 13, "#1f7a43", weight=600))
# bastion
s.append(box(48, 122, 120, 72, "#1f9d55"))
s.append(text(108, 146, "跳板机", 12.5, INK, "middle", 600))
s.append(text(108, 165, "SSM Session Manager", 9.5, INK2, "middle"))
s.append(text(108, 180, "kubectl / 压测编排", 9.5, INK2, "middle"))
s.append(arrow(108, 236, 108, 198))
s.append(text(108, 254, "运维人员", 11.5, INK, "middle", 600))
s.append(text(108, 271, "经 SSM 接入", 10, MUTED, "middle"))
s.append(arrow(170, 158, 188, 158))
# EKS
s.append(box(190, 112, 476, 380, "#f59b00", "white", 12, None, 2.5))
s.append('<path d="M190,124 Q190,112 202,112 H654 Q666,112 666,124 V138 H190 Z" fill="#f59b00"/>')
s.append(text(428, 130, "Amazon EKS（私有 API endpoint）", 12.5, "white", "middle", 600))
s.append(box(206, 150, 444, 46, "#9aa3b2", "#f2f4f8", 8, None, 1.2))
s.append(text(222, 170, "系统节点组", 11.5, INK, weight=600))
s.append(text(222, 187, "CoreDNS · AWS Load Balancer Controller · Karpenter · Pod Identity Agent", 9.5, INK2))
s.append(box(206, 206, 444, 274, "#f59b00", "#fffbec", 8, None, 1.5))
s.append(text(222, 228, "Karpenter NodePool：trn2.48xlarge", 12, "#b35c00", weight=600))
s.append(text(222, 246, "16 颗 Trainium2，每颗 96 GiB HBM · 本地 NVMe：模型权重与 NEFF 编译缓存", 9.5, INK2))
# chips grid 2 x 8
cx0, cy0, cw, ch, g = 222, 260, 46, 40, 7
for i in range(16):
    r, c = divmod(i, 8)
    x, y = cx0 + c * (cw + g), cy0 + r * (ch + g)
    if i == 0:
        s.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="6" fill="{BLUE}"/>')
        s.append(text(x + cw / 2, y + 17, "芯片 0", 9.5, "white", "middle", 600))
        s.append(text(x + cw / 2, y + 31, "服务实例", 8.5, "white", "middle"))
    else:
        s.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="6" fill="white" stroke="#d9c9a0" stroke-width="1.2" stroke-dasharray="4,3"/>')
        s.append(text(x + cw / 2, y + 24, f"芯片 {i}", 9, MUTED, "middle"))
s.append(box(222, 362, 412, 58, "#9aa3b2", "white", 8, None, 1.2))
s.append(text(236, 382, "每颗芯片一个服务实例：vLLM + vllm-neuron，Qwen3-30B-A3B", 10, INK, weight=600))
s.append(text(236, 398, "TP4 / LNC=2（4 个逻辑核，每个由 2 个物理 NeuronCore 组成）", 9.5, INK2))
s.append(text(236, 412, "其余芯片可放更多副本（本文只测单个副本）", 9.5, MUTED))
s.append(text(222, 442, "Neuron 组件：device plugin · scheduler extension（多芯片分配）", 9.5, INK2))
s.append(text(222, 460, "多机时：EFA device plugin（大 MoE 两机测试）", 9.5, INK2))
# right side services
s.append(box(700, 112, 152, 92, "#8b5cf6"))
s.append(text(776, 140, "Amazon ECR Public", 12, INK, "middle", 600))
s.append(text(776, 160, "Neuron 官方", 9.5, INK2, "middle"))
s.append(text(776, 175, "vLLM 推理镜像", 9.5, INK2, "middle"))
s.append(f'<line x1="700" y1="158" x2="676" y2="158" stroke="#8b5cf6" stroke-width="2"/>')
s.append(arrow(684, 158, 668, 158, "#8b5cf6"))
s.append(box(700, 228, 152, 132, "#1f9d55"))
s.append(text(776, 254, "Amazon S3", 12.5, INK, "middle", 600))
s.append(text(776, 274, "模型权重", 9.5, INK2, "middle"))
s.append(text(776, 290, "NEFF 编译缓存", 9.5, INK2, "middle"))
s.append(text(776, 312, "启动前拉取缓存", 9.5, "#1f7a43", "middle"))
s.append(text(776, 327, "服务就绪后推送", 9.5, "#1f7a43", "middle"))
s.append(text(776, 346, "经 S3 Gateway Endpoint", 9, "#1f7a43", "middle"))
s.append(f'<line x1="700" y1="294" x2="690" y2="294" stroke="#1f9d55" stroke-width="2"/>')
s.append('<rect x="660" y="282" width="32" height="24" rx="4" fill="#1f9d55"/>')
s.append(text(676, 298, "GW", 9.5, "white", "middle", 600))
s.append(box(700, 384, 152, 84, "#dc2f45"))
s.append(text(776, 410, "EKS Pod Identity", 12, INK, "middle", 600))
s.append(text(776, 430, "Pod → S3 最小权限角色", 9.5, INK2, "middle"))
s.append(text(776, 446, "无长期凭证", 9.5, INK2, "middle"))
save("fig1-architecture.svg", s)

# ---------- Figure 2: four optimization steps, two panels ----------
steps = ["基线（BF16）", "① 去掉 padding 白算", "② 专家权重改 FP8", "③ decode 批上限 32", "④ 打包 prefill（推荐）"]
tpot = [22.53, 20.20, 15.86, 15.81, 14.43]
p99 = [1288, 1304, 1188, 1141, 276]
W, H = 880, 350
s = svg_open(W, H)
s.append(text(24, 30, "ShareGPT 16 并发：四项优化逐步叠加（越低越好）", 16, INK, weight=700))
s.append(text(24, 50, "每一步叠加在上一步之上；① 为纯 BF16，② 起为 W8A16（只有专家权重为 FP8）", 11, INK2))


def panel(x0, x1, title, vals, vmax, fmt, ticks):
    out = [text(x0, 82, title, 12.5, INK, weight=600)]
    top, band = 98, 40
    sx = (x1 - x0) / vmax
    for t in ticks:
        gx = x0 + t * sx
        out.append(f'<line x1="{gx:.1f}" y1="{top-4}" x2="{gx:.1f}" y2="{top+band*len(vals)}" stroke="{GRID}"/>')
        out.append(text(gx, top + band * len(vals) + 16, f"{t:g}", 10, MUTED, "middle"))
    for i, v in enumerate(vals):
        y = top + i * band + 10
        out.append(hbar(x0, y, v * sx, 20, BLUE))
        out.append(text(x0 + v * sx + 6, y + 15, fmt(v), 10.5, INK, weight=600 if i == len(vals) - 1 else None))
    out.append(f'<line x1="{x0}" y1="{top-4}" x2="{x0}" y2="{top+band*len(vals)}" stroke="{MUTED}"/>')
    return out


top, band = 98, 40
for i, lab in enumerate(steps):
    s.append(text(178, top + i * band + 25, lab, 11, INK2, "end", 600 if i == 4 else None))
s += panel(190, 500, "TPOT（ms）", tpot, 25, lambda v: f"{v:.2f}", [0, 5, 10, 15, 20, 25])
s += panel(560, 840, "TTFT P99（ms）", p99, 1400, lambda v: f"{v:,}", [0, 350, 700, 1050, 1400])
s.append(text(24, H - 14, "TPOT 的改善约 3/4 来自 FP8、1/4 来自打包；TTFT P99 的改善几乎全部来自打包。", 10.5, MUTED))
save("fig2-steps.svg", s)

# ---------- Figure 3: same-precision throughput ----------
groups = [("W8A16 · 16 并发", (1075, 1659, 1137)), ("W8A16 · 32 并发", (1753, 2658, 1790)),
          ("BF16 · 16 并发", (763, 1224, 714))]
series = [("1 颗 Trainium2", BLUE), ("1 张 H100", ORANGE), ("1 张 RTX PRO 6000 Blackwell", AQUA)]
W, H = 880, 410
s = svg_open(W, H)
s.append(text(24, 30, "同精度输出吞吐（ShareGPT，tok/s，越高越好）", 16, INK, weight=700))
s.append(text(24, 50, "Qwen3-30B-A3B；同一批 prompt、贪心解码；比较单位为 1 颗芯片对 1 张 GPU，不是每美元性能", 11, INK2))
s.append(legend(series, 24, 78))
base, ytop, vmax = 330, 110, 2800
sy = (base - ytop) / vmax
xl, xr = 80, 850
for t in range(0, vmax + 1, 700):
    gy = base - t * sy
    s.append(f'<line x1="{xl}" y1="{gy:.1f}" x2="{xr}" y2="{gy:.1f}" stroke="{GRID}"/>')
    s.append(text(xl - 8, gy + 4, f"{t:,}", 10, MUTED, "end"))
gw = (xr - xl) / len(groups)
bw, gap = 62, 2
for i, (lab, vals) in enumerate(groups):
    cx = xl + gw * i + gw / 2
    x = cx - (3 * bw + 2 * gap) / 2
    for j, v in enumerate(vals):
        xb = x + j * (bw + gap)
        s.append(vbar(xb, base, bw, v * sy, series[j][1]))
        s.append(text(xb + bw / 2, base - v * sy - 6, f"{v:,}", 10.5, INK, "middle", 600))
    s.append(text(cx, base + 20, lab, 11.5, INK2, "middle"))
s.append(f'<line x1="{xl}" y1="{base}" x2="{xr}" y2="{base}" stroke="{MUTED}"/>')
s.append(text(24, H - 30, "W8A16：只有 MoE 专家权重为 FP8，计算与 KV cache 为 BF16。Trn2 的 W8A16 为推荐配置，BF16 为第 ① 步配置（不打包）。", 10, MUTED))
s.append(text(24, H - 14, "GPU 均为 vLLM 0.33 默认调度；GPU 的 W8A16 走 Marlin 路径。", 10, MUTED))
save("fig3-compare.svg", s)

# ---------- Figure 4: prefill MoE timeline schematic ----------
W, H = 880, 400
s = svg_open(W, H)
s.append(text(24, 30, "prefill MoE kernel 改造前后的执行时间线（示意）", 16, INK, weight=700))
s.append(text(24, 50, "5 个真实块（专家 E5 E5 E9 E14 E14）+ 2 个空块；时长按实测量级示意，不是实测时间线", 11, INK2))
s.append(legend([("搬专家权重", BLUE), ("gather token / 写结果", ORANGE), ("PE 矩阵计算", AQUA), ("空块", NEUTRAL)], 24, 78))
X0, SX = 170, 3.6


def seg(y, t0, d, color, label=None, h=22):
    out = [f'<rect x="{X0+t0*SX+1:.1f}" y="{y}" width="{max(d*SX-2,1):.1f}" height="{h}" rx="3" fill="{color}"/>']
    if label and d * SX > 22:
        out.append(text(X0 + (t0 + d / 2) * SX, y + 15, label, 9.5, "white" if color != NEUTRAL else INK2, "middle", 600))
    return out


# original
s.append(text(24, 116, "原版", 12.5, INK, weight=600))
lanes_o = [("软件 DGE 队列", 128), ("PE", 156)]
for name, y in lanes_o:
    s.append(text(X0 - 10, y + 15, name, 10.5, INK2, "end"))
    s.append(f'<line x1="{X0}" y1="{y+24}" x2="{X0+182*SX}" y2="{y+24}" stroke="{GRID}"/>')
blocks = ["E5", "E5", "E9", "E14", "E14", "空", "空"]
t = 0
for b in blocks:
    if b == "空":
        s += seg(128, t, 17, NEUTRAL, "空")
        s += seg(128, t + 17, 8, NEUTRAL)
        s += seg(156, t + 25, 7, NEUTRAL)
    else:
        s += seg(128, t, 17, BLUE, b)
        s += seg(128, t + 17, 8, ORANGE)
        s += seg(156, t + 25, 7, AQUA)
    t += 25
s.append(text(X0 + 182 * SX + 4, 172, "结束", 10, MUTED))
s.append(text(X0, 202, "每块都重搬整个专家；全部搬运挤在一条队列上，PE 大部分时间在等数据；空块照样执行", 10, MUTED))
# new
s.append(text(24, 240, "新 kernel", 12.5, INK, weight=600))
lanes_n = [("硬件 DGE（Sync 队列）", 252), ("软件 DGE 队列", 280), ("PE", 308)]
for name, y in lanes_n:
    s.append(text(X0 - 10, y + 15, name, 10.5, INK2, "end"))
    s.append(f'<line x1="{X0}" y1="{y+24}" x2="{X0+182*SX}" y2="{y+24}" stroke="{GRID}"/>')
for k, (lab, t0) in enumerate([("E5", 0), ("E9", 12), ("E14", 24)]):
    s += seg(252, t0, 12, BLUE, lab)
for k in range(5):
    s += seg(280, k * 8, 8, ORANGE)
pe = [("E5", 12), ("E5", 20), ("E9", 28), ("E14", 36), ("E14", 44)]
for lab, t0 in pe:
    s += seg(308, t0, 8, AQUA, lab)
s += seg(308, 52, 4, NEUTRAL)
s += seg(308, 56, 4, NEUTRAL)
s.append(text(X0 + 60 * SX + 6, 324, "结束", 10, MUTED))
s.append(text(X0, 354, "权重只在专家变化时搬一次，走硬件 DGE；token gather 走软件 DGE，两条队列并行；", 10, MUTED))
s.append(text(X0, 370, "PE 跨块流水连续工作；空块不搬任何数据（仍占少量 PE 时间）。实测每层快 1.9–2.8 倍。", 10, MUTED))
save("fig4-prefill-timeline.svg", s)

# ---------- Figure 5: per-layer prefill MoE time ----------
Ts = [128, 512, 1024, 2048, 4096, 8192]
old = [2.10, 2.33, 2.48, 3.04, 4.25, 6.63]
new = [0.76, 0.95, 1.06, 1.46, 2.16, 3.57]
W, H = 880, 400
s = svg_open(W, H)
s.append(text(24, 30, "每层 MoE prefill 耗时（ms，越低越好）", 16, INK, weight=700))
s.append(text(24, 50, "Qwen3-30B-A3B，单个 TP4 rank，路由取自真实模型在 ShareGPT 上的输出；均含分组（mapping），新版另含合并（combine）", 11, INK2))
s.append(legend([("原版（vllm-neuron moe_cte）", ORANGE), ("新 kernel", BLUE)], 24, 78))
base, ytop, vmax = 320, 120, 7
sy = (base - ytop) / vmax
xl, xr = 70, 850
for tk in range(0, 8):
    gy = base - tk * sy
    s.append(f'<line x1="{xl}" y1="{gy:.1f}" x2="{xr}" y2="{gy:.1f}" stroke="{GRID}"/>')
    s.append(text(xl - 8, gy + 4, f"{tk}", 10, MUTED, "end"))
gw = (xr - xl) / len(Ts)
bw = 40
for i, T in enumerate(Ts):
    cx = xl + gw * i + gw / 2
    xa, xb = cx - bw - 1, cx + 1
    s.append(vbar(xa, base, bw, old[i] * sy, ORANGE))
    s.append(text(xa + bw / 2, base - old[i] * sy - 6, f"{old[i]:.2f}", 10, INK, "middle"))
    s.append(vbar(xb, base, bw, new[i] * sy, BLUE))
    s.append(text(xb + bw / 2, base - new[i] * sy - 6, f"{new[i]:.2f}", 10, INK, "middle"))
    s.append(text(cx, base + 20, f"T = {T:,}", 11.5, INK2, "middle"))
    s.append(text(cx, base + 37, f"{old[i]/new[i]:.1f}×", 11, INK, "middle", 700))
s.append(f'<line x1="{xl}" y1="{base}" x2="{xr}" y2="{base}" stroke="{MUTED}"/>')
s.append(text(24, H - 12, "T 为一次 prefill 的 token 数；第二行为加速倍数。48 层累计每次 prefill 节省 64 ms（T = 128）到 147 ms（T = 8,192）。", 10, MUTED))
save("fig5-prefill-layer.svg", s)

# ---------- Figure 6: large MoE per-layer ----------
panels = [("DeepSeek-V3，EP64（每 rank 4 个专家）", [("2K", 3.95, 0.87, 4.6), ("8K", 8.10, 1.33, 6.1), ("32K", 27.56, 3.90, 7.1)], 30, [0, 10, 20, 30]),
          ("Qwen3-235B-A22B，EP16（每 rank 8 个专家）", [("2K", 6.25, 0.96, 6.5), ("8K", 18.95, 2.62, 7.2), ("32K", 57.50, 7.75, 7.4)], 60, [0, 20, 40, 60])]
W, H = 880, 350
s = svg_open(W, H)
s.append(text(24, 30, "大 MoE 每层 prefill MoE 耗时（ms，越低越好）", 16, INK, weight=700))
s.append(text(24, 50, "单个 rank（LNC=2）microbench，合成路由，取负载最重的 rank；按全局 prefill token 数分组", 11, INK2))
s.append(legend([("原版（vllm-neuron moe_cte）", ORANGE), ("新 kernel", BLUE)], 24, 78))
for pi, (title, rows, vmax, ticks) in enumerate(panels):
    x0 = 70 + pi * 430
    x1 = x0 + 340
    s.append(text(x0 - 46, 112, title, 12, INK, weight=600))
    top, band = 126, 54
    sx = (x1 - x0) / vmax
    for tk in ticks:
        gx = x0 + tk * sx
        s.append(f'<line x1="{gx:.1f}" y1="{top-4}" x2="{gx:.1f}" y2="{top+band*3}" stroke="{GRID}"/>')
        s.append(text(gx, top + band * 3 + 16, f"{tk}", 10, MUTED, "middle"))
    for i, (lab, a, b, sp) in enumerate(rows):
        y = top + i * band + 8
        s.append(text(x0 - 10, y + 19, lab, 11, INK2, "end"))
        s.append(hbar(x0, y, a * sx, 15, ORANGE))
        s.append(text(x0 + a * sx + 6, y + 12, f"{a:.2f}", 10, INK))
        s.append(hbar(x0, y + 17, max(b * sx, 2), 15, BLUE))
        s.append(text(x0 + b * sx + 6, y + 29, f"{b:.2f}（{sp}×）", 10, INK, weight=600))
    s.append(f'<line x1="{x0}" y1="{top-4}" x2="{x0}" y2="{top+band*3}" stroke="{MUTED}"/>')
s.append(text(24, H - 12, "原版按“本 rank 实际收到的 token”输入，对它偏有利；按 vllm-neuron 真实的 all-gather 输入，DeepSeek-V3 EP64 8K 时为 67.1 → 2.26 ms。", 10, MUTED))
save("fig6-large-moe.svg", s)
print("ok")
