# Qwen3-30B-A3B (BF16) on Trn2 — high-concurrency throughput research · 2026-11-08

## 结论摘要（2026-11-09 定稿，工作停在 RS7）
- **推荐配置 = RS7**：FP8 专家权重（W8A16，每专家一个标量 scale）+ T≤32 的 fused decode MoE kernel（p2_fused11r）+ bs32 bucket + 多序列打包 prefill（pack_max 16、budget 4096）。1 device、TP4 / LNC2、MML 8192，启动参数见第 3 节。
- **相对 NJ 基线**（同机同口径，ShareGPT）：c16 TPOT 22.53 → **14.43 ms（−36%）**，吞吐 688 → **1075 tok/s（+56%）**，TTFT P99 1288 → **276 ms（−79%）**；c1 TPOT 6.45 → 5.85（−9%）；c32 **1753 tok/s**（NJ c16 的 2.55×），TTFT P99 581 ms。
- **对 H100 TP1，同精度比**（vLLM 0.33，同 prompt，详见 2.8）：W8A16 对 W8A16（RS7 对 H100 W8A16 补测）c16 TPOT **1.54×**、吞吐 **0.65×**、TTFT P99 1.7×；c32 TPOT 1.51×、吞吐 **0.66×**；c1 TPOT 1.19×。纯 BF16（RS1 对 H100 BF16）c16 TPOT **1.58×**、吞吐 0.62×。两种精度下 decode 差距几乎一样，原因是 H100 用上 FP8 后自己也快了 27%。
  - 跨精度参考（RS7 W8A16 对 H100 BF16，只作参考）：c16 TPOT 1.13×、吞吐 0.88×、TTFT P99 1.9×（NJ 时分别是 1.76×、0.56×、8.8×）；c1 TPOT 1.23×；trn2 c32 吞吐是 H100 BF16 c16 的 1.43×。
  - RS1 → RS7 的 c16 TPOT 改善（20.20 → 14.43 ms）约 3/4 来自 FP8 权重，约 1/4 来自打包；TTFT P99 的改善几乎都来自打包。
- **对 g7e**（g7e.4xlarge，1 张 RTX PRO 6000 Blackwell，vLLM 0.33，同 prompt，详见 2.8）：W8A16 对 W8A16 基本持平，c16 TPOT 1.04×、吞吐 0.95×，c32 TPOT 1.00×、吞吐 0.98×；TTFT P99 c16 1.7×、c32 2.0×。纯 BF16 时 trn2 RS1 略快，c16 TPOT 0.91×、吞吐 1.07×。prefill 重的档位 g7e W8A16 明显更好（1K→512 c16 吞吐是 RS3 的 1.25×，4K→512 是 1.31×）。
- **数值与稳定性**：GSM8K 500 = 0.946（NJ 0.934；H100 BF16 0.930、H100 W8A16 0.932；g7e BF16 0.930、g7e W8A16 0.932；500 题标准误约 1.1 个点）。三次 soak 都是 0 失败：RS5（RS3 配置）71 min / 8376 请求；RS4（打包）51 min / 5587 请求；RS7（c32 + 打包）51 min / 6089 请求，其中 2 条重复字符告警经重跑确认是正常内容。所有 NaN 检查 PASS，服务日志错误行 0。
- **试过但没有收益的**：c32 下 pack_max 32 或 budget 8192、decode MoE kernel 的每专家开销优化、权重 DMA 改 hw-DGE（≤1.4%）。
- **用户 11/9 决定不做的**：W8A8 decode MoE（上限约 15%，有数值风险）、转置 K cache 布局（约 5–7%）。更早就判定不做的：EP4、P/D 分离、c64、EAGLE3（理由见 2.5）。
- 现状：**rs-b 已于 11/9 按用户决定删除**（节点 <NODE> 和节点上的 /data/cache 保留）；原始结果已拉回本地 `results/`，如何用暖缓存重启见第 5 节。

## 0. 环境与口径
- 节点 <NODE>（<EKS_CLUSTER>, ap-south-2），只用空闲 device；pod `rs-a/b/c/d/e`（各 1 device，my-scheduler，BDF b5/b6/cb/6f/58；nz-b 在 cc，未触碰）。
- 基线 = NJ（pkg_nj，NM_MOE=1，1 device TP4/LNC2，MNS16，MML8192，NGB4352，buckets 同 nj_srv.sh）。同机复测 = **RS0**（与 NJA 一致：sg c16 s42 TPOT 22.53 / 688 tok/s / TTFT P99 1288，GSM8K 0.934）。
- 同口径 bench：`rs_bench.sh`（= nj_bench.sh + `dec` 纯 decode 扫描 + `b2` 1K→512 + `sgx` 高并发 ShareGPT），每次 reset prefix cache；GSM8K 500 题 cbv_cli。
- **精度**：RS0、RS1 是纯 BF16（权重、计算、KV）。RS2 起（含推荐的 RS7）是 W8A16：只有专家权重是 FP8 e4m3（每个专家一个标量 scale），计算和 KV cache 都是 BF16。
- H100 对照取自 <GPU_H100_BF16>/REPORT.md（TP1/TP2，vLLM 0.33，贪心），**是纯 BF16**：`--dtype bfloat16`，`quantization=None`，`kv_cache_dtype=auto`。2.2、2.3 和第 1 节里的 H100 都是这一组。同精度对照是 H100 FP8 补测（`<GPU_H100_FP8>/`）的 W8A16 组和 g7e（`<GPU_G7E>/`）的 BF16 / W8A16 组，见 2.8。

## 1. 阶段 1 诊断

### 1.1 纯 decode 每步耗时（RS0，random 输入 128/1024，输出 512 ignore-eos，ITL 中位数）
| bs | ctx~128-640 | ctx~1024-1536 |
|---|---|---|
| 1 | 6.44 ms | 6.47 |
| 4 | 10.76 | 10.87 |
| 16 | 21.09 | 19.34 |

ShareGPT c16 ITL 中位数 21.0 ms ≈ 纯 decode bs16 → c16 TPOT 22.6 中只有 ~1.6 ms（7%）来自 prefill 插入；**差距主体是 decode 单步本身**（H100 TP1 BF16 c16 TPOT 12.8）。

### 1.2 带宽利用
- bs16 每步读取：MoE union ≈ 128·(1−(120/128)^16) ≈ 82 专家/层 × 9.44 MB × 48 = 37.1 GB，+ attention 1.8 GB + lm_head 0.6 GB ≈ 39.6 GB。
- trn2 21.1 ms → **1.88 TB/s ≈ 65%** of 2.9 TB/s；H100（BF16）12.8 ms → 3.1 TB/s ≈ 92% of 3.35 TB/s。BF16 trn2 下限 ≈ 13.7 ms。
- MoE 占 bs16 步的 ~85%（kernel 微基准 0.378 ms/层 × 48 = 18.2 ms）。

### 1.3 MoE decode kernel（p2_fused11n）的空档（kernel 微基准 wsB_t1.py，随机路由，每层 µs）
- 静态展开循环 NSL=min(64,4T) 个位置/物理核；padding 位置只 oob-skip DMA，**matmul/ACT/DVE 照算**，且都排在最后形成串行尾巴。
  - 证据：T=16 强制 nsl=64/48/40（数值仍正确，说明 40 以上全是 padding）→ 379/350/323 µs；nsl=32 才截到真实专家。⇒ T=16 每层 ~56 µs 白算（×48 ≈ 2.7 ms/步，≈ c16 TPOT 的 12%）。
- 真实位置每个 ~7.5 µs ≈ 2.36 MB DMA → ~315 GB/s/物理核，已接近 HBM 峰值（~87%）。⇒ “大块权重搬运”（外部参考方案①）在本 kernel 已基本做到（gu 每分区 12 KiB 连续），空间很小。
- 设备 profile（NRT inspect session 模式，单逻辑核微基准 T=16 L=4）：11n PE 61% / DVE 29% / ACT 32% / GpSimd 50% / DMA active 80%，HBM 读 ~790 GB/s/逻辑核；FP8 版 11r 字节和执行时间同时减半（0.190→0.098 s），HBM 读速率不变（~775 GB/s）→ **decode MoE 是 HBM 带宽受限**；11r 下 PE 78%、GpSimd（sw-DGE 描述符生成）68% 成为下一瓶颈。
- 注：e2e 中 vLLM 的 /start_profile（session，异步执行）抓不到 decode NEFF（vllm_neuron 源码自述的已知问题，且 KV override 下 HBM 不够分配默认 profiler 缓冲），所以整步设备时间线未取到；整步分解用 kernel 微基准 + e2e 差分代替。

### 1.4 TTFT P99（c16 1.25 s vs H100 BF16 146 ms）
- NJA c16 逐请求 TTFT：开头 16 个同时到达的请求呈阶梯 65→128→199→…→1311 ms（每级 75–90 ms），之后的 48 个请求 TTFT 中位 227 ms、最大 284 ms。
- 根因：vllm_neuron 调度器 `max_prefills_per_batch = 1`（scheduler.py:282，runner 里 `assert len(req_ids)==1`），prefill 逐条串行；而短 prompt prefill 是权重带宽受限的（每步读全部 128 专家 ≈ 58 GB → ≥20 ms），16 条 = 读 16 遍。H100 vLLM 把 16 条打包进一个 8192-token step。

## 2. 阶段 2 方向与结果

### 2.1 配置一览（全部同口径：TP4/LNC2 单 device，MML 8192，bucket 同 NJ，除特别说明）
| tag | 内容 | 包 / 关键 env |
|---|---|---|
| RS0 | NJ 基线同机复测 | pkg_nj，NM_MOE=1 |
| RS1 | + decode MoE kernel **p2_fused11p**（padding 交错 + T≤32 支持），仍 BF16 | pkg_rs1，`Q3_FUSED_MOE_KMOD=p2_fused11p` |
| RS2 | + **FP8 专家权重**（加载时在线量化 e4m3，per-expert scalar scale，W8A16），decode kernel p2_fused11r + prefill kernel nm_k9f | pkg_rs2，`RS_FP8=1 Q3_FUSED_MOE_KMOD=p2_fused11r NM_KMOD=nm_k9f` |
| RS3 | RS2 + bs32 bucket（NSB [1,4,8,16,32]，MNS 32，`Q3_FUSED_MOE_MAX_T=32`）+ NGB 8704（FP8 省出 ~7 GB/rank HBM，MML 仍 8192，KV 并发上限 34） | 同 RS2 |
| RS4 | RS2 + 多序列打包 prefill（P5，pack_max 16，pack_budget 4096，打包 bucket 512/1024/2048/4096） | pkg_rs3 + scripts/p5_patch_pack.py（P5P=1），`P5_PACK=1` |
| RS5 | RS3 配置的 71 min soak | 同 RS3 |

### 2.2 ShareGPT（seed 42，c16 另含 s123；每次 reset prefix cache）
| 指标 | RS0 (NJ, BF16) | RS1 (11p, BF16) | RS2 (FP8, W8A16) | RS3 (FP8+c32 cfg, W8A16) | H100 TP1 (BF16) | H100 TP2 (BF16) |
|---|---|---|---|---|---|---|
| c1 TTFT 均值/P99 | 71.5 / 100.5 | 70.5 / 98.2 | 68.6 / 92.6 | 64.5 / 88.7 | 50.1 / 66.6 | 62.0 / 86.8 |
| c1 TPOT | 6.45 | 6.44 | 5.86 | 5.85 | 4.75 | 3.90 |
| c4 TPOT / tok/s | 11.13 / 349 | 10.98 / 350 | 8.81 / 438 | 8.81 / 438 | 8.34 / 463 | 5.52 / 694 |
| c16 TTFT 均值/P99 | 353 / 1288 | 317 / 1304 | 320 / 1188 | 299 / 1141 | 123 / 146 | 148 / 233 |
| c16 TPOT（P99） | 22.53 (23.10) | 20.20 (20.78) | **15.86 (16.26)** | 15.81 (16.28) | 12.82 (13.20) | 8.65 (8.98) |
| c16 tok/s | 688 | 763 | **969** | 974 | 1224 | 1784 |
| c16 s123 TPOT / tok/s | 22.80 / 659 | 20.12 / 753 | 16.00 / 941 | 15.94 / 946 | — | — |
| c32 TPOT / tok/s / TTFT P99 | — | — | — | **20.29 / 1481 / 2405** | 未测 | 未测 |
| GSM8K 500 | 0.934 | 0.934 | **0.946** | **0.946**（RS6 = 同配置复测） | 0.930 | 0.930 |

表里的 H100 是 BF16，RS2/RS3 和它比是跨精度的。H100 W8A16、g7e 的同精度对照见 2.8。

### 2.3 1K→512（bench2 口径，random，ignore-eos）
| | RS0 (BF16) | RS1 (BF16) | RS2 (W8A16) | RS3 (W8A16) | H100 TP1 (BF16) | H100 TP2 (BF16) |
|---|---|---|---|---|---|---|
| c1 TTFT / TPOT / tok/s | 95 / 6.50 / 150 | 98 / 6.46 / 151 | 91 / 5.90 / 165 | 88 / 5.90 / 165 | 66 / 4.80 / 203 | 76 / 3.94 / 245 |
| c4 TPOT / tok/s | 10.81 / 356 | 10.79 / 357 | 8.99 / 426 | 8.97 / 427 | 7.75 / 505 | 5.21 / 742 |
| c16 TTFT / TPOT / tok/s | 485 / 21.86 / 700 | 482 / 19.00 / 800 | 434 / 16.14 / 940 | 429 / 16.08 / 943 | 187 / 11.46 / 1355 | 164 / 8.09 / 1905 |
| c32 TPOT / tok/s | — | — | — | 21.01 / 1420 | 未测 | 未测 |
| 4096→512 c16 TPOT / tok/s | 26.73 / 546 | 24.61 / 589 | 22.13 / 647 | 22.33 / 647 | 12.79 / 1159 | 8.48 / 1730 |

### 2.4 纯 decode 每步（ITL 中位数，ms；random in=128 / 1024，out 512 ignore-eos）
| bs | RS0 | RS1 | RS2/RS3 (FP8) |
|---|---|---|---|
| 1 | 6.44 / 6.47 | 6.44 / 6.46 | 5.85 / 5.90 |
| 4 | 10.76 / 10.87 | 10.28 / 10.66 | 8.39 / 8.77 |
| 16 | 21.09 / 19.34 | 18.25 / 17.08 | 14.00 / 14.25 |
| 32 | — | — | **16.75 / 16.75**（1520 / 1400 tok/s） |

FP8 下 bs16 的 MoE e2e ≈ (14.0−2.9)/48 ≈ 0.23 ms/层（单核微基准 0.20：4 rank 并发争 HBM）。bs16→bs32 步长只 +20%，所以 c32 吞吐 ≈ c16 的 1.5×。

### 2.5 各方向结论
1. **权重搬运粒度（外部参考方案①）→ 改为“去掉 padding 白算 + FP8 减字节”**：decode MoE kernel 的真实专家部分已 ~87% HBM 峰值、gu 每分区 12 KiB 大块，粒度没有空间；真正的空档是 padding 位置的串行 PE 尾巴（11o/11p，bs16 −13.5%）和 BF16 字节数（FP8，bs16 再 −23%）。
   - kernel 微基准（每层 µs，T=1/4/8/16/32）：11n 58/143/236/378/(2×T16) → 11p 58/142/226/338/442 → 11r(FP8) 48/94/140/202/252。
   - FP8 量化误差（真实权重）：per-expert-tensor 2.65% vs per-channel 2.64%（e4m3 浮点格式与 scale 粒度基本无关）→ 选 per-expert scalar，scale 并进 SiLU activation 的 per-partition scale 与 affinity，kernel 内零额外开销（per-channel 版 11q 因每专家一次碎描述符 scale DMA 多 ~60 µs/层，已弃）。
   - prefill nm_k9f：T=256/2048/8192 每层 836→661 / 1470→1234 / 3609→3517 µs。
2. **prefill 干扰 decode**：c16 TPOT 中只占 ~7%。dsbp/dsbpc 旋钮扫描（c16 s42）：默认 1/1 = TTFT 387/1379、TPOT 22.59；0/0 = 574/1098、21.85、ITL max 1049 ms；1/0 = 498/1008、22.10 —— 只在 P99 与均值/ITL 尾部之间搬移。TTFT P99 的根因是单 prefill/step（见 1.4），解决办法是多序列打包（2.6）。同机 16 device 的 P/D 分离：prefill 侧仍是单条串行，突发 P99 不变，只能消掉 ~7% 的 TPOT 干扰且要 2 device —— 未做。
3. **单芯片 EP4 vs TP4**：未上机。理由：同一步需要读的专家字节数相同（union 不变），EP 只会引入 rank 间负载不均（4 个 rank 各自 union 的最大值比均值高 15–20%）和 token all-to-all；本 kernel 已是带宽受限，TP 的“专家切分”在 I=192/rank 时 DMA 已是 12 KiB 连续块，EP 的大块优势只剩 ~13% 的峰值余量，且被负载不均吃掉。前一轮（ni2）也已实测“片内 EP（decode 代价）”无效。
4. **c32/c64**：kernel 扩到 T≤32（11p/11r）+ bs32 bucket；FP8 省出的 HBM 让 MML 8192 下 KV 并发到 34。c32 ShareGPT 1481 tok/s（RS0 c16 的 2.15×，超过 H100 TP1 BF16 c16 的 1224），但 TTFT P99 2.4 s（32 条突发串行 prefill）。c64 未做：kernel 上限 T=32，c64 需拆两次调用，MoE 每 token 成本不再下降，估计吞吐只比 c32 高 ~5–10%，且 KV 需 NGB ~17K（HBM 不够）。H100 BF16 没有测 c32。FP8 补测里 H100 W8A16 c32 是 2658 tok/s，为 RS3 c32 的 1.79×。c16→c32 的吞吐倍数：H100 W8A16 ×1.60（c16 取 MNS16 server 的 1659；按 MNS32 server 的 c16 1643 算是 ×1.62），trn2 RS3 ×1.52（见 2.8）。
5. **FP8 权重 + BF16 计算**：已做，是本轮最大收益（见上）。GSM8K 0.946（基线 0.934，H100 BF16 0.930，H100 / g7e W8A16 0.932），soak 见 2.7。
6. **EAGLE3**：未做。NxDI 路径不输出 aux hidden states；PR56 路径此前实测 k=1 接受率 68%、c1 TPOT 仅 −10%，c4 起无收益；只作延迟参考，优先级最低。

### 2.5b decode MoE FP8 kernel 的每专家开销（11/9，单逻辑核 microbench，`k/p2_fused11s.py` = 11r + 开关，`wsB_t1.py` LAYOUT=fp8s L=8，II_DGE=sw 同生产）
| 变体 | T=4 每层 ms | T=16 | T=32 | 数值 rel-fro（对反量化参考） |
|---|---|---|---|---|
| 11r（生产） | 0.1112 | 0.2195 | 0.2662 | 5.2e-3 / 4.7e-3 / 4.7e-3 |
| 11s 全关（应等于 11r） | 0.1107 | 0.2193 | 0.2663 | 相同 |
| SPLIT：GU 结果 PSUM→SBUF 的拷贝拆给 ACT + DVE 两个引擎 | 0.1113 | 0.2195 | 0.2663 | 相同 |
| SC：SEL 段 3 条 DVE 指令合成 2 条 scalar_tensor_tensor | 0.1114 | 0.2217 | 0.2662 | 相同 |
| ND：拷贝不再复制列，SEL 的 stationary 用 stride-0 broadcast 读 | 编译失败（PE stationary 不接受 stride-0 访问模式） | | | |
| HWW：专家权重 DMA 改走 hw-DGE，只用 Sync 队列 | 0.1096（−1.4%） | 0.2168（−1.2%） | 0.2655（−0.3%） | 相同 |
- 结论：**每专家的 ACT / DVE 指令和 GpSimd 生成描述符都不在关键路径上**，减少它们不会变快。按每个 slot 算，T=16 约 4.9 µs/slot（每层约 41 个 slot/核），其中权重 DMA 1.18 MB/slot，按 315–390 GB/s/核是 3.0–3.7 µs。剩下的时间在 PE 上：GU 和 DN 都是"权重当 moving 流过 PE、token 当 stationary（只占 T/128 列）"，每个 slot 要流过 9216 列。
- HWW 收益太小（≤1.4%），而生产里 hw-DGE 有过 BAD_OPCODE 历史，不值得再跑 soak 验证，**不采用**。
- **整步设备 profile（11/9 首次取到）**：用 `scripts/prof/ie_prof.sh` 对 RS7 缓存里最新一代的 decode NEFF 做 `neuron-explorer capture`（4 rank，profile rank 0，第 6 次执行；输入由 capture 生成，不是真实 hidden，路由分布和真实请求不同）。按 all-reduce 边界切层（`ie_ana.py`）：
  | 每层 µs（中位数） | bs16 ctx1024 | bs16 ctx2048 | bs1 ctx1024 |
  |---|---|---|---|
  | attention（含 qkv / o_proj、KV 写回） | 74.3 | 85.9 | 43.2 |
  | AR（attn） | 10.2 | 10.1 | 7.4 |
  | MoE | 157.2 | 157.9 | 53.2 |
  | AR（MoE） | 9.4 | 9.2 | 7.5 |
  | 整步（含 embed、lm_head + 采样 ≈0.52 / 0.31 ms） | 12.70 ms | 13.25 ms | 5.82 ms |
  注意：`ie_pick.py` 默认取到的是最老一代（NJ 基线 BF16）NEFF，因为缓存目录是三代配置复制下来的，每个 (B, ctx) 有 3 个 NEFF；改成按修改时间取最新（`/tmp/rs/pick2.py`）才是 RS7。
- **bs16 时 decode MoE 同时贴着 PE 和 HBM 两个上限**：MoE 段里 4514 条 MATMUL（每条约 311 ns），8 个 PE 小块（2 核 × 4 个 32 列）在整个窗口里一直忙；fp8 权重作为 moving 操作数，单条耗时和 BF16 版本一样（moving 384 列都是 326 ns），**说明 PE 是按 bf16 速率在跑，fp8 权重没有换来 PE 加速**（只减少了 HBM 字节）。HBM 侧：bs16 每层约 82 个专家 × 1.18 MB ≈ 97 MB / 157 µs ≈ 620 GB/s/rank，单 rank 的 HBM 份额约 725 GB/s。因此"对调 PE 角色"（kernel 已经用 4 个 32 列小块并行，对调后反而只剩 1 路）没有意义，已放弃；W8A8（activation 也量化成 fp8，让 PE 跑 fp8 速率）在 HBM 侧最多还剩约 15% 余量，而且有数值风险，本轮不做。
- PE 探针 `k/pe_probe.py` 的计时不可信（R=64 到 8192 条 matmul 的调用耗时都是约 175 µs），已弃用，改用整步 profile 里的单条指令耗时。

### 2.5c bs16 attention decode（11/9，同上 profile，第 20 层）
- decode attention 是 nkilib `attention_block_tkg` megakernel 一次调用（rmsnorm + qkv + RoPE + attention + o_proj + KV 写回）。bs16 ctx1024 每层 74 µs，但读取量只有约 17.6 MB（qkv + o 权重 9.2 MB + KV 8.4 MB），≈ 240 GB/s，**不是带宽受限**。
- 时间线：**前 10.5 µs 只有 GpSimd 在做 `DMA_GATHER_TRANSPOSE`**（按 block table 从分页 cache 收集 K，同时转成 [d, ctx]），PE、ACT、DVE 全空；接着约 5 µs 几乎空闲；21–57 µs 是计算主体（MATMUL 420 条、合计 110 µs，LDWEIGHTS 388 条、47 µs）；57–68 µs 有大量 Sync 信号量等待，加上 KV 写回。K 的收集转置在 ctx1024 时 14.3 µs、ctx2048 时 24.2 µs（2 核合计），基本随 ctx 线性增长。bs1 时没有这一段（走的是另一条路径）。
- 结论：bs16 每层大约有 10–17 µs 的串行段（K 收集转置）加 5 µs 空档，48 层合计每步约 0.7–1.0 ms，相当于 c16 TPOT 14.4 ms 的 5–7%。**没有现成开关能去掉它**：kernel 支持 `K_cache_transposed=True`（直接按 [d, ctx] 存 K），但 vllm-neuron 里所有模型都传 False，prefill 写 cache（qkv kernel 内写）、分段 attention、打包用的 full 分支都只支持 [ctx, d] 布局。要做就得同时改这三条路径加 decode，这是多天的工作，加上每次约 1 h 的重编译。本轮不改，推荐配置不变。
- 可以继续做的：(1) 转置 K cache 布局（收益上限约每步 0.7–1.0 ms，ctx 越长越大）；(2) 在 nkilib kernel 里让 qkv 权重预取 / qkv 投影和 K 收集重叠（需要改 nkilib 源码）。

### 2.6 多序列打包 prefill（P5）
- 08:20 UTC 第一次上机（RS4 = RS2 + pkg_rs3 + `P5_PACK=1`）：48 层图编译通过（+27 NEFF），NaN 检查 PASS；但 **打包从未生效**——steplog 里每个 prefill step 都只有 1 条请求。原因：vLLM 0.29 的 `coordinator.find_longest_cache_hit` / `get_computed_blocks` 返回 3 元组，补丁按 2 元组解包，前缀命中探测抛异常后保守地关闭了打包（日志 "P5 prefix-hit probe failed ... packing disabled"）。这次的打包对比无效。
  - 顺带得到的基线：同一组 16 条 prompt 贪心 64 token，"逐条（c1）" vs "同时发（c16，不打包）" 相同 12/16，两次 c16 之间相同 13/16，分歧都出现在第 5–49 个 token。即 **decode 的 batch 组成本身就会改变贪心结果**（bf16 下不同 T 的 kernel 数值不同），打包正确性要和这个基线比，而不是要求 100% 相同。
- 修复（`scripts/p5_patch_pack.py` 两处改为取 `[1]`，只动调度器的 Python 代码，不需要重新编译；重启约 40 min，new NEFF=0）后，09:05 UTC 复测：
  - **打包生效**：16 条 prompt（37–500 token，合计 4064）同时发，prefill step 最多装 8 / 12 条（两次运行各有 2 个多请求 step），16 条全部完成的墙钟时间 **2.06 s → 1.24 s（−40%）**。
  - 数值：逐条 vs 打包相同 11/16、12/16（两次）；打包 vs 打包 13/16。不打包时的对照是 12/16 和 13/16，处在同一水平。所有分歧都在第 9 个 token 之后，**首 token（直接来自打包 prefill 的 logits）16/16 一致**，所以和 batch 组成噪声分不开，没有看到打包特有的错误。更强的检查是下面的 GSM8K（并发请求下会大量打包）。
  - **基准（RS4 vs RS2，同为 FP8 + 11r，唯一差别是打包；RS4 拿到了 bench 锁，RS2 那次是 LOCK TIMEOUT 后跑的，但它的 c16 数字和 RS3 15.81 一致）**：
    | 负载 | RS2 TTFT 均值 / 中位 / P99 | RS4 TTFT 均值 / 中位 / P99 | RS2 TPOT | RS4 TPOT | RS2 tok/s | RS4 tok/s |
    |---|---|---|---|---|---|---|
    | ShareGPT c16 s42 | 320 / 224 / 1188 | 251 / 269 / **308** | 15.86 | **14.70** | 969 | **1047** |
    | ShareGPT c16 s123 | 312 / 221 / 1255 | 229 / 255 / **380** | 16.00 | 14.62 | 941 | 1040 |
    | ShareGPT c4 s42 | 174 / 186 / 293 | 128 / 124 / 200 | 8.81 | 8.64 | 438 | 449 |
    | ShareGPT c4 s123 | 137 / 111 / 325 | 127 / 115 / 253 | 9.06 | 8.83 | 418 | 427 |
    | ShareGPT c1 | 68.6 / 63.4 / 92.6 | 62.3 / 55.6 / 89.4 | 5.86 | 5.85 | 167 | 168 |
    | random 256 c4 | 156 / 163 / 263 | 108 / 121 / 193 | 8.70 | 8.43 | 431 | 453 |
    | random 1024 c4 | 210 / 224 / 351 | 229 / 270 / 281 | 9.29 | 8.89 | 396 | 410 |
    | random 3000 / 6000 c4、7000 c16 | 与 RS2 相同（±1%） | | | | | |
    | GSM8K 500 | 0.946 | **0.946** | | | | |
    - 结论：**ShareGPT c16 TTFT P99 −74%（1188 → 308 ms），H100 TP1（BF16）是 146 ms，差距从 8.1× 缩到 2.1×**；c16 TPOT 也 −7%、吞吐 +8%（prefill step 数减少，对 decode 的插入干扰变少）。长 prompt（≥3000）合计超过 pack_budget 4096，不会打包，结果与 RS2 相同，符合预期。均值和中位数基本不变或略升（打包 step 里最短的请求要等最长的一起完成）。
    - steplog：从重启到 bench 结束共 769 个 prefill step，其中 58 个是多请求 step（2–15 条，含打包对比的 4 个）。
    - 单请求 TTFT 扫描与 RS2 相同（128→53 ms … 8000→392 ms），说明打包补丁对单条路径没有额外开销。
  - **soak 通过**（rs-b，10:08–11:03 UTC，RS4 同一个服务器进程，c1/c4/c16 混合）：50.7 分钟，**5587 个请求，0 失败、0 坏样本**；ShareGPT 2466 条，随机长 prompt 3121 条（≥4096 token 2085 条、≥6000 token 1330 条，prefill 共 1398 万 token）；金丝雀 65/65；开始、中间、结束三次 NaN 检查都 PASS；服务日志错误行 0。最长 TTFT 4.6 s（RS5 是 7.7 s，但 RS5 有 c32，不完全可比）。从重启到 soak 结束共 6218 个 prefill step，其中 261 个是多请求打包 step，整个过程中前缀命中探测没有再失败。

- **RS7 = RS3（c32 配置）+ 打包**（pkg_rs3，`P5P=1`，NSB [1,4,8,16,32]，MNS32，NGB8704，`Q3_FUSED_MOE_MAX_T=32 P5_PACK=1`；以 rs_RS3 缓存为底重新编译约 1 h；bench 期间 knob 文件里 pack_max=16，是打包对比脚本最后写入的值，pack_max=32 没有测）。打包对比：16 条同时发的墙钟 6.90 s（逐条）→ 1.17–1.20 s，一个 step 最多 14 条；三组相同率 11–12/16，和 RS4 同级。
  | ShareGPT | RS3 TTFT 均值 / 中位 / P99 | RS7 TTFT 均值 / 中位 / P99 | RS3 TPOT | RS7 TPOT | RS3 tok/s | RS7 tok/s |
  |---|---|---|---|---|---|---|
  | c32 s42（128 条） | 461 / 215 / 2405 | 354 / 300 / **581** | 20.29 | **17.24** | 1481 | **1753** |
  | c16 s42 | 299 / 213 / 1141 | 243 / 255 / **276** | 15.81 | **14.43** | 974 | **1075** |
  | c16 s123 | 293 / 196 / 1213 | 265 / 266 / 381 | 15.94 | 14.44 | 946 | 1049 |
  | c4 s42 | 169 / 180 / 287 | 107 / 97 / 208 | 8.81 | 8.85 | 438 | 432 |
  | c1 | 64.5 / 58.1 / 88.7 | 63.6 / 55.8 / 96.5 | 5.85 | 5.85 | 168 | 168 |
  random 输入（256–7000，c1/c4/c16）与 RS4 相同（±2%），单请求 TTFT 扫描也相同；GSM8K 500 = **0.946**。
  - **c32 TTFT P99 −76%、TPOT −15%、吞吐 +18%**：c32 时 prefill 串行排队更长，打包收益比 c16 更大；TPOT 变好是因为 prefill step 少了，插进 decode 的次数也少了。**c32 吞吐 1753 tok/s = H100 TP1 BF16 c16（1224）的 1.43×**（跨精度、跨并发，只作参考）。同精度同并发：H100 W8A16 c32 2658 tok/s，RS7 是它的 0.66×；g7e W8A16 c32 1790 tok/s，RS7 是它的 0.98×（见 2.8）。
  - c32 soak 已跑完，0 失败，见 2.7。
- **c32 下调打包参数（11/9）**：
  - 2a. pack_max 16 → 32（budget 4096 不变，运行时改 knob，不重启；结果目录 `rs_results/RS7pm32`）：sg c32 TTFT 均值 / 中位 / P99 327 / 338 / 586（RS7：354 / 300 / 581），TPOT 17.40（17.24），1740 tok/s（1753）；c16 s42 14.28 / 1085、s123 14.70 / 1033（RS7：14.43 / 1075、14.44 / 1049）；其它负载 ±2%；GSM8K 0.946。单个 step 最多装到 19 条（RS7 是 16），但 4096 的 budget 先成为上限。**没有收益，结果在噪声范围内**，所以不改推荐值。
  - 2b. RS8 = RS7 + 打包 bucket 增加 8192（`P5_PACK_BUCKETS=512,1024,2048,4096,8192`，+1 NEFF）+ pack_max 32 + pack_budget 8192（结果目录 `rs_results/RS8`）：bench 期间 754 个 prefill step 中 61 个是打包 step，最多一次 29 条，10 个 step 超过 4096 token。sg c32 TTFT 均值 / 中位 / P99 **429 / 458 / 544**（RS7：354 / 300 / 581），TPOT 17.26，1737 tok/s；sg c16 s123 P99 461（RS7 381）；rn3000 c4 TTFT 均值 543（RS7 471）；其它 ±2%；GSM8K 0.946。**P99 略好，但均值、中位数明显变差，吞吐不变**：打包超过 4096 token 后，step 里每条请求都要等更长的 prefill 完成。**不采用**，推荐配置保持 pack_max 16 / budget 4096。

### 2.7 稳定性
- **RS5 = RS3 推荐配置 soak 通过**（rs-d，07:35–08:47 UTC，FP8 + T≤32 kernel + bs32 bucket，MNS32，NGB8704）：71.2 分钟，7 段 × 10 分钟，并发 c1/c4/c16/c32 混合（c32 占 3869 条）。共 **8376 个请求，0 失败、0 坏样本**；ShareGPT 4154 条（输出 121 万 token），随机长 prompt 4222 条（其中 ≥4096 token 2833 条、≥6000 token 1809 条，prefill 共 1896 万 token）；金丝雀 91/91；开始、中间、结束三次 NaN 检查都 PASS；服务日志里 nrta / SEQUENCER / BAD_OPCODE / NRT_TIMEOUT / Traceback 都是 0。最长 e2e 37.9 s，最长 TTFT 7.7 s（出现在 c32 加长 prompt 的串行 prefill 排队里，见 1.4）。原始数据：`/data/cache/rs_results/RS5/soak/`。
- **RS7 = 推荐配置 soak（c32 + 打包）跑完，0 失败**（rs-b，11/8 12:19–13:10 UTC，RS7 bench 后的同一个服务器进程，SOAK32=1，knob pack_max 16 / pack_budget 4096）：50.8 分钟，5 段 × 10 分钟，并发 c1/c4/c16/c32 混合（c32 占 2827 条）。**6089 个负载请求，0 失败**；ShareGPT 3054 条（输出 90 万 token），随机长 prompt 3035 条（≥4096 token 2037 条、≥6000 token 1295 条，prefill 共 1362 万 token）；金丝雀 65/65；开始、中间、结束三次 NaN 检查都 PASS；服务日志错误行（nrta / SEQUENCER / BAD_OPCODE / NRT_TIMEOUT / Traceback / NaN 等）0；健康检查一直是 200。从启动到 soak 结束共 6534 个 prefill step，其中 373 个是多请求打包 step。最长 e2e 37.0 s，最长 TTFT 7.1 s。
  - **2 条"坏样本"复查（11/9，在 RS7 服务器上重跑，全文存在 `/data/cache/rs_results/RS7_final/soak/charrun_rerun{,2,3}.json`）**：两条都触发了重复字符启发式 `(.)\1{40,}`。soak 只存了前 80 个字符，所以按 `L=len(prompt)//4` 和输出开头在 ShareGPT 里反查出原 prompt（两条的输出开头都和 soak 记录逐字一致）。
    1. "difference between brute force and dictionary attack in tabular format"（L=17）：c1 贪心重跑两次结果逐字一致，都复现了告警。触发处是 markdown 表格：表头单元格用 65 个空格补齐列宽，分隔行是 88 个 `-`（`|----…----|`）。**是正常内容，属于误报**。c16 批次里的那次输出换了一种表格写法，没有触发。
    2. 一段 CSS 代码（`body{ box-sizing: border-box; … /*-------`，L=778）：prompt 本身就有 58、52、49 个连续 `-` 组成的注释分隔线 `/*------…*/`。c1 重跑 1 次、和 31 条其它 ShareGPT 组成 c32 批次重跑 3 次，输出都没有出现 ≥41 个连续字符（不同 batch 组成下，贪心输出在后段会分叉，见 2.6）。soak 里那次最可能是思考过程中照抄了用户的注释分隔线。**没有直接复现，但 prompt 里有同样的长串，判断为正常内容**。
    - 结论：两条都不是数值退化，RS7 soak 按 6089 个请求、0 失败、0 真实坏样本计。
  - 原始数据：`/data/cache/rs_results/RS7_final/soak/`（req.jsonl、slog、summary.txt、nan_*.txt；`rs_results/RS7` 在 11/9 恢复服务器时被 rs_main 覆盖，以 RS7_final 为准），本地摘要 `results/RS7/soak_summary.txt`。


### 2.8 跨平台同精度对比（H100 BF16 / W8A16、g7e BF16 / W8A16；11/9 补）
2.2、2.3 里的 H100 是纯 BF16，所以 RS2 起的配置和它比都是 **W8A16 对 BF16 的跨精度比较**（结论摘要里旧的 1.13× 就属于这种）。本节按精度分开比。数据来源和口径：
- **H100 BF16**：`<GPU_H100_BF16>/REPORT.md`，p5.48xlarge，vLLM 0.33.0，TP1。启动命令 `--dtype bfloat16`，没有 `--quantization` 和 `--kv-cache-dtype`。server-tp1.log 里是 `dtype=torch.bfloat16, quantization=None, kv_cache_dtype=auto`，权重、计算、KV 都是 BF16。
- **H100 W8A16**：同一天的 FP8 补测 `<GPU_H100_FP8>/`，单卡 TP1，镜像相同。启动参数 `--quantization online --quantization-config '{"moe":{"weight":"fp8_per_tensor_static"}}' --moe-backend marlin`，只把专家量化成 e4m3，每个专家的 w13、w2 各一个标量 scale；attention、dense、lm_head 仍是 BF16；Marlin 反量化后用 BF16 计算；KV 是 auto（BF16）。另加 `--block-size 32 --num-gpu-blocks-override 4352 / 8704`，KV 容量和 trn2 一样；c32 用 MNS 32 的 server。量化方案和 trn2 RS2 相同，kernel 不同。
- **g7e**：`<GPU_G7E>/`，g7e.4xlarge，1 张 RTX PRO 6000 Blackwell Server Edition（96 GB GDDR7，规格带宽约 1.8 TB/s），vLLM 0.33.0（镜像 digest 和 H100 相同），TP1。BF16 组和 W8A16 组的参数分别和 H100 的 BF16、W8A16 相同（server 日志已核对：BF16 组 `quantization=None`，W8A16 组 `Using MARLIN Fp8 MoE backend`，KV 都是 auto）。server 参数、block 32 和 KV 容量设置也和 H100 FP8 补测相同。c1–c16 用 MNS 16 的 server，c32 用另一个 MNS 32 的 server。
- **trn2**：RS0、RS1 的权重、计算、KV 全是 BF16。RS2 起只有专家权重是 FP8，计算和 KV 都是 BF16（W8A16）。server.log 里都是 `dtype=torch.bfloat16`、`quantization=None`、`kv_cache_dtype=auto`（RS0、RS1、RS7_final 已核对），因为 FP8 在自研模型代码里做，vLLM 层看不到。
- **口径一致**：三个平台都是贪心（GPU 用 `--override-generation-config '{"temperature":0.0}'`），MML 8192，MNBT 8192。GPU 的 bench 脚本 `fp8_bench.sh` 两边 sha256 相同，其中 ShareGPT 的 `sg()` 和本目录 `rs_bench.sh` 逐字相同。ShareGPT 各档 `total_input_tokens` 三边逐一相同（c1 2510、c4 8869、c16 17268、c32 31028），1K / 4K→512 的输入总量也相同（49152 / 131072）。
- **调度不同**：GPU 都是 vLLM 默认调度（chunked prefill，prefill 和 decode 混在同一个 step），trn2 是 1.4 节说的单条 prefill 加 2.6 节的打包。

**(a) 纯 BF16：trn2 RS1 对 H100 BF16、g7e BF16**（RS1 = 11p BF16 kernel，去掉 padding 白算，不打包，MNS 16，没有 c32）

| 并发 | 指标 | trn2 RS1（BF16） | H100 TP1（BF16） | g7e（BF16） | RS1 / H100 | RS1 / g7e |
|---|---|---|---|---|---|---|
| c1 | TTFT 均值 / 中位 / P99 | 70.5 / 63.1 / 98.2 | 50.1 / 56.8 / 66.6 | 39.2 / 39.5 / 54.8 | P99 1.47× | P99 1.79× |
| c1 | TPOT / tok/s | 6.44 / 152.2 | 4.75 / 206.6 | 6.72 / 147.5 | 1.36× / 0.74× | 0.96× / 1.03× |
| c4 | TTFT 均值 / 中位 / P99 | 185.6 / 201.7 / 293.1 | 124.0 / 108.0 / 581.2（复跑 P99 122.9） | 86.1 / 89.2 / 108.6 | P99 2.4×（按复跑） | P99 2.7× |
| c4 | TPOT / tok/s | 10.98 / 350.3 | 8.34 / 462.8 | 12.68 / 311.9 | 1.32× / 0.76× | 0.87× / 1.12× |
| c16 | TTFT 均值 / 中位 / P99 | 317.1 / 223.9 / 1304.0 | 123.3 / 125.0 / 145.9 | 148.1 / 153.2 / 161.5 | P99 8.9× | P99 8.1× |
| c16 | TPOT（P99）/ tok/s | 20.20（20.78）/ 763.0 | 12.82（13.20）/ 1224.2 | 22.15（22.52）/ 714.3 | **1.58× / 0.62×** | **0.91× / 1.07×** |
| 1K→512 c16 | TPOT / tok/s | 19.00 / 800 | 11.46 / 1355 | 19.84 / 782 | 1.66× / 0.59× | 0.96× / 1.02× |
| 4K→512 c16 | TPOT / tok/s | 24.61 / 589 | 12.79 / 1159 | 22.12 / 666 | 1.92× / 0.51× | 1.11× / 0.88× |
| — | GSM8K 500 | 0.934 | 0.930 | 0.930 | | |

**(b) W8A16 对 W8A16：trn2 RS7 对 H100 W8A16、g7e W8A16**（三方都是只把专家权重量化成 FP8 e4m3、每个专家一个标量 scale、BF16 计算。kernel 不同：trn2 是自研 NKI 的 p2_fused11r / nm_k9f，GPU 是 vLLM 的 Marlin）

| 并发 | 指标 | trn2 RS7（W8A16 + 打包） | H100 TP1（W8A16） | g7e（W8A16） | RS7 / H100 | RS7 / g7e |
|---|---|---|---|---|---|---|
| c1 | TTFT 均值 / 中位 / P99 | 63.6 / 55.8 / 96.5 | 50.1 / 57.6 / 67.4 | 32.7 / 36.3 / 42.8 | P99 1.43× | P99 2.3× |
| c1 | TPOT / tok/s | 5.85 / 167.7 | 4.92 / 199.7 | 5.66 / 175.0 | 1.19× / 0.84× | 1.03× / 0.96× |
| c4 | TTFT 均值 / 中位 / P99 | 107.3 / 96.7 / 207.7 | 96.4 / 104.6 / 117.5 | 60.6 / 64.8 / 73.5 | P99 1.8× | P99 2.8× |
| c4 | TPOT / tok/s | 8.85 / 432.4 | 6.49 / 592.7 | 8.92 / 440.5 | 1.36× / 0.73× | 0.99× / 0.98× |
| c16 | TTFT 均值 / 中位 / P99 | 242.8 / 255.3 / 276.1 | 144.0 / 147.7 / 160.0 | 109.4 / 101.0 / 159.7 | P99 1.7× | P99 1.7× |
| c16 | TPOT（P99）/ tok/s | 14.43（14.94）/ 1074.9 | 9.38（9.55）/ 1659.2 | 13.81（14.14）/ 1137.2 | **1.54× / 0.65×** | **1.04× / 0.95×** |
| c32 | TTFT 均值 / 中位 / P99 | 354.1 / 299.5 / 580.5 | 186.8 / 195.6 / 254.3 | 211.1 / 200.2 / 290.0 | P99 2.3× | P99 2.0× |
| c32 | TPOT（P99）/ tok/s | 17.24（17.89）/ 1753.0 | 11.40（11.81）/ 2658.0 | 17.16（17.66）/ 1789.8 | **1.51× / 0.66×** | **1.00× / 0.98×** |
| 1K→512 c16 | TPOT / tok/s（trn2 用 RS3） | 16.08 / 943 | 9.25 / 1628 | 12.89 / 1182 | 1.74× / 0.58× | 1.25× / 0.80× |
| 4K→512 c16 | TPOT / tok/s（trn2 用 RS3） | 22.33 / 647 | 11.19 / 1240 | 16.92 / 844 | 2.00× / 0.52× | 1.32× / 0.77× |
| — | GSM8K 500 | 0.946 | 0.932 | 0.932 | | |

**(c) 怎么读**
- **同精度下，trn2 和 H100 的 decode 差距在两种精度下几乎一样**：c16 TPOT，BF16 是 1.58×（RS1），W8A16 是 1.54×（RS7）。原因是 H100 用上 FP8 后自己也快了：c16 TPOT 12.82 → 9.38 ms（−27%）。和 FP8 补测里参数对齐的 BF16 组（12.60 ms）比是 −26%。所以 RS7 对 H100 BF16 的 1.13× 基本只反映了 trn2 这边上了 FP8、H100 那边没上，不能说成 trn2 的 decode 追近了 H100。
- **RS1 → RS7 的收益拆分**（c16 s42，按 RS1 → RS2 → RS3 → RS7 的叠加顺序拆，**不是正交分解**）：TPOT 共降 5.77 ms（20.20 → 14.43），其中 FP8 权重（RS1 → RS2，20.20 → 15.86）占 4.34 ms，约 75%；T≤32 + bs32（RS2 → RS3）占 0.05 ms；打包（RS3 → RS7，15.81 → 14.43）占 1.38 ms，约 24%。吞吐共增加 312 tok/s（763 → 1075），FP8 占 206（约 66%），打包占 101（约 32%）。
- **TTFT P99 的改善几乎都来自打包，不来自 FP8**：c16 P99 RS1 1304 → RS2 1188 → RS3 1141 → RS7 276 ms。对 H100 BF16 的 P99 比值从 8.9× 降到 1.9×，靠的是打包。
- **和 g7e 比**：BF16 时 trn2 略占优（c16 TPOT 0.91×，c4 0.87×），W8A16 时基本持平（c16 1.04×，c32 1.00×）。原因是 g7e 从 FP8 拿到的收益更大：c16 TPOT 22.15 → 13.81 ms（−38%），而 trn2 RS1 → RS2 是 −21%。g7e 的显存带宽更紧（规格约 1.8 TB/s），权重字节减半的收益更直接（这是 g7e README 的解释，**〔未单独剖析〕**）。prefill 比重大的档位 g7e 明显更好：1K→512 c16 吞吐是 RS3 的 1.25×，4K→512 是 1.31×；TTFT P99 也更低（1K→512 c16 482 对 1439 ms，4K→512 c16 2123 对 3332 ms）。
- **口径提醒**：GPU 的 W8A16 走 Marlin（先反量化再做 BF16 MMA），c1 时比 BF16 还慢（H100 4.92 对 4.75 ms），长 prefill 也更慢。所以 W8A16 对 W8A16 比较的是各自现成的最佳实现，不是理论最优。H100 有官方 FP8 checkpoint（W8A8 block），c16 TPOT 8.98 ms、1729 tok/s，比 W8A16 再快约 4%；trn2 没有 W8A8 对应项，没有放进表里。

## 3. 推荐配置与预期收益

**推荐 = RS7 配置**（FP8 专家权重 + T≤32 fused decode kernel + bs32 bucket + 多序列打包 prefill）：
```
PKG=pkg_rs3 (+ scripts/p5_patch_pack.py 修复版，rs_srv.sh P5P=1)  NM_MOE=1  Q3_FUSED_MOE=1 Q3_FUSED_MOE_KMOD=p2_fused11r Q3_FUSED_MOE_MAX_T=32 RS_FP8=1 NM_KMOD=nm_k9f P5_PACK=1
WSA_FN=1 II_DGE=sw P3_KVIX=1 P4_DSBPC=1 P4_DSBP=1        # 同 NJ；knob json：pack_max 16，pack_budget 4096
vllm serve ... --tensor-parallel-size 4 --max-num-seqs 32 --max-model-len 8192 --max-num-batched-tokens 8192 --num-gpu-blocks-override 8704
  neuron_config: num_seqs_buckets [1,4,8,16,32]，on_device_sampling all_greedy（打包要求），其余（token buckets、decode ctx buckets、kvseg）同 NJ
```
启动脚本：`scripts/rs_srv.sh`（`P5P=1 NSB="[1,4,8,16,32]" MNS=32 NGB=8704 NMX="Q3_FUSED_MOE_KMOD=p2_fused11r RS_FP8=1 NM_KMOD=nm_k9f Q3_FUSED_MOE_MAX_T=32 P5_PACK=1"`）；暖缓存 `/data/cache/rs_RS7`（节点 hostPath）。

相对 NJ 的收益（同口径实测）：ShareGPT c1 TPOT −9%；**c16 TPOT 22.53 → 14.43（−36%）、吞吐 688 → 1075 tok/s（+56%）、TTFT P99 1288 → 276 ms（−79%）**；**c32 吞吐 1753 tok/s（NJ c16 的 2.55×），TTFT P99 581 ms**；GSM8K 0.934 → 0.946；50.8 min c32 soak 6089 个请求 0 失败（2 条重复字符告警经重跑确认是表格补齐 / 照抄 prompt 的注释分隔线，见 2.7）。
与 H100 TP1 的差距（同精度，见 2.8）：W8A16 对 W8A16，c16 TPOT **1.54×**、吞吐 **0.65×**、TTFT P99 1.7×，c32 吞吐 **0.66×**，c1 TPOT 1.19×；BF16 对 BF16（RS1），c16 TPOT 1.58×、吞吐 0.62×。和 g7e 同为 W8A16 时基本持平：c16 TPOT 1.04×，c32 吞吐 0.98×。
跨精度参考（RS7 W8A16 对 H100 BF16，只作参考）：c16 TPOT 1.76× → 1.13×，c16 吞吐 0.56× → 0.88×，c16 TTFT P99 8.8× → 1.9×；c1 TPOT 1.36× → 1.23×；trn2 c32 吞吐是 H100 BF16 c16 的 1.43×。

只要 c≤16 时可以用 RS4（MNS16，无 bs32 bucket，暖缓存 `/data/cache/rs_RS4`）：c16 数字与 RS7 基本相同（TPOT 14.70、1047 tok/s、TTFT P99 308）。不需要打包时用 RS3（pkg_rs2，`/data/cache/rs_RS3`）：c16 TPOT 15.81、974 tok/s，但 TTFT P99 1.1 s、c32 时 2.4 s。

若不接受 FP8：RS1（pkg_rs1，`Q3_FUSED_MOE_KMOD=p2_fused11p`，纯 BF16，数值与 NJ 同级、GSM 0.934）c16 TPOT −10% / 吞吐 +11%，零风险。

## 4. 剩余差距与下一步
- decode bs16（FP8）：c16 TPOT 14.43 ms，H100 W8A16 9.38、H100 BF16 12.82、g7e W8A16 13.81。整步设备 profile（2.5b、2.5c）给出每层 MoE 157 µs（60%）、attention 74 µs（28%）、两次 AR 20 µs、lm_head + 采样 0.5 ms。MoE 同时接近 PE（bf16 速率）和 HBM 上限，每专家指令开销与 hw-DGE 已实测无收益；attention 有约 0.7–1.0 ms/步的串行 K 收集转置，需要改 cache 布局。剩余的大方向：W8A8 decode MoE（PE 跑 fp8 速率，上限约 15%，有数值风险）、转置 K cache 布局。
- **11/9 用户决定：先停在 RS7，不继续 W8A8 和转置 K cache 这两项。**
- TTFT P99：打包后 c16 降到 276–380 ms，c32 581 ms（H100 TP1 c16：BF16 146、W8A16 160；c32 W8A16 254）。剩下的差距：(1) 打包的总 token 上限 pack_budget 4096，长 prompt 仍然串行；(2) 打包 step 里最短的请求要等最长的；(3) 单条 prefill 本身仍比 H100 慢（ShareGPT c1 TTFT 62 vs 50 ms，H100 BF16 和 W8A16 都是 50.1）。
- RS3 + 打包（RS7）已做；pack_max 32 / budget 8192 也试过，没有收益（2.6）。
- 未做：EP4（理由 2.5-3）、P/D 分离（2.5-2）、c64（2.5-4）、EAGLE3（2.5-6）、整步设备 profile（1.3 注）。

## 5. 文件
- **pod 状态（11/9 16:1x UTC+8）**：rs-b 已删除（`kubectl -n trn2-perf delete pod rs-b`，rc 0；删除后节点 <NODE> 上已没有 pod 申请 neuron device，allocatable 16）。节点本身、节点 hostPath `/mnt/k8s-disks/0/trn2-perf`（pod 内 `/data`）下的暖缓存 `/data/cache/rs_RS7`（以及 rs_RS3、rs_RS4、rs_RS8 等）和 `/data/cache/rs_results/` 都保留；rs_* 的 native_neff_cache 也已在 S3 `s3://<NEFF_CACHE_BUCKET>/v1/vn0.29.0.1.2.0_cc2.28.7631.0/rs_RS*/`（rs_RS0..7，是 11/8 灌入的；rs_RS8 没推）。
- **重启 RS7**：在 <NODE> 上重建 1 device 的 pod（模板 `/tmp/trn2-hyd/wsA/kdev-a2.yaml`，`schedulerName: my-scheduler`，hostPath 挂到 `/data`；按 neff-cache 规则加 `serviceAccountName: neff-cache` 和 ConfigMap `neff-cache-scripts`）。把 `scripts/` 下的 rs_srv.sh、rs_main.sh、rs_bench.sh、p5_patch_pack.py 等放到 pod 的 `/tmp/ni2e/`，然后执行：
  `TAG=RS7 PKG=/data/cache/rs_pkg/pkg_rs3 NMV=1 SRC=/data/cache/rs_RS3 PORT=8802 COMM=62802 P5P=1 KEEP=1 NSB="[1,4,8,16,32]" MNS=32 NGB=8704 NMX="Q3_FUSED_MOE_KMOD=p2_fused11r RS_FP8=1 NM_KMOD=nm_k9f Q3_FUSED_MOE_MAX_T=32 P5_PACK=1" PHASES=nan bash /tmp/ni2e/rs_main.sh`
  启动后把 knob 写成 `{"dsbpc":1,"dsbp":1,"pack_max":16,"pack_budget":4096}`（`/tmp/ni2_knobs_RS7.json`）。同节点暖缓存实测约 40 min READY、new NEFF=0（11/9 06:32 那次）。注意：(a) 启动前要把 `$C/vc/neuron/` 下的 compile_cache 和 torch_compile 移开（rs_main 在缓存目录已存在时不会做，参考 `_joblogs/job_RS7r.sh`），否则可能 NCC_EVRF059；(b) rs_main 结束时会 `rm -rf /data/cache/rs_results/RS7`，正式结果以 RS7_final 和本地 `results/` 为准；(c) 换节点时，节点本地缓存没有用，要先用 `neff_cache.sh pull rs_RS7` 从 S3 拉 NEFF，trace / warmup 仍需约 28 min。
- 报告：`qwen3-30b/REPORT.md`；多序列打包设计：`multi_prefill_design.md`
- 包：`pkg_rs1/`（11p）、`pkg_rs2/`（FP8：model_bf16.py RS_FP8 段、q3_fused_moe.py、nm_glue.py、p2_fused11r.py、nm_k9f.py）、`pkg_rs3/`（pkg_rs2 + P5 打包）
- kernel 与微基准：`k/`（p2_fused11o/p/q/r/s.py、wsB_t1.py、nm_e2e8.py、pa_test.py、pe_probe.py）
- 脚本：`scripts/`（rs_srv.sh、rs_main.sh、rs_bench.sh、rs_soak.*、p5_patch_pack.py、p5_pack_check.py、rsh/rrun/rship 工具）
- 结果（本地 `results/`）：每个目录顶层是整理过的摘要（`summary_blog.txt`、`soak_summary.txt` 等），`node_raw/` 是 11/9 从节点 `/data/cache/rs_results/<目录名>/` 原样拉回的原始数据（bench json 和日志、server.log、info、nan / gsm / ttft、soak 的 req.jsonl / slog 等）。
  - RS0–RS6：各配置的 bench（RS5 = RS3 配置的 71 min soak，RS6 = RS3 同配置复测 GSM8K）；RS4 = 打包 + soak；**RS7_final = 推荐配置 RS7 的 bench + c32 soak + 2 条告警重跑全文**（`node_raw/soak/charrun_rerun*.json`）；RS7_prev_out = 同一次运行恢复前的输出目录副本；RS7pm32 = pack_max 32；RS8 = budget 8192；KNOB = dsbp / dsbpc 旋钮扫描（原 `/tmp/rs_out_RSP`）；RSP = 旋钮扫描用的服务器（NJ 配置）；**RS7_restart_1109** = 11/9 停服务器做 microbench 后恢复 RS7 的那次输出（只有 NaN 检查，节点上叫 `RS7`）。
  - `_joblogs/`：pod 内每个任务的脚本和日志（job_*.sh、log_*.txt，含 microbench、PE 探针、profile、告警重跑脚本）；`_prof/`：整步 decode profile 的 sum.txt / cap.log（prof.json 每个约 200 MB，没拉回，仍在 rs-b 的 `/tmp/ie_prof_*`）；`PROF/decode_profile_RS7.txt`：整步 profile 的分析输出。
  - 校验：`results/MANIFEST_node_raw.txt` 是节点上生成的清单（820 个文件、125,792,085 字节），本地按清单逐个核对大小，全部一致。传输路径是节点 → S3 `s3://<NEFF_CACHE_BUCKET>/results/qwen3-30b-rs/20261109/rs_results/`（用只读挂载 hostPath 的一次性 pod `rs-results-push`，已删除）→ 本机 `aws s3 sync`。
- 跨平台对照：H100 BF16 `<GPU_H100_BF16>/REPORT.md`、`results/{TP1,TP2}`、`results/SUMMARY_all.txt`、`results/server-tp*.log`；H100 FP8 补测 `<GPU_H100_FP8>/REPORT.md`、`results/{w8a16,w8a16x32}/summary.txt`、`results/server-w8a16*.log`；g7e `<GPU_G7E>/README.md`、`results/COMPARE.md`、`results/{bf16,bf16x32,w8a16,w8a16x32}/summary.txt`、`results/server-*.log`、`scripts/fp8_bench.sh`。
- profile 分析脚本：`scripts/prof/`（ie_prof.sh 抓取，pick2.py 按修改时间选最新一代 NEFF，ie_ana.py 按层切，att3/att4/moe_ana/moe_ana2.py 看段内指令）；kernel 变体 `k/p2_fused11s.py`（每专家开销 / hw-DGE 开关）、`k/pe_probe.py`（计时不可信，已弃用）。
