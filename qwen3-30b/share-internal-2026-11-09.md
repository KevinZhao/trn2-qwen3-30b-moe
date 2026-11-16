# Qwen3-30B-A3B 在单颗 Trn2 上的吞吐优化复盘（2026-11-08 ~ 11-09）

> 面向懂推理、没跟过这件事的同事。约 10 分钟读完。trn2 数字取自 `REPORT.md` 和 `results/`，H100 取自 `<GPU_H100_BF16>/`，g7e 取自 `<GPU_G7E>/`，都已和原始 summary 逐一核对；标 **〔待确认〕** 的是推断或口径有差异的地方。
> 本文缩写：TTFT = 首 token 时间，TPOT = 每输出 token 时间，cN = 并发 N，NJ = 优化前基线，RS0–RS8 = 本轮各配置。W8A16 = 只把 MoE 专家权重存成 FP8，激活和计算仍是 BF16。

---

## 1. 一页结论

1. **推荐配置是 RS7**，由四项改动组成：FP8 专家权重（W8A16，每个专家一个标量 scale）、T≤32 的 fused decode MoE kernel、bs32 decode bucket、多序列打包 prefill（pack_max 16，pack_budget 4096）。运行在 1 个 device 上，TP4 / LNC2，max-model-len 8192。
2. **相对 NJ 基线**（同机同口径，ShareGPT）：c16 的 TPOT 从 22.53 降到 **14.43 ms（−36%）**，吞吐从 688 升到 **1075 tok/s（+56%）**，TTFT P99 从 1288 降到 **276 ms（−79%）**。c1 的 TPOT 从 6.45 降到 5.85 ms（−9%）。c32 吞吐 **1753 tok/s**（NJ c16 的 2.55×），TTFT P99 581 ms。
3. **对比 H100 TP1，同精度比**（vLLM 0.33，同一批 prompt，详见 5.2）：W8A16 对 W8A16（RS7 对 H100 W8A16 补测）时，c16 TPOT 是 H100 的 **1.54×**，吞吐 **0.65×**，TTFT P99 1.7×；c32 TPOT 1.51×，吞吐 **0.66×**；c1 TPOT 1.19×。纯 BF16（RS1 对 H100 BF16）时，c16 TPOT **1.58×**，吞吐 0.62×。两种精度下 decode 差距几乎一样，原因是 H100 用上 FP8 后自己也快了 27%。RS7 相对 RS1 的 c16 TPOT 改善（20.20 → 14.43 ms）里约 3/4 来自 FP8 权重，约 1/4 来自打包；TTFT P99 的改善几乎都来自打包。
   - 跨精度参考（RS7 W8A16 对 H100 BF16，**只作参考，不代表 decode 追近了 H100**）：c16 TPOT 1.13×，吞吐 0.88×，TTFT P99 1.9×（NJ 时分别是 1.76×、0.56×、8.8×）；c1 TPOT 1.23×；trn2 c32 吞吐是 H100 BF16 c16 的 1.43×。
4. **对比 g7e**（g7e.4xlarge，1 张 RTX PRO 6000 Blackwell，vLLM 0.33，同一批 prompt）：同为 W8A16 时，RS7 和 g7e 基本持平。c16 TPOT 14.43 对 13.81 ms（1.04×），吞吐 0.95×；c32 吞吐 1753 对 1790 tok/s（0.98×）。TTFT P99 还落后：c16 276 对 160 ms（1.7×），c32 581 对 290 ms（2.0×）。纯 BF16 时，trn2 RS1 比 g7e 略快：c16 TPOT 20.20 对 22.15 ms，吞吐 1.07×。
5. **精度和稳定性**：GSM8K 500 题得分 0.946（NJ 0.934；H100 BF16 0.930，H100 / g7e W8A16 0.932）。三次 soak 共 20,052 个请求，0 失败：RS5 71 min，RS4 51 min，RS7 51 min。soak 里有 2 条重复字符告警，重跑后确认是正常内容（表格补齐，以及照抄 prompt 里的注释分隔线）。
6. **停在 RS7 的原因**：剩下的两个大方向是 W8A8 decode MoE（收益上限约 15%，有数值风险）和转置 K cache（约 5–7%，要改多天），用户在 11/9 决定不做。打包参数、kernel 微优化都实测过，没有收益。

---

## 2. 背景与测试口径

| 项 | 内容 |
|---|---|
| 模型 | Qwen3-30B-A3B：48 层，128 专家选 top-8，BF16 checkpoint。每个专家每层 9.44 MB（BF16） |
| 硬件 | 1 个 Trn2 device，TP4 / LNC2（4 个逻辑核，每个逻辑核 2 个物理核），HBM 按 2.9 TB/s 计 |
| 软件 | vLLM 0.29.0（server.log 里写的是 V1 engine v0.29.0）+ vllm-neuron 0.29 native torch 路径（Neuron SDK 2.33）。镜像应为 `pytorch-inference-vllm-neuronx:0.29.0.1.2.0-…-sdk2.33.0` **〔待确认：这是根据同期 k8s 清单和 RESULTS.md 推断的，rs 系列 pod 的结果里没记镜像 tag〕**。qwen3_moe 的模型和 kernel 代码由自研包 `pkg_*` 拷进 site-packages，调度器和 runner 用补丁脚本修改 |
| 服务参数 | max-num-seqs 16（c32 配置为 32），max-model-len 8192，max-num-batched-tokens 8192，`on_device_sampling all_greedy`（贪心） |
| 基线 NJ | 上一轮的产出（`pkg_nj`，fused decode kernel p2_fused11n，BF16）。本轮在同一台机器上复测，记为 RS0：c16 TPOT 22.53 / 688 tok/s / TTFT P99 1288 ms，GSM8K 0.934 |
| bench | `rs_bench.sh`，即 `vllm bench serve` 加 ShareGPT 输出 512（c1 跑 12 条，c4 32 条，c16 64 条，c32 128 条；seed 42，c4/c16 另用 seed 123 复跑）、random 输入 256–7000、1K→512、纯 decode 扫描（ignore-eos）。每跑一次都 reset prefix cache |
| 精度 | GSM8K 500 题（cbv_cli，zero-shot chat，关闭 thinking，温度 0） |
| H100 对照 | 取自另一轮实测（`<GPU_H100_BF16>/REPORT.md`）：p5.48xlarge，vLLM 0.33.0，TP1 和 TP2 各一组。启动参数和 trn2 对齐（MML 8192、MNS 16、MNBT 8192），用 `--override-generation-config '{"temperature":0.0}'` 强制贪心。bench 脚本照搬 trn2 版本，两边 ShareGPT 每档的 `total_input_tokens` 逐一相同，说明测的是同一批 prompt。同一天还做过一轮 FP8 补测（`h100-syd-test/fp8-2026-11-08/`，单卡 TP1，镜像相同，另加 `--block-size 32 --num-gpu-blocks-override 4352/8704`，KV 容量和 trn2 一样），本文 5.2 用其中的 W8A16 组 |
| g7e 对照 | 取自 `<GPU_G7E>/README.md`：g7e.4xlarge（1 张 RTX PRO 6000 Blackwell Server Edition，96 GB GDDR7，规格带宽约 1.8 TB/s），us-east-1 Spot，vLLM 0.33.0（和 H100 同一个镜像 digest），TP1，测了 BF16、W8A16、W8A8 三组，本文只用前两组。server 参数和 H100 FP8 补测逐项相同：MML 8192，MNBT 8192，block 32，c1–c16 用 MNS 16 / NGB 4352，c32 另起一个 MNS 32 / NGB 8704 的 server，贪心。bench 脚本 `fp8_bench.sh` 和 H100 FP8 补测的 sha256 相同，其中 ShareGPT 的 `sg()` 和 trn2 `rs_bench.sh` 逐字相同（每档条数、seed 42/123、输出 512、percentile 都一样）。ShareGPT 各档 `total_input_tokens` 和 trn2 逐一相同（c1 2510、c4 8869、c16 17268、c32 31028），全部 0 失败。**结论：口径一致，可以直接比** |
| 精度：trn2 | **RS0、RS1**：权重、计算、KV cache 全部 BF16。**RS2 起（RS2–RS8，包括推荐的 RS7）**：MoE 专家权重 FP8 e4m3，加载时在线量化，每个专家一个标量 scale（`RS_FP8=1`）；专家以外的权重（attention 等）仍是 BF16；计算 BF16，即 W8A16；KV cache BF16。server.log 里都是 `dtype=torch.bfloat16`、`quantization=None`、`kv_cache_dtype=auto`（RS0、RS1、RS7_final 的日志已核对）：FP8 是在自研模型代码里做的，vLLM 这一层看不到 |
| 精度：H100 | **主对照（TP1 / TP2）是纯 BF16**：启动命令 `--dtype bfloat16`，没有 `--quantization`，也没有 `--kv-cache-dtype`。server-tp1.log / server-tp2.log 里是 `dtype=torch.bfloat16, quantization=None, quantization_config=None, kv_cache_dtype=auto`，auto 即跟随模型 dtype，也就是 BF16 KV。权重用 HF 原始 BF16 checkpoint。**FP8 补测的 W8A16 组**：`--quantization online --quantization-config '{"moe":{"weight":"fp8_per_tensor_static"}}' --moe-backend marlin`，只量化专家（e4m3，每个专家的 w13、w2 各一个标量 scale），attention、dense、lm_head 仍是 BF16，Marlin 反量化后用 BF16 计算，KV 是 auto（BF16）。量化方案和 trn2 RS2 相同，kernel 不同 |
| 精度：g7e | **BF16 组**：和 H100 主对照相同（`--dtype bfloat16`，`quantization=None`，`kv_cache_dtype=auto`）。**W8A16 组**：命令和 H100 W8A16 相同，日志里是 `Using MARLIN Fp8 MoE backend`，`kv_cache_dtype=auto`。两组都已在 server-*.log 里核对过 |

---

## 3. 基线诊断：时间花在哪

**3.1 TPOT 的差距主要在 decode 单步本身，不在 prefill 干扰**

| 纯 decode 每步耗时（RS0，ITL 中位数） | ctx 约 128–640 | ctx 约 1024–1536 |
|---|---|---|
| bs1 | 6.44 ms | 6.47 |
| bs4 | 10.76 | 10.87 |
| bs16 | 21.09 | 19.34 |

ShareGPT c16 的 ITL 中位数是 21.0 ms，和纯 decode bs16 基本相同。也就是说 c16 TPOT 22.6 ms 里只有约 1.6 ms（7%）是 prefill 插进来造成的。H100 TP1（BF16）c16 的 TPOT 是 12.8 ms。

**3.2 带宽利用率只有 65%，其中 MoE 占约 85%**

- bs16 每步要读：专家并集约 82 个/层 × 9.44 MB × 48 层 = 37.1 GB，attention 1.8 GB，lm_head 0.6 GB，合计约 39.6 GB。
- trn2 一步 21.1 ms，折合 1.88 TB/s，**约为峰值的 65%**；H100（BF16）一步 12.8 ms，折合 3.1 TB/s，约为峰值的 92%。BF16 下 trn2 的理论下限约 13.7 ms。
- MoE 约占 bs16 一步的 85%（kernel 微基准 0.378 ms/层 × 48 = 18.2 ms）。

**3.3 MoE decode kernel 的问题是 padding 位置在白算，而不是搬运粒度**

- 基线 kernel 11n 静态展开 min(64, 4T) 个位置。padding 位置只跳过 DMA，matmul/ACT/DVE 仍然照算，而且都排在最后，形成一条串行尾巴。在 T=16 时，每层约 56 µs 是白算，48 层合计约 2.7 ms/步，约占 c16 TPOT 的 12%。
- 真实专家部分每个位置约 7.5 µs，搬 2.36 MB，单物理核约 315 GB/s，已接近 HBM 峰值（约 87%），每个分区是 12 KiB 连续块。所以"大块权重搬运"这条路已经没有空间。
- 设备 profile 显示 FP8 版 kernel 的字节数和执行时间都减半，HBM 读速率不变（约 775–790 GB/s/逻辑核）。**decode MoE 受 HBM 带宽限制**，减少字节数就是直接收益。

**3.4 TTFT P99 的根因是每个 step 只能跑一条 prefill**

- NJ c16 开头 16 个请求同时到达，它们的 TTFT 呈阶梯状：65 → 128 → 199 → … → 1311 ms，每级 75–90 ms。之后的 48 个请求 TTFT 中位数只有 227 ms。
- vllm-neuron 调度器写死了 `max_prefills_per_batch = 1`，prefill 只能逐条串行。短 prompt 的 prefill 也受权重带宽限制（每次都要读全部 128 个专家，约 58 GB，至少 20 ms），16 条就要读 16 遍。H100 上的 vLLM 会把这 16 条打包进一个 8192-token 的 step。
- 调 dsbp/dsbpc 旋钮（控制两次 prefill 之间插几个 decode step）只能在 P99 和均值/ITL 尾部之间挪：默认 1/1 时 TTFT 均值/P99 是 387/1379 ms；0/0 时 574/1098 ms，但 ITL 最大值涨到 1049 ms。治本的办法只有打包。

---

## 4. 做了什么

每一步都在上一步的基础上叠加。下表的收益是 ShareGPT c16 s42 实测。

| 步骤 | 配置 | c16 TPOT | c16 tok/s | c16 TTFT P99 |
|---|---|---|---|---|
| 基线 | RS0（NJ） | 22.53 | 688 | 1288 |
| 4.1 去掉 padding 白算（BF16） | RS1 | 20.20（−10%） | 763（+11%） | 1304 |
| 4.2 FP8 专家权重 | RS2 | 15.86（−21%） | 969（+27%） | 1188 |
| 4.3 T≤32 + bs32 bucket | RS3 | 15.81 | 974 | 1141 |
| 4.4 多序列打包 prefill | RS7（RS3 + 打包） | **14.43**（−9%） | **1075**（+10%） | **276**（−76%） |

括号里是相对上一行的变化。

### 4.1 decode MoE kernel：padding 交错，支持 T≤32（11n → 11p，仍是 BF16）
- **做法**：padding 位置不再全部排在最后，而是和真实位置交错，消掉串行的 PE 尾巴。同时把 kernel 支持的 T 扩到 32。
- **kernel 微基准**（每层 µs，T=1/4/8/16/32）：11n 是 58/143/236/378/（T=32 要调两次 T=16）；11p 是 58/142/226/338/442。
- **e2e**：纯 decode bs16 从 21.09 降到 18.25 ms（−13.5%）；c16 TPOT −10%，吞吐 +11%；GSM8K 0.934 不变。这一步纯 BF16，**不接受 FP8 时可以单独用这一步，零数值风险**。

### 4.2 FP8 专家权重 + fused decode kernel（11r）+ prefill kernel（nm_k9f）
- **做法**：加载时把专家权重在线量化成 e4m3，每个专家一个标量 scale，计算仍是 BF16（W8A16）。scale 合并进 SiLU 激活的 per-partition scale 和路由 affinity，kernel 里没有额外开销。
- **为什么选 per-expert 标量**：用真实权重测，per-expert-tensor 的量化误差 2.65%，per-channel 2.64%，几乎一样（e4m3 是浮点格式，误差和 scale 粒度基本无关）。per-channel 版本（11q）每个专家要多一次零碎的 scale DMA，每层多约 60 µs，已放弃。
- **kernel 微基准**：11r（FP8）每层 48/94/140/202/252 µs（T=1/4/8/16/32），T=16 时比 11p 少 40%。prefill nm_k9f 在 T=256/2048/8192 时每层分别从 836 降到 661、1470 降到 1234、3609 降到 3517 µs。
- **e2e**：纯 decode bs16 从 18.25 降到 14.00 ms（−23%）；c1 TPOT 6.44 → 5.86；c16 TPOT 20.20 → 15.86，吞吐 763 → 969。**这是本轮最大的单项收益。** GSM8K 0.946。FP8 每个 rank 还省出约 7 GB HBM。

### 4.3 bs32 bucket + T≤32（RS3）
- **做法**：decode bucket 改成 [1,4,8,16,32]，max-num-seqs 32，`Q3_FUSED_MOE_MAX_T=32`。用 FP8 省下的 HBM 把 KV block 数从 4352 加到 8704，在 MML 8192 下 KV 并发上限是 34。
- **为什么**：decode 受 HBM 带宽限制，bs 越大，每个 token 分摊的权重字节越少。纯 decode bs16 → bs32 每步只慢了 20%（14.00 → 16.75 ms）。
- **e2e**：c32 ShareGPT 1481 tok/s（RS0 c16 的 2.15×），但 TTFT P99 达到 2405 ms，原因是 32 条突发请求的 prefill 仍在串行排队。c≤16 的数字和 RS2 一样。

### 4.4 多序列打包 prefill（P5，RS4 / RS7）
- **为什么能做**：runner 的输入构建继承自 GPU vLLM，本来就是多请求的写法（`cu_num_tokens`、逐 token 的 slot_mapping、逐请求的 logits_indices）。attention 用 nkilib `attention_cte` 自带的 sequence packing（用 `bound_min/bound_max` 加上 causal 实现块对角 mask），不用自己写 kernel。sampling 本来就按行处理。真正写死"只有一条"的地方只有 4 处：调度器准入、调度器按请求补 bucket、prefill 图里请求数那一维的形状、attention。
- **做法**：
  - 调度器只打包满足以下条件的请求：新请求、整条 prompt 能放进 budget、前缀缓存命中为 0、没有结构化输出或多模态。其余请求仍走原来的单条路径。
  - 按 FCFS 累计，直到 `pack_max` 条或总长达到 `pack_budget` token。所有 padding 挂在最后一条请求上，总长补到打包 bucket（512/1024/2048/4096）。
  - 请求数那一维固定为 16，每个打包 T bucket 只多 1 个 NEFF。打包 step 走 full attention 分支，加上 bound。
  - MoE 的 padding mask 改成显式 mask（见第 9 节第 2 条）。
  - 用 knob 文件运行时调 `pack_max` 和 `pack_budget`，`pack_max=1` 就是关闭，不用重启。
  - 实现方式是对 site-packages 做带计数断言的字符串替换补丁（`p5_patch_pack.py`），模型侧改动直接写进 `pkg_rs3`。
- **效果**：16 条 prompt 同时发，全部完成的墙钟时间从 2.06 s 降到 1.24 s（RS4）。在 c32 配置上（RS7），逐条跑要 6.90 s，打包后 1.17–1.20 s，一个 step 最多装 14 条。

| ShareGPT | 不打包 TTFT 均值/中位/P99 | 打包 TTFT 均值/中位/P99 | TPOT | tok/s |
|---|---|---|---|---|
| c16 s42（RS3 → RS7） | 299 / 213 / 1141 | 243 / 255 / **276** | 15.81 → 14.43 | 974 → 1075 |
| c16 s123（RS3 → RS7） | 293 / 196 / 1213 | 265 / 266 / 381 | 15.94 → 14.44 | 946 → 1049 |
| c32 s42（RS3 → RS7） | 461 / 215 / 2405 | 354 / 300 / **581** | 20.29 → 17.24 | 1481 → 1753 |
| c4 s42（RS3 → RS7） | 169 / 180 / 287 | 107 / 97 / 208 | 8.81 → 8.85 | 438 → 432 |

- P99 大幅下降。TPOT 也有改善，因为 prefill step 变少，插进 decode 的次数也少了。中位数略有上升，因为同一个打包 step 里最短的请求要等最长的请求一起算完。
- 长 prompt（≥3000 token）超过 4096 的 budget，不会被打包，结果和不打包相同（±1–2%）。单请求 TTFT 扫描也不变，说明打包补丁对单条路径没有额外开销。
- 正确性：首 token（直接来自打包 prefill 的 logits）16/16 一致。贪心 64 token 的完全一致率是 11–13/16，和"不打包时 c1 对比 c16"的 12–13/16 处在同一水平（原因见第 9 节第 3 条）。GSM8K 0.946，与不打包相同。

---

## 5. 数据总表

### 5.1 ShareGPT（seed 42，输出 512），NJ → RS7，对比 H100 和 g7e

表头括号里是各配置的精度（权重精度；计算和 KV cache 都是 BF16）。TTFT 一栏依次是均值 / 中位 / P99。

| 并发 | 指标 | NJ（RS0，BF16） | **RS7（W8A16）** | H100 TP1（BF16） | H100 TP2（BF16，2 卡，参考） | g7e（BF16） | g7e（W8A16） | RS7 / H100 TP1（跨精度） | RS7 / g7e W8A16 |
|---|---|---|---|---|---|---|---|---|---|
| c1 | TTFT（ms） | 71.5 / 65.3 / 100.5 | 63.6 / 55.8 / 96.5 | 50.1 / 56.8 / 66.6 | 62.0 / 70.7 / 86.8 | 39.2 / 39.5 / 54.8 | 32.7 / 36.3 / 42.8 | P99 1.45× | P99 2.3× |
| c1 | TPOT（ms） | 6.45 | 5.85 | 4.75 | 3.90 | 6.72 | 5.66 | 1.23× | 1.03× |
| c1 | 输出 tok/s | 152.0 | 167.7 | 206.6 | 249.2 | 147.5 | 175.0 | 0.81× | 0.96× |
| c4 | TTFT | 183.6 / 201.3 / 301.4 | 107.3 / 96.7 / 207.7 | 124.0 / 108.0 / 581.2（复跑 106.7 / 110.0 / 122.9） | 127.4 / 133.1 / 232.5 | 86.1 / 89.2 / 108.6 | 60.6 / 64.8 / 73.5 | 见注 | P99 2.8× |
| c4 | TPOT | 11.13 | 8.85 | 8.34 | 5.52 | 12.68 | 8.92 | 1.06× | 0.99× |
| c4 | tok/s | 348.7 | 432.4 | 462.8 | 694.3 | 311.9 | 440.5 | 0.93× | 0.98× |
| c16 | TTFT | 353.0 / 246.2 / 1288.4 | 242.8 / 255.3 / **276.1** | 123.3 / 125.0 / 145.9 | 148.4 / 138.4 / 233.2 | 148.1 / 153.2 / 161.5 | 109.4 / 101.0 / 159.7 | P99 1.9× | P99 1.7× |
| c16 | TPOT（P99） | 22.53（23.10） | **14.43**（14.94） | 12.82（13.20） | 8.65（8.98） | 22.15（22.52） | 13.81（14.14） | 1.13× | 1.04× |
| c16 | tok/s | 688.3 | **1074.9** | 1224.2 | 1784.0 | 714.3 | 1137.2 | 0.88× | 0.95× |
| c32 | TTFT | — | 354.1 / 299.5 / 580.5 | 未测 | 未测 | 192.5 / 189.9 / 275.4 | 211.1 / 200.2 / 290.0 | — | P99 2.0× |
| c32 | TPOT（P99）/ tok/s | — | 17.24（17.89）/ **1753.0** | 未测 | 未测 | 28.35（29.04）/ 1088.9 | 17.16（17.66）/ 1789.8 | 吞吐为 H100 TP1 c16 的 1.43× | TPOT 1.00×，吞吐 0.98× |
| — | GSM8K 500 | 0.934 | 0.946 | 0.930 | 0.930 | 0.930 | 0.932 | | |

注：
- **"RS7 / H100 TP1" 这一列是 W8A16 对 BF16 的跨精度比较。** 同精度的比较见 5.2。
- H100 TP1 c4 第一轮 TTFT P99 581 ms 是被单个 786 ms 的请求拉高的，复跑是 122.9 ms。按复跑算，RS7 c4 的 P99 约为 H100 的 1.7×。
- RS7 c1 的 TTFT P99 96.5 ms 比 RS3 的 88.7 ms 高。c1 只有 12 条请求，P99 基本就是最大值，属于噪声 **〔判断，没有专门复测〕**。
- 测法差异：RS7 的 c1–c32 都在同一个 MNS 32 的 server 上测；g7e 的 c1–c16 用 MNS 16 的 server，c32 用另一个 MNS 32 的 server。g7e W8A16 在 MNS 32 server 上的 c16 是 13.71 ms / 1144.9 tok/s，和 MNS 16 server 相差不到 1%。g7e 加了 `--block-size 32 --num-gpu-blocks-override`，KV 容量和 trn2 一样；H100 主对照没加（KV 145,536 token），这些档位下两者的 KV 都够用。
- 调度方式不同：GPU 两边都是 vLLM 默认调度（chunked prefill，prefill 和 decode 混在同一个 step），trn2 是自己实现的打包。TTFT 的差距主要来自调度，见 3.4 和第 8 节。
- trn2 是 1 个 device，H100 / g7e 都是 1 张卡（H100 TP2 是 2 张）。本轮没有做成本对比。g7e README 里有一份按 Spot 价的每美元吞吐估算，其中 trn2 的价格是按整机均摊推算的，本文不引用。

### 5.2 同精度对比：RS7 的优势里有多少来自 FP8

5.1 里 RS7 对 H100 的 1.13×，是 W8A16 的 trn2 对 BF16 的 H100。下面按精度分开比较。H100 W8A16 来自同一天的 FP8 补测：单卡 TP1，同一个镜像，同一批 prompt，c32 用 MNS 32 的 server。

**(a) 纯 BF16：trn2 RS1 对 H100 BF16、g7e BF16**（RS1 = BF16 kernel 去掉 padding 白算，不打包，MNS 16，没有 c32）

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
- **同精度下，trn2 和 H100 的 decode 差距在两种精度下几乎一样**：c16 TPOT，BF16 是 1.58×（RS1），W8A16 是 1.54×（RS7）。原因是 H100 用上 FP8 后自己也快了：c16 TPOT 12.82 → 9.38 ms（−27%）。和 FP8 补测里参数对齐的 BF16 组（12.60 ms）比是 −26%。所以 5.1 里的 1.13× 基本只反映了 trn2 这边上了 FP8、H100 那边没上，不能说成 trn2 的 decode 追近了 H100。
- **RS1 → RS7 的收益拆分**（c16 s42，按 RS1 → RS2 → RS3 → RS7 的叠加顺序拆，**不是正交分解**）：TPOT 共降 5.77 ms（20.20 → 14.43），其中 FP8 权重（RS1 → RS2，20.20 → 15.86）占 4.34 ms，约 75%；T≤32 + bs32（RS2 → RS3）占 0.05 ms；打包（RS3 → RS7，15.81 → 14.43）占 1.38 ms，约 24%。吞吐共增加 312 tok/s（763 → 1075），FP8 占 206（约 66%），打包占 101（约 32%）。
- **TTFT P99 的改善几乎都来自打包，不来自 FP8**：c16 P99 RS1 1304 → RS2 1188 → RS3 1141 → RS7 276 ms。对 H100 BF16 的 P99 比值从 8.9× 降到 1.9×，靠的是打包。
- **和 g7e 比**：BF16 时 trn2 略占优（c16 TPOT 0.91×，c4 0.87×），W8A16 时基本持平（c16 1.04×，c32 1.00×）。原因是 g7e 从 FP8 拿到的收益更大：c16 TPOT 22.15 → 13.81 ms（−38%），而 trn2 RS1 → RS2 是 −21%。g7e 的显存带宽更紧（规格约 1.8 TB/s），权重字节减半的收益更直接（这是 g7e README 的解释，**〔未单独剖析〕**）。prefill 比重大的档位 g7e 明显更好：1K→512 c16 吞吐是 RS3 的 1.25×，4K→512 是 1.31×；TTFT P99 也更低（1K→512 c16 482 对 1439 ms，4K→512 c16 2123 对 3332 ms）。
- **口径提醒**：GPU 的 W8A16 走 Marlin（先反量化再做 BF16 MMA），c1 时比 BF16 还慢（H100 4.92 对 4.75 ms），长 prefill 也更慢。所以 W8A16 对 W8A16 比较的是各自现成的最佳实现，不是理论最优。H100 有官方 FP8 checkpoint（W8A8 block），c16 TPOT 8.98 ms、1729 tok/s，比 W8A16 再快约 4%；trn2 没有 W8A8 对应项，没有放进表里。

### 5.3 其它口径（RS4 和 RS7 没跑 1K→512 和纯 decode 扫描，下面用 RS3 的数据。RS3 和 RS7 的 decode 路径相同，只差打包）

| 口径 | NJ（RS0，BF16） | RS3（W8A16） | H100 TP1（BF16） | H100 TP2（BF16） | g7e（BF16） | g7e（W8A16） |
|---|---|---|---|---|---|---|
| 1K→512 c1 TTFT / TPOT / tok/s | 95 / 6.50 / 150 | 88 / 5.90 / 165 | 66 / 4.80 / 203 | 76 / 3.94 / 245 | 62 / 6.74 / 146 | 50 / 5.68 / 173 |
| 1K→512 c16 TTFT / TPOT / tok/s | 485 / 21.86 / 700 | 429 / 16.08 / 943 | 187 / 11.46 / 1355 | 164 / 8.09 / 1905 | 333 / 19.84 / 782 | 341 / 12.89 / 1182 |
| 1K→512 c32 TPOT / tok/s | — | 21.01 / 1420 | 未测 | 未测 | 27.30 / 1131.5 | 17.50 / 1726 |
| 4096→512 c16 TPOT / tok/s | 26.73 / 546 | 22.33 / 647 | 12.79 / 1159 | 8.48 / 1730 | 22.12 / 666 | 16.92 / 844 |
| 纯 decode bs1 / bs16 / bs32 每步（ms，in=128） | 6.44 / 21.09 / — | 5.85 / 14.00 / 16.75 | — | — | — | — |

长 prompt 场景（4096→512）离 H100 还差得多：TPOT 是 H100 TP1 BF16 的 1.75×，是 g7e W8A16 的 1.32×。这部分不受打包影响，attention 占比也更高。g7e 的 1K / 4K→512 用的也是 bench2 口径（`--ignore-eos`，不设 seed），`total_input_tokens` 和 trn2 相同（49152 / 131072）。

---

## 6. 稳定性与精度

| soak | 配置 | 时长 | 请求数 | 结果 | 负载构成 |
|---|---|---|---|---|---|
| RS5 | RS3（FP8 + bs32，不打包） | 71.2 min | 8376 | 0 失败，0 坏样本 | c1/c4/c16/c32 混合（c32 占 3869 条）；随机长 prompt 4222 条，其中 ≥4096 token 的 2833 条；prefill 共 1896 万 token |
| RS4 | RS2 + 打包（MNS16） | 50.7 min | 5587 | 0 失败，0 坏样本 | c1/c4/c16 混合；随机长 prompt 3121 条；服务进程从启动到 soak 结束共 261 个多请求打包 step（含 bench 期间） |
| RS7 | 推荐配置（c32 + 打包） | 50.8 min | 6089 | 0 失败，2 条告警（重跑后判定为正常） | c32 占 2827 条；随机长 prompt 3035 条；prefill 共 1362 万 token；服务进程从启动到 soak 结束共 373 个多请求打包 step（含 bench 期间） |

- 三次 soak 的开始、中间、结束三次 NaN 检查都 PASS，金丝雀请求全部通过，服务日志里 nrta / SEQUENCER / BAD_OPCODE / NRT_TIMEOUT / Traceback 都是 0。RS7 soak 中最长 TTFT 7.1 s、最长 e2e 37.0 s，都来自 c32 加长 prompt 时的串行 prefill 排队。
- **2 条重复字符告警**（启发式规则 `(.)\1{40,}`）的复查：
  1. "brute force 和 dictionary attack 的对比表"：c1 贪心重跑两次，逐字一致，都复现了告警。触发处是 markdown 表格：表头用 65 个空格补齐列宽，分隔行是 88 个 `-`。这是正常内容，属于误报。
  2. 一段 CSS 代码：prompt 本身就有 58、52、49 个连续 `-` 组成的注释分隔线。重跑 4 次（c1 一次，c32 批次三次）都没有复现，判断是模型在思考过程中照抄了用户的分隔线。
  - 结论：两条都不是数值退化。
- **GSM8K 500**：RS2、RS4、RS6（RS3 同配置复测）、RS7、RS7pm32、RS8 全部是 0.946，NJ 0.934，H100 BF16 0.930（H100 / g7e 的 W8A16 都是 0.932，g7e BF16 0.930）。H100 报告给出的标准误约 1.1 个点，所以这些差异都在噪声范围内，结论是"FP8 没有让精度变差"，不是"FP8 让精度变好"。

---

## 7. 试过但没收益、或判断不做的

| 方向 | 结论与理由 |
|---|---|
| c32 下 pack_max 16 → 32（RS7pm32） | 一个 step 最多装到 19 条，但 4096 的 budget 先成为上限。c32 TTFT P99 586 ms（RS7 581），吞吐 1740（1753），在噪声范围内 |
| pack_budget 8192 + 8192 打包 bucket（RS8） | c32 P99 544 ms，略好；但 TTFT 均值从 354 升到 429，中位数从 300 升到 458，吞吐没变。原因是打包超过 4096 token 后，step 里每条请求都要等更长的 prefill。不采用 |
| decode MoE kernel 每专家开销优化（11s：拆 PSUM 拷贝、合并 DVE 指令） | T=4/16/32 都在 ±1% 以内。每专家的 ACT/DVE 指令和 GpSimd 描述符生成都不在关键路径上 |
| 专家权重 DMA 改走 hw-DGE | 只快 ≤1.4%，而且生产环境里 hw-DGE 出过 BAD_OPCODE，不值得再跑一轮 soak 验证 |
| 对调 PE 的 moving / stationary 角色 | kernel 已经用 4 个 32 列小块并行，对调后只剩 1 路，放弃 |
| dsbp/dsbpc 旋钮 | 只在 TTFT P99 和均值 / ITL 尾部之间挪，见 3.4 |
| 单芯片 EP4 代替 TP4 | 同一步要读的专家字节数一样（并集不变）。EP 只会带来 rank 间负载不均（4 个 rank 各自并集的最大值比均值高 15–20%）和 all-to-all 开销。TP 切分下 DMA 已经是 12 KiB 连续块。上一轮也实测过片内 EP 无效 |
| P/D 分离（同机） | prefill 侧仍是单条串行，突发时 P99 不变，只能消掉约 7% 的 TPOT 干扰，还要多占 1 个 device |
| c64 | kernel 上限是 T=32，c64 要拆成两次调用，每 token 的 MoE 成本不再下降，估计吞吐只比 c32 高 5–10%；而且 KV 需要约 17K block，HBM 不够 |
| EAGLE3 | NxDI 路径不输出 aux hidden states；PR56 路径以前实测 k=1 接受率 68%，c1 TPOT 只降 10%，c4 起没有收益 |
| W8A8 decode MoE | 用户 11/9 决定不做。收益上限约 15%（理由见第 8 节），有数值风险 |
| 转置 K cache 布局 | 用户 11/9 决定不做。收益约每步 0.7–1.0 ms（5–7%），要同时改 prefill 写 cache、分段 attention、打包分支和 decode 四条路径，每次重编译约 1 h，是多天的工作 |

---

## 8. 为什么停在 RS7，剩下的差距在哪

**decode（c16 TPOT 14.43 对 H100 BF16 12.82、H100 W8A16 9.38）**：RS7 的 decode NEFF 整步 profile 结果如下（bs16 ctx1024，capture 自己生成的输入）。

| 每层（中位数） | µs | 占比 |
|---|---|---|
| MoE | 157.2 | 59.6% |
| attention（qkv / o_proj / KV 写回） | 74.3 | 28.3% |
| 两次 all-reduce | 10.2 + 9.4 | 7.5% |
| 整步（含 lm_head + 采样约 0.52 ms） | 12.70 ms | |

profile 整步 12.70 ms，比 e2e 的 TPOT 14.43 ms 少约 1.7 ms。差额应该主要来自 prefill 插入和主机侧开销 **〔推测，没有拆分〕**。

- **MoE 同时卡在 PE 和 HBM 两个上限**。每层 4514 条 MATMUL，8 个 PE 小块在整个时间窗里一直忙。FP8 权重作为 moving 操作数时，单条 matmul 耗时和 BF16 一样（moving 384 列都是 326 ns），也就是 **PE 按 bf16 速率在跑，FP8 只省了 HBM 字节，没有换来 PE 加速**。HBM 侧每个 rank 约 620 GB/s，而单 rank 能分到的 HBM 带宽约 725 GB/s。W8A8 能让 PE 跑 FP8 速率，但 HBM 只剩约 15% 余量，所以收益上限约 15%。
- **attention 不受带宽限制**（bs16 每层只读约 17.6 MB，折合约 240 GB/s）。开头 10–17 µs 只有 GpSimd 在按 block table 收集 K 并做转置，其它引擎都空着。去掉这一段要改 K cache 的存储布局，没有现成开关，见第 7 节。

**TTFT（c16 P99 276–381 ms，H100 BF16 146 ms、W8A16 160 ms）**：剩下的差距有三个来源：
1. pack_budget 4096 限制了打包，长 prompt 仍然串行。
2. 打包 step 里最短的请求要等最长的。
3. 单条 prefill 本身就比 H100 慢：ShareGPT c1 TTFT 62–64 ms，H100 50 ms。单请求 TTFT 扫描：128 token 54 ms，2048 token 119 ms，4096 token 197 ms，8000 token 390 ms。

**停下来的理由**：kernel 层面的低成本改动（每专家指令、hw-DGE、打包参数）都实测过，没有收益。剩下的两项（W8A8、转置 K cache）合计上限约 20%，但都是多天工作量、每次要重编译约 1 h，W8A8 还要重新做精度验证。用户在 11/9 决定先停在 RS7。

---

## 9. 踩过的坑和经验

1. **打包补丁会静默失效，要用 steplog 确认是否生效。** 第一次上机时，48 层图编译通过，NaN 检查也 PASS，但 steplog 里每个 prefill step 都只有 1 条请求。原因是 vLLM 0.29 的 `find_longest_cache_hit` / `get_computed_blocks` 返回 3 元组，补丁按 2 元组解包，前缀命中探测抛异常后，代码保守地关闭了打包，只留下一行日志。修复只改调度器的 Python 代码，不需要重编译（重启约 40 min，新增 NEFF 为 0）。**经验：新功能要有一个"确实生效了"的正向指标（这里是 steplog 里多请求 step 的数量），不能只看没有报错。**
2. **MoE prefill 的 padding mask 会静默出错。** 原代码用 `arange <= argmax(positions)` 推断最后一个真实 token。打包后 positions 是锯齿形的，除了第一条最长的序列，其它 token 都会被判成 padding，MoE 输出被置零，不报错，只会给出错误答案。这一点是设计阶段读代码时发现的，已改成显式 mask。边界用例里一定要有"多条序列等长"的情况。
3. **batch 组成本身就会改变贪心结果。** 同一组 16 条 prompt，逐条发和 16 条同时发（不打包）相比，完全一致的只有 12/16；两次 c16 之间也只有 13/16 一致。分歧出现在第 5–49 个 token，原因是 bf16 下不同 T 的 kernel 数值不同。**所以验证打包这类改动时，要和"同一路径的 batch 噪声"这条基线比，不能要求 100% 一致**；更可靠的检查是首 token 的一致性加上 GSM8K。
4. **NEFF 编译缓存的代次问题。** 缓存目录是三代配置复制下来的，每个 (B, ctx) 有 3 个 NEFF。profile 工具默认选到的是最老一代（NJ 的 BF16 NEFF），要按修改时间选最新的才是 RS7。另外，新配置以旧缓存为底增量编译能省不少时间（RS7 以 RS3 的缓存为底，新增 31 个 NEFF，约 1 h）。每次启动都记录 `neffs_before/after new=N`，可以发现意外的重编译（比如 dynamo guard 被 Python int 触发）。
5. **怎么抓 decode 的整步设备 profile。** vLLM 的 `/start_profile` 抓不到 decode NEFF：vllm_neuron 源码里自己写了这是已知问题，而且在 KV override 下 HBM 不够分配默认的 profiler 缓冲。可行的做法是对缓存里的 decode NEFF 离线跑 `neuron-explorer capture`（4 rank，只 profile rank 0，取第 6 次执行），再按 all-reduce 边界切出每一层（`scripts/prof/ie_prof.sh`、`ie_ana.py`）。注意这样得到的输入是 capture 自己生成的，不是真实 hidden state，路由分布和真实请求不同。
6. **微基准也会骗人。** 专门写的 PE 探针，从 64 条到 8192 条 matmul，调用耗时都是约 175 µs，结果不可信，后来改用整步 profile 里单条指令的耗时。另外单核微基准（0.20 ms/层）比 e2e 推算（0.23 ms/层）乐观，因为 e2e 里 4 个 rank 在争 HBM。
7. **soak 要保存完整输出。** soak 只存了前 80 个字符，复查告警时只能靠 `len(prompt)//4` 加输出开头，去 ShareGPT 里反查原 prompt 再重跑。
8. **对比 H100 时要对齐采样方式，并且充分预热。** vLLM 0.33 默认读取 generation_config（温度 0.6），必须在 server 端强制温度 0 才是贪心。trn2 原来的预热覆盖不到 Triton fused_moe 的 JIT 形状，H100 c1 会出现 600–760 ms 的 TTFT 尖峰，要加长预热。
9. **共享机器上跑 bench 要加锁。** 同一台机器上多个 pod 同时跑 bench 会互相干扰，本轮 bench 用了文件锁。RS2 那一组是等锁超时后跑的，事后用 RS3 的同口径数字交叉确认过。

---

## 10. 附录

### 10.1 推荐配置（RS7）启动参数
```
PKG=pkg_rs3，启动前打 scripts/p5_patch_pack.py（修复版），即 rs_srv.sh 里的 P5P=1
env: NM_MOE=1 Q3_FUSED_MOE=1 Q3_FUSED_MOE_KMOD=p2_fused11r Q3_FUSED_MOE_MAX_T=32 RS_FP8=1 NM_KMOD=nm_k9f P5_PACK=1
     WSA_FN=1 II_DGE=sw P3_KVIX=1 P4_DSBPC=1 P4_DSBP=1        # knob json：pack_max 16，pack_budget 4096
vllm serve Qwen3-30B-A3B --tensor-parallel-size 4 --max-num-seqs 32 --max-model-len 8192 \
  --max-num-batched-tokens 8192 --num-gpu-blocks-override 8704 \
  --additional-config '{"neuron_config":{"quantization":"bf16","on_device_sampling_config":{"all_greedy":true},
    "num_batched_tokens_buckets":[128,256,512,1024,2048,4096,8192],"num_seqs_buckets":[1,4,8,16,32],
    "decode_context_length_buckets":[1024,2048,4096],"kv_segment_size_buckets":[8192]}}'
```
- 一键启动：`scripts/rs_srv.sh`，参数为 `P5P=1 NSB="[1,4,8,16,32]" MNS=32 NGB=8704 NMX="Q3_FUSED_MOE_KMOD=p2_fused11r RS_FP8=1 NM_KMOD=nm_k9f Q3_FUSED_MOE_MAX_T=32 P5_PACK=1"`。打包要求 all_greedy。`quantization` 写的是 bf16，FP8 是由 `RS_FP8=1` 在加载时在线量化的。
- 以 RS3 缓存为底启动，新增编译 31 个 NEFF（缓存里合计 89 个，含继承的旧配置 NEFF），从启动到 ready 约 1 h。启动期间 pod 内存（cgroup）峰值 76 GiB。
- 备选：
  - 只跑 c≤16：用 RS4（MNS16，不加 bs32 bucket），c16 TPOT 14.70、1047 tok/s、TTFT P99 308 ms。
  - 不要打包：用 RS3（pkg_rs2），c16 TPOT 15.81、974 tok/s，但 TTFT P99 1.1 s，c32 时 2.4 s。
  - 不接受 FP8：用 RS1（pkg_rs1，`Q3_FUSED_MOE_KMOD=p2_fused11p`），纯 BF16，c16 TPOT −10%、吞吐 +11%，GSM8K 0.934。

### 10.2 文件和原始数据位置（都在 `qwen3-30b/` 下）
| 内容 | 位置 |
|---|---|
| 完整技术报告 | `REPORT.md`；打包设计评估：`multi_prefill_design.md` |
| 代码包 | `pkg_rs1/`（11p）、`pkg_rs2/`（FP8：model_bf16.py 的 RS_FP8 段、p2_fused11r.py、nm_k9f.py 等）、`pkg_rs3/`（pkg_rs2 加打包） |
| kernel 和微基准 | `k/`（p2_fused11o/p/q/r/s.py、wsB_t1.py 等） |
| 脚本 | `scripts/`（rs_srv.sh、rs_bench.sh、rs_soak.*、p5_patch_pack.py、p5_pack_check.py）；profile：`scripts/prof/` |
| 各配置汇总 | `results/<RSx>/summary_blog.txt`（每行依次是 TTFT 均值 / 中位 / P99、TPOT 均值 / P99、ITL、tok/s）；推荐配置的 soak：`results/RS7/soak_summary.txt` |
| 原始数据 | `results/<目录>/node_raw/`（bench json、server.log、steplog、gsm、soak 的 req.jsonl 等）。推荐配置以 `results/RS7_final/` 为准（含 c32 soak 和告警重跑的全文 `soak/charrun_rerun*.json`）；`RS7pm32`、`RS8` 是打包参数实验；`KNOB` 是 dsbp/dsbpc 扫描 |
| 整步 decode profile | `results/PROF/decode_profile_RS7.txt`；`results/_prof/` |
| 校验清单 | `results/MANIFEST_node_raw.txt`（820 个文件，本地逐个核对大小一致） |
| H100 对照 | BF16：`<GPU_H100_BF16>/REPORT.md` 和同目录下的 `results/TP1`、`results/TP2`（含 `server-tp*.log`、`SUMMARY_all.txt`）。FP8 补测：`<GPU_H100_FP8>/REPORT.md`，W8A16 组见 `results/w8a16`、`results/w8a16x32` 的 `summary.txt` 和 `server-w8a16*.log` |
| g7e 对照 | `<GPU_G7E>/README.md`；全量对比 `results/COMPARE.md`；各组 `results/{bf16,bf16x32,w8a16,w8a16x32}/summary.txt`、`runs/*.json`；server 日志 `results/server-*.log`；bench 脚本 `scripts/fp8_bench.sh` |
