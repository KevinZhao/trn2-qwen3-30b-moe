# blog 草稿说明：数字出处核对（2026-11-15）

> 公开版：账号、资源名、本地路径已替换成 `<占位符>`（含义见 `trn2-handoff.md` 第 7 节）。文中路径都指原始工作区，和本仓库目录的对照见 `trn2-handoff.md` 第 7 节；`node_raw/` 下的原始数据没有发布。

## 草稿在哪

| 文件 | 说明 |
|---|---|
| `blog-docx/blog-trn2-aws-cn.md` | AWS-CN 版正文（447 行，2026-11-14 22:58）。**以它为准** |
| `blog-docx/4. 在 AWS Trainium2 上服务 MoE 模型：Qwen3-30B-A3B 单芯片优化实践与 kernel 原理.docx` | 由上面的 md 生成的 Word（同一时间） |
| `blog-docx/img/*.svg`、`blog-docx/img-png/*.png` | 图 1–6（fig6 是大 MoE 的图） |
| `scripts/blog/make_charts.py`、`scripts/blog/build_docx.py` | 出图和生成 docx 的脚本，用法见 内部笔记 |
| `blog-trn2-draft.md`（工作区根目录） | 更早的草稿，带〔待确认〕标记，可以对照看哪些点当时就没定 |

本次没有改动上面任何文件。

## 怎么读下表

- **状态**：
  - **原始**：工作区里有原始 bench JSON 或日志，数字已经逐个核对过；
  - **原始（S3）**：原始日志在 S3 上，本地没有；
  - **原始（工作区外）**：在原始工作区之外的本地目录，已经核对过；
  - **仅报告**：只出现在报告类文件里（REPORT.md、cte_result.json、prefill-moe-kernel.html、内部笔记），**没有原始日志**；
  - **推导**：由其它数字计算得到，写出了算法；
  - **无依据**：工作区里找不到出处；
  - **范围外**：大模型（Qwen3-235B、DeepSeek-V3）的数字。用户 11/15 说"更大的模型以后再说"，这次不核对。
- 路径简写：`R/` = `research-qwen3-30b/results/`，`RS7F/` = `R/RS7_final/node_raw/`，`H100/` = `<GPU_H100_BF16>/`，`H100F/` = `<GPU_H100_FP8>/results/`，`G7E/` = `<GPU_G7E>/results/`（后三个在工作区外）。
- 每条 bench JSON 里用的字段是 `mean_tpot_ms`、`output_throughput`、`mean_ttft_ms`、`p99_ttft_ms`、`median_itl_ms`。

## 发现的问题（先看这里）

1. **镜像 tag 是推断的**（10.1 节）：Qwen3-30B 那几个 pod 的清单没有保存下来。间接证据见 `handoff/trn2-handoff.md` 6.1 第 1 条。
2. **冷启动 38.9 → 27.8 分钟、306 MB、2 秒**（7.2 节）：原始日志在 S3 `s3://<NEFF_CACHE_BUCKET>/_meta/cache-test/20261108T134029Z/`（`summary.txt`：`startup_sec=1667` / `2334`、`neff=31 MB=306 sec=2`），本地没有。测的是 RS0 / NJ 配置，不是推荐配置。
3. **大 MoE（第六章，以及 1.2、5.5、7.1、7.2、9、10 节里相关的句子）要不要保留**：待用户决定，本次没有核对。
4. **5.3 节（prefill MoE kernel）整节没有原始日志**：数字都来自 `cte_result.json` 和 `prefill-moe-kernel.html`。原始产物在 `<内部临时路径>` 和节点 `/data/cache/nj_results`，都已经不在了。
5. **5.2 节"T=16 每层 378 → 338 → 202 µs"没有原始日志**（只在 REPORT §2.5 里有）。11/9 重测的 `R/_joblogs/node_raw/log_MB1.txt` 里 11r T=16 是 0.2195 ms（L=8，测法不同），和 202 µs 对不上。
6. **7.2 节"编译峰值约 76 GiB"**：RS7 是 76 GiB（`RS7F/info`），但 RS3 是 82 GiB（`R/RS3/node_raw/info`）。建议改成"约 76–82 GiB"，或者注明说的是推荐配置。
7. **5.2 节"ITL 中位数 21.0 ms"**：RS0 原始 JSON 里是 20.90 ms（`R/RS0/node_raw/runs/sg_c16_s42.json` 的 `median_itl_ms`）。21.0 来自更早的 NJA 那次运行（REPORT §1.1），原始数据不在工作区。差别很小。
8. **5.4 节 TTFT 阶梯"65 → 128 → 199 → … → 1,311 ms"**：精确数字来自 NJA 那次运行（REPORT §1.4），原始数据不在工作区。RS0 复测的原始 JSON 里阶梯形状相同（`R/RS0/node_raw/runs/sg_c16_s42.json` 前 16 条排序后是 68、132、233、335 … 1338 ms），可以用它替代。
9. **7.3 节"本次测试中途就有节点被回收"**：被回收的是大模型测试的 lx-d 节点（`RESULTS.md` 第 120 行），不是 Qwen3-30B 的节点。

## 逐节数字表

### 摘要、1.2 本文要点

| 数字 | 出处 | 状态 |
|---|---|---|
| 16 并发吞吐 +56%（688 → 1,075 tok/s） | `R/RS0/node_raw/runs/sg_c16_s42.json`（688.3）、`RS7F/runs/sg_c16_s42.json`（1074.9） | 原始 |
| TTFT P99 −79%（1,288 → 276 ms） | 同上（1288.4、276.1） | 原始 |
| TPOT 22.5 → 14.4 ms（−36%） | 同上（22.53、14.43） | 原始 |
| 32 并发 1,753 tok/s | `RS7F/runs/sg_c32_s42.json`（1753.0） | 原始 |
| GSM8K 没有下降、51 分钟 soak 0 失败 | `R/RS0/node_raw/gsm.txt`（0.934）、`RS7F/gsm.txt`（0.946）、`RS7F/soak/summary.txt`（6089 / fail 0 / span_min 50.8） | 原始 |
| 对 RTX PRO 6000 是 1.04 倍，对 H100 是 0.65 倍 / 1.54 倍 | trn2：`RS7F/runs/sg_c16_s42.json`；`G7E/w8a16/summary.txt`（13.81 / 1137.2）；`H100F/w8a16/summary.txt`（9.38 / 1659.2） | 原始（工作区外） |
| prefill MoE 每层快 1.9–2.8 倍 | `cte_result.json` 的 `numbers` 表 | 仅报告 |
| 大 MoE 4.6–7.9 倍、DeepSeek-V3 TTFT −53–59% | —— | 范围外 |

### 1.1 典型问题

| 数字 | 出处 | 状态 |
|---|---|---|
| 128 个专家选 8 个 | 模型配置（REPORT §1.2） | 公开模型参数（工作区内没有 config.json 原件） |
| 单个专家每层约 9.4 MB | 3 × 2048 × 768 × 2 B = 9.44 MB | 推导 |
| 16 请求每层专家并集约 82 个，48 层约 37 GB | 128 × (1 − (120/128)^16) ≈ 82；82 × 9.44 MB × 48 ≈ 37.1 GB（REPORT §1.2） | 推导 |

### 1.3 测试说明、10.1 版本

| 数字 | 出处 | 状态 |
|---|---|---|
| vLLM 0.29 | `RS7F/server.log`（`version 0.29.0`） | 原始 |
| vllm-neuron 0.29 / neuronx-cc | S3 STACK 前缀 `vn0.29.0.1.2.0_cc2.28.7631.0`；`k8s/neff-cache/README.md` | 原始（S3） |
| Neuron SDK 2.33、镜像 tag `...:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `k8s/neff-cache/rs-cache-test.yaml`；`BRIEF.md` 第 59 行；Qwen3-30B 的 rs pod 清单没有保存 | **推断** |
| `Qwen/Qwen3-30B-A3B`（BF16） | `research-qwen3-30b/scripts/rs_srv.sh`（`/data/models/Qwen3-30B-A3B`）；下载命令在 `<早期测试归档>/local-scripts/qwen3-moe-tp4.yaml` | 原始；HF revision **无依据** |
| H100：p5.48xlarge、Spot、驱动 580.178.04 / CUDA 13.0、vLLM 0.33.0 | `H100/REPORT.md` 第 1、40、45、51 行 | 原始（工作区外） |
| g7e.4xlarge、RTX PRO 6000、vLLM 0.33.0 同 digest | `G7E/image.txt`、`G7E/host.txt`；REPORT §2.8 | 原始（工作区外，image.txt 没有逐字核对） |
| 测试时间 Qwen3-30B 2026-11-06 ～ 11-09 | RS 系列的 `info` 里是 11/08–11/09；NJ 是 11/06–11/07（内部笔记） | 原始 + 仅报告 |
| 大 MoE 的测试时间和模型 | —— | 范围外 |

### 2.2 关键设计

| 数字 | 出处 | 状态 |
|---|---|---|
| 1 颗芯片、TP4、LNC=2 | `R/_joblogs/node_raw/job_RS7.sh`（`NEURON_LOGICAL_NC_CONFIG=2`）；`RS7F/server.log`（`tensor_parallel_size=4`） | 原始 |
| trn2.48xlarge 有 16 颗芯片 | 实例规格；内部笔记（neuron=16） | 仅报告 |

### 3.1 被测对象与负载

| 数字 | 出处 | 状态 |
|---|---|---|
| 48 层、max-model-len 8192、max-num-batched-tokens 8192、max-num-seqs 16 / 32 | `RS7F/info`、`RS7F/cfg`、`research-qwen3-30b/scripts/rs_srv.sh` | 原始 |
| ShareGPT 1 / 4 / 16 / 32 并发分别 12 / 32 / 64 / 128 条 | 各 `runs/sg_c*_s42.json` 的 completed（`R/RS7/summary_blog.txt` 里的 n=） | 原始 |
| 三方每档输入 token 总数逐一相同 | trn2 `in=2510 / 8869 / 17268 / 31028`（`R/RS7/summary_blog.txt`）；`H100F/w8a16*/summary.txt`、`G7E/w8a16*/summary.txt` 的 in_tok 列相同 | 原始（含工作区外） |
| GSM8K 标准误约 1.1 个百分点 | √(0.93 × 0.07 / 500) ≈ 1.14%；`H100/REPORT.md` 第 134 行 | 推导 |
| soak 50–70 分钟 | `RS7F/soak/summary.txt`（50.8）、`R/RS5/node_raw/soak/summary.txt`（71.2） | 原始 |

### 4.1 从基线到推荐配置

| 数字 | 出处 | 状态 |
|---|---|---|
| 基线 22.53 / 688 / 1,288 | `R/RS0/node_raw/runs/sg_c16_s42.json` | 原始 |
| ① 20.20 / 763 / 1,304 | `R/RS1/node_raw/runs/sg_c16_s42.json` | 原始 |
| ② 15.86 / 969 / 1,188 | `R/RS2/node_raw/runs/sg_c16_s42.json`（这一轮是 LOCK TIMEOUT 后跑的，见 `R/RS2/summary_blog.txt`） | 原始 |
| ③ 15.81 / 974 / 1,141 | `R/RS3/node_raw/runs/sg_c16_s42.json` | 原始 |
| ④ 14.43 / 1,075 / 276 | `RS7F/runs/sg_c16_s42.json` | 原始 |
| 括号里的百分比（−10%、+11%、−21%、+27%、−9%、+10%、−76%） | 相邻两行相除 | 推导 |
| 推荐配置 c1 TPOT 5.85、TTFT P99 96.5；c4 8.85 / 432；c32 17.24 / 1,753 / 581 | `RS7F/runs/sg_c1_s42.json`、`sg_c4_s42.json`、`sg_c32_s42.json` | 原始 |
| 共降 5.77 ms，FP8 占 4.34，打包占 1.38；约 3/4 和 1/4 | 20.20 − 14.43；20.20 − 15.86；15.81 − 14.43 | 推导 |
| 纯 decode 14.0 → 16.8 ms（约 +20%） | `R/RS3/node_raw/runs/dc128_c16_a.json`（ITL 14.03）、`dc128_c32_a.json`（16.74） | 原始 |
| 不打包时 32 并发 TTFT P99 2.4 s | `R/RS3/node_raw/runs/sg_c32_s42.json`（2404.9） | 原始 |

### 4.2 精度与稳定性

| 数字 | 出处 | 状态 |
|---|---|---|
| GSM8K 基线 0.934 | `R/RS0/node_raw/gsm.txt` | 原始 |
| FP8 各配置 0.946 | `R/RS2`、`RS4`、`RS6`、`RS7_final`、`RS7pm32`、`RS8` 下的 `node_raw/gsm.txt`（RS3 本身没有跑 GSM，RS6 是同配置复测） | 原始 |
| 三次 soak 共 20,052 个请求，0 失败 | 8376（`R/RS5/node_raw/soak/summary.txt`）+ 5587（`R/RS4/node_raw/soak/summary.txt`）+ 6089（`RS7F/soak/summary.txt`） | 原始 |
| 推荐配置 50.8 分钟、6,089 个请求、32 并发 2,827 个、随机长 prompt 3,035 个、prefill 1,362 万 token | `RS7F/soak/summary.txt`（span_min 50.8、conc 32 n 2827、rn n 3035、prefill tokens total 13624036） | 原始 |
| 2 条重复字符样本是正常内容 | `RS7F/soak/charrun_rerun{,2,3}.json`；分析在 REPORT §2.7 | 原始 + 仅报告 |

### 4.3 与 GPU 的同精度对比

| 数字 | 出处 | 状态 |
|---|---|---|
| W8A16 表 Trainium2 一列 | `RS7F/runs/sg_c{1,4,16,32}_s42.json` | 原始 |
| W8A16 表 H100 一列（4.92、6.49 / 593、9.38 / 1,659 / 160、11.40 / 2,658 / 254、GSM 0.932） | `H100F/w8a16/summary.txt`、`H100F/w8a16x32/summary.txt`（GSM 见同目录 / REPORT §2.8） | 原始（工作区外；GSM 没有逐个核对） |
| W8A16 表 RTX PRO 6000 一列（5.66、8.92 / 441、13.81 / 1,137 / 160、17.16 / 1,790 / 290、0.932） | `G7E/w8a16/summary.txt`、`G7E/w8a16x32/summary.txt` | 原始（工作区外；GSM 没有逐个核对） |
| BF16 表 Trainium2 一列（6.44、10.98 / 350、20.20 / 763 / 1,304、0.934） | `R/RS1/node_raw/runs/sg_c*_s42.json`、`R/RS1/node_raw/gsm.txt` | 原始 |
| BF16 表 H100 一列（4.75、8.34 / 463、12.82 / 1,224 / 146、0.930） | `H100/results/SUMMARY_all.txt`（TP1 段） | 原始（工作区外） |
| BF16 表 RTX PRO 6000 一列（6.72、12.68 / 312、22.15 / 714 / 162、0.930） | `G7E/bf16/summary.txt` | 原始（工作区外） |
| "相对"两列 | 相除 | 推导 |
| 长 prompt 表 Trainium2（16.08 / 943 / 1,439；22.33 / 647 / 3,332） | `R/RS3/node_raw/runs/b2_in1024_out512_c16.json`、`b2_in4096_out512_c16.json` | 原始 |
| 长 prompt 表 H100（9.25 / 1,628 / 412；11.19 / 1,240 / 1,590） | `H100F/w8a16/summary.txt` 的 `r5_in1024_out512_c16`、`r5_in4096_out512_c16` | 原始（工作区外） |
| 长 prompt 表 RTX PRO 6000（12.89 / 1,182 / 482；16.92 / 844 / 2,123） | `G7E/w8a16/summary.txt` 同名行 | 原始（工作区外） |

### 4.4 怎么读这组对比

| 数字 | 出处 | 状态 |
|---|---|---|
| H100 FP8 后快 27%（12.82 → 9.38） | 上面两个文件 | 推导 |
| RTX PRO 6000 带宽约 1.8 TB/s | g7e README（REPORT §2.8 引用），规格值 | 仅报告 |
| TPOT −38%（g7e 22.15 → 13.81）、Trainium2 −21%（20.20 → 15.86） | 上面的文件 | 推导 |
| 吞吐是 H100 的 0.52–0.58 倍、RTX PRO 6000 的 0.77–0.80 倍 | 长 prompt 表 | 推导 |
| 跨精度 1.13 倍 / 0.88 倍；基线时 1.76 倍 / 0.56 倍 | 14.43 / 12.82；1075 / 1224；22.53 / 12.82；688 / 1224 | 推导 |
| H100 W8A16 1 并发 4.92 vs BF16 4.75 | `H100F/w8a16/summary.txt`、`H100/results/SUMMARY_all.txt` | 原始（工作区外） |
| H100 官方 FP8（W8A8 block）16 并发 TPOT 8.98 ms，快约 4% | `H100F/fp8b/summary.txt`（`sg_c16_s42` 8.98） | 原始（工作区外） |

### 5.1 执行模型

| 数字 | 出处 | 状态 |
|---|---|---|
| PE 128×128、SBUF 每核约 24 MiB、HBM 每芯片 96 GiB、约 2.9 TB/s | 硬件规格（`BRIEF.md`、REPORT §1.2 引用） | 仅报告（工作区内没有官方出处） |
| 软件 DGE 大块连续搬运约 348 GB/s/物理核 | `prefill-moe-kernel.html`；内部笔记 | 仅报告 |

### 5.2 decode

| 数字 | 出处 | 状态 |
|---|---|---|
| 基线 16 并发 decode 一步约 21 ms | `R/RS0/node_raw/runs/dc128_c16_a.json`（ITL 21.09） | 原始 |
| 39.6 GB（37.1 + 1.8 + 0.6）、1.88 TB/s、约 65%；H100 约 92% | REPORT §1.2 的公式 | 推导 |
| MoE 约占一步的 85% | 0.378 ms × 48 = 18.2 ms，÷ 21.1 ms（REPORT §1.2）；0.378 来自 microbench | 推导（输入是仅报告的数） |
| 16 并发 ITL 中位数 21.0 ms | REPORT §1.1（NJA）；RS0 原始是 20.90 | 仅报告（见"发现的问题" 7） |
| prefill 插入约占 7% | (22.6 − 21.0) / 22.6（REPORT §1.1） | 推导 |
| min(64, 4T) 个位置；T=16 每层约 56 µs、48 层约 2.7 ms、约 12% | REPORT §1.3（nsl=64/48/40/32 的实验） | 仅报告 |
| 每层 378 → 338 µs | REPORT §2.5 | 仅报告（见"发现的问题" 5） |
| 端到端 TPOT −10% | `R/RS0`、`R/RS1` 的 `runs/sg_c16_s42.json` | 原始 |
| 每物理核约 315 GB/s、约 87%；每分区 12 KiB 连续块 | REPORT §1.3 | 仅报告 |
| FP8 版字节和执行时间都减半（0.190 → 0.098 s） | REPORT §1.3（profile 结论） | 仅报告 |
| T=16 每层 338 → 202 µs | REPORT §2.5 | 仅报告（与 `R/_joblogs/node_raw/log_MB1.txt` 的 0.2195 ms 不一致） |
| 纯 decode 16 批一步 18.3 → 14.0 ms | `R/RS1/node_raw/runs/dc128_c16_a.json`（ITL 18.18；REPORT 写 18.25，是 a/b 两次的值）、`R/RS2/node_raw/runs/dc128_c16_a.json`（14.00） | 原始 |
| 量化误差 2.65% vs 2.64%；per-channel 每层多约 60 µs | REPORT §2.5 | 仅报告 |
| profile 表 157.2 / 74.3 / 19.6 µs，59.6% / 28.3% / 7.5% | `R/PROF/decode_profile_RS7.txt`（AR 10.2 + 9.4 = 19.6；3.9% + 3.6% = 7.5%） | 原始 |
| FP8 作为 moving 操作数，单条耗时和 BF16 一样 | REPORT §2.5b；`R/PROF/decode_profile_RS7.txt`（MATMUL median 311 ns） | 原始 + 仅报告 |
| HBM 侧余量约 15% | 620 vs 725 GB/s/rank（REPORT §2.5b） | 推导 |
| attention 每层读约 17.6 MB；开头 10–17 µs 只有 GpSimd 在收集转置 K | REPORT §2.5c；`R/PROF/decode_profile_RS7.txt`（0–10.5 µs 只有 `DMA_GATHER_TRANSPOSE`） | 原始 + 仅报告 |
| 收益 5–7%；两项合计约 20% | 0.7–1.0 ms / 14.4 ms；15% + 5–7% | 推导 |

### 5.3 prefill（整节没有原始日志）

| 数字 | 出处 | 状态 |
|---|---|---|
| T=2048 时每块重搬约 2.25 MB，每块约 3.4 MB | `cte_result.json`（summary）、`prefill-moe-kernel.html` | 仅报告 |
| 98% DMA 字节走软件 DGE、约 135 GB/s、GpSimd 71% 忙 | 同上 | 仅报告 |
| 预留 191 块、真实约 145 块 | 同上 | 仅报告 |
| 分组每层约 0.33 ms | 同上；`BRIEF.md`（约 330 µs） | 仅报告 |
| 每块约 28 µs vs 理想约 7 µs | 同上 | 仅报告 |
| 8 条 lane（2 核 × 4） | 同上 | 仅报告（设计描述） |
| 误差 4.36e-3 → 2.87e-3 | `cte_result.json` | 仅报告 |
| 动态循环 4.3 ms vs 原版 2.7 ms | `cte_result.json`、`prefill-moe-kernel.html` | 仅报告 |
| DeepSeek-V3 每组约 0.7 ms | —— | 范围外 |
| 每层耗时表（2.10 … 6.63；0.76 … 3.57；2.8× … 1.9×） | `cte_result.json` 的 `numbers` 表（stock + mapping 列、nm + mapping + combine 列） | 仅报告 |
| 端到端 A/B 表（119.9 / 154.2 → 67.9 / 93.8；308.3 / 543.8 → 219.1 / 420.2；584.9 / 561.5 → 354.5 / 336.5；23.83 / 24.16 → 22.54 / 22.71；640 / 618 → 688 / 667；1 并发 6.44） | `prefill-moe-kernel.html` | 仅报告。"改造后"的 22.54 / 688 与 RS0 复测原始值（22.53 / 688.3）一致，可以作为间接佐证 |
| 71 分钟 soak 0 失败 | `prefill-moe-kernel.html`（6,557 个请求） | 仅报告 |
| T=2048 kernel 1.06 ms vs 下限 0.79 ms；空块占 20–28%；PE 平均 75–80% | `cte_result.json`（open_issues）、`prefill-moe-kernel.html` | 仅报告 |

### 5.4 调度

| 数字 | 出处 | 状态 |
|---|---|---|
| 阶梯 65 → 128 → 199 → … → 1,311 ms，每级 75–90 ms | REPORT §1.4（NJA） | 仅报告（见"发现的问题" 8） |
| 每次读全部 128 个专家，约 58 GB | 128 × 9.44 MB × 48 | 推导 |
| 最多 16 条或 4,096 token；请求数那一维固定为 16 | `R/_joblogs/node_raw/job_RS7.sh`、REPORT §3（knob `pack_max 16 / pack_budget 4096`）；`RS7F/p5check.txt`（pack_max=16） | 原始 |
| 只有 4 处写死"只有一条" | `research-qwen3-30b/multi_prefill_design.md` | 仅报告（设计文档） |
| 6.90 s → 约 1.2 s，一个 step 最多 14 条 | `RS7F/p5check.txt`；`R/RS7/summary_blog.txt`（6.90 / 1.17 / 1.20，max 14） | 原始 |
| 16 并发 TTFT P99 1,141 → 276；另一组 seed 1,213 → 381 | `R/RS3`、`RS7F` 的 `runs/sg_c16_s42.json`、`sg_c16_s123.json`（1212.9 → 380.5） | 原始 |
| TTFT 中位数 213 → 255 ms | 同上（212.9 → 255.3） | 原始 |
| 单请求 TTFT 128 / 2,048 / 4,096 / 8,000 token 是 54 / 119 / 197 / 390 ms | `RS7F/ttft.txt`（53.5 / 118.7 / 197.2 / 390.4） | 原始 |

### 5.5 哪些结论可以推广

| 数字 | 出处 | 状态 |
|---|---|---|
| 每个专家中间维 768，TP4 后每 rank 192 | 模型配置；`prefill-moe-kernel.html` | 公开模型参数 |
| DeepSeek-V3 单个专家约 88 MB | —— | 范围外 |

### 第六章 大 MoE

整章（6.1–6.3 和图 6）都是范围外，本次没有核对。出处大致在 `RESULTS.md`、`data/`、`k/`、`e2e/`；保留与否待用户决定。

### 7.1 什么负载适合

| 数字 | 出处 | 状态 |
|---|---|---|
| 32 并发 1,753 tok/s；第 ① 步 −10% / +11%；4K → 512 是 0.52 倍 / 0.77 倍；TTFT P99 276 vs 约 160 | 同 4.1、4.3 | 原始 |
| 打包要求贪心解码 | `RS7F/cfg`（`all_greedy:true`）；REPORT §3 | 原始 |
| DeepSeek-V3 那一行 | —— | 范围外 |

### 7.2 编译与冷启动

| 数字 | 出处 | 状态 |
|---|---|---|
| 空缓存完整编译约 43 分钟，其中 8,192 prefill 图约 25 分钟 | `cte_result.json`（integration：about 43 minutes / about 25 minutes） | 仅报告 |
| 推荐配置增量编译 31 个图，从启动到可服务约 1 小时 | `RS7F/info`（11:04:16 → 12:02:57，`neffs_after 89 new=31`）；`RS7F/server.log`（init engine took 3328 s） | 原始 |
| Qwen3-235B 2.3–2.5 小时、DeepSeek-V3 约 7 小时 | —— | 范围外 |
| 306 MB / 2 秒 / 全部命中；38.9 → 27.8 分钟 | S3 `_meta/cache-test/20261108T134029Z/summary.txt`（`neff=31 MB=306 sec=2`；ccW `startup_sec=1667` new=0；ccC `startup_sec=2334` new=23）；`k8s/neff-cache/README.md` 第 132–137 行 | 原始（S3），本地只有 README |
| 2,048 → 2,049 token 时 TTFT 119 → 197 ms | `RS7F/ttft.txt`（118.7 → 196.9） | 原始 |
| 单芯片编译峰值约 76 GiB | `RS7F/info`（`mem_peak_GiB 76`）；RS3 是 82 | 原始（见"发现的问题" 6） |
| Qwen3-235B 1.3–1.5 TiB；多机 barrier `14400` | —— | 范围外（单芯片用不到） |

### 7.3 Kubernetes、容量与上线检查

| 数字 | 出处 | 状态 |
|---|---|---|
| neuron-ls 的 PCI 地址核对 | 内部笔记；REPORT §0（BDF b5/b6/cb/6f/58） | 仅报告 |
| `NEURON_SKIP_EFA_AFFINITY=1` | `RS7F/info` | 原始 |
| 中途有节点被 Spot 回收 | `RESULTS.md` 第 120 行（大模型的 lx-d 节点） | 原始（但不是 Qwen3-30B 的节点） |
| 打包补丁第一次上机没有生效 | REPORT §2.6 | 仅报告 |
| 逐条与同时发完全一致只有 12/16 | REPORT §2.6；同类数据见 `R/RS4/node_raw/p5check.txt`（打包对打包 13/16） | 原始 + 仅报告 |
| DMA 引擎放置问题 | 内部笔记 | 仅报告 |
| "显存"一行（引用 6.3 节） | —— | 范围外（来自 Qwen3-235B） |
| soak 45 分钟以上 | 建议值 | —— |

### 7.4 成本

只有公式，没有数字。

### 第八章 POC 压测注意事项

| 数字 | 出处 | 状态 |
|---|---|---|
| Qwen3 generation_config 的 temperature 0.6 | `H100/REPORT.md` 第 83 行 | 原始（工作区外，报告） |
| 600–760 ms 的 TTFT 尖峰 | `H100/REPORT.md` 第 85 行 | 仅报告（工作区外） |
| 4 并发双峰 | 内部笔记 | 仅报告 |
| 200 题标准误约 1.3 个点 | 同上。按 √(p(1−p)/200) 算：p≈0.96（当时 PR56 的 95.5–96.5%）是约 1.4 个点；按本文 0.93–0.95 的准确率算是 1.5–1.8 个点 | 仅报告，数字建议复核 |

### 第九章 总结

数字都和前文重复：0.65 倍（4.3）、1.9–2.8 倍（5.3，仅报告）、+56%（4.1）、4.6–7.9 倍（范围外）。

### 10.2 推荐配置的关键参数

| 数字 | 出处 | 状态 |
|---|---|---|
| 启动参数和 `--additional-config`（buckets、num_seqs_buckets、decode_context_length_buckets、kv_segment_size_buckets、all_greedy） | `RS7F/cfg`、`RS7F/info`、`R/_joblogs/node_raw/job_RS7.sh` | 原始 |
| KV block 4,352 → 8,704 | `R/RS0/node_raw/server.log`、`RS7F/server.log`（`num_gpu_blocks_override=4352` / `8704`）；`R/_joblogs/node_raw/job_RS7.sh`（`NGB=8704`） | 原始 |
| 约可容纳 34 个满长请求 | 8704 × 32 / 8192 = 34 | 推导 |
| 9216 让块大小从 32 缩到 4 | 内部笔记 | 仅报告 |
| 备选 14.70 ms / 1,047 tok/s / 308 ms | `R/RS4/node_raw/runs/sg_c16_s42.json` | 原始 |
| 不打包时 1.1 s / 2.4 s | `R/RS3/node_raw/runs/sg_c16_s42.json`（1141.2）、`sg_c32_s42.json`（2404.9） | 原始 |

### 10.3 局限与声明

没有数字。其中"推荐配置没有跑 1K / 4K → 512"属实：`RS7F/runs/` 下没有 `b2_*`。

## 建议（不在本次交代范围内，没有做）

- 把 S3 `_meta/cache-test/20261108T134029Z/` 下的几个小文件拉到本地归档，让冷启动数字也有本地依据。
- 5.3 节和 5.2 节 microbench 的数字，在 blog 里加一句"来自 kernel 作者的 microbench 报告"；或者在复现环境里用 `research-qwen3-30b/k/wsB_t1.py` 重测。
- 第八章第 5 条"200 题标准误约 1.3 个点"：按本文 0.93–0.95 的准确率算是 1.5–1.8 个点，建议复核后再定。
