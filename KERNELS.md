# 自研 kernel 与补丁清单

本次测试自研的 NKI kernel、模型代码、调度补丁和测试工具都在本仓库里。下表列出每个组件的文件、开关和使用它的配置。

## 1. 部署方式（服务器启动时由 `qwen3-30b/scripts/rs_srv.sh` 自动完成）

1. 把一个模型代码包（`qwen3-30b/pkg_rs1/`、`pkg_rs2/` 或 `pkg_rs3/`）的全部 `.py` 拷进镜像的 `vllm_neuron/model/qwen3_moe/`。
2. 在 `vllm_neuron/model/registry.py` 里注册 `Qwen3MoeForCausalLM`（rs_srv.sh 里内联的 Python 片段）。
3. 先把 `vllm_neuron/vllm/core/scheduler.py` 恢复成原版，再打调度补丁 `p4_patch_sched.py`。
4. `P5P=1` 时再打打包补丁 `p5_patch_pack.py`（改 scheduler、`neuron_model_runner.py`、`neuron_worker.py`）。
5. 按配置导出环境变量，然后 `vllm serve`。

镜像：`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`（vLLM 0.29.0、vllm-neuron 0.29.0.1.2.0、neuronx-cc 2.28.7631.0）。

## 2. 模型代码包

| 包 | 对应配置 | 内容 |
|---|---|---|
| `pkg_rs1/` | RS0 基线（`Q3_FUSED_MOE_KMOD=p2_fused11n`）、RS1（`=p2_fused11p`） | 包含基线 pkg_nj 的全部文件（md5 逐个一致），另加 11o、11p 两个 decode kernel。纯 BF16 |
| `pkg_rs2/` | RS2、RS3、RS5、RS6 | pkg_rs1 + FP8 专家权重（`RS_FP8=1`：加载时在线量化成 e4m3、每专家一个标量 scale）+ `p2_fused11r` + `nm_k9f` |
| `pkg_rs3/` | **RS4、RS7（推荐）**、RS7pm32、RS8 | pkg_rs2 + 多请求打包 prefill 的模型侧支持（full prefill 分支、显式 padding mask）。15 个文件的 md5 与 RS7 运行时记录一致 |

包里的模型代码起点是 vllm-neuron 社区 PR #56 的 Qwen3 MoE 实现（文件头保留了原有的 `SPDX-License-Identifier: Apache-2.0`）。

## 3. Kernel

### 3.1 decode MoE（`Q3_FUSED_MOE=1`，用 `Q3_FUSED_MOE_KMOD` 选择）

| 文件 | 说明 | 用在 |
|---|---|---|
| `ii_fused11a.py` | 第一版融合 decode MoE kernel（blob 权重布局），加了整图 BAD_OPCODE 的规避 | 早期集成；`Q3_FUSED_MOE_LAYOUT` 不是 nat 时的默认值 |
| `p2_fused11n.py` | 11a 改成直接读 PR56 原生权重布局（gu `[E,H,2I]`、dn `[E,I,H]`） | RS0 基线 |
| `p2_fused11o.py` | 11n + padding slot 按位反转顺序和真实 slot 交错，消掉串行尾巴 | 过渡版本 |
| `p2_fused11p.py` | 11o + 支持 T≤32（DN 拆成 2 个 64 行 PE tile） | RS1 |
| `p2_fused11r.py` | 11p + FP8（e4m3）专家权重、每专家标量 scale（gate scale 并进 SiLU activation scale，up×down 并进 affinity），BF16 activation | **RS2 及以后、RS7** |
| `q3_fused_moe.py` | decode kernel 的集成层（选 kernel 模块、`Q3_FUSED_MOE_MAX_T`） | 所有配置 |
| `k/p2_fused11q.py` | FP8 per-channel scale 版本，每层多约 60 µs，**已弃用** | 只用于 microbench |
| `k/p2_fused11s.py` | 11r + 开关（SPLIT / SC / ND / HWW），用来测每专家开销，都没有收益 | 只用于 microbench |

### 3.2 prefill MoE（`NM_MOE=1`，用 `NM_KMOD` 选择）

| 文件 | 说明 | 用在 |
|---|---|---|
| `nm_k9.py` | 新 prefill 专家 kernel：按专家顺序切成 2 核 × 4 条 lane，专家变化时才重新搬权重；权重走 Sync 队列硬件 DGE，token gather 走软件 DGE；空块越界跳过；静态展开加跨块流水 | RS0、RS1（`NM_KMOD` 默认值） |
| `nm_k9f.py` | nm_k9 的 FP8 专家权重版 | RS2 及以后（`NM_KMOD=nm_k9f`） |
| `nm_glue.py` | 设备端 mapping（每个 rank 本地扫描专家，不再做集合通信）和集成 | 所有 `NM_MOE=1` 的配置 |
| `nm_comb.py` | combine kernel：按 token 排列 gather 后以 fp32 求和 | 同上 |

原理和数据：`prefill-kernel/prefill-moe-kernel.html`、`prefill-kernel/cte_result.json`；hw-DGE 崩溃根因和安全写法：`prefill-kernel/dge_result.json`。

### 3.3 模型代码里的其它开关（`model_bf16.py`）

| 开关 | 作用 |
|---|---|
| `RS_FP8=1`（`RS_FMAX` 默认 240） | 加载时把专家权重量化成 FP8 |
| `WSA_FN=1` | decode attention 融合 input RMSNorm |
| `P3_KVIX=1` | segmented prefill 改用显式 KV scatter（原来 kernel 内写会拷贝整块 KV cache） |
| `II_DGE` | decode kernel 的 DGE 方式。**代码默认是 `hw`，生产必须设 `II_DGE=sw`**（hw 会触发 BAD_OPCODE），rs_srv.sh 已设 |
| `P5_PACK=1`、`P5_PACK_BUCKETS`（默认 `512,1024,2048,4096`） | 打包 prefill 的模型侧开关和打包长度档位 |

## 4. 调度补丁

| 文件 | 作用 | 开关 |
|---|---|---|
| `scripts/p4_patch_sched.py` | 移植 vllm-neuron PR #68（两个 prefill chunk 之间插 decode step），并扩展为"两次 prefill 之间插 decode step"；支持运行时 knob 文件和 step 日志 | `P4_DSBPC=1 P4_DSBP=1`；`P4_KNOBS=<json>`；`P4_STEPLOG=<file>` |
| `scripts/p5_patch_pack.py` | 多请求打包 prefill：调度器按先到先服务把符合条件的新请求（整条 prompt 放得下、前缀缓存未命中、无结构化输出）打进同一个 prefill step | `P5P=1`（rs_srv.sh）+ `P5_PACK=1`；knob 文件里的 `pack_max`（推荐 16）和 `pack_budget`（推荐 4096）；**要求 `on_device_sampling_config.all_greedy=true`** |
| `scripts/p5_pack_check.py` | 打包检查：逐条发和同时发对比，从 steplog 统计多请求 prefill step 数 | 服务 READY 后运行 |

## 5. 推荐配置 RS7 的完整开关

```
NM_MOE=1 Q3_FUSED_MOE=1 Q3_FUSED_MOE_KMOD=p2_fused11r Q3_FUSED_MOE_MAX_T=32 RS_FP8=1 NM_KMOD=nm_k9f P5_PACK=1
WSA_FN=1 II_DGE=sw P3_KVIX=1 P4_DSBPC=1 P4_DSBP=1 P4_KNOBS=/tmp/ni2_knobs_RS7.json
NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1
knob: {"dsbpc":1,"dsbp":1,"pack_max":16,"pack_budget":4096}
vllm serve ... --tensor-parallel-size 4 --max-num-seqs 32 --max-model-len 8192 --max-num-batched-tokens 8192 --num-gpu-blocks-override 8704
  neuron_config: num_seqs_buckets [1,4,8,16,32]、on_device_sampling all_greedy、其余见 qwen3-30b/REPORT.md §3
```

实际执行过的命令：`qwen3-30b/jobs/job_RS7.sh`（启动 + 打包检查 + bench）、`job_RS7s.sh`（soak）、`job_RS7r.sh`（暖缓存重启）。

## 6. 测试和分析工具

| 路径 | 用途 |
|---|---|
| `qwen3-30b/k/wsB_t1.py`、`k/nm_common.py` | decode MoE kernel 的数值检查和 microbench（`KMOD`、`T`、`L`、`LAYOUT`） |
| `qwen3-30b/k/nm_e2e8.py` | 单层 prefill MoE（mapping + 专家 + combine）对比：stock vs nm |
| `qwen3-30b/k/pa_test.py` | 打包 causal flash_attention 对逐序列参考的正确性测试 |
| `qwen3-30b/k/pe_probe.py` | PE 单条矩阵乘耗时探针（**计时不可信，已弃用**） |
| `qwen3-30b/scripts/prof/` | 整步 decode profile：`ie_prof.sh` 抓取，`pick2.py` 选最新一代 NEFF，`ie_ana.py` 按层切，`att3/att4/moe_ana*.py` 看段内指令 |
| `qwen3-30b/tools/prefill-prof/` | prefill kernel profile 分析：`nm_capt.py`（用 dump 的张量构造输入后 capture）、`nm_util.py`、`nm_tl.py`、`nm_gaps.py`、`nm_sum.py` |
| `qwen3-30b/jobs/` | pod 内每个任务的脚本和日志（microbench、PE 探针、profile、旋钮扫描、告警重跑） |

## 7. 没有包含的（本地已经找不到）

以下组件出现在过程记录里，但源码只存在于已经删除的临时目录或节点上，本仓库**没有**：

- prefill kernel 的早期 microbench 框架 `nm_stock.py`、`nm_mine.py`、`nm_hf.py`，以及更早的 `nm_k5.py`。现有的 `k/nm_e2e8.py` 和 `k/nm_common.py` 可以替代做 stock vs nm 的对比。
- 第二轮 prefill 改进 NI1 / NZ1（`nf_map2` 设备端 mapping、`nh_k2` 64-token 半块，包 `pkg_ni`、`pkg_nz`）。不在 RS 系列和推荐配置里。
- hw-DGE 安全变体 `p2d_fused.py`（TPOT 只快 1–3%，opt-in），以及早期的 `wsB_fused9.py`（已被 11a / 11n 取代）。

RS 系列（RS0–RS8，含推荐配置 RS7）需要的代码都已经包含。
