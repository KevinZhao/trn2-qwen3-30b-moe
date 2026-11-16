# trn2 MoE 推理测试交接：Qwen3-30B-A3B 单芯片（2026-11-15）

> **范围**：只覆盖 trn2 上 Qwen3-30B-A3B 的测试与验证。更大的模型（Qwen3-235B、DeepSeek-V3）以后再说，本文不展开。
> **脱敏**：本文是公开版，账号、资源名、实例 ID、内部路径都已替换成 `<占位符>`，见第 7 节。
> **标记**："无依据"表示工作区里找不到原始数据或记录；"未核实"表示今天没能用只读命令确认。
> **路径约定**：没有前缀的相对路径都相对于 `<工作区>/`；`R/` = `research-qwen3-30b/results/`；工作区外的路径写全（`<工作区外>`），并注明"工作区外"。

---

## 1. 测试目标和主要结论

### 1.1 目标

在 1 颗 Trainium2（取自 trn2.48xlarge，TP4 / LNC=2）上服务 Qwen3-30B-A3B（BF16 checkpoint，max-model-len 8192），回答三个问题：能不能跑、同精度下和 GPU 比怎样、需要多少工程量。在此基础上做高并发吞吐优化，并与 1 张 H100、1 张 RTX PRO 6000 Blackwell（g7e）做同口径、同精度对比。

主报告：`research-qwen3-30b/REPORT.md`（2026-11-09 定稿，工作停在 RS7）。

### 1.2 配置演进（全部 1 device、TP4/LNC2、MML 8192）

| tag | 内容 | 精度 | 包 | 原始数据 |
|---|---|---|---|---|
| RS0 | NJ 基线同机复测（已做过 decode 优化 + 新 prefill MoE kernel `NM_MOE=1`） | BF16 | pkg_nj（原目录已不在；`pkg_rs1/` 包含其全部文件，见 6.2） | `R/RS0/node_raw/` |
| RS1 | + decode MoE kernel p2_fused11p（去掉 padding 白算，支持 T≤32） | BF16 | `research-qwen3-30b/pkg_rs1/` | `R/RS1/node_raw/` |
| RS2 | + FP8 专家权重（W8A16，每专家一个标量 scale），p2_fused11r + nm_k9f | W8A16 | `research-qwen3-30b/pkg_rs2/` | `R/RS2/node_raw/` |
| RS3 | RS2 + bs32 bucket（MNS32、NGB 8704） | W8A16 | pkg_rs2 | `R/RS3/node_raw/` |
| RS4 | RS2 + 多请求打包 prefill（P5，pack_max 16 / budget 4096） | W8A16 | `research-qwen3-30b/pkg_rs3/` + `research-qwen3-30b/scripts/p5_patch_pack.py` | `R/RS4/node_raw/` |
| **RS7（推荐）** | RS3 + 打包 | W8A16 | pkg_rs3 + P5 补丁 | `R/RS7_final/node_raw/` |
| RS5 | RS3 配置的 71 min soak | W8A16 | pkg_rs2 | `R/RS5/node_raw/soak/` |
| RS6 | RS3 同配置复测 GSM8K | W8A16 | pkg_rs2 | `R/RS6/node_raw/` |
| RS7pm32 / RS8 | 调打包参数（pack_max 32；budget 8192），无收益 | W8A16 | pkg_rs3 | `R/RS7pm32/`、`R/RS8/` |

### 1.3 关键数字（ShareGPT，seed 42，每次 reset prefix cache；均已对原始 JSON 核对）

| 指标 | RS0 基线 | RS7 推荐 | 变化 | 出处 |
|---|---|---|---|---|
| c16 TPOT（ms） | 22.53 | **14.43** | −36% | `R/RS0/node_raw/runs/sg_c16_s42.json`、`R/RS7_final/node_raw/runs/sg_c16_s42.json` |
| c16 输出吞吐（tok/s） | 688.3 | **1074.9** | +56% | 同上 |
| c16 TTFT P99（ms） | 1288.4 | **276.1** | −79% | 同上 |
| c1 TPOT（ms） | 6.45 | 5.85 | −9% | `.../runs/sg_c1_s42.json` |
| c32 TPOT / tok/s / TTFT P99 | —（MNS16） | 17.24 / **1753.0** / 580.5 | — | `R/RS7_final/node_raw/runs/sg_c32_s42.json` |
| GSM8K 500 | 0.934 | 0.946 | 噪声内（标准误约 1.1 点） | `R/RS0/node_raw/gsm.txt`、`R/RS7_final/node_raw/gsm.txt` |
| soak | — | 50.8 min，6089 请求，0 失败 | — | `R/RS7_final/node_raw/soak/summary.txt` |

- 逐步拆分（c16 s42）：RS1 20.20 ms / 763 tok/s → RS2 15.86 / 969 → RS3 15.81 / 974 → RS7 14.43 / 1075。TPOT 的改善约 3/4 来自 FP8，约 1/4 来自打包；TTFT P99 的改善几乎都来自打包（RS3 1141 → RS7 276）。各 `R/RS*/node_raw/runs/sg_c16_s42.json`。
- 三次 soak 合计 20,052 请求、0 失败：RS5 8376（`R/RS5/node_raw/soak/summary.txt`）+ RS4 5587（`R/RS4/node_raw/soak/summary.txt`）+ RS7 6089。RS7 soak 有 2 条"重复字符"告警，重跑确认是正常内容（`R/RS7_final/node_raw/soak/charrun_rerun{,2,3}.json`）。
- 不接受 FP8 时的零风险选项是 RS1：c16 TPOT −10%、吞吐 +11%，GSM8K 0.934（`R/RS1/node_raw/`）。

### 1.4 同精度对比（GPU 数据在工作区外，已对 summary.txt 逐项核对）

| ShareGPT | trn2 RS7（W8A16） | H100 TP1 W8A16 | g7e W8A16 | trn2 / H100 | trn2 / g7e |
|---|---|---|---|---|---|
| c16 TPOT（ms） | 14.43 | 9.38 | 13.81 | 1.54× | 1.04× |
| c16 tok/s | 1075 | 1659 | 1137 | 0.65× | 0.95× |
| c32 tok/s | 1753 | 2658 | 1790 | 0.66× | 0.98× |
| c16 TTFT P99（ms） | 276 | 160 | 160 | 1.7× | 1.7× |

出处（工作区外）：`<GPU_H100_FP8>/results/{w8a16,w8a16x32}/summary.txt`，`<GPU_G7E>/results/{w8a16,w8a16x32}/summary.txt`。BF16 对 BF16（RS1）：c16 TPOT 对 H100 是 1.58×，对 g7e 是 0.91×（`<GPU_H100_BF16>/results/SUMMARY_all.txt`、`<GPU_G7E>/results/bf16/summary.txt`）。长 prompt（4K→512 c16）trn2 吞吐只有 H100 的 0.52×（`R/RS3/node_raw/runs/b2_in4096_out512_c16.json`）。

### 1.5 原理层结论（详见 REPORT §1、§2.5）

- decode MoE 受 HBM 带宽限制：收益来自少搬字节（FP8）和不白算 padding。整步 profile（bs16 ctx1024）中每层 MoE 157.2 µs（59.6%），attention 74.3 µs（28.3%）（`R/PROF/decode_profile_RS7.txt`）。
- TTFT 长尾的根因是 vllm-neuron 调度器每步只跑一条 prefill（`max_prefills_per_batch=1`），解决办法是打包。
- 剩下的方向（W8A8 decode MoE 上限约 15%，转置 K cache 约 5–7%）用户在 11/9 决定不做。

---

## 2. 环境与资源现状

**结论先说：测试用的 EKS 集群 `<EKS_CLUSTER>` 已在 2026-11-10 删除，trn2 节点、bastion、集群 VPC 都已不在。还在的只有 S3 桶 `<NEFF_CACHE_BUCKET>`（NEFF 缓存加结果）和 IAM 角色 `<NEFF_CACHE_ROLE>`。复现要从建集群开始。** 此前内部笔记里"保留 trn2-spot / neuron-helm-chart / ns trn2-perf"的记录已经过时。

2026-11-15 只调用 describe/list/get 类只读 API在 ap-south-2 做了只读核实：

| 资源 | 此前记录 | 2026-11-15 只读核实结果 | 状态 |
|---|---|---|---|
| EKS 集群 `<EKS_CLUSTER>`（私有 endpoint） | 在 | `aws eks describe-cluster` → ResourceNotFoundException；`list-clusters` 为空（ap-south-2/ap-south-1/ap-southeast-2/us-east-1/us-east-2/us-west-2/eu-north-1 都查了）；CloudTrail `DeleteCluster` 记录：2026-11-10T20:51:19+08:00，发起方 `<INSTANCE_ID>` | **已删** |
| bastion `<INSTANCE_ID>` | 在 | `describe-instances` 无记录；ap-south-2 目前没有任何 EC2 实例 | **不在**（删除时间未核实） |
| trn2 节点（含 Qwen3-30B 用的 <NODE> / `<INSTANCE_ID>`） | 2026-11-10 02:47 UTC 回收 | ap-south-2 无 trn2 实例 | **已删** |
| 节点本地缓存 `/mnt/k8s-disks/0/trn2-perf`（pod 内 `/data`：模型、数据集、`rs_RS*` 暖缓存、`rs_results`、`rs_pkg`、pkg_nj） | REPORT §5 写"保留" | 实例存储随节点终止而消失 | **已不存在**（推断，无法直接查） |
| NodePool / EC2NodeClass `trn2-spot`、ns `trn2-perf`、helm `neuron-helm-chart` 1.11.2、my-scheduler、ConfigMap `neff-cache-scripts`、SA `neff-cache`、Pod Identity 关联 | 保留 | 集群已不存在，k8s 对象无法单独查询 | **随集群删除**（推断；Pod Identity 关联未单独核实） |
| NodePool `lx-trn2`、DaemonSet `lx-efa-device-plugin`、Deployment `trn2-holder`、k8s-neuron-scheduler | 11/10 已删 / 缩到 0 | 同上 | 已删 |
| 集群 VPC 和 S3 Gateway endpoint | —— | ap-south-2 只剩默认 VPC（`IsDefault=True`，172.31.0.0/16），VPC endpoint 为空 | **不在** |
| S3 桶 `<NEFF_CACHE_BUCKET>` | 在，81 个 TAG、约 163 GB，暂不清理 | 存在；标签 owner/cluster=<EKS_CLUSTER>/purpose=trn2-neff-cache；STACK `vn0.29.0.1.2.0_cc2.28.7631.0` 下有 **83 个 TAG、59,781 个对象、约 419.0 GB** | **在**（此前内部笔记里的数量已过时） |
| └ NEFF：`v1/vn0.29.0.1.2.0_cc2.28.7631.0/rs_RS0..rs_RS7`、`rs_RSP`、`nj_*`、`nzc_NZ1` | rs_RS0..7 在，rs_RS8 没推 | `rs_RS7`：179 个对象、840 MB，其中 89 个 NEFF，与 `R/RS7_final/node_raw/neffs_after` 的 89 个文件名**完全一致**（`comm -3` 差集为 0）；`rs_RS8` 不在 | 在 |
| └ 结果：`results/qwen3-30b-rs/20261109/rs_results/` | 在 | 821 个对象、125.8 MB（= 本地 `R/MANIFEST_node_raw.txt` 的 820 个文件 + 清单） | 在 |
| └ 冷启动实测原始日志：`_meta/cache-test/20261108T134029Z/` | 未提 | 在：`summary.txt`、`cc{W,C}_info.txt`、`cc{W,C}_server.log.gz` 等（见 6.1） | 在 |
| "S3 结果桶" | —— | 没有单独的结果桶，结果都放在上面这个桶的 `results/` 前缀下 | —— |
| IAM 角色 `<NEFF_CACHE_ROLE>` | 在 | `get-role` 返回，创建于 2026-11-08T13:32:48Z | **在** |
| trn2.48xlarge Spot 价（ap-south-2b） | $10.3157/h | `describe-spot-price-history` 11-14～11-15 一直是 10.315700 | 核实 |

只有 ap-south-2b 有 trn2 Spot（来源：`<内部笔记>`，今天没有再核实可用区）。

---

## 3. 怎么从零复现 RS7

### 3.1 需要的东西和路径

| 东西 | 路径 | 说明 |
|---|---|---|
| 镜像 | `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | rs-a…e 这几个 pod 的清单没有保存下来，tag 是推断的（见 6.1）。`k8s/neff-cache/rs-cache-test.yaml`（RS0 配置）和 `k8s/neff-cache/*.yaml` 用的都是这个 tag |
| 集群基础设施 | `aws-samples/sample-eks-enterprise-quickstart`（工作区外；commit `0c01085` 起支持 Neuron） | 当时 trn2 部分没有走这个仓库的 tfvars，是带外加的（见下一行） |
| trn2 NodePool / EC2NodeClass / ns | `<早期测试归档>/bastion/trn2/trn2-spot.yaml`（工作区外，与 `local-scripts/trn2-spot.yaml` 相同） | 注意文件里 NodePool `limits.cpu: "192"` = 1 台；里面的 `trn2-holder` 占位 Deployment 用完要删 |
| Neuron helm | 命令记录：`<早期测试归档>/bastion/trn2/_run_1793784090_65170_9975.sh`、`_run_1793784227_65964_23334.sh`（工作区外） | `neuron-helm-chart` 1.11.2，`scheduler.enabled=true`，scheduler 加 nodeSelector `karpenter.sh/nodepool=trn2-spot` 和 neuron toleration |
| S3 权限 | `k8s/neff-cache/neff-cache-sa.yaml`、`iam-trust.json`、`iam-policy-neff-cache-bucket-rw.json`、`README.md` | IAM 角色还在，Pod Identity 关联要对新集群重建 |
| NEFF 工具 | `k8s/neff-cache/neff_cache.sh`（做成 ConfigMap `neff-cache-scripts` 挂到 `/scripts`） | `pull|push|ls|stack` |
| pod 模板 | `k8s/neff-cache/rs-cache-test.yaml`（带 SA、ConfigMap、1 device、160Gi） | 原模板 `<内部临时路径>` 本机已不存在；`tmp/mkpod.sh` 里 base64 的 lx-a 模板也可以参考 |
| 模型代码包 | `research-qwen3-30b/pkg_rs3/` | 15 个 .py 的 md5 与 RS7 运行时记录（`R/RS7_final/node_raw/info`）**逐个一致**，NEFF 缓存可以命中 |
| 服务和压测脚本 | `research-qwen3-30b/scripts/`：`rs_srv.sh rs_main.sh rs_bench.sh rs_soak.sh rs_soak.py p4_patch_sched.py p5_patch_pack.py p5_pack_check.py cbv_cli.py cbv_ana.py ie_nan.py` | 脚本里写死了 pod 内路径 `/tmp/ni2e/`、`/data/models/Qwen3-30B-A3B`、`/data/datasets/...`、`/data/cache/ie_bench.lock` |
| 实际执行过的命令 | `R/_joblogs/node_raw/job_RS7.sh`（启动 + 打包检查 + bench）、`job_RS7s.sh`（soak）、`job_RS7r.sh`（暖缓存重启） | 照抄即可 |
| 模型下载 | `snapshot_download('Qwen/Qwen3-30B-A3B', ...)`，见 `<早期测试归档>/local-scripts/qwen3-moe-tp4.yaml`（工作区外） | HF revision 没有记录（无依据） |
| ShareGPT | `anon8231489123/ShareGPT_Vicuna_unfiltered` 的 `ShareGPT_V3_unfiltered_cleaned_split.json`，见 `<早期测试归档>/local-scripts/s_ds.sh` | 放到 `/data/datasets/` |
| GSM8K | `cbv_cli.py` 读 `/data/datasets/gsm8k/test.jsonl` | 下载来源**无依据**（工作区没有记录），取前 500 题 |

### 3.2 步骤、耗时、花费

花费按 trn2.48xlarge Spot $10.3157/h（ap-south-2b，2026-11-15 核实）计算。只用 1 颗芯片，但按整台计费。

| # | 步骤 | 预计耗时 | 花费 | 依据 |
|---|---|---|---|---|
| 1 | 建 VPC + EKS（私有 endpoint）+ bastion + Karpenter：用 `sample-eks-enterprise-quickstart` | **无依据**（工作区没有记录建集群用了多久） | 非 trn2 部分**无依据** | —— |
| 2 | 装 `trn2-spot.yaml`、neuron-helm-chart（scheduler.enabled）；建 SA `neff-cache` 和 Pod Identity 关联；建 ConfigMap `neff-cache-scripts` | 无依据 | —— | 3.1 表 |
| 3 | 创建 1 device pod（`schedulerName: my-scheduler`、`serviceAccountName: neff-cache`、hostPath `/mnt/k8s-disks/0/trn2-perf`→`/data`、memory 160Gi），Karpenter 拉起 trn2.48xlarge | 节点拉起时间无依据 | 从这一步开始按 $10.32/h 计费 | `k8s/neff-cache/rs-cache-test.yaml` |
| 4 | **先核对镜像**：pod 里执行 `bash /scripts/neff_cache.sh stack`，必须输出 `vn0.29.0.1.2.0_cc2.28.7631.0`，否则 NEFF 不会命中 | 1 min | —— | `k8s/neff-cache/neff_cache.sh` |
| 5 | 下载模型、ShareGPT、GSM8K 到 `/data/models`、`/data/datasets` | 无依据 | 计费中 | 3.1 表 |
| 6 | 把 `research-qwen3-30b/pkg_rs3/` 放到 `/data/cache/rs_pkg/pkg_rs3`，`research-qwen3-30b/scripts/` 放到 `/tmp/ni2e/` | 几分钟 | —— | `R/_joblogs/node_raw/job_RS7.sh` |
| 7 | 拉 NEFF：`mkdir -p /data/cache/rs_RS7/vc && bash /scripts/neff_cache.sh pull rs_RS7 /data/cache/rs_RS7/vc`（89 个 NEFF / 840 MB） | RS7 没测过；RS0 配置拉 306 MB 用了 2 s | —— | S3 `_meta/cache-test/20261108T134029Z/summary.txt` |
| 8 | 启动 RS7：照 `R/_joblogs/node_raw/job_RS7.sh` 执行（`TAG=RS7 PKG=/data/cache/rs_pkg/pkg_rs3 NMV=1 SRC=<任意> P5P=1 KEEP=1 NSB="[1,4,8,16,32]" MNS=32 NGB=8704 NMX="Q3_FUSED_MOE_KMOD=p2_fused11r RS_FP8=1 NM_KMOD=nm_k9f Q3_FUSED_MOE_MAX_T=32 P5_PACK=1 P4_STEPLOG=..." PHASES=nan bash /tmp/ni2e/rs_main.sh`）。`/data/cache/rs_RS7` 已经存在时，rs_main 不会去拷贝 SRC | **约 40–45 min（推断）**：RS7 带完整暖缓存重启实测 41.6 min，new NEFF=0；RS0 配置从 S3 拉缓存后启动实测 27.8 min，RS7 的 bucket 更多 | 约 $7–8 | `R/RS7_restart_1109/node_raw/info`（05:51:12→06:32:47）；S3 `_meta/cache-test/.../summary.txt` |
| 8' | （S3 不可用时）空缓存冷编 | **> 1 h（推断，没有 RS7 从零编译的实测）**：RS7 在 RS3 缓存上增量编 31 个 NEFF 用了 58.7 min；RS0 配置空缓存 38.9 min | 约 $10–15 | `R/RS7_final/node_raw/info`（11:04:16→12:02:57，new=31）；S3 cache-test `ccC startup_sec=2334` |
| 9 | 写 knob：`{"dsbpc":1,"dsbp":1,"pack_max":16,"pack_budget":4096}` → `/tmp/ni2_knobs_RS7.json`；跑 `p5_pack_check.py`，确认 steplog 里有多请求 prefill step | 1 min | —— | REPORT §5；`R/RS7_final/node_raw/p5check.txt` |
| 10 | bench：`PHASES="sgx perf ttft gsm" bash /tmp/ni2e/rs_bench.sh RS7` | **约 15 min**（READY 12:02:57 → bench done 12:18:10） | 约 $2.6 | `R/RS7_final/node_raw/blog` |
| 11 | soak：`SOAK32=1 SOAK_MIN=50 bash /tmp/ni2e/rs_soak.sh RS7` | **约 51 min** | 约 $9 | `R/RS7_final/node_raw/soak/summary.txt`（span_min 50.8） |
| 12 | 结果写到 S3（`aws s3 sync /data/cache/rs_results/<TAG> s3://<桶>/results/<run id>/`），本机 `aws s3 sync` 拉回；新编的 NEFF 在 READY 且 20 分钟无写入后 `neff_cache.sh push <TAG>` | 几分钟 | —— | 内部笔记、`trn2-neff-cache-s3` |
| 13 | **释放**：删 pod → 删 `trn2-holder` → 把 k8s-neuron-scheduler 缩到 0 → 等 WhenEmpty 回收节点 → 用 `aws ec2 describe-instances` 确认实例已 terminated | WhenEmpty 1h（`trn2-spot.yaml` 里是 `consolidateAfter: 1h`） | 空转 1 h 约 $10 | 内部笔记 |

**合计**（节点就绪之后，从 S3 拉缓存走到 soak 跑完）：约 2 h，Spot 约 $20。集群重建、节点拉起、模型下载的时间和非 trn2 的费用没有记录，另算。走冷编再加约 1 h（约 $10）。

注意事项：
- 别在 pod 里改 `model_bf16.py` 的行号布局，否则所有图都要重编（约 45 min，见 4.2）。
- `rs_main.sh` 的 `save()` 会 `rm -rf /data/cache/rs_results/<TAG>`。在已经有结果的 TAG 下重启服务器之前，先备份。
- `rs_srv.sh` 本身**没有**内置 S3 pull/push（内置的是大模型那几个 `e2e/*_srv.sh`），所以第 7 步和第 12 步要手动做。

---

## 4. 踩过的坑和规避办法

### 4.1 测量与压测

| 坑 | 规避 | 来源 |
|---|---|---|
| Neuron 上贪心解码不能逐字复现：同一组 prompt 逐条发和同时发，完全相同的只有 12/16（batch 组成改变了 kernel 数值）；同一实例重复跑只有 31–36% 完全一致 | 不要求 100% 一致：跟"同路径的 batch 噪声"这条基线比，再结合首 token 一致率和 GSM8K 500 题 | REPORT §2.6；内部笔记 |
| GSM8K 200 题的标准误约 1.3 点，分不出常见的差异 | 至少 500 题（标准误约 1.1 点） | 同上；REPORT §0 |
| 不 reset prefix cache 时结果偏好；c4 有客户端相位造成的双峰 | 每轮之前 `POST /reset_prefix_cache`；c4、c16 用 seed 42 和 123 各跑一次 | `research-qwen3-30b/scripts/rs_bench.sh` 的 `RST()`；内部笔记 同上 |
| 同一节点上多个 pod 同时压测会互相干扰 | 用 `flock /data/cache/ie_bench.lock` 串行；日志里出现 `LOCK TIMEOUT` 表示这一轮是没拿到锁跑的（RS2 就是这样），这种结果要标注 | `research-qwen3-30b/scripts/rs_bench.sh` 的 `LOCK()`；`R/RS2/summary_blog.txt` |
| 单核 microbench 比 e2e 乐观（e2e 里 4 个 rank 争 HBM）：FP8 kernel micro 0.20 ms/层，e2e 约 0.23 | 关键结论一律在服务里 e2e 复测 | 内部笔记 |
| microbench 每次迭代 `.cpu()` 会把整个输出拷回来，引入约 30 ms 常数和 ±0.5 ms 抖动 | 编译后的 step 只返回很小的切片（`SMALLOUT=1`），按 L 层减 1 层的斜率计时 | 内部笔记 |
| nkilib `moe_cte` 的 `skip_weight` 在真正发生权重复用时数值错误 | 不要把"stock + skip_weight"当成有效基线 | 同上 |
| pod 内的任务链：`pgrep -f "a\|b"` 不能当"或"用；改一个正在运行的 bash 脚本不会生效；`lx_batch` 的输出是块缓冲 | 分开调用 pgrep；要改就 kill 后重启；按 CASE_END 计数轮询 | 同上 |
| GPU 对照：vLLM 0.33 默认读 generation_config（temperature 0.6），客户端不传温度时做的是随机采样；Triton MoE JIT 预热不够时，c1 TTFT 出现 600–760 ms 尖峰 | 服务端加 `--override-generation-config '{"temperature":0.0}'`；预热覆盖所有 JIT 形状 | `<GPU_H100_BF16>/REPORT.md` 第 83、85 行（工作区外） |
| vLLM `/start_profile` 抓不到 decode NEFF，默认 profiler 缓冲在 NGB 4352 下也分配不出来 | 停掉服务，用 `research-qwen3-30b/scripts/prof/ie_prof.sh`（`neuron-explorer capture`）抓缓存里的 NEFF；用 `pick2.py` 按修改时间取最新一代（`ie_pick.py` 会取到最老的那一代） | REPORT §2.5b；内部笔记 |

### 4.2 编译和缓存

| 坑 | 规避 | 来源 |
|---|---|---|
| 拷别处的缓存，或在一次**启动失败**留下的缓存目录上重启 → `NCC_EVRF059`（`torch_compile`、`compile_cache` 引用了节点本地 `/var/tmp/nki-*`） | 把 `vc/neuron/compile_cache` 和 `torch_compile` 移开，只保留 `native_neff_cache`；清 `/var/tmp/nki-*` 时一并清 `$HOME/.cache/neuron_libtorch` | 内部笔记、`qwen3-30b-fp8-decode-research`；`research-qwen3-30b/scripts/rs_main.sh`；`BRIEF.md` |
| 只有 `native_neff_cache` 能跨节点用。命中要求同镜像、同包（源码行号布局一致）、同 bucket 配置、同 LNC/编译器版本 | 新 pod 先跑 `neff_cache.sh stack` 核对 STACK；包用 `pkg_rs3/` 原样，不要改 | 内部笔记；`k8s/neff-cache/README.md` |
| `model_bf16.py` 里任何改动行号的编辑都会触发全部重编（约 45 min） | 原地改、保持行数不变 | 内部笔记 |
| NKI 编译缓存不认 `II_DGE` 环境变量，不重建缓存就切换会静默用旧 kernel | 切 `II_DGE` 时换新的缓存目录 | 内部笔记 |
| `max-model-len 9216` 会让 KV block 从 32 缩到 4，性能明显下降 | 用 8192 或 16384 | 同上 |
| 编译峰值主机内存：RS7 76 GiB，RS3 82 GiB | pod memory 设 160Gi（模板里就是） | `R/RS7_final/node_raw/info`、`R/RS3/node_raw/info` 的 `mem_peak_GiB` |
| SDK 2.33 镜像在没有挂 EFA 的节点上直接报错 | 设 `NEURON_SKIP_EFA_AFFINITY=1`（rs_srv.sh 已设） | `R/RS7_final/node_raw/info` |

### 4.3 Kubernetes 和容量

| 坑 | 规避 | 来源 |
|---|---|---|
| 只要有 Pod 走默认调度器，neuron scheduler extension 的占用表里就没有它，会把已经在用的芯片再分配出去 | 所有 Neuron pod 都设 `schedulerName: my-scheduler`；核对时比较各 pod 里 `neuron-ls` 显示的 PCI BDF（容器内设备名都会重编号成 neuron0） | 内部笔记；REPORT §0 |
| extender 不接受 2 卡请求，1 卡和 4 卡可以 | 要 2 卡就申请 4 卡，再用 `NEURON_RT_VISIBLE_CORES` 限制 | 同上 |
| k8s-neuron-scheduler 处于 Pending 时，会让 Karpenter 再拉一台 trn2 | 不用时把它缩到 0；要用多卡之前再扩回 1 | 内部笔记 |
| Spot 回收会把节点本地缓存一起带走（11/08 大模型测试的 lx-d 节点就被回收过） | 编译产物 READY 后推 S3；新节点启动前先 pull | `RESULTS.md` 第 120 行；内部笔记 |
| 节点空转很贵：11/04 那一轮约 21.5 h，其中约 9 h 是测完后的空转 | 测完立刻按 3.2 第 13 步释放，并确认实例已 terminated | 内部笔记 |

### 4.4 结果回传

| 坑 | 规避 | 来源 |
|---|---|---|
| SSM 单次输出上限约 24 KB；tar 包曾经因为 SSM 往返丢块而损坏；`kubectl exec` 的 websocket 长连接会断 | 结果从 pod 直接 `aws s3 sync` 到带 run id 前缀的位置，本机再 `aws s3 sync` 拉回；SSM 只用来发命令和看简短状态；不要全局 `export AWS_PROFILE` | 内部笔记 |
| soak 的 req.jsonl 只存了每条输出的前 80 个字符，2 条告警只能靠反查 prompt 重跑来确认 | soak 要保存完整输出 | REPORT §2.7 |

### 4.5 自研 kernel 和补丁上线

| 坑 | 规避 | 来源 |
|---|---|---|
| NKI kernel 在动态循环（`nl.fori_loop`，上界在寄存器里）里做 sw-DGE gather/scatter，会让**下一个**集合通信挂死（随机 1–3 个 rank 收不到 MODEL_STOP）；单设备 microbench 永远测不出来 | 进 vLLM 图（有集合通信）的 kernel，sw-DGE 只能放在静态循环里；动态循环里只用 hw-DGE。复现用 4 层截断（约 10 min） | 内部笔记（发现于大模型，规则对 Qwen3-30B 的 TP4 图同样适用） |
| 普通 `dma_copy` 交给编译器放置引擎，在特定执行历史下会触发 BAD_OPCODE | kernel 里每条 DMA 都显式指定引擎；权重类 DMA 只从一条队列（Sync）发出 | 内部笔记 |
| `II_DGE=hw` 在 bs4→bs8 切换时会确定性地触发 BAD_OPCODE | 生产用 `II_DGE=sw`（rs_srv.sh 默认） | 同上；REPORT §2.5b |
| 打包补丁第一次上机时，编译和 NaN 检查都通过，但打包从未生效：vLLM 0.29 的 `find_longest_cache_hit` 返回 3 元组，补丁按 2 元组解包，异常被吞掉了 | 每个新功能都要有"确实生效"的正向指标。打包看 steplog 里 `S <t> P <tok> <nreq>` 中 nreq>1 的 step 数，或 `p5check.txt` 的 `max_reqs_per_prefill_step` | REPORT §2.6；`R/RS7_final/node_raw/p5check.txt`、`steplog.txt` |
| MoE prefill 的 padding mask 用 `argmax(positions)` 推断最后一个真实 token，打包后会把其它序列静默判成 padding，MoE 输出被置零 | 改用显式 mask；边界用例要包含"多条序列等长"的情况 | `research-qwen3-30b/multi_prefill_design.md` 第 18 行 |
| combine 依赖每个 token 恰好有 8 个非零 affinity，router 出 NaN 会写坏相邻 token | 每轮都跑 NaN 检查（`ie_nan.py`，soak 开始、中间、结束各一次） | 内部笔记 |

---

## 5. 辅导客户测试：建议的矩阵和顺序

原则：**先复现我们的基准，确认环境和代码没问题，再换成客户的负载**。每一步都有"怎么判断结果对"的标准，任何一步不通过就停下排查，不往下走。

### 5.1 顺序

| 阶段 | 内容 | 看什么 | 怎么判断对不对 |
|---|---|---|---|
| **0. 环境验收**（约 0.5 h） | pod 起来后 `neuron-ls`；`neff_cache.sh stack`；单条请求冒烟（"The capital of France is"） | BDF 不和其它 pod 重复；STACK 等于 `vn0.29.0.1.2.0_cc2.28.7631.0`；能输出 " Paris." | 三项都满足才继续。cache-test 冒烟的参考输出见 S3 `_meta/cache-test/.../summary.txt` |
| **1. 启动和数值** | 按 3.2 启动 RS7；`ie_nan.py`；`p5_pack_check.py` | `neffs_after` 与 `new=` 的数量；NaN 检查结果；打包检查 | 用 S3 缓存时 **new=0**（有新 NEFF 说明镜像、包或配置不一致）；NaN `RESULT PASS`；`max_reqs_per_prefill_step` > 1；逐条与打包两种方式的相同率在 11–13/16 这个水平，首 token 全部一致 |
| **2. 复现基准**（约 15 min） | `rs_bench.sh` 的 `sgx perf ttft gsm`：ShareGPT c1/c4/c16（seed 42、123）/c32、随机 256–7000、单请求 TTFT 扫描、GSM8K 500 | TPOT 均值和 P99、输出 tok/s、TTFT 均值/中位数/P99、ITL 中位数和最大值、fail 数 | 对照 `R/RS7/summary_blog.txt`：TPOT ±3% 以内、tok/s ±5% 以内（我们自己两组 seed 的差异：c16 TPOT 14.43 对 14.44，tok/s 1075 对 1049）；TTFT P99 受 seed 影响大（c16 是 276 对 381），按同一个 seed 比；GSM8K 应在 0.946 ± 0.02 之内（约 2 个标准误）；fail=0 |
| **3. 客户负载** | 用客户的真实 prompt 长度分布和请求回放替换 ShareGPT | 同上，外加按长度分档统计的 TTFT | 先看长度分布落在哪些 bucket（单请求 TTFT 在 2048→2049 从 119 跳到 197 ms，4096→4097 从 197 跳到 390 ms，见 `R/RS7_final/node_raw/ttft.txt`）；超过 4096 token 的 prompt 不会打包；按需调整 bucket（档位越多，编译越久） |
| **4. 采样方式确认** | 问清楚客户是否需要 temperature > 0 | —— | 打包要求 `all_greedy`。需要随机采样时，只能用不打包的配置（RS3），c16 TTFT P99 会回到约 1.1 s，c32 约 2.4 s（`R/RS3/node_raw/runs/sg_c16_s42.json`、`sg_c32_s42.json`） |
| **5. 稳定性** | soak ≥ 50 min，c1/c4/c16/c32 混合，加入长 prompt；开始、中间、结束各做一次 NaN 检查；金丝雀请求 | fail 数、坏样本、服务日志错误行（nrta / SEQUENCER / BAD_OPCODE / NRT_TIMEOUT / Traceback / NaN）、健康检查 | 0 失败、错误行 0、NaN 全部 PASS、金丝雀全部通过；**保存完整输出**（我们只存了 80 个字符，见 4.4） |
| **6.（可选）GPU 同口径对照** | 同一批 prompt，服务端温度强制 0，MML 和 MNBT 都是 8192，同精度 | 同阶段 2 | 每档的 `total_input_tokens` 两边要逐一相同（ShareGPT c1 2510、c4 8869、c16 17268、c32 31028）；预热要覆盖 JIT 形状 |

### 5.2 指标定义（与我们的数据口径一致）

- 工具：`vllm bench serve`（参数见 `research-qwen3-30b/scripts/rs_bench.sh` 里的 `sg()`、`rn()`、`b2()`），`--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99`，`--save-detailed`。
- 吞吐一律用**输出** tok/s；ShareGPT 输出长度 512；c1/c4/c16/c32 分别跑 12/32/64/128 条。
- GSM8K：`cbv_cli.py gsm`，500 题，zero-shot chat，关闭 thinking，温度 0，串行。
- 汇总：`cbv_ana.py <out dir>`。

### 5.3 选配置

| 客户场景 | 建议配置 | 依据 |
|---|---|---|
| 接受 FP8，16–32 并发 | RS7 | 1.3 节 |
| 只跑 c≤16 | RS4（MNS16，无 bs32 bucket）：c16 TPOT 14.70、1047 tok/s、TTFT P99 308 | `R/RS4/node_raw/runs/sg_c16_s42.json` |
| 不接受 FP8 | RS1（纯 BF16） | `R/RS1/node_raw/` |
| 需要随机采样 | RS3（不打包） | 5.1 阶段 4 |
| prompt 以 4K 以上为主、或 TTFT SLO 很严 | 先用客户数据实测，不要先给结论 | 1.4 节；REPORT §4 |

---

## 6. 还没结的事

### 6.1 blog 提交前要确认的三件

1. **镜像 tag 是推断的**。Qwen3-30B 的 rs-a…e pod 清单没有保存下来（模板 `<内部临时路径>` 本机已不存在），server.log 里也没有记录 tag。间接证据：
   - `R/RS7_final/node_raw/server.log` 里是 `version 0.29.0` 和 `python3.14` 路径；
   - `k8s/neff-cache/rs-cache-test.yaml` 明确写了这个 tag，用它启动的 pod 从 `rs_RS0` 拉缓存 31/31 命中、new=0。缓存命中要求同编译器版本，可以推出 rs pod 用的是同一套 vllm-neuron 0.29.0.1.2.0 和 neuronx-cc 2.28.7631.0；
   - `BRIEF.md` 第 59 行说"模板里已配置"这个 tag。
   
   确认办法：复现时用这个 tag 启动，`neff_cache.sh stack` 输出一致，并且拉 `rs_RS7` 后 new=0，就算坐实。在此之前，blog 10.1 节应保留"推断"的说明，或者由用户决定怎么写。
2. **冷启动 38.9 → 27.8 分钟：原始日志其实有，在 S3 上，本地没有**。`s3://<NEFF_CACHE_BUCKET>/_meta/cache-test/20261108T134029Z/summary.txt`里是 `ccW startup_sec=1667`（从 S3 拉缓存，31/31 命中）和 `ccC startup_sec=2334`（空缓存，编 23 个 NEFF），旁边还有 `cc{W,C}_info.txt` 和 `cc{W,C}_server.log.gz`。注意两点：一是测的是 RS0 / NJ 配置（pkg_nj，MNS16），不是推荐的 RS7；二是这两次是同一个 pod 先后跑的。blog 里写的是"一个 Qwen3-30B 配置"，这个说法没错。建议把这几个小文件拉到本地归档（本次没做，交代范围外）。
3. **235B 和 DeepSeek-V3 两段（blog 第六章，以及摘要、1.2、5.5、7.1、7.2、9、10 节里相关的句子）要不要留**：用户 11/15 说"更大的模型以后再说"，这次交接不覆盖；blog 里这些段落是否保留仍待用户决定。

### 6.2 其它

- **更大的模型以后再说**（Qwen3-235B、DeepSeek-V3 的工作在 `RESULTS.md`、`k/`、`e2e/`、`k8s/lx-*.yaml`，本文不涉及）。
- **集群已删**：要复现就得重建（第 2、3 节）。基线 RS0 用的 pkg_nj 原目录已经不在，但 `pkg_rs1/` 包含了 pkg_nj 的全部文件：和 `R/RS0/node_raw/info` 里记录的 md5 逐个比对，完全一致，只多了 `p2_fused11o.py`、`p2_fused11p.py`。所以 **RS0 基线可以用 `pkg_rs1` 加 `Q3_FUSED_MOE_KMOD=p2_fused11n`（rs_srv.sh 的默认值）复现**。RS1、RS2、RS3、RS7 的包都在本地。
- **5.3 节 prefill MoE kernel 的数字没有原始日志**：只在 `cte_result.json` 和 `prefill-moe-kernel.html` 这两份报告里有，原始产物在 `<内部临时路径>`（已不在）。详见 `handoff/blog-draft-notes.md`。
- **decode microbench（11n/11p/11r 每层 378/338/202 µs）没有原始日志**，只在 REPORT §2.5 里有。11/9 重测的 `R/_joblogs/node_raw/log_MB1.txt` 里，11r 在 T=16 时是 0.2195 ms（L=8，测法不同），和 202 µs 不一致。
- Qwen3-30B-A3B 的 HF revision、GSM8K 数据集的下载来源都没有记录（无依据）。
- 自研代码（pkg_rs1/2/3、`p4_patch_sched.py`、`p5_patch_pack.py`、NKI kernel）是基于 vllm-neuron 社区 PR #56 的未合入实现。**能不能把代码交给客户、以什么形式交**，需要用户决定。
- S3 桶现在约 419 GB，用户 11/09 定的是"暂不清理"。存储费用没有算。

---

## 7. 脱敏说明

本文由内部交接文档脱敏而来：

| 占位符 | 含义 |
|---|---|
| `<ACCOUNT_ID>` | 测试用的 AWS 账号 |
| `<EKS_CLUSTER>` | 测试用的 EKS 集群名 |
| `<NEFF_CACHE_BUCKET>` | 存 NEFF 编译缓存和测试结果的 S3 桶 |
| `<NEFF_CACHE_ROLE>` | 通过 EKS Pod Identity 访问该桶的 IAM 角色 |
| `<INSTANCE_ID>`、`<NODE>`、`<VPC_ID>` | EC2 实例 ID、节点主机名、VPC ID |
| `<工作区>`、`<工作区外>`、`<内部临时路径>`、`<bastion 路径>`、`<内部笔记>` | 原始工作区、其它本地目录、临时目录、跳板机上的目录、内部笔记 |
| `<GPU_H100_BF16>`、`<GPU_H100_FP8>`、`<GPU_G7E>`、`<早期测试归档>` | GPU 对照测试和早期测试的数据目录 |

另外：
- 客户名已改成"客户"。
- 账号凭证方式、服务配额、对 Neuron 内部版本和路线的调研都已删去。
- 路径对照：文中原始工作区的 `research-qwen3-30b/` 在本仓库是 `qwen3-30b/`；`k8s/neff-cache/` 相同；`blog-docx/` 是 `blog/`，`scripts/blog/` 是 `blog/scripts/`；`cte_result.json`、`dge_result.json`、`prefill-moe-kernel.html` 在 `prefill-kernel/`。`R/_joblogs/node_raw/`（任务脚本和日志）在 `qwen3-30b/jobs/`。自研 kernel 和补丁的完整清单见根目录 `KERNELS.md`。
- `R/<TAG>/node_raw/` 下的原始 bench JSON、server.log、soak 明细（约 120 MB）**没有发布**（`_joblogs` 除外），本仓库的 `qwen3-30b/results/` 只有各配置的摘要。其它没发布的有：大模型（Qwen3-235B、DeepSeek-V3）相关的文件，`research-qwen3-30b/ref/`（vllm-neuron 上游源码的参考副本）。

---

## 附：文件索引（Qwen3-30B 部分）

- 报告：`research-qwen3-30b/REPORT.md`；打包设计：`research-qwen3-30b/multi_prefill_design.md`
- 包：`research-qwen3-30b/pkg_rs{1,2,3}/`；kernel 和 microbench：`research-qwen3-30b/k/`
- 脚本：`research-qwen3-30b/scripts/`（服务 / 压测 / soak / 补丁），`research-qwen3-30b/scripts/prof/`（整步 profile）
- 原始结果：`R/<TAG>/node_raw/`（runs/*.json、server.log、info、gsm.txt、ttft.txt、soak/…），校验清单 `R/MANIFEST_node_raw.txt`（820 个文件），pod 任务脚本和日志 `R/_joblogs/node_raw/`
- K8s 和 S3 缓存：`k8s/neff-cache/`（README.md、neff_cache.sh、各种模板）
- blog：`blog-docx/`（说明见 `handoff/blog-draft-notes.md`）
