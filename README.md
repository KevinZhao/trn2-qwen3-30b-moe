# trn2-qwen3-30b-moe

在 AWS Trainium2（trn2.48xlarge 的 1 颗芯片，TP4 / LNC=2）上用 vLLM 0.29 + vllm-neuron 0.29 服务 Qwen3-30B-A3B 的优化代码、测试脚本和文档（2026-11，脱敏版）。

推荐配置 RS7（FP8 专家权重 W8A16 + T≤32 融合 decode MoE kernel + bs32 + 多请求打包 prefill），相对基线：ShareGPT 16 并发 TPOT 22.53 → 14.43 ms，吞吐 688 → 1,075 tok/s，TTFT P99 1,288 → 276 ms；32 并发 1,753 tok/s；GSM8K 500 = 0.946。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/trn2-handoff.md` | **从这里开始**：结论、环境现状、从零复现（步骤 / 耗时 / 花费）、踩过的坑、测试矩阵和判定标准、未结事项 |
| `KERNELS.md` | **自研 kernel 和补丁清单**：每个文件做什么、用哪个环境变量开关、哪个配置在用、怎么部署进镜像 |
| `docs/blog-draft-notes.md` | 博客草稿里每个关键数字的出处 |
| `qwen3-30b/REPORT.md` | 完整研究报告（诊断、各方向结果、跨平台同精度对比、推荐配置、重启方法） |
| `qwen3-30b/multi_prefill_design.md`、`share-*.md/html`、`anim-*.html` | 多请求打包 prefill 设计；分享材料 |
| `qwen3-30b/pkg_rs1/`、`pkg_rs2/`、`pkg_rs3/` | 替换进 `vllm_neuron/model/qwen3_moe/` 的模型代码包（RS1 = BF16 decode kernel，RS2 = FP8，RS3 = FP8 + 打包；推荐用 pkg_rs3） |
| `qwen3-30b/k/` | NKI kernel 变体和 microbench |
| `qwen3-30b/scripts/` | 服务（`rs_srv.sh`、`rs_main.sh`）、压测（`rs_bench.sh`、`cbv_cli.py`）、soak、调度和打包补丁（`p4_patch_sched.py`、`p5_patch_pack.py`）、`prof/` 整步 profile |
| `qwen3-30b/jobs/` | 实际执行过的 pod 任务脚本和日志（`job_RS7.sh` 启动 + bench、`job_RS7s.sh` soak、`job_RS7r.sh` 暖缓存重启等） |
| `qwen3-30b/tools/prefill-prof/` | prefill kernel 的 profile 分析脚本 |
| `qwen3-30b/results/` | 各配置的结果摘要（原始数据没有发布） |
| `prefill-kernel/` | prefill MoE kernel 的说明和报告（`prefill-moe-kernel.html`、`cte_result.json`、`dge_result.json`） |
| `k8s/neff-cache/` | NEFF 编译缓存放 S3 的工具（`neff_cache.sh`）、IAM 策略、pod 模板 |
| `blog/` | 技术博客草稿（md 和图）及生成脚本，**尚未发布**。Word 版不放在仓库里，可以用 `blog/scripts/build_docx.py` 从 md 生成 |

## 说明

- 账号、桶名、集群名、实例 ID、主机名、内部路径都已替换成 `<占位符>`，含义见 `docs/trn2-handoff.md` 第 7 节。使用 `k8s/neff-cache/neff_cache.sh` 前要设置 `NEFF_CACHE_BUCKET`。
- 脚本是在测试集群上直接用的工具脚本，里面写死了 pod 内路径（`/tmp/ni2e/`、`/data/...`）和跳板机调用方式（`rsh.sh`、`rrun.sh`、`rship.sh`），换环境前需要调整。
- MoE kernel、调度补丁和模型代码是为本次测试自研的，基于 vllm-neuron 社区 PR 的 Qwen3 MoE 实现，不在 Neuron / vllm-neuron 的官方发布中。生产使用前请自行验证。
- 推荐配置的开关：`II_DGE` 在代码里默认是 `hw`，必须显式设 `II_DGE=sw`（rs_srv.sh 已设），完整开关见 `KERNELS.md` 第 5 节。
- 所有数字都是 2026-11 的版本（Neuron SDK 2.33）在少量 Spot 节点上测得，仅供参考。
