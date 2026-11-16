# trn2 Neuron 编译缓存（NEFF）S3 共享

2026-11-08 建。目的：换节点、Spot 回收重建、多个会话之间不再重复跑 neuronx-cc。

## 1. 资源（账号 <ACCOUNT_ID>，ap-south-2，集群 <EKS_CLUSTER>）

| 资源 | 值 |
|---|---|
| S3 桶 | `<NEFF_CACHE_BUCKET>`（标签 purpose=trn2-neff-cache / cluster=<EKS_CLUSTER> / owner=<OWNER>；Block Public Access 四项全开；SSE-S3 AES256 + Bucket Key；BucketOwnerEnforced；桶策略只拒绝非 TLS；生命周期：7 天清理未完成的分段上传） |
| 网络 | VPC 已有 S3 Gateway endpoint `vpce-06cd227cebdc268f8`（策略全放行），节点直连 S3，不走 NAT |
| IAM 角色 | `<NEFF_CACHE_ROLE>`：信任 `pods.eks.amazonaws.com`，条件限定 SourceAccount、集群 ARN、namespace=trn2-perf、SA=neff-cache（`iam-trust.json`） |
| 角色内联策略 | `neff-cache-bucket-rw`（`iam-policy-neff-cache-bucket-rw.json`）：桶上 ListBucket/GetBucketLocation；对象 Get/Put/AbortMultipartUpload/ListMultipartUploadParts；DeleteObject 只允许 `*.lock`（给 libneuronxla 的 S3 锁用）。**没有其他删除权限**，缓存只增不删 |
| K8s SA | `trn2-perf/neff-cache`（`neff-cache-sa.yaml`） |
| Pod Identity 关联 | `a-re5pcspiilm0bfvwh`（trn2-perf/neff-cache → <NEFF_CACHE_ROLE>）；eks-pod-identity-agent 已在所有 trn2 节点上 |
| ConfigMap | `trn2-perf/neff-cache-scripts`（`neff_cache.sh`、`neff_sync_host.sh`）；`trn2-perf/neff-cache-test`（实测用） |

没有改节点角色、NodePool、EC2NodeClass。现有 pod 用 default SA，**没有 S3 凭证**（IMDS 也拿不到节点角色），见第 5 节。

## 2. 什么可以共享、键由什么决定

vllm-neuron 0.29（SDK 2.33，`libtorch_neuronx_lite`）的缓存都在 `VLLM_CACHE_ROOT/neuron/` 下（会话脚本里 `VLLM_CACHE_ROOT=/data/cache/<tag>/vc`）：

| 子目录 | 内容 | 能否跨节点 |
|---|---|---|
| `native_neff_cache/<pk[0:2]>/<pk[2:4]>/<pk>.neff` | neuronx-cc 产物，文件名 = 编译器内部的 persistent key（由 HLO + 编译参数决定） | **能**，内容寻址 |
| `native_neff_cache/cache_key_index/<sha[0:2]>/<sha[2:4]>/<sha256(cache_key)>/<determinant>` | FX cache_key → persistent key 的索引（只给 precompile restore 用） | 能（每组编译器输入一个文件，多写者不冲突，设计上就支持共享目录） |
| `torch_compile/`、`compile_cache/`（含 `nki/`） | dynamo/inductor 产物、NKI kernel 缓存，引用节点本地 `/var/tmp/nki-*` | **不能**。带到别处会 `NCC_EVRF059` |

缓存命中条件（`libtorch_neuronx_lite/compile/native_backend.py`、`backend.py`）：
- FX cache_key = 图结构（**含模型源码的行号布局**：改 model_bf16.py 导致行号移动会让全部图失效）+ 输入形状/dtype（即 bucket：num_seqs、num_batched_tokens、context 长度、KV segment 等）+ 编译参数（按图的 -O 级别、平台注入的 fp8 参数等）。
- determinant = 平台 target（trn2）+ `NEURON_LOGICAL_NC_CONFIG` + `NEURON_COMPILER_OPT_LEVEL` + `NEURON_CC_FLAGS` + neuronx-cc 版本。
- 所以同镜像、同包（同源码文件）、同 bucket 配置就能命中，与节点、pod、缓存目录名、端口无关（RS0 就是把 nj_CAP 拷成 rs_RS0 后 31/31 命中）。
- NKI 缓存不认 `II_DGE` 这类 env 开关（见 neuron-eks pitfalls）——但它在 `compile_cache/nki`，不进 S3，所以不影响。

Neuron SDK 对 S3 的原生支持：
- **2.33 native 路径（vllm-neuron 0.29）没有 S3 支持**。`NEURON_LIBTORCH_REMOTE_CACHE` 只是共享文件系统路径（os/shutil 拷贝，给 NFS/Lustre 用），且 native NEFF 不看它；`TORCH_NEURONX_NEFF_CACHE_DIR` 也是本地/共享 FS 路径。所以这里用"启动前 `aws s3 sync` 拉、READY 后推"的方式。
- **S3 Mountpoint 不适合当可写缓存目录**：native 缓存用 `os.replace` 原子改名、FileLock，Mountpoint 不支持。
- **2.31.1 NxDI 镜像（vllm-neuron 0.5.3，neuronx-cc 2.26）原生支持** `NEURON_COMPILE_CACHE_URL=s3://…`（`libneuronxla/neuron_cc_cache.py` 的 `CompileCacheS3`，boto3，用 `.lock`/`.done` 对象做并发控制，所以策略放开了 `*.lock` 的删除）。约定路径：`s3://<NEFF_CACHE_BUCKET>/v1/vn0.5.3_cc2.26.6360.0/neuronxcc`。本次未实测（当时没有在跑的 NxDI 任务）。

## 3. S3 前缀约定

```
s3://<NEFF_CACHE_BUCKET>/
  v1/<STACK>/<TAG>/native_neff_cache/...        与本地 /data/cache/<TAG>/vc/neuron/native_neff_cache 一一对应
  v1/<STACK>/<TAG>/_manifest/<node>-<utc>.txt   每次 push 写一份：来源、NEFF 数、大小、编译相关 env、可选包 md5
  _meta/sync-logs/                              首次从节点灌入的日志
  _meta/cache-test/<run>/                       实测结果
```
- `STACK` = `vn<vllm-neuron 版本>_cc<neuronx-cc 版本>`，由 `neff_cache.sh stack` 在镜像里自动算。当前 2.33 镜像 = `vn0.29.0.1.2.0_cc2.28.7631.0`。换镜像就自动换前缀，不会混用。
- `TAG` = 会话原来的本地缓存目录名（`lx_q235_stock`、`rs_RS3`、`nzc_NZ1`……），保持和会话记录/memory 里的名字一致。
- 不同配置分开放；想"借"别的配置的图（例如 LX 复用 stock 的 decode 图）就在本地 pull 多个 TAG 取并集：NEFF 内容寻址，多余的文件不会被误用。

## 4. 用法

### 4.0 规则：重建时切换（用户 2026-11-09 定）

- **正在跑的 pod/服务不动**（不重启、不改 SA、不为了接 S3 而重建）。它们编出来的新 NEFF 由同节点 helper pod 代推（4.3）。
- **以后任何 trn2 pod 重建时**（Spot 回收、换节点、换配置、日常重建）都切到 S3 缓存：
  1. pod 用 `serviceAccountName: neff-cache`，并把 ConfigMap `neff-cache-scripts` 挂到 `/scripts`（`k8s/lx-c.yaml`、`lx-d.yaml`、`lx-e.yaml` 已改好；`lx-f` 是 2.31.1 的 CPU 辅助 pod，不编译，没改）；
  2. 启动 vllm 之前 `neff_cache.sh pull <TAG>`（同 TAG，必要时再加借用的 TAG）；
  3. 服务 READY 之后 `neff_cache.sh push <TAG>`。
- `e2e/lx_srv.sh`、`ds_srv.sh`、`mn_srv.sh` 已内置这两步，pod 按上面配好就自动生效：
  - S3 TAG = 本地缓存目录名 `lx_$TAG`（脚本里 `C=/data/cache/lx_$TAG`）；`NEFF_PULL="lx_xxx lx_yyy"` 额外拉别的 TAG 取并集；`NEFF_S3=0` 关闭。
  - pull 在 `vllm serve` 之前；没挂 `/scripts` 或没凭证（老 pod 用 default SA）时只记日志、照常冷启动，不会挡住启动。
  - push 在检测到 "Application startup complete" 之后、`touch READY` 之前同步执行（`FORCE=1`，因为此时所有 bucket 已编完；bench 等 READY 才开始，所以不会和 bench 抢资源）。增量 sync，全部命中时只需几秒。
  - `mn_srv.sh` 多机：node 0 同上；headless 节点没有 READY，后台等本机打出 "Model warmup completed" 后每 5 分钟试推一次（20 分钟内有写入会被拒绝），最多 1 小时。
  - 日志在 `/data/lx_out/<TAG>/neff_cache.log`。
- **多机 DS 两个节点都要 pull 同一个 TAG**：实测 `lx_ds_stock` 两机各 1408 个 NEFF，**文件名完全不重叠**（S3 并集 2816 个 / 80 GB），每个节点只会命中自己那一半，但拉整份最省事（约 2–3 min）。
- bastion 上 `/root/work/trn2/lx/` 和 pod 里 `/tmp/lx/` 的旧脚本没有同步改（避免影响正在跑的会话）；下次重建时从本仓库 `e2e/` 拷新版。


### 4.1 `neff_cache.sh`（ConfigMap `neff-cache-scripts`，挂到 `/scripts`）

```bash
bash /scripts/neff_cache.sh stack                 # 本镜像的 STACK
bash /scripts/neff_cache.sh ls                    # 列出所有 TAG
bash /scripts/neff_cache.sh ls rs_RS3             # 某 TAG 的 manifest
bash /scripts/neff_cache.sh pull rs_RS3 /data/cache/rs_RS9/vc   # S3 → 本地（只增不删）
bash /scripts/neff_cache.sh push rs_RS9 /data/cache/rs_RS9/vc   # 本地 → S3（READY 之后再推）
```
- `pull` 用 `aws s3 sync`，306 MB 约 2 s，11 GB 约 1 min 量级（Gateway endpoint）。
- `push` 拒绝推 20 分钟内还有写入的目录（可能仍在编译、NEFF 可能写了一半），`FORCE=1` 跳过；只推 `native_neff_cache`，跳过 `*.tmp`/`*.lock`。
- 需要 pod 用 `serviceAccountName: neff-cache`。镜像自带 aws-cli 2.38。

### 4.2 新 pod

本项目的 lx pod 直接用 `k8s/lx-*.yaml` + `e2e/*_srv.sh`（4.0）。其他会话的新 pod 见 `example-vllm-s3cache.yaml`：`serviceAccountName: neff-cache` + initContainer 预拉 + 主容器 READY 后 push。改动最小的写法是在现有 `*_srv.sh` 里加两行：

```bash
bash /scripts/neff_cache.sh pull $TAG $C/vc          # 启动 vllm serve 之前（C=/data/cache/$TAG）
...  # 等到 "Application startup complete"
bash /scripts/neff_cache.sh push $TAG $C/vc          # touch READY 之后
```

### 4.3 现有 pod（没有凭证、不想重建）

在同节点起不占 Neuron 设备的 helper pod 代拉/代推（只挂 hostPath，不碰现有 pod）：
- 代推（只读挂载）：`neff-sync-pod.tmpl.yaml` + `neff_sync_host.sh`（推整个节点的全部 TAG；`SKIP` 里列正在编译的 TAG；另外自动跳过 20 分钟内有写入的目录）。
- 只推指定 TAG（只读挂载）：`neff-push-pod.tmpl.yaml`，`__TAGS__` 空格分隔。先确认服务 READY，推完删 pod。
- 代拉（读写挂载）：`neff-pull-pod.tmpl.yaml`，`PAIRS="src_tag:dst_tag ..."`，拉进 `/data/cache/<dst_tag>/vc/neuron/native_neff_cache`，然后会话照常用 `VLLM_CACHE_ROOT=/data/cache/<dst_tag>/vc` 启动。
- 两种 helper 都用 `nodeName` 直接钉到节点、默认调度器，不申请 neuron 资源，不会和 my-scheduler 的占卡表冲突。bastion 上的副本在 `/root/work/trn2/neff-cache/`。

## 5. 注意事项

1. **只同步 native_neff_cache。** 不要把 `torch_compile`/`compile_cache` 放进 S3 或拷到别的节点。从别处拉来 NEFF 的目录里如果有失败启动留下的 `torch_compile`/`compile_cache`，先挪开（`NCC_EVRF059`）。
2. **S3 只省 neuronx-cc 的时间。** 启动时 dynamo 重新 trace、NKI kernel 重编、加载权重、warmup 照常进行；实测数字见第 6 节。
3. **push 时机**：服务 READY（所有 bucket 编译完）之后。多节点（DS 两机）两边同 TAG 都 push，NEFF 取并集；`cache_key_index` 同名文件可能被后写者覆盖，只影响 precompile restore（跨节点本来就不带 torch_compile，不走这条路）。
4. **并发**：native 路径不需要锁（文件名内容寻址，同名同内容覆盖无害）；两个会话同时冷编同一配置仍会各编各的——想省就让后起的等先起的 push 完再 pull。
5. **改源码**：模型 .py 行号移动 → 全部图新 key，旧 NEFF 不再命中（不会出错，只是白存）。改了包就换 TAG（或沿用 TAG 但 manifest 里记 `NEFF_CACHE_PKG` 的 md5）。
6. **清理**：角色没有删除权限，旧前缀由管理员手工删（`aws s3 rm --recursive s3://…/v1/<STACK>/<TAG>/`）。2026-11-09 补推后约 163 GB（38,772 个对象），S3 标准存储约 $3.9/月。用户 2026-11-09 定：**暂不清理**（S3 对象和节点本地缓存都先保留）。
7. 不要全局 `export AWS_PROFILE`；pod 内凭证来自 Pod Identity，本机按命令带 `--region ap-south-2`。

## 6. 已灌入的缓存与实测

### 6.1 首次灌入（2026-11-08 13:37–13:40 UTC，helper pod 只读挂 hostPath）

| 节点 | 会话 | TAG 数 | NEFF | MB | 跳过 |
|---|---|---|---|---|---|
| <NODE>（lx-c） | 大型 MoE | 16 | 8259 | 28527 | `lx_q235_lx`（正在编译）、`lx_q4_lx_noapc`（空） |
| <NODE>（lx-e） | 大型 MoE | 4 | 4404 | 15956 | `lx_ds_stock`（正在编译） |
| <NODE>（lx-d） | 大型 MoE | 1 | 768 | 1965 | `lx_ds_stock`（正在编译） |
| <NODE>（rs-b） | Qwen3-30B | 59 | 1795 | 17809 | — |

S3 合计 79 个 TAG、30,525 个对象、67.2 GB（`lx_dsmn5_stock` 两机取并集）。重点：`lx_q235_stock`（235B stock，1296 NEFF / 14.8 GB）、`lx_ds5_stock`（DS-V3 5 层，2560 / 11.3 GB）、`lx_dsmn5_stock`、`lx_q4_*`、`rs_RS0..RS7`、`nzc_NZ1`、`ni2_NI1`、`nj_CAP`。当时没灌的 `lx_q235_lx`、`lx_ds_stock` 已于 2026-11-09 补推（见 6.3）。日志在 `_meta/sync-logs/`。

### 6.2 命中实测（rs-cache-test，<NODE> 的空闲 device，RS0 配置）

RS0 配置：Qwen3-30B，pkg_nj，NM_MOE=1，1 device TP4/LNC2，MNS16，buckets 和 rs_srv.sh 相同，8 CPU / 160 Gi（与 rs pod 同规格）。同一个 pod 里先跑 W，再跑 C。结果在 `s3://…/_meta/cache-test/20261108T134029Z/`。

| | W：先 `neff_cache.sh pull rs_RS0` | C：空缓存冷启动 |
|---|---|---|
| S3 拉取 | 31 NEFF / 306 MB，**2 s** | — |
| 新编译 NEFF | **0**（31/31 命中） | 23 |
| 权重加载结束 → 开始 prefill 预热（编译段） | **10.0 min** | 21.0 min |
| prefill 预热（7 个 bucket） | 9.0 min | 9.0 min |
| decode 预热（16 个组合） | 7.5 min | 7.4 min |
| **启动到 READY 总计** | **1667 s = 27.8 min** | **2334 s = 38.9 min** |
| 推理结果 | 正确（"Paris…"） | 一致 |

- 冷编出来的 23 个 NEFF 文件名全部包含在 S3 那 31 个里（`only_cold=0`；多出的 8 个是 nj_CAP 的 NM_CAP 变体）。说明 persistent key 跨 pod、跨缓存目录名是确定的，命中可靠。
- 这个小模型只省 11 min（−29%）。剩下约 28 min 是 trace、NKI 重编和逐 bucket 预热（每个 bucket 约 60–70 s，与 NEFF 是否命中无关），S3 解决不了，属于另一类优化（例如研究 NKI 缓存能否安全携带、减少 bucket）。大模型的 neuronx-cc 时间占比更高（235B stock 约 2h20m、DS-V3 5 层首编 61 min），预期省得更多，但未实测。

### 6.3 补推（2026-11-09 01:21–01:22 UTC，`neff-push-pod.tmpl.yaml`，只推这两个 TAG 的 native_neff_cache）

推之前确认：两个服务都已 READY（`/data/lx_out/q235_lx/READY` 11-08 14:56 UTC，服务 15:04 已退出；`/data/lx_out/ds_stock/READY` 11-08 16:10 UTC，两机 warmup 完成，服务仍在、空闲），缓存目录最后写入分别是 14:56 和 16:09/16:10，远超 20 分钟。正在跑的 lx-c/d/e 没碰。

| TAG | 节点（pod） | 本地 NEFF / MB | 用时 | S3 结果 |
|---|---|---|---|---|
| `lx_q235_lx` | <NODE>（lx-c） | 1296 / 14751 | 41 s | 1296 NEFF，2592 个对象，15.46 GB |
| `lx_ds_stock` | <NODE>（lx-d，node 0） | 1408 / 38169 | 72 s | 两机并集 2816 NEFF（两边文件名不重叠），5632 个对象，80.03 GB |
| `lx_ds_stock` | <NODE>（lx-e，headless） | 1408 / 38168 | 73 s | 同上 |

每次 push 都写了 `_manifest/<node>-<utc>.txt`（`neff_cache.sh ls lx_q235_lx` 1 份，`ls lx_ds_stock` 2 份）。之后 STACK 下共 81 个 TAG，桶合计 38,772 个对象、162.7 GB。
