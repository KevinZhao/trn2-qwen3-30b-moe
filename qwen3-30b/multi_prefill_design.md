# 多请求打包 prefill（multi-prefill packing）设计评估

对象：Qwen3-30B-A3B（48 层，128 专家 top-8，TP4），vllm-neuron 0.29 native torch 路径，生产包 `pkg_rs2`，调度器已打 P4 补丁。
代码基准：`ref/vn_src/vllm_neuron`（以下简称 `vn/`，其中的 `scheduler.py` 已经是 P4 补丁之后的版本）、`pkg_rs2/model_bf16.py`。
结论来自静态阅读代码和已有实测（`results/RS0/summary_blog.txt`），**没有上机验证**。凡是标"估计"的数字都需要实测。

---

## 0. 结论

**可行，而且性价比高。** 关键依据有四条：

1. runner 的输入构建是从 GPU vLLM 继承来的，本来就是多请求通用写法：`req_indices`、`cu_num_tokens`、逐请求的 `padding_map`、逐 token 的 slot_mapping、逐请求的 `logits_indices`。真正写死"单条"的只有 4 处：调度器准入、调度器按请求各自补 bucket、prefill 图的输入形状（K 维）、attention。
2. attention 不用自己写 kernel。nkilib 的 `attention_cte`（也就是 `NF.flash_attention`）**原生支持 sequence packing**：用 `bound_min/bound_max` 给出每个 query 能看到的 KV 区间，并且可以和 `causal_mask=True` 同时用。vllm_neuron 的 `flash_attention` 包装已经把这两个参数透传给 kernel。qwen3_vl 的视觉编码器在生产里就是这样用的。
3. on-device sampling（all_greedy）本来就按行处理：`index_select(sampling_positions)` 得到 `lm_head` 的 `[K, V/TP]`，再做分布式 argmax 得到 `[K]`。一次输出 K 行没有障碍。
4. 实测 c1 TTFT（RS0）：bucket 128 约 64 ms，256 约 67 ms，1024 约 95 ms，4096（3000 token）约 218 ms，8192（6000 token）约 420 ms。近似为 **每个 step 固定约 58–60 ms，另加约 0.04 ms/token**。把 16 条约 262 token 的 ShareGPT prompt 合进 1–3 个 step，可以省掉 13–15 次固定开销（每次约 60 ms）。估计 c16 TTFT P99 从约 1.3 s 降到 **约 0.3–0.45 s**（3–4 倍）。

**最大的坑**：MoE prefill 的 padding mask 用 `argmax(positions)` 推断"最后一个真实 token"（`model_bf16.py:1309-1312`）。打包以后这会**静默地**把除第一条最长序列之外的 token 判成 padding，MoE 输出被置零，不会报错，只会出错答案。必须改成显式的 mask。

---

## 1. 需要改的地方（文件 / 函数 / 行号）

只打包同时满足以下条件的请求：新请求（`status==WAITING`、`num_output_tokens==0`、`num_computed_tokens==0`）、整条 prompt 能放进打包预算、前缀缓存命中为 0、不是 SO/多模态/prompt_embeds/KV-transfer。其余请求仍走原来的单条路径，原有 NEFF 不受影响。

### 1.1 调度器 `vn/vllm/core/scheduler.py`

| 位置 | 现状 | 改动 |
|---|---|---|
| L285 `self.max_prefills_per_batch: int = 1` | 硬编码 1。L1245（3.2 分支里给 SO 的 override）和 L1298（3.5 SO 准入）也读它 | **不要改它**（保留 SO 路径语义）。新增 `self._pack_max`（默认 16 = max_num_seqs）和 `self._pack_budget`（默认 2048 或 4096 token）。在 `_p4_refresh`（L1396-1426）里加 `pack_max`/`pack_budget` 两个 knob：`pack_max=1` 即关闭，不用重启就能回滚。另外 `pack_budget` 要满足 ≤ `self.max_num_scheduled_tokens`（P4 的 `budget` knob 会改这个值） |
| L599-610 `at_capacity` | 只看 `len(self.running)` | 打包准入时要按 `len(running)+len(waiting) < max_num_seqs` 计数 |
| L677-681 `_kv_in_use` | running + send-lane | 打包准入时再加上 `len(self.waiting)`，否则一次放进 K 条可能超过 `_max_kv_concurrent` |
| L820-876 `can_schedule`，L848 `if len(self.waiting) >= self.max_prefills_per_batch: return False` | 一次只放 1 条 | 改为：`waiting` 为空时照旧；否则只有在"waiting 里已有的请求和当前候选都可打包、累计 prompt 长度 ≤ pack_budget、`len(waiting) < pack_max`"时才放行，遇到第一条不满足的就停（保持 FCFS）。前缀命中探测用 `self.kv_cache_manager.coordinator.find_longest_cache_hit(request.block_hashes, ...)`。也可以用 `get_computed_blocks`，但它会重复计入 prefix-cache 统计。探测只对 ≤16 条候选做，开销可以忽略 |
| L1131-1139 Step 1/2（holdback→waiting 循环） | 逐条调用 `can_schedule` | 不用改，自动受益 |
| L1147-1152 `can_start_prefill`，L1233-1257 分支 3.2 | 隐藏所有 decode，交给 base scheduler | 不用改。Step 4（L1323-1342）已经按隐藏数扣减 `max_num_running_reqs`，K 条都能放进去。由于准入保证累计长度 ≤ pack_budget ≤ token_budget，base scheduler 会把每条都**完整**调度，不会出现半条 chunk |
| L924-1099 `_apply_padding_and_log_stats`，L1002-1050 | **每条 prefill 各自补到 bucket**（例如 3×100 token → 128+128+128=384，不是已编译的 bucket，会触发运行时编译或报错） | prefill 条数 >1 时：`padded_total = bucket(sum(actual))`，前 K-1 条 `padded = actual`，**全部 padding 挂在最后一条**（建议由 runner 按自己的行序决定"最后一条"，见 1.2）。L1081 的 mixed 检查保持不变 |
| L1182-1192（P4 3.0b）、L1165-1175（3.0 dsbpc） | 两次 prefill 之间插 dsbp 个 decode step；长 prompt 的 chunk 之间插 dsbpc 个 | 语义不变：一个打包 step 算"一次 prefill"，之后照样插 dsbp 个 decode。dsbpc 只影响仍走单条路径的长 prompt |
| L1374-1380 P4 steplog | `P/D nt nr ...` | 建议多记一个 K，便于看打包效果 |

### 1.2 Runner `vn/vllm/worker/neuron_model_runner.py`（生产用 V1 runner）

先澄清 **L3458 的 `assert len(req_ids) == 1`**：它在 `_gather_mm_embeddings` 里，只在多模态且 encoder cache 存在时调用（L4351-4366），**文本路径不经过**，不用管。

| 位置 | 现状 | 改动 |
|---|---|---|
| L3926-3931 `padded_num_reqs = get_decode_padded_batch_size(...)`（`utils/bucket_utils.py:649-690`：prefill 时直接返回 num_reqs） | prefill 时 = num_reqs | 打包 step（prefill 条数 >1）令 `padded_num_reqs = PACK_KMAX`（固定 16，原因见 §3）。这样 block_table 行数、`sampling_params` 行数（L8934-8948 本来就按 block_table 行数补齐）都自动对齐 |
| L4069-4075、L4093-4124 input_ids/positions | 逐请求通用 | 不用改。positions = `num_computed + arange`，每条序列从 0 开始，RoPE 自然正确 |
| L4138-4147 `_create_padded_inputs`（L4640-4827） | 逐请求在各自末尾补 padding，padding 位置 = 该请求最后一个位置 | 可以不改。建议在打包 step 里把 padding 统一挂到 `input_batch` 行序的最后一条（runner 侧重写 `num_scheduled_tokens_padded`，只要保证总和 = bucket），这样 padding 全在尾部，MoE mask 和 bound 都更简单 |
| L4155-4209 slot_mapping | 逐 token；padding 指向 NULL_BLOCK | **已支持，不用改** |
| L4223-4232 decode slot padding / 拷贝到设备 | prefill 直接跳过 | 不用改 |
| L4235-4246 `_compute_cached_seq_len` + `_build_attention_metadata` | cached_seq_len 取 max（打包时 =0） | 传 `max_query_len = total_padded`（=T），**不要传每条最大长度**。`max_query_len` 是 attn_metadata 里的 Python int，dynamo 会对它做值 guard，必须和 warmup 一致，否则每个新的值都会重新编译 |
| L4248-4289 logits_indices | 逐请求取最后一个真实 token | 不用改（K 个元素） |
| L4325-4338 `_pad_to_compiled_shapes`（L5108-5272，prefill 在 L5163-5170 直接返回） | prefill 不补 | 打包 step：`logits_indices` 用最后一个真实下标补到 `PACK_KMAX`（复用 L5228-5238 的写法） |
| L5497-5701 `_build_attention_metadata`：L5533 `get_device_tensor(padded_num_reqs)`；L5586-5595 `kv_segment_size = pick_kv_segment_size(...)` | 生产走 segmented（kv_segment_size_buckets=[8192]） | 打包 step：强制 `kv_segment_size=0`（进 full 分支）；增加 `pack_bound_min` / `pack_bound_max`（`[T]` int32，设备上）和 `pack_valid`（`[T]` bool）三个 key。`bound_min[t]` = t 所属序列在打包缓冲里的起点；`bound_max[t]` = 该序列终点（padding 也算作最后一条序列的一部分），也可以直接取 T（上界由 causal 保证） |
| L7420-7450 logit_mask（仅在 `enable_structured_outputs` 时存在；prefill 的 pad 目标是 `input_ids.shape[0]`，L5108 以下的 `_pad_logit_mask_for_decode` 遇到 >max_num_reqs 就不补） | 生产没开 SO，logit_mask 为 None | MVP 要求 SO 关闭（init 时 assert）。如果以后要开，把打包 step 的 pad 目标改成 `PACK_KMAX` |
| L5703-5880 `_build_warmup_attention_metadata`，L5882-6009 `_build_prefill_synthetic_inputs`（num_reqs=1，logits_indices `[1]`，sampling_params `[1,3]`） | 只有单条 | 加一个打包变体（`kv_segment_size=-1` 作为哨兵）：num_reqs=PACK_KMAX，`max_query_len=T`，logits_indices `[16]`，sampling_params `[16,3]`，bound/valid 张量（构造在 `device` 参数指定的设备上，parallel trace 时是 meta） |
| L6011 `extract_prefill_graphs`，L6103 `warmup_prefill` | 按 (bucket, kv_seg) | 透传哨兵值即可 |
| L9093-9104 NEFF 指标名 | `prefill_s{T}[_kv]` | 打包时加 `_pk` 后缀 |
| L10293-10350 `_sample`、L10054-10183 `_generate_model_runner_output` | `[[x] for x in out.tolist()]`，按行对应 `input_batch.req_ids` | **不用改**。多出的 16-K 行和 decode padding 一样，由 `req_id_to_index` 忽略；`_get_partial_prefill_req_ids` 按请求判断，整条 prompt 不会被丢弃 |

`vn/vllm/worker/neuron_worker.py`：
- L2133-2165 `_prefill_compile_targets`：对 `pack_buckets` 里的每个 T 追加 `(T, -1)`。
- L2420-2450 `_warmup_prefill`：**这里自己写了一套 bucket×kv 双循环，没有调用 `_prefill_compile_targets`**，要单独加打包循环。
- L1734 `_prefill_buckets`：不用改。

### 1.3 模型 `pkg_rs2/model_bf16.py`

| 位置 | 改动 |
|---|---|
| L501-703 `forward_prefill`：L547 `if kv_segment_size:` 是 segmented 分支；L640-690 是 full 分支（`NF.qkv_proj` → `_write_paged_kv_cache`（index_put，已兼容 FP8/packed-K）→ `NF.flash_attention(causal)`） | 在 full 分支里，如果 attn_metadata 带有 `pack_bound_min`，就给 `flash_attention` 加上 `bound_min=bmin.view(1,T,1).expand(Nh,T,1).contiguous()` 和同样处理的 `bound_max`。kernel 按 `[B, S, 1]` 连续布局、`batch_id*S` 的偏移去 DMA，所以必须 contiguous。约 15 行 |
| **L1305-1312 MoE `padding_mask = arange <= argmax(positions)`** | **必改**。打包后 positions 是锯齿形，argmax 只会指向第一条最长序列的末尾，后面所有序列都被当成 padding（nm_glue L69-70/L125/L139 会把 affinity 乘 0）。改成：attn_metadata 里有 `pack_valid` 时就用它（通过 `Qwen3MoeDecoderLayer._forward_prefill` 的 L1625 `self.mlp(...)` 透传一个 `padding_mask=` 参数，或者在 `Qwen3MoeModel.forward` 里算一次后挂在模块上）。nm 的 mask 是逐元素的，不要求是前缀。约 10–15 行 |
| L1944-1948 SP 检查 `T % world_size` | bucket 都是 128 的倍数，没问题 |
| L1967-1969 `index_select(hidden, sampling_positions)`，L2001 `B_local = sampling_positions.shape[0]`，L2037 `self.sampler(...)` | 不用改，自然输出 `[16]` |
| RMSNorm / router / `nm_prefill_layer`（L1333）/ all-gather / reduce-scatter | 都按 token 处理，与序列边界无关，不用改 |

### 1.4 attention 能不能做块对角 causal

- **segmented kernel 不行**。`functional/attention/attention_segmented_cte.py` 的 torch 实现在 L367 用 `bt = block_tables[0]`，`prior_tokens` 是全局标量（L362）。NKI 版接口（L904-1087）也只有单个 `prior_tokens`、没有逐 query 的 bound。它从 cache 读 KV，K 条序列的块分散在不同的 block 上，没法表达。
- **flash_attention（attention_cte）可以**。`functional/attention/attention_cte.py` L189-347 的签名里有 `bound_min/bound_max`，并原样传给 nkilib kernel。nkilib 源码（本地副本 `/private/tmp/trn2-hyd/p3/ref/nkiattn/attention/attention_cte.py`）L62-67 写明："Sequence Packing … Compatible with causal masking (both masks are applied simultaneously); Not compatible with prefix caching"。L660-683 是形状和互斥断言（`(batch, seqlen_q, 1)`），L1668-1669 打开 `dynamic_sel_mask`，L1922-1940 从 HBM 读 bound 并做 range_select。生产先例：`vllm-neuron-main/.../qwen3_vl/vision_encoder_bf16.py` L328-344（那里用的是 `causal_mask=False`）。
- **代价（估计）**：packing 模式下静态 causal 仍会跳过上三角 tile，但跨序列的下三角 tile 由于 bound 是运行时值，**照样会计算**，只是用动态 range_select 屏蔽。所以计算量约等于对整个 T 做 causal：每 rank 8 个 Q 头，每层 2·T²·D·H，T=4096 时约 34 GFLOP/层，48 层约 1.65 TFLOP。按 LNC2 每 rank 167 TF、50% 利用率算约 20 ms，T=2048 约 5 ms。现有单条 4096 bucket 的 segmented 路径（prior=0）本来也是对 4096 做全 causal，所以**和实测的 218 ms 同一量级，额外开销主要是动态 mask，估计 0–10 ms**。
- **torch 按 query 分块的 masked attention（兜底方案）**：可以接受，但明显更慢。按 512 一块切 query，第 c 块只看 KV[0:(c+1)·512]（切片是静态的），每层 scores 约 8·T²/2 个 fp32 元素，读写 3–4 次。T=4096 约 0.8 GB/层，48 层约 38 GB/rank，按约 0.7 TB/s 是 55 ms，再加上非融合 matmul，估计 **4096 时每个打包 step 多 +60–90 ms，2048 时多 +15–25 ms**。另外这些大的中间张量可能挤占 NEFF 里 MoE kernel 的 SBUF（RESULTS.md 里遇到过 NCC_INLA001），这是额外风险。所以只作为 kernel 编译失败时的退路。

### 1.5 KV 写入与输出

- KV：slot_mapping 本来就是逐 token 的。full 分支的 `_write_paged_kv_cache`（L453-497）按 slot scatter，padding 写到 NULL block，**已支持**。生产开着 `P3_KVIX=1`，segmented 分支用的也是这个函数，所以写入代价不变。
- 输出：sampler 产出 `[16]`，`_sample` 逐行转成 list，`ModelRunnerOutput(req_ids=input_batch.req_ids)` 按行对应，**已支持**。async scheduling 下 `prev_req_ids_ordered` 按 req_id 对应，16 ≥ K，L3081 的断言成立。

---

## 2. 有没有现成的多序列 prefill 路径可以复用

| 候选 | 结论 |
|---|---|
| 调大 `max_prefills_per_batch` | **不能单独调**。会同时撞上 5 个问题：(a) 每条请求各自补 bucket，总长不是已编译形状；(b) `padded_num_reqs=K`，block_table 和 sampling_positions 是没编译过的 `[K]` 形状；(c) segmented attention 只用 `block_tables[0]`，相当于跨序列做 attention，数值错误；(d) MoE `argmax(positions)` 静默置零；(e) 超出 token budget 的最后一条会被 chunk。另外 L1245/L1298 的 SO 逻辑依赖它等于 1 |
| V1 runner 输入构建 | **可以复用**，本来就是多请求写法（见 1.2） |
| V2 runner（`vn/vllm/worker/neuron_model_runner_v2.py`） | 也是单条：L2628 写着 "prefill (R=1)"，L2662 写着 "single active prefill row"。没有复用价值 |
| DI / prefill-async pipeline（scheduler L182-206、`_release_dispatched_producers`） | 仍然是 "batch-1 compute slot"，只是把 NIXL 发送和下一条 prefill 重叠，不是多条 prefill |
| 投机解码 drafter | `extract_prefill_graphs` 里 `drafter.graph_extract(num_reqs=1)`，也是单条 |
| EPD / 多模态 | `_gather_mm_embeddings` 断言单条（L3458） |
| **nkilib attention_cte 的 sequence packing** | **可以复用**，这是唯一现成的"多序列"能力，qwen3_vl 视觉编码器在用 |
| `num_seqs` 维度的 prefill 图 | 不存在。`num_seqs_buckets` 只用于 decode |

---

## 3. NEFF 数量和 bucket 划分

- 现状：RS0 一共 31 个 NEFF（`neffs_before 31`），其中 prefill 是 7 个 T bucket × 1 个 kv_seg = 7。
- 打包图相对现有 prefill 图，在形状上只多了 `sampling_positions`/`sampling_params`/block_table 的行数（K 维）和 3 个 `[T]` 张量。
- **建议 K 维固定成 PACK_KMAX = max_num_seqs = 16，不分 bucket**。原因：每 rank 读一次 lm_head 权重（151936×2048×2B/4 ≈ 156 MB，约 0.2 ms）的代价和行数无关，多算 16 行可以忽略。这样**每个打包 T bucket 只多 1 个 NEFF**。
- 打包 T bucket 建议 `{256, 512, 1024, 2048, 4096}`。128 一般只能装 1–2 条，收益小，可以不加。8192 那一步要约 420 ms，会让所有人一起等，也会把 ITL 拉得很长，不建议加。所以 **+5 个 NEFF（31→36，prefill 7→12）**，编译时间按现有同 bucket 的 prefill NEFF 再来一遍估算，可以并入 parallel trace pool。
- 如果 K 维也按 `{4,8,16}` 分 bucket，是 +15 个 NEFF，没有收益，不推荐。
- K=1 继续走原来的 segmented NEFF，不变。

---

## 4. 最小可行改造（MVP）

实现方式沿用 `scripts/p4_patch_sched.py` 的做法：写一个 `p5_patch_pack.py`，对 site-packages 里的 `scheduler.py`、`neuron_model_runner.py`、`neuron_worker.py` 做"带计数断言的字符串替换"。模型改动直接写进 `pkg_rs2`。用环境变量 `P5_PACK=1` 加上 knob 文件里的 `pack_max`/`pack_budget` 控制开关。

| 步骤 | 内容 | 改动量 | 人天 |
|---|---|---|---|
| S0 | **先单独验证 attention 和 full 分支**：(a) 单层 microbench，在设备上对比 `flash_attention(causal, bound_min/max)` 打包结果和逐条 `flash_attention` 的结果（rel-fro、max abs）；(b) 把单条请求强制走 full 分支（kv_segment_size=0），和 segmented 结果对比 logits。生产从来没跑过 full 分支（所有 bucket 都是 128 的倍数），而它和 segmented 分支的 `qkv_proj` 参数写法不一样（例如 q_gamma 有没有 unsqueeze），要先排除这个分支本身的问题 | ~100 行测试 | 1 |
| S1 | 模型：full 分支加 bound；MoE `padding_mask` 改成显式的 `pack_valid` | ~30 行 | 0.5 |
| S2 | runner：识别打包 step、`padded_num_reqs=16`、`max_query_len=T`、`kv_segment_size=0`、生成 bound/valid、补 `logits_indices`、warmup 和编译目标的哨兵（worker 里两处）、指标名；init 时断言 all_greedy、无 spec/DP/SO/DCP/mm | ~150 行 | 1.5 |
| S3 | 调度器：可打包判定（含前缀命中探测）、容量计数、`_apply_padding` 只补一次、knob、steplog | ~100 行 | 1 |
| S4 | 编译 +5 个 NEFF，跑数值验证（见下）和基准（sg c1/c4/c16，rn128/1024/3000 的 c4/c16，P4 knob 在 dsbp=0/1、pack_budget=2048/4096 之间切换），跑 soak（≥45 min，混合长短 prompt，开前缀缓存并故意构造重复前缀） | 测试脚本 ~100 行 | 2–3 |

**合计**：代码约 300 行，测试约 200 行；**工程量 6–7 人天**，按编译排队的日历时间约 1.5–2 周。只求原型能跑出 TTFT 数字的话（跳过 soak 和 SO/async 等边角）约 3–4 人天。

### 数值验证
1. **首 token logits**：开一个调试服务（关掉 on-device sampling，或者用 `_debug_write_logits`，runner L9640），同一组 16 条 prompt 分别 (a) 逐条单发 (b) 一次打包，比较首 token logits：top-1 一致率应为 100%，max abs 和 KL 应该和"segmented 与 full 分支"之间的差异同量级（S0 测出这条基线）。
2. **贪心续写**：同一组 prompt，每条贪心 128 token，比较打包前后输出完全一致的比例。基线是"同一路径跑两次"（应为 100% 确定）和"segmented 与 full 分支"之间的差异。这一项同时验证了打包 step 写进去的 KV 被后续 decode 读取是否正确。
3. **专门的边界用例**：长度完全相同的多条（专门打 argmax 那个 bug）、长度为 1–2 的极短 prompt、恰好等于 bucket 的总长、K=16 满载、和一条前缀命中的请求混在一起（应该回退到单条）。
4. **端到端准确率**：GSM8K 500 题，打包和不打包对比，差异应在 ±0.5pt 以内。

### 主要风险

| 风险 | 说明 / 缓解 |
|---|---|
| **正确性：MoE padding_mask** | 见 1.3。改错会静默出错答案，边界用例必须覆盖"等长序列" |
| 正确性：dynamo guard | `max_query_len`、`kv_segment_size` 这些 int 必须和 warmup 时完全一致，否则会在运行时触发编译（分钟级卡顿）。要盯住 steplog 和 `neffs_after new=0` |
| 编译：sequence packing 和 causal 同时开 | 开了 `dynamic_sel_mask`，视觉编码器只验证过 causal=False 的组合。放进 48 层的大图之后，可能遇到 SBUF / DMA scratch 冲突（RESULTS.md 遇到过 NCC_INLA001 和运行时 DMA abort）。退路是 1.4 的 torch 分块方案 |
| 性能：大打包 step 拉长 ITL | 一个 4096 的打包 step 约 220 ms，所有 decode 都要停这么久（现在 ITL max 约 115 ms）。用 `pack_budget` knob 权衡；整个突发里 prefill 总时长下降（16×65 ms → 2–3 个 step），TPOT 均值反而可能更好 |
| 与 P4 的交互 | 打包 step 算一次 prefill，之后照样插 dsbp 个 decode。dsbpc 只影响仍走单条路径的长 prompt。P4 的 `budget` knob 调小时，`pack_budget` 必须跟着取 min，否则最后一条会被 chunk（准入时就要按 min 判断） |
| 前缀缓存 | 打包路径只放命中为 0 的请求。同一个打包 step 里如果有两条前缀相同，第二条不会命中（第一条的 block 还没算完），两条各自计算，和 GPU vLLM 的行为一致。打包 step 写进去的 block 照常进入前缀缓存（hash 在调度器侧算，和 runner 无关） |
| async scheduling | 打包 step 输出 16 行 future，之后的 decode step 按 req_id 对应。soak 时重点看紧跟在打包 step 后面的那个 decode |

### 预期收益（估计，按 RS0 的 c1 bucket 耗时线性拟合：t ≈ 60 ms + 0.04 ms × bucket）
ShareGPT c16 突发：16 条，平均约 262 token，合计约 4200 token。现在实测 P99 是 1288–1383 ms，中位数 225–246 ms。
- pack_budget=4096：两个 step，4096（约 218 ms）+ 一个小 bucket（约 65 ms），中间夹一个 decode（约 20 ms）→ **P99 约 0.3 s**，中位数约 0.22 s。
- pack_budget=2048：三个 2048 step（每个约 136 ms）→ P99 约 0.45 s，**中位数约 0.14 s**。
- 离 H100 的 146 ms 还有差距，剩下的主要是每个 step 约 60 ms 的固定开销（见 §5）。

---

## 5. 替代或互补方案（不能替代打包，但可以叠加）

1. **dsbp=0（零工程量）**：现在两次 prefill 之间都插一个约 20 ms 的 decode step。16 条串行时，最后一条要多等约 15×20 ≈ 300 ms。突发时把 dsbp 设为 0，估计 P99 从 1.3 s 降到约 1.0 s，代价是 decode 停顿更长。可以用 knob 文件按负载切换。依据：P4 的 3.0b 分支（scheduler L1176-1192）。
2. **降低单次 prefill 的固定开销**：128 token 一次 prefill 约 64 ms，而权重带宽下限只有约 20 ms（每 rank 14.5 GB / 约 0.72 TB/s）。小 T 下的 MoE 是主要部分：nm_k9 在 T=512 时 0.94 ms/层 × 48 ≈ 45 ms，每层权重下限约 0.42 ms（RESULTS.md L163）。如果把小 T 的 MoE kernel 做到接近带宽下限，每次 prefill 能省约 20 ms，串行 16 条的 P99 能省约 300 ms。FP8 专家权重（pkg_rs2 里 `RS_FP8` / nm_k9f 已在做）能让带宽下限再减半。这两项和打包是叠加关系：打包减少固定开销的**次数**，这两项减少每次的**大小**。
3. **小 block 的 prefill MoE**：本质和第 2 条相同。T 很小时 128-token block 的填充率低，但瓶颈在权重流，换 block 大小的收益有限。
4. 单靠 1–3，16 条串行的下限也在 16×(约 40 ms) ≈ 0.65 s 左右，比打包（约 0.3 s）差。所以**推荐的顺序是：先切 dsbp knob（立即生效），再做打包 MVP，然后继续优化小 T 的 MoE 和 FP8**。

---

## 附：引用位置
- 调度器：`ref/vn_src/vllm_neuron/vllm/core/scheduler.py` L285、L599-623、L677-681、L820-876、L924-1099、L1101-1394、L1396-1426
- runner：`ref/vn_src/vllm_neuron/vllm/worker/neuron_model_runner.py` L2996-3014、L3422-3463、L3872-4397、L4640-4827、L5108-5272、L5497-5701、L5703-6009、L6011、L6103、L7420-7450、L8925-8967、L9093-9104、L10054-10183、L10293-10350
- worker：`ref/vn_src/vllm_neuron/vllm/worker/neuron_worker.py` L1734、L2133-2165、L2189-2256、L2420-2450
- bucket：`ref/vn_src/vllm_neuron/utils/bucket_utils.py` L649-690
- attention：`ref/vn_src/vllm_neuron/functional/attention/attention_cte.py` L189-347；`attention_segmented_cte.py` L246-301、L362-367、L904-1087；nkilib `/private/tmp/trn2-hyd/p3/ref/nkiattn/attention/attention_cte.py` L62-67、L594-683、L1655-1669、L1907-1940
- 打包先例：`/private/tmp/nxdi_research/vllm-neuron-main/vllm_neuron/model/qwen3_vl/vision_encoder_bf16.py` L328-344
- 模型：`pkg_rs2/model_bf16.py` L413-451、L453-497、L501-703、L1277-1333、L1598-1625、L1918-2069；`pkg_rs2/nm_glue.py` L60-70、L117-158
- 实测：`results/RS0/summary_blog.txt`（c1 各 bucket 的 TTFT，sg_c16 的 P99）
- 服务配置：`scripts/rs_srv.sh` L19-26；P4 补丁：`scripts/p4_patch_sched.py`
