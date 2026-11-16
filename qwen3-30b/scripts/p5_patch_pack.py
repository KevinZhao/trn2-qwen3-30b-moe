# P5: packed multi-request prefill for vllm-neuron 0.29 (Qwen3-30B-A3B, pkg_rs3).
#
# usage: python3 p5_patch_pack.py <vllm_neuron install dir>
#
#   vllm/core/scheduler.py            must be "stock + p4_patch_sched.py" (patched in place)
#   vllm/worker/neuron_model_runner.py  backed up once to *.p5orig, always re-patched from it
#   vllm/worker/neuron_worker.py        backed up once to *.p5orig, always re-patched from it
#
# Runtime switches (all off => behaviour identical to the unpatched files):
#   P5_PACK=1                 enable packing (scheduler + runner + worker/compile targets)
#   P5_PACK_BUCKETS=512,...   packed-prefill T buckets (default 512,1024,2048,4096); K = max_num_seqs
#   knob json (P4_KNOBS):     "pack_max" (default max_num_seqs, 1 = off), "pack_budget" (default 4096)
#
# Requires the model package to understand attn_metadata keys pack_bound_min / pack_bound_max /
# pack_valid (pkg_rs3/model_bf16.py).
import glob
import os
import shutil
import sys

VN = sys.argv[1]


def _apply(s, edits, path):
    for old, new, cnt in edits:
        n = s.count(old)
        assert n == cnt, f"{path}: anchor count {n} != {cnt}: {old[:100]!r}"
        s = s.replace(old, new)
    return s


def patch_from_backup(path, edits):
    orig = path + ".p5orig"
    if not os.path.exists(orig):
        shutil.copy2(path, orig)
    s = open(orig).read()
    assert "_p5_" not in s, f"{orig} already contains P5 code (bad backup)"
    open(path, "w").write(_apply(s, edits, path))


# =====================================================================================
# scheduler.py  (stock + p4)
# =====================================================================================
SCHED_EDITS = [
    # --- init: P5 state (after the P4 init log line) ---
    ("""        logger.info("P4 interleave: decode_steps_between_prefill_chunks=%d decode_steps_between_prefills=%d knobs=%s",
                    self._decode_steps_between_chunks, self._decode_steps_between_prefills, self._p4_knob_path)
""", """        logger.info("P4 interleave: decode_steps_between_prefill_chunks=%d decode_steps_between_prefills=%d knobs=%s",
                    self._decode_steps_between_chunks, self._decode_steps_between_prefills, self._p4_knob_path)
        # P5 packed multi-request prefill (p5_patch_pack.py). Off unless P5_PACK=1.
        self._p5_on = _p4os.environ.get("P5_PACK", "0") == "1"
        self._p5_buckets = sorted({int(_x) for _x in _p4os.environ.get(
            "P5_PACK_BUCKETS", "512,1024,2048,4096").split(",") if _x.strip()})
        self._p5_pack_max = self.max_num_seqs if self._p5_on else 1
        self._p5_pack_budget = 4096
        self._p5_memo = {}
        self._p5_probe_warned = False
        if self._p5_on:
            logger.info("P5 packed prefill: pack_max=%d pack_budget=%d buckets=%s",
                        self._p5_pack_max, self._p5_pack_budget, self._p5_buckets)
""", 1),
    # --- knob file: pack_max / pack_budget ---
    ("""            self._decode_steps_between_prefills = int(d.get("dsbp", 0))
""", """            self._decode_steps_between_prefills = int(d.get("dsbp", 0))
            if self._p5_on:
                self._p5_pack_max = max(1, min(int(d.get("pack_max", self.max_num_seqs)),
                                               self.max_num_seqs))
                self._p5_pack_budget = max(1, int(d.get("pack_budget", 4096)))
                logger.info("P5 knobs: pack_max=%d pack_budget=%d", self._p5_pack_max,
                            self._p5_pack_budget)
                if self._p4_steplog is not None:
                    self._p4_steplog.write("K5 %.4f pack_max=%d pack_budget=%d\\n" % (
                        _p4time.time(), self._p5_pack_max, self._p5_pack_budget))
""", 1),
    # --- per-schedule() memo reset ---
    ("""        self._p4_refresh()
""", """        self._p4_refresh()
        self._p5_memo = {}
""", 1),
    # --- admission: allow >1 waiting prefill only when the whole group is packable ---
    ("""        # Check prefill batch capacity (only 1 prefill at a time)
        if len(self.waiting) >= self.max_prefills_per_batch:
            return False
""", """        # Check prefill batch capacity (only 1 prefill at a time)
        if len(self.waiting) >= self.max_prefills_per_batch:
            # P5: a 2nd..Kth waiter is admitted only into a packed prefill step.
            if not self._p5_can_pack(request):
                return False
""", 1),
    # --- padding: collect prefill reqs ---
    ("""        prefill_count = 0
        decode_count = 0
""", """        prefill_count = 0
        decode_count = 0
        _p5_pf: list[tuple[str, int]] = []
""", 1),
    ("""                prefill_count += 1
""", """                prefill_count += 1
                _p5_pf.append((req_id, num_tokens))
""", 1),
    # --- padding: packed step pads the TOTAL once to a packed bucket ---
    ("""        # Log batch summary
        if prefill_count > 0 and total_padding_this_batch > 0:
""", """        if self._p5_on and len(_p5_pf) > 1:
            self._p5_fix_padding(scheduler_output, _p5_pf, req_num_computed)

        # Log batch summary
        if prefill_count > 0 and total_padding_this_batch > 0:
""", 1),
    # --- helper methods (before _p4_refresh) ---
    ("""    def _p4_refresh(self) -> None:
""", """    # ---------------------------------------------------------------- P5 helpers
    def _p5_prefix_hit(self, request) -> bool:
        km = self.kv_cache_manager
        if not getattr(km, "enable_caching", False):
            return False
        if getattr(request, "skip_reading_prefix_cache", False):
            return False
        bh = getattr(request, "block_hashes", None)
        if not bh:
            return False  # shorter than one block: nothing can hit
        try:
            n = km.coordinator.find_longest_cache_hit(bh, request.num_tokens - 1)[1]  # 0.29: (blocks, hit_len, n_uncached_common)
            return n > 0
        except Exception as e:  # noqa: BLE001
            try:
                n = km.get_computed_blocks(request)[1]
                return n > 0
            except Exception as e2:  # noqa: BLE001
                if not self._p5_probe_warned:
                    logger.warning("P5 prefix-hit probe failed (%s / %s); packing disabled", e, e2)
                    self._p5_probe_warned = True
                return True

    def _p5_packable(self, request) -> bool:
        rid = request.request_id
        m = self._p5_memo.get(rid)
        if m is not None:
            return m
        ok = (
            request.status == RequestStatus.WAITING
            and request.num_computed_tokens == 0
            and getattr(request, "num_output_tokens", 0) == 0
            and request.num_prompt_tokens >= 2
            and getattr(request, "structured_output_request", None) is None
            and not getattr(request, "mm_features", None)
            and not getattr(request, "has_encoder_inputs", False)
            and getattr(request, "kv_transfer_params", None) is None
            and getattr(request, "prompt_embeds", None) is None
            and getattr(request, "lora_request", None) is None
            and getattr(request, "pooling_params", None) is None
            and not self._p5_prefix_hit(request)
        )
        self._p5_memo[rid] = ok
        return ok

    def _p5_can_pack(self, request) -> bool:
        if not self._p5_on or self._p5_pack_max <= 1:
            return False
        waiting = list(self.waiting)
        n = len(waiting) + 1
        if n > self._p5_pack_max:
            return False
        if len(self.running) + n > self.max_num_seqs:
            return False
        if self._kv_in_use() + n > self._max_kv_concurrent:
            return False
        cap = min(self._p5_pack_budget, self.max_num_scheduled_tokens, self._p5_buckets[-1])
        total = request.num_prompt_tokens + sum(r.num_prompt_tokens for r in waiting)
        if total > cap:
            return False
        if not self._p5_packable(request):
            return False
        heads = set()
        for r in waiting:
            if not self._p5_packable(r):
                return False
            bh = getattr(r, "block_hashes", None)
            if bh:
                heads.add(bh[0])
        # Same first block as an earlier member: the base scheduler would give it a
        # prefix hit on blocks cached (but not yet computed) in this very step.
        bh = getattr(request, "block_hashes", None)
        if bh and bh[0] in heads:
            return False
        return True

    def _p5_fix_padding(self, scheduler_output, pf, req_num_computed) -> None:
        total = 0
        for rid, n in pf:
            req = self.requests[rid]
            if req_num_computed.get(rid, req.num_computed_tokens) != 0 or n != req.num_prompt_tokens:
                raise RuntimeError(
                    f"P5: packed prefill request {rid} is not a whole uncached prompt "
                    f"(computed={req_num_computed.get(rid)}, scheduled={n}, "
                    f"prompt={req.num_prompt_tokens})")
            total += n
        if total > self._p5_buckets[-1]:
            raise RuntimeError(f"P5: packed total {total} > largest packed bucket {self._p5_buckets[-1]}")
        padded_total = get_bucket_for_count(total, self._p5_buckets)
        padded = scheduler_output.num_scheduled_tokens_padded
        for rid, n in pf:
            padded[rid] = n
        padded[pf[-1][0]] = pf[-1][1] + (padded_total - total)
        logger.debug("P5 packed prefill: %d reqs, %d tokens -> bucket %d", len(pf), total, padded_total)

    def _p4_refresh(self) -> None:
""", 1),
]

# =====================================================================================
# neuron_model_runner.py  (stock)
# =====================================================================================
RUNNER_METHODS = '''    # ------------------------------------------------------------------
    # P5: packed multi-request prefill (p5_patch_pack.py)
    # ------------------------------------------------------------------
    def _p5_init(self) -> None:
        import os as _p5os

        self._p5_on = _p5os.environ.get("P5_PACK", "0") == "1"
        self._p5_pack = False
        self._p5_kmax = int(self.max_num_reqs)
        self._p5_buckets = sorted(
            {
                int(x)
                for x in _p5os.environ.get(
                    "P5_PACK_BUCKETS", "512,1024,2048,4096"
                ).split(",")
                if x.strip()
            }
        )
        if not self._p5_on:
            return
        cfg = self.neuron_config.on_device_sampling_config
        all_greedy = (
            cfg.get("all_greedy", False)
            if isinstance(cfg, dict)
            else bool(getattr(cfg, "all_greedy", False))
        )
        assert self.on_device_sampling and all_greedy, (
            "P5_PACK requires on_device_sampling_config.all_greedy=true"
        )
        assert self.speculative_config is None, "P5_PACK: no speculative decoding"
        assert not getattr(self.neuron_config, "enable_structured_outputs", False), (
            "P5_PACK: structured outputs must be disabled"
        )
        assert self.vllm_config.parallel_config.data_parallel_size == 1, "P5_PACK: no DP"
        assert getattr(self, "lm_head_dp_size", 1) == 1, "P5_PACK: no lm_head DP"
        assert not self.supports_mm_inputs, "P5_PACK: no multimodal"
        assert not self.enable_prompt_embeds, "P5_PACK: no prompt_embeds"
        assert not self.uses_mrope, "P5_PACK: no mrope"
        assert self.cp_world_size == 1, "P5_PACK: no DCP"
        assert not getattr(self.neuron_config, "max_jump_forward_tokens", 0), (
            "P5_PACK: no jump-forward decoding"
        )
        assert self._p5_buckets and all(
            b % 128 == 0 and b % self._p5_kmax == 0 for b in self._p5_buckets
        ), f"P5_PACK_BUCKETS must be multiples of 128 and of K={self._p5_kmax}"
        logger.info(
            "P5 packed prefill enabled: K=%d buckets=%s", self._p5_kmax, self._p5_buckets
        )

    def _p5_prepare_pack(self, scheduler_output: Any, num_reqs: int) -> bool:
        """Detect a packed prefill step and move all padding to the last row."""
        if not self._p5_on or num_reqs < 2:
            return False
        req_ids = list(self.input_batch.req_ids)[:num_reqs]
        n_prefill = 0
        for i, rid in enumerate(req_ids):
            st = self.requests.get(rid)
            if (
                st is not None
                and self.input_batch.num_computed_tokens_cpu[i] < st.num_prompt_tokens
            ):
                n_prefill += 1
        if n_prefill < 2:
            return False
        if n_prefill != num_reqs:
            raise RuntimeError(
                f"P5: mixed batch ({n_prefill} prefill of {num_reqs} requests)"
            )
        padded = scheduler_output.num_scheduled_tokens_padded
        actual = scheduler_output.num_scheduled_tokens
        tot_act = 0
        tot_pad = 0
        for i, rid in enumerate(req_ids):
            st = self.requests[rid]
            computed = int(self.input_batch.num_computed_tokens_cpu[i])
            if computed != 0 or actual[rid] != st.num_prompt_tokens:
                raise RuntimeError(
                    f"P5: packed request {rid} is not a whole uncached prompt "
                    f"(computed={computed}, scheduled={actual[rid]}, "
                    f"prompt={st.num_prompt_tokens})"
                )
            tot_act += actual[rid]
            tot_pad += padded[rid]
        if tot_pad not in self._p5_buckets or tot_act > tot_pad:
            raise RuntimeError(
                f"P5: packed step total {tot_act} padded to {tot_pad}, "
                f"not one of the compiled packed buckets {self._p5_buckets}"
            )
        for rid in req_ids:
            padded[rid] = actual[rid]
        padded[req_ids[-1]] = actual[req_ids[-1]] + (tot_pad - tot_act)
        return True

    def _p5_finish_pack(
        self, attn_metadata, padded_np, actual_np, total_tokens: int, device
    ) -> None:
        """Route a packed step to the full-prefill branch and attach packing tensors."""
        padded_np = np.asarray(padded_np, dtype=np.int64)
        actual_np = np.asarray(actual_np, dtype=np.int64)
        assert int(padded_np.sum()) == int(total_tokens), (padded_np, total_tokens)
        starts = np.cumsum(padded_np) - padded_np
        ends = starts + padded_np
        bmin = np.repeat(starts, padded_np).astype(np.int32)
        bmax = np.repeat(ends, padded_np).astype(np.int32)
        valid = np.zeros(int(total_tokens), dtype=np.bool_)
        for s0, a in zip(starts.tolist(), actual_np.tolist()):
            valid[s0 : s0 + a] = True
        bmin_t = torch.from_numpy(bmin).to(device)
        bmax_t = torch.from_numpy(bmax).to(device)
        valid_t = torch.from_numpy(valid).to(device)
        seen = set()
        for meta in attn_metadata.values():
            if id(meta) in seen:
                continue
            seen.add(id(meta))
            meta["kv_segment_size"] = 0
            meta["max_query_len"] = int(total_tokens)
            meta["pack_bound_min"] = bmin_t
            meta["pack_bound_max"] = bmax_t
            meta["pack_valid"] = valid_t

    def _p5_pack_synthetic_inputs(self, bucket_size: int, device) -> dict:
        """Warmup/trace inputs for the packed prefill graph (K = max_num_seqs rows)."""
        k = self._p5_kmax
        assert bucket_size % k == 0 and bucket_size // k >= 2, (bucket_size, k)
        seg = bucket_size // k
        lens = np.full(k, seg, dtype=np.int64)
        input_ids = torch.arange(1, bucket_size + 1, dtype=torch.int32).to(device)
        positions = torch.from_numpy(
            np.tile(np.arange(seg, dtype=np.int64), k)
        ).to(device)
        logits_indices = torch.arange(
            seg - 1, bucket_size, seg, dtype=torch.long
        ).to(device)
        attn_metadata = self._build_warmup_attention_metadata(
            num_tokens=bucket_size,
            num_reqs=k,
            cached_seq_len=0,
            decode_token_threshold=1,
            device=device,
            kv_segment_size=0,
        )
        self._p5_finish_pack(attn_metadata, lens, lens, bucket_size, device)
        sampling_params = torch.tensor(
            [[0, 1.0, 1.0]] * k, dtype=torch.float32, device=device
        )
        return dict(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=logits_indices,
            sampling_params=sampling_params,
            spec_decode_metadata=None,
            rank=self.rank_tensor.to(device),
            logit_mask=self._build_noop_logit_mask(
                k, device, compiled_graph_input=True
            ),
        )

    def _build_prefill_synthetic_inputs(
'''

RUNNER_EDITS = [
    # --- init: config + assertions ---
    ("""        # block_size is logged in initialize_kv_cache (post-override). At this
""", """        self._p5_init()

        # block_size is logged in initialize_kv_cache (post-override). At this
""", 1),
    # --- detect packed step, move padding to last row (before any use of padded counts) ---
    ("""        pre_strip_max_query_len = max(
""", """        self._p5_pack = self._p5_prepare_pack(scheduler_output, num_reqs)
        pre_strip_max_query_len = max(
""", 1),
    # --- packed step uses K = max_num_seqs rows for block_table / sampling ---
    ("""        self._save_slot_mapping_to_device(total_num_scheduled_tokens_padded)
""", """        self._save_slot_mapping_to_device(total_num_scheduled_tokens_padded)
        if self._p5_pack:
            padded_num_reqs = self._p5_kmax
""", 1),
    # --- packed step: full-prefill branch + bound/valid tensors, max_query_len = T ---
    ("""            cached_seq_len,
            max_decode_ctx_len=max_decode_ctx_len,
        )

        # Spec decoding
""", """            cached_seq_len,
            max_decode_ctx_len=max_decode_ctx_len,
        )
        if self._p5_pack:
            self._p5_finish_pack(
                attn_metadata,
                num_scheduled_tokens_padded,
                actual_num_tokens,
                total_num_scheduled_tokens_padded,
                self.device,
            )

        # Spec decoding
""", 1),
    # --- packed step: sampling_positions padded to K ---
    ("""        # Multimodal path: gather encoder embeddings from on-device encoder cache.
""", """        if self._p5_pack and logits_indices.shape[0] < self._p5_kmax:
            logits_indices = torch.cat(
                [
                    logits_indices,
                    logits_indices[-1:].repeat(
                        self._p5_kmax - logits_indices.shape[0]
                    ),
                ]
            )

        # Multimodal path: gather encoder embeddings from on-device encoder cache.
""", 1),
    # --- helper methods ---
    ("""    def _build_prefill_synthetic_inputs(
""", RUNNER_METHODS, 1),
    # --- kv_segment_size == -1 selects the packed warmup/trace variant ---
    ("""        if device is None:
            device = self.device
        cached_seq_len = 0
        num_reqs = 1
""", """        if device is None:
            device = self.device
        if kv_segment_size == -1:
            return self._p5_pack_synthetic_inputs(bucket_size, device)
        cached_seq_len = 0
        num_reqs = 1
""", 1),
]

# =====================================================================================
# neuron_worker.py  (stock)
# =====================================================================================
_P5_BKTS = """sorted({int(_x) for _x in _p5os.environ.get("P5_PACK_BUCKETS", "512,1024,2048,4096").split(",") if _x.strip()})"""
WORKER_EDITS = [
    ("""                targets[(bucket_size, kv_seg_size_for_bucket)] = None
        return list(targets)
""", """                targets[(bucket_size, kv_seg_size_for_bucket)] = None
        import os as _p5os

        if _p5os.environ.get("P5_PACK", "0") == "1":
            # P5 packed prefill graphs: kv_segment_size sentinel -1 (see runner).
            for _b in """ + _P5_BKTS + """:
                targets[(_b, -1)] = None
        return list(targets)
""", 1),
    ("""                    ) from e

    def _decode_compile_targets(self)""", """                    ) from e

        import os as _p5os

        if _p5os.environ.get("P5_PACK", "0") == "1":
            for _b in """ + _P5_BKTS + """:
                logger.info("[P5] Warming up packed prefill: bucket_size=%s", _b)
                try:
                    self.model_runner.warmup_prefill(_b, -1)
                except Exception as e:
                    raise RuntimeError(
                        f"P5 packed prefill warmup failed for bucket {_b}: {e}"
                    ) from e

    def _decode_compile_targets(self)""", 1),
]


def main():
    sched = os.path.join(VN, "vllm/core/scheduler.py")
    runner = os.path.join(VN, "vllm/worker/neuron_model_runner.py")
    worker = os.path.join(VN, "vllm/worker/neuron_worker.py")

    s = open(sched).read()
    assert "_p4_refresh" in s, "scheduler.py must be patched by p4_patch_sched.py first"
    assert "_p5_" not in s, "scheduler.py already P5-patched (restore .njorig + p4 first)"
    open(sched, "w").write(_apply(s, SCHED_EDITS, sched))

    patch_from_backup(runner, RUNNER_EDITS)
    patch_from_backup(worker, WORKER_EDITS)

    for d in ("vllm/core/__pycache__", "vllm/worker/__pycache__"):
        for f in glob.glob(os.path.join(VN, d, "*.pyc")):
            b = os.path.basename(f)
            if b.startswith(("scheduler.", "neuron_model_runner.", "neuron_worker.")):
                os.remove(f)
    print("p5 patched:", sched, runner, worker)


if __name__ == "__main__":
    main()
