# Patch vllm_neuron scheduler: PR#68 "decode steps between prefill chunks" port + P4 extension
# "decode steps between consecutive prefills" + runtime knob file + step log.
import sys, re
p = sys.argv[1]; s = open(p).read()
assert "_p4_refresh" not in s, "already patched"
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old)); s = s.replace(old, new)

rep("import logging\n", "import json as _p4json\nimport logging\nimport os as _p4os\nimport time as _p4time\n")
rep("""        self.max_prefills_per_batch: int = 1
""", """        self.max_prefills_per_batch: int = 1

        # P4 (port of vllm-neuron PR#68 0cbaeb6106 + extension). Decode-only steps run for the
        # other running requests (a) after each chunk of a segmented prefill (PR#68 semantics,
        # decode_steps_between_prefill_chunks) and (b) after a prefill completes before the next
        # waiting request is admitted (decode_steps_between_prefills). 0/0 == stock behaviour.
        self._decode_steps_between_chunks = int(neuron_config.get(
            "decode_steps_between_prefill_chunks", _p4os.environ.get("P4_DSBPC", 0)))
        self._decode_steps_between_prefills = int(neuron_config.get(
            "decode_steps_between_prefills", _p4os.environ.get("P4_DSBP", 0)))
        assert self._decode_steps_between_chunks >= 0 and self._decode_steps_between_prefills >= 0
        self._interleave_decodes_left = 0
        self._interleave_new_left = 0
        self._p4_knob_path = _p4os.environ.get("P4_KNOBS", "")
        self._p4_knob_mtime = None
        self._p4_knob_check = 0.0
        _sl = _p4os.environ.get("P4_STEPLOG", "")
        self._p4_steplog = open(_sl, "a", buffering=1) if _sl else None
        logger.info("P4 interleave: decode_steps_between_prefill_chunks=%d decode_steps_between_prefills=%d knobs=%s",
                    self._decode_steps_between_chunks, self._decode_steps_between_prefills, self._p4_knob_path)
""")
rep("""    def _call_base_schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        \"\"\"Delegate to base vllm Scheduler. Override for async behavior.\"\"\"
""", """    def _p4_refresh(self) -> None:
        if not self._p4_knob_path:
            return
        now = _p4time.monotonic()
        if now - self._p4_knob_check < 0.5:
            return
        self._p4_knob_check = now
        try:
            m = _p4os.stat(self._p4_knob_path).st_mtime
        except OSError:
            return
        if m == self._p4_knob_mtime:
            return
        self._p4_knob_mtime = m
        try:
            d = _p4json.load(open(self._p4_knob_path))
            self._decode_steps_between_chunks = int(d.get("dsbpc", 0))
            self._decode_steps_between_prefills = int(d.get("dsbp", 0))
            # optional runtime prefill chunk size (token budget); 0 -> startup value
            if not hasattr(self, "_p4_budget0"):
                self._p4_budget0 = self.max_num_scheduled_tokens
            _b = int(d.get("budget", 0))
            self.max_num_scheduled_tokens = _b if _b > 0 else self._p4_budget0
            if self._p4_steplog is not None:
                self._p4_steplog.write("K %.4f dsbpc=%d dsbp=%d budget=%d tag=%s\\n" % (
                    _p4time.time(), self._decode_steps_between_chunks,
                    self._decode_steps_between_prefills, self.max_num_scheduled_tokens, d.get("tag", "")))
            logger.info("P4 knobs: dsbpc=%d dsbp=%d", self._decode_steps_between_chunks,
                        self._decode_steps_between_prefills)
        except Exception as e:  # noqa: BLE001
            logger.warning("P4 knob read failed: %s", e)

    def _call_base_schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        \"\"\"Delegate to base vllm Scheduler. Override for async behavior.\"\"\"
""")
rep("""        # Step 1: Move all waiting to holdback queue (preserving priority)
        while self.waiting:""", """        self._p4_refresh()
        # Step 1: Move all waiting to holdback queue (preserving priority)
        while self.waiting:""")
rep("""        max_num_running_reqs_override: int | None = None
        if self.has_prefill_in_running:
            # 3.1 Ongoing prefill segments in running queue - hide decode requests
            running_holdback = [
                req for req in self.running if not self._is_prefill_request(req)
            ]
            self.running = [
                req for req in self.running if self._is_prefill_request(req)
            ]
            if has_structured_output_waiting:
                max_num_running_reqs_override = len(self.running)
""", """        max_num_running_reqs_override: int | None = None
        decode_running = [
            req for req in self.running if not self._is_prefill_request(req)
        ]
        _p4_kind = ""
        if (
            self.has_prefill_in_running
            and decode_running
            and self._interleave_decodes_left > 0
        ):
            # 3.0 (PR#68) Decode step between two chunks of an ongoing prefill - hide the
            # prefill. No waiting request can be admitted (can_schedule() holds them back
            # while a prefill runs; the override blocks ready SO waiters).
            running_holdback = [
                req for req in self.running if self._is_prefill_request(req)
            ]
            self.running = decode_running
            max_num_running_reqs_override = len(self.running)
            self._interleave_decodes_left -= 1
            _p4_kind = "dc"
        elif (
            can_start_prefill
            and decode_running
            and self._interleave_new_left > 0
            and not has_structured_output_waiting
            and not self._skip_decode_holdback
        ):
            # 3.0b (P4) Decode step between two consecutive prefills: a prefill just
            # finished and another request waits. Put the admitted waiter(s) back so the
            # base scheduler cannot admit them, and run the decodes once more.
            _p4_back = []
            while self.waiting:
                _p4_back.append(self.waiting.popleft())
            for _r in reversed(_p4_back):
                self.holdback_queue.appendleft(_r)
            max_num_running_reqs_override = len(self.running)
            self._interleave_new_left -= 1
            _p4_kind = "dn"
        elif self.has_prefill_in_running:
            # 3.1 Ongoing prefill segments in running queue - hide decode requests
            running_holdback = decode_running
            self.running = [
                req for req in self.running if self._is_prefill_request(req)
            ]
            if has_structured_output_waiting:
                max_num_running_reqs_override = len(self.running)
            # After this chunk, let the hidden decodes run before the next one.
            if running_holdback:
                self._interleave_decodes_left = self._decode_steps_between_chunks
                self._interleave_new_left = self._decode_steps_between_prefills
""")
rep("""            # a step for nothing. ``running_holdback`` keeps its empty default, so the capacity
            # decrement is a no-op and the restore concatenates an empty list.
            logger.debug(""", """            # a step for nothing. ``running_holdback`` keeps its empty default, so the capacity
            # decrement is a no-op and the restore concatenates an empty list.
            self._interleave_decodes_left = 0
            self._interleave_new_left = 0
            logger.debug(""")
rep("""            running_holdback = self.running
            self.running = []
            if has_structured_output_waiting:
                available_slots""", """            running_holdback = self.running
            self.running = []
            if running_holdback:
                self._interleave_decodes_left = self._decode_steps_between_chunks
                self._interleave_new_left = self._decode_steps_between_prefills
            if has_structured_output_waiting:
                available_slots""")
rep("""            # 3.3 No prefill requests - no need to hide decodes
            if (""", """            # 3.3 No prefill requests - no need to hide decodes
            self._interleave_decodes_left = 0
            self._interleave_new_left = 0
            if (""")
rep("""        # Step 8: Update states
        self._update_state(scheduler_output)
""", """        # Step 8: Update states
        self._update_state(scheduler_output)
        if self._p4_steplog is not None:
            _nt = scheduler_output.total_num_scheduled_tokens
            _nr = len(scheduler_output.num_scheduled_tokens)
            if _nt:
                self._p4_steplog.write("S %.4f %s %d %d %d %d %s\\n" % (
                    _p4time.time(), "P" if _nt > _nr else "D", _nt, _nr, len(self.running),
                    len(self.waiting), _p4_kind))
""")
open(p, "w").write(s); print("patched OK", p)
