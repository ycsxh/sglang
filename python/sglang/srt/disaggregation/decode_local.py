"""Opt-in TP1 decode-local continuation with exclusive, bounded state leases.

The scheduler is the sole owner. A lease retains the whole request KV row and
hybrid recurrent slot, without inserting either into the decode radix cache.
No defaults or prediction policy live here: the operator supplies a frozen JSON
configuration and a completion evaluator. Non-overlap/no-spec execution is
required so the committed FULL prefix and live recurrent state have one length.
"""

import base64
import importlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass

PREFIX = "dlp."


def identity(rid):
    if not isinstance(rid, str) or not rid.startswith(PREFIX):
        return None
    try:
        obj = json.loads(base64.urlsafe_b64decode(rid[len(PREFIX) :]))
        if not isinstance(obj, dict):
            raise ValueError("identity is not an object")
        for key in ("run", "session", "request", "generation", "op"):
            if not isinstance(obj[key], str) or not obj[key]:
                raise ValueError("invalid identity")
        if type(obj["turn"]) is not int or obj["turn"] < 0:
            raise ValueError("invalid turn")
        if obj["op"] not in ("local", "pd", "close"):
            raise ValueError("invalid operation")
        if obj["turn"] and not isinstance(obj.get("predecessor"), str):
            raise ValueError("missing predecessor")
        return obj
    except Exception as exc:
        raise ValueError("malformed decode-local identity") from exc


@dataclass
class Lease:
    owner: object
    ident: dict
    prefix: tuple
    born_ns: int
    expires_ns: int
    size_bytes: int


class LeaseStore:
    """Scheduler-thread-only atomic consume and exactly-once request admission."""

    def __init__(self, config, release, record, clock=time.monotonic_ns):
        self.config, self.release, self.record, self.clock = (
            config,
            release,
            record,
            clock,
        )
        self.leases = {}
        self.seen = set()
        self.frontiers = {}
        self.closed = set()
        self.bytes = 0

    def key(self, ident):
        return ident["run"], ident["session"]

    def drop(self, key, reason):
        lease = self.leases.pop(key, None)
        if lease is None:
            return
        self.bytes -= lease.size_bytes
        now = self.clock()
        self.record(
            "unpin",
            identity=lease.ident,
            reason=reason,
            bytes=lease.size_bytes,
            held_ns=now - lease.born_ns,
            byte_ns=lease.size_bytes * (now - lease.born_ns),
        )
        self.release(lease.owner)

    def expire(self):
        now = self.clock()
        for key, lease in list(self.leases.items()):
            if now >= lease.expires_ns:
                self.drop(key, "expired")

    def validate(self, ident):
        self.expire()
        key = self.key(ident)
        logical = key + (ident["request"],)
        if (
            ident["run"] != self.config["run"]
            or ident["generation"] != self.config["generation"]
        ):
            return "generation", False
        if key in self.closed:
            return "closed", False
        if logical in self.seen:
            return "duplicate", False
        previous = self.frontiers.get(key)
        if previous is not None:
            if (
                ident["turn"] != previous[0] + 1
                or ident.get("predecessor") != previous[1]
            ):
                return "stale_or_predecessor", False
        elif ident["turn"] != 0:
            return "no_frontier", False
        if len(self.seen) >= self.config["max_requests"]:
            return "identity_budget", False
        return None, True

    def admit(
        self, ident, tokens, capacity_ok=lambda lease: True, restore=lambda lease: None
    ):
        reason, _ = self.validate(ident)
        if reason:
            return None, reason, False
        key = self.key(ident)
        if ident["op"] == "local":
            lease = self.leases.get(key)
            if lease is None:
                return None, "no_lease", True
            if lease.ident["request"] != ident.get("predecessor"):
                return None, "lease_predecessor", True
            n = len(lease.prefix)
            if len(tokens) <= n or tuple(tokens[:n]) != lease.prefix:
                return None, "prefix_mismatch", True
            if not capacity_ok(lease):
                return None, "capacity", True
            # No yield, callback or output can intervene between validating the
            # state, moving its ownership into the execution and consuming it.
            restore(lease)
            del self.leases[key]
            self.bytes -= lease.size_bytes
            self.seen.add(key + (ident["request"],))
            self.frontiers[key] = (ident["turn"], ident["request"])
            now = self.clock()
            self.record(
                "consume",
                identity=ident,
                predecessor=lease.ident,
                bytes=lease.size_bytes,
                held_ns=now - lease.born_ns,
                byte_ns=lease.size_bytes * (now - lease.born_ns),
                matched_tokens=n,
                new_tokens=len(tokens) - n,
            )
        else:
            lease = None
            self.drop(key, "pd_replacement")
        self.seen.add(key + (ident["request"],))
        self.frontiers[key] = (ident["turn"], ident["request"])
        return lease, None, False

    def pin(self, owner, ident, prefix, size_bytes):
        self.expire()
        key = self.key(ident)
        if key in self.closed or key in self.leases:
            return False
        if (
            len(self.leases) >= self.config["max_leases"]
            or self.bytes + size_bytes > self.config["max_pin_bytes"]
        ):
            return False
        now = self.clock()
        lease = Lease(
            owner, ident, tuple(prefix), now, now + self.config["ttl_ns"], size_bytes
        )
        self.leases[key] = lease
        self.bytes += size_bytes
        self.record(
            "pin",
            identity=ident,
            bytes=size_bytes,
            valid_tokens=len(prefix),
            pin_ns=now,
            expires_ns=lease.expires_ns,
        )
        return True

    def close(self, ident):
        key = self.key(ident)
        self.closed.add(key)
        self.drop(key, "close")


class DecodeLocalManager:
    def __init__(self, scheduler, config):
        from sglang.srt.mem_cache.chunk_cache import ChunkCache
        from sglang.srt.runtime_context import get_parallel

        if scheduler.enable_overlap or not scheduler.spec_algorithm.is_none():
            raise ValueError("decode-local requires non-overlap and no speculation")
        if get_parallel().tp_size != 1 or type(scheduler.tree_cache) is not ChunkCache:
            raise ValueError("decode-local requires TP1 exclusive ChunkCache")
        if scheduler.enable_hisparse:
            raise ValueError("decode-local does not support HiSparse")
        self.scheduler, self.config = scheduler, config
        pool = scheduler.req_to_token_pool
        if hasattr(pool, "mamba_pool"):
            state = pool.mamba_pool.mamba_cache
            tensors = [*state.conv, state.temporal]
            actual = sum(t[:, 0].numel() * t.element_size() for t in tensors)
            if actual != config["mamba_bytes_per_slot"]:
                raise ValueError(f"recurrent state byte identity mismatch: {actual}")
        module, name = config["evaluator"].split(":")
        self.evaluate = getattr(importlib.import_module(module), name)
        self.queue = deque()
        self.active = {}
        self.after_prefill = None
        self.decode_due = False
        self._occupancy_at_ns = 0
        self._occupancy_previous = None
        self.store = LeaseStore(config, self.release, self.record)
        scheduler.tree_cache.decode_local_manager = self
        self.record("start", config=config)

    def record(self, event, **fields):
        row = dict(
            event=event,
            monotonic_ns=time.monotonic_ns(),
            wall_ns=time.time_ns(),
            **fields,
        )
        with open(self.config["events"], "a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def release(self, req):
        from sglang.srt.mem_cache.common import release_kv_cache

        req.decode_local_releasing = True
        release_kv_cache(req, self.scheduler.tree_cache)

    def retain(self, req):
        ident = getattr(req, "decode_local_identity", None)
        if ident is None or getattr(req, "decode_local_releasing", False):
            return False
        self.active.pop(req.rid, None)
        # Only the completion result (after GPU synchronization) reaches here.
        t0 = time.monotonic_ns()
        valid = req.kv.kv_committed_len
        expected = len(req.origin_input_ids) + len(req.output_ids) - 1
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        if (
            isinstance(req.finished_reason, FINISH_ABORT)
            or not req.finished()
            or valid != expected
            or valid <= 0
            or req.kv.cache_protected_len != 0
        ):
            self.record(
                "no_pin",
                identity=ident,
                reason="aborted_or_invalid_state",
                valid=valid,
                expected=expected,
            )
            return False
        # Count physical page rounding and every exclusively retained recurrent
        # slot (main + ping/pong buffers). Sizes are frozen and GPU-audited.
        page = self.scheduler.page_size
        slots = 1 + (
            len(req.kv.mamba_ping_pong_track_buffer)
            if req.kv.mamba_ping_pong_track_buffer is not None
            else 0
        )
        size = (req.kv.kv_allocated_len + page - 1) // page * page * self.config[
            "kv_bytes_per_token"
        ] + (slots * self.config["mamba_bytes_per_slot"] if req.kv.holds_mamba else 0)
        text = req.tokenizer.decode(req.output_ids, skip_special_tokens=True)
        decision = self.evaluate(text, t0, time.monotonic_ns, self.config)
        self.record(
            "completion",
            identity=ident,
            completed_ns=t0,
            decision=decision,
            completion_hold_ns=time.monotonic_ns() - t0,
            completion_hold_bytes=size,
            asynchronous_temporary_pin_bytes=0,
            valid_tokens=valid,
            allocated_tokens=req.kv.kv_allocated_len,
            row=req.kv.req_pool_idx,
            mamba_slot=int(req.kv.mamba_pool_idx) if req.kv.holds_mamba else None,
            output_ids=list(req.output_ids),
        )
        if not decision["pin"]:
            return False
        prefix = (req.origin_input_ids + req.output_ids)[:valid]
        ok = self.store.pin(req, ident, prefix, size)
        if not ok:
            self.record("no_pin", identity=ident, reason="pin_budget")
        return ok

    def handle(self, req):
        from sglang.srt.disaggregation.utils import prepare_abort
        from sglang.srt.managers.schedule_batch import ReqKvInfo

        try:
            ident = identity(req.rid)
        except ValueError:
            ident = None
            reason, fallback = "malformed_identity", False
        else:
            if ident is None:
                return False
            if ident["op"] == "close":
                if (
                    ident["run"] == self.config["run"]
                    and ident["generation"] == self.config["generation"]
                ):
                    self.store.close(ident)
                    # Cancel admitted execution as well as leases on trusted close.
                    for active in list(self.active.values()):
                        if self.store.key(
                            active.decode_local_identity
                        ) == self.store.key(ident):
                            active.to_finish = __import__(
                                "sglang.srt.managers.schedule_batch",
                                fromlist=["FINISH_ABORT"],
                            ).FINISH_ABORT("session closed", 499)
                    reason, fallback = "closed", False
                else:
                    reason, fallback = "generation", False
            else:

                def capacity(lease):
                    needed = (
                        len(req.origin_input_ids)
                        - len(lease.prefix)
                        + req.sampling_params.max_new_tokens
                    )
                    available = (
                        self.scheduler.token_to_kv_pool_allocator.available_size()
                    )
                    return (
                        available >= needed + self.scheduler.page_size
                        and len(self.active) < self.config["max_local_running"]
                    )

                def restore(lease):
                    old = lease.owner
                    assert old.kv.holds_kv and old.kv.kv_committed_len == len(
                        lease.prefix
                    )
                    indices = self.scheduler.req_to_token_pool.req_to_token[
                        old.kv.req_pool_idx, : len(lease.prefix)
                    ].to(dtype=__import__("torch").int64, copy=True)
                    req.kv, old.kv = old.kv, ReqKvInfo()
                    req.kv.mamba_needs_clear = False
                    req.kv.mamba_cow_src_index = None
                    req.prefix_indices = indices
                    req.decode_local_valid = len(lease.prefix)
                    req.decode_local_mamba_len = len(lease.prefix)

                lease, reason, fallback = self.store.admit(
                    ident, req.origin_input_ids, capacity, restore
                )
                if reason is None:
                    req.decode_local_identity = ident
                    if ident["op"] == "pd":
                        self.record("pd_admit", identity=ident)
                        return False
                    self.record(
                        "state_handoff",
                        identity=ident,
                        row=req.kv.req_pool_idx,
                        full_valid=req.decode_local_valid,
                        mamba_valid=req.decode_local_mamba_len,
                        mamba_slot=int(req.kv.mamba_pool_idx)
                        if req.kv.holds_mamba
                        else None,
                    )
                    req.init_next_round_input()
                    self.queue.append(req)
                    self.active[req.rid] = req
                    self.record("accepted", identity=ident)
                    return True
        self.record(
            "rejected", identity=ident, reason=reason, fallback_allowed=fallback
        )
        marker = "DLOCAL_NOT_ACCEPTED:" if fallback else "DLOCAL_TERMINAL:"
        prepare_abort(req, marker + reason, status_code=409)
        self.scheduler.output_streamer.stream_output([req], req.return_logprob)
        return True

    def cancel(self, abort):
        from sglang.srt.disaggregation.utils import prepare_abort
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        def matches(req):
            return abort.abort_all or req.rid.startswith(abort.rid)

        for key, lease in list(self.store.leases.items()):
            if matches(lease.owner):
                self.store.drop(key, "cancel")
        pending = deque()
        for req in self.queue:
            if matches(req):
                prepare_abort(req, "decode-local cancelled", status_code=499)
                self.release(req)
                self.active.pop(req.rid, None)
                self.scheduler.output_streamer.stream_output([req], req.return_logprob)
                self.record(
                    "cancelled", identity=req.decode_local_identity, queued=True
                )
            else:
                pending.append(req)
        self.queue = pending
        for req in self.active.values():
            if matches(req) and not req.finished():
                req.to_finish = FINISH_ABORT("decode-local cancelled", 499)
                self.record(
                    "cancelled", identity=req.decode_local_identity, queued=False
                )

    def record_occupancy(self, running):
        """Optional scheduler-owned queue/lease gauges, independent of HTTP slots.

        Counts describe resident scheduler work, not simultaneous GPU kernels.
        Sampling is opt-in and bounded; lease/execution ownership is untouched.
        """
        interval = self.config.get("occupancy_interval_ns", 0)
        if interval <= 0:
            return
        now = time.monotonic_ns()
        counts = (
            len(self.queue),
            len(self.active),
            len(running.reqs),
            len(self.store.leases),
            self.store.bytes,
            len(self.store.seen),
        )
        idle = not any(counts[:4])
        if now - self._occupancy_at_ns < interval:
            return
        if idle and counts == self._occupancy_previous:
            return
        self._occupancy_at_ns = now
        self._occupancy_previous = counts
        self.record(
            "occupancy",
            queued_local=counts[0],
            active_local=counts[1],
            running_decode=counts[2],
            leases=counts[3],
            pin_bytes=counts[4],
            seen_requests=counts[5],
        )

    def yield_on_idle(self):
        """Let completion/control threads run while polling retained leases.

        Never wait across a lease deadline or when local work is queued. The
        default remains the MVP's busy polling; experiments opt in explicitly.
        """
        interval = self.config.get("idle_yield_ns", 0)
        if interval <= 0 or self.queue or not self.store.leases:
            return
        if interval > 10_000_000:
            raise ValueError("decode-local idle yield must be <= 10 ms")
        now = time.monotonic_ns()
        remaining = min(lease.expires_ns for lease in self.store.leases.values()) - now
        time.sleep(min(interval, max(0, remaining)) / 1e9)

    def next_batch(self, running):
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        self.store.expire()
        self.record_occupancy(running)
        if self.after_prefill is not None:
            batch = self.after_prefill
            self.after_prefill = None
            batch.filter_batch()
            if not batch.is_empty():
                if running.is_empty():
                    running = batch
                else:
                    running.merge_batch(batch)
        while self.queue and self.queue[0].finished():
            self.queue.popleft()
        if not self.queue or (self.decode_due and not running.is_empty()):
            self.decode_due = False
            return None, running
        req = self.queue[0]
        start = req.decode_local_valid
        end = min(len(req.origin_input_ids), start + self.config["chunk_size"])
        req.init_next_round_input()
        req.set_extend_range(start, end)
        middle = end < len(req.origin_input_ids)
        req.inflight_middle_chunks += int(middle)
        req.prefix_indices = self.scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :start
        ].to(dtype=__import__("torch").int64, copy=True)
        batch = ScheduleBatch.init_new(
            [req],
            self.scheduler.req_to_token_pool,
            self.scheduler.token_to_kv_pool_allocator,
            self.scheduler.tree_cache,
            self.scheduler.model_config,
            False,
            self.scheduler.spec_algorithm,
        )
        batch.prepare_for_extend()
        req.decode_local_valid = end
        self.record(
            "local_prefill",
            identity=req.decode_local_identity,
            prefix_tokens=start,
            computed_tokens=end - start,
            end=end,
            mamba_start=req.decode_local_mamba_len,
            middle=middle,
        )
        req.decode_local_mamba_len = end
        if not middle:
            self.queue.popleft()
            self.after_prefill = batch
        self.decode_due = True
        return batch, running


def manager_for(scheduler):
    if not hasattr(scheduler, "decode_local_manager"):
        path = os.environ.get("SGLANG_DECODE_LOCAL_CONFIG")
        scheduler.decode_local_manager = (
            DecodeLocalManager(scheduler, json.load(open(path))) if path else None
        )
    return scheduler.decode_local_manager
