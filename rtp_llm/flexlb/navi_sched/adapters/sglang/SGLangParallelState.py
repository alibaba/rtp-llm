from __future__ import annotations


class SGLangParallelState:
    """Keep attention-DP state separate while reporting one serving node."""

    def __init__(self, size):
        if type(size) is not int or size <= 0:
            raise ValueError("attention-DP size must be a positive integer")
        self.size = size
        self.policies = {}
        self.workloads = {}
        self.request_ids = {}
        self.caches = {rank: [set(), set(), set()] for rank in range(size)}
        self.pending = {}
        self.execution_depth = {rank: 0 for rank in range(size)}
        self.policy = None

    def _rank(self, rank):
        if type(rank) is not int or not 0 <= rank < self.size:
            raise ValueError("invalid attention-DP rank")

    def ready(self, rank, policy):
        self._rank(rank)
        if rank in self.policies:
            raise ValueError("duplicate scheduler readiness")
        for previous in self.policies.values():
            for name in (
                "page_size",
                "linear_step",
                "hybrid_tail_required",
                "latency_fingerprint",
            ):
                if previous[name] != policy[name]:
                    raise ValueError("attention-DP schedulers disagree on " + name)
        self.policies[rank] = dict(policy)
        if len(self.policies) != self.size:
            return None
        return dict(
            policy, total_blocks=sum(x["total_blocks"] for x in self.policies.values())
        )

    def _combined_cache(self):
        cached, full, reusable_tail = set(), set(), set()
        for rank_cached, rank_full, rank_tail in self.caches.values():
            cached.update(rank_cached)
            full.update(rank_full)
            # A Full page on one rank cannot supply a Mamba tail on another.
            reusable_tail.update(rank_full & rank_tail)
        return cached, full, reusable_tail

    def cache_delta(self, rank, values):
        self._rank(rank)
        if len(values) != 6:
            raise ValueError("cache update requires three added/removed pairs")
        changes = [
            (set(added), set(removed))
            for added, removed in zip(values[::2], values[1::2])
        ]
        for added, removed in changes:
            if added & removed:
                raise ValueError("one cache update cannot add and remove the same page")
        changed = set().union(*(added | removed for added, removed in changes))

        def membership():
            cached = {
                key
                for key in changed
                if any(key in item[0] for item in self.caches.values())
            }
            full = {
                key
                for key in changed
                if any(key in item[1] for item in self.caches.values())
            }
            tail = {
                key
                for key in changed
                if any(
                    key in item[1] and key in item[2] for item in self.caches.values()
                )
            }
            return cached, full, tail

        before = membership()
        for cache, (added, removed) in zip(self.caches[rank], changes):
            cache.difference_update(removed)
            cache.update(added)
        after = membership()
        return [
            items
            for old, new in zip(before, after)
            for items in (sorted(new - old), sorted(old - new))
        ]

    def workload(self, rank, values, request_ids):
        self._rank(rank)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("duplicate scheduler request IDs")
        self.workloads[rank] = dict(values)
        self.request_ids[rank] = set(request_ids)
        result = {}
        for snapshot in self.workloads.values():
            for name, value in snapshot.items():
                if name == "event_time_us":
                    result[name] = max(result.get(name, 0), value)
                elif type(value) in (int, float):
                    result[name] = result.get(name, 0) + value
                elif type(value) is bool:
                    result[name] = result.get(name, False) or value
                elif isinstance(value, list):
                    result[name] = sorted(set(result.get(name, ())) | set(value))
        return result, sorted(set().union(*self.request_ids.values()))

    def reserve_rank(self, rid, hashes, token_count, requested_rank=None):
        if len(self.policies) != self.size:
            raise RuntimeError("attention-DP schedulers are not ready")
        if rid in self.pending:
            raise ValueError("request already has a rank reservation")

        def cost(rank):
            _, full, tail = self.caches[rank]
            matched = reusable = 0
            policy = self.policies[rank]
            for key in hashes:
                if key not in full:
                    break
                matched += 1
                if not policy["hybrid_tail_required"] or key in tail:
                    reusable = matched
            waiting = sum(
                tokens
                for pending_rid, (pending_rank, tokens) in self.pending.items()
                if pending_rank == rank
                and pending_rid not in self.request_ids.get(rank, ())
            )
            observed = self.workloads.get(rank, {}).get(
                "remaining_total_token_count", 0
            )
            return -reusable, waiting + observed, rank

        if requested_rank is None:
            rank = min(range(self.size), key=cost)
        else:
            self._rank(requested_rank)
            rank = requested_rank
        self.pending[rid] = rank, token_count
        return rank

    def release_rank(self, rid):
        self.pending.pop(rid, None)

    def snapshot(self):
        return {
            "size": self.size,
            "ready_ranks": sorted(self.policies),
            "workloads": dict(self.workloads),
            "cache_counts": {
                rank: [len(keys) for keys in groups]
                for rank, groups in self.caches.items()
            },
            "execution_depth": dict(self.execution_depth),
            "pending_rank_count": len(self.pending),
        }

    def reduce(self, message):
        rank = message["dp_rank"]
        self._rank(rank)
        kind = message["kind"]
        if kind == "ready":
            self.policy = self.ready(rank, message)
            if self.policy is None:
                return []
            cached, full, tail = self._combined_cache()
            result = [
                self.policy,
                {
                    "kind": "cache_ready",
                    "values": [sorted(cached), [], sorted(full), [], sorted(tail), []],
                },
            ]
            if self.workloads:
                values, ids = self.workload(
                    rank, self.workloads.get(rank, {}), self.request_ids.get(rank, ())
                )
                result.append(
                    {"kind": "workload", "values": values, "request_ids": ids}
                )
            return result
        if kind in ("cache", "cache_ready"):
            values = message.get("values")
            if values is None:
                values = [
                    message["inserted"],
                    message["removed"],
                    message["inserted"],
                    message["removed"],
                    [],
                    [],
                ]
            delta = self.cache_delta(rank, values)
            if self.policy is None:
                return []
            return [{"kind": "cache_ready", "values": delta}]
        if kind == "workload":
            values, ids = self.workload(rank, message["values"], message["request_ids"])
            return (
                []
                if self.policy is None
                else [{"kind": "workload", "values": values, "request_ids": ids}]
            )
        if kind in ("execution_started", "execution_finished"):
            before = sum(self.execution_depth.values())
            self.execution_depth[rank] += 1 if kind == "execution_started" else -1
            if self.execution_depth[rank] < 0:
                self.execution_depth[rank] = 0
                raise ValueError("execution finished without a matching start")
            after = sum(self.execution_depth.values())
            if before == 0 and after > 0:
                return [message]
            if before > 0 and after == 0:
                return [{"kind": "execution_finished"}]
            return []
        return [message] if self.policy is not None else []
