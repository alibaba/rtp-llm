#!/usr/bin/env python3
"""Deterministic, GPU-free policy experiment from query_features.csv.

Runs the *Java production shedding controllers* through the test bridge. The
event model supplies FIFO Prefill, aggregate FIFO Decode, delayed status/cache
visibility, TTL and topology changes; it is not the production scheduler/RPC
stack. Results are model evidence, not a production capacity measurement.
Raw query IDs, IPs and CSV records are never written to the output directory.
"""

import argparse
import array
import collections
import csv
import heapq
import json
import math
import pathlib
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class Feature:
    at: float
    input: int
    output: int
    keys: array.array
    prefix: int
    stream: str = ""
    sample_id: int = 0


def read_features(path):
    result, by_id = [], {}
    next_key = 1
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            parent = by_id.get(row["best_prefix_query_id"])
            size = int(row["input_len"]) // 512
            prefix = min(size, int(row["best_prefix_len_512_tokens"]) // 512)
            prefix = min(prefix, len(parent.keys)) if parent else 0
            keys = array.array("q", parent.keys[:prefix] if parent else [])
            keys.extend(range(next_key, next_key + size - prefix))
            next_key += size - prefix
            feature = Feature(
                int(row["query_start_epoch_ms"]) / 1000,
                int(row["input_len"]),
                int(row["output_len"]),
                keys,
                prefix,
            )
            result.append(feature)
            by_id[row["query_id"]] = feature
    return result


class Policy:
    def __init__(self, java, classpath, policy):
        self.process = subprocess.Popen(
            [
                java,
                "-cp",
                str(classpath),
                "org.flexlb.balance.scheduler.PrefillSheddingReplayBridge",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.policy = policy

    def victims(
        self, now, workers, saturated, outstanding, queue, arrivals, cold, expired
    ):
        header = f"{self.policy} {round(now * 1000)} {workers} {saturated} {outstanding} {len(queue)} {arrivals} {cold} {expired}\n"
        rows = [
            f"{r.id} {r.f.input} {hit} {round((now-r.at)*1000)}\n" for r, hit in queue
        ]
        self.process.stdin.write(header + "".join(rows))
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Java policy bridge exited")
        return {int(x) for x in line.strip().split(",") if x}

    def close(self):
        self.process.stdin.close()
        if self.process.wait(timeout=10):
            raise RuntimeError("Java policy bridge failed")


@dataclass
class Request:
    id: int
    at: float
    f: Feature
    keys: array.array
    cold: bool
    state: str = "global"
    work: int = 0
    actual_work: int = 0
    pending_work: int = 0
    admitted_at: float = 0


@dataclass
class Worker:
    id: int
    cache: dict = field(default_factory=dict)
    queue: collections.deque = field(default_factory=collections.deque)
    busy: bool = False
    pending: int = 0
    reported_waiting: int = 0
    above_since: float = math.inf
    active: bool = True
    ready: bool = True
    cache_version: int = 0
    hit_memo: dict = field(default_factory=dict)
    transit: int = 0


class Simulation:
    def __init__(
        self,
        features,
        policy,
        seed,
        prefill,
        decode,
        scenario,
        output,
        base_qps=40.0,
        copy_ratio=0.8,
        copy_step=600.0,
        status_prefix="actual",
    ):
        self.features, self.policy, self.seed = features, policy, seed
        self.rng = random.Random(seed)
        self.workers = [Worker(i) for i in range(prefill)]
        self.decode = [Worker(i) for i in range(decode)]
        self.initial_prefill, self.initial_decode = prefill, decode
        self.scenario, self.output = scenario, output
        self.events, self.serial, self.next_id = [], 0, 0
        self.global_queue = []
        self.requests = {}
        self.bins = collections.defaultdict(collections.Counter)
        self.arrivals = self.cold_arrivals = self.expired = 0
        self.max_queue = self.first_shed = 0
        self.next_dispatch = -1
        self.latencies = []
        self.latencies_by_bin = collections.defaultdict(list)
        self.ttft_by_bin = collections.defaultdict(list)
        self.base_qps, self.copy_ratio, self.copy_step = base_qps, copy_ratio, copy_step
        self.status_prefix = status_prefix
        self.first = min(f.at for f in features)
        self.ttl, self.delay = 1200, 0.020
        self.duration = 1800
        self.burst_start, self.burst_end = 600, 645
        if scenario == "sustained":
            self.burst_end = 780
        self.schedule(0, "status", None)
        self.schedule(0, "policy", None)
        self.schedule(0, "sample", None)

    def schedule(self, at, kind, value):
        self.serial += 1
        heapq.heappush(self.events, (at, self.serial, kind, value))

    def plan(self):
        if self.scenario in {"mixed_control", "mixed_thin"}:
            self.plan_mixed_trace()
            return
        if self.scenario == "trace":
            self.duration = max(f.at for f in self.features) - self.first
            for f in self.features:
                self.schedule(f.at - self.first, "arrival", (f, False, f.keys))
            return
        # Preserve empirical input/output correlation and prefix shape. Repeated
        # templates are synthetic warm traffic, not claimed to be original tokens.
        pool = self.rng.sample(self.features, min(512, len(self.features)))
        base_qps = self.base_qps
        at = 0
        while at < self.duration:
            f = self.rng.choice(pool)
            self.schedule(at, "arrival", (f, False, f.keys[: f.prefix]))
            at += self.rng.expovariate(base_qps)
        if self.scenario in {
            "burst",
            "brief",
            "sustained",
            "scale_out",
            "shrink_burst",
        }:
            rate = (
                10
                if self.scenario == "brief"
                else 15 if self.scenario == "sustained" else 40
            )
            if self.scenario == "brief":
                self.burst_end = 610
            at = self.burst_start
            while at < self.burst_end:
                f = self.rng.choice(self.features)
                self.schedule(at, "arrival", (f, True, array.array("q")))
                at += self.rng.expovariate(rate)
        if self.scenario == "scale_out":
            self.schedule(610, "scale", (len(self.workers) + 3, len(self.decode) + 8))
        if self.scenario in {"shrink", "shrink_burst"}:
            self.schedule(
                610,
                "scale",
                (max(1, len(self.workers) - 3), max(1, len(self.decode) - 8)),
            )

    def plan_mixed_trace(self):
        """Two independent copies; thin only stream A, keeping retained timestamps/keys.

        Time compression sets the total offered QPS. Cache TTL remains 1200s;
        the resulting inter-request gaps are therefore an explicit model assumption.
        """
        span = max(f.at for f in self.features) - self.first
        scale = self.base_qps * span / (2 * len(self.features))
        self.duration = span / scale + 0.01
        if not 0 < self.copy_step < self.duration:
            raise ValueError("copy-step must fall inside the scaled trace")
        self.burst_start = self.burst_end = self.copy_step
        sampling = random.Random(self.seed)
        for stream in range(2):
            for index, source in enumerate(self.features):
                at = (source.at - self.first) / scale + stream * 0.005
                keep = sampling.random() < self.copy_ratio
                if (
                    self.scenario == "mixed_thin"
                    and stream == 0
                    and at >= self.copy_step
                    and not keep
                ):
                    continue
                keys = array.array("q", (k + stream * (1 << 40) for k in source.keys))
                feature = Feature(
                    at,
                    source.input,
                    source.output,
                    keys,
                    source.prefix,
                    "a" if stream == 0 else "b",
                    stream * len(self.features) + index + 1,
                )
                self.schedule(at, "arrival", (feature, False, keys))

    def cache_hit(self, worker, request, now, visible=True):
        if not request.keys:
            return 0
        memo_key = (id(request.f), len(request.keys), visible)
        memo = worker.hit_memo.get(memo_key)
        if memo and memo[0] == worker.cache_version and now < memo[1]:
            return memo[2]
        # Keys identify complete prefixes, and writes refresh every ancestor.
        # Presence/expiry is therefore monotone along a request's key sequence.
        low, high = 0, len(request.keys)
        while low < high:
            middle = (low + high) // 2
            times = worker.cache.get(request.keys[middle])
            if times and now < times[0] + self.ttl and (not visible or times[1] <= now):
                low = middle + 1
            else:
                high = middle
        valid_until = (
            worker.cache[request.keys[low - 1]][0] + self.ttl if low else math.inf
        )
        if low < len(request.keys):
            missing = worker.cache.get(request.keys[low])
            if missing and visible and missing[1] > now:
                valid_until = min(valid_until, missing[1])
        hit = min(request.f.input, low * 512)
        worker.hit_memo[memo_key] = (worker.cache_version, valid_until, hit)
        return hit

    def publish_cache(self, worker, keys, now):
        for key in keys:
            previous = worker.cache.get(key)
            visible_at = (
                previous[1]
                if previous and now < previous[0] + self.ttl
                else now + self.delay
            )
            worker.cache[key] = (now, visible_at)
        if keys:
            worker.cache_version += 1

    def active(self):
        return [w for w in self.workers if w.active and w.ready]

    def matches(self, request, now, workers):
        hits = [(self.cache_hit(w, request, now), w) for w in workers]
        best = max((h for h, _ in hits), default=0)
        hot = best * 100 > request.f.input * 5
        return hits, best if hot else 0

    def request_dispatch(self, now):
        if self.next_dispatch < now:
            self.next_dispatch = now + 0.020
            self.schedule(self.next_dispatch, "dispatch", None)

    def start_prefill(self, worker, now):
        if worker.busy:
            return
        while worker.queue:
            r = worker.queue.popleft()
            if r.state != "prefill":
                worker.pending -= r.pending_work
                continue
            worker.busy = True
            hit = self.cache_hit(worker, r, now, visible=False)
            r.actual_work = r.f.input - hit
            # Per-request deterministic jitter makes A/B independent of eviction order.
            speed = 70000 * (
                1
                + random.Random(self.seed * 1000003 + (r.f.sample_id or r.id)).uniform(
                    -0.1, 0.1
                )
            )
            self.schedule(
                now + max(0.001, (r.f.input - hit) / speed), "prefill_done", (worker, r)
            )
            break

    def start_decode(self, worker, now):
        if worker.busy:
            return
        while worker.queue:
            r = worker.queue.popleft()
            if r.state != "decode":
                continue
            worker.busy = True
            speed = 2000 * (
                1
                + random.Random(self.seed * 2000003 + (r.f.sample_id or r.id)).uniform(
                    -0.1, 0.1
                )
            )
            self.schedule(now + r.f.output / speed, "decode_done", (worker, r))
            break

    def run(self):
        self.plan()
        while self.events:
            now, _, kind, value = heapq.heappop(self.events)
            if now > self.duration + 120:
                break
            bucket = self.bins[int(now // 10)]
            if kind == "arrival":
                f, cold, keys = value
                self.next_id += 1
                r = Request(self.next_id, now, f, keys, cold)
                self.requests[r.id] = r
                self.global_queue.append(r)
                self.arrivals += 1
                self.cold_arrivals += int(cold)
                bucket["offered"] += 1
                bucket["offered_input_tokens"] += f.input
                if f.stream:
                    bucket["offered_" + f.stream] += 1
                bucket["cold_offered"] += int(cold)
                self.schedule(now + 100, "expire", r)
                self.request_dispatch(now)
            elif kind == "dispatch":
                workers = self.active()
                kept = []
                for r in self.global_queue:
                    if r.state != "global":
                        continue
                    matches, best = self.matches(r, now, workers)
                    available = [
                        (hit, w)
                        for hit, w in matches
                        if (best == 0 or hit == best)
                        and w.reported_waiting <= 30
                        and len(w.queue) + int(w.busy) + w.transit < 256
                        and (
                            self.policy.policy == "legacy"
                            or best > 0
                            or w.pending < 200000
                        )
                    ]
                    if not available:
                        kept.append(r)
                        continue
                    hit, w = min(available, key=lambda x: (x[1].pending, x[1].id))
                    r.work = r.f.input - hit
                    r.pending_work = r.work
                    r.admitted_at = now
                    r.state = "transit"
                    w.pending += r.work
                    w.transit += 1
                    self.schedule(now + self.delay, "enqueue", (w, r))
                    bucket["dispatched"] += 1
                    bucket["dispatched_input_tokens"] += r.f.input
                    bucket["dispatched_cached_tokens"] += hit
                    bucket["cold_dispatched"] += int(best == 0)
                    bucket["dispatch_wait_ms"] += round((now - r.at) * 1000)
                self.global_queue = kept
                self.max_queue = max(self.max_queue, len(kept))
            elif kind == "enqueue":
                w, r = value
                w.transit -= 1
                if r.state == "transit":
                    r.state = "prefill"
                    w.queue.append(r)
                    self.start_prefill(w, now)
                    if self.status_prefix == "enqueue-zero":
                        self.schedule(
                            (math.floor(now * 10) + 1) / 10 + self.delay,
                            "prefix_observation",
                            (w, r),
                        )
                else:
                    w.pending -= r.pending_work
            elif kind == "prefix_observation":
                w, r = value
                if r.state == "prefill":
                    w.pending += r.f.input - r.pending_work
                    r.pending_work = r.f.input
            elif kind == "prefill_done":
                w, r = value
                w.pending -= r.pending_work
                w.busy = False
                if r.state == "prefill":
                    self.publish_cache(w, r.keys, now)
                    bucket["computed_tokens"] += r.work
                    bucket["actual_compute_tokens"] += r.actual_work
                    bucket["prefill_input_tokens"] += r.f.input
                    bucket["prefill_cached_tokens"] += r.f.input - r.actual_work
                    self.ttft_by_bin[int(now // 10)].append(now - r.at)
                    r.state = "decode_transit"
                    self.schedule(now + self.delay, "decode_enqueue", r)
                self.start_prefill(w, now)
                self.request_dispatch(now)
            elif kind == "decode_enqueue":
                r = value
                if r.state == "decode_transit":
                    w = min(
                        (w for w in self.decode if w.active),
                        key=lambda w: len(w.queue) + int(w.busy),
                    )
                    r.state = "decode"
                    w.queue.append(r)
                    self.start_decode(w, now)
            elif kind == "decode_done":
                w, r = value
                w.busy = False
                if r.state == "decode":
                    r.state = "success"
                    bucket["success"] += 1
                    bucket["success_tokens"] += r.f.input + r.f.output
                    bucket["output_tokens"] += r.f.output
                    bucket["warm_success"] += int(not r.cold)
                    self.latencies.append(now - r.at)
                    self.latencies_by_bin[int(now // 10)].append(now - r.at)
                self.start_decode(w, now)
            elif kind == "expire":
                r = value
                # timeoutMs is inactivity, not an end-to-end client deadline:
                # accepted/running work keeps receiving status and remains alive.
                if r.state in {"global", "transit"}:
                    if r.state == "global":
                        self.expired += 1
                    bucket["expired_" + r.state] += 1
                    bucket["expired"] += 1
                    r.state = "expired"
            elif kind == "status":
                for w in self.workers:
                    self.schedule(now + self.delay, "status_apply", (w, len(w.queue)))
                self.schedule(now + 0.1, "status", None)
            elif kind == "status_apply":
                w, waiting = value
                w.reported_waiting = waiting
                w.above_since = min(w.above_since, now) if waiting > 30 else math.inf
                self.request_dispatch(now)
            elif kind == "policy":
                workers = self.active()
                queue = [
                    (r, self.matches(r, now, workers)[1])
                    for r in self.global_queue
                    if r.state == "global"
                ]
                saturated = sum(
                    (
                        w.reported_waiting >= 27 or w.pending >= 200000
                        if self.policy.policy == "budget"
                        else now - w.above_since >= 5
                    )
                    for w in workers
                )
                victims = self.policy.victims(
                    now,
                    len(workers),
                    saturated,
                    sum(w.pending for w in workers),
                    queue,
                    self.arrivals,
                    self.cold_arrivals,
                    self.expired,
                )
                self.arrivals = self.cold_arrivals = self.expired = 0
                for r, hit in queue:
                    if r.id in victims:
                        r.state = "shed"
                        bucket["shed"] += 1
                        bucket["shed_hot"] += int(hit > 0)
                        self.first_shed = self.first_shed or now
                self.global_queue = [
                    r for r in self.global_queue if r.state == "global"
                ]
                self.schedule(now + 1, "policy", None)
            elif kind == "scale":
                np, nd = value
                for group, count in [(self.workers, np), (self.decode, nd)]:
                    while len(group) < count:
                        group.append(Worker(len(group), ready=False))
                        self.schedule(now + self.delay, "ready", group[-1])
                    for w in group:
                        w.active = w.id < count
                bucket["scale_prefill"] = np
                bucket["scale_decode"] = nd
            elif kind == "ready":
                value.ready = True
                self.request_dispatch(now)
            elif kind == "sample":
                bucket["global_queue"] = sum(
                    r.state == "global" for r in self.global_queue
                )
                bucket["active_prefill"] = len(self.active())
                bucket["active_decode"] = sum(w.active for w in self.decode)
                bucket["prefill_pending"] = sum(w.pending for w in self.workers)
                active = self.active()
                bucket["prefill_pending_max"] = max(
                    (w.pending for w in active), default=0
                )
                bucket["prefill_busy_workers"] = sum(w.busy for w in active)
                bucket["prefill_waiting_max"] = max(
                    (len(w.queue) for w in active), default=0
                )
                self.schedule(now + 10, "sample", None)
                if now % 300 == 0:
                    print(
                        f"progress {self.scenario}/{self.policy.policy}/{self.seed}: {now:.0f}s queue={bucket['global_queue']}",
                        file=sys.stderr,
                        flush=True,
                    )
        self.policy.close()
        for samples, prefix in [
            (self.latencies_by_bin, "latency"),
            (self.ttft_by_bin, "ttft"),
        ]:
            for index, values in samples.items():
                ordered = sorted(values)
                for quantile in (50, 95, 99):
                    self.bins[index][f"{prefix}_p{quantile}_s"] = ordered[
                        min(len(ordered) - 1, int(len(ordered) * quantile / 100))
                    ]
        total = collections.Counter()
        for b in self.bins.values():
            total.update(
                {
                    k: v
                    for k, v in b.items()
                    if k
                    in {
                        "success",
                        "offered",
                        "cold_offered",
                        "shed",
                        "shed_hot",
                        "expired",
                        "success_tokens",
                    }
                    or k.startswith("expired_")
                }
            )
        before = [self.bins[i]["warm_success"] / 10 for i in range(48, 60)]
        baseline = sum(before) / len(before)
        recovery = None
        for i in range(math.ceil(self.burst_end / 10), int(self.duration / 10) - 5):
            if all(
                self.bins[j]["warm_success"] / 10 >= baseline * 0.9
                and self.bins[j]["global_queue"] < 10
                for j in range(i, i + 6)
            ):
                recovery = i * 10 - self.burst_end
                break
        lat = sorted(self.latencies)
        unfinished = collections.Counter(
            r.state
            for r in self.requests.values()
            if r.state not in {"success", "shed", "expired"}
        )
        assert total["offered"] == total["success"] + total["shed"] + total[
            "expired"
        ] + sum(unfinished.values())
        summary = dict(
            total,
            unfinished=sum(unfinished.values()),
            unfinished_by_state=dict(unfinished),
            scenario=self.scenario,
            policy=self.policy.policy,
            seed=self.seed,
            prefill_start=self.initial_prefill,
            decode_start=self.initial_decode,
            duration_s=self.duration,
            base_qps=self.base_qps,
            copy_ratio=self.copy_ratio if self.scenario == "mixed_thin" else 1.0,
            copy_step_s=self.copy_step,
            status_prefix=self.status_prefix,
            max_global_queue=self.max_queue,
            first_shed_s=self.first_shed or None,
            recovery_s=recovery,
            latency_p99_s=lat[int(0.99 * len(lat))] if lat else None,
            steady_success_qps=baseline,
            steady_tail_qps=sum(self.bins[i]["warm_success"] for i in range(150, 170))
            / 200,
            burst_window_success=sum(self.bins[i]["success"] for i in range(60, 90)),
        )
        stem = f"{self.scenario}-{self.policy.policy}-{self.seed}"
        (self.output / (stem + ".json")).write_text(json.dumps(summary, indent=2))
        fields = sorted(set().union(*(x.keys() for x in self.bins.values())))
        with open(self.output / (stem + ".csv"), "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=["second"] + fields)
            writer.writeheader()
            for i, values in sorted(self.bins.items()):
                writer.writerow(dict(values, second=i * 10))
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--prefill", type=int, default=9)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--base-qps", type=float, default=40.0)
    parser.add_argument("--copy-ratio", type=float, default=0.8)
    parser.add_argument("--copy-step", type=float, default=600.0)
    parser.add_argument(
        "--status-prefix", choices=("actual", "enqueue-zero"), default="actual"
    )
    parser.add_argument(
        "--scenarios",
        default="trace,steady,brief,burst,sustained,scale_out,shrink,shrink_burst",
    )
    parser.add_argument("--modes", default="legacy,budget")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing results; use only with identical model/parameters",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(__file__).resolve().parents[2]
    # A Maven compile replaces target/classes. Freeze the tiny controller runtime
    # so a concurrent test build cannot invalidate a long experiment mid-matrix.
    runtime = args.output / ".policy-classes"
    for module, kind, package, names in [
        (
            "flexlb-sync",
            "test-classes",
            "org/flexlb/balance/scheduler",
            ["PrefillSheddingReplayBridge"],
        ),
        (
            "flexlb-sync",
            "classes",
            "org/flexlb/balance/scheduler",
            [
                "PrefillQueueShedding",
                "PrefillDeadlineBudget",
                "PrefillColdRequestProtection",
                "PrefillTrafficTrend",
            ],
        ),
        (
            "flexlb-common",
            "classes",
            "org/flexlb/config",
            ["PrefillQueueSheddingConfig"],
        ),
    ]:
        target = runtime / package
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            sources = list(
                (root / module / "target" / kind / package).glob(name + "*.class")
            )
            if not sources:
                raise RuntimeError(
                    f"Compile the test bridge before replay: missing {name}"
                )
            for source in sources:
                shutil.copy2(source, target / source.name)
    features = read_features(args.csv)
    results = []
    for scenario in args.scenarios.split(","):
        for seed in map(int, args.seeds.split(",")):
            for mode in args.modes.split(","):
                previous = args.output / f"{scenario}-{mode}-{seed}.json"
                if args.resume and previous.exists():
                    results.append(json.loads(previous.read_text()))
                    continue
                policy = Policy(args.java, runtime, mode)
                result = Simulation(
                    features,
                    policy,
                    seed,
                    args.prefill,
                    args.decode,
                    scenario,
                    args.output,
                    args.base_qps,
                    args.copy_ratio,
                    args.copy_step,
                    args.status_prefix,
                ).run()
                results.append(result)
                print(json.dumps(result), flush=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
