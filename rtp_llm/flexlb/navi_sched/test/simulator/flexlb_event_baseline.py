#!/usr/bin/env python3

import argparse
import collections
import hashlib
import heapq
import json
import math
import random
import statistics


class Request:
    def __init__(self, sequence, request_id, arrival_ms, prompt_tokens, block_hashes,
                 forced_node_index=-1):
        self.sequence = sequence
        self.request_id = request_id
        self.arrival_ms = arrival_ms
        self.prompt_tokens = prompt_tokens
        self.block_hashes = block_hashes
        self.forced_node_index = forced_node_index
        self.node_index = -1
        self.selected_hit_tokens = 0
        self.dispatch_ms = 0.0
        self.start_ms = 0.0
        self.finish_ms = 0.0
        self.predicted_total_ms = 0.0
        self.actual_hit_tokens = 0


class Batch:
    def __init__(self, batch_id, node_index, requests, dispatch_ms, reason,
                 predicted_execution_ms):
        self.batch_id = batch_id
        self.node_index = node_index
        self.requests = requests
        self.dispatch_ms = dispatch_ms
        self.reason = reason
        self.predicted_execution_ms = predicted_execution_ms
        self.start_ms = 0.0
        self.finish_ms = 0.0
        self.actual_execution_ms = 0.0


class Node:
    def __init__(self, index, cache_capacity_blocks):
        self.index = index
        self.cache_capacity_blocks = cache_capacity_blocks
        self.cache = collections.OrderedDict()
        self.waiting = collections.deque()
        self.execution_queue = collections.deque()
        self.running = None
        self.inflight = []
        self.reported_cache = set()
        self.reported_pending_count = 0
        self.timeout_generation = 0
        self.batches = []

    def cache_hit_blocks(self, block_hashes):
        count = 0
        for block_hash in block_hashes:
            if block_hash not in self.cache:
                break
            count += 1
        return count

    def reported_cache_hit_blocks(self, block_hashes):
        count = 0
        for block_hash in block_hashes:
            if block_hash not in self.reported_cache:
                break
            count += 1
        return count

    def read_cache(self, block_hashes):
        count = 0
        for block_hash in block_hashes:
            if block_hash not in self.cache:
                break
            count += 1
            self.cache.move_to_end(block_hash)
        return count

    def update_cache(self, block_hashes):
        for block_hash in block_hashes:
            if block_hash in self.cache:
                self.cache.move_to_end(block_hash)
                continue
            self.cache[block_hash] = None
            while len(self.cache) > self.cache_capacity_blocks:
                self.cache.popitem(last=False)


class FlexLBEventSimulator:
    EVENT_COMPLETE = 0
    EVENT_TIMEOUT = 1
    EVENT_ARRIVAL = 2

    def __init__(self, config):
        self.config = config
        block_bytes = config.block_size * config.token_cache_bytes
        cache_blocks = max(1, config.cache_capacity_bytes // block_bytes)
        self.nodes = [Node(index, cache_blocks) for index in range(config.node_count)]
        self.events = []
        self.event_sequence = 0
        self.next_batch_id = 1
        self.requests = []
        self.batches = []
        self.next_cache_status_ms = 0.0
        self.cache_status_interval_ms = config.cache_status_initial_interval_ms
        self.cache_diff_history = collections.deque(maxlen=30)
        self.cache_status_refresh_count = 0

    def push_event(self, time_ms, event_type, payload):
        self.event_sequence += 1
        heapq.heappush(
            self.events, (time_ms, event_type, self.event_sequence, payload))

    def run(self, requests):
        self.requests = requests
        for request in requests:
            self.push_event(request.arrival_ms, self.EVENT_ARRIVAL, request)
        while self.events:
            now_ms, event_type, _, payload = heapq.heappop(self.events)
            if event_type == self.EVENT_COMPLETE:
                self.complete_batch(payload, now_ms)
            elif event_type == self.EVENT_TIMEOUT:
                node_index, generation = payload
                self.handle_timeout(self.nodes[node_index], generation, now_ms)
            else:
                self.handle_arrival(payload, now_ms)
        return make_summary(self)

    def handle_arrival(self, request, now_ms):
        if request.forced_node_index >= 0:
            node, predicted_total_ms = self.select_forced_node(request, now_ms)
        else:
            node, predicted_total_ms = self.select_node(request, now_ms)
        request.node_index = node.index
        request.predicted_total_ms = predicted_total_ms
        if self.config.batching_mode == "engine_dynamic":
            self.dispatch_dynamic_request(node, request, now_ms)
            return
        was_empty = not node.waiting
        node.waiting.append(request)
        if len(node.waiting) >= self.config.batch_size_max:
            self.dispatch(node, now_ms, "batch_full")
        elif was_empty:
            node.timeout_generation += 1
            self.push_event(
                now_ms + self.config.fixed_window_ms,
                self.EVENT_TIMEOUT,
                (node.index, node.timeout_generation))

    def handle_timeout(self, node, generation, now_ms):
        if generation != node.timeout_generation or not node.waiting:
            return
        self.dispatch(node, now_ms, "fixed_window_timeout")

    def dispatch(self, node, now_ms, reason):
        picked = []
        while node.waiting and len(picked) < self.config.batch_size_max:
            request = node.waiting.popleft()
            request.dispatch_ms = now_ms
            picked.append(request)
        if not picked:
            return
        predicted_ms = flexlb_formula_latency_ms(
            picked, use_selected_hits=True,
            compute_token_cost=self.config.compute_token_cost,
            cache_hit_token_cost=self.config.cache_hit_token_cost)
        batch = Batch(
            self.next_batch_id, node.index, picked, now_ms, reason, predicted_ms)
        self.next_batch_id += 1
        node.batches.append(batch)
        node.execution_queue.append(batch)
        node.inflight.append(batch)
        self.batches.append(batch)
        if node.waiting:
            node.timeout_generation += 1
            self.push_event(
                node.waiting[0].arrival_ms + self.config.fixed_window_ms,
                self.EVENT_TIMEOUT,
                (node.index, node.timeout_generation))
        else:
            node.timeout_generation += 1
        self.start_next_batch(node, now_ms)

    def select_forced_node(self, request, now_ms):
        if request.forced_node_index >= len(self.nodes):
            raise ValueError("forced node index is outside the simulated cluster")
        node = self.nodes[request.forced_node_index]
        hit_blocks = node.cache_hit_blocks(request.block_hashes)
        request.selected_hit_tokens = bounded_hit_tokens(
            request.prompt_tokens, hit_blocks, self.config.block_size)
        predicted_total_ms = (self.predicted_endpoint_wait_ms(node, now_ms) +
                              nonlinear_latency_ms(
                                  [request], self.node_parameters(node.index),
                                  selected_hits=True))
        return node, predicted_total_ms

    def dispatch_dynamic_request(self, node, request, now_ms):
        request.dispatch_ms = now_ms
        predicted_ms = flexlb_formula_latency_ms(
            [request], use_selected_hits=True,
            compute_token_cost=self.config.compute_token_cost,
            cache_hit_token_cost=self.config.cache_hit_token_cost)
        reservation = Batch(
            -request.sequence - 1, node.index, [request], now_ms,
            "individual_route", predicted_ms)
        if self.config.selector_inflight_tracking:
            node.inflight.append(reservation)
        node.execution_queue.append(request)
        self.start_next_batch(node, now_ms)

    def start_next_batch(self, node, now_ms):
        if node.running is not None or not node.execution_queue:
            return
        if self.config.batching_mode == "engine_dynamic":
            requests = []
            while (node.execution_queue and
                   len(requests) < self.config.engine_batch_size_max):
                requests.append(node.execution_queue.popleft())
            batch = Batch(
                self.next_batch_id, node.index, requests, now_ms,
                "engine_dynamic", flexlb_formula_latency_ms(
                    requests, use_selected_hits=True,
                    compute_token_cost=self.config.compute_token_cost,
                    cache_hit_token_cost=self.config.cache_hit_token_cost))
            self.next_batch_id += 1
            node.batches.append(batch)
            self.batches.append(batch)
        else:
            batch = node.execution_queue.popleft()
        batch.start_ms = now_ms
        node.running = batch
        for request in batch.requests:
            request.start_ms = now_ms
            hit_blocks = node.read_cache(request.block_hashes)
            request.actual_hit_tokens = bounded_hit_tokens(
                request.prompt_tokens, hit_blocks, self.config.block_size)
        batch.actual_execution_ms = nonlinear_latency_ms(
            batch.requests, self.node_parameters(node.index))
        batch.finish_ms = now_ms + batch.actual_execution_ms
        self.push_event(batch.finish_ms, self.EVENT_COMPLETE, batch)

    def complete_batch(self, batch, now_ms):
        node = self.nodes[batch.node_index]
        if node.running is not batch:
            raise RuntimeError("completion does not match running batch")
        for request in batch.requests:
            request.finish_ms = now_ms
            node.update_cache(request.block_hashes)
        node.running = None
        if self.config.batching_mode == "engine_dynamic":
            completed = {request.sequence for request in batch.requests}
            if self.config.selector_inflight_tracking:
                node.inflight = [reservation for reservation in node.inflight
                                 if reservation.requests[0].sequence not in completed]
        else:
            node.inflight.remove(batch)
        self.start_next_batch(node, now_ms)

    def select_node(self, request, now_ms):
        self.refresh_cache_status(now_ms)
        candidates = []
        for node in self.nodes:
            if self.config.cache_status_initial_interval_ms > 0.0:
                hit_blocks = node.reported_cache_hit_blocks(request.block_hashes)
            else:
                hit_blocks = node.cache_hit_blocks(request.block_hashes)
            hit_tokens = bounded_hit_tokens(
                request.prompt_tokens, hit_blocks, self.config.block_size)
            endpoint_wait_ms = self.predicted_endpoint_wait_ms(node, now_ms)
            batcher_wait_ms = 0.0
            if node.waiting:
                batcher_wait_ms = max(
                    0.0,
                    node.waiting[0].arrival_ms + self.config.fixed_window_ms - now_ms)
            heuristic_ms = ((request.prompt_tokens - hit_tokens) *
                            self.config.compute_token_cost +
                            hit_tokens * self.config.cache_hit_token_cost)
            # The audited RTP runtime leaves waiting_query_len unset. With the
            # FlexLB batch flag enabled, non-batch routes also have no local
            # inflight reservation, so only explicitly tracked work is visible.
            pending_count = (sum(len(batch.requests) for batch in node.inflight) +
                             len(node.waiting))
            score = heuristic_ms + endpoint_wait_ms + batcher_wait_ms
            candidates.append({
                "node": node,
                "hit_tokens": hit_tokens,
                "endpoint_wait_ms": endpoint_wait_ms,
                "batcher_wait_ms": batcher_wait_ms,
                "pending_count": pending_count,
                "score": score,
            })

        average_wait = mean(
            item["endpoint_wait_ms"] for item in candidates)
        average_pending = mean(
            item["pending_count"] for item in candidates)
        survivors = []
        for item in candidates:
            if (self.config.hotspot_multiplier > 0 and average_pending > 0 and
                    item["pending_count"] >
                    average_pending * self.config.hotspot_multiplier):
                continue
            if (self.config.imbalance_multiplier > 0 and average_wait > 0 and
                    item["endpoint_wait_ms"] >
                    average_wait * self.config.imbalance_multiplier):
                continue
            survivors.append(item)
        if not survivors:
            survivors = [min(candidates, key=lambda item: item["endpoint_wait_ms"])]
        minimum = min(item["score"] for item in survivors)
        threshold = max(
            minimum * self.config.score_tie_threshold_pct,
            self.config.score_tie_threshold_ms)
        tied = [item for item in survivors if item["score"] <= minimum + threshold]
        selected = min(
            tied,
            key=lambda item: stable_tie_rank(
                request.request_id, item["node"].index, self.config.random_seed))
        request.selected_hit_tokens = selected["hit_tokens"]

        single_model_ms = nonlinear_latency_ms(
            [request], self.node_parameters(selected["node"].index),
            selected_hits=True)
        predicted_total_ms = (selected["endpoint_wait_ms"] +
                              selected["batcher_wait_ms"] + single_model_ms)
        return selected["node"], predicted_total_ms

    def refresh_cache_status(self, now_ms):
        interval_ms = self.cache_status_interval_ms
        if interval_ms <= 0.0 or now_ms < self.next_cache_status_ms:
            return
        for node in self.nodes:
            diff_size = len(set(node.cache).symmetric_difference(
                node.reported_cache))
            node.reported_cache = set(node.cache)
            self.update_dynamic_cache_interval(diff_size)
        self.cache_status_refresh_count += 1
        self.next_cache_status_ms = now_ms + self.cache_status_interval_ms

    def update_dynamic_cache_interval(self, diff_size):
        if self.config.cache_status_interval_mode != "flexlb_dynamic":
            return
        self.cache_diff_history.append(diff_size)
        if len(self.cache_diff_history) < 3:
            return
        average = mean(self.cache_diff_history)
        deviation = ((average - self.config.cache_status_target_diff) /
                     self.config.cache_status_target_diff)
        if abs(deviation) < 0.1:
            return
        factor = 0.7 if average > self.config.cache_status_target_diff else 1.3
        self.cache_status_interval_ms = min(
            self.config.cache_status_max_interval_ms,
            max(self.config.cache_status_min_interval_ms,
                math.floor(self.cache_status_interval_ms * factor + 0.5)))

    def predicted_endpoint_wait_ms(self, node, now_ms):
        if not node.inflight:
            return 0.0
        total = sum(batch.predicted_execution_ms for batch in node.inflight)
        progress_base = min(batch.dispatch_ms for batch in node.inflight)
        return max(0.0, total - max(0.0, now_ms - progress_base))

    def node_parameters(self, node_index):
        parameters = self.config.formula_parameters_by_node
        return parameters[node_index] if parameters else self.config.formula_parameters


class Config:
    def __init__(self, traffic_log, output, node_count, formula_parameters,
                 formula_parameters_by_node=None,
                 fixed_window_ms=50.0, batch_size_max=8, block_size=1024,
                 batching_mode="fixed_window", engine_batch_size_max=128,
                 selector_inflight_tracking=False,
                 cache_status_interval_mode="flexlb_dynamic",
                 cache_status_initial_interval_ms=100.0,
                 cache_status_min_interval_ms=50.0,
                 cache_status_max_interval_ms=3000.0,
                 cache_status_target_diff=30.0,
                 cache_capacity_bytes=300 * 1024 * 1024 * 1024,
                 token_cache_bytes=10 * 1024, compute_token_cost=1.0,
                 cache_hit_token_cost=0.3, score_tie_threshold_pct=0.1,
                 score_tie_threshold_ms=20.0, hotspot_multiplier=3.0,
                 imbalance_multiplier=3.0, random_seed=1,
                 steady_discard_fraction=0.5):
        self.traffic_log = traffic_log
        self.output = output
        self.node_count = node_count
        self.formula_parameters = formula_parameters
        self.formula_parameters_by_node = formula_parameters_by_node or []
        self.fixed_window_ms = fixed_window_ms
        self.batch_size_max = batch_size_max
        self.batching_mode = batching_mode
        self.engine_batch_size_max = engine_batch_size_max
        self.selector_inflight_tracking = selector_inflight_tracking
        self.cache_status_interval_mode = cache_status_interval_mode
        self.cache_status_initial_interval_ms = cache_status_initial_interval_ms
        self.cache_status_min_interval_ms = cache_status_min_interval_ms
        self.cache_status_max_interval_ms = cache_status_max_interval_ms
        self.cache_status_target_diff = cache_status_target_diff
        self.block_size = block_size
        self.cache_capacity_bytes = cache_capacity_bytes
        self.token_cache_bytes = token_cache_bytes
        self.compute_token_cost = compute_token_cost
        self.cache_hit_token_cost = cache_hit_token_cost
        self.score_tie_threshold_pct = score_tie_threshold_pct
        self.score_tie_threshold_ms = score_tie_threshold_ms
        self.hotspot_multiplier = hotspot_multiplier
        self.imbalance_multiplier = imbalance_multiplier
        self.random_seed = random_seed
        self.steady_discard_fraction = steady_discard_fraction


def bounded_hit_tokens(prompt_tokens, hit_blocks, block_size):
    hit_tokens = hit_blocks * block_size
    if hit_tokens >= prompt_tokens:
        hit_tokens = max(0, prompt_tokens - block_size)
    return min(prompt_tokens, max(0, hit_tokens))


def nonlinear_latency_ms(requests, parameters, selected_hits=False):
    if len(parameters) != 11:
        raise ValueError("the nonlinear latency model requires 11 parameters")
    batch_size = len(requests)
    reuse = 0.0
    compute = 0.0
    compute_square = 0.0
    reuse_compute = 0.0
    for request in requests:
        hit = (request.selected_hit_tokens if selected_hits
               else request.actual_hit_tokens)
        miss = request.prompt_tokens - hit
        reuse += hit / 1024.0
        compute += miss / 1024.0
        compute_square += (miss / 1024.0) ** 2
        reuse_compute += (hit / 1024.0) * (miss / 1024.0)
    values = [1.0, float(batch_size), reuse, compute,
              compute_square, reuse_compute, 1.0]
    linear = sum(value * weight for value, weight in zip(values, parameters)) / 320.0
    scaled_linear = linear * (parameters[9] + 1.0)
    nonlinear = scaled_linear + math.sqrt(scaled_linear * scaled_linear + 4.0)
    total_prompt = sum(request.prompt_tokens for request in requests) / 1024.0
    latency = (parameters[7] / 0.005 + parameters[8] / 0.02 * nonlinear +
               parameters[10] * total_prompt)
    return max(0.0, latency)


def flexlb_formula_latency_ms(requests, use_selected_hits,
                              compute_token_cost, cache_hit_token_cost):
    total = 0.0
    for request in requests:
        hit = request.selected_hit_tokens if use_selected_hits else request.actual_hit_tokens
        total += ((request.prompt_tokens - hit) * compute_token_cost +
                  hit * cache_hit_token_cost)
    return total


def stable_tie_rank(request_id, node_index, seed):
    value = f"{seed}:{request_id}:{node_index}".encode("utf-8")
    return hashlib.blake2b(value, digest_size=8).digest()


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def coefficient_of_variation(values):
    average = mean(values)
    return statistics.pstdev(values) / average if average else 0.0


def metric_summary(requests, node_count):
    latencies = [request.finish_ms - request.arrival_ms for request in requests]
    predictions = [request.predicted_total_ms for request in requests]
    errors = [actual - predicted for actual, predicted in zip(latencies, predictions)]
    node_counts = [0] * node_count
    for request in requests:
        node_counts[request.node_index] += 1
    return {
        "request_count": len(requests),
        "latency_mean_ms": mean(latencies),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_max_ms": max(latencies, default=0.0),
        "predicted_total_mean_ms": mean(predictions),
        "actual_total_mean_ms": mean(latencies),
        "total_actual_minus_predicted_bias_ms": (
            mean(errors)),
        "total_mae_ms": mean(abs(value) for value in errors),
        "total_mape_pct": (mean(
            abs(error) / actual for error, actual in zip(errors, latencies)
            if actual > 0.0) * 100.0 if latencies else 0.0),
        "node_request_counts": node_counts,
        "node_count_cv": coefficient_of_variation(node_counts),
    }


def make_summary(simulator):
    config = simulator.config
    discard = int(len(simulator.requests) * config.steady_discard_fraction)
    steady_requests = simulator.requests[discard:]
    steady_ids = {request.sequence for request in steady_requests}
    steady_batches = [batch for batch in simulator.batches
                      if any(request.sequence in steady_ids for request in batch.requests)]
    node_summaries = []
    for node in simulator.nodes:
        requests = [request for request in steady_requests
                    if request.node_index == node.index]
        batches = [batch for batch in steady_batches if batch.node_index == node.index]
        node_summaries.append({
            "node_index": node.index,
            "request_count": len(requests),
            "batch_count": len(batches),
            "average_batch_size": mean(len(batch.requests) for batch in batches),
            "card_time_ms": sum(batch.actual_execution_ms for batch in batches),
        })
    result = metric_summary(steady_requests, config.node_count)
    result.update({
        "discarded_request_count": discard,
        "batch_count": len(steady_batches),
        "average_batch_size": mean(
            len(batch.requests) for batch in steady_batches),
        "card_time_ms": sum(batch.actual_execution_ms for batch in steady_batches),
        "dispatch_reasons": dict(collections.Counter(
            batch.reason for batch in steady_batches)),
    })
    return {
        "baseline": "flexlb_node_local_event_v1",
        "effective_configuration": dict(vars(config)),
        "configuration_audit": {
            "node_local_fixed_window": config.batching_mode == "fixed_window",
            "engine_dynamic_batching": config.batching_mode == "engine_dynamic",
            "per_request_selection": True,
            "fifo_execution_per_node": True,
            "execution_noise_stddev_ratio": 0.0,
            "forced_node_request_count": sum(
                request.forced_node_index >= 0 for request in simulator.requests),
            "engine_waiting_query_len_observed": 0,
            "cache_status_refresh_count": simulator.cache_status_refresh_count,
            "cache_status_final_interval_ms": simulator.cache_status_interval_ms,
        },
        "result": result,
        "nodes": node_summaries,
        "batches": [{
            "batch_id": batch.batch_id,
            "node_index": batch.node_index,
            "request_sequences": [request.sequence for request in batch.requests],
            "dispatch_reason": batch.reason,
            "dispatch_ms": batch.dispatch_ms,
            "start_ms": batch.start_ms,
            "finish_ms": batch.finish_ms,
            "batch_size": len(batch.requests),
            "flexlb_predicted_execution_ms": batch.predicted_execution_ms,
            "actual_execution_ms": batch.actual_execution_ms,
        } for batch in simulator.batches],
    }


def load_requests(path):
    requests = []
    first_timestamp = None
    with open(path, encoding="utf-8") as source:
        for sequence, line in enumerate(source):
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp = record_timestamp_ms(record, sequence + 1)
            if first_timestamp is None:
                first_timestamp = timestamp
            prompt_tokens = int(record.get(
                "input_length", record.get("prompt_token_count", 0)))
            if prompt_tokens <= 0:
                raise ValueError(f"traffic line {sequence + 1} has no input length")
            hashes = record.get(
                "input_block_hash_ids", record.get("block_cache_keys", []))
            requests.append(Request(
                len(requests), str(record.get("request_id", sequence + 1)),
                timestamp - first_timestamp, prompt_tokens, tuple(map(str, hashes)),
                int(record.get("forced_node_index", -1))))
    if not requests:
        raise ValueError("traffic log is empty")
    return requests


def record_timestamp_ms(record, line_number=0):
    if "timestamp_ms" in record:
        return float(record["timestamp_ms"])
    if "timestamp" in record:
        timestamp = record["timestamp"]
        return float(timestamp) * 1000.0 if isinstance(timestamp, float) else float(timestamp)
    if "reqTime" in record:
        return float(record["reqTime"])
    raise ValueError(f"traffic line {line_number} has no timestamp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a node-local event simulation of the FlexLB baseline")
    parser.add_argument("--traffic-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--node-count", required=True, type=int)
    parser.add_argument("--formula-parameters-json", required=True)
    parser.add_argument("--formula-parameters-by-node-json", default="")
    parser.add_argument("--fixed-window-ms", type=float, default=50.0)
    parser.add_argument("--batch-size-max", type=int, default=8)
    parser.add_argument("--batching-mode", choices=("fixed_window", "engine_dynamic"),
                        default="fixed_window")
    parser.add_argument("--engine-batch-size-max", type=int, default=128)
    parser.add_argument("--selector-inflight-tracking", type=int, choices=(0, 1),
                        default=0)
    parser.add_argument("--cache-status-interval-mode",
                        choices=("fixed", "flexlb_dynamic"),
                        default="flexlb_dynamic")
    parser.add_argument("--cache-status-initial-interval-ms", type=float,
                        default=100.0)
    parser.add_argument("--cache-status-min-interval-ms", type=float, default=50.0)
    parser.add_argument("--cache-status-max-interval-ms", type=float, default=3000.0)
    parser.add_argument("--cache-status-target-diff", type=float, default=30.0)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--cache-capacity-bytes", type=int,
                        default=300 * 1024 * 1024 * 1024)
    parser.add_argument("--token-cache-bytes", type=int, default=10 * 1024)
    parser.add_argument("--compute-token-cost", type=float, default=1.0)
    parser.add_argument("--cache-hit-token-cost", type=float, default=0.3)
    parser.add_argument("--score-tie-threshold-pct", type=float, default=0.1)
    parser.add_argument("--score-tie-threshold-ms", type=float, default=20.0)
    parser.add_argument("--hotspot-multiplier", type=float, default=3.0)
    parser.add_argument("--imbalance-multiplier", type=float, default=3.0)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--steady-discard-fraction", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    parameters = json.loads(args.formula_parameters_json)
    parameters_by_node = json.loads(args.formula_parameters_by_node_json or "[]")
    config = Config(
        traffic_log=args.traffic_log,
        output=args.output,
        node_count=args.node_count,
        formula_parameters=parameters,
        formula_parameters_by_node=parameters_by_node,
        fixed_window_ms=args.fixed_window_ms,
        batch_size_max=args.batch_size_max,
        batching_mode=args.batching_mode,
        engine_batch_size_max=args.engine_batch_size_max,
        selector_inflight_tracking=bool(args.selector_inflight_tracking),
        cache_status_interval_mode=args.cache_status_interval_mode,
        cache_status_initial_interval_ms=args.cache_status_initial_interval_ms,
        cache_status_min_interval_ms=args.cache_status_min_interval_ms,
        cache_status_max_interval_ms=args.cache_status_max_interval_ms,
        cache_status_target_diff=args.cache_status_target_diff,
        block_size=args.block_size,
        cache_capacity_bytes=args.cache_capacity_bytes,
        token_cache_bytes=args.token_cache_bytes,
        compute_token_cost=args.compute_token_cost,
        cache_hit_token_cost=args.cache_hit_token_cost,
        score_tie_threshold_pct=args.score_tie_threshold_pct,
        score_tie_threshold_ms=args.score_tie_threshold_ms,
        hotspot_multiplier=args.hotspot_multiplier,
        imbalance_multiplier=args.imbalance_multiplier,
        random_seed=args.random_seed,
        steady_discard_fraction=args.steady_discard_fraction)
    if (config.node_count <= 0 or config.batch_size_max <= 0 or
            config.engine_batch_size_max <= 0 or
            config.block_size <= 0 or config.cache_capacity_bytes <= 0 or
            config.token_cache_bytes <= 0 or
            config.cache_status_initial_interval_ms < 0.0 or
            config.cache_status_min_interval_ms <= 0.0 or
            config.cache_status_max_interval_ms < config.cache_status_min_interval_ms or
            config.cache_status_target_diff <= 0.0 or
            not 0.0 <= config.steady_discard_fraction < 1.0):
        raise ValueError("count, cache, and discard configuration is invalid")
    if (len(config.formula_parameters) != 11 or
            (config.formula_parameters_by_node and
             (len(config.formula_parameters_by_node) != config.node_count or
              any(len(parameters) != 11
                  for parameters in config.formula_parameters_by_node)))):
        raise ValueError("formula parameters must contain 11 values for every node")
    summary = FlexLBEventSimulator(config).run(load_requests(config.traffic_log))
    with open(config.output, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "output": config.output,
        "effective_configuration": summary["effective_configuration"],
        "result": summary["result"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
