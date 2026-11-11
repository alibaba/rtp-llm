"""Plan every BF16 checkpoint read across the separate 64 training ranks."""

from dataclasses import asdict
import hashlib
import ipaddress
import json

from q38_remote_checkpoint import PARAMETERS, TRAIN_EP


def plan_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_plan(checkpoint, train_ips, rollout_ips, port):
    train_ips = tuple(str(ipaddress.IPv4Address(value)) for value in train_ips)
    rollout_ips = tuple(str(ipaddress.IPv4Address(value)) for value in rollout_ips)
    if len(train_ips) != 16 or len(rollout_ips) != 8 or len(set(train_ips + rollout_ips)) != 24:
        raise ValueError("expected 16 training and eight separate rollout containers")
    if type(port) is not int or not 1024 <= port < 65536:
        raise ValueError("invalid checkpoint relay port")
    by_file = {name: [] for name in checkpoint.metadata.sizes}
    owner_bytes = [0] * TRAIN_EP
    for name in sorted(checkpoint.metadata.policy):
        location = checkpoint.locations[name]
        routed = ".mlp.experts." in name
        if routed and location.shape[0] != 512:
            raise ValueError(f"expected 512 expert rows for {name}")
        ranks = range(TRAIN_EP) if routed else (0,)
        for rank in ranks:
            start, stop = (8*rank, 8*(rank+1)) if routed else (0, location.shape[0])
            for selected in location.ranges(start, stop):
                by_file[location.shard].append((train_ips[rank//4], selected))
                owner_bytes[rank] += selected.length
    if sum(owner_bytes) != 2*PARAMETERS or min(owner_bytes) <= 0:
        raise ValueError("initial reads must cover every BF16 policy byte exactly once")

    source_bytes = [0] * len(rollout_ips)
    relay_plans = {host: {} for host in rollout_ips}
    endpoints = {}
    byte_counts = {name: sum(selected.length for _, selected in requests)
                   for name, requests in by_file.items()}
    for name in sorted(by_file, key=lambda name: (-byte_counts[name], name)):
        requests = by_file[name]
        source = min(range(len(rollout_ips)), key=lambda index: (source_bytes[index], index))
        host = rollout_ips[source]
        endpoints[name] = [host, port]
        source_bytes[source] += byte_counts[name]
        seen = set()
        for peer, selected in sorted(requests, key=lambda row: row[1].offset):
            if selected.offset in seen:
                raise ValueError(f"duplicate physical source range at {name} offset {selected.offset}")
            seen.add(selected.offset)
            relay_plans[host].setdefault(peer, []).append(asdict(selected))
    if sum(source_bytes) != sum(owner_bytes):
        raise ValueError("relay byte totals differ from the training plan")
    source_ranges = {host: sum(map(len, plan.values())) for host, plan in relay_plans.items()}
    return endpoints, relay_plans, {
        "total_bytes": sum(source_bytes),
        "per_source_bytes": dict(zip(rollout_ips, source_bytes)),
        "per_source_ranges": source_ranges,
        "per_source_plan_sha256": {host: plan_digest(plan) for host, plan in relay_plans.items()},
        "per_rank_bytes": owner_bytes,
        "range_count": sum(source_ranges.values()),
        "plan_sha256": plan_digest(relay_plans),
    }
