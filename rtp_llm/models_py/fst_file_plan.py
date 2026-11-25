"""Select collective vs rank-local FST transport before opening payloads."""

import hashlib
import os

import torch
from safetensors import safe_open


def plan_fst_files(files, root, process_group, model_filter, is_fixed_host, catalog=None,
                   *, resolver=None, expander=None):
    """Only share files whose collective tensor schedule agrees on every peer.

    Native FST brokers a file sequence, not a list of tensor names per rank.
    Different files/keys at the same collective position can silently exchange
    equally shaped bytes. Keep such shards on the independent native range
    reader. A shared file containing fixed-host weights also takes that route:
    even ignoring a key cannot prevent the FST file copier from staging its bytes.
    """
    by_name = {os.path.relpath(path, root): path for path in files}
    world = process_group.size()
    if world == 1 and is_fixed_host is None and catalog is None and expander is None:
        return list(files), []
    local = {}
    error = None
    try:
        for name, path in by_name.items():
            layout, schedule, placement = (hashlib.sha256() for _ in range(3))
            rank = process_group.rank() if world > 1 else 0
            independent = False
            def record(digest, *items):
                raw = repr(items).encode("utf-8")
                digest.update(len(raw).to_bytes(8, "little"))
                digest.update(raw)
            with safe_open(path, framework="pt", device="cpu") as shard:
                # safe_open validated this header; keys() can be alphabetically
                # sorted even when the native file buffer visits another order.
                # Compare serialized order/offsets too, without reading weights.
                if world > 1:
                    with open(path, "rb") as stream:
                        remaining = int.from_bytes(stream.read(8), "little")
                        while remaining:
                            chunk = stream.read(min(remaining, 1 << 16))
                            if not chunk:
                                raise ValueError(f"Truncated safetensors header: {name}")
                            layout.update(chunk)
                            remaining -= len(chunk)
                keys = shard.keys()
                if catalog is not None and len(catalog.shard_keys(path, keys)) != len(keys):
                    independent = True
                else:
                    for key in keys:
                        view = shard.get_slice(key)
                        shape = tuple(view.get_shape())
                        selected = model_filter is None or model_filter(key)
                        record(layout, key, shape, view.get_dtype(), selected)
                        if is_fixed_host is not None and is_fixed_host(key):
                            independent = True
                        if not selected:
                            continue
                        if expander is not None and expander.handles_safetensor(key):
                            # Native collective requires equal canonical shards.
                            if shape[0] % expander.ep_size:
                                independent = True
                            mapping = expander._placement(key)
                            record(placement, key, mapping.physical_to_logical if mapping is not None else None)
                            record(schedule, key, "stacked", expander.ep_size)
                        elif resolver is not None:
                            spec = resolver(key, shape)
                            targets = spec.target_ranks
                            if targets is not None:
                                if any(t >= world for t in targets) or spec.load != (rank in targets):
                                    raise ValueError(f"Inconsistent FST target ranks for {key}")
                                record(placement, key, targets)
                            if spec.segments is not None:
                                op = "range"
                            elif targets is not None and targets != tuple(range(world)):
                                op = ("route", targets)
                            elif not spec.load:
                                op = "skip"
                            elif spec.shard_dim is not None and spec.shard_count == world and spec.shard_rank == rank:
                                op = ("scatter", spec.shard_dim)
                            else:
                                op = "broadcast"
                            record(schedule, key, op)
            local[name] = (layout.digest(), schedule.digest(), placement.digest(), independent)
    except Exception as exc:
        # Peers must see preflight failures before entering an FST collective.
        error = f"{type(exc).__name__}: {exc}"
    peers = [(error, local)]
    if world > 1:
        peers = [None] * world
        torch.distributed.all_gather_object(peers, (error, local), group=process_group)
    for failure, _ in peers:
        if failure is not None:
            raise RuntimeError(f"Invalid native FST file plan: {failure}")
    # Every participant checks the same table, including ranks missing the file.
    placements = {}
    for _, peer in peers:
        for name, (header, _, placement, _) in peer.items():
            if placements.setdefault((name, header), placement) != placement:
                raise ValueError(f"Inconsistent expert placement across FST peers for {name}")
    shared, independent = [], []
    for name, path in sorted(by_name.items()):
        signature = local[name]
        collective = not signature[3] and all(peer.get(name) == signature for _, peer in peers)
        (shared if collective else independent).append(path)
    return shared, independent
