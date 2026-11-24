"""Select collective vs rank-local FST transport before opening payloads."""

import hashlib
import os

import torch
from safetensors import safe_open


def plan_fst_files(files, root, process_group, model_filter, is_fixed_host, catalog=None):
    """Only share files whose collective tensor schedule agrees on every peer.

    Native FST brokers a file sequence, not a list of tensor names per rank.
    Different files/keys at the same collective position can silently exchange
    equally shaped bytes. Keep such shards on the independent native range
    reader. A shared file containing fixed-host weights also takes that route:
    even ignoring a key cannot prevent the FST file copier from staging its bytes.
    """
    by_name = {os.path.relpath(path, root): path for path in files}
    world = process_group.size()
    if world == 1 and is_fixed_host is None and catalog is None:
        return list(files), []
    local = {}
    error = None
    try:
        for name, path in by_name.items():
            if model_filter is None and is_fixed_host is None and catalog is None:
                local[name] = ("all", False)
                continue
            selected = []
            host = False
            with safe_open(path, framework="pt", device="cpu") as shard:
                keys = shard.keys()
                if catalog is not None and len(catalog.shard_keys(path, keys)) != len(keys):
                    # A whole-file transport could stage unindexed weights or
                    # cross-wire duplicate names from unrelated physical tails.
                    local[name] = ("indexed subset", True)
                    continue
                if model_filter is None and is_fixed_host is None:
                    local[name] = ("all", False)
                    continue
                for key in keys:
                    if model_filter is None or model_filter(key):
                        selected.append(key)
                    if is_fixed_host is not None and is_fixed_host(key):
                        host = True
            if len(selected) == len(keys):
                digest = "all"
            else:
                h = hashlib.sha256()
                for key in sorted(selected):
                    raw = key.encode()
                    h.update(len(raw).to_bytes(8, "little"))
                    h.update(raw)
                digest = h.digest()
            local[name] = (digest, host)
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
    shared, independent = [], []
    for name, path in sorted(by_name.items()):
        signature, host = local[name]
        collective = not host and all(
            peer.get(name) == (signature, False) for _, peer in peers
        )
        (shared if collective else independent).append(path)
    return shared, independent
