#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Opt-in two-host synthetic integration of the production startup RPC.

Run once on EACH host with private startup configs for N=2, P=2, world_size=2.
Each process holds its local allocations while two spawned workers use their
instance TCPStore, the real mTLS RPC, bootstrap validation and CUDA mappings.
No full model or latency acceptance is implied. The operator creates STOP only
AFTER both hosts report PASS and all consumer workers have exited.
"""
import argparse
import base64
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from uuid import uuid4

from rtp_llm.models_py.modules.dsv41.engram_exchange import encode
from rtp_llm.models_py.modules.dsv41.engram_startup import prepare_static_engram

HASH = 'sha256:' + 'b' * 64


def private_write(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encode(value))


def worker(rank, env, store_port, rows, device):
    import torch
    from torch.distributed import TCPStore
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import staged_package, _verify_samples, _load_native
    from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout
    from rtp_llm.models_py.modules.dsv41.engram_storage import FabricEngramStorage
    from rtp_llm.models_py.modules.dsv41.engram_shard_storage import CudaDriverOps

    class CountingOps(CudaDriverOps):
        imports = releases = maps = unmaps = reserves = frees = 0

        def _check(self, result, what):
            value = super()._check(result, what)
            counter = {'import_handle': 'imports', 'release': 'releases',
                       'map': 'maps', 'unmap': 'unmaps',
                       'address_reserve': 'reserves', 'address_free': 'frees'}.get(what)
            if counter:
                setattr(self, counter, getattr(self, counter) + 1)
            return value
    torch.cuda.set_device(device)
    native = _load_native()
    baseline = native.resource_counters()
    store = TCPStore('127.0.0.1', store_port, 2, rank == 0, timedelta(seconds=120))
    started = time.monotonic()
    prepare_static_engram(store, rank=rank, world_size=2, environ=env)
    startup_ms = (time.monotonic() - started) * 1000
    package = staged_package()
    if package is None:
        raise RuntimeError('startup exchange did not stage a package')
    # Test oracle only: prove graph replay does not require a live RPC listener.
    host, port = env['RTP_ENGRAM_COHORT'].split(',')[0].rsplit(':', 1)
    deadline = time.monotonic() + 10
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=0.2):
                pass
        except OSError:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('exchange listener still active after startup')
        time.sleep(0.05)
    layout = SimpleNamespace(layer_ids=(1,), num_embeddings=(rows,), head_dim=64)
    expected = ExpectedLayout(layers=tuple(zip(layout.layer_ids, layout.num_embeddings)),
                              head_dim=layout.head_dim, model_revision=package['model_revision'],
                              hash_abi=HASH)
    ops = CountingOps(device)
    storage = FabricEngramStorage.attach(package['raw'], expected_layout=expected,
        dataset_id=package['dataset_id'], generation=package['generation'],
        handles=package['capabilities'], device=device, native=native, shard_ops=ops)
    storage.verify(lambda views, verified: _verify_samples(views, verified, package['samples'], torch.device('cuda', device)))
    w, scale = storage.table_views(1)
    ids = torch.tensor(package['samples']['layers.1.engram.embed.weight']['row_ids'], device=device)
    # Values and ordering are checked against checkpoint samples before this.
    reference_w, reference_s = w.index_select(0, ids).clone(), scale.index_select(0, ids).clone()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        output_w, output_s = w.index_select(0, ids), scale.index_select(0, ids)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output_w, output_s = w.index_select(0, ids), scale.index_select(0, ids)
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(output_w, reference_w) and torch.equal(output_s, reference_s)
    graph.reset()
    del graph, output_w, output_s, reference_w, reference_s, w, scale, ids
    gc.collect()
    storage.close()
    assert ops.imports == ops.releases == ops.maps == ops.unmaps == 4
    assert ops.reserves == ops.frees == 2
    assert native.resource_counters() == baseline
    print(json.dumps(dict(status='WORKER_PASS', instance=env['RTP_ENGRAM_INSTANCE_INDEX'], rank=rank,
        startup_ms=round(startup_ms, 2), dataset_id=package['dataset_id'],
        rpc_listener_closed=True, graph_replays=100, counters_balanced=True,
        imports=ops.imports, releases=ops.releases, maps=ops.maps, unmaps=ops.unmaps,
        reserves=ops.reserves, frees=ops.frees)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--shard', required=True, type=int, choices=(0, 1))
    parser.add_argument('--exchange-port', required=True, type=int)
    parser.add_argument('--store-port', required=True, type=int)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--numa', type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch.multiprocessing as mp
    from safetensors.torch import save_file
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _load_native
    from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards
    from rtp_llm.models_py.modules.dsv41.engram_startup import (build_group, COHORT_ENV,
        COUNT_ENV, INDEX_ENV, REVISION_ENV, DESCRIPTORS_ENV, TIMEOUT_ENV, CHECKPOINT_ENV)
    root = args.workdir
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    endpoint = f'127.0.0.1:{args.exchange_port}'
    env = {
        COHORT_ENV: f'{endpoint},{endpoint}',
        INDEX_ENV: str(args.shard),
        COUNT_ENV: '2',
        REVISION_ENV: 'probe-revision',
        TIMEOUT_ENV: '120',
        CHECKPOINT_ENV: str(root / 'checkpoint'),
        DESCRIPTORS_ENV: str(root / 'shard.json'),
    }
    config = build_group(env, 2)
    member = next(m for m in config['group']['members'] if m['instance_id'] == config['instance_id'])
    assert member['world_size'] == 2 and len(member['shards']) == 1 and member['shards'][0]['index'] == args.shard
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    native = _load_native()
    baseline = native.resource_counters()
    gran = native.host_numa_granularity(args.numa, args.device)
    plan = plan_shards(2 * (gran // 2) - 7, 64, 2, granularity=gran)
    assert plan.shard_bytes <= 256 << 20
    owners, tensors = [], {}
    begin, end = plan.row_range(args.shard)
    for role, width, size, value in [('weight', 64, plan.weight_bytes, args.shard + 1),
                                    ('scale', 2, plan.scale_bytes, args.shard + 101)]:
        owner = native.Owner(size, args.numa, args.device)
        owners.append(owner)
        logical_bytes = (end - begin) * width
        hasher = hashlib.sha256()
        for offset in range(0, logical_bytes, 8 << 20):
            data = bytes([value]) * min(8 << 20, logical_bytes - offset)
            owner.load(data, offset)
            hasher.update(data)
        tensors[f'layers.1.engram.embed.{role}'] = dict(shape=[plan.total_rows, width], row_begin=begin,
            row_end=end, allocation_bytes=size, allocation_granularity=gran, logical_bytes=logical_bytes,
            allocation_id=str(uuid4()), sha256='sha256:' + hasher.hexdigest(),
            handle=base64.b64encode(owner.export_handle()).decode())
    host = member['shards'][0]['host_id']
    descriptor = dict(schema_version=2, shard=args.shard, shards=2, generation=config['group']['generation'],
        model_revision=config['group']['model_revision'], hash_abi=HASH, host_id=host, owner_id=host,
        owner_incarnation=str(uuid4()), tensors=tensors)
    path = Path(env[DESCRIPTORS_ENV])
    private_write(path, descriptor)
    # Only the coordinator needs the tiny synthetic reference checkpoint.
    if args.shard == 0:
        checkpoint = Path(env[CHECKPOINT_ENV])
        checkpoint.mkdir(mode=0o700, parents=True, exist_ok=True)
        weights = {}
        for role, width, base in [('weight', 64, 1), ('scale', 2, 101)]:
            table = torch.empty((plan.total_rows, width), dtype=torch.uint8)
            for i in range(2):
                b, e = plan.row_range(i)
                table[b:e].fill_(base + i)
            weights[f'layers.1.engram.embed.{role}'] = table
        save_file(weights, str(checkpoint / 'probe.safetensors'))
        private_write(checkpoint / 'model.safetensors.index.json', {'weight_map': {k: 'probe.safetensors' for k in weights}})
        del table, weights
    print(json.dumps(dict(status='HOLDER_READY', host=host, shard=args.shard, backing_bytes=plan.shard_bytes)), flush=True)
    mp.spawn(worker, args=(env, args.store_port, plan.total_rows, args.device), nprocs=2, join=True)
    print(json.dumps(dict(status='PASS', instance=config['instance_id'], consumers_exited=True)), flush=True)
    # Test-only owner retirement. Production cohort launcher must coordinate drain.
    while not (root / 'STOP').exists():
        time.sleep(0.2)
    for owner in reversed(owners):
        owner.close()
    assert native.resource_counters() == baseline
    print(json.dumps(dict(status='CLOSED', counters_balanced=True)), flush=True)


if __name__ == '__main__':
    main()
