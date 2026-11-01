#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a static distributed Engram launch plan without remote deployment.

With exchange configuration, existing per-instance Store and temporary mTLS RPC
collect handles automatically during BackendManager startup. Without exchange,
the explicit offline bundle workflow remains available for diagnostics.
Start each holder on its declared host; retain all holders until every model
worker and CUDA graph in the fixed cohort has exited. No live reshard/election.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from uuid import UUID

from rtp_llm.models_py.modules.dsv41.engram_shard_config import validate_count


def render_plan(config):
    """Storage host count P and the instance/worker topology are independent."""
    from rtp_llm.models_py.modules.dsv41.engram_capacity import validate_capacity
    allocation, total = validate_capacity(config.get('max_allocation_bytes', 128 << 30),
                                          config.get('max_total_allocation_bytes', 256 << 30))
    p = validate_count(config['storage_host_count'])
    hosts = config['storage_hosts']
    if type(hosts) is not list or len(hosts) != p:
        raise ValueError('storage_hosts length must equal P')
    ids = [h['host_id'] for h in hosts]
    if any(type(i) is not str or not i.strip() or len(i) > 256 for i in ids) or len(set(ids)) != p:
        raise ValueError('P distinct physical storage host IDs required')
    generation = config['generation']
    if str(UUID(generation)) != generation:
        raise ValueError('canonical generation UUID required')
    for key in ('checkpoint', 'publication_root'):
        if type(config[key]) is not str or not Path(config[key]).is_absolute():
            raise ValueError('checkpoint and publication_root must be absolute')
    if type(config['model_revision']) is not str or not config['model_revision'].strip():
        raise ValueError('model revision required')
    revision = config['model_revision']
    root = str(Path(config['publication_root']) / generation)
    framework = 'tools.engram_sidecar.framework.'
    shared = ['--checkpoint', config['checkpoint'], '--model-revision', revision,
              '--generation', generation, '--shards', str(p)]
    holders = []
    for index, host in enumerate(hosts):
        for key in ('device', 'numa'):
            if type(host[key]) is not int or host[key] < 0:
                raise ValueError('local device and NUMA ordinals must be nonnegative integers')
        command = ['python3', '-m', framework + 'sharded_checkpoint_owner', *shared,
                   '--host-id', host['host_id'], '--shard', str(index),
                   '--publication', f'{root}/shard-{index}',
                   '--device', str(host['device']), '--numa', str(host['numa']),
                   '--max-allocation-bytes', str(allocation), '--budget-bytes', str(total)]
        holders.append(dict(host_id=host['host_id'], shard_index=index, argv=command))
    instances = config['instances']
    if type(instances) is not list or not instances:
        raise ValueError('at least one instance required')
    n = config.get('instance_count', len(instances))
    if type(n) is not int or n < 1 or n != len(instances):
        raise ValueError('instance_count N must equal the complete static instance list')
    seen, consumers = set(), []
    for instance in instances:
        identifier = instance['instance_id']
        workers = instance['hosts']
        if type(identifier) is not str or not identifier or identifier in seen:
            raise ValueError('instance IDs must be nonempty and distinct')
        seen.add(identifier)
        if (type(workers) is not list or not workers
                or any(type(h) is not str or not h.strip() for h in workers)
                or len(set(workers)) != len(workers)):
            raise ValueError('each instance needs distinct consumer host IDs')
        consumers.append(dict(instance_id=identifier, hosts=workers,
            env={'RTP_ENGRAM_MAX_ALLOCATION_BYTES': str(allocation),
                 'RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES': str(total),
                 'ENGRAM_SHARD_COUNT': str(p)}))
    # Every declared instance participates in holding backing. This is a fixed
    # cohort, not an expandable consumer list; joining/removing requires a new
    # generation. One holder per physical storage host is the current contract.
    assignments = {identifier: [] for identifier in seen}
    for host, holder in zip(hosts, holders):
        candidates = [instance['instance_id'] for instance in instances
                      if host['host_id'] in instance['hosts']]
        owner = host.get('instance_id')
        if owner is None:
            if len(candidates) != 1:
                raise ValueError('storage host must belong to exactly one instance, or specify instance_id')
            owner = candidates[0]
        if owner not in candidates:
            raise ValueError('storage holder instance must contain its physical host')
        holder['instance_id'] = owner
        assignments[owner].append(holder['shard_index'])
    if any(not indices for indices in assignments.values()):
        raise ValueError('every static instance must hold at least one shard; consumer-only instances are not supported')
    for consumer in consumers:
        consumer['held_shards'] = assignments[consumer['instance_id']]
    result = dict(instance_count=n, membership='static', storage_host_count=p,
                  generation=generation, holders=holders, consumers=consumers,
                  lifecycle='all N instances start their assigned P holders -> '
                            'existing instance Store -> one RPC connection per remote '
                            'representative -> all-member ACK -> model attach')
    if config.get('exchange') is None:
        raise ValueError('exchange configuration is required; offline bundles were removed')
    exchange = config['exchange']
    coordinator = exchange.get('coordinator')
    if type(coordinator) is not dict or set(coordinator) != {'host', 'port'}:
        raise ValueError('exchange.coordinator must be {host, port}')
    endpoints = []
    for instance in instances:
        ip = instance.get('ip') or instance['hosts'][0]
        endpoints.append(f'{ip}:{coordinator["port"]}')
    cohort = ','.join(endpoints)
    for member_index, consumer in enumerate(consumers):
        descriptors = ','.join(f'{root}/shard-{index}/shard.json'
                               for index in consumer['held_shards'])
        env = consumer['env']
        env.pop('RTP_ENGRAM_EXTERNAL_BUNDLE', None)
        env.update({
            'RTP_ENGRAM_COHORT': cohort,
            'RTP_ENGRAM_INSTANCE_INDEX': str(member_index),
            'ENGRAM_SHARD_COUNT': str(p),
            'RTP_ENGRAM_MODEL_REVISION': revision,
            'RTP_ENGRAM_LOCAL_DESCRIPTORS': descriptors,
            'RTP_ENGRAM_COHORT_TIMEOUT': str(exchange.get('timeout_seconds', 1800)),
            'RTP_ENGRAM_CHECKPOINT': config['checkpoint'],
        })
        if exchange.get('tls_dir'):
            env['RTP_ENGRAM_TLS_DIR'] = exchange['tls_dir']
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, help='Write the private launch plan for launcher distribution')
    args = parser.parse_args()
    plan = render_plan(json.loads(args.config.read_text()))
    if args.output_directory:
        import os
        destination = args.output_directory.resolve()
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
        fd = os.open(destination / 'launch-plan.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(plan, stream, indent=2)
        print(destination / 'launch-plan.json')
    else:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
