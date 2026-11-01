# SPDX-License-Identifier: Apache-2.0
"""Backend startup adapter: existing intra-instance Store + temporary leader RPC.

Only called before engine construction. No lookup/kernel/Graph code calls this
module. Holders remain owned by the launcher's static cohort lifecycle.

The exchanged package stays in memory and is handed to the model attach through
``engram_bootstrap.stage_package``; serving never reads bundle files.
"""
from __future__ import annotations
import base64
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import ssl
import threading
import time
from uuid import NAMESPACE_URL, uuid5

from .engram_exchange import (Cohort, ExchangeError, ExchangeServer, decode, digest, encode,
                              remote_exchange, tls_context, validate_group)
from .engram_bootstrap import _json, _private_read, stage_package
from .engram_descriptors import build_manifest
from .engram_manifest import parse_manifest
from .engram_shard_config import validate_count

COHORT_ENV = 'RTP_ENGRAM_COHORT'
INDEX_ENV = 'RTP_ENGRAM_INSTANCE_INDEX'
REVISION_ENV = 'RTP_ENGRAM_MODEL_REVISION'
DESCRIPTORS_ENV = 'RTP_ENGRAM_LOCAL_DESCRIPTORS'
TIMEOUT_ENV = 'RTP_ENGRAM_COHORT_TIMEOUT'
TLS_DIR_ENV = 'RTP_ENGRAM_TLS_DIR'
CHECKPOINT_ENV = 'RTP_ENGRAM_CHECKPOINT'
COUNT_ENV = 'ENGRAM_SHARD_COUNT'
RANKS_ENV = 'RTP_ENGRAM_RANKS_PER_INSTANCE'
DEFAULT_TIMEOUT = 1800


def checkpoint_samples(checkpoint, manifest):
    """Hash every shard boundary from the trusted coordinator's checkpoint."""
    import torch
    from safetensors import safe_open
    root = Path(checkpoint).resolve(strict=True)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    result = {}
    for tensor in manifest.tensors:
        rows = sorted({0, tensor.shape[0] // 2, tensor.shape[0] - 1} |
                      {r for part in tensor.shards for r in (part.row_begin, part.row_end - 1)})
        relative = Path(index[tensor.name])
        if relative.is_absolute() or '..' in relative.parts:
            raise ExchangeError('UNSAFE_CHECKPOINT_PATH')
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ExchangeError('UNSAFE_CHECKPOINT_PATH')
        with safe_open(path, framework='pt', device='cpu') as source:
            view = source.get_slice(tensor.name)
            payload = b''.join(view[r:r + 1].contiguous().view(torch.uint8).numpy().tobytes() for r in rows)
        result[tensor.name] = dict(row_ids=rows, sha256='sha256:' + digest(payload))
    return result


def assemble_package(group, documents, checkpoint, samples=checkpoint_samples):
    p = sum(len(m['shards']) for m in group['members'])
    manifest = build_manifest(documents, p, group['generation'], group['model_revision'], documents[0]['hash_abi'])
    raw = encode(manifest)
    sample_values = samples(checkpoint, parse_manifest(raw))
    capabilities = {entry['allocation_id']: entry['handle'] for d in documents for entry in d['tensors'].values()}
    return dict(manifest=manifest, capabilities=capabilities, samples=sample_values)


def verify_package(group, package):
    """Reject identity/membership/handle mismatches before publishing to workers."""
    if type(package) is not dict or set(package) != {'manifest', 'capabilities', 'samples'}:
        raise ExchangeError('INVALID_PACKAGE')
    manifest = parse_manifest(encode(package['manifest']))
    if manifest.generation != group['generation'] or manifest.model_revision != group['model_revision']:
        raise ExchangeError('PACKAGE_IDENTITY_MISMATCH')
    expected = {s['index']: s['host_id'] for m in group['members'] for s in m['shards']}
    handles = package['capabilities']
    ids = set()
    owners = None
    for tensor in manifest.tensors:
        if len(tensor.shards) != len(expected):
            raise ExchangeError('PACKAGE_SHARD_COUNT_MISMATCH')
        current = [(s.owner_id, s.owner_incarnation) for s in tensor.shards]
        if [v[0] for v in current] != [expected[i] for i in range(len(expected))]:
            raise ExchangeError('PACKAGE_HOST_MISMATCH')
        if owners is not None and current != owners:
            raise ExchangeError('PACKAGE_OWNER_INCONSISTENT')
        owners = current
        ids.update(s.allocation_id for s in tensor.shards)
    if type(handles) is not dict or set(handles) != ids:
        raise ExchangeError('PACKAGE_CAPABILITIES_MISMATCH')
    for value in handles.values():
        if type(value) is not str or not 0 < len(value) <= 1024:
            raise ExchangeError('INVALID_CAPABILITY')
        try:
            if not base64.b64decode(value, validate=True):
                raise ValueError()
        except (ValueError, TypeError):
            raise ExchangeError('INVALID_CAPABILITY') from None
    if type(package['samples']) is not dict or set(package['samples']) != {t.name for t in manifest.tensors}:
        raise ExchangeError('PACKAGE_SAMPLES_MISMATCH')
    # Full layout/hash ABI and GPU content are still checked by model bootstrap.
    return manifest


def to_memory_package(group, package):
    """Validate the wire package and convert it to the in-memory attach form."""
    manifest = verify_package(group, package)
    return dict(
        raw=encode(package['manifest']),
        dataset_id=manifest.dataset_id,
        generation=manifest.generation,
        model_revision=manifest.model_revision,
        capabilities={aid: base64.b64decode(value, validate=True)
                      for aid, value in package['capabilities'].items()},
        samples=package['samples'],
    )


def offline_group(documents):
    """Synthetic single-rank cohort for offline probes over published shards."""
    ordered = sorted(documents, key=lambda item: item['shard'])
    return dict(
        generation=ordered[0]['generation'],
        model_revision=ordered[0]['model_revision'],
        members=[dict(instance_id=f'i{item["shard"]}', world_size=1,
                      shards=[dict(index=item['shard'], host_id=item['host_id'],
                                   publisher_rank=0)])
                 for item in ordered],
    )


def package_from_descriptors(paths, checkpoint, samples=checkpoint_samples):
    """Build an in-memory package from published shard descriptors (no bundle)."""
    documents = [_json(_private_read(Path(path), 1 << 20)) for path in paths]
    group = offline_group(documents)
    validate_group(group)
    package = assemble_package(group, documents, checkpoint, samples=samples)
    return to_memory_package(group, package)


def _wait(store, keys, deadline):
    # c10d Store.wait()/check() is collective: every rank must pass the same key
    # set. Ranks here wait on different keys (own config vs. the shared package),
    # so a collective wait deadlocks across the gang. Blocking get() is per-key.
    pending = list(keys)
    while pending:
        if time.monotonic() >= deadline:
            raise ExchangeError('INSTANCE_STARTUP_TIMEOUT')
        try:
            store.get(pending[0])
        except KeyError:
            # A local store answers immediately; retry until the peer publishes.
            time.sleep(0.05)
            continue
        pending.pop(0)


def _cohort(env):
    raw = env.get(COHORT_ENV)
    if raw is None:
        return None
    members = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.rpartition(':')
        if not host or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ExchangeError('INVALID_COHORT')
        members.append((host, int(port)))
    if not members:
        raise ExchangeError('INVALID_COHORT')
    return members


def _placement(env, name):
    raw = env.get(name)
    if raw is None:
        return 0
    if not raw.isdigit() or int(raw) > 4096:
        raise ExchangeError('INVALID_LOCAL_PLACEMENT')
    return int(raw)


def _descriptors(env, count):
    """Descriptors may come from files (harness) or be published in-process."""
    raw = env.get(DESCRIPTORS_ENV)
    if raw is None:
        return None
    paths = [item.strip() for item in raw.split(',') if item.strip()]
    if len(paths) != count:
        raise ExchangeError('LOCAL_DESCRIPTOR_COUNT_MISMATCH')
    for path in paths:
        if not Path(path).is_absolute():
            raise ExchangeError('DESCRIPTOR_PATH_MUST_BE_ABSOLUTE')
    return paths


def _tls(env, index, instances):
    directory = env.get(TLS_DIR_ENV)
    if not directory:
        return None
    root = Path(directory)
    if not root.is_absolute():
        raise ExchangeError('TLS_DIR_MUST_BE_ABSOLUTE')
    fingerprints = {}
    for member in range(instances):
        pem = root / f'member-{member}.pem'
        if not pem.is_file():
            raise ExchangeError('MISSING_MEMBER_CERTIFICATE')
        der = ssl.PEM_cert_to_DER_cert(pem.read_text())
        fingerprints[f'i{member}'] = hashlib.sha256(der).hexdigest()
    if len(set(fingerprints.values())) != len(fingerprints):
        raise ExchangeError('DISTINCT_MEMBER_CERTIFICATES_REQUIRED')
    return dict(ca_file=str(root / 'ca.pem'), cert_file=str(root / f'member-{index}.pem'),
                key_file=str(root / f'member-{index}.key'),
                instance_cert_sha256=fingerprints)


def build_group(env, world_size):
    """Derive cohort membership and shard ownership from the environment only."""
    cohort = _cohort(env)
    if cohort is None:
        return None
    raw_count = env.get(COUNT_ENV)
    if raw_count is None or not raw_count.isdigit():
        raise ExchangeError('STORAGE_HOST_COUNT_REQUIRED')
    p = validate_count(int(raw_count))
    instances = len(cohort)
    if p < instances:
        raise ExchangeError('STORAGE_HOST_COUNT_BELOW_INSTANCE_COUNT')
    revision = env.get(REVISION_ENV)
    if not revision:
        raise ExchangeError('MODEL_REVISION_REQUIRED')
    raw_index = env.get(INDEX_ENV, '0')
    if not raw_index.isdigit() or not 0 <= int(raw_index) < instances:
        raise ExchangeError('INVALID_INSTANCE_INDEX')
    index = int(raw_index)
    raw_ranks = env.get(RANKS_ENV)
    if raw_ranks is not None and (not raw_ranks.isdigit() or int(raw_ranks) <= 0):
        raise ExchangeError('INVALID_RANKS_PER_INSTANCE')
    generation = str(uuid5(NAMESPACE_URL, 'engram|' + revision + '|' +
                           ','.join(f'{host}:{port}' for host, port in cohort) + '|' + str(p)))
    # A cohort member is either a whole gang on its own host (prefill) or one
    # host of a gang shared by every member (decode); RANKS_ENV states which.
    local_world_size = world_size if raw_ranks is None else int(raw_ranks)
    if local_world_size != world_size and local_world_size * instances != world_size:
        raise ExchangeError('WORLD_SIZE_NOT_DIVISIBLE_BY_INSTANCES')
    base, extra = divmod(p, instances)
    members = []
    for member_index in range(instances):
        start = member_index * base + min(member_index, extra)
        count = base + (1 if member_index < extra else 0)
        if local_world_size % count:
            raise ExchangeError('WORLD_SIZE_NOT_DIVISIBLE_BY_LOCAL_HOSTS')
        publishers = local_world_size // count
        host = cohort[member_index][0]
        shards = [dict(index=start + offset, host_id=f'{host}#{start + offset}',
                       publisher_rank=offset * publishers) for offset in range(count)]
        members.append(dict(instance_id=f'i{member_index}', world_size=local_world_size, shards=shards))
    group = dict(generation=generation, model_revision=revision, members=members)
    validate_group(group)
    local = members[index]
    checkpoint = env.get(CHECKPOINT_ENV) or env.get('CHECKPOINT_PATH')
    if not checkpoint:
        raise ExchangeError('CHECKPOINT_REQUIRED')
    raw_timeout = env.get(TIMEOUT_ENV)
    timeout = DEFAULT_TIMEOUT if raw_timeout is None else float(raw_timeout)
    if not 0 < timeout <= 86400:
        raise ExchangeError('INVALID_TIMEOUT')
    return dict(
        group=group,
        instance_id=local['instance_id'],
        index=index,
        checkpoint=checkpoint,
        coordinator=dict(host=cohort[0][0], port=cohort[0][1]),
        timeout_seconds=timeout,
        local_shards=local['shards'],
    )


def build_config(env, world_size):
    """Add the local-only inputs (descriptor paths, optional mTLS material)."""
    config = build_group(env, world_size)
    if config is None:
        return None
    config['tls'] = _tls(env, config['index'], len(config['group']['members']))
    config['local_descriptors'] = _descriptors(env, len(config['local_shards']))
    config['local_device'] = _placement(env, 'RTP_ENGRAM_LOCAL_DEVICE')
    config['local_numa'] = _placement(env, 'RTP_ENGRAM_LOCAL_NUMA')
    config['release_page_cache'] = env.get('RTP_ENGRAM_RELEASE_INPUT_PAGE_CACHE') == '1'
    return config


def exchange_leaders(config, documents, deadline, build=None):
    group = config['group']
    group_digest = validate_group(group)
    ident = config['instance_id']
    coordinator = group['members'][0]['instance_id']
    builder = build or (lambda docs: assemble_package(group, docs, config['checkpoint']))
    verifier = lambda package: verify_package(group, package)
    if len(group['members']) == 1:
        cohort = Cohort(group, builder, max(0.001, deadline - time.monotonic()))
        return decode(cohort.submit(ident, group_digest, documents))['package']
    address = config['coordinator']
    if (type(address) is not dict or set(address) != {'host', 'port'}
            or type(address['port']) is not int or not 1 <= address['port'] <= 65535):
        raise ExchangeError('INVALID_COORDINATOR')
    tls = config['tls']
    context = tls_context(tls) if tls else None
    server_context = tls_context(tls, server=True) if tls else None
    fingerprints = tls['instance_cert_sha256'] if tls else {}
    if ident != coordinator:
        return remote_exchange((address['host'], address['port']), context, ident, group_digest,
                               documents, deadline, verifier,
                               server_fingerprint=fingerprints.get(coordinator))
    cohort = Cohort(group, builder, max(0.001, deadline - time.monotonic()))
    server = ExchangeServer((address['host'], address['port']), cohort, server_context, fingerprints)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
    thread.start()
    try:
        payload = cohort.submit(ident, group_digest, documents)
        package = decode(payload)['package']
        verifier(package)
        cohort.ack(ident, digest(payload))
        cohort.wait_acknowledged()
        return package
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def prepare_static_engram(store, *, rank, world_size, environ=None, leader_exchange=exchange_leaders):
    """All ranks enter once. Only the member-local rank 0 opens cross-instance connections."""
    env = os.environ if environ is None else environ
    config = build_config(env, world_size)
    if config is None:
        return None
    group = config['group']
    group_digest = validate_group(group)
    members = {m['instance_id']: m for m in group['members']}
    member = members[config['instance_id']]
    local_world_size = member['world_size']
    if local_world_size == world_size:
        # Whole gang per member: ranks are already member-local.
        local_rank = rank
    elif local_world_size * len(group['members']) == world_size:
        local_rank = rank - config['index'] * local_world_size
    else:
        raise ExchangeError('INSTANCE_WORLD_SIZE_MISMATCH')
    if not 0 <= local_rank < local_world_size:
        raise ExchangeError('INSTANCE_WORLD_SIZE_MISMATCH')
    p = sum(len(m['shards']) for m in group['members'])
    timeout = config['timeout_seconds']
    deadline = time.monotonic() + timeout
    if store is None:
        if local_world_size != 1:
            raise ExchangeError('INSTANCE_STORE_REQUIRED')
        from torch.distributed import HashStore
        store = HashStore()
    prefix = 'engram/' + group['generation'] + '/' + config['instance_id'] + '/'
    # Each worker must agree on the public contract, even if its config differs.
    store.set(prefix + f'config/{local_rank}', group_digest.encode())
    if local_rank == 0:
        config_keys = [prefix + f'config/{i}' for i in range(local_world_size)]
        _wait(store, config_keys, deadline)
        if any(store.get(k) != group_digest.encode() for k in config_keys):
            raise ExchangeError('INSTANCE_GROUP_CONFIG_MISMATCH')
    descriptors = config['local_descriptors']
    for offset, shard in enumerate(member['shards']):
        if shard['publisher_rank'] != local_rank:
            continue
        if descriptors is None:
            # This member owns the shard: allocate it here and describe it.
            from rtp_llm.models_py.modules.dsv41.engram_publish import publish_shard
            document = publish_shard(
                config['checkpoint'], group['model_revision'], member['host_id'],
                group['generation'], shard['index'], p,
                device=config['local_device'], numa=config['local_numa'],
                release_input_page_cache=config['release_page_cache'])
            payload = encode(document)
        else:
            descriptor = Path(descriptors[offset])
            while not descriptor.exists():
                if time.monotonic() >= deadline:
                    raise ExchangeError('LOCAL_SHARD_NOT_READY')
                time.sleep(0.05)
            payload = _private_read(descriptor, 1 << 20)
        store.set(prefix + f'shard/{shard["index"]}', payload)
    if local_rank == 0:
        keys = [prefix + f'shard/{s["index"]}' for s in member['shards']]
        _wait(store, keys, deadline)
        documents = [decode(store.get(key)) for key in keys]
        package = leader_exchange(config, documents, deadline)
        verify_package(group, package)
        store.set(prefix + 'package', encode(package))
    _wait(store, [prefix + 'package'], deadline)
    package = decode(store.get(prefix + 'package'))
    stage_package(to_memory_package(group, package))
    env[COUNT_ENV] = str(p)
    store.set(prefix + f'received/{local_rank}', b'1')
    if local_rank == 0:
        _wait(store, [prefix + f'received/{i}' for i in range(local_world_size)], deadline)
        # Every worker holds the package in memory. No serving-time Store reads.
        for key in ([prefix + 'package'] + keys + config_keys +
                    [prefix + f'received/{i}' for i in range(local_world_size)]):
            store.delete_key(key)
    return None
