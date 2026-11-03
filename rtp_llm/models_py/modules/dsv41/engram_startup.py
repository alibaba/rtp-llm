# SPDX-License-Identifier: Apache-2.0
"""Backend startup adapter: existing intra-instance Store + temporary leader RPC.

Only called before engine construction. No lookup/kernel/Graph code calls this
module. Backing is retained by the publishing RTP worker until process exit.

The exchanged package stays in memory and is handed to the model attach through
``engram_bootstrap.stage_package``; serving never reads bundle files.
"""
from __future__ import annotations
import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import ssl
import threading
import time
from uuid import NAMESPACE_URL, UUID, uuid5
from datetime import timedelta

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
GENERATION_ENV = 'RTP_ENGRAM_GENERATION'
PORT_ENV = 'RTP_ENGRAM_COHORT_PORT'
GANGINFO_ENV = 'RTP_ENGRAM_GANGINFO_DIR'
# Only the first member listens, only during startup; everything else is outbound.
DEFAULT_COHORT_PORT = 29483
DEFAULT_TIMEOUT = 1800
# Where the platform drops the topology, and the Store prefix used to derive it
# from a gang that already spans the hosts.
GANGINFO_DIR = '/etc/ganginfo'
TOPOLOGY_PREFIX = 'engram/topology/'


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
    # TCPStore.wait is a key-availability operation, not a rank collective.
    # Use its per-call timeout: get() otherwise inherits the Store's (possibly
    # very long) timeout. Never mutate the shared Store's global timeout.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ExchangeError('INSTANCE_STARTUP_TIMEOUT')
    try:
        store.wait(keys, timedelta(seconds=remaining))
    except RuntimeError:
        raise ExchangeError('INSTANCE_STORE_WAIT_FAILED_OR_TIMEOUT') from None


def _parse_members(raw, default_port):
    """Turn one "host[,host...]" string into validated (host, port) members."""
    if default_port is not None and (not default_port.isdigit() or not 1 <= int(default_port) <= 65535):
        raise ExchangeError('INVALID_COHORT')
    members = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        # A bare host takes the shared cohort port; "host:port" still works for
        # a deployment that has to spread the exchange servers over ports.
        host, _, port = item.rpartition(':')
        if not host:
            host, port = item, ''
        if not port:
            port = default_port
        if not port:
            port = str(DEFAULT_COHORT_PORT)
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ExchangeError('INVALID_COHORT')
        members.append((host, int(port)))
    if not members:
        raise ExchangeError('INVALID_COHORT')
    if len({host for host, _ in members}) != len(members):
        raise ExchangeError('DISTINCT_STORAGE_HOST_ADDRESSES_REQUIRED')
    return members


def _cohort(env):
    raw = env.get(COHORT_ENV)
    if raw is None:
        return None
    return _parse_members(raw, env.get(PORT_ENV))


def _ganginfo_entries(root):
    """Member list from the directory the platform writes the topology into."""
    for name in ('cohort', 'hosts', 'members'):
        path = root / name
        if path.is_file():
            lines = [line.split('#', 1)[0].strip() for line in path.read_text().splitlines()]
            return ','.join(line for line in lines if line) or None
    # Fall back to one file per part, ordered by the number in the name.
    numbered = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        for prefix in ('rank-', 'rank', 'part-', 'part', 'node-', 'node'):
            suffix = path.name[len(prefix):]
            if path.name.startswith(prefix) and suffix.isdigit():
                numbered.append((int(suffix), path))
                break
    items = []
    for _, path in sorted(numbered):
        content = path.read_text().splitlines()
        if content and content[0].strip():
            items.append(content[0].strip())
    return ','.join(items) or None


def _ganginfo_cohort(env):
    root = Path(env.get(GANGINFO_ENV) or GANGINFO_DIR)
    if not root.is_dir():
        return None
    try:
        raw = _ganginfo_entries(root)
    except OSError as error:
        # A directory the deployment points at but cannot read is a
        # misconfiguration; silently serving with Engram off would hide it.
        raise ExchangeError('INVALID_GANGINFO') from error
    if raw is None:
        return None
    return _parse_members(raw, env.get(PORT_ENV))


def _gang_cohort(env, store, rank, world_size):
    """One gang spanning several hosts already knows its own parts.

    Consecutive ranks on the same address form one part, so both the number of
    parts and the ranks each member owns fall out of the gang itself. Only used
    when the deployment asked for Engram without giving an address list: a
    single-host gang (prefill's independent instances) still needs that list,
    because it cannot see the other hosts.
    """
    raw_count = env.get(COUNT_ENV)
    raw_ranks = env.get(RANKS_ENV)
    if raw_count is None and raw_ranks is None:
        return None, None
    if store is None:
        raise ExchangeError('INSTANCE_STORE_REQUIRED')
    timeout = DEFAULT_TIMEOUT if env.get(TIMEOUT_ENV) is None else float(env[TIMEOUT_ENV])
    if not 0 < timeout <= 86400:
        raise ExchangeError('INVALID_TIMEOUT')
    local = sorted(address for address in _local_addresses() if not address.startswith('127.'))
    if not local:
        raise ExchangeError('INSTANCE_INDEX_REQUIRED')
    key = TOPOLOGY_PREFIX + 'host/'
    store.set(key + str(rank), ','.join(local).encode())
    keys = [key + str(i) for i in range(world_size)]
    _wait(store, keys, time.monotonic() + timeout)
    blocks = []
    for i in range(world_size):
        address = store.get(key + str(i)).decode()
        if blocks and blocks[-1][0] == address:
            blocks[-1][1] += 1
        else:
            blocks.append([address, 1])
    if len(blocks) < 2:
        # One address for the whole gang: the other parts are other gangs.
        raise ExchangeError('COHORT_REQUIRED')
    sizes = {size for _, size in blocks}
    if len(sizes) != 1:
        raise ExchangeError('UNEVEN_PART_SIZE')
    per_instance = sizes.pop()
    if raw_ranks is not None and (not raw_ranks.isdigit() or int(raw_ranks) != per_instance):
        raise ExchangeError('RANKS_PER_INSTANCE_MISMATCH')
    if raw_count is not None and (not raw_count.isdigit() or int(raw_count) != len(blocks)):
        raise ExchangeError('STORAGE_HOST_COUNT_MUST_MATCH_COHORT')
    hosts = [sorted(address.split(','))[0] for address, _ in blocks]
    return _parse_members(','.join(hosts), env.get(PORT_ENV)), per_instance


def _local_addresses():
    """IPv4 addresses this process can be reached on, for index discovery."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    return found


def _checkpoint_revision(checkpoint):
    """Identity of the weights this host serves, read from the checkpoint itself.

    Every member has the same checkpoint, so hashing its index gives the whole
    cohort the same namespace without anyone having to name a revision.
    """
    root = Path(checkpoint)
    for name in ('model.safetensors.index.json', 'config.json'):
        path = root / name
        if path.is_file():
            return 'ckpt@' + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    raise ExchangeError('MODEL_REVISION_REQUIRED')


def _instance_index(env, cohort):
    raw = env.get(INDEX_ENV)
    if raw is not None:
        if not raw.isdigit() or not 0 <= int(raw) < len(cohort):
            raise ExchangeError('INVALID_INSTANCE_INDEX')
        return int(raw)
    # The platform only has to hand over the address list: a host that appears
    # in it exactly once knows which shard it is.
    local = _local_addresses()
    matches = [i for i, (host, _) in enumerate(cohort) if host in local]
    if not matches:
        raise ExchangeError('INSTANCE_INDEX_REQUIRED')
    if len(matches) > 1:
        raise ExchangeError('AMBIGUOUS_LOCAL_ADDRESS')
    return matches[0]


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


def build_group(env, world_size, store=None, rank=0):
    """Derive cohort membership and shard ownership from the environment only.

    The address list may come from the environment, from the topology directory
    the platform writes, or from the gang itself when one gang spans the hosts.
    """
    cohort = _cohort(env)
    if cohort is None:
        cohort = _ganginfo_cohort(env)
    discovered = None
    if cohort is None:
        cohort, discovered = _gang_cohort(env, store, rank, world_size)
    if cohort is None:
        return None
    instances = len(cohort)
    # One shard per storage host: the cohort list is the topology, so the count
    # is only needed as a cross-check when a launcher still sets it.
    raw_count = env.get(COUNT_ENV)
    if raw_count is None:
        p = validate_count(instances)
    elif raw_count.isdigit():
        p = validate_count(int(raw_count))
        if p != instances:
            raise ExchangeError('STORAGE_HOST_COUNT_MUST_MATCH_COHORT')
    else:
        raise ExchangeError('STORAGE_HOST_COUNT_REQUIRED')
    checkpoint = env.get(CHECKPOINT_ENV) or env.get('CHECKPOINT_PATH')
    if not checkpoint:
        raise ExchangeError('CHECKPOINT_REQUIRED')
    revision = env.get(REVISION_ENV) or _checkpoint_revision(checkpoint)
    index = _instance_index(env, cohort)
    raw_ranks = env.get(RANKS_ENV)
    if raw_ranks is not None and (not raw_ranks.isdigit() or int(raw_ranks) <= 0):
        raise ExchangeError('INVALID_RANKS_PER_INSTANCE')
    # Namespace for this cohort's store keys. Derived from the deployment
    # identity by default, so the platform only has to supply the address list;
    # an explicit UUID can be passed to distinguish restarts in diagnostics.
    # Partial restarts are unsupported either way: a member that comes back
    # alone would read handles published by an allocation that no longer exists.
    generation = env.get(GENERATION_ENV)
    if generation is None:
        generation = str(uuid5(NAMESPACE_URL, 'engram|' + revision + '|' +
                               ','.join(f'{host}:{port}' for host, port in cohort) + '|' + str(p)))
    else:
        try:
            if str(UUID(generation)) != generation:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise ExchangeError('INVALID_GENERATION') from None
    # A cohort member is either a whole gang on its own host (prefill) or one
    # host of a gang shared by every member (decode). A list the platform gave
    # us describes prefill, so the member owns the whole gang; a gang that
    # reported its own split tells us how many ranks each member owns.
    if raw_ranks is not None and discovered is not None and int(raw_ranks) != discovered:
        raise ExchangeError('RANKS_PER_INSTANCE_MISMATCH')
    if raw_ranks is not None:
        local_world_size = int(raw_ranks)
    else:
        local_world_size = world_size if discovered is None else discovered
    if local_world_size != world_size and local_world_size * instances != world_size:
        raise ExchangeError('WORLD_SIZE_NOT_DIVISIBLE_BY_INSTANCES')
    members = [dict(instance_id=f'i{i}', world_size=local_world_size,
                    shards=[dict(index=i, host_id=host, publisher_rank=0)])
               for i, (host, _) in enumerate(cohort)]
    group = dict(generation=generation, model_revision=revision, members=members)
    validate_group(group)
    local = members[index]
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


def build_config(env, world_size, store=None, rank=0):
    """Add the local-only inputs (descriptor paths, optional mTLS material)."""
    config = build_group(env, world_size, store, rank)
    if config is None:
        return None
    config['tls'] = _tls(env, config['index'], len(config['group']['members']))
    config['local_descriptors'] = _descriptors(env, len(config['local_shards']))
    config['local_device'] = _placement(env, 'RTP_ENGRAM_LOCAL_DEVICE')
    config['local_numa'] = _placement(env, 'RTP_ENGRAM_LOCAL_NUMA')
    config['release_page_cache'] = env.get('RTP_ENGRAM_RELEASE_INPUT_PAGE_CACHE') == '1'
    from .engram_capacity import capacity_from_environ
    config['max_allocation_bytes'], config['budget_bytes'] = capacity_from_environ(env)
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
    config = build_config(env, world_size, store, rank)
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
            if env.get('RTP_ENGRAM_LOCAL_DEVICE') is None or env.get('RTP_ENGRAM_LOCAL_NUMA') is None:
                logging.getLogger(__name__).warning(
                    'RTP_ENGRAM_LOCAL_DEVICE/RTP_ENGRAM_LOCAL_NUMA unset; the self-published'
                    ' shard allocates on device %d / NUMA %d by default',
                    config['local_device'], config['local_numa'])
            document = publish_shard(
                config['checkpoint'], group['model_revision'], shard['host_id'],
                group['generation'], shard['index'], p,
                device=config['local_device'], numa=config['local_numa'],
                release_input_page_cache=config['release_page_cache'], deadline=deadline,
                max_allocation_bytes=config['max_allocation_bytes'], budget_bytes=config['budget_bytes'])
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
