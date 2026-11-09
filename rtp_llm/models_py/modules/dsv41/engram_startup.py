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
from uuid import NAMESPACE_URL, uuid5
from datetime import timedelta

from .engram_exchange import (Cohort, ExchangeError, ExchangeServer, decode, digest, encode,
                              remote_exchange, tls_context, validate_group)
from .engram_bootstrap import stage_package
from .engram_descriptors import build_manifest
from .engram_ganginfo import (ANNOTATION_ENV, DEFAULT_ANNOTATION, GangInfoError,
                              family, read_parts)
from .engram_manifest import parse_manifest
from .engram_shard_config import shared_storage_requested, validate_count

COHORT_ENV = 'RTP_ENGRAM_COHORT'
INDEX_ENV = 'RTP_ENGRAM_INSTANCE_INDEX'
REVISION_ENV = 'RTP_ENGRAM_MODEL_REVISION'
TIMEOUT_ENV = 'RTP_ENGRAM_COHORT_TIMEOUT'
TLS_DIR_ENV = 'RTP_ENGRAM_TLS_DIR'
CHECKPOINT_ENV = 'RTP_ENGRAM_CHECKPOINT'
COUNT_ENV = 'ENGRAM_SHARD_COUNT'
RANKS_ENV = 'RTP_ENGRAM_RANKS_PER_INSTANCE'
PORT_ENV = 'RTP_ENGRAM_COHORT_PORT'
GANGINFO_ENV = 'RTP_ENGRAM_GANGINFO_DIR'
# Only the first member listens, only during startup; everything else is outbound.
DEFAULT_COHORT_PORT = 29483
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
        # Only the first address listens. Its explicit port overrides the
        # shared port; non-coordinator ports never affect the group identity.
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
    directory = env.get(GANGINFO_ENV)
    if directory is None:
        return None
    root = Path(directory)
    if not root.is_dir():
        raise ExchangeError('INVALID_GANGINFO')
    try:
        raw = _ganginfo_entries(root)
    except OSError as error:
        # A directory the deployment points at but cannot read is a
        # misconfiguration; silently serving with Engram off would hide it.
        raise ExchangeError('INVALID_GANGINFO') from error
    if raw is None:
        raise ExchangeError('EMPTY_GANGINFO')
    return _parse_members(raw, env.get(PORT_ENV))


def _annotation_cohort(env, parts):
    """Storage hosts from the platform's gang annotation, when unambiguous.

    Only a one-host instance can be read from addresses alone: every listed pod
    is then one storage host.  A multi-host instance keeps the existing
    derivation (the gang itself is the cohort), because the annotation does not
    say how several multi-host instances group, and guessing would silently
    change an existing per-instance layout.

    TODO(engram): multi-host instances sharing one storage cohort are out of
    scope for now.  Supporting them needs the platform to publish the
    pod-to-instance grouping (or an explicit index), not just a flat address
    list; until then such a gang keeps one shard per host of its own instance.
    """
    path = env.get(ANNOTATION_ENV)
    try:
        entries = read_parts(path or DEFAULT_ANNOTATION)
    except GangInfoError:
        # Deployments without platform annotations (manual containers) fall
        # through to the explicit inputs; a path the operator set is a
        # misconfiguration and must not be ignored.
        if path is None:
            return None
        raise ExchangeError('INVALID_GANGINFO') from None
    if parts is None or len(parts) != 1:
        return None
    local = [entry for entry in entries if entry[1] == parts[0]]
    if len(local) != 1:
        raise ExchangeError('LOCAL_HOST_NOT_IN_GANGINFO')
    name = family(local[0][0])
    listed = entries if name is None else [e for e in entries if family(e[0]) == name]
    default_port = env.get(PORT_ENV)
    if default_port is not None and (not default_port.isdigit() or not 1 <= int(default_port) <= 65535):
        raise ExchangeError('INVALID_COHORT')
    port = int(default_port) if default_port else DEFAULT_COHORT_PORT
    return [(host, explicit or port) for _, host, explicit in listed]


def _gang_parts(gang_hosts, world_size):
    """Use the completed DistributedServer topology, never rediscover NICs."""
    if gang_hosts is None:
        return None, None
    if len(gang_hosts) != world_size or any(type(h) is not str or not h for h in gang_hosts):
        raise ExchangeError('INVALID_GANG_TOPOLOGY')
    blocks = []
    for host in gang_hosts:
        if blocks and blocks[-1][0] == host:
            blocks[-1][1] += 1
        else:
            blocks.append([host, 1])
    if len({host for host, _ in blocks}) != len(blocks):
        raise ExchangeError('NONCONTIGUOUS_GANG_HOST')
    if len({size for _, size in blocks}) != 1:
        raise ExchangeError('UNEVEN_PART_SIZE')
    return [host for host, _ in blocks], blocks[0][1]


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
    """Checkpoint metadata namespace, NOT a weight-content checksum.

    Equal indices do not prove equal tensor bytes. Publication hashes and GPU
    sample validation remain separate; deployments must use one immutable model.
    """
    root = Path(checkpoint)
    for name in ('model.safetensors.index.json', 'config.json'):
        path = root / name
        if path.is_file():
            return 'ckpt@' + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    raise ExchangeError('MODEL_REVISION_REQUIRED')


def _instance_index(env, cohort, local_host=None):
    raw = env.get(INDEX_ENV)
    if raw is not None:
        if not raw.isdigit() or not 0 <= int(raw) < len(cohort):
            raise ExchangeError('INVALID_INSTANCE_INDEX')
        if local_host is not None and cohort[int(raw)][0] != local_host:
            raise ExchangeError('INSTANCE_INDEX_HOST_MISMATCH')
        return int(raw)
    # The platform only has to hand over the address list: a host that appears
    # in it exactly once knows which shard it is.
    local = {local_host} if local_host is not None else _local_addresses()
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


def build_group(env, world_size, *, rank=0, gang_hosts=None, checkpoint=None):
    """Derive membership from explicit sharing inputs and the completed gang.

    The address list may come from the environment, from the topology directory
    the platform writes, from the gang annotation, or from the gang itself when
    one gang spans the hosts.
    """
    # Ordinary deployments must not opt in merely because /etc/ganginfo exists.
    if not shared_storage_requested(env):
        return None
    if type(world_size) is not int or world_size <= 0 or not 0 <= rank < world_size:
        raise ExchangeError('INVALID_WORLD_SIZE')
    parts, discovered = _gang_parts(gang_hosts, world_size)
    source = 'cohort'
    cohort = _cohort(env)
    if cohort is None:
        source = 'ganginfo'
        cohort = _ganginfo_cohort(env)
    if cohort is None:
        source = 'annotation'
        cohort = _annotation_cohort(env, parts)
    if cohort is None:
        source = 'gang'
        if parts is None:
            raise ExchangeError('GANG_TOPOLOGY_REQUIRED')
        if len(parts) < 2:
            raise ExchangeError('COHORT_REQUIRED')
        cohort = _parse_members(','.join(parts), env.get(PORT_ENV))
    if parts is not None and not set(parts) <= {host for host, _ in cohort}:
        raise ExchangeError('GANG_HOST_NOT_IN_COHORT')
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
    # The launcher resolves an oss:// checkpoint to its local fuse mount before
    # the engine starts; prefer that over the raw URI, which is not readable here.
    checkpoint = env.get(CHECKPOINT_ENV) or checkpoint or env.get('CHECKPOINT_PATH')
    if not checkpoint:
        raise ExchangeError('CHECKPOINT_REQUIRED')
    revision = env.get(REVISION_ENV) or _checkpoint_revision(checkpoint)
    index = _instance_index(env, cohort, gang_hosts[rank] if gang_hosts is not None else None)
    raw_ranks = env.get(RANKS_ENV)
    if raw_ranks is not None and (not raw_ranks.isdigit() or int(raw_ranks) <= 0):
        raise ExchangeError('INVALID_RANKS_PER_INSTANCE')
    # Internal topology namespace, NOT a deployment input or restart token.
    # Store attempt counters below isolate repeated/partial startup attempts.
    generation = str(uuid5(NAMESPACE_URL, 'engram|' + revision + '|' +
                           ','.join(host for host, _ in cohort) + '|' + str(cohort[0][1]) + '|' + str(p)))
    if raw_ranks is not None and discovered is not None and int(raw_ranks) != discovered:
        raise ExchangeError('RANKS_PER_INSTANCE_MISMATCH')
    if raw_ranks is not None:
        local_world_size = int(raw_ranks)
    else:
        local_world_size = world_size if discovered is None else discovered
    if world_size % local_world_size or instances % (world_size // local_world_size):
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
    single_gang = parts is not None and len(parts) == instances
    # One line that answers "which topology did this pod pick, and why".
    logging.getLogger(__name__).info(
        'engram topology source=%s instances=%d p=%d index=%d coordinator=%s:%d '
        'local=%s single_gang=%s',
        source, instances, p, index, cohort[0][0], cohort[0][1],
        parts[0] if parts else '-', single_gang)
    return dict(
        group=group,
        instance_id=local['instance_id'],
        index=index,
        checkpoint=checkpoint,
        coordinator=dict(host=cohort[0][0], port=cohort[0][1]),
        timeout_seconds=timeout,
        rank_base=(rank // local_world_size) * local_world_size if parts is not None
                  else (index % (world_size // local_world_size)) * local_world_size,
        single_gang=single_gang,
    )


def build_config(env, world_size, *, rank=0, gang_hosts=None, checkpoint=None):
    """Add local allocation policy and optional mTLS material."""
    config = build_group(env, world_size, rank=rank, gang_hosts=gang_hosts,
                         checkpoint=checkpoint)
    if config is None:
        return None
    if 'RTP_ENGRAM_LOCAL_DESCRIPTORS' in env:
        raise ExchangeError('REMOVED_LOCAL_DESCRIPTOR_MODE')
    config['tls'] = None if config['single_gang'] else _tls(
        env, config['index'], len(config['group']['members']))
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
    logging.getLogger(__name__).info(
        'engram exchange: role=%s instance=%s coordinator=%s:%d members=%d tls=%s',
        'coordinator' if ident == coordinator else 'remote', ident,
        address['host'], address['port'], len(group['members']), bool(tls))
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


def prepare_static_engram(store, *, rank, world_size, environ=None, gang_hosts=None,
                          checkpoint=None, leader_exchange=exchange_leaders):
    """All ranks enter once; only host representatives use cross-gang RPC."""
    env = os.environ if environ is None else environ
    started = time.monotonic()
    config = build_config(env, world_size, rank=rank, gang_hosts=gang_hosts,
                          checkpoint=checkpoint)
    if config is None:
        return None
    group = config['group']
    group_digest = validate_group(group)
    members = {m['instance_id']: m for m in group['members']}
    member = members[config['instance_id']]
    local_world_size = member['world_size']
    local_rank = rank - config['rank_base']
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
    # One monotonically increasing counter per rank, retained in the gang Store.
    # A restarted rank advances to a fresh namespace and cannot read old packages.
    # Partial restart fails boundedly; recovery recreates the whole gang/Store.
    attempt = store.add(f'engram/start_count/{rank}', 1)
    shared = config['single_gang']
    receiver_rank = rank if shared else local_rank
    receivers = world_size if shared else local_world_size
    leader = receiver_rank == 0
    scope = 'gang' if shared else config['instance_id']
    prefix = f'engram/{group["generation"]}/attempt/{attempt}/{scope}/'
    logging.getLogger(__name__).info(
        'engram startup: rank=%d world=%d instance=%s local_rank=%d p=%d shared=%s '
        'leader=%s attempt=%d timeout=%.0fs',
        rank, world_size, config['instance_id'], local_rank, p, shared, leader,
        attempt, timeout)
    store.set(prefix + f'config/{receiver_rank}', group_digest.encode())
    config_keys = [prefix + f'config/{i}' for i in range(receivers)]
    if leader:
        logging.getLogger(__name__).info(
            'engram startup: waiting for %d group digests', receivers)
        _wait(store, config_keys, deadline)
        if any(store.get(k) != group_digest.encode() for k in config_keys):
            raise ExchangeError('INSTANCE_GROUP_CONFIG_MISMATCH')
    for shard in member['shards']:
        if shard['publisher_rank'] != local_rank:
            continue
        from rtp_llm.models_py.modules.dsv41.engram_publish import publish_shard
        if env.get('RTP_ENGRAM_LOCAL_DEVICE') is None or env.get('RTP_ENGRAM_LOCAL_NUMA') is None:
            logging.getLogger(__name__).warning(
                'RTP_ENGRAM_LOCAL_DEVICE/RTP_ENGRAM_LOCAL_NUMA unset; the self-published'
                ' shard allocates on device %d / NUMA %d by default',
                config['local_device'], config['local_numa'])
        logging.getLogger(__name__).info(
            'engram startup: publishing shard %d of %d on %s (device %d, NUMA %d)',
            shard['index'] + 1, p, shard['host_id'],
            config['local_device'], config['local_numa'])
        shard_started = time.monotonic()
        document = publish_shard(
            config['checkpoint'], group['model_revision'], shard['host_id'],
            group['generation'], shard['index'], p,
            device=config['local_device'], numa=config['local_numa'],
            release_input_page_cache=config['release_page_cache'], deadline=deadline,
            max_allocation_bytes=config['max_allocation_bytes'], budget_bytes=config['budget_bytes'])
        logging.getLogger(__name__).info(
            'engram startup: published shard %d in %.1fs',
            shard['index'], time.monotonic() - shard_started)
        store.set(prefix + f'shard/{shard["index"]}', encode(document))
    if leader:
        shards = [sh for m in group['members'] for sh in m['shards']] if shared else member['shards']
        keys = [prefix + f'shard/{sh["index"]}' for sh in shards]
        logging.getLogger(__name__).info(
            'engram startup: waiting for %d shard descriptors', len(keys))
        _wait(store, keys, deadline)
        documents = [decode(store.get(key)) for key in keys]
        # A single multi-host gang already has a control channel. No extra socket.
        package = (assemble_package(group, documents, config['checkpoint']) if shared
                   else leader_exchange(config, documents, deadline))
        verify_package(group, package)
        payload = encode(package)
        store.set(prefix + 'package', payload)
        logging.getLogger(__name__).info(
            'engram startup: package assembled (%d bytes, %d shards)',
            len(payload), len(documents))
    _wait(store, [prefix + 'package'], deadline)
    package = decode(store.get(prefix + 'package'))
    stage_package(to_memory_package(group, package))
    logging.getLogger(__name__).info(
        'engram startup: package staged in %.1fs', time.monotonic() - started)
    env[COUNT_ENV] = str(p)
    store.set(prefix + f'received/{receiver_rank}', b'1')
    if leader:
        _wait(store, [prefix + f'received/{i}' for i in range(receivers)], deadline)
        # Every worker holds the package in memory. No serving-time Store reads.
        for key in ([prefix + 'package'] + keys + config_keys +
                    [prefix + f'received/{i}' for i in range(receivers)]):
            store.delete_key(key)
    logging.getLogger(__name__).info(
        'engram startup: ready in %.1fs', time.monotonic() - started)
    return None
