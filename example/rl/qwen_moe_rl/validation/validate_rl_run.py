#!/usr/bin/env python3
"""Fail-closed recorded-evidence acceptance for full Qwen3.5-397B FSDP32 RL.

The accepted 122B implementation's reward, actual Python Tool, exact delivered
trace, and observed CUDA loss checks live unchanged in _validated_core.py.
This file adds the configured 397B action-token/turn budget to the existing
trace-integrity check, alongside model initialization and topology contracts.
It launches no GPU work or upgrades old evidence into a new-model pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from checkpoint_contract import MODEL_ID, REVISION, PINNED_HASHES, default_model, identity
from _validated_core import (Validator as Core, Check, canonical_gpu_uuid, cold_loader_source_layout,
                             sha, digest_ids, fp32_bytes, positive, valid_sha, steps, normalize)
from q397_evidence_contracts import hf_residency, native_boot, native_shutdown, reference_gpu_membership

WORLD = 32
TP = 16
REPLICAS = 2
NODES = 8
LOCAL_GPUS = 4
TEXT_NUMEL = 396346350336
VISUAL_NUMEL = 456010480
ORIGINAL_PROMPTS_SHA256 = '2642350d3158fb11da15589c3e848598efc09d6852e57cdd0311688c635339cf'
OFFICIAL_OBJECT_MANIFEST_SHA256 = '8160574fdcfe30895d1e0e419b34dbc510e1750f855e92d1ce2e08c2d100f926'
ACTION_CALL_TOKEN_LIMIT = 2048
ACTION_TOTAL_TOKEN_LIMIT = 3072
ACTION_TURN_LIMIT = 3
NCCL_VERSION = '2.29.3'
NCCL_VERSION_CODE = 22903
NCCL_LIBRARY_SHA256 = 'b7039d11fe9b00e897308754bbd1cbcb1afc46813e2be5d65d5b85e534ce859c'
NCCL_ROOT_SUFFIX = 'logs/dependencies/nccl-2.29.3-arm64/extracted'


def recorded_nccl_scope(proof, project_root, pid, *, self_process=False, verify_source=None):
    """Check original mapped-library evidence without inspecting a gone PID.

    Torch's header macro may remain 2.28.9. Only a genuine in-process
    RTLD_NOLOAD ncclGetVersion can supply a runtime version; other processes
    are bound to the official binary SHA and their live /proc mappings.
    """
    def need(value, message):
        if not value:
            raise ValueError(message)
    distribution = Path(project_root).absolute()/NCCL_ROOT_SUFFIX
    library = str(distribution/'nvidia/nccl/lib/libnccl.so.2')
    need(isinstance(proof, dict) and proof.get('schema') == 'q397-nccl-scope-v1'
         and proof.get('passed') is True, 'Missing original NCCL mapped-library proof')
    need(type(pid) is int and pid > 0 and type(proof.get('pid')) is int and proof['pid'] == pid,
         'NCCL proof PID differs from the actual worker')
    need(type(proof.get('proc_start_ticks')) is int and proof['proc_start_ticks'] > 0,
         'Original live NCCL process-start identity is absent')
    need(proof.get('expected_version') == NCCL_VERSION and proof.get('expected_version_code') == NCCL_VERSION_CODE
         and proof.get('library_path') == library and proof.get('library_sha256') == NCCL_LIBRARY_SHA256
         and proof.get('mapped_nccl_paths') == [library],
         'Actual NCCL mappings do not identify only the exact pinned official 2.29.3 binary')
    env = proof.get('scope_environment') or {}
    expected = dict(Q397_NCCL_ROOT=str(distribution), Q397_NCCL_SO=library,
                    Q397_NCCL_SHA256=NCCL_LIBRARY_SHA256, Q397_NCCL_VERSION=NCCL_VERSION)
    need(isinstance(env, dict) and all(env.get(key) == value for key, value in expected.items()),
         'Actual worker did not inherit the pinned NCCL scope')
    preload = [value for value in re.split(r'[\s:]+', str(env.get('LD_PRELOAD', ''))) if value]
    need(library in preload and all(not Path(value).name.startswith('libnccl.so') or value == library for value in preload),
         'Actual worker explicit NCCL preload is absent or conflicting')
    need(str(distribution) in str(env.get('PYTHONPATH', '')).split(':')
         and str(Path(library).parent) in str(env.get('LD_LIBRARY_PATH', '')).split(':'),
         'Actual worker NCCL Python/library search scope is absent')
    if self_process:
        need(proof.get('runtime_version_code') == NCCL_VERSION_CODE
             and proof.get('version_evidence') == 'self_ncclGetVersion_and_official_library_sha256',
             'Training rank has no original in-process NCCL runtime-version proof')
    else:
        need(proof.get('runtime_version_code') is None
             and proof.get('version_evidence') == 'official_library_sha256_binding',
             'External native PID was assigned an unobserved in-process NCCL version')
    need(proof.get('torch_nccl_version_kind') == 'compile_time_header',
         'Torch NCCL compile-header version was mislabeled as a runtime version')
    need(positive(proof.get('inspected_unix')), 'Actual NCCL inspection time is absent')
    source = proof.get('scope_source') or {}
    source_path = Path(project_root).absolute()/'adapter/q397_nccl_scope.py'
    need(source.get('path') == str(source_path) and positive(source.get('bytes')) and valid_sha(source.get('sha256')),
         'Original loaded shared NCCL scope source fingerprint is absent or belongs to another module')
    if verify_source is not None:
        verify_source(source)
    else:
        need(source_path.is_file() and not source_path.is_symlink()
             and source_path.stat().st_size == source['bytes'] and sha(source_path.read_bytes()) == source['sha256'],
             'Actual shared NCCL scope source differs from its recorded loaded fingerprint')
    return dict(pid=pid, proc_start_ticks=proof['proc_start_ticks'], library_path=library,
                library_sha256=NCCL_LIBRARY_SHA256, version_evidence=proof['version_evidence'],
                runtime_version_code=proof.get('runtime_version_code'),
                torch_compile_version=proof.get('torch_nccl_version'), scope_source=source)


def native_nccl_scope(boot, audit_root, project_root, *, expected_replica, read_record=None, verify_source=None):
    """Bind all original native NCCL wrappers/files to the real boot owners."""
    def need(value, message):
        if not value:
            raise ValueError(message)
    audit_root = Path(audit_root)
    if read_record is None:
        def read_record(path):
            return json.loads(path.read_text())
    need(boot.get('replica_rank') == expected_replica and type(boot.get('native_boot')) is int
         and boot['native_boot'] >= 0, 'NCCL wrappers belong to another native replica/boot')
    nodes = boot.get('nodes', [])
    need(len(nodes) == 4 and {node.get('node_index') for node in nodes} == set(range(4)),
         'All four actual native NCCL owner nodes are required')
    summaries = []
    for node in nodes:
        index = node['node_index']
        process = node.get('process_identity') or {}
        owned = {process.get('pid'): process.get('create_time')}
        owned.update({row.get('pid'): row.get('create_time') for row in process.get('descendants', [])})
        manifests = {row.get('tp_rank'): row for row in node.get('rank_manifests', [])}
        wrappers = node.get('nccl_scope_proofs', [])
        expected_ranks = set(range(index*LOCAL_GPUS, (index+1)*LOCAL_GPUS))
        need(set(manifests) == expected_ranks and len(wrappers) == LOCAL_GPUS
             and {row.get('tp_rank') for row in wrappers} == expected_ranks,
             'Missing or duplicated original native rank NCCL wrappers')
        for row in wrappers:
            rank = row['tp_rank']
            manifest = manifests[rank]
            pid = manifest.get('pid')
            need(row.get('schema') == 'q397-native-rank-nccl-scope-v1'
                 and row.get('replica_rank') == expected_replica and row.get('pid') == pid
                 and row.get('create_time') == owned.get(pid) and positive(row.get('create_time'))
                 and row.get('owner_ip') == node.get('node_ip')
                 and row.get('native_incarnation') == boot.get('native_incarnation')
                 and canonical_gpu_uuid(row.get('gpu_uuid')) == canonical_gpu_uuid(manifest.get('gpu_uuid')),
                 'Original native NCCL wrapper differs from its boot rank PID/creation time/owner/GPU')
            proof = row.get('proof') or {}
            summary = recorded_nccl_scope(proof, project_root, pid, self_process=False, verify_source=verify_source)
            need(row['create_time'] <= proof['inspected_unix'] <= node.get('checked_unix', -1),
                 'Native NCCL proof is not from the original live startup readiness interval')
            relative = Path(row.get('proof_file', 'missing'))
            expected = Path(f'boots/boot-{boot["native_boot"]:03d}/node-{index:02d}/nccl-scope/replica-{expected_replica}/rank-{rank:05d}.json')
            need(not relative.is_absolute() and '..' not in relative.parts and relative == expected,
                 'Native NCCL proof filename is unsafe or lacks original boot/node/replica/rank identity')
            path = audit_root/relative
            need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(audit_root.resolve())
                 and read_record(path) == row, 'Original native NCCL proof file differs from the boot wrapper')
            summaries.append(dict(replica_rank=expected_replica, tp_rank=rank, hostname=node.get('hostname'),
                                  node_ip=node.get('node_ip'), create_time=row['create_time'],
                                  gpu_uuid=canonical_gpu_uuid(row['gpu_uuid']), proof_file=str(path), **summary))
    need(len(summaries) == TP and {row['tp_rank'] for row in summaries} == set(range(TP))
         and len({(row['node_ip'], row['pid'], row['proc_start_ticks']) for row in summaries}) == TP,
         'Native NCCL proofs do not identify sixteen distinct actual rank processes')
    return summaries


def episode_action_budget(episode):
    """Check original delivered action IDs, independently of reported usage."""
    failures = []
    def need(value, message):
        if not value:
            failures.append(message)
    trajectories = episode.get('trajectories')
    need(isinstance(trajectories, list) and bool(trajectories), 'Actual trajectories are absent')
    actual_steps = []
    for trajectory in trajectories if isinstance(trajectories, list) else []:
        values = trajectory.get('steps') if isinstance(trajectory, dict) else None
        need(isinstance(values, list), 'Actual trajectory steps are absent')
        if isinstance(values, list):
            actual_steps.extend(values)
    need(0 < len(actual_steps) <= ACTION_TURN_LIMIT,
         f'Actual delivered turns {len(actual_steps)} exceed limit {ACTION_TURN_LIMIT} or are empty')
    response_lengths, prompt_lengths = [], []
    used = 0
    calls = []
    for number, step in enumerate(actual_steps):
        step = step if isinstance(step, dict) else {}
        response, prompt = step.get('response_ids'), step.get('prompt_ids')
        valid_response = isinstance(response, list) and bool(response) and all(type(x) is int and x >= 0 for x in response)
        valid_prompt = isinstance(prompt, list) and bool(prompt) and all(type(x) is int and x >= 0 for x in prompt)
        need(valid_response and valid_prompt, f'Step {number}: Actual prompt/action token arrays are malformed')
        count = len(response) if isinstance(response, list) else 0
        remaining = ACTION_TOTAL_TOKEN_LIMIT - used
        need(0 < count <= ACTION_CALL_TOKEN_LIMIT,
             f'Step {number}: Actual action tokens {count} exceed per-call limit {ACTION_CALL_TOKEN_LIMIT} or are empty')
        need(count <= remaining,
             f'Step {number}: Actual action tokens {count} exceed remaining episode budget {remaining}')
        calls.append(dict(agent_step=number, actual_action_tokens=count, used_before=used,
                          remaining_before=remaining, maximum_allowed=min(ACTION_CALL_TOKEN_LIMIT, remaining)))
        used += count
        response_lengths.append(count)
        prompt_lengths.append(len(prompt) if isinstance(prompt, list) else 0)
    need(used <= ACTION_TOTAL_TOKEN_LIMIT,
         f'Actual delivered action-token total {used} exceeds limit {ACTION_TOTAL_TOKEN_LIMIT}')
    metrics = episode.get('metrics') or {}
    need(isinstance(metrics, dict), 'Original episode metrics are absent')
    metrics = metrics if isinstance(metrics, dict) else {}
    expected = dict(agent_completion_tokens=used, steps_used=len(actual_steps), steps_collected=len(actual_steps),
                    num_trajectories=len(trajectories) if isinstance(trajectories, list) else 0)
    if actual_steps:
        expected.update(mean_response_len=used/len(actual_steps), max_response_len=max(response_lengths),
                        min_response_len=min(response_lengths), max_prompt_len=max(prompt_lengths),
                        min_prompt_len=min(prompt_lengths))
    for key, value in expected.items():
        reported = metrics.get(key)
        valid = type(reported) in (int, float) and math.isfinite(reported)
        equal = math.isclose(reported, value, rel_tol=1e-12, abs_tol=1e-9) if valid and key == 'mean_response_len' else valid and reported == value
        need(equal, f'Metric {key}={reported!r} differs from actual token/step arrays ({value})')
    return dict(episode_id=episode.get('episode_id'), passed=not failures, failures=failures,
                actual_action_tokens=used, actual_turns=len(actual_steps), calls=calls,
                expected_metrics=expected, reported_metrics={key: metrics.get(key) for key in expected})


def shard_extent(shape, rank):
    width = math.ceil(shape[0] / WORLD)
    begin = min(rank * width, shape[0])
    end = min(begin + width, shape[0])
    return begin, end, [end-begin, *shape[1:]]


class Validator(Core):
    def __init__(self, args):
        super().__init__(args)
        self.replica_syncs = {}
        self.replica_probes = {}
        self.boots = {}
        self.boot_worlds = {}
        self.load_uuid_by_rank = {}
        self.load_owner_by_rank = {}
        self.checkpoint_process_by_rank = {}
        self._loaded_nccl_config = None
        self.recorded_root = (getattr(args, 'recorded_run_root', None) or args.run_root).resolve()
        self.collection = None

    def delivered_trace_linkage(self, c):
        # Keep the original exact-history/retry/receipt association, then check
        # budgets on the unchanged action IDs selected for the agent/training.
        super().delivered_trace_linkage(c)
        budgets = []
        for path, episode in self.episodes:
            budget = episode_action_budget(episode)
            for failure in budget['failures']:
                c.need(False, path.name+': '+failure)
            budgets.append(budget)
        for path, receipt in self.generation_receipts:
            count = receipt.get('response_tokens')
            c.need(type(count) is int and 0 < count <= ACTION_CALL_TOKEN_LIMIT,
                   path.name+': Actual native generation, including retry-only calls, exceeds the per-call token limit')
        c.details.update(action_token_budget=dict(per_call=ACTION_CALL_TOKEN_LIMIT,
                         per_episode=ACTION_TOTAL_TOKEN_LIMIT, maximum_turns=ACTION_TURN_LIMIT,
                         accounting='Actual delivered response_ids only; retry-only calls retain the per-call limit and are excluded from episode usage.',
                         episodes=budgets))

    def load_records(self, c):
        super().load_records(c)
        self.collection = self.read(self.root/'EVIDENCE-COLLECTION.json')
        collection = self.collection
        c.need(collection.get('schema') == 'qwen397-eight-node-audit-collection-v1'
               and collection.get('passed') is True and collection.get('errors') == []
               and collection.get('conflicts') == [] and len(collection.get('nodes', [])) == NODES,
               'An unambiguous checksummed structured-evidence collection from all eight nodes is required')
        recorded = collection.get('source_run_root')
        if not getattr(self.args, 'recorded_run_root', None) and isinstance(recorded, str):
            self.recorded_root = Path(recorded).resolve()
        c.need(recorded == str(self.recorded_root), 'Aggregate source RUN_ROOT differs from the explicit recorded run identity')
        for relative, row in collection.get('files', {}).items():
            relative = Path(relative)
            if not c.need(not relative.is_absolute() and '..' not in relative.parts,
                          'Unsafe aggregate evidence member name'):
                continue
            actual = self.root/relative
            c.need(actual.is_file() and actual.stat().st_size == row.get('bytes')
                   and sha(actual.read_bytes()) == row.get('sha256'),
                   'Copied aggregate evidence differs from its original source checksum: '+str(relative))
        c.need(self.checkpoint.text_numel == TEXT_NUMEL and self.checkpoint.visual_numel == VISUAL_NUMEL
               and self.checkpoint.num_layers == 60 and self.checkpoint.num_experts == 512,
               'Actual official shard headers do not describe the required 397B model')
        c.need(len(self.checkpoint.text) == 1038 and len(self.checkpoint.visual) == 333
               and len(self.checkpoint.experts) == 120 and len(self.checkpoint.gdn) == 90,
               'Official text/vision/expert/GDN tensor structure differs from the pinned model')
        self.replica_syncs[0], self.replica_probes[0] = self.syncs, self.probes
        for replica in range(REPLICAS):
            if replica:
                self.replica_syncs[replica], self.replica_probes[replica] = {}, {}
                for step in range(self.args.minimum_updates+1):
                    for prefix, target in [('rtp_sync', self.replica_syncs[replica]),
                                           ('rtp_probe', self.replica_probes[replica])]:
                        path = self.rtp / f'{prefix}_rank{replica}_step{step}.json'
                        if path.exists():
                            target[step] = (path, self.read(path))
            self.boots[replica] = self.read(self.rtp / f'rtp_boot_{replica}_step0.json')
        c.details.update(model_id=MODEL_ID, model_revision=REVISION, native_replicas=REPLICAS)

    def model_load(self, c):
        records = self.records(self.root/'model-load', 'model-load-rank*.json')
        seen = Counter()
        owners = {}
        for path, record in records:
            rank = record.get('rank')
            seen[rank] += 1
            prefix = path.name+': '
            c.need(record.get('schema') == 'qwen36-text-only-load-v1', prefix+'Unexpected model-load schema')
            c.need(record.get('world_size') == WORLD and rank in range(WORLD), prefix+'Wrong FSDP32 world/rank')
            c.need(record.get('checkpoint_load_mode') == 'meta_sharded_checkpoint_slice_world32'
                   and record.get('pre_shard_parameter_devices') == ['meta'],
                   prefix+'The 397B actor must be initialized from bounded checkpoint slices on all meta ranks')
            c.need(record.get('model_class') == 'Qwen3_5MoeForConditionalGeneration'
                   and record.get('model_type') == 'qwen3_5_moe', prefix+'Wrong actual full-model class')
            c.need(record.get('text_trainable_numel') == TEXT_NUMEL
                   and record.get('visual_frozen_numel') == VISUAL_NUMEL,
                   prefix+'Full text training/frozen vision parameter counts differ from the checkpoint')
            c.need(record.get('mtp_layers') == record.get('mtp_frozen_numel') == 0
                   and record.get('use_cache') is False, prefix+'MTP/cache must remain disabled in training')
            c.need(record.get('parameter_prefix_preserved') == 'model.language_model', prefix+'HF source prefix changed')
            c.need(record.get('master_dtype') == 'torch.float32'
                   and record.get('master_dtypes') == ['torch.float32'], prefix+'Training masters are not FP32')
            owner = (record.get('hostname'), record.get('node_ip'))
            c.need(all(isinstance(value, str) and value for value in owner), prefix+'Missing node-local owner')
            owners[rank] = owner
        c.need(set(seen) == set(range(WORLD)) and all(count == 1 for count in seen.values()),
               'Exactly one actual model-load record is required for each of the 32 ranks')
        c.need(len(set(owners.values())) == NODES and sorted(Counter(owners.values()).values()) == [LOCAL_GPUS]*NODES,
               'Model initialization did not use four ranks on each of eight distinct nodes')
        self.load_owner_by_rank = owners
        collection_owners = {(row.get('hostname'), row.get('node_ip')) for row in (self.collection or {}).get('nodes', [])}
        c.need(collection_owners == set(owners.values())
               and all(row.get('run_root') == str(self.recorded_root) for row in (self.collection or {}).get('nodes', [])),
               'Eight-node evidence collection does not identify the actual training worker owners and original run root')
        slices = self.records(self.root/'model-load', 'checkpoint-load-rank*.json')
        seen_slices = Counter()
        aggregate = Counter()
        all_sources = self.checkpoint.text | self.checkpoint.visual
        mapping = self.read(self.args.model_snapshot/'model.safetensors.index.json').get('weight_map', {})
        for path, record in slices:
            rank = record.get('rank')
            seen_slices[rank] += 1
            prefix = path.name+': '
            if not c.need(type(rank) is int and rank in range(WORLD), prefix+'Invalid checkpoint slice rank'):
                continue
            c.need(record.get('schema') == 'qwen397-streamed-checkpoint-load-v2'
                   and record.get('world_size') == WORLD, prefix+'Missing actual FSDP32 checkpoint loader audit')
            c.need(record.get('model_id') == MODEL_ID and record.get('model_revision') == REVISION
                   and record.get('config_sha256') == PINNED_HASHES['config.json']
                   and record.get('checkpoint_index_sha256') == PINNED_HASHES['model.safetensors.index.json'],
                   prefix+'Loaded checkpoint identity is not the pinned official 397B model')
            c.need(Path(record.get('checkpoint_path', 'missing')).resolve() == self.args.model_snapshot.resolve(),
                   prefix+'Checkpoint slice loader read another model path')
            c.need(record.get('pre_shard_parameter_devices') == ['meta']
                   and record.get('rank0_full_model_materialized') is False
                   and record.get('buffers_verified') is True and positive(record.get('buffer_count')),
                   prefix+'Meta initialization or real rotary buffer restoration is not proved')
            c.need(record.get('master_dtype') == 'torch.float32'
                   and str(record.get('master_device', '')).startswith('cuda:'), prefix+'Loaded masters are not FP32 CUDA shards')
            c.need((record.get('hostname'), record.get('node_ip')) == owners.get(rank), prefix+'Loader owner differs from its model rank')
            c.need(type(record.get('pid')) is int and record['pid'] > 0, prefix+'Original checkpoint worker PID is absent')
            self.checkpoint_process_by_rank[rank] = record
            gpu = canonical_gpu_uuid(record.get('gpu_uuid'))
            self.load_uuid_by_rank[rank] = gpu
            rows = record.get('source_slices', [])
            names = [row.get('name') for row in rows]
            c.need(set(names) == set(all_sources) and len(names) == len(set(names)) == 1371,
                   prefix+'Actual loaded shards must cover exactly all 1038 text and 333 visual source tensors')
            local_total = local_text = source_bytes = max_source = 0
            for row in rows:
                name = row.get('name')
                shape = all_sources.get(name)
                if not c.need(shape is not None, prefix+'Unexpected checkpoint source '+str(name)):
                    continue
                begin, end, local_shape = shard_extent(shape, rank)
                count = math.prod(local_shape)
                c.need(row.get('global_shape') == shape and row.get('shard_dim') == 0
                       and row.get('shard_begin') == begin and row.get('shard_end') == end
                       and row.get('local_shape') == local_shape and row.get('local_numel') == count,
                       prefix+'Actual DTensor slice differs from its official rank interval: '+name)
                c.need(row.get('target_dtype') == 'torch.float32' and row.get('local_bytes') == 4*count
                       and row.get('source_dtype') in {'torch.bfloat16', 'torch.float32'},
                       prefix+'Checkpoint source/master dtype or local byte extent is wrong: '+name)
                width = 2 if row.get('source_dtype') == 'torch.bfloat16' else 4
                c.need(row.get('source_slice_bytes') == width*count and row.get('checkpoint_file') == mapping.get(name),
                       prefix+'Source shard filename/byte extent differs from the pinned index: '+name)
                c.need(row.get('sample_coordinates_match') is True and row.get('sample_count') == min(64, count)
                       and valid_sha(row.get('source_sample_sha256')),
                       prefix+'Actual CUDA FP32 coordinates do not match sampled official source values: '+name)
                is_text = name in self.checkpoint.text
                c.need(row.get('requires_grad') is is_text, prefix+'Text/visual trainability differs from the original path: '+name)
                local_total += count
                local_text += count if is_text else 0
                source_bytes += width*count
                max_source = max(max_source, width*count)
                aggregate[name] += count
            c.need(record.get('loaded_local_numel') == local_total
                   and record.get('loaded_local_bytes') == 4*local_total
                   and record.get('trainable_local_numel') == local_text
                   and record.get('maximum_checkpoint_slice_bytes') == max_source,
                   prefix+'Actual local checkpoint allocation/byte totals are inconsistent')
            c.need(record.get('shard_coverage_allreduce') == [TEXT_NUMEL+VISUAL_NUMEL, TEXT_NUMEL],
                   prefix+'Actual all-rank checkpoint coverage reduction is wrong')
        c.need(set(seen_slices) == set(range(WORLD)) and all(count == 1 for count in seen_slices.values()),
               'Actual checkpoint initialization evidence is missing or duplicated on one or more ranks')
        c.need(dict(aggregate) == {name: math.prod(shape) for name, shape in all_sources.items()},
               'The 32 real local shard intervals do not reconstruct every official parameter exactly once')
        c.need(len(set(self.load_uuid_by_rank.values())) == WORLD,
               'Checkpoint loading did not populate 32 distinct physical GPUs')
        c.details.update(training_world=WORLD, nodes={str(rank): list(owner) for rank, owner in owners.items()},
                         actual_loaded_parameters=len(aggregate), trainable_text_numel=TEXT_NUMEL,
                         frozen_visual_numel=VISUAL_NUMEL,
                         empty_tail_policy='Empty small-parameter first-axis tail shards are valid; global element coverage must be exact.')
        self._training_nccl_scope(c)

    def _current_nccl_scope(self):
        if self._loaded_nccl_config is None:
            path = self.args.project_root/'adapter/q397_nccl_scope.py'
            spec = importlib.util.spec_from_file_location('q397_acceptance_pinned_nccl_scope', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config = module.scope_config(self.args.project_root)
            expected_library = str(self.args.project_root.resolve()/NCCL_ROOT_SUFFIX/'nvidia/nccl/lib/libnccl.so.2')
            if (config.get('library_path') != expected_library or config.get('library_sha256') != NCCL_LIBRARY_SHA256
                    or config.get('version') != NCCL_VERSION or config.get('version_code') != NCCL_VERSION_CODE):
                raise ValueError('Current project-local official NCCL artifact/config differs from the required mapped library')
            self._loaded_nccl_config = config
        return self._loaded_nccl_config

    def _nccl_source_fingerprint(self, c, source):
        self._source_fingerprint(c, source['path'], source['sha256'], 'Loaded shared NCCL scope source', source['bytes'])

    def _training_nccl_scope(self, c):
        self._current_nccl_scope()
        records = self.records(self.root/'model-load', 'comm-preflight-rank*.json')
        seen = Counter()
        summaries = []
        for path, record in records:
            rank = record.get('rank')
            seen[rank] += 1
            if not c.need(type(rank) is int and rank in range(WORLD), path.name+': Invalid actual communication rank'):
                continue
            original = self.checkpoint_process_by_rank.get(rank, {})
            c.need(record.get('schema') == 'qwen397-pre-checkpoint-communication-v1' and record.get('passed') is True
                   and record.get('world_size') == WORLD and record.get('pid') == original.get('pid')
                   and (record.get('hostname'), record.get('node_ip')) == self.load_owner_by_rank.get(rank)
                   and canonical_gpu_uuid(record.get('gpu_uuid')) == self.load_uuid_by_rank.get(rank),
                   path.name+': Communication proof does not identify its actual FSDP32 checkpoint worker PID/host/GPU')
            c.need(record.get('coverage') == [528, WORLD] and record.get('shard_weight_files_opened_before_preflight') == 0
                   and record.get('nccl_ib_gid_index') == '3' and record.get('nccl_ib_disable') in (None, '0')
                   and record.get('collective') == 'existing default process group CUDA int64 all_reduce'
                   and record.get('nccl_version_kind') == 'compile_time_header',
                   path.name+': Actual original CUDA communication preflight, transport or before-weight-load boundary differs')
            proof = record.get('nccl_scope') or {}
            summary = recorded_nccl_scope(proof, self.args.project_root, record.get('pid'), self_process=True,
                                          verify_source=lambda source: self._nccl_source_fingerprint(c, source))
            c.need(proof['inspected_unix'] <= record.get('started_unix', -1) <= record.get('finished_unix', -1)
                   <= original.get('started_unix', -1),
                   path.name+': NCCL runtime inspection/real CUDA collective did not precede the actual checkpoint load')
            summaries.append(dict(rank=rank, hostname=record.get('hostname'), node_ip=record.get('node_ip'), **summary))
        c.need(set(seen) == set(range(WORLD)) and all(count == 1 for count in seen.values()) and len(summaries) == WORLD,
               'Exactly one original new-NCCL runtime/collective proof is required from all32 actual training ranks')
        c.need(len({(row['node_ip'], row['pid'], row['proc_start_ticks']) for row in summaries}) == WORLD,
               'Training NCCL proofs do not identify32 distinct original worker processes')
        c.details.update(nccl_runtime_version=NCCL_VERSION, nccl_library_sha256=NCCL_LIBRARY_SHA256,
                         training_rank_nccl_runtime_proofs=summaries,
                         torch_nccl_version_scope='Recorded compile-header label; runtime acceptance uses actual mapped SO and in-process ncclGetVersion.')

    def official_model_objects(self, c):
        path = Path(__file__).with_name('official-model-objects.json')
        official = self.read(path)
        c.need(self.inventory.get(str(path.resolve()), {}).get('sha256') == OFFICIAL_OBJECT_MANIFEST_SHA256
               and official.get('model_id') == MODEL_ID and official.get('revision') == REVISION,
               'Pinned official model object manifest differs from the audited 397B revision')
        objects = {row['name']:row for row in official.get('objects', [])}
        c.need(len(objects) == 105 and sum(row.get('bytes', 0) for row in objects.values()) == 806819469998
               and len([row for row in objects.values() if row['name'].endswith('.safetensors') and row.get('lfs_sha256')]) == 94,
               'Pinned 397B object list is incomplete')
        receipt_path = self.args.model_download_receipt or self.args.model_snapshot.resolve().parent/'download-receipt.json'
        receipt = self.read(receipt_path)
        c.need(receipt.get('revision') == REVISION and receipt.get('status') == 'completed'
               and receipt.get('verified') is True and receipt.get('all_official_object_hashes_match') is True
               and receipt.get('errors') == [] and receipt.get('total_bytes') == official.get('total_bytes')
               and Path(receipt.get('destination', 'missing')).resolve() == self.args.model_snapshot.resolve(),
               'Official download/readback receipt is missing or belongs to another snapshot')
        downloaded = receipt.get('files', {})
        c.need(set(downloaded) == set(objects), 'Download receipt does not cover the complete pinned runtime/model objects')
        for name, expected in objects.items():
            row = downloaded.get(name, {})
            file = self.args.model_snapshot/name
            c.need(row.get('status') == 'verified' and row.get('bytes') == expected['bytes']
                   and row.get('official_object_hash_match') is True and row.get('readback_sha256_match') is True
                   and valid_sha(row.get('sha256')) and file.is_file() and file.stat().st_size == expected['bytes'],
                   'Model object download/readback size or identity is wrong: '+name)
            if expected.get('lfs_sha256'):
                c.need(row.get('sha256') == expected['lfs_sha256'], 'Weight shard checksum differs from the pinned official LFS object: '+name)
            elif file.is_file():
                payload = file.read_bytes()
                git_blob = hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
                c.need(git_blob == expected.get('git_blob') and sha(payload) == row.get('sha256'),
                       'Actual tokenizer/config/runtime bytes differ from the official git object: '+name)
        current = self.read(self.root/'model-objects-current.json')
        c.need(current.get('schema') == 'qwen397-eight-node-current-model-objects-v1'
               and current.get('revision') == REVISION and current.get('status') == 'complete'
               and Path(current.get('model_snapshot', 'missing')).resolve() == self.args.model_snapshot.resolve(),
               'Current full SHA256 readback is missing or belongs to another model')
        nodes = current.get('nodes', [])
        c.need(len(nodes) == NODES and {node.get('node_rank') for node in nodes} == set(range(NODES)),
               'Current object hash computation did not use all eight distinct node partitions')
        observed = {}
        ordered = sorted(objects)
        for node in nodes:
            rank = node.get('node_rank')
            if type(rank) is not int or rank not in range(NODES):
                continue
            c.need(node.get('revision') == REVISION and node.get('status') == 'complete'
                   and positive(node.get('started_unix'))
                   and node.get('finished_unix', 0) >= node.get('started_unix', float('inf')),
                   f'Current model hash partition {rank} did not finish')
            rows = node.get('files', [])
            names = [row.get('name') for row in rows]
            c.need(len(names) == len(set(names)) and set(names) == set(ordered[rank::NODES]),
                   f'Current model hash partition {rank} does not cover its exact assigned source files')
            for row in rows:
                name = row.get('name')
                source = downloaded.get(name, {})
                expected = objects.get(name, {})
                c.need(name not in observed and row.get('verified') is True
                       and row.get('bytes') == expected.get('bytes')
                       and row.get('sha256') == source.get('sha256') and valid_sha(row.get('sha256')),
                       'Current full model object readback differs from the official source/download: '+str(name))
                observed[name] = row
        c.need(set(observed) == set(objects), 'Current full SHA readback does not reconstruct every one of the 105 required model objects')
        c.details.update(official_objects=len(objects), official_weight_shards=94,
                         total_actual_file_bytes=806819469998, current_sha_nodes=len(nodes),
                         source='Pinned official LFS SHA256 and git blob IDs, completed initial download/readback plus independent current eight-way full SHA256 readback')

    def optimizer_updates(self, c):
        super().optimizer_updates(c)
        for path, record in self.optimizers:
            rank = record.get('rank')
            c.need(record.get('master_dtype') == 'torch.float32'
                   and record.get('master_dtypes') == ['torch.float32']
                   and record.get('gradient_dtypes') == ['torch.float32'],
                   path.name+': FP32 master/gradient precision changed')
            c.need((record.get('hostname'), record.get('node_ip')) == self.load_owner_by_rank.get(rank),
                   path.name+': Optimizer owner differs from checkpoint initialization')
            c.need(canonical_gpu_uuid(record.get('gpu_uuid')) == self.load_uuid_by_rank.get(rank),
                   path.name+': Optimizer GPU differs from the actual loaded shard')

    def topology(self, c):
        files = self.files(self.rtp, 'rtp_topology_*.json')
        c.need(len(files) == REPLICAS, 'Expected exactly two external rollout replicas, each native pure TP16')
        all_uuids = set()
        all_hosts = set()
        endpoints = set()
        for replica in range(REPLICAS):
            topology = self.read(self.rtp/f'rtp_topology_{replica}.json')
            c.need(topology.get('replica_rank') == replica and topology.get('tp_size') == topology.get('world_size') == TP
                   and topology.get('dp_size') == topology.get('pp_size') == topology.get('ep_size') == 1,
                   f'Replica {replica}: topology is not native pure TP16/DP1/PP1/EP1')
            hosts, gpu_ids = topology.get('rank_hosts', []), topology.get('gpu_ids', [])
            c.need(len(hosts) == len(gpu_ids) == TP and len(set(zip(hosts, gpu_ids))) == TP,
                   f'Replica {replica}: native rank host/GPU assignments are incomplete or duplicated')
            c.need(len(set(hosts)) == 4 and sorted(Counter(hosts).values()) == [4]*4,
                   f'Replica {replica}: native owners are not four distinct four-GPU nodes')
            addresses = topology.get('control_addresses', [])
            c.need(len(addresses) == len(set(addresses)) == TP and not endpoints.intersection(addresses),
                   f'Replica {replica}: native control endpoints are incomplete or overlap another replica')
            endpoints.update(addresses)
            c.need(not all_hosts.intersection(hosts), 'Native external replicas share a physical node')
            all_hosts.update(hosts)
            world = native_boot(self.boots[replica], TP)
            self.boot_worlds[replica] = world
            c.need(not all_uuids.intersection(world['uuids']), 'Native external replicas share physical GPUs')
            all_uuids.update(world['uuids'])
            expected_ranks = list(range(replica*TP, (replica+1)*TP))
            for local, rank in enumerate(expected_ranks):
                record = self.read(self.rtp/f'rtp_worker_layout_rank{rank}.json')
                c.need(record.get('global_rank') == rank and record.get('tp_rank') == local
                       and record.get('tp_size') == TP and record.get('replica_rank') == replica
                       and record.get('replica_count') == REPLICAS and record.get('group_ranks') == expected_ranks,
                       f'Training rank {rank}: no correct real native TP16 replica group')
                manifest = self.read(self.rtp/f'rtp_weight_manifest_{replica}'/f'rank-{local:05d}.json')
                original = world['manifests'][local]
                c.need(manifest == original, f'Replica {replica} rank {local}: canonical loader manifest differs from its original owner proof')
                c.need(canonical_gpu_uuid(manifest.get('gpu_uuid')) == self.load_uuid_by_rank.get(rank)
                       and (manifest.get('owner_hostname'), manifest.get('owner_ip')) == self.load_owner_by_rank.get(rank),
                       f'Training/native rank {rank}: CUDA IPC sender and receiver are not on the same physical GPU/node')
                names = manifest.get('names', [])
                c.need(len(names) == len(set(names)) and set(names) == set(self.checkpoint.text),
                       f'Replica {replica} rank {local}: not all exact official text HF refit source keys')
        self.source_names = set(self.checkpoint.text)
        c.need(len(all_hosts) == NODES and all_uuids == self.optimizer_uuids == set(self.load_uuid_by_rank.values()),
               'Native TP16x2, actual FSDP32 checkpoint loading and Adam updates did not use the same 32 GPUs')
        c.details.update(training_world=WORLD, independent_native_replicas=REPLICAS,
                         tp_size=TP, owner_nodes=len(all_hosts), physical_gpu_uuids=sorted(all_uuids),
                         exact_hf_sources=len(self.source_names))

    def weight_refit(self, c):
        summaries = {}
        saved = self.syncs
        for replica in range(REPLICAS):
            self.syncs = self.replica_syncs[replica]
            self.expected_source_training_ranks = list(range(replica*TP, (replica+1)*TP))
            one = Check()
            Core.weight_refit(self, one)
            c.errors.extend(f'Replica {replica}: '+error for error in one.errors)
            summaries[replica] = one.details
            boot = self.boots[replica]
            for step, (_, record) in self.syncs.items():
                prefix = f'Replica {replica} step {step}: '
                c.need(record.get('replica_rank') == replica and record.get('native_boot') == 0
                       and record.get('native_incarnation') == boot.get('native_incarnation')
                       and record.get('restarted_since_previous_commit') is False,
                       prefix+'Resident native process incarnation changed or is unbound')
                c.need(record.get('native_lifecycle') == 'resident_native_pause_training_refit_resume',
                       prefix+'The accepted pause/refit/resume lifecycle changed')
                if step:
                    previous = self.syncs.get(step-1, (None, {}))[1]
                    c.need(record.get('previous_committed_version') == previous.get('model_version'),
                           prefix+'Prior committed policy identity changed')
                bindings = record.get('ipc_bindings', [])
                c.need(len(bindings) == TP and {row.get('tp_rank') for row in bindings} == set(range(TP)),
                       prefix+'Per-rank CUDA IPC ownership evidence is incomplete')
                for row in bindings:
                    local = row.get('tp_rank')
                    if type(local) is not int or local not in range(TP):
                        continue
                    global_rank = replica*TP+local
                    manifest = self.boot_worlds.get(replica, {}).get('manifests', [{}]*TP)[local]
                    c.need(row.get('replica_rank') == replica and row.get('global_rank') == global_rank
                           and row.get('global_step') == step and row.get('native_pid') == manifest.get('pid')
                           and row.get('native_boot') == 0 and row.get('native_incarnation') == boot.get('native_incarnation')
                           and canonical_gpu_uuid(row.get('gpu_uuid')) == self.load_uuid_by_rank.get(global_rank)
                           and (row.get('owner_hostname'), row.get('owner_ip')) == self.load_owner_by_rank.get(global_rank),
                           prefix+f'CUDA IPC binding differs from the original training/native process pair at rank {global_rank}')
        self.syncs = saved
        for step in range(self.args.minimum_updates+1):
            records = [self.replica_syncs[r].get(step, (None, {}))[1] for r in range(REPLICAS)]
            c.need(all(records) and len({row.get('full_source_sha256') for row in records}) == 1,
                   f'Step {step}: The two real native replicas did not receive identical full trained source payloads')
        c.details.update(replicas=summaries, source_release='Every physical source is retained until all 16 native readers ACK; both groups reconstruct the same FSDP32 actor.')

    def training_pauses(self, c):
        summary = []
        for replica in range(REPLICAS):
            for step in range(self.args.minimum_updates):
                sync = self.replica_syncs[replica].get(step, (None, {}))[1]
                version = sync.get('model_version', 'missing')
                path = self.rtp/f'rtp_training_pause_{replica}_version{version}.json'
                pause = self.read(path)
                c.need(pause.get('replica_rank') == replica and pause.get('paused') is True
                       and pause.get('native_boot') == 0
                       and pause.get('native_incarnation') == self.boots[replica].get('native_incarnation')
                       and pause.get('previous_committed_version') == version
                       and pause.get('native_processes_released') is False,
                       path.name+': Native training pause does not identify the actual resident committed policy')
                statuses = pause.get('statuses', [])
                c.need(len(statuses) == TP and {row.get('tp_rank') for row in statuses} == set(range(TP))
                       and all(row.get('tp_size') == TP and row.get('phase') == 'paused'
                               and row.get('model_version') == version and row.get('global_step') == step for row in statuses),
                       path.name+': Every native rank must acknowledge pause before training')
                c.need(pause.get('kv_pool_retained') == sync.get('kv_pool_retained_across_training'),
                       path.name+': Recorded native KV lifecycle differs from the actual refit configuration')
                times = [record.get('time_ns', 0)/1e9 for _, record in self.optimizers
                         if record.get('optimizer_step_max') == step+1]
                c.need(len(times) == WORLD and positive(pause.get('finished_at_unix'))
                       and max(sync.get('finished_unix', float('inf')), 0) <= pause.get('finished_at_unix', 0)
                       and pause.get('finished_at_unix', float('inf')) < min(times, default=0),
                       path.name+': Training pause did not precede the corresponding real 32-rank optimizer update')
                summary.append(dict(replica=replica, policy_step=step, version=version,
                                    pause_finished_unix=pause.get('finished_at_unix')))
        c.details.update(pauses=summary, native_lifecycle='resident_native_pause_training_refit_resume')

    def _source_fingerprint(self, c, path, expected, label, size=None):
        path = Path(path)
        allowed = (self.args.project_root.resolve(), self.args.rtp_source.resolve())
        if not c.need(path.suffix == '.py' and path.is_file()
                      and any(path.resolve().is_relative_to(root) for root in allowed),
                      label+': Recorded Python source is outside the exact project/native source trees'):
            return
        payload = path.read_bytes()
        self.inventory[str(path.resolve())] = dict(size_bytes=len(payload), sha256=sha(payload))
        c.need(valid_sha(expected) and sha(payload) == expected and (size is None or size == len(payload)),
               label+': Loaded Python source differs from its recorded fingerprint')

    def _arm_backend(self, c, audit_root, worlds):
        records = self.records(audit_root, 'backend-replica*.json', recursive=True)
        by_owner_pid = defaultdict(list)
        for path, row in records:
            by_owner_pid[(row.get('hostname'), row.get('pid'))].append((path, row))
        source_sets = set()
        matched = []
        for replica, world in worlds.items():
            for manifest in world['manifests']:
                key = (manifest.get('owner_hostname'), manifest.get('pid'))
                candidates = by_owner_pid[key]
                if not c.need(len(candidates) == 1,
                              f'Replica {replica} TP rank {manifest["tp_rank"]}: No unique original ARM backend process proof'):
                    continue
                path, row = candidates[0]
                prefix = str(path)+': '
                c.need(row.get('backend') == 'torch_bf16_v1' and row.get('explicit_project_source') is True
                       and row.get('external_rtp_kernel_wheel') is False and row.get('architecture') == 'aarch64'
                       and row.get('torch') == '2.11.0+cu130' and str(row.get('python', '')).startswith('3.10.9'),
                       prefix+'Actual native worker did not explicitly identify the reviewed ARM BF16 compatibility backend')
                c.need(str(row.get('replica_rank')) == str(replica)
                       and str(row.get('local_node_world_rank')) == str(manifest['tp_rank']//LOCAL_GPUS*LOCAL_GPUS)
                       and len(str(row.get('cuda_visible_devices', '')).split(',')) == LOCAL_GPUS,
                       prefix+'ARM source proof belongs to another native replica/node rank group')
                sources = row.get('source_manifest', [])
                c.need(len(sources) >= 3 and len({item.get('path') for item in sources}) == len(sources),
                       prefix+'Backend source file coverage is missing or duplicated')
                for item in sources:
                    self._source_fingerprint(c, item.get('path', 'missing'), item.get('sha256'), prefix+'backend', item.get('bytes'))
                source_sets.add(tuple(sorted((item.get('path'), item.get('sha256')) for item in sources)))
                fusion = row.get('fusion') or {}
                c.need(fusion.get('canonical_triton_retained') is True and fusion.get('bf16_only_guard') is True,
                       prefix+'The reviewed canonical BF16 QK normalization/RoPE path is not retained')
                self._source_fingerprint(c, fusion.get('canonical_source', 'missing'),
                                         fusion.get('canonical_source_sha256'), prefix+'QK fusion')
                decode = row.get('decode_dispatch') or {}
                c.need(decode.get('sequence_lengths_used') is True and decode.get('context_cu_seqlens_not_passed') is True,
                       prefix+'Actual decode caller does not enforce the explicit sequence-length ABI')
                self._source_fingerprint(c, decode.get('caller_source', 'missing'),
                                         decode.get('caller_source_sha256'), prefix+'decode caller')
                matched.append(dict(replica=replica, tp_rank=manifest['tp_rank'], hostname=key[0], pid=key[1]))
        c.need(len(source_sets) == 1, 'Actual native ARM workers loaded different compatibility source versions')
        return matched

    def native_math_backend(self, c):
        self._current_nccl_scope()
        matched = self._arm_backend(c, self.rtp, self.boot_worlds)
        c.need(len(matched) == WORLD, 'The reviewed ARM backend is not bound to all 32 actual native worker PIDs')
        c.details.update(backend='torch_bf16_v1', external_rtp_kernel_wheel=False,
                         native_worker_processes=matched,
                         scope='Loaded native-worker source identity and explicit BF16 path. Actual model arithmetic is checked independently by the full-checkpoint numeric comparison.')
        native_scopes = []
        for replica in range(REPLICAS):
            native_scopes.extend(native_nccl_scope(self.boots[replica], self.rtp, self.args.project_root,
                                 expected_replica=replica, read_record=self.read,
                                 verify_source=lambda source: self._nccl_source_fingerprint(c, source)))
        c.need(len(native_scopes) == WORLD, 'All32 actual native worker NCCL mappings are required')
        cold_scopes = []
        for path in self.args.hf_comparison or []:
            companion = self.read(path.parent/'native-topology.json')
            cold_scopes.extend(native_nccl_scope(companion.get('boot', {}), path.parent/'rtp', self.args.project_root,
                               expected_replica=0, read_record=self.read,
                               verify_source=lambda source: self._nccl_source_fingerprint(c, source)))
        c.need(bool(self.args.hf_comparison) and len(cold_scopes) == TP*len(self.args.hf_comparison or []),
               'Every fresh independent HF comparison must contain all16 original new-NCCL native process mappings')
        c.details.update(nccl_library_sha256=NCCL_LIBRARY_SHA256,
                         native_rank_nccl_mappings=native_scopes, cold_native_nccl_mappings=cold_scopes,
                         external_pid_version_scope='Exact original mapped official SO SHA; no inferred in-process ncclGetVersion.')

    def post_refit_scores(self, c):
        summaries = []
        for replica in range(REPLICAS):
            baseline = None
            for step in range(self.args.minimum_updates+1):
                entry = self.replica_probes[replica].get(step)
                if not c.need(entry is not None, f'Replica {replica}: missing actual post-refit probe at step {step}'):
                    continue
                path, probe = entry
                values, targets = probe.get('raw_log_probs', []), probe.get('response_ids', [])
                valid = bool(values) and len(values) == len(targets) and all(math.isfinite(x) for x in values)
                if not c.need(valid, path.name+': Invalid actual selected-token scores'):
                    continue
                bits = fp32_bytes(values)
                c.need(probe.get('raw_fp32_bits_hex') == bits.hex() and probe.get('score_sha256') == sha(bits),
                       path.name+': Raw native FP32 bytes/hash differ from actual selected scores')
                sync = self.replica_syncs[replica].get(step, (None, {}))[1]
                c.need(probe.get('transport') == 'native_GenerateStreamCall_prompt_target_scores'
                       and probe.get('tp_size') == TP and probe.get('global_step') == step
                       and probe.get('model_version') == sync.get('model_version')
                       and probe.get('full_source_sha256') == sync.get('full_source_sha256'),
                       path.name+': Probe is not linked to the actual committed native replica policy')
                if baseline is None:
                    baseline = probe
                else:
                    c.need((probe.get('prompt_ids'), targets) == (baseline.get('prompt_ids'), baseline.get('response_ids')),
                           path.name+': Native score probe token inputs changed')
                    c.need(bits != fp32_bytes(baseline.get('raw_log_probs', [])),
                           path.name+': Native inference distribution did not change after actual training')
                distribution = self.read(self.rtp/f'rtp_policy_distribution_replica{replica}_step{step}.json')
                signatures = distribution.get('signatures', [])
                c.need(len(signatures) == WORLD and {(x.get('replica_rank'), x.get('tp_rank')) for x in signatures}
                       == {(r, rank) for r in range(REPLICAS) for rank in range(TP)},
                       path.name+': Probe signatures do not cover all 32 workers in their correct two groups')
                for row in signatures:
                    owner = row.get('replica_rank')
                    other = self.replica_probes.get(owner, {}).get(step, (None, {}))[1]
                    c.need(row.get('tp_size') == TP and row.get('global_step') == step
                           and row.get('model_version') == other.get('model_version')
                           and row.get('source_sha256') == other.get('full_source_sha256')
                           and fp32_bytes(row.get('raw_log_probs', [])) == fp32_bytes(other.get('raw_log_probs', [])),
                           path.name+': Training worker signature differs from its actual independent group probe')
                if signatures and all(len(x.get('raw_log_probs', [])) == len(values) for x in signatures):
                    first = signatures[0]['raw_log_probs']
                    maximum = max(abs(a-b) for row in signatures for a, b in zip(first, row['raw_log_probs']))
                    c.need(maximum <= self.args.logprob_tolerance
                           and math.isclose(distribution.get('max_score_error_across_replicas', -1), maximum,
                                            rel_tol=1e-10, abs_tol=1e-12),
                           path.name+': Independent native replica score agreement exceeds the requested tolerance')
                summaries.append(dict(replica=replica, step=step, model_version=probe.get('model_version'), score_sha256=sha(bits)))
        c.details.update(probes=summaries,
                         scope='One actual TP16 model computation per independent replica; the result is broadcast to its 16 corresponding training workers.')

    def generation_logprobs(self, c):
        index = defaultdict(list)
        seen = set()
        for path, record in self.generation_receipts:
            replica, policy_step = record.get('replica_rank'), record.get('global_step')
            sync = self.replica_syncs.get(replica, {}).get(policy_step, (None, {}))[1]
            prefix = f'{path.name} request {record.get("request_id")}: '
            c.need(record.get('engine') == 'rtp-llm' and record.get('logprobs_mode') == 'raw_logprobs'
                   and replica in range(REPLICAS), prefix+'Missing actual native generation metadata')
            c.need(bool(sync) and record.get('model_version') == sync.get('model_version')
                   and record.get('native_boot') == sync.get('native_boot')
                   and record.get('native_incarnation') == sync.get('native_incarnation'),
                   prefix+'Generation is not bound to its own committed native incarnation/policy')
            native_request_id = record.get('native_request_id')
            c.need(type(native_request_id) is int and 0 < native_request_id < 2**63,
                   prefix+'Actual native RPC request identity must be a positive signed int63')
            c.need(positive(record.get('prompt_tokens')) and positive(record.get('response_tokens'))
                   and record.get('response_tokens') == record.get('logprobs'), prefix+'Generation token/logprob lengths disagree')
            key = tuple(record.get(name) for name in ('prompt_sha256', 'response_sha256', 'raw_logprobs_fp32_sha256'))
            c.need(all(valid_sha(value) for value in key), prefix+'Missing actual token/FP32 payload hashes')
            index[key].append(record)
        c.need(bool(index), 'No actual native generations are recorded')
        for path, episode in self.episodes:
            policy_steps = set()
            for number, step in enumerate(steps(episode)):
                output = step.get('model_output') or {}
                prompt = step.get('prompt_ids') or output.get('prompt_ids') or []
                response = step.get('response_ids') or output.get('completion_ids') or []
                values = step.get('logprobs') or output.get('logprobs') or []
                prefix = f'{path.name} step {number}: '
                valid = bool(prompt) and bool(response) and len(response) == len(values)
                valid &= all(type(value) is int and value >= 0 for value in prompt+response)
                valid &= all(isinstance(value, (int, float)) and math.isfinite(value) and value <= 1e-5 for value in values)
                if not c.need(valid, prefix+'Missing full actual token/logprob arrays'):
                    continue
                bits = fp32_bytes(values)
                import struct
                c.need(list(struct.unpack(f'<{len(values)}f', bits)) == values,
                       prefix+'Native FP32 logprob values were rounded or reconstructed')
                matches = index.get((digest_ids(prompt), digest_ids(response), sha(bits)), [])
                if not c.need(bool(matches), prefix+'Delivered episode payload has no actual native generation receipt'):
                    continue
                versions = {row.get('global_step') for row in matches}
                c.need(len(versions) == 1, prefix+'Identical payload is ambiguously bound to different trained policy steps')
                policy_step = next(iter(versions))
                # Identical bytes from two native replicas are legitimate only
                # because both complete commits are independently checked above.
                source_hashes = {self.replica_syncs[row['replica_rank']][policy_step][1].get('full_source_sha256')
                                 for row in matches}
                c.need(len(source_hashes) == 1 and all(valid_sha(value) for value in source_hashes),
                       prefix+'Identical payload came from replicas with different source policies')
                policy_steps.add(policy_step)
                for row in matches:
                    seen.add((row['replica_rank'], policy_step))
                weight_version = step.get('weight_version', output.get('weight_version'))
                if weight_version is not None:
                    c.need(weight_version == policy_step, prefix+'rLLM weight version differs from the actual native policy step')
                self.raw_segments.append(dict(episode_id=episode.get('episode_id'), source_path=str(path),
                    prompt_token_ids=prompt, output_token_ids=response, raw_log_probs=values,
                    global_step=policy_step, model_version=matches[0].get('model_version'),
                    native_replica_candidates=sorted({row['replica_rank'] for row in matches}),
                    agent_action=step.get('action'), model_response=step.get('model_response', ''),
                    chat_completions=step.get('chat_completions', [])))
            c.need(len(policy_steps) == 1, path.name+': One episode spans missing or multiple policy versions')
            if len(policy_steps) == 1:
                self.episode_policy_steps[episode.get('episode_id')] = next(iter(policy_steps))
        for replica in range(REPLICAS):
            for step in range(self.args.minimum_updates):
                c.need((replica, step) in seen, f'No actual sampled rollout from replica {replica} policy step {step}')
        c.need(bool(self.raw_segments), 'No actual native token-level rollout trajectories are available')
        c.details.update(native_receipts=len(self.generation_receipts), linked_segments=len(self.raw_segments),
                         sampled_replica_steps=sorted(seen))

    def grounded_rewards(self, c):
        super().grounded_rewards(c)
        payload = self.args.original_prompts.read_bytes()
        self.inventory[str(self.args.original_prompts.resolve())] = dict(size_bytes=len(payload), sha256=sha(payload))
        c.need(sha(payload) == ORIGINAL_PROMPTS_SHA256, 'The exact original eight-question Python Tool fixture was replaced')
        original = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        c.need(len(original) == 8, 'Original question fixture must contain exactly eight rows')
        by_row = {row['source_row_index']: row for row in original}
        observed = defaultdict(set)
        for path, episode in self.episodes:
            task = episode.get('task') or {}
            row = task.get('source_row_index')
            expected = by_row.get(row)
            c.need(expected is not None, path.name+': Episode is outside the preserved original eight questions')
            if expected:
                for key in ('question', 'original_question', 'ground_truth', 'data_source'):
                    c.need(normalize(task.get(key, '')) == normalize(expected.get(key, '')),
                           path.name+': Original Python Tool task field changed: '+key)
                observed[self.episode_policy_steps.get(episode.get('episode_id'))].add(row)
        for step in range(self.args.minimum_updates):
            c.need(observed[step] == set(by_row), f'Policy step {step}: Actual rollout did not use every original question')

    def hf_reference(self, c):
        if not c.need(bool(self.args.hf_comparison), 'An independent full-checkpoint BF16 HF/native cold comparison is required'):
            return
        summaries = []
        for path in self.args.hf_comparison:
            record = self.read(path)
            if not c.need(record.get('schema') == 'qwen35-397b-rtp-tp16-hf-reference-v1',
                          str(path)+': Old-model or configuration-only numeric evidence is forbidden'):
                continue
            reference_path = Path(record.get('reference_path', 'missing'))
            if not reference_path.is_absolute():
                reference_path = path.parent/reference_path
            reference = self.read(reference_path)
            companion = self.read(path.parent/'native-topology.json')
            controller = self.read(path.parent/'controller.json')
            c.need(record.get('tp_size') == record.get('native_gpu_count') == TP
                   and record.get('checkpoint_identity') == controller.get('checkpoint_identity') == identity(),
                   str(path)+': Cold native world/identity is wrong')
            c.need(record.get('model_path') == reference.get('model_path') == controller.get('model_path')
                   and Path(record.get('model_path', 'missing')).resolve() == self.args.model_snapshot.resolve(),
                   str(path)+': Cold numeric reference model paths differ')
            world = native_boot(companion.get('boot', {}), TP)
            c.need(world['uuids'] <= self.optimizer_uuids, str(path)+': Cold native devices are outside this run\'s actual 32 training GPUs')
            hf_placement=hf_residency(reference, self.checkpoint, self.optimizer_uuids)
            reference_gpu_membership(hf_placement['devices'],world['uuids'],self.optimizer_uuids)
            backend_processes = self._arm_backend(c, path.parent/'rtp', {0:world})
            c.need(len(backend_processes) == TP, str(path)+': Cold native ARM backend is not bound to all actual TP16 worker PIDs')
            layouts = [cold_loader_source_layout(row.get('names', []), self.source_names,
                       self.checkpoint.num_layers, self.checkpoint.num_experts) for row in world['manifests']]
            c.need(all(layouts) and len(set(layouts)) == 1, str(path)+': Cold native checkpoint source coverage is incomplete')
            statuses = world['statuses']
            baseline = reference.get('cases', [])
            cases = record.get('cases', [])
            c.need(len(baseline) == len(cases) == 3, str(path)+': Three independent full-model token cases are required')
            errors = []
            for number, case in enumerate(cases):
                original = baseline[number] if number < len(baseline) else {}
                prompt, targets = case.get('prompt_token_ids', []), case.get('output_token_ids', [])
                expected, actual = case.get('hf_raw_log_probs', []), case.get('rtp_raw_log_probs', [])
                valid = bool(prompt) and bool(targets) and len(targets) == len(expected) == len(actual)
                valid &= all(type(value) is int and value >= 0 for value in prompt+targets)
                valid &= all(isinstance(value, (int, float)) and math.isfinite(value) for value in expected+actual)
                if not c.need(valid, f'{path}: Case {number} token/FP32 arrays are malformed'):
                    continue
                c.need((prompt, targets, expected) == (original.get('prompt_token_ids'), original.get('output_token_ids'), original.get('raw_log_probs')),
                       f'{path}: Case {number} differs from the independently generated HF target case')
                c.need(all(row.get('model_version') == case.get('model_version')
                           and row.get('global_step') == case.get('global_step') for row in statuses.values()),
                       f'{path}: Case {number} is not bound to the live native checkpoint incarnation')
                differences = [abs(a-b) for a, b in zip(actual, expected)]
                errors.extend(differences)
                c.need(max(differences) <= self.args.logprob_tolerance
                       and math.isclose(case.get('max_abs_error', -1), max(differences), rel_tol=1e-10, abs_tol=1e-12),
                       f'{path}: Case {number} actual selected-token logprob error is excessive or misreported')
            shutdown = self.read(path.parent/'shutdown.json')
            native_shutdown(companion.get('boot', {}), shutdown.get('native', {}))
            c.need(shutdown.get('all_owned_native_gpu_processes_gone') is True
                   and shutdown.get('owned_gpu_processes_after') == [] and shutdown.get('errors') == [],
                   str(path)+': The cold native harness did not release its owned GPU processes')
            c.need(bool(errors) and max(errors) <= self.args.logprob_tolerance,
                   str(path)+': No actual full-checkpoint numeric comparison meets the tolerance')
            summaries.append(dict(path=str(path), selected_tokens=len(errors),
                                  recomputed_max_abs_error=max(errors) if errors else None,
                                  native_gpus=sorted(world['uuids']), hf_gpus=len(reference.get('devices', []))))
        c.need(bool(summaries), 'No actual 397B numeric comparisons were accepted for checking')
        c.details.update(comparisons=summaries, tolerance=self.args.logprob_tolerance,
                         scope='Independent complete official BF16 HF checkpoint on four CUDA GPUs versus native TP16 cold teacher forcing; actual FSDP32 updates are checked separately.')

    def cleanup(self, c):
        driver = self.read(self.rtp/'driver-cleanup.json')
        recorded_rtp = self.recorded_root/'rtp'
        c.need(driver.get('passed') is True and driver.get('audit_dir') == str(recorded_rtp),
               'Final driver cleanup is absent or belongs to another run')
        entries = driver.get('replicas', [])
        c.need(len(entries) == REPLICAS, 'Final cleanup must cover both native TP16 replicas')
        released = []
        for replica in range(REPLICAS):
            boot = self.boots[replica]
            identity_path = self.rtp/f'rtp_cluster_{replica}.json'
            cluster = self.read(identity_path)
            shutdown = self.read(self.rtp/f'rtp_cluster_shutdown_{replica}.json')
            c.need(cluster.get('replica_rank') == replica and cluster.get('node_owner_count') == 4,
                   f'Replica {replica}: Stable controller identity has the wrong node topology')
            summary = native_shutdown(boot, shutdown)
            item = [row for row in entries if row.get('identity_file') == str(recorded_rtp/identity_path.name)]
            c.need(len(item) == 1 and item[0].get('passed') is True,
                   f'Replica {replica}: Driver cleanup has no unique original controller identity')
            final = self.replica_syncs[replica].get(self.args.minimum_updates, (None, {}))[1]
            c.need(positive(final.get('finished_unix'))
                   and shutdown.get('finished_at_unix', 0) > final.get('finished_unix', float('inf'))
                   and driver.get('finished_at_unix', 0) >= shutdown.get('finished_at_unix', float('inf')),
                   f'Replica {replica}: Final cleanup preceded the last actual weight commit')
            released.append(summary)
        c.need(positive(driver.get('started_at_unix'))
               and driver.get('finished_at_unix', 0) >= driver.get('started_at_unix', float('inf')),
               'Final driver cleanup timestamps are inconsistent')
        c.details.update(released_native_replicas=released,
                         scope='Original node/PID/create-time identities, all native rank manifests, owner-local termination ACKs and no owned GPU processes remain.')

    def execute(self):
        report = super().execute()
        report['schema'] = 'qwen35-397b-thirty-two-gpu-real-rl-acceptance-v1'
        report['expected_rollout_replicas'] = REPLICAS
        for check in report['checks']:
            check['name'] = check['name'].replace('full_122b_model_on_8_training_ranks', 'full_397b_model_on_32_training_ranks')
            check['name'] = check['name'].replace('two_real_adamw_updates_on_8_distinct_gpus', 'two_real_adamw_updates_on_32_distinct_gpus')
            check['name'] = check['name'].replace('native_tp8_topology_and_loader_sources', 'native_tp16x2_topology_and_loader_sources')
        report['scope_notes'] = [
            'No GPU stages run inside this offline validator. Old-model and tiny CPU proofs cannot pass the new full 397B GPU requirements.',
            '32 actual FP32-master FSDP2 training shards, CPU FP32 AdamW moments, BF16 forwards, and two native TP16 rollout replicas are required.',
            'An actual independent native computation is required per replica; signatures broadcast inside a TP group are not independent inference computations.',
            'Original eight DeepScaleR questions, actual Python Tool execution/replay, exact delivered/retry trace accounting and observed CUDA GRPO loss remain mandatory.',
            'Functional acceptance does not establish resource saturation or end-to-end RL performance.',
        ]
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--rtp-source', type=Path,
                        default=Path(os.environ.get('RTP_SOURCE', str(PROJECT.parents[2]))))
    parser.add_argument('--recorded-run-root', type=Path,
                        help='Original immutable RUN_ROOT; automatically read from the eight-node collection receipt')
    parser.add_argument('--model-snapshot', type=Path, default=default_model(PROJECT))
    parser.add_argument('--model-download-receipt', type=Path)
    parser.add_argument('--source-dataset', type=Path, required=True)
    parser.add_argument('--agent-source', type=Path, required=True)
    parser.add_argument('--rllm-source', type=Path, required=True)
    parser.add_argument('--original-prompts', type=Path, default=PROJECT/'training/data/eight-original-prompts.jsonl')
    parser.add_argument('--minimum-updates', type=int, default=2)
    parser.add_argument('--world-size', type=int, default=WORLD)
    parser.add_argument('--tp-size', type=int, default=TP)
    parser.add_argument('--tool-replay-limit', type=int, default=3)
    parser.add_argument('--logprob-tolerance', type=float, default=.05)
    parser.add_argument('--hf-comparison', type=Path, action='append', default=[])
    parser.add_argument('--cpu-proof', type=Path)
    args = parser.parse_args()
    if args.world_size != WORLD or args.tp_size != TP or args.minimum_updates < 2 or args.tool_replay_limit < 1:
        parser.error('Requires FSDP32, two native TP16 replicas, at least two real updates and an actual Python Tool replay')
    if not 0 < args.logprob_tolerance <= .05:
        parser.error('Selected-token error tolerance must be in (0,.05]')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report = Validator(args).execute()
    output = args.output or args.run_root/'validation/rl-run-acceptance.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'], output=str(output), checks=len(report['checks']),
                         failed_checks=[check['name'] for check in report['checks'] if check['required'] and not check['passed']])))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
