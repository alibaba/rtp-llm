#!/usr/bin/env python3
"""Offline, fail-closed acceptance of the recorded Qwen3.5-397B-A17B eight-GPU RL run.

This script starts no CUDA work. It checks recorded tensors, hashes and linked
episodes rather than accepting a producer's ``passed`` flag. CPU fixture proofs
are explicitly separate from full-model GPU evidence. Exit 0 means every
required check passed; exit 2 means missing or inconsistent evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
import unicodedata

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from checkpoint_contract import MODEL_ID, REVISION, default_model, identity, load_checkpoint

DEEPSCALER_SHA256 = '4e9e7b18248d982b4498a54a6bdb37b1bb653cf5f77bc309fa67a6c2533c8ba4'
LEGACY = Path(os.environ.get('RLLM_MATH_ROOT', str(PROJECT / 'external/rllm-math')))



def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest_ids(ids):
    # Match the actual native generation receipt, including json.dumps spacing.
    return sha(json.dumps(ids).encode())


def fp32_bytes(values):
    return struct.pack(f'<{len(values)}f', *values)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def canonical_gpu_uuid(value):
    """Reconcile Torch's bare UUID with NVML's GPU- prefix, strictly."""
    if not isinstance(value, str):
        raise ValueError('Missing physical GPU UUID')
    bare = value[4:] if value.startswith('GPU-') else value
    if re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', bare) is None:
        raise ValueError(f'Invalid physical GPU UUID: {value!r}')
    return bare.lower()


def cold_loader_source_layout(names, physical, layers, experts):
    """Accept only the exact physical key set or its *entire* defined expert expansion."""
    names, physical = set(names), set(physical)
    if names == physical:
        return 'physical_hf_sources'
    packed = {f'model.language_model.layers.{layer}.mlp.experts.{leaf}'
              for layer in range(layers) for leaf in ('gate_up_proj', 'down_proj')}
    aliases = {f'layers.{layer}.moe.partial_moe_weights.{kind}.kernel.{expert}.0'
               for layer in range(layers) for kind in ('intermediate_weight', 'intermediate_weight2')
               for expert in range(experts)}
    if packed.issubset(physical) and names == (physical - packed) | aliases:
        return f'exact_native_expansion_{len(aliases)}_aliases_for_{len(packed)}_packed_sources'
    return None


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).split())


def steps(episode):
    for trajectory in episode.get('trajectories', []):
        yield from trajectory.get('steps', [])


class Check:
    def __init__(self):
        self.errors = []
        self.details = {}

    def need(self, condition, reason):
        if not condition:
            self.errors.append(reason)
        return bool(condition)


class Validator:
    def __init__(self, args):
        self.args = args
        self.root = args.run_root.resolve()
        self.rtp = self.root/'rtp'
        self.inventory = {}
        self.input_errors = []
        self.checks = []
        self.optimizers = []
        self.syncs = {}
        self.probes = {}
        self.episodes = []
        self.training_episodes = {}
        self.generation_receipts = []
        self.raw_segments = []
        self.episode_policy_steps = {}
        self.checkpoint = None
        self.source_names = set()
        self.optimizer_uuids = set()
        self._grader = None
        self._tool_module = None
        self._tokenizer = None
        self._parser = None

    def read(self, path, jsonl=False):
        path = Path(path)
        try:
            data = path.read_bytes()
            self.inventory[str(path.resolve())] = {'size_bytes': len(data), 'sha256': sha(data)}
            def reject(value):
                raise ValueError(f'Nonstandard JSON number: {value}')
            if jsonl:
                return [json.loads(line, parse_constant=reject) for line in data.decode().splitlines() if line.strip()]
            return json.loads(data, parse_constant=reject)
        except Exception as exc:
            self.input_errors.append(f'{path}: {type(exc).__name__}: {exc}')
            return [] if jsonl else {}

    def files(self, folder, pattern, recursive=False):
        iterator = folder.rglob(pattern) if recursive else folder.glob(pattern)
        return sorted(p for p in iterator if p.is_file() and not p.name.startswith('._'))

    def records(self, folder, pattern, recursive=False):
        return [(p, self.read(p)) for p in self.files(folder, pattern, recursive)]

    def run_check(self, name, function, required=True):
        check = Check()
        try:
            function(check)
        except Exception as exc:
            check.errors.append(f'{type(exc).__name__}: {exc}')
        self.checks.append(dict(name=name, required=required,
                                passed=not check.errors, failures=check.errors,
                                details=check.details))

    def load_records(self, c):
        self.checkpoint = load_checkpoint(self.args.model_snapshot)
        c.need(self.root.is_dir(), f'RUN_ROOT does not exist: {self.root}')
        self.optimizers = self.records(self.root/'optimizer', 'optimizer-rank*.json')
        self.episodes = self.records(self.root/'tool-episodes', 'enriched_*.json')
        for path, episode in self.records(self.root/'episodes', 'episode_*.json', recursive=True):
            key = episode.get('episode_id')
            if key:
                c.need(key not in self.training_episodes, f'Duplicate training episode ID: {key}')
                self.training_episodes[key] = (path, episode)
        for path in self.files(self.rtp, 'rtp_generations_*.jsonl'):
            self.generation_receipts.extend((path, record) for record in self.read(path, jsonl=True))
        for step in range(self.args.minimum_updates+1):
            path = self.rtp/f'rtp_sync_rank0_step{step}.json'
            if path.exists():
                self.syncs[step] = (path, self.read(path))
            path = self.rtp/f'rtp_probe_rank0_step{step}.json'
            if path.exists():
                self.probes[step] = (path, self.read(path))
        c.details.update(optimizer_records=len(self.optimizers), enriched_episodes=len(self.episodes),
                         logged_training_episodes=len(self.training_episodes),
                         native_generation_receipts=len(self.generation_receipts))

    def model_load(self, c):
        records = self.records(self.root/'model-load', 'model-load-rank*.json')
        seen = Counter()
        for path, record in records:
            rank = record.get('rank')
            seen[rank] += 1
            prefix = f'{path.name}: '
            c.need(record.get('schema') == 'qwen36-text-only-load-v1', prefix+'Unexpected model-load audit schema')
            c.need(record.get('world_size') == self.args.world_size, prefix+'Model world size is not eight')
            mode, devices = ('checkpoint_cpu_rank0', ['cpu']) if rank == 0 else ('meta_receiver_from_rank0', ['meta'])
            c.need(record.get('checkpoint_load_mode') == mode and record.get('pre_shard_parameter_devices') == devices,
                   prefix+'Expected rank-zero CPU checkpoint initialization and meta receivers on the other ranks')
            c.need(record.get('model_class') == 'Qwen3_5MoeForConditionalGeneration', prefix+'Wrong full-model architecture')
            c.need(record.get('model_type') == 'qwen3_5_moe', prefix+'Wrong model type')
            c.need(record.get('text_trainable_numel') == self.checkpoint.text_numel, prefix+'Text actor size differs from the official shard headers')
            c.need(record.get('visual_frozen_numel') == self.checkpoint.visual_numel, prefix+'Visual parameters are not all frozen')
            c.need(record.get('mtp_layers') == record.get('mtp_frozen_numel') == 0 and record.get('use_cache') is False, prefix+'MTP/cache still enabled')
            c.need(record.get('parameter_prefix_preserved') == 'model.language_model', prefix+'Refit parameter prefix changed')
        c.need(set(seen) == set(range(self.args.world_size)), 'Full-model load records do not cover every training rank')
        c.need(all(value == 1 for value in seen.values()), 'Multiple model-load records per rank; use a fresh, unambiguous RUN_ROOT')
        c.details.update(ranks=sorted(seen), trainable_text_parameters=self.checkpoint.text_numel,
                         frozen_visual_parameters=self.checkpoint.visual_numel)

    def optimizer_updates(self, c):
        by_step = defaultdict(dict)
        rank_uuids = defaultdict(set)
        for path, record in self.optimizers:
            rank = record.get('rank')
            step = record.get('optimizer_step_max')
            prefix = f'{path.name}: '
            c.need(record.get('schema') == 'qwen36-bounded-real-optimizer-v1', prefix+'Unexpected optimizer schema')
            c.need(record.get('world_size') == self.args.world_size, prefix+'Optimizer world size is not eight')
            c.need(rank in range(self.args.world_size), prefix+'Unexpected optimizer rank')
            c.need(record.get('optimizer') == 'AdamW', prefix+'Not the real AdamW path')
            moments = record.get('moment_placement', {})
            grad_count = record.get('gradient_tensor_count', 0)
            grad_numel = record.get('gradient_local_numel', 0)
            state_numel = moments.get('parameter_numel_with_state', 0)
            inactive_numel = moments.get('state_numel_with_no_current_gradient')
            c.need(moments.get('implementation') == 'torch.optim.adamw.adamw' and
                   moments.get('moment_devices') == ['cpu'] and moments.get('moment_dtypes') == ['torch.float32'],
                   prefix+'The FP32 AdamW moments are not persistently on CPU')
            valid_state = (positive(grad_count) and positive(grad_numel) and type(inactive_numel) is int and
                           inactive_numel >= 0 and grad_numel + inactive_numel == state_numel and
                           state_numel <= record.get('trainable_local_numel', 0))
            count = moments.get('moment_tensor_count', 0)
            c.need(valid_state and type(count) is int and count % 2 == 0 and
                   (count == 2 * grad_count if inactive_numel == 0 else count > 2 * grad_count) and
                   moments.get('persistent_moment_bytes') == 8 * state_numel,
                   prefix+'Expected two FP32 moments for each local gradient-bearing parameter element')
            staged = moments.get('max_staged_moment_bytes_per_step', 0)
            chunk = moments.get('configured_chunk_bytes_per_moment', 0)
            c.need(positive(staged) and positive(chunk) and staged <= 2 * chunk <= 64 * 1024**2,
                   prefix+'GPU staging of the two moments did not stay within the 64 MiB bound')
            c.need(record.get('optimizer_step_min') == step and positive(step) and int(step) == step, prefix+'Adam parameter step counters disagree')
            c.need(positive(record.get('gradient_l2')), prefix+'Gradient norm is zero or nonfinite')
            c.need(positive(record.get('sample_delta_l2')) and positive(record.get('sample_max_abs_delta')), prefix+'Observed weights did not change')
            c.need(positive(record.get('changed_sampled_elements')) and positive(record.get('gradient_tensor_count')), prefix+'No gradient/update samples')
            c.need(record.get('sample_before_sha256') != record.get('sample_after_sha256') and
                   valid_sha(record.get('sample_before_sha256')) and valid_sha(record.get('sample_after_sha256')), prefix+'Bounded weight hashes did not change')
            c.need(abs(record.get('trainable_local_numel', 0) - self.checkpoint.text_numel / self.args.world_size) < 1_000_000,
                   prefix+'Local parameter count does not match a full text actor shard')
            devices = record.get('parameter_devices', [])
            c.need(bool(devices) and all(str(x).startswith('cuda:') for x in devices), prefix+'Optimizer parameters are not on CUDA')
            gpu_uuid = canonical_gpu_uuid(record.get('gpu_uuid'))
            rank_uuids[rank].add(gpu_uuid)
            if positive(step) and int(step) == step:
                step = int(step)
                c.need(rank not in by_step[step], prefix+'Duplicate optimizer step/rank; evidence mixes runs')
                by_step[step][rank] = record
        summary = []
        for step in range(1, self.args.minimum_updates+1):
            ranks = by_step[step]
            c.need(set(ranks) == set(range(self.args.world_size)), f'Adam step {step} did not occur on every rank')
            total = sum(x.get('trainable_local_numel', 0) for x in ranks.values())
            c.need(abs(total-self.checkpoint.text_numel) < 1_000_000, f'Adam step {step} shard totals do not reconstruct the full text actor')
            summary.append(dict(step=step, ranks=sorted(ranks), sharded_parameter_total=total,
                                minimum_gradient_l2=min((x['gradient_l2'] for x in ranks.values()), default=None),
                                changed_sampled_elements=sum(x.get('changed_sampled_elements', 0) for x in ranks.values())))
            if step > 1:
                for rank in set(ranks) & set(by_step[step-1]):
                    prior, now = by_step[step-1][rank], ranks[rank]
                    c.need(prior['time_ns'] < now['time_ns'], f'Rank {rank} optimizer timestamps did not advance')
                    c.need(prior['sample_after_sha256'] == now['sample_before_sha256'], f'Rank {rank} bounded master-weight update chain is broken')
        c.need(set(rank_uuids) == set(range(self.args.world_size)) and all(len(v) == 1 for v in rank_uuids.values()), 'Physical GPU assignment is missing or changed within this run')
        uuids = {uuid for values in rank_uuids.values() for uuid in values}
        self.optimizer_uuids = uuids
        c.need(len(uuids) == self.args.world_size, 'Eight ranks did not use eight distinct physical GPUs')
        c.details.update(updates=summary, gpu_uuid_by_rank={str(k): sorted(v) for k, v in rank_uuids.items()},
                         audit_scope='Full local gradient norms; deterministic bounded master-weight coordinates, not a full weight snapshot')

    def topology(self, c):
        topology_files = self.files(self.rtp, 'rtp_topology_*.json')
        c.need(len(topology_files) == 1, 'Expected one TP8 rollout replica')
        topology = self.read(self.rtp/'rtp_topology_0.json')
        c.need(topology.get('tp_size') == self.args.tp_size and topology.get('world_size') == self.args.tp_size, 'Native rollout is not TP8')
        c.need(topology.get('dp_size') == topology.get('pp_size') == 1, 'Native RTP DP/PP is not one')
        gpu_ids = topology.get('gpu_ids', [])
        addresses = topology.get('control_addresses', [])
        c.need(len(gpu_ids) == len(set(gpu_ids)) == self.args.tp_size, 'Native topology does not assign eight unique GPUs')
        c.need(len(addresses) == len(set(addresses)) == self.args.tp_size, 'Eight native control endpoints are missing')
        for rank in range(self.args.world_size):
            record = self.read(self.rtp/f'rtp_worker_layout_rank{rank}.json')
            c.need(record.get('global_rank') == rank and record.get('tp_rank') == rank and
                   record.get('tp_size') == self.args.tp_size and record.get('replica_rank') == 0 and
                   record.get('replica_count') == 1 and record.get('group_ranks') == list(range(self.args.world_size)),
                   f'Training rank {rank} does not belong to the real TP8 group')
        for rank in range(self.args.tp_size):
            manifest = self.read(self.rtp/'rtp_weight_manifest_0'/f'rank-{rank:05d}.json')
            names = manifest.get('names', [])
            c.need(manifest.get('tp_rank') == rank and manifest.get('tp_size') == self.args.tp_size, f'Native loader manifest {rank} has wrong rank/world')
            c.need(bool(names) and len(names) == len(set(names)), f'Native loader manifest {rank} is empty or duplicated')
            if rank == 0:
                self.source_names = set(names)
            else:
                c.need(set(names) == self.source_names, f'Native loader rank {rank} requires different HF sources')
        c.need({name for name in self.source_names if '.mlp.experts.' in name} == self.checkpoint.experts,
               'Native manifests must contain both packed expert tensors for every official layer')
        checkpoint = self.read(self.args.model_snapshot/'model.safetensors.index.json')
        expected_text = {name for name in checkpoint.get('weight_map', {})
                         if name.startswith('model.language_model.') or name == 'lm_head.weight'}
        c.need(bool(expected_text) and self.source_names == expected_text == set(self.checkpoint.text),
               'Native refit source keys are not the exact text parameters from the fixed official checkpoint')
        c.details.update(tp_size=topology.get('tp_size'), gpu_ids=gpu_ids, control_addresses=addresses,
                         required_hf_sources=len(self.source_names), packed_expert_sources=len(self.checkpoint.experts))

    def weight_refit(self, c):
        summary = []
        previous = None
        for step in range(self.args.minimum_updates+1):
            if not c.need(step in self.syncs, f'Missing TP8 weight commit at global step {step}'):
                continue
            path, record = self.syncs[step]
            prefix = f'{path.name}: '
            tensors = record.get('tensors', [])
            names = [item.get('name') for item in tensors]
            c.need(record.get('global_step') == step and record.get('tp_size') == self.args.tp_size, prefix+'Wrong commit step or TP size')
            c.need(record.get('source_training_ranks') == self.expected_source_training_ranks, prefix+'Not all FSDP ranks contributed')
            c.need(record.get('transport') == 'native_rl_cuda_ipc_v2_per_tp_rank', prefix+'Refit did not use the native per-rank IPC path')
            c.need(record.get('source_release') == 'per-tensor all-TP ACK', prefix+'IPC lifetime was not protected by all-rank ACKs')
            c.need(bool(names) and len(names) == len(set(names)) and set(names) == self.source_names,
                   prefix+'Source tensor coverage does not match native loader manifests')
            c.need(record.get('hf_tensor_count') == record.get('native_received_weights') == len(tensors), prefix+'Native received/source tensor counts disagree')
            c.need(positive(record.get('native_applied_weights')), prefix+'Native engine applied no internal tensors')
            c.need(valid_sha(record.get('full_source_sha256')), prefix+'Missing full source payload hash')
            c.need(bool(record.get('model_version')) and record.get('model_version') != record.get('before_model_version'), prefix+'Native model version did not advance')
            ack = record.get('rank_status', [])
            c.need(len(ack) == self.args.tp_size and {x.get('tp_rank') for x in ack} == set(range(self.args.tp_size)), prefix+'Native commit ACK set is incomplete')
            c.need(all(x.get('tp_size') == self.args.tp_size and x.get('global_step') == step and
                       x.get('model_version') == record.get('model_version') and x.get('phase') in ('paused', 'running') for x in ack),
                   prefix+'Native ranks did not acknowledge the same committed policy')
            special = []
            for tensor in tensors:
                name = tensor.get('name', '')
                preserve = name.endswith(('.linear_attn.A_log', '.linear_attn.dt_bias'))
                expected_dtype = 'torch.float32' if preserve else 'torch.bfloat16'
                c.need(tensor.get('dtype') == expected_dtype, prefix+f'Incorrect source dtype: {name}')
                c.need(valid_sha(tensor.get('sha256')) and tensor.get('shape') == self.checkpoint.text.get(name), prefix+f'Invalid tensor payload description: {name}')
                if preserve:
                    special.append(name)
            c.need(set(special) == self.checkpoint.gdn and set(record.get('fp32_sources', [])) == self.checkpoint.gdn, prefix+'GDN FP32 sources were lost or downcast')
            if step == 0:
                observed = {t['name']: t['sha256'] for t in tensors}
                c.need(all(observed.get(name) == expected for name, expected in self.checkpoint.initial_bf16_fingerprints.items()),
                       prefix+'Initial actor/refit tensor bytes differ from the official BF16 checkpoint fingerprints')
            changed = None
            if previous is not None:
                old = {x['name']: x['sha256'] for x in previous.get('tensors', [])}
                changed = sum(old.get(x['name']) != x.get('sha256') for x in tensors)
                c.need(changed > 0 and record.get('full_source_sha256') != previous.get('full_source_sha256'), prefix+'Exported policy payload did not change after training')
                c.need(record.get('before_model_version') == previous.get('model_version'), prefix+'Committed version chain is broken')
            summary.append(dict(global_step=step, model_version=record.get('model_version'),
                                source_sha256=record.get('full_source_sha256'), tensor_count=len(tensors),
                                changed_tensor_hashes=changed, acknowledged_tp_ranks=sorted(x['tp_rank'] for x in ack)))
            previous = record
        c.details.update(commits=summary,
            limitation='Payload hashes are recorded by the actual CUDA IPC exporter. This offline check verifies manifest identity and hash/version changes; it does not reread all GPU tensor payloads.')

    def post_refit_scores(self, c):
        baseline = None
        summary = []
        for step in range(self.args.minimum_updates+1):
            if not c.need(step in self.probes, f'Missing real native post-refit score probe at step {step}'):
                continue
            path, record = self.probes[step]
            values = record.get('raw_log_probs', [])
            targets = record.get('response_ids', [])
            bits = fp32_bytes(values)
            c.need(bool(values) and len(values) == len(targets) and all(math.isfinite(x) for x in values), f'{path.name}: Invalid selected-token raw scores')
            c.need(record.get('raw_fp32_bits_hex') == bits.hex() and record.get('score_sha256') == sha(bits), f'{path.name}: FP32 bytes/hash do not match actual score values')
            c.need(record.get('transport') == 'native_GenerateStreamCall_prompt_target_scores' and record.get('tp_size') == self.args.tp_size,
                   f'{path.name}: Probe is not the native TP8 prompt-target scoring route')
            sync = self.syncs.get(step, (None, {}))[1]
            c.need(record.get('global_step') == step and record.get('model_version') == sync.get('model_version') and
                   record.get('full_source_sha256') == sync.get('full_source_sha256'), f'{path.name}: Probe is not bound to the committed trained policy')
            if step == 0:
                baseline = record
            elif baseline is not None:
                c.need((record.get('prompt_ids'), targets) == (baseline.get('prompt_ids'), baseline.get('response_ids')), f'{path.name}: Probe token inputs changed')
                c.need(bits != fp32_bytes(baseline.get('raw_log_probs', [])), f'{path.name}: Native inference scores stayed identical to the pre-update model')
            distribution = self.read(self.rtp/f'rtp_policy_distribution_step{step}.json')
            signatures = distribution.get('signatures', [])
            c.need(len(signatures) == self.args.world_size and {x.get('tp_rank') for x in signatures} == set(range(self.args.world_size)),
                   f'Step {step}: Distribution signature coverage is incomplete')
            c.need(all(x.get('model_version') == record.get('model_version') and x.get('global_step') == step and
                       x.get('source_sha256') == record.get('full_source_sha256') and
                       fp32_bytes(x.get('raw_log_probs', [])) == bits for x in signatures), f'Step {step}: Training ranks did not receive the same native probe/version')
            summary.append(dict(step=step, model_version=record.get('model_version'), score_sha256=sha(bits),
                                tokens=len(values), observed_change_from_initial=bool(baseline and bits != fp32_bytes(baseline.get('raw_log_probs', [])))))
        c.details.update(probes=summary,
                         scope='One real TP8 forward per probe. Its score signature is broadcast within the group; this is not eight independent model-output parity tests.')

    def generation_logprobs(self, c):
        index = defaultdict(list)
        seen_steps = set()
        for path, record in self.generation_receipts:
            prefix = f'{path.name} request {record.get("request_id")}: '
            c.need(record.get('engine') == 'rtp-llm' and record.get('logprobs_mode') == 'raw_logprobs', prefix+'Native raw-logprob metadata is missing')
            c.need(positive(record.get('prompt_tokens')) and positive(record.get('response_tokens')) and
                   record.get('response_tokens') == record.get('logprobs'), prefix+'Generation token/logprob lengths disagree')
            key = tuple(record.get(name) for name in ('prompt_sha256', 'response_sha256', 'raw_logprobs_fp32_sha256'))
            c.need(all(valid_sha(x) for x in key), prefix+'Generation receipt has missing payload hashes')
            index[key].append(record)
        c.need(bool(index), 'No actual native model generations were recorded')
        for path, episode in self.episodes:
            policy_steps = set()
            for number, step in enumerate(steps(episode)):
                model_output = step.get('model_output') or {}
                prompt = step.get('prompt_ids') or model_output.get('prompt_ids') or []
                response = step.get('response_ids') or model_output.get('completion_ids') or []
                values = step.get('logprobs') or model_output.get('logprobs') or []
                prefix = f'{path.name} trajectory step {number}: '
                valid = bool(prompt) and bool(response) and len(response) == len(values)
                valid &= all(type(x) is int and x >= 0 for x in prompt+response)
                valid &= all(isinstance(x, (int, float)) and math.isfinite(x) and x <= 1e-5 for x in values)
                if not c.need(valid, prefix+'Missing or invalid raw generated token/logprob arrays'):
                    continue
                bits = fp32_bytes(values)
                c.need(list(struct.unpack(f'<{len(values)}f', bits)) == values, prefix+'Raw logprob values were rounded or converted away from native FP32')
                key = (digest_ids(prompt), digest_ids(response), sha(bits))
                receipts = index.get(key, [])
                if not c.need(bool(receipts), prefix+'Actual episode token/logprob payload is not linked to a native generation receipt'):
                    continue
                versions = {(x.get('global_step'), x.get('model_version')) for x in receipts}
                c.need(len(versions) == 1, prefix+'Identical payload is ambiguously associated with different policy versions')
                native_step, version = next(iter(versions))
                policy_steps.add(native_step)
                seen_steps.add(native_step)
                sync = self.syncs.get(native_step, (None, {}))[1]
                c.need(bool(sync) and version == sync.get('model_version'), prefix+'Generation model version is not a recorded committed policy')
                weight_version = step.get('weight_version', model_output.get('weight_version'))
                if weight_version is not None:
                    c.need(weight_version == native_step, prefix+'rLLM generation weight_version differs from the native policy step')
                self.raw_segments.append(dict(episode_id=episode.get('episode_id'), source_path=str(path),
                    prompt_token_ids=prompt, output_token_ids=response, raw_log_probs=values,
                    global_step=native_step, model_version=version,
                    agent_action=step.get('action'),
                    model_response=step.get('model_response', ''),
                    chat_completions=step.get('chat_completions', [])))
            c.need(len(policy_steps) == 1, f'{path.name}: One episode spans missing or multiple model versions')
            if len(policy_steps) == 1:
                self.episode_policy_steps[episode.get('episode_id')] = next(iter(policy_steps))
        for policy_step in range(self.args.minimum_updates):
            c.need(policy_step in seen_steps, f'No real sampled rollout from policy step {policy_step}; cannot establish the update/rollout loop')
        c.need(bool(self.raw_segments), 'No full token-level generated trajectories are available')
        c.details.update(native_receipts=len(self.generation_receipts), linked_trajectory_segments=len(self.raw_segments),
                         sampled_policy_steps=sorted(seen_steps), selected_tokens=sum(len(x['output_token_ids']) for x in self.raw_segments))

    def load_grader_and_tool(self):
        # Hide CUDA only inside this offline validation process before imports.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.path.insert(0, str(self.args.rllm_source))
        sys.path.insert(0, str(self.args.agent_source))
        if self._grader is None:
            module = importlib.import_module('rllm.rewards.math_utils.utils')
            self._grader = (module.grade_answer_mathd, module.grade_answer_sympy)
            self.read_python_source(Path(module.__file__))
        if self._tool_module is None:
            self._tool_module = importlib.import_module('python_math_agent')
            self.read_python_source(Path(self._tool_module.__file__))

    def read_python_source(self, path):
        data = path.read_bytes()
        self.inventory[str(path.resolve())] = {'size_bytes': len(data), 'sha256': sha(data)}

    def grounded_rewards(self, c):
        source = self.read(self.args.source_dataset)
        if not c.need(isinstance(source, list) and len(source) == 40315, 'Original 40,315-row DeepScaleR dataset is absent or replaced'):
            return
        c.need(self.inventory[str(self.args.source_dataset.resolve())]['sha256'] == DEEPSCALER_SHA256,
               'Original DeepScaleR source SHA256 differs from the previously deployed and preserved dataset')
        self.load_grader_and_tool()
        cached = {}
        groups = defaultdict(list)
        matched_training_steps = set()
        for path, episode in self.episodes:
            prefix = f'{path.name}: '
            task = episode.get('task') or {}
            row_number = task.get('source_row_index')
            if not c.need(type(row_number) is int and 0 <= row_number < len(source), prefix+'Missing original DeepScaleR row index'):
                continue
            original = source[row_number]
            question = task.get('original_question', task.get('question', ''))
            truth = str(task.get('ground_truth', task.get('answer', '')))
            c.need(task.get('data_source') == 'deepscaler_math' and normalize(question) == normalize(original['problem']), prefix+'Question is not grounded in the original DeepScaleR row')
            c.need(truth == str(original['answer']), prefix+'Ground-truth answer differs from the original dataset')
            artifacts = episode.get('artifacts') or {}
            answer = str(artifacts.get('answer', ''))
            c.need(answer == self._tool_module.extract_boxed(str(artifacts.get('final_content', ''))), prefix+'Graded answer does not come from the generated final boxed answer')
            key = (answer, truth)
            if key not in cached:
                cached[key] = bool(answer and truth and (self._grader[0](answer, truth) or self._grader[1](answer, truth)))
            correct = cached[key]
            c.need(episode.get('reward') == float(correct) and episode.get('is_correct') is correct, prefix+'Reward was not obtained by the original math correctness evaluator')
            episode_id = episode.get('episode_id')
            logged = self.training_episodes.get(episode_id)
            if c.need(logged is not None, prefix+'This evaluated episode is not linked to a logged real training batch'):
                logged_path, training = logged
                matched_training_steps.add(training.get('training_step'))
                c.need(training.get('is_correct') is correct, f'{logged_path.name}: Training correctness differs from the original evaluator')
            policy_step = self.episode_policy_steps.get(episode_id)
            if policy_step is not None:
                groups[(policy_step, row_number)].append(float(correct))
        mixed_by_step = defaultdict(list)
        group_summary = []
        for (policy_step, row), rewards in sorted(groups.items()):
            mixed = len(set(rewards)) > 1
            if mixed:
                mixed_by_step[policy_step].append(row)
            group_summary.append(dict(policy_step=policy_step, source_row_index=row, samples=len(rewards),
                                      reward_counts=dict(Counter(rewards)), mixed=mixed))
        for policy_step in range(self.args.minimum_updates):
            c.need(bool(mixed_by_step[policy_step]), f'Policy step {policy_step}: No same-question group has genuine nonuniform rewards')
        c.need(len(matched_training_steps) >= self.args.minimum_updates, 'Fewer than two distinct real training batches were logged')
        c.need(bool(self.episodes), 'No original PythonMathEvaluator enriched trajectories exist')
        c.details.update(source_dataset=str(self.args.source_dataset), source_rows=len(source),
                         source_sha256=self.inventory[str(self.args.source_dataset.resolve())]['sha256'],
                         real_episodes=len(self.episodes), logged_training_steps=sorted(x for x in matched_training_steps if x is not None),
                         groups=group_summary,
                         reward_definition='Original final boxed-answer correctness only. No reward or tool-call injection.')

    def python_tools(self, c):
        self.load_grader_and_tool()
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.args.model_snapshot, local_files_only=True)
        sys.path.insert(0, str(self.args.project_root/'adapter'))
        parser_module = importlib.import_module('rtp_gateway_adapter')
        self.read_python_source(Path(parser_module.__file__))
        parse = parser_module.parse_generated_tool_calls
        generated_by_episode = defaultdict(list)
        final_contents_by_episode = defaultdict(list)
        checked_prompts = 0
        for segment in self.raw_segments:
            raw = self._tokenizer.decode(segment['output_token_ids'], skip_special_tokens=False)
            parsed = parse(raw, self._tool_module.TOOLS)
            c.need(parsed['content'] == segment['model_response'],
                   f'{segment["source_path"]}: Visible model response is not decoded from the actual native generated tokens')
            c.need(parsed['content'] == segment['agent_action'],
                   f'{segment["source_path"]}: Original agent action does not match the delivered native token response')
            generated_by_episode[segment['episode_id']].extend(parsed['tool_calls'])
            if not parsed['tool_calls']:
                final_contents_by_episode[segment['episode_id']].append(parsed['content'])
            history = segment['chat_completions']
            if c.need(bool(history) and history[-1].get('role') == 'assistant',
                      f'{segment["source_path"]}: Actual agent prompt history is absent'):
                prompt = self._tokenizer.apply_chat_template(
                    parser_module.normalize_tool_history(history[:-1]), tools=self._tool_module.TOOLS,
                    tokenize=False, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
                prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
                c.need(prompt_ids == segment['prompt_token_ids'],
                       f'{segment["source_path"]}: Actual native prompt tokens differ from the official full Python Tool chat template')
                checked_prompts += 1
        successful = []
        counts = Counter()
        for path, episode in self.episodes:
            artifacts = episode.get('artifacts') or {}
            final_content = artifacts.get('final_content', '')
            if final_content:
                c.need(final_content in final_contents_by_episode[episode.get('episode_id')],
                       f'{path.name}: Graded final answer was not emitted as actual native model text')
            calls = artifacts.get('python_tools', [])
            messages = [message for step in steps(episode) for message in step.get('chat_completions', [])]
            generated = generated_by_episode[episode.get('episode_id')]
            metrics = episode.get('metrics') or {}
            c.need(metrics.get('python_tool_calls') == len(calls), f'{path.name}: Python-call metric disagrees with actual recorded executions')
            c.need(metrics.get('python_tool_successes') == sum(bool(x.get('result', {}).get('ok')) for x in calls), f'{path.name}: Python-success metric disagrees with actual results')
            for call in calls:
                code, call_id = call.get('code'), call.get('tool_call_id')
                result = call.get('result') or {}
                prefix = f'{path.name} call {call_id}: '
                counts['recorded_calls'] += 1
                c.need(isinstance(code, str) and bool(code.strip()) and bool(call_id), prefix+'Empty code or call ID')
                c.need(any(x.get('name') == 'python' and x.get('arguments', {}).get('code') == code for x in generated), prefix+'Executed Python code is not present in actual generated tool-call tokens')
                linked = False
                for message in messages:
                    for tool in message.get('tool_calls', []):
                        function = tool.get('function') or {}
                        arguments = function.get('arguments') or {}
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except ValueError:
                                arguments = {}
                        if tool.get('id') == call_id and function.get('name') == 'python' and arguments.get('code') == code:
                            linked = True
                c.need(linked, prefix+'Executed call does not match the assistant tool-call ID/code')
                returns = [x for x in messages if x.get('role') == 'tool' and x.get('tool_call_id') == call_id]
                for returned in returns:
                    c.need(json.loads(returned.get('content', '{}')) == result, prefix+'Returned tool message differs from the actual execution result')
                if returns:
                    counts['calls_with_model_visible_return'] += 1
                c.need(type(result.get('ok')) is bool and isinstance(result.get('stdout'), str), prefix+'Malformed Python sandbox result')
                if result.get('ok'):
                    c.need(positive(result.get('elapsed_s')), prefix+'Successful sandbox execution has no positive elapsed time')
                    counts['successful_calls'] += 1
                    successful.append((path, code, result))
        c.need(counts['successful_calls'] > 0 and counts['calls_with_model_visible_return'] > 0, 'No successful real Python Tool execution with a result returned to the model')
        replays = []
        unique = set()
        for path, code, expected in successful:
            if code in unique:
                continue
            unique.add(code)
            actual = self._tool_module.execute_python(code)
            match = actual.get('ok') == expected.get('ok') and actual.get('stdout') == expected.get('stdout')
            c.need(match, f'{path.name}: Original sandbox replay does not reproduce the recorded generated Python result')
            replays.append(dict(episode=str(path), code_sha256=sha(code.encode()),
                                stdout_sha256=sha(str(actual.get('stdout', '')).encode()), reproduced=match))
            if len(replays) >= self.args.tool_replay_limit:
                break
        c.need(bool(replays), 'No real generated Python call could be replayed with the original sandbox')
        c.details.update(counts=dict(counts), independently_replayed_calls=replays,
                         official_template_prompts_checked=checked_prompts,
                         scope='Original execution artifacts + native generated XML/JSON call tokens + matching tool messages. CPU replays verify reproducible results; they are not GPU-training evidence.')

    def tensor_values(self, c, record, label, *, allow_empty=False):
        if not c.need(isinstance(record, dict), label+': Missing actual tensor observation'):
            return []
        raw = bytes.fromhex(record.get('selected_fp32_bits_hex', ''))
        count = record.get('selected_numel')
        if not c.need(type(count) is int and (count > 0 or (allow_empty and count == 0)) and len(raw) == count*4, label+': Invalid observed FP32 tensor byte length'):
            return []
        values = list(struct.unpack(f'<{count}f', raw))
        finite = all(math.isfinite(x) for x in values)
        c.need(finite and record.get('finite') is True, label+': Nonfinite tensor values')
        c.need(record.get('selected_fp32_sha256') == sha(raw), label+': Tensor byte/hash mismatch')
        c.need(record.get('nonzero') == sum(x != 0 for x in values) and
               record.get('positive') == sum(x > 0 for x in values) and
               record.get('negative') == sum(x < 0 for x in values), label+': Observed tensor sign counts are inconsistent')
        if finite:
            c.need(record.get('minimum') == (min(values) if values else None) and
                   record.get('maximum') == (max(values) if values else None),
                   label+': Recorded extrema disagree with actual tensor bytes')
        return values

    def delivered_trace_linkage(self, c):
        """Bind actual agent messages to selected native payloads after retries."""
        source = self.args.project_root/'training/overrides/qwen36_trace_enrichment.py'
        self.read_python_source(source)
        spec = importlib.util.spec_from_file_location('qwen36_linkage_validation', source)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        records = self.records(self.root/'trace-linkage', 'trace-linkage-*.json')
        episodes = {episode.get('episode_id'): episode for _, episode in self.episodes}
        seen = Counter()
        observed_payloads = Counter()
        orphaned_count = selected_count = 0
        for path, record in records:
            uid = record.get('uid')
            seen[uid] += 1
            prefix = path.name+': '
            c.need(record.get('schema') == 'qwen36-delivered-trace-linkage-v1' and
                   record.get('passed') is True and record.get('strict') is True,
                   prefix+'Strict delivered-message trace association did not succeed')
            c.need(record.get('tokens_logprobs_and_agent_artifacts_modified') is False,
                   prefix+'Trace selection modified native payloads or original agent artifacts')
            episode = episodes.get(uid)
            if not c.need(episode is not None, prefix+'Trace audit is not linked to an actual training episode'):
                continue
            actual_steps = list(steps(episode))
            selected = record.get('selected', [])
            orphaned = record.get('orphaned', [])
            observed = record.get('observed_traces', [])
            c.need(record.get('agent_steps') == record.get('selected_count') == len(selected) == len(actual_steps),
                   prefix+'Selected traces do not cover every delivered agent step')
            c.need(record.get('orphaned_count') == len(orphaned), prefix+'Orphan count is inconsistent')
            identifiers = [item.get('trace_id') for item in observed]
            c.need(all(isinstance(value, str) and value for value in identifiers) and
                   len(set(identifiers)) == len(identifiers), prefix+'Observed trace identifiers are missing/duplicated')
            by_id = {item.get('trace_id'): item for item in observed}
            partition = [item.get('trace_id') for item in selected+orphaned]
            c.need(Counter(partition) == Counter(identifiers), prefix+'Observed traces are not exactly partitioned into delivered and retry-only calls')
            request_hashes = set()
            policy_step = self.episode_policy_steps.get(uid)
            for number, item in enumerate(selected):
                if number >= len(actual_steps):
                    break
                step = actual_steps[number]
                history = helper._history(step.get('chat_completions', []))
                c.need(bool(history) and history[-1]['role'] == 'assistant', prefix+'Delivered assistant history is missing')
                if not history:
                    continue
                c.need(item.get('agent_step') == number and item.get('session_id') == uid,
                       prefix+'Selected trace order/session differs from the delivered step')
                c.need(item.get('delivered_history_sha256') == sha(helper._json(history).encode()) and
                       item.get('request_history_sha256') == sha(helper._json(history[:-1]).encode()) and
                       item.get('assistant_sha256') == sha(helper._json(history[-1]).encode()),
                       prefix+'Selected trace does not represent the exact delivered message/history')
                prompt, response, values = step.get('prompt_ids', []), step.get('response_ids', []), step.get('logprobs', [])
                c.need((item.get('prompt_sha256'), item.get('response_sha256'), item.get('raw_logprobs_fp32_sha256')) ==
                       (digest_ids(prompt), digest_ids(response), sha(fp32_bytes(values))),
                       prefix+'Selected native trace payload differs from the actual trained step')
                c.need(item.get('prompt_tokens') == len(prompt) and item.get('response_tokens') == item.get('logprobs') == len(response),
                       prefix+'Selected trace token/score lengths differ from the trained step')
                c.need(item.get('tool_call_ids') == [call['id'] for call in history[-1].get('tool_calls', [])],
                       prefix+'Delivered tool-call identity differs from its selected trace')
                request_hashes.add(item.get('request_history_sha256'))
            for item in selected+orphaned:
                original = by_id.get(item.get('trace_id'), {})
                c.need(bool(original) and all(item.get(key) == value for key, value in original.items()),
                       prefix+'Selected/orphan trace summary differs from its observed original')
                c.need(item.get('session_id') == uid and item.get('weight_version') == policy_step,
                       prefix+'Observed trace session or policy version is inconsistent')
                c.need(isinstance(item.get('response_id'), str) and bool(item.get('response_id')),
                       prefix+'Observed response ID is missing')
                observed_payloads[(item.get('prompt_sha256'), item.get('response_sha256'),
                                   item.get('raw_logprobs_fp32_sha256'), policy_step)] += 1
            c.need(all(item.get('request_history_sha256') in request_hashes for item in orphaned),
                   prefix+'Discarded extra trace is not a retry of a delivered request')
            selected_count += len(selected)
            orphaned_count += len(orphaned)
        c.need(set(seen) == set(episodes) and all(count == 1 for count in seen.values()),
               'Exactly one successful trace-linkage audit is required for every actual training episode')
        native_payloads = Counter((item.get('prompt_sha256'), item.get('response_sha256'),
                                   item.get('raw_logprobs_fp32_sha256'), item.get('global_step'))
                                  for _, item in self.generation_receipts)
        c.need(observed_payloads == native_payloads,
               'Delivered plus retry-only traces do not account for the actual native generation receipts')
        c.details.update(episodes=len(seen), selected_agent_steps=selected_count,
                         retry_only_calls_excluded_from_training=orphaned_count,
                         native_generation_calls=sum(native_payloads.values()),
                         scope='Every native generation is accounted for. Only exact agent-delivered histories select training token/logprob payloads; retry-only responses are excluded without changing the agent or rewards.')

    def actual_advantages(self, c):
        records = self.records(self.root/'policy-loss', 'policy-loss-rank*.json')
        by_rank = defaultdict(list)
        for _, record in self.optimizers:
            by_rank[record.get('rank')].append(record)
        for values in by_rank.values():
            values.sort(key=lambda x: x.get('time_ns', 0))
        observed = defaultdict(lambda: defaultdict(list))
        absent_fields = Counter()
        padding_events = 0
        for path, record in records:
            rank, timestamp = record.get('rank'), record.get('time_ns', 0)
            prefix = path.name+': '
            c.need(record.get('schema') == 'qwen36-real-policy-loss-v1' and record.get('world_size') == self.args.world_size,
                   prefix+'Policy loss was not observed on the real eight-rank actor')
            c.need(record.get('loss_mode') == 'vanilla' and record.get('loss_requires_grad') is True and bool(record.get('loss_grad_fn')),
                   prefix+'Observed policy loss is not the real autograd PPO/GRPO objective')
            advantage = record.get('advantages') or {}
            c.need(str(advantage.get('device', '')).startswith('cuda:'), prefix+'CPU fixture advantage is not full-model GPU evidence')
            mask = record.get('response_mask') or {}
            # verl pads rows to the eight-rank DP divisor. Such rows still pass
            # through the real loss, with zero active tokens; their selected
            # advantage byte string is legitimately empty. Only this explicitly
            # observed all-zero CUDA response mask permits an empty selection.
            padding = (
                type(mask.get('numel')) is int and mask['numel'] > 0 and
                mask.get('selected_numel') == mask['numel'] and
                mask.get('nonzero') == mask.get('positive') == mask.get('negative') == 0 and
                mask.get('minimum') == mask.get('maximum') == mask.get('sum') == 0 and
                mask.get('finite') is True and str(mask.get('device', '')).startswith('cuda:') and
                mask.get('shape') == advantage.get('shape') and
                mask.get('numel') == advantage.get('numel'))
            values = self.tensor_values(c, advantage, prefix+'advantages', allow_empty=padding)
            padding_events += int(padding)
            self.tensor_values(c, record.get('loss'), prefix+'returned loss')
            c.need(mask.get('nonzero') == len(values), prefix+'Advantage selection does not match actual response-mask tokens')
            context = record.get('batch_context') or {}
            for field in ('rewards', 'group_ids', 'step_fields'):
                if not context.get(field):
                    absent_fields[field] += 1
            for key, tensor in context.get('rewards', {}).items():
                self.tensor_values(c, tensor, prefix+'actual batch '+key)
            later = [x for x in by_rank[rank] if x.get('time_ns', 0) > timestamp]
            if not c.need(bool(later), prefix+'Policy-loss event cannot be linked to a subsequent real optimizer update on this rank'):
                continue
            step = int(later[0]['optimizer_step_max'])
            observed[step][rank].extend(values)
        summary = []
        for step in range(1, self.args.minimum_updates+1):
            ranks = observed[step]
            all_values = [x for values in ranks.values() for x in values]
            c.need(set(ranks) == set(range(self.args.world_size)), f'Update {step}: Actual policy-loss inputs are missing on one or more GPUs')
            c.need(any(x < 0 for x in all_values) and any(x > 0 for x in all_values), f'Update {step}: Actual GRPO advantages lack nonzero positive/negative values')
            summary.append(dict(optimizer_step=step, ranks=sorted(ranks), observed_active_tokens=len(all_values),
                                positive=sum(x > 0 for x in all_values), negative=sum(x < 0 for x in all_values),
                                minimum=min(all_values) if all_values else None, maximum=max(all_values) if all_values else None))
        c.details.update(actual_policy_loss_events=len(records), zero_mask_padding_events=padding_events, updates=summary,
                         missing_batch_context_fields=dict(absent_fields),
                         linkage='Actual loss inputs are assigned to the next observed optimizer step on the same rank using nanosecond timestamps. Original episode rewards are independently validated; no reconstructed advantage is accepted as an observed tensor.')

    def hf_reference(self, c):
        paths = self.args.hf_comparison
        if not c.need(bool(paths), 'Provide --hf-comparison with an actual full-model GPU HF selected-token comparison; configuration-only/CPU proofs do not count'):
            return
        signatures = set()
        for segment in self.raw_segments:
            signatures.add((tuple(segment['prompt_token_ids']), tuple(segment['output_token_ids']), sha(fp32_bytes(segment['raw_log_probs']))))
        for _, probe in self.probes.values():
            signatures.add((tuple(probe.get('prompt_ids', [])), tuple(probe.get('response_ids', [])), sha(fp32_bytes(probe.get('raw_log_probs', [])))))
        summary = []
        for path in paths:
            record = self.read(path)
            if record.get('schema') == 'qwen35-397b-rtp-tp8-hf-reference-v1':
                summary.append(self.harness_hf_reference(c, path, record))
                continue
            c.need(record.get('schema') == 'qwen35-397b-hf-selected-logprob-comparison-v1' and
                   str(record.get('device', '')).startswith('cuda:') and all(record.get(k) == v for k, v in identity().items()) and
                   Path(record.get('checkpoint','missing')).resolve() == self.args.model_snapshot.resolve() and
                   record.get('cuda_numel') == self.checkpoint.text_numel + self.checkpoint.visual_numel and
                   record.get('peak_allocated_bytes', 0) >= self.checkpoint.active_bf16_bytes,
                   f'{path}: HF comparison did not use the fixed full-model GPU checkpoint')
            c.need(record.get('parameter_dtype') == 'bfloat16' and 'FP32' in str(record.get('comparison_dtype', '')),
                   f'{path}: Wrong reference precision path')
            c.need(record.get('gdn_scalar_dtype') == 'bf16',
                   f'{path}: FP32 GDN diagnostic does not match the actual FSDP BF16 forward precision path')
            errors = []
            cases = record.get('cases', [])
            c.need(bool(cases), f'{path}: No actual selected-token HF comparison cases')
            for case in cases:
                expected, actual = case.get('raw_log_probs', []), case.get('hf_log_probs', [])
                targets = case.get('output_token_ids', [])
                valid = bool(expected) and len(expected) == len(actual) == len(targets)
                valid &= all(math.isfinite(x) for x in expected+actual)
                if not c.need(valid, f'{path}: Invalid or nonfinite HF/raw token arrays'):
                    continue
                c.need((tuple(case.get('prompt_token_ids', [])), tuple(targets), sha(fp32_bytes(expected))) in signatures,
                       f'{path}: HF case is not linked to a real native generation or post-refit probe in this RUN_ROOT')
                differences = [a-b for a, b in zip(actual, expected)]
                errors.extend(abs(x) for x in differences)
                if 'hf_minus_rtp' in case:
                    c.need(case['hf_minus_rtp'] == differences, f'{path}: Reported per-token errors do not match actual values')
            maximum = max(errors) if errors else None
            c.need(bool(errors) and maximum <= self.args.logprob_tolerance, f'{path}: Actual max selected-token logprob error exceeds {self.args.logprob_tolerance}')
            summary.append(dict(path=str(path), selected_tokens=len(errors), recomputed_max_abs_error=maximum,
                                recomputed_mean_abs_error=sum(errors)/len(errors) if errors else None,
                                gdn_scalar_dtype=record.get('gdn_scalar_dtype')))
        c.details.update(comparisons=summary, tolerance=self.args.logprob_tolerance,
                         scope='Independent HF forward numerics on supplied native token cases. This does not itself validate FSDP backward or optimizer updates.')

    def harness_hf_reference(self, c, path, record):
        """Read the existing full-checkpoint HF + TP8-native harness as written.

        Its companion reference and process topology provide the provenance;
        fields from the newer reference script are deliberately not invented.
        """
        prefix = str(path)+': '
        reference_path = Path(record.get('reference_path', 'missing-reference.json'))
        if not reference_path.is_absolute():
            reference_path = path.parent/reference_path
        reference = self.read(reference_path)
        topology = self.read(path.parent/'native-topology.json')
        controller = self.read(path.parent/'controller.json')
        c.need(record.get('tp_size') == record.get('native_gpu_count') == 8,
               prefix+'HF/native harness did not use native TP8 on eight GPUs')
        c.need(record.get('checkpoint_identity') == controller.get('checkpoint_identity') == identity() and
               reference.get('schema') == 'qwen35-397b-hf-cold-reference-v1' and
               all(reference.get(k) == v for k, v in identity().items()) and
               Path(record.get('model_path', 'missing-model')).resolve() == self.args.model_snapshot.resolve(),
               prefix+'Harness model path differs from the fixed official snapshot')
        c.need(reference.get('model_path') == record.get('model_path') == controller.get('model_path'),
               prefix+'Reference, native controller and comparison used different checkpoints')
        c.need(reference.get('dtype') == 'bfloat16' and reference.get('mtp') is False and
               str(reference.get('device', '')).startswith('cuda:'),
               prefix+'HF reference is not the real BF16 full-checkpoint GPU path')
        c.need(reference.get('peak_allocated_bytes', 0) >= self.checkpoint.active_bf16_bytes and
               reference.get('cuda_numel') == self.checkpoint.text_numel + self.checkpoint.visual_numel and
               reference.get('model_class') == 'Qwen3_5MoeForConditionalGeneration',
               prefix+'Full-checkpoint BF16 residency was not observed on the HF CUDA device')
        native = topology.get('topology') or {}
        c.need(native.get('tp_size') == native.get('world_size') == 8 and
               native.get('dp_size') == native.get('pp_size') == 1,
               prefix+'Companion native topology is not one TP8 group')
        manifests = topology.get('rank_manifests', [])
        statuses = topology.get('rank_status', [])
        processes = topology.get('native_gpu_processes', [])
        pids = {x.get('pid') for x in manifests}
        manifest_gpu_by_pid = {x.get('pid'): canonical_gpu_uuid(x.get('gpu_uuid')) for x in manifests}
        uuids = set(manifest_gpu_by_pid.values())
        c.need(len(manifests) == 8 and {x.get('tp_rank') for x in manifests} == set(range(8)) and
               all(x.get('tp_size') == 8 for x in manifests), prefix+'Native rank manifests are incomplete')
        c.need(len(pids) == len(uuids) == 8 and all(isinstance(x, int) and x > 0 for x in pids) and
               all(isinstance(x, str) and len(x) > 12 for x in uuids),
               prefix+'Native manifests do not identify eight real processes and distinct physical GPUs')
        c.need(len(processes) == 8 and {x.get('pid') for x in processes} == pids and
               {x.get('pid'): canonical_gpu_uuid(x.get('gpu_uuid')) for x in processes} == manifest_gpu_by_pid and
               all(positive(float(x.get('used_memory_mib', '0'))) for x in processes),
               prefix+'nvidia-smi process/GPU memory map does not confirm all eight native ranks')
        c.need(uuids.issubset({canonical_gpu_uuid(x.get('uuid')) for x in topology.get('devices', [])}),
               prefix+'Native process GPUs are absent from the physical device inventory')
        c.need(canonical_gpu_uuid(reference.get('gpu_uuid')) in uuids,
               prefix+'HF reference device differs from the selected eight GPUs')
        c.need(uuids == self.optimizer_uuids, prefix+'Cold native reference and actual training did not use the same eight physical GPUs')
        c.need(len(statuses) == 8 and {x.get('tp_rank') for x in statuses} == set(range(8)) and
               all(x.get('tp_size') == 8 and x.get('phase') == 'running' for x in statuses),
               prefix+'Eight live native rank statuses are missing')
        if self.source_names:
            source_layouts = [cold_loader_source_layout(x.get('names', []), self.source_names, self.checkpoint.num_layers, self.checkpoint.num_experts) for x in manifests]
            c.need(all(source_layouts) and len(set(source_layouts)) == 1,
                   prefix+'Cold native harness sources are neither exact physical HF keys nor the exact historical MoE expansion')
            for number, manifest in enumerate(manifests):
                names = manifest.get('names', [])
                c.need(len(names) == len(set(names)), prefix+f'Cold rank {number} has duplicate source names')
        baseline_cases = reference.get('cases', [])
        cases = record.get('cases', [])
        c.need(len(cases) == len(baseline_cases) and len(cases) >= 3,
               prefix+'Original three actual HF/native token cases are missing')
        errors = []
        for number, case in enumerate(cases):
            prompt = case.get('prompt_token_ids', [])
            targets = case.get('output_token_ids', [])
            expected = case.get('hf_raw_log_probs', [])
            actual = case.get('rtp_raw_log_probs', [])
            valid = bool(prompt) and bool(targets) and len(targets) == len(expected) == len(actual)
            valid &= all(type(x) is int and x >= 0 for x in prompt+targets)
            valid &= all(isinstance(x, (int, float)) and math.isfinite(x) for x in expected+actual)
            if not c.need(valid, prefix+f'Case {number} has invalid actual token/logprob arrays'):
                continue
            original = baseline_cases[number] if number < len(baseline_cases) else {}
            c.need((prompt, targets, expected) == (original.get('prompt_token_ids'), original.get('output_token_ids'), original.get('raw_log_probs')),
                   prefix+f'Case {number} is not linked to the independent full-checkpoint HF token case')
            c.need(all(x.get('model_version') == case.get('model_version') and
                       x.get('global_step') == case.get('global_step') for x in statuses),
                   prefix+f'Case {number} used a native version different from the eight live rank statuses')
            differences = [abs(a-b) for a, b in zip(actual, expected)]
            errors.extend(differences)
            c.need(max(differences) <= self.args.logprob_tolerance,
                   prefix+f'Case {number} actual selected-token error exceeds {self.args.logprob_tolerance}')
            c.need(math.isclose(case.get('max_abs_error', -1), max(differences), rel_tol=1e-10, abs_tol=1e-12),
                   prefix+f'Case {number} recorded max error does not match actual token arrays')
        maximum = max(errors) if errors else None
        c.need(bool(errors) and maximum <= self.args.logprob_tolerance,
               prefix+'No valid full-model HF/native comparison satisfies the requested tolerance')
        return dict(path=str(path), schema=record.get('schema'), reference_path=str(reference_path),
                    selected_tokens=len(errors), recomputed_max_abs_error=maximum,
                    recomputed_mean_abs_error=sum(errors)/len(errors) if errors else None,
                    hf_peak_allocated_bytes=reference.get('peak_allocated_bytes'), native_physical_gpu_uuids=sorted(uuids),
                    source_layouts=source_layouts if self.source_names else [],
                    scope='Independent official-checkpoint BF16 HF and cold TP8 native teacher forcing. It is not a post-training HF reload or a sampled RL trajectory comparison.')

    def cleanup(self, c):
        identity = self.read(self.rtp/'rtp_process_0.json')
        shutdown = self.read(self.rtp/'rtp_shutdown_0.json')
        driver = self.read(self.rtp/'driver-cleanup.json')
        c.need(driver.get('passed') is True and driver.get('audit_dir') == str(self.rtp),
               'Final driver cleanup did not succeed for this exact run')
        replicas = driver.get('replicas', [])
        if not c.need(len(replicas) == 1, 'Final cleanup must cover the one TP8 replica'):
            return
        item = replicas[0]
        process = item.get('process_receipt') or {}
        c.need(item.get('passed') is True and item.get('identity_file') == str(self.rtp/'rtp_process_0.json'),
               'Cleanup receipt does not identify this native replica')
        c.need(process.get('process_tree_terminated') is True and shutdown.get('process_tree_terminated') is True,
               'Native process-tree termination acknowledgments are missing')
        for field in ('pid', 'pgid', 'create_time'):
            c.need(process.get(field) == identity.get(field) and positive(identity.get(field)),
                   f'Cleanup process identity mismatch: {field}')
        for field in ('pgid', 'create_time'):
            c.need(shutdown.get(field) == identity.get(field), f'Native shutdown identity mismatch: {field}')
        manifests = [self.read(self.rtp/'rtp_weight_manifest_0'/f'rank-{rank:05d}.json')
                     for rank in range(self.args.tp_size)]
        rank_pids = {x.get('pid') for x in manifests}
        c.need(len(rank_pids) == 8 and all(type(pid) is int and pid > 0 for pid in rank_pids),
               'Eight native process identities are not recorded')
        required = rank_pids | {identity.get('pid')} | {x.get('pid') for x in identity.get('descendants', [])}
        c.need(required.issubset(set(process.get('owned_pids', []))) and
               required.issubset(set(shutdown.get('owned_pids', []))),
               'Final cleanup omitted a native TP rank or its leader')
        c.need(positive(driver.get('started_at_unix')) and
               driver.get('finished_at_unix', 0) >= driver['started_at_unix'],
               'Final driver cleanup timestamps are inconsistent')
        final_commit = self.syncs.get(self.args.minimum_updates, (None, {}))[1]
        c.need(positive(final_commit.get('finished_unix')) and
               shutdown.get('finished_at_unix', 0) > final_commit['finished_unix'] and
               driver.get('finished_at_unix', 0) >= shutdown.get('finished_at_unix', 0),
               'Final native cleanup did not occur after the last real weight commit')
        c.details.update(owned_native_pids=sorted(set(process.get('owned_pids', []))),
                         method=item.get('method'), prior_actor_rpc_error=item.get('rpc_error'),
                         scope='Recorded identity-checked cleanup; a separately observed empty nvidia-smi process list confirms the live final state.')

    def training_exit(self, c):
        launch = self.read(self.root/'launch.json')
        status = self.read(self.root/'exit.json')
        cleanup = self.read(self.rtp/'driver-cleanup.json')
        c.need(status.get('returncode') == 0 and type(status.get('returncode')) is int,
               'The actual training driver did not exit successfully')
        c.need(status.get('pid') == launch.get('pid') and positive(launch.get('pid')),
               'Exit status does not identify the launched training driver')
        c.need(status.get('source_commit') == launch.get('source_commit') and
               isinstance(launch.get('source_commit'), str) and len(launch['source_commit']) == 40,
               'Training launch/exit source commit is not recorded consistently')
        c.need(status.get('started_unix') == launch.get('started_unix') and
               positive(status.get('started_unix')) and
               status.get('finished_unix', 0) >= cleanup.get('finished_at_unix', float('inf')),
               'Successful driver exit was not recorded after native cleanup')
        c.details.update(pid=status.get('pid'), returncode=status.get('returncode'),
                         source_commit=status.get('source_commit'),
                         finished_unix=status.get('finished_unix'))

    def cpu_proof(self, c):
        if not self.args.cpu_proof:
            c.details.update(status='not_requested', counts_as_full_model_gpu_evidence=False)
            return
        record = self.read(self.args.cpu_proof)
        c.need(record.get('schema') == 'qwen35-397b-training-meta-contract-v1' and
               all(record.get(k) == v for k, v in identity().items()) and
               record.get('parameter_count') == self.checkpoint.text_numel + self.checkpoint.visual_numel and
               record.get('text_numel') == self.checkpoint.text_numel and
               record.get('visual_numel') == self.checkpoint.visual_numel and
               record.get('missing_checkpoint_names') == [] and
               set(record.get('ignored_checkpoint_mtp_names', [])) == set(self.checkpoint.mtp) and
               record.get('packed_expert_parameter_count') == len(self.checkpoint.experts) and
               {r.get('name') for r in record.get('gdn_fp32_parameters', []) if r.get('dtype') == 'torch.float32'} == self.checkpoint.gdn,
               'CPU/meta names, parameter counts or precision disagree with the official shard headers')
        tiny = record.get('cpu_tiny_test', {})
        c.need(tiny.get('visual_frozen') is True and tiny.get('real_optimizer_update') is True and
               tiny.get('cuda_initialized') is False and isinstance(tiny.get('loss'), (float, int)) and math.isfinite(tiny['loss']),
               'The CPU autograd/AdamW smoke is missing')
        c.need(record.get('cuda_initialized') is False, 'CPU proof unexpectedly initialized CUDA')
        c.details.update(path=str(self.args.cpu_proof), counts_as_full_model_gpu_evidence=False,
                         scope='Meta parameter naming and a tiny CPU implementation check only')

    def execute(self):
        for name, function in [
            ('artifact_layout', self.load_records),
            ('all_pinned_official_model_objects_current_full_sha256', self.official_model_objects),
            ('full_122b_model_on_8_training_ranks', self.model_load),
            ('two_real_adamw_updates_on_8_distinct_gpus', self.optimizer_updates),
            ('native_tp8_topology_and_loader_sources', self.topology),
            ('explicit_arm_bf16_backend_bound_to_all_native_ranks', self.native_math_backend),
            ('complete_distributed_refit_and_policy_version_advance', self.weight_refit),
            ('resident_native_pause_before_both_real_updates', self.training_pauses),
            ('real_post_refit_native_score_change', self.post_refit_scores),
            ('native_fp32_generation_logprobs_linked_to_episodes', self.generation_logprobs),
            ('original_deepscaler_rewards_and_nonuniform_groups', self.grounded_rewards),
            ('genuine_generated_python_tool_execution', self.python_tools),
            ('exact_delivered_trace_linkage_and_retry_exclusion', self.delivered_trace_linkage),
            ('observed_gpu_policy_loss_and_nonuniform_advantages', self.actual_advantages),
            ('full_model_hf_selected_token_numerical_comparison', self.hf_reference),
            ('identity_checked_native_process_cleanup', self.cleanup),
            ('actual_training_driver_exited_successfully', self.training_exit),
        ]:
            self.run_check(name, function)
        self.run_check('optional_cpu_prerequisite_proof', self.cpu_proof, required=bool(self.args.cpu_proof))
        self.run_check('all_input_artifacts_parse_and_hash', lambda c: (
            c.need(not self.input_errors, '; '.join(self.input_errors)),
            c.details.update(artifact_count=len(self.inventory), errors=self.input_errors)))
        passed = all(check['passed'] for check in self.checks if check['required'])
        return dict(schema='qwen35-397b-eight-gpu-real-rl-acceptance-v1',
                    run_root=str(self.root), created_unix=time.time(),
                    expected_model=MODEL_ID, expected_revision=REVISION,
                    expected_training_world_size=self.args.world_size, expected_rollout_tp_size=self.args.tp_size,
                    minimum_real_updates=self.args.minimum_updates,
                    passed=passed, status='accepted' if passed else 'not_accepted',
                    checks=self.checks, evidence_files=self.inventory,
                    scope_notes=[
                        'This validator starts no CUDA work and never treats a tiny CPU test as a 397B GPU training pass.',
                        'The native TP probe is one distributed model computation with a broadcast signature, not eight independent inference replicas.',
                        'Offline checks establish consistency of actual recorded artifacts; original producer passed flags do not substitute for tensor/trajectory checks.',
                        'Resource saturation and end-to-end RL performance are separate from this functional acceptance.',
                    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--model-snapshot', type=Path, default=default_model(PROJECT))
    parser.add_argument('--source-dataset', type=Path, default=LEGACY/'data/deepscaler.json')
    parser.add_argument('--agent-source', type=Path, default=LEGACY/'agent')
    parser.add_argument('--rllm-source', type=Path, default=LEGACY/'source')
    parser.add_argument('--minimum-updates', type=int, default=2)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--tp-size', type=int, default=8)
    parser.add_argument('--tool-replay-limit', type=int, default=3)
    parser.add_argument('--logprob-tolerance', type=float, default=.05)
    parser.add_argument('--hf-comparison', type=Path, action='append', default=[])
    parser.add_argument('--cpu-proof', type=Path)
    args = parser.parse_args()
    if args.minimum_updates < 2 or args.world_size != 8 or args.tp_size != 8 or args.tool_replay_limit < 1:
        parser.error('This acceptance requires at least two updates, training world8, native TP8, and at least one actual tool replay')
    if not 0 < args.logprob_tolerance <= .05:
        parser.error('Selected-token logprob tolerance must be in (0, 0.05]')
    # The validator is always a CPU process, including tokenizer/grader imports.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report = Validator(args).execute()
    output = args.output or args.run_root/'validation/rl-run-acceptance.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'], output=str(output),
                          checks=len(report['checks']),
                          failed_checks=[x['name'] for x in report['checks'] if x['required'] and not x['passed']]), ensure_ascii=False))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
