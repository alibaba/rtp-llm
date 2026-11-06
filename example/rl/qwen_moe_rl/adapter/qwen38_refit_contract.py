"""Pure CPU guards for the exact world72 canonical source-lease protocol."""
from __future__ import annotations

import hashlib
import json
import math
import struct

WORLD = 72
MAIN_SOURCES = 1590


def validate_combined_verl_mesh(mesh, world_rank):
    """verl combines rollout TP×attention-DP in its ``infer_tp`` dimension.

    This worker mesh is (dp=1, infer_tp=72, infer_pp=1); the independent native
    mapping remains TP rank r%8 and attention-DP rank r//8.
    """
    names=tuple(mesh.mesh_dim_names or ())
    sizes={name:int(mesh.size(name)) for name in names}
    if names!=('dp','infer_tp','infer_pp') or sizes!={'dp':1,'infer_tp':72,'infer_pp':1}:
        raise ValueError('The actual verl rollout mesh must combine TP8×DP9 as infer_tp=72, with dp=PP=1')
    if (mesh.get_local_rank('infer_tp')!=world_rank or mesh.get_local_rank('dp')!=0
            or mesh.get_local_rank('infer_pp')!=0):
        raise ValueError('The combined verl infer_tp rank must equal the complete native world rank')
    return dict(dimension_names=list(names),dimension_sizes=sizes,
                combined_infer_tp_rank=world_rank,native_tp_rank=world_rank%8,native_dp_rank=world_rank//8)


def export_descriptor(weights, required_sources, *, peft_config=None, step=0):
    if peft_config is not None:
        raise ValueError('Qwen3.8 uses full-policy updates')
    if type(step) is not int or step < 0:
        raise ValueError('The refit global step must be a nonnegative integer')
    if any(not callable(getattr(weights, name, None))
           for name in ('acknowledge', 'close', 'confirm_native_cleanup', '__iter__')):
        raise TypeError('The exact leased canonical exporter must reach RTP without prefetch wrapping')
    order = tuple(weights.order)
    if (len(order) != MAIN_SOURCES or len(set(order)) != MAIN_SOURCES
            or set(order) != set(required_sources) or any(not isinstance(name, str) for name in order)):
        raise ValueError('Canonical exporter must cover every 1590 main-policy source exactly once')
    return dict(source_count=len(order), order_sha256=hashlib.sha256(
        json.dumps(order, separators=(',', ':')).encode()).hexdigest(), global_step=step)


def validate_collective_stage(records, stage, *, identity_fields=()):
    if len(records) != WORLD:
        raise RuntimeError(f'{stage}: all 72 training ranks must participate')
    errors = [dict(rank=rank, error=row.get('error')) for rank, row in enumerate(records) if row.get('error')]
    if errors:
        raise RuntimeError(f'{stage}: rank-local failure: {errors}')
    for field in identity_fields:
        if any(row.get(field) != records[0].get(field) for row in records):
            raise RuntimeError(f'{stage}: {field} differs across the 72 training ranks')
    return records


def exact_native_ack(ranks):
    if (not isinstance(ranks, (list, tuple)) or len(ranks) != WORLD
            or any(type(rank) is not int for rank in ranks) or set(ranks) != set(range(WORLD))):
        raise RuntimeError('Every source lease requires exactly one ACK from each of the 72 native ranks')
    return ranks


def validate_policy_probe_sources(sync, hf, *, step, version, boot, incarnation):
    """Bind pre-refit actual HF scores to the precise full native transaction."""
    if (sync.get('global_step') != step or sync.get('model_version') != version
            or sync.get('native_boot') != boot or sync.get('native_incarnation') != incarnation
            or sync.get('logical_model_replicas') != 1 or sync.get('received_weights') != MAIN_SOURCES):
        raise ValueError('Post-refit probe does not bind the complete current source transaction')
    exact_native_ack(sync.get('ranks'))
    digest = sync.get('full_source_sha256', '')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Post-refit probe requires the actual full canonical source digest')
    if (hf.get('schema') != 'q38-real-HF-policy-probe-v1' or hf.get('world_size') != WORLD
            or hf.get('rank') != 0 or hf.get('global_step') != step
            or hf.get('master_dtype') != 'bfloat16' or hf.get('compute_dtype') != 'bfloat16'
            or hf.get('score_dtype') != 'float32' or hf.get('all_rank_HF_probe_agreement') is not True
            or not isinstance(hf.get('max_HF_rank_error'), (int, float))
            or not 0 <= hf['max_HF_rank_error'] <= .05):
        raise ValueError('Actual complete-world BF16 HF reference for this policy is required before native admission')
    scores = hf.get('raw_log_probs', [])
    if (not hf.get('prompt_ids') or not hf.get('response_ids') or len(scores) != len(hf['response_ids'])
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scores)
            or hf.get('raw_fp32_bits_hex') != struct.pack(f'<{len(scores)}f', *scores).hex()):
        raise ValueError('The actual HF reference target IDs and FP32 score bits differ')
    return hf
