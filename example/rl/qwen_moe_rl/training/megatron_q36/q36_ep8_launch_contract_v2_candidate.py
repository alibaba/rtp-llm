"""TP4/EP8 training + TP4/EP4 rollout configuration candidate.

The existing TP4 BF16 contract is reused on a deep copy after the exact EP8
and pool geometry are checked. No mutable real config is rewritten.
"""
import copy
import hashlib
import importlib
import importlib.machinery
import sys
import json
import os
from pathlib import Path


def validate(config):
    from q36_launch_contract import get, validate as validate_tp4
    expected = {
        ('trainer', 'nnodes'): 2,
        ('trainer', 'n_gpus_per_node'): 4,
        ('actor_rollout_ref', 'actor', 'megatron', 'tensor_model_parallel_size'): 4,
        ('actor_rollout_ref', 'actor', 'megatron', 'expert_model_parallel_size'): 8,
        ('actor_rollout_ref', 'actor', 'megatron', 'expert_tensor_parallel_size'): 1,
        ('actor_rollout_ref', 'rollout', 'tensor_model_parallel_size'): 4,
        ('actor_rollout_ref', 'rollout', 'data_parallel_size'): 1,
    }
    for path, value in expected.items():
        actual = get(config, *path)
        if type(actual) is not type(value) or actual != value:
            raise ValueError('.'.join(path) + ' differs from the qualified EP8 profile')
    if (os.environ.get('Q36_DISAGG_TRAIN_GPUS') != '8' or
        os.environ.get('Q36_DISAGG_ROLLOUT_GPUS') != '4' or
        os.environ.get('RTP_EP_SIZE') != '4' or os.environ.get('RTP_LLM_RL_Q36_EP4') != '1' or
        os.environ.get('RTP_RL_SHARED_BACKING', '0') != '0'):
        raise ValueError('requires an explicit independent 8 training + 4 rollout EP4 allocation')
    base = copy.deepcopy(config)
    base.trainer.nnodes = 1
    base.actor_rollout_ref.actor.megatron.expert_model_parallel_size = 1
    base.actor_rollout_ref.actor.megatron.expert_tensor_parallel_size = 4
    validate_tp4(base)
    path = Path(os.environ.get('Q36_EP8_SR_MANIFEST', ''))
    digest = os.environ.get('Q36_EP8_SR_MANIFEST_SHA256', '')
    if (path.is_symlink() or not path.is_file() or len(digest) != 64 or
            hashlib.sha256(path.read_bytes()).hexdigest() != digest):
        raise ValueError('independently reviewed EP8 live-provider manifest required')
    value = json.loads(path.read_bytes())
    if (value.get('schema') != 'q36-ep8-live-provider-sr-v1' or
        value.get('binding_scope') != 'independent-source-and-live-provider-reviewed' or
        (value.get('tp'), value.get('ep'), value.get('etp'), value.get('pp')) != (4, 8, 1, 1) or
        set(value.get('rank_parameters', {})) != {str(x) for x in range(8)}):
        raise ValueError('reviewed EP8 manifest geometry or rank coverage mismatch')
    # Audit against the fixed model, checkpoint index, independently captured
    # EP1 provider and Bridge registry; never derive expectations from value.
    root = Path(os.environ['Q36_CODE_DIR']).resolve(strict=True)
    source = root / 'q36_ep8_independent_coverage_candidate.py'
    spec = importlib.machinery.PathFinder.find_spec('q36_ep8_independent_coverage_candidate', sys.path)
    loaded = sys.modules.get('q36_ep8_independent_coverage_candidate')
    if (source.is_symlink() or not source.is_file() or
        hashlib.sha256(source.read_bytes()).hexdigest() != '2265fdcbc966fda7af86fee4c855aca64285787480ec4ad63423a88f5d6320dc' or
        spec is None or not spec.origin or Path(spec.origin).resolve() != source or
        loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != source):
        raise ValueError('independent expert coverage auditor source is not locked')
    audit = importlib.import_module('q36_ep8_independent_coverage_candidate')
    model = Path(os.environ['Q36_MODEL_PATH'])
    independent = audit.derive_expected(model / 'config.json', model / 'model.safetensors.index.json',
                                        os.environ['Q36_SR_MANIFEST'], os.environ['Q36_EP8_BRIDGE_REGISTRY'])
    coverage = audit.audit_ledger(value, independent)
    return {'expected_train_gpus': 8, 'expected_rollout_gpus': 4, 'sr_manifest_sha256': digest,
            'independent_expert_coverage': coverage}
