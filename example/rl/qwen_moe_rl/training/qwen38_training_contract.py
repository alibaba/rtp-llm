"""Pinned Qwen3.8 policy shapes, exact 72-rank shards, and resource lower bounds.

This module uses only the standard library. It is a preflight, not a training
runner or a claim that native RTP pipeline parallelism has been implemented.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3.8-2.4T-A95B'
REVISION = '207bd685a7e3696cfaff12ded7c6a7ea0f88c996'
CONFIG_SHA256 = '4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987'
INDEX_SHA256 = 'e36c40d4e99b2714fff821218a0433bda2dec46afdb1ebee8ce96ced997928ee'
CHECKPOINT_NUMEL = 2_446_182_725_504
POLICY_NUMEL = 2_419_804_697_984
MTP_NUMEL = 26_378_027_520
WORLD_SIZE = 72
NODES = 18
GPUS_PER_NODE = 4


def numel(shape):
    if not shape or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError('Expected a nonempty nonnegative integer shape')
    return math.prod(shape)


def validate_config(config, *, allow_fp32_master=False):
    required = dict(model_type='qwen3_5_moe_text', num_hidden_layers=92,
                    hidden_size=8192, num_experts=512, num_experts_per_tok=10,
                    moe_intermediate_size=2048, shared_expert_intermediate_size=2048,
                    num_attention_heads=64, num_key_value_heads=4, head_dim=256,
                    linear_num_key_heads=16, linear_num_value_heads=128,
                    linear_key_head_dim=128, linear_value_head_dim=128,
                    linear_conv_kernel_dim=4, vocab_size=248320,
                    dtype='bfloat16', tie_word_embeddings=False)
    if allow_fp32_master and config.get('dtype') == 'float32':
        required['dtype'] = 'float32'
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('Configuration is not the pinned Qwen3.8 text policy')
    if config.get('architectures') != ['Qwen3_5MoeForCausalLM']:
        raise ValueError('Qwen3.8 needs the text CausalLM architecture')
    expected_layers = ['linear_attention'] * 3 + ['full_attention']
    if config.get('layer_types') != expected_layers * 23:
        raise ValueError('The 69 GDN / 23 full-attention layer order differs')
    if config.get('quantization_config') is not None:
        raise ValueError('This profile requires the unquantized BF16 checkpoint')


def training_config(original):
    """A separate runtime copy: official checkpoint bytes are never rewritten."""
    validate_config(original)
    result = copy.deepcopy(original)
    result.update(use_cache=False, output_router_logits=False,
                  mtp_num_hidden_layers=0)
    return result


def layer_shapes(prefix, kind, config):
    h = config['hidden_size']; e = config['num_experts']
    i = config['moe_intermediate_size']; shared = config['shared_expert_intermediate_size']
    shapes = {
        prefix + '.input_layernorm.weight': (h,),
        prefix + '.post_attention_layernorm.weight': (h,),
        prefix + '.mlp.experts.gate_up_proj': (e, 2 * i, h),
        prefix + '.mlp.experts.down_proj': (e, h, i),
        prefix + '.mlp.gate.weight': (e, h),
        prefix + '.mlp.shared_expert.gate_proj.weight': (shared, h),
        prefix + '.mlp.shared_expert.up_proj.weight': (shared, h),
        prefix + '.mlp.shared_expert.down_proj.weight': (h, shared),
        prefix + '.mlp.shared_expert_gate.weight': (1, h),
    }
    if kind == 'linear_attention':
        key = config['linear_num_key_heads'] * config['linear_key_head_dim']
        value = config['linear_num_value_heads'] * config['linear_value_head_dim']
        heads = config['linear_num_value_heads']; conv = 2 * key + value
        shapes.update({
            prefix + '.linear_attn.A_log': (heads,),
            prefix + '.linear_attn.dt_bias': (heads,),
            prefix + '.linear_attn.conv1d.weight': (conv, 1, config['linear_conv_kernel_dim']),
            prefix + '.linear_attn.norm.weight': (config['linear_value_head_dim'],),
            prefix + '.linear_attn.in_proj_qkv.weight': (conv, h),
            prefix + '.linear_attn.in_proj_z.weight': (value, h),
            prefix + '.linear_attn.in_proj_a.weight': (heads, h),
            prefix + '.linear_attn.in_proj_b.weight': (heads, h),
            prefix + '.linear_attn.out_proj.weight': (h, value),
        })
    elif kind == 'full_attention':
        head = config['head_dim']; query = config['num_attention_heads'] * head
        kv = config['num_key_value_heads'] * head
        shapes.update({
            prefix + '.self_attn.q_proj.weight': (2 * query, h),
            prefix + '.self_attn.k_proj.weight': (kv, h),
            prefix + '.self_attn.v_proj.weight': (kv, h),
            prefix + '.self_attn.o_proj.weight': (h, query),
            prefix + '.self_attn.q_norm.weight': (head,),
            prefix + '.self_attn.k_norm.weight': (head,),
        })
    else:
        raise ValueError('Unknown Qwen3.8 layer type')
    return shapes


def parameter_shapes(config):
    """The shape formula must match all 1,609 official index keys and numel."""
    validate_config(config)
    h = config['hidden_size']; vocab = config['vocab_size']
    policy = {'model.embed_tokens.weight': (vocab, h), 'model.norm.weight': (h,),
              'lm_head.weight': (vocab, h)}
    for index, kind in enumerate(config['layer_types']):
        policy.update(layer_shapes(f'model.layers.{index}', kind, config))
    mtp = layer_shapes('mtp.layers.0', 'full_attention', config)
    mtp.update({'mtp.fc.weight': (h, 2 * h), 'mtp.norm.weight': (h,),
                'mtp.pre_fc_norm_embedding.weight': (h,),
                'mtp.pre_fc_norm_hidden.weight': (h,)})
    if (len(policy), sum(map(numel, policy.values()))) != (1590, POLICY_NUMEL):
        raise ValueError('Policy shape formula does not match its pinned contract')
    if (len(mtp), sum(map(numel, mtp.values()))) != (19, MTP_NUMEL):
        raise ValueError('MTP shape formula does not match its pinned contract')
    return policy, mtp


def load_metadata(checkpoint):
    checkpoint = Path(checkpoint)
    values = {}
    for filename, expected in [('config.json', CONFIG_SHA256),
                               ('model.safetensors.index.json', INDEX_SHA256)]:
        raw = (checkpoint / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'Pinned official {filename} SHA256 differs')
        values[filename] = json.loads(raw)
    config = values['config.json']; index = values['model.safetensors.index.json']
    policy, mtp = parameter_shapes(config)
    mapping = index['weight_map']
    if set(mapping) != set(policy) | set(mtp):
        raise ValueError('Checkpoint source keys differ from the full policy+MTP shape contract')
    if len(set(mapping.values())) != 213:
        raise ValueError('The pinned checkpoint must reference all 213 weight shards')
    if any(Path(name).name != name for name in mapping.values()):
        raise ValueError('Checkpoint index contains a nonlocal weight path')
    declared = index['metadata']['total_size']
    # The official file uses an integral JSON float. Do not change its bytes or
    # reject a valid official file just because the JSON number is not an int.
    if (isinstance(declared, bool) or not isinstance(declared, (int, float))
            or not math.isfinite(declared) or int(declared) != declared
            or int(declared) != 2 * CHECKPOINT_NUMEL):
        raise ValueError('Official BF16 data payload size differs')
    return config, index, policy, mtp


def shard_interval(size, world_size, rank):
    """Match ordinary torch Shard/torch.chunk, including zero-length tail ranks."""
    if type(size) is not int or size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid sharding input')
    chunk = (size + world_size - 1) // world_size
    start = min(rank * chunk, size)
    return start, min(start + chunk, size)


def shard_dim(name, shape, policy='packed_axis1'):
    if policy not in ('axis0', 'packed_axis1', 'flat_chunks16', 'flat_chunks16_vocab16384'):
        raise ValueError('Unknown explicit FSDP sharding policy')
    if (policy == 'packed_axis1' and len(shape) == 3
            and name.endswith(('.mlp.experts.gate_up_proj', '.mlp.experts.down_proj'))):
        return 1
    return 0


def runtime_parameter_shapes(shapes, policy):
    if policy not in ('flat_chunks16','flat_chunks16_vocab16384'):
        return dict(shapes)
    result = {}
    for name, shape in shapes.items():
        if policy=='flat_chunks16_vocab16384' and name in ('model.embed_tokens.weight','lm_head.weight'):
            prefix=name.rsplit('.',1)[0]
            for index,begin in enumerate(range(0,shape[0],16384)):
                result[f'{prefix}.tiles.{index}.weight']=(min(16384,shape[0]-begin),shape[1])
            continue
        if len(shape) != 3 or not name.endswith(('.mlp.experts.gate_up_proj', '.mlp.experts.down_proj')):
            result[name] = shape
            continue
        if shape[0] != 512:
            raise ValueError('Pinned expert count differs before flattened chunk planning')
        prefix, field = name.rsplit('.', 1)
        for index in range(32):
            result[f'{prefix}.chunks.{index}.{field}'] = (16 * shape[1], shape[2])
    if sum(map(numel, result.values())) != sum(map(numel, shapes.values())):
        raise RuntimeError('Flattened expert chunks changed the logical parameter count')
    return result


def flat_source_fragments(source_shape, expert_begin, expert_end, row_begin, row_end):
    """Map owned flat rows to bounded original HF [expert,feature,input] slices."""
    if len(source_shape) != 3:
        raise ValueError('Canonical packed expert source must be three-dimensional')
    experts, features, _ = source_shape
    rows = (expert_end - expert_begin) * features
    if not (0 <= expert_begin < expert_end <= experts and 0 <= row_begin <= row_end <= rows):
        raise ValueError('Flattened source interval is out of bounds')
    result = []; cursor = row_begin
    while cursor < row_end:
        expert, feature = divmod(cursor, features)
        end = min(row_end, cursor + features - feature)
        result.append({'expert': expert_begin + expert, 'feature_begin': feature,
                       'feature_end': feature + end - cursor,
                       'destination_row_begin': cursor - row_begin,
                       'destination_row_end': end - row_begin})
        cursor = end
    return result


def slice_contract(shape, dimension, world_size, rank):
    numel(shape)
    if not 0 <= dimension < len(shape):
        raise ValueError('Invalid shard dimension')
    begin, end = shard_interval(shape[dimension], world_size, rank)
    local_shape = list(shape); local_shape[dimension] = end - begin
    offset = [0] * len(shape); offset[dimension] = begin
    return {'shard_dim': dimension, 'begin': begin, 'end': end,
            'global_shape': list(shape), 'local_shape': local_shape,
            'global_offset': offset, 'local_numel': numel(local_shape)}


def pipeline_ranges(layers, stages):
    if stages < 1 or layers < stages:
        raise ValueError('Pipeline needs at least one layer on every stage')
    quotient, remainder = divmod(layers, stages); result = []; offset = 0
    for stage in range(stages):
        count = quotient + (stage < remainder)
        result.append({'stage': stage, 'begin_layer': offset, 'end_layer': offset + count})
        offset += count
    assert offset == layers
    return result


def validate_topology(tp, pp, replicas=1, allow_kv_replication=False):
    if type(tp) is not int or type(pp) is not int or tp < 1 or pp < 1 or replicas != 1:
        raise ValueError('This allocation requires one positive TP×PP model instance')
    if tp * pp != WORLD_SIZE:
        raise ValueError('Exactly all 72 allocated GPUs must participate in the single instance')
    if any(size % tp for size in (8192, 64, 512)):
        raise ValueError('TP does not divide hidden channels, query heads and expert counts')
    if 4 % tp and not (allow_kv_replication and tp % 4 == 0):
        raise ValueError('KV head replication needs an explicitly validated backend')
    return {'tp': tp, 'pp': pp, 'replicas': 1,
            'kv_replication_required': 4 % tp != 0,
            'uneven_pp_required': 92 % pp != 0,
            'pipeline_layers': pipeline_ranges(92, pp),
            'native_PP_implementation_proven': False}


def memory_report(config, shapes, policy):
    runtime_shapes = runtime_parameter_shapes(shapes, policy)
    ranks = []
    for rank in range(WORLD_SIZE):
        owned = sum(slice_contract(shape, shard_dim(name, shape, policy), WORLD_SIZE, rank)['local_numel']
                    for name, shape in runtime_shapes.items())
        ranks.append({'rank': rank, 'local_numel': owned,
                      'gpu_fp32_param_and_grad_bytes': 8 * owned,
                      'gpu_bf16_param_and_grad_bytes': 4 * owned,
                      'cpu_fp32_moments_bytes': 8 * owned,
                      'cpu_fp32_master_plus_moments_bytes': 12 * owned,
                      'cpu_offloaded_fp32_param_grad_moments_bytes': 16 * owned})
    if sum(row['local_numel'] for row in ranks) != POLICY_NUMEL:
        raise RuntimeError('The 72 actual shard intervals do not cover the policy exactly once')
    nodes = []
    for node in range(NODES):
        group = ranks[node * 4:(node + 1) * 4]
        nodes.append({'node_index': node, **{field: sum(row[field] for row in group)
                      for field in group[0] if field != 'rank'}})
    full_layers = [sum(numel(shape) for name, shape in shapes.items()
                       if name.startswith(f'model.layers.{index}.'))
                   for index in range(config['num_hidden_layers'])]
    return {'shard_policy': policy, 'runtime_parameter_tensors': len(runtime_shapes),
            'torch213_backward_policy_supported': policy != 'packed_axis1',
            'torch213_axis1_blocker': ('foreach_reduce requires size(shard_dim) divisible by world size'
                                     if policy == 'packed_axis1' else None),
            'ranks': ranks, 'nodes': nodes,
            'min_owned_numel': min(row['local_numel'] for row in ranks),
            'max_owned_numel': max(row['local_numel'] for row in ranks),
            'largest_full_layer_bf16_bytes': 2 * max(full_layers),
            'largest_full_packed_parameter_bf16_bytes': max(numel(s) * 2 for s in shapes.values()),
            'largest_full_packed_parameter_fp32_bytes': max(numel(s) * 4 for s in shapes.values()),
            'bounds_are_persistent_only': True,
            'additional_runtime_memory_required': ['all-gather and reduction buffers', 'activations',
                'optimizer staging', 'RTP KV cache', 'bounded weight-refit transport'],
            'no_runtime_memory_fit_claim': True}


def hybrid_memory_report(config,shapes):
    profile=memory_report(config,shapes,'flat_chunks16_vocab16384')
    gpu_grad=40*1024**3;gpu_master_refit=48*1024**3
    cpu_moment_per_node=60_000_000_000;cpu_reserve=64*1024**3
    max_parameter=max(row['local_numel']*4 for row in profile['ranks'])
    nodes=[]
    for node in profile['nodes']:
        parameter=node['local_numel']*4
        spill=max(0,parameter-4*gpu_grad)
        cold_master=max(0,parameter-4*gpu_master_refit)
        nodes.append({'node_index':node['node_index'],
            'fp32_master_bytes':parameter,'training_cpu_gradient_spill_bytes':spill,
            'training_cpu_moments_plus_gradient_and_reserve_bytes':spill+cpu_moment_per_node+cpu_reserve,
            'rollout_cpu_cold_masters_plus_moments_and_reserve_bytes':cold_master+cpu_moment_per_node+cpu_reserve,
            'nvme_two_moments_lower_bound_bytes':max(0,parameter*2-cpu_moment_per_node)})
    return {'schema':'qwen38-hybrid-phase-persistent-budgets-v1',
        'source_policy_numel':POLICY_NUMEL,'runtime_parameter_tensors':profile['runtime_parameter_tensors'],
        'maximum_fp32_gpu_master_bytes':max_parameter,
        'gpu_reduced_gradient_cache_bytes':gpu_grad,
        'gpu_refit_master_cache_bytes':gpu_master_refit,
        'cpu_moment_cache_bytes_per_node':cpu_moment_per_node,'cpu_reserve_bytes_per_node':cpu_reserve,
        'expert_chunk16_bf16_allgather_bytes':16*(4096*8192+8192*2048)*2,
        'expert_chunk16_fp32_reduction_input_bytes':16*(4096*8192+8192*2048)*4,
        'vocab_tile_bf16_allgather_bytes':16384*8192*2,
        'vocab_tile_fp32_reduction_input_bytes':16384*8192*4,
        'canonical_cuda_backing_bytes':512*4096*8192*2,
        'canonical_chunk_gather_maximum_bytes':16*4096*8192*2,
        'all_fp32_master_grad_and_two_moments_bytes':POLICY_NUMEL*16,
        'full_BF16_training_model_persistent':False,
        'optimizer_master_gradient_moments_dtype':'float32',
        'nodes':nodes,'runtime_memory_fit_proven':False,
        'unbudgeted_peak_must_be_measured':['activations','CUDA/NCCL context','native expert layout',
            'native shared weights','KV','loader conversion temporaries','NVMe page-cache residency']}


def report(checkpoint):
    config, index, policy, mtp = load_metadata(checkpoint)
    return {'schema': 'qwen38-72gpu-training-preflight-v1', 'model_id': MODEL_ID,
            'revision': REVISION, 'config_sha256': CONFIG_SHA256, 'index_sha256': INDEX_SHA256,
            'policy_tensors': len(policy), 'mtp_tensors': len(mtp),
            'policy_numel': POLICY_NUMEL, 'excluded_mtp_numel': MTP_NUMEL,
            'checkpoint_numel': CHECKPOINT_NUMEL, 'world_size': WORLD_SIZE,
            'native_single_instance_candidates': [validate_topology(4, 18),
                validate_topology(8, 9, allow_kv_replication=True)],
            'memory_profiles': {p: memory_report(config, policy, p)
                                for p in ('axis0', 'packed_axis1', 'flat_chunks16','flat_chunks16_vocab16384')},
            'hybrid_phase_budgets':hybrid_memory_report(config,policy),
            'enable_thinking': True, 'reasoning_effort': 'low',
            'gpu_execution_performed': False, 'training_passed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(args.checkpoint)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['schema', 'policy_numel', 'policy_tensors',
          'mtp_tensors', 'world_size', 'gpu_execution_performed', 'training_passed']}))
