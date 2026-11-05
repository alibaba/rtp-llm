"""No-CUDA model-contract and tiny differentiable forward/backward validation."""
from __future__ import annotations
import argparse
import math
import json
import os
from pathlib import Path
import sys
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE/'overrides'))
from checkpoint_contract import identity, metadata, partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=HERE.parent/'metadata')
    parser.add_argument('--output-dir', type=Path, default=HERE/'cpu-check')
    parser.add_argument('--meta-only', action='store_true', help='Dump actual meta parameters and an unexecuted FSDP32 interval plan')
    args = parser.parse_args()
    import torch
    from transformers import AutoConfig
    import qwen36_training_overrides as hook
    from verl.utils.model import get_hf_auto_model_class

    root = HERE.parent
    config = AutoConfig.from_pretrained(args.metadata, local_files_only=True, attn_implementation='sdpa')
    hook.configure_text_only(config)
    factory = get_hf_auto_model_class(config)
    with torch.device('meta'):
        model = factory.from_config(config, dtype=torch.float32)
    rows = [{'name':n,'shape':list(p.shape),'dtype':str(p.dtype),'requires_grad':p.requires_grad}
            for n,p in model.named_parameters()]
    text_config, index = metadata(args.metadata)
    weight_map = index['weight_map']
    expected_names = set(weight_map)
    parameters = {r['name']:r for r in rows}
    missing = sorted(set(parameters)-expected_names)
    extras = sorted(expected_names-set(parameters))
    packed = [r for r in rows if '.experts.' in r['name']]
    scalars = [r for r in rows if r['name'].endswith(('.linear_attn.A_log','.linear_attn.dt_bias'))]
    layers, experts = text_config['num_hidden_layers'], text_config['num_experts']
    inner, hidden = text_config['moe_intermediate_size'], text_config['hidden_size']
    expected_packed = {f'model.language_model.layers.{i}.mlp.experts.{leaf}': shape
        for i in range(layers) for leaf, shape in
        [('gate_up_proj', [experts,2*inner,hidden]), ('down_proj', [experts,hidden,inner])]}
    expected_gdn = {f'model.language_model.layers.{i}.linear_attn.{leaf}'
        for i, kind in enumerate(text_config['layer_types']) if kind == 'linear_attention'
        for leaf in ('A_log', 'dt_bias')}
    assert {r['name']:r['shape'] for r in packed} == expected_packed
    assert {r['name'] for r in scalars} == expected_gdn and all(r['dtype']=='torch.float32' for r in scalars)
    assert not missing and set(extras) == {name for name in expected_names if partition(name)=='mtp'}
    assert all(r['requires_grad'] == (partition(r['name']) == 'text') for r in rows)
    text_numel = sum(math.prod(r['shape']) for r in rows if r['requires_grad'])
    visual_numel = sum(math.prod(r['shape']) for r in rows if partition(r['name'])=='visual')
    meta = {'schema':'qwen35-397b-training-meta-contract-v1', **identity(),
            'text_numel':text_numel, 'visual_numel':visual_numel,'parameter_count':sum(p.numel() for p in model.parameters()),
            'state_names_match_checkpoint_except_disabled_mtp':True,
            'missing_checkpoint_names':missing,'ignored_checkpoint_mtp_names':extras,
            'packed_expert_parameter_count':len(packed),'packed_expert_examples':packed[:4],
            'gdn_fp32_parameters':scalars,'frozen_visual_parameters':sum(r['name'].startswith('model.visual.') for r in rows),
            'parameters':rows,
            'model_class':type(model).__name__,'cuda_initialized':torch.cuda.is_initialized()}
    assert (text_numel, visual_numel, layers, experts) == (396346350336,456010480,60,512)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rank_totals = [dict(rank=i,local_numel=0,trainable_local_numel=0,empty_parameter_shards=0) for i in range(32)]
    planned = []
    for row in rows:
        shape = row['shape']
        width = math.ceil(shape[0]/32)
        intervals = []
        for rank in range(32):
            begin = min(rank*width,shape[0])
            end = min(begin+width,shape[0])
            local_shape = [end-begin,*shape[1:]]
            count = math.prod(local_shape)
            intervals.append(dict(rank=rank,shard_begin=begin,shard_end=end,local_shape=local_shape,local_numel=count))
            rank_totals[rank]['local_numel'] += count
            rank_totals[rank]['trainable_local_numel'] += count if row['requires_grad'] else 0
            rank_totals[rank]['empty_parameter_shards'] += int(count==0)
        assert sum(x['local_numel'] for x in intervals) == math.prod(shape)
        planned.append(dict(name=row['name'],global_shape=shape,requires_grad=row['requires_grad'],intervals=intervals))
    plan = dict(schema='qwen397-meta-fsdp32-interval-plan-v1',**identity(),world_size=32,
                source='actual Transformers meta parameter shapes; mathematical intervals, not observed GPU DTensors',
                parameters=planned,rank_totals=rank_totals,cuda_initialized=torch.cuda.is_initialized(),
                gpu_stages_executed=False,counts_as_full_model_gpu_evidence=False)
    (args.output_dir/'fsdp32-checkpoint-plan.json').write_text(json.dumps(plan,separators=(',',':'))+'\n')
    del model
    if args.meta_only:
        meta['schema'] = 'qwen35-397b-training-meta-plan-v1'
        meta['counts_as_full_model_gpu_evidence'] = False
        (args.output_dir/'meta-model-parameters.json').write_text(json.dumps(meta,indent=2)+'\n')
        print('QWEN397_META_PARAMETER_PLAN_COMPLETE',len(rows),text_numel,visual_numel,flush=True)
        return

    # Small 3-GDN/1-attention MoE network exercises CPU autograd through the same
    # native Transformers modules, including padded attention and text-only input.
    tiny_dict = config.to_dict()
    tiny_dict['text_config'].update(vocab_size=64,hidden_size=32,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        linear_num_key_heads=2,linear_num_value_heads=2,
        linear_key_head_dim=8,linear_value_head_dim=8,num_experts=4,
        num_experts_per_tok=2,moe_intermediate_size=16,shared_expert_intermediate_size=16,
        layer_types=['linear_attention']*3+['full_attention'],bos_token_id=1,eos_token_id=2,
        pad_token_id=0,partial_rotary_factor=1.0,
        rope_parameters={'rope_type':'default','rope_theta':10000.0,'partial_rotary_factor':1.0,
                         'mrope_section':[3,3,2],'mrope_interleaved':True})
    tiny_dict['vision_config'].update(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
        out_hidden_size=32,patch_size=2,temporal_patch_size=1,spatial_merge_size=2,num_position_embeddings=64)
    tiny_dict.update(image_token_id=61,video_token_id=62,vision_start_token_id=59,vision_end_token_id=60)
    tiny_config = AutoConfig.for_model(tiny_dict.pop('model_type'), **tiny_dict)
    tiny_config._attn_implementation = 'sdpa'
    tiny_config.text_config._attn_implementation = 'sdpa'
    tiny_config.vision_config._attn_implementation = 'sdpa'
    tiny = get_hf_auto_model_class(tiny_config).from_config(tiny_config, dtype=torch.float32)
    torch.manual_seed(17)
    ids = torch.randint(3,59,(2,16))
    attention_mask = torch.ones_like(ids)
    attention_mask[1,:3] = 0
    result = tiny(input_ids=ids,attention_mask=attention_mask,labels=ids,use_cache=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    text_grads = [p.grad for n,p in tiny.named_parameters() if p.requires_grad and p.grad is not None]
    assert text_grads and any(bool(torch.count_nonzero(g)) for g in text_grads)
    assert not any(p.requires_grad or p.grad is not None for n,p in tiny.named_parameters() if n.startswith('model.visual.'))
    # Test the bounded optimizer with only the registration shim mocked; its
    # AdamW implementation, gradients and weight updates are real CPU tensors.
    sys.modules['install_rtp_bridge'] = types.SimpleNamespace(worker_setup=lambda:None)
    audit = args.output_dir/'optimizer'
    audit.mkdir(parents=True,exist_ok=True)
    os.environ['RLLM_OPTIMIZER_AUDIT_DIR'] = str(audit)
    from qwen36_optimizer_audit import AuditedAdamW
    optimizer = AuditedAdamW([p for p in tiny.parameters() if p.requires_grad],lr=1e-4,weight_decay=0)
    optimizer.step()
    proof = json.loads(sorted(audit.glob('optimizer-*.json'))[-1].read_text())
    assert proof['audit_passed'] and proof['optimizer_step_min']==1
    assert hook.export_dtype('model.language_model.layers.0.linear_attn.A_log') == torch.float32
    assert hook.export_dtype('model.language_model.layers.0.linear_attn.dt_bias') == torch.float32
    assert hook.export_dtype('model.language_model.layers.0.mlp.experts.gate_up_proj') == torch.bfloat16
    assert not torch.cuda.is_initialized()
    meta['cpu_tiny_test'] = {'loss':float(result.loss.detach()),'real_optimizer_update':True,
                             'visual_frozen':True,'cuda_initialized':False}
    output = args.output_dir/'meta-and-cpu-contract.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(meta,indent=2)+'\n')
    print('QWEN35_397B_CPU_CONTRACT_PASSED',json.dumps({k:v for k,v in meta.items() if k not in ['gdn_fp32_parameters','ignored_checkpoint_mtp_names']}))


if __name__=='__main__':main()
