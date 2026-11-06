"""Actual Transformers architecture and CPU forward/backward smoke.

The official 2.4T policy is constructed on meta only. The executable forward
uses an explicitly tiny synthetic config and opens no checkpoint weight files.
"""
from __future__ import annotations
import argparse,hashlib,inspect,json,os,time
from pathlib import Path


def tiny_config_dict(original):
    config=dict(original)
    config.update(vocab_size=289,hidden_size=144,num_hidden_layers=4,
        num_experts=8,num_experts_per_tok=2,moe_intermediate_size=72,
        shared_expert_intermediate_size=72,num_attention_heads=4,num_key_value_heads=2,head_dim=32,
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=32,linear_value_head_dim=32,
        layer_types=['linear_attention']*3+['full_attention'],mtp_num_hidden_layers=0,
        use_cache=False,output_router_logits=False,pad_token_id=3,bos_token_id=1,eos_token_id=2,
        _experts_implementation='eager')
    return config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--metadata',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from accelerate import init_empty_weights
    import transformers
    from transformers import AutoConfig,AutoModelForCausalLM
    from qwen38_training_contract import load_metadata,POLICY_NUMEL
    from qwen38_chunked_experts import install_meta_chunks
    from qwen38_vocab_tiles import install_meta_vocab_tiles
    from qwen38_training_runtime import install
    install()
    import verl.workers.engine.fsdp.transformer_impl as engine
    import verl.utils.model as model_utils
    hooks={name:getattr(engine.FSDPEngine,name).__module__ for name in [
        '_build_optimizer','optimizer_step','to','get_per_tensor_param','q38_confirm_native_cleanup']}
    if set(hooks.values())!={'qwen38_training_runtime'}:raise RuntimeError('Actual verl engine hooks are not active')
    if model_utils.get_hf_auto_model_class.__module__!='qwen38_training_runtime':
        raise RuntimeError('The actual HF model factory was not replaced')
    original,_,policy,_=load_metadata(args.metadata)
    config=AutoConfig.from_pretrained(args.metadata,local_files_only=True,trust_remote_code=False)
    config.use_cache=False;config.mtp_num_hidden_layers=0;config._experts_implementation='eager'
    started=time.time()
    with torch.device('cpu'),init_empty_weights(include_buffers=False):
        full=AutoModelForCausalLM.from_config(config,dtype=torch.float32)
    if type(full).__name__!='Qwen3_5MoeForCausalLM':raise RuntimeError('Actual HF class differs from the flat-text checkpoint')
    if set(dict(full.named_parameters()))!=set(policy):raise RuntimeError('Actual HF source keys differ from official policy')
    wrong={name:{'actual':list(parameter.shape),'expected':list(policy[name])}
           for name,parameter in full.named_parameters() if tuple(parameter.shape)!=tuple(policy[name])}
    if wrong:raise RuntimeError('Actual HF source tensor shapes differ '+str(wrong))
    if any(not p.is_meta for p in full.parameters()):raise RuntimeError('An official-sized parameter was materialized')
    source_count=len(list(full.parameters()))
    install_meta_chunks(full);install_meta_vocab_tiles(full)
    if sum(p.numel() for p in full.parameters())!=POLICY_NUMEL or len(list(full.parameters()))!=7324:
        raise RuntimeError('Actual HF meta subdivision changed the pinned policy coordinates')
    meta={'class':type(full).__name__,'source_parameters':source_count,
          'runtime_parameters':len(list(full.parameters())),'policy_numel':POLICY_NUMEL,
          'all1590_canonical_shapes_match_pinned_config_and_index':True,
          'all_parameters_meta':all(p.is_meta for p in full.parameters()),
          'buffer_numel':sum(b.numel() for b in full.buffers()),
          'all_initialized_buffers_CPU':all(b.device.type=='cpu' for b in full.buffers())}
    del full
    tiny_dict=tiny_config_dict(original)
    tiny_config=AutoConfig.for_model('qwen3_5_moe_text',**{k:v for k,v in tiny_dict.items() if k!='model_type'})
    tiny=AutoModelForCausalLM.from_config(tiny_config,dtype=torch.float32)
    tiny.train();generator=torch.Generator().manual_seed(937)
    ids=torch.randint(4,289,(2,8),generator=generator)
    result=tiny(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False)
    if not result.loss.requires_grad or not torch.isfinite(result.loss) or result.loss.item()<=0:
        raise RuntimeError('The actual tiny HF language-model loss is invalid')
    result.loss.backward()
    missing=[name for name,p in tiny.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite=[name for name,p in tiny.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if missing or nonfinite:raise RuntimeError('Actual HF gradient coverage/finite check failed '+str((missing,nonfinite)))
    norm=sum(float(p.grad.double().square().sum()) for p in tiny.parameters())**0.5
    if norm<=0:raise RuntimeError('The actual tiny HF backward produced no gradient')
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling
    attention_source=inspect.getsource(modeling.Qwen3_5MoeAttention.forward)
    source_path=Path(inspect.getfile(modeling))
    report={'schema':'qwen38-actual-HF-meta-CPU-v1','passed':True,'started_unix':started,'finished_unix':time.time(),
        'torch_version':torch.__version__,'transformers_version':transformers.__version__,
        'actual_verl_engine_hooks':hooks,'full_meta_policy':meta,
        'tiny_synthetic_config':tiny_dict,'tiny_loss':float(result.loss.detach()),'tiny_gradient_norm':norm,
        'tiny_parameter_numel':sum(p.numel() for p in tiny.parameters()),'missing_gradients':missing,
        'all_gradients_finite':True,'actual_modeling_path':str(source_path),
        'actual_modeling_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'full_attention_forward_sha256':hashlib.sha256(attention_source.encode()).hexdigest(),
        'full_attention_sigmoid_in_source':'sigmoid' in attention_source,
        'full_attention_output_gate_type_field_used':'output_gate_type' in attention_source,
        'CUDA_initialized':torch.cuda.is_initialized(),'official_weight_files_opened':False,
        'full_2_4T_training_accepted':False}
    if report['CUDA_initialized']:raise RuntimeError('CPU source/meta validation initialized a GPU')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream:json.dump(report,stream,indent=2)
    print('ACTUAL_HF_META_CPU_PASS',json.dumps({k:v for k,v in report.items() if k!='tiny_synthetic_config'}),flush=True)


if __name__=='__main__':main()
