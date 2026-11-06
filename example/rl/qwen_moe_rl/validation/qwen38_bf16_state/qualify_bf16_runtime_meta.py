"""CPU-only actual HF meta and live verl-hook qualification; opens no weights."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(CUDA_VISIBLE_DEVICES='',Q38_ENABLE_BF16_STATE_RUNTIME='1')
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM
from qwen38_training_contract import load_metadata,POLICY_NUMEL,shard_interval
from qwen38_chunked_experts import install_meta_chunks,ExpertChunk
from qwen38_vocab_tiles import install_meta_vocab_tiles,VocabTile
import qwen38_bf16_training_runtime
from qwen38_bf16_streamed_checkpoint import (save_buffers,remember_initialized_buffers,
                                           restore_initialized_buffers_after_engine_cast)


def main():
    started=time.time()
    import verl.workers.engine.fsdp.transformer_impl as engine
    hooks={name:getattr(engine.FSDPEngine,name).__module__ for name in
           ['_build_module','_build_optimizer','optimizer_step','to','get_per_tensor_param','q38_confirm_native_cleanup']}
    assert set(hooks.values())=={'qwen38_bf16_training_runtime'},hooks
    mounted='/mnt/fuse/Qwen/Qwen3.8-2.4T-A95B'
    _,_,official,_=load_metadata(mounted)
    config=AutoConfig.from_pretrained(mounted,local_files_only=True,trust_remote_code=False)
    config.use_cache=False;config.mtp_num_hidden_layers=0;config._experts_implementation='eager'
    with torch.device('cpu'),init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16)
    assert set(dict(model.named_parameters()))==set(official)
    assert all(tuple(p.shape)==tuple(official[name]) for name,p in model.named_parameters())
    assert all(p.is_meta and p.dtype==torch.bfloat16 for p in model.parameters())
    initialized_buffers,aliases=save_buffers(model)
    remember_initialized_buffers(model)
    model.to(torch.bfloat16)  # Exact unconditional cast in real verl _build_module.
    cast_types={name:str(buffer.dtype) for name,buffer in model.named_buffers()}
    assert any(b.dtype==torch.float32 for b in initialized_buffers.values())
    assert any(b.dtype!=dict(model.named_buffers())[name].dtype for name,b in initialized_buffers.items())
    restore_initialized_buffers_after_engine_cast(model)
    for name,buffer in model.named_buffers():
        expected=initialized_buffers[name]
        assert buffer.dtype==expected.dtype and torch.equal(buffer,expected),(name,buffer.dtype,expected.dtype)
    install_meta_chunks(model);install_meta_vocab_tiles(model)
    assert sum(p.numel() for p in model.parameters())==POLICY_NUMEL
    assert len(list(model.parameters()))==7324
    assert all(p.is_meta and p.dtype==torch.bfloat16 for p in model.parameters())
    ranks=[]
    params=list(model.parameters())
    for rank in range(72):
        counts=[]
        for p in params:
            lo,hi=shard_interval(p.shape[0],72,rank)
            counts.append((hi-lo)*p.numel()//p.shape[0])
        count=sum(counts);largest=max(counts)*4
        cache=80*1024**3;retain=24*1024**3
        # The production all-rank placement check repeats this on real DTensors.
        ranks.append(dict(rank=rank,local_policy_numel=count,GPU_BF16_master=count*2,
            CPU_BF16_moments=count*4,CPU_FP32_grad_spill_upper=max(0,count*4-cache+largest),
            CPU_rollout_master_upper=max(0,count*2-(retain-largest//2)),
            largest_local_gradient_bytes=largest))
    assert sum(r['local_policy_numel'] for r in ranks)==POLICY_NUMEL
    nodes=[]
    for node in range(18):
        peers=ranks[node*4:node*4+4]
        train=sum(r['CPU_BF16_moments']+r['CPU_FP32_grad_spill_upper'] for r in peers)+96*1024**3
        rollout=sum(r['CPU_BF16_moments']+r['CPU_rollout_master_upper'] for r in peers)+96*1024**3
        assert max(train,rollout)<=800*1024**3,(node,train,rollout)
        assert sum(r['CPU_BF16_moments'] for r in peers)<=512*1024**3
        nodes.append(dict(node_index=node,train_CPU_with_reserve_bytes=train,
                          rollout_CPU_with_reserve_bytes=rollout))
    result=dict(schema='q38-BF16-runtime-HF-meta-v1',passed=True,
        actual_verl_engine_hooks=hooks,source_keys=len(official),runtime_parameters=len(params),
        policy_numel=POLICY_NUMEL,all_policy_parameters_meta_BF16=True,
        initialized_real_CPU_buffers=sum(v.numel() for v in initialized_buffers.values()),
        buffer_aliases=len(aliases),source_actual_dtypes={'bfloat16':len(params)},
        actual_engine_unconditional_BF16_buffer_cast_reproduced=True,
        original_initialized_buffers_restored_bit_exactly=True,cast_buffer_dtypes=cast_types,
        planned_ranks=ranks,planned_nodes=nodes,actual_DTensors_still_require_runtime_plan=True,
        CUDA_initialized=torch.cuda.is_initialized(),official_weight_files_opened=False,
        full_Qwen38_RL_validated=False,started_unix=started,finished_unix=time.time())
    assert not result['CUDA_initialized']
    Path('runtime-meta-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BF16_META_RUNTIME_PASS',json.dumps({k:v for k,v in result.items() if k not in
        ('planned_ranks','planned_nodes')}))


if __name__=='__main__':main()
