"""Actual HF tiny CPU equivalence; no official weight reads or GPU work."""
import argparse, copy, json, os, time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import torch
from transformers import AutoConfig,AutoModelForCausalLM
from hf_qwen38_meta_and_cpu import tiny_config_dict
from micro_hf_bf16_state import install_tiny_storage
from qwen38_bounded_checkpoint import install_bounded_checkpoints


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(927)
    source=json.loads((Path(__file__).resolve().parents[1] / 'qwen38_native_bootstrap/fixtures/config.json').read_text())
    small=tiny_config_dict(source)
    c=AutoConfig.for_model('qwen3_5_moe_text',**{k:v for k,v in small.items() if k!='model_type'})
    c._attn_implementation='sdpa';c._experts_implementation='eager'
    old=AutoModelForCausalLM.from_config(c,dtype=torch.float32);install_tiny_storage(old)
    new=copy.deepcopy(old);installation=install_bounded_checkpoints(new,expected_layers=4)
    old.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'early_stop':False})
    new.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'early_stop':False})
    before=tuple(n for n,_ in old.named_parameters())
    if before!=tuple(n for n,_ in new.named_parameters()):raise RuntimeError('Changed actual HF parameter names')
    rounds=[]
    for step in range(2):
        old.train();new.train();old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
        g=torch.Generator().manual_seed(743+step)
        ids=torch.randint(4,289,(2,16),generator=g)
        a=old(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False)
        b=new(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False)
        logits_error=(a.logits.detach()-b.logits.detach()).abs().max().item()
        loss_error=abs(a.loss.item()-b.loss.item())
        a.loss.backward();b.loss.backward()
        oldp=dict(old.named_parameters());newp=dict(new.named_parameters())
        missing=[n for n in before if oldp[n].grad is None or newp[n].grad is None]
        errors={n:float((oldp[n].grad-newp[n].grad).abs().max()) for n in before if n not in missing}
        numerator=sum((oldp[n].grad.double()-newp[n].grad.double()).square().sum().item() for n in errors)
        denominator=sum(oldp[n].grad.double().square().sum().item() for n in errors)
        rel=(numerator/max(denominator,1e-100))**.5
        if missing or logits_error!=0 or loss_error!=0 or rel>1e-5:
            raise RuntimeError(f'Actual HF equivalence failed {missing,logits_error,loss_error,rel}')
        audits=[dict(l._q38_bounded_checkpoint_audit) for l in new.model.layers]
        if any(r['normalized_inputs_deferred']<6 or r['normalized_inputs_recreated']<6 for r in audits):
            raise RuntimeError('Real checkpoint input recomputation did not execute')
        rounds.append(dict(step=step,logits_max_abs_error=logits_error,loss_abs_error=loss_error,
            gradient_max_abs_error=max(errors.values()),gradient_relative_L2_error=rel,
            all_parameters_have_actual_gradients=True,layer_audits=audits))
    new.gradient_checkpointing_disable()
    if any(l._q38_bounded_checkpoint_enabled for l in new.model.layers):raise RuntimeError('Disable ignored')
    new.eval();old.eval()
    with torch.no_grad():
        if not torch.equal(new(input_ids=ids,use_cache=False).logits,old(input_ids=ids,use_cache=False).logits):
            raise RuntimeError('Eval/no-grad original forward changed')
    report=dict(schema='q38-bounded-checkpoint-real-HF-CPU-v1',passed=True,installation=installation,
        rounds=rounds,CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),
        official_weight_files_opened=False,fullmodel_RL_accepted=False,time=time.time())
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
