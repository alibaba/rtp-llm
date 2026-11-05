"""Teacher-forced selected-token HF comparison; launched on GPUs only by root."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from checkpoint_contract import default_model, identity, load_checkpoint, metadata

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',type=Path,required=True)
    ap.add_argument('--model',type=Path,default=default_model(PROJECT))
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tolerance',type=float,default=.05)
    ap.add_argument('--logits-chunk-tokens',type=int,default=128)
    ap.add_argument('--temperature',type=float,default=1.0)
    ap.add_argument('--gdn-scalar-dtype',choices=['bf16','fp32'],default='bf16',
                    help='bf16 matches FSDP mixed-precision forward; fp32 diagnoses native RTP scalar preservation')
    ap.add_argument('--report-only',action='store_true')
    ap.add_argument('--config-only',action='store_true',help='Validate actual checkpoint configuration and case schema, without loading weights or starting CUDA')
    args=ap.parse_args()
    if args.logits_chunk_tokens<1 or args.temperature<=0:ap.error('Invalid chunk size/temperature')
    args.model=args.model.resolve(strict=True)
    metadata(args.model)
    if not 0 < args.tolerance <= .05:ap.error('Tolerance must be in (0, .05]')
    raw=json.loads(args.cases.read_text())
    cases=raw['cases'] if isinstance(raw,dict) else raw
    if not isinstance(cases,list) or not cases:raise ValueError('Expected non-empty list of token cases')
    for case in cases:
        p,o,l=(case[name] for name in ['prompt_token_ids','output_token_ids','raw_log_probs'])
        if not p or not o or len(o)!=len(l):raise ValueError('Each case needs prompt tokens and matching generated token/logprob lists')
        if not all(type(v) is int for v in p+o):raise ValueError('Token IDs must be integers')
    import torch
    from transformers import AutoConfig,AutoModelForImageTextToText
    sys.path.insert(0,str(Path(__file__).parent/'overrides'))
    from qwen36_training_overrides import configure_text_only,prepare_text_only
    torch.manual_seed(36)
    config=AutoConfig.from_pretrained(args.model,local_files_only=True,attn_implementation='sdpa')
    configure_text_only(config)
    if args.config_only:
        plan={'status':'configuration_only','model':str(args.model),'model_type':config.model_type,
              'architectures':config.architectures,'layers':config.text_config.num_hidden_layers,
              'checkpoint_config_sha256':hashlib.sha256((args.model/'config.json').read_bytes()).hexdigest(),
              'cases':len(cases),'cuda_initialized':torch.cuda.is_initialized(),'numerical_comparison_run':False}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(plan,indent=2)+'\n')
        print('HF_REFERENCE_CONFIG_CHECK',json.dumps(plan))
        return 0
    model=AutoModelForImageTextToText.from_pretrained(args.model,config=config,dtype=torch.bfloat16,
                                                    local_files_only=True,trust_remote_code=False,
                                                    attn_implementation='sdpa')
    prepare_text_only(model)
    special=[]
    index=json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    for name,p in model.named_parameters():
        if name.endswith(('.linear_attn.A_log','.linear_attn.dt_bias')):
            if args.gdn_scalar_dtype=='fp32':
                # Promoting an already-BF16 value would not recover the original
                # FP32 scalars. Read these tiny tensors directly from the source.
                from safetensors import safe_open
                with safe_open(args.model/index[name],framework='pt',device='cpu') as shard:
                    original=shard.get_tensor(name)
                p.data=original.float()
            else:
                p.data=p.data.bfloat16()
            special.append({'name':name,'dtype':str(p.dtype)})
    model=model.to(args.device).eval()
    checkpoint=load_checkpoint(args.model)
    device=torch.device(args.device)
    loaded={name:list(param.shape) for name,param in model.named_parameters() if param.device==device}
    if loaded != checkpoint.text | checkpoint.visual:
        raise RuntimeError('HF reference parameters differ from the pinned checkpoint on CUDA')
    report={'schema':'qwen35-122b-hf-selected-logprob-comparison-v1', **identity(), 'cuda_numel':sum(math.prod(shape) for shape in loaded.values()),'checkpoint':str(args.model),
            'device':args.device,'parameter_dtype':'bfloat16','gdn_scalar_dtype':args.gdn_scalar_dtype,
            'reference_path':'padded single-sequence HF ConditionalGeneration / SDPA / eager experts / no cache / no MTP',
            'comparison_dtype':'FP32 log_softmax and selected-token log probabilities',
            'scope_note':'No FSDP/NCCL collectives in this independent HF reference; bf16 parameter compute matches the training forward policy',
            'tolerance':args.tolerance,'started_unix':time.time(),'cases':[]}
    total_error=0.;count=0;maximum=0.;all_finite=True
    with torch.inference_mode():
        for number,case in enumerate(cases):
            prompt=list(case['prompt_token_ids']);outputs=list(case['output_token_ids'])
            expected=[float(v) for v in case['raw_log_probs']]
            actual=[]
            # Only the selected output rows reach the LM head; logits are never
            # serialized. Chunking bounds the temporary [tokens,vocabulary] tensor.
            for start in range(0,len(outputs),args.logits_chunk_tokens):
                end=min(len(outputs),start+args.logits_chunk_tokens)
                ids=torch.tensor([prompt+outputs[:end-1]],dtype=torch.int64,device=args.device)
                mask=torch.ones_like(ids)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                    result=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=end-start)
                logits=result.logits[0].float()/float(case.get('temperature',args.temperature))
                targets=torch.tensor(outputs[start:end],dtype=torch.int64,device=args.device)
                selected=logits.log_softmax(dim=-1).gather(-1,targets[:,None]).squeeze(-1)
                actual.extend(selected.cpu().tolist())
                del result,logits,selected,ids,mask,targets
            differences=[a-b for a,b in zip(actual,expected)]
            finite=all(math.isfinite(v) for v in actual+expected+differences)
            absolute=[abs(v) for v in differences]
            max_error=max(absolute) if finite else float('inf')
            mean_error=sum(absolute)/len(absolute) if finite else float('inf')
            row={'case_index':number,'id':case.get('id',str(number)),
                 'prompt_tokens':len(prompt),'output_tokens':len(outputs),
                 'prompt_token_ids':prompt,'output_token_ids':outputs,'raw_log_probs':expected,'hf_log_probs':actual,
                 'hf_minus_rtp':differences,'max_abs_error':max_error,'mean_abs_error':mean_error,
                 'finite':finite,'passed':finite and max_error<=args.tolerance}
            report['cases'].append(row)
            maximum=max(maximum,max_error);total_error+=sum(absolute);count+=len(absolute);all_finite &= finite
            print('HF_LOGPROB_CASE',json.dumps({k:v for k,v in row.items() if not isinstance(v,list)}),flush=True)
    report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device), max_abs_error=maximum,mean_abs_error=total_error/count if all_finite else float('inf'),
                  finite=all_finite,selected_tokens=count,passed=all_finite and maximum<=args.tolerance,
                  gdn_scalar_parameter_examples=special[:4],finished_unix=time.time())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('HF_LOGPROB_SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ['cases','gdn_scalar_parameter_examples']}),flush=True)
    return 0 if report['passed'] or args.report_only else 2


if __name__=='__main__':raise SystemExit(main())
