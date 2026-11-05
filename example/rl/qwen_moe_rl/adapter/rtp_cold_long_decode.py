"""Optional one-request diagnostic using the same live controller's gated client.

No weights, scheduler, tool or training path is changed. Root must explicitly
request this probe after the independent three-case HF comparison succeeds.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import uuid

FIXTURE_SHA='2642350d3158fb11da15589c3e848598efc09d6852e57cdd0311688c635339cf'


def _strict_prompt_ids(encoded,vocab_size):
    if type(vocab_size) is not int or vocab_size<=0:
        raise ValueError('Tokenizer validation requires a positive integer model vocabulary size')
    if not isinstance(encoded,(list,tuple)) or not encoded:
        raise ValueError('Chat template must return a nonempty flat token-ID list')
    prompt=list(encoded)
    for index,token in enumerate(prompt):
        if type(token) is not int or not 0<=token<vocab_size:
            raise ValueError(f'Invalid chat-template token at {index}: {token!r} '
                             f'({type(token).__name__}); expected Python int in [0,{vocab_size})')
    return prompt


def _prompt_ids(tokenizer,messages,vocab_size):
    # New Transformers defaults may return BatchEncoding. Never list() its keys.
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,
        add_generation_prompt=True,enable_thinking=False)
    return _strict_prompt_ids(encoded,vocab_size)


async def run(server,expected_version):
    if not server._native_available() or server.tp_size!=16 or server.poisoned:
        raise RuntimeError('Cold long decode requires the accepted live TP16 cold checkpoint')
    if getattr(server.client,'_max_active_requests',None)!=1:
        raise RuntimeError('Cold long decode requires the actual shared native GATE1 client')
    from rtp_native_server import _deployment_root,_new_control
    from rtp_llm.RL.token_rpc_client import TokenSamplingParams
    fixture=_deployment_root()/'training/data/eight-original-prompts.jsonl'
    payload=fixture.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=FIXTURE_SHA:
        raise RuntimeError('Original eight-question fixture identity changed')
    task=json.loads(payload.splitlines()[0])
    messages=[{'role':'user','content':task['original_question']}]
    prompt=_prompt_ids(server.tokenizer,messages,server.client.vocab_size)
    if not prompt or len(prompt)+2048>8192:
        raise RuntimeError('Cold long-decode token budget exceeds the existing context limit')
    control=await _new_control(server.replica_rank)
    try:
        statuses=await control.status()
        if len(statuses)!=16 or any(row.phase!='running' or row.model_version!=expected_version for row in statuses):
            raise RuntimeError('Long decode does not identify one live accepted native checkpoint')
        step=statuses[0].global_step
    finally:await control.close()
    params=TokenSamplingParams(max_new_tokens=2048,min_new_tokens=2048,ignore_eos=True,
        temperature=1.,top_p=1.,top_k=0,stop_token_ids=(),logprobs=True,logprobs_mode='raw_logprobs')
    request_id=uuid.uuid4().int&((1<<63)-1);started=time.time()
    result=await server.client.generate(request_id,prompt,params,expected_version=expected_version)
    values=result.log_probs or [];ids=result.token_ids
    if len(ids)!=2048 or len(values)!=2048 or result.model_version!=expected_version or result.global_step!=step:
        raise RuntimeError('Long decode length/raw-score/policy metadata differs from the actual request')
    if not all(type(token) is int and token>=0 for token in ids) or not all(math.isfinite(v) and v<=0 for v in values):
        raise RuntimeError('Long decode native IDs or FP32 selected scores are invalid')
    bits=struct.pack('<2048f',*values)
    if list(struct.unpack('<2048f',bits))!=values or result.logprobs_mode!='raw_logprobs':
        raise RuntimeError('Long decode did not preserve complete native FP32 raw selected-token scores')
    return dict(schema='q397-cold-long-decode-smoke-v1',passed=True,counts_as_rl_acceptance=False,
        scope='one original DeepScaleR long-decode stability request; no answer grading, Python Tool or RL update',
        replica_rank=server.replica_rank,tp_size=16,dp_size=1,pp_size=1,ep_size=1,max_active_requests=1,
        native_boot=server.boot,native_incarnation=server.incarnation,native_request_id=request_id,
        model_version=result.model_version,global_step=result.global_step,fixture_sha256=FIXTURE_SHA,
        source_row_index=task['source_row_index'],original_question=task['original_question'],messages=messages,
        chat_template_options=dict(enable_thinking=False,add_generation_prompt=True),
        sampling=dict(temperature=1.,top_p=1.,top_k=0,min_new_tokens=2048,max_new_tokens=2048,ignore_eos=True),
        prompt_token_ids=prompt,output_token_ids=ids,raw_log_probs=values,raw_fp32_bits_hex=bits.hex(),
        raw_logprobs_fp32_sha256=hashlib.sha256(bits).hexdigest(),response_tokens=2048,raw_logprob_count=2048,
        logprobs_mode=result.logprobs_mode,finish_reason=result.finish_reason,
        started_unix=started,finished_unix=time.time(),step_measurement=os.environ.get('RTP_STEP_MEASUREMENT'))


def tokenizer_check(model,fixture,output=None):
    """CPU-only actual tokenizer check; loads no model or native RPC client."""
    from collections.abc import Mapping
    import sys
    import transformers
    from transformers import AutoTokenizer,AutoConfig

    payload=Path(fixture).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=FIXTURE_SHA:
        raise RuntimeError('Original eight-question fixture identity changed')
    task=json.loads(payload.splitlines()[0])
    messages=[{'role':'user','content':task['original_question']}]
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    config=AutoConfig.from_pretrained(str(model),local_files_only=True)
    text_config=getattr(config,'text_config',config)
    vocab_size=text_config.vocab_size
    prompt=_prompt_ids(tokenizer,messages,vocab_size)
    default=tokenizer.apply_chat_template(messages,tokenize=True,
        add_generation_prompt=True,enable_thinking=False)
    if not isinstance(default,Mapping) or 'input_ids' not in default:
        raise RuntimeError('Current default chat-template result must expose BatchEncoding input_ids')
    reference=_strict_prompt_ids(default['input_ids'],vocab_size)
    if prompt!=reference:
        raise RuntimeError('Explicit return_dict=False IDs differ from default BatchEncoding input_ids')
    if len(prompt)+2048>8192:
        raise RuntimeError('Cold long-decode token budget exceeds the existing context limit')
    rejected=0
    for bad in [list(default),[],[True],[1.0],[-1],[vocab_size],{'input_ids':prompt}]:
        try:_strict_prompt_ids(bad,vocab_size)
        except ValueError:rejected+=1
        else:raise RuntimeError('Strict prompt validation accepted malformed or out-of-vocabulary input')
    torch_module=sys.modules.get('torch')
    cuda=getattr(torch_module,'cuda',None) if torch_module is not None else None
    cuda_initialized=bool(cuda.is_initialized()) if cuda is not None else False
    if cuda_initialized:
        raise RuntimeError('CPU tokenizer check unexpectedly initialized CUDA')
    result=dict(schema='q397-cold-tokenizer-cpu-check-v1',passed=True,used_gpu=False,
        cuda_initialized=cuda_initialized,external_model_requests=False,
        model_path=str(Path(model).resolve()),transformers_version=transformers.__version__,
        tokenizer_class=type(tokenizer).__name__,config_model_type=config.model_type,
        text_model_type=getattr(text_config,'model_type',None),vocab_size=vocab_size,
        fixture_sha256=FIXTURE_SHA,source_row_index=task['source_row_index'],
        explicit_return_dict_false=True,default_return_type=type(default).__name__,
        default_keys=list(default.keys()),default_input_ids_equal=True,
        all_python_integer_ids=all(type(token) is int for token in prompt),
        prompt_tokens=len(prompt),min_token_id=min(prompt),max_token_id=max(prompt),
        prompt_token_ids=prompt,prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        strict_negative_cases_rejected=rejected,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checked_unix=time.time())
    if output is not None:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Actual CPU tokenizer check for the cold long-decode helper')
    parser.add_argument('--tokenizer-check',action='store_true',required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,
        default=Path(__file__).resolve().parents[1]/'training/data/eight-original-prompts.jsonl')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(tokenizer_check(args.model,args.fixture,args.output),indent=2))


if __name__=='__main__':
    main()
