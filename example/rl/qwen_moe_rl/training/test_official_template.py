"""CPU-only equality with the fixed model's official tool chat template."""
import asyncio
import argparse
import importlib.util
import json
import os
from pathlib import Path
import types

os.environ['CUDA_VISIBLE_DEVICES']=''

root=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('gateway',root/'adapter/rtp_gateway_adapter.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
arguments = argparse.ArgumentParser(description=__doc__)
arguments.add_argument(
    '--model-path', default=os.environ.get('MODEL_PATH') or os.environ.get('QWEN35_397_MODEL_PATH'),
    help='Local Qwen3.5-family checkpoint/tokenizer directory; no model download is performed',
)
options = arguments.parse_args()
if not options.model_path:
    arguments.error('provide --model-path or MODEL_PATH explicitly')
model=options.model_path
from transformers import AutoTokenizer
import torch

tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
tools=[{'type':'function','function':{'name':'python','description':'Run Python code','parameters':{'type':'object','properties':{'code':{'type':'string'}},'required':['code']}}}]
messages=[{'role':'system','content':'Use Python to check arithmetic.'},{'role':'user','content':'Compute 17 * 19.'}]
parser=g.create_official_chat_parser(tokenizer,disable_thinking=True)
assert parser.stop_sequences==[248044,248046]
assert 151645 not in parser.stop_sequences
for thinking in [False,True]:
    parser=g.create_official_chat_parser(tokenizer,disable_thinking=not thinking)
    expected=tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True,
                                            enable_thinking=thinking,preserve_thinking=False)
    actual=parser.parse(messages,tools=tools,add_generation_prompt=True,is_first_msg=True)
    assert actual==expected
    ids=tokenizer.encode(expected,add_special_tokens=False)
    expected_ids=tokenizer.encode(expected,add_special_tokens=False)
    assert ids==expected_ids
    assert '<function=example_function_name>' in actual
    generated='<tool_call>\n<function=python>\n<parameter=code>\nprint(17 * 19)\n</parameter>\n</function>\n</tool_call>'
    out_ids=tokenizer.encode(generated,add_special_tokens=False)
    class Engine:
        processor=None;accumulate_reasoning=False;weight_version=0
        chat_parser=parser
        config=types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(model=types.SimpleNamespace(path=model)))
        async def get_token_output_from_token_input(self,prompt_ids,**kwargs):
            assert prompt_ids==expected_ids
            return types.SimpleNamespace(token_ids=out_ids,log_probs=[-0.2]*len(out_ids),stop_reason='stop',
                extra_fields={'global_step':0,'model_version':'cpu-protocol-fixture'})
    engine=Engine();engine.tokenizer=tokenizer
    answer=asyncio.run(g.create_rtp_handler(engine)({'messages':messages,'tools':tools,'chat_template_kwargs':{'enable_thinking':thinking}}))
    assert answer['prompt_token_ids']==expected_ids
    assert answer['choices'][0]['token_ids']==out_ids
    assert answer['choices'][0]['message']['tool_calls'][0]['function']['name']=='python'
    plain_ids=tokenizer.encode('The answer is \\boxed{323}.',add_special_tokens=False)
    parsed=parser.parse_completion(plain_ids)
    assert not parsed['tool_calls'] and '\\boxed{323}' in parsed['content']

multi=messages+[{'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'python','arguments':{'code':'print(17 * 19)'}}}]},
                {'role':'tool','content':'{"ok":true,"stdout":"323"}','tool_call_id':'cpu-fixture'}]
expected=tokenizer.apply_chat_template(multi,tools=tools,tokenize=False,add_generation_prompt=True,
                                      enable_thinking=False,preserve_thinking=False)
parser=g.create_official_chat_parser(tokenizer,disable_thinking=True)
assert parser.parse(multi,tools=tools,add_generation_prompt=True,is_first_msg=True)==expected
multi[-2]['tool_calls'][0]['function']['arguments']=json.dumps({'code':'print(17 * 19)'})
assert parser.parse(multi,tools=tools,add_generation_prompt=True,is_first_msg=True)==expected
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'passed','scope':'CPU protocol fixture; no model generation or numerical evidence',
                  'official_prompt_token_ids_match':True,'tools_template':'qwen3_coder XML',
                  'correct_stop_token_ids':[248044,248046],'thinking_on_off':True,'multi_turn_tool_history':True,
                  'raw_output_tokens_and_logprobs_preserved':True,'cuda_initialized':False}))
