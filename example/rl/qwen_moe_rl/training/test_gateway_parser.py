"""CPU checks of generated-call decoding and exact token/logprob passthrough."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import types

path=Path(__file__).resolve().parent.parent/'adapter/rtp_gateway_adapter.py'
spec=importlib.util.spec_from_file_location('gateway',path);gateway=importlib.util.module_from_spec(spec);spec.loader.exec_module(gateway)
tools=[{'type':'function','function':{'name':'python','parameters':{'type':'object','properties':{'code':{'type':'string'}}}}}]
code='import math\nprint(math.factorial(7))'
raw='<think>I should check.</think>\n<tool_call>\n<function=python>\n<parameter=code>\n'+code+'\n</parameter>\n</function>\n</tool_call><|im_end|>'
parsed=gateway.parse_generated_tool_calls(raw,tools)
assert parsed['tool_calls']==[{'name':'python','arguments':{'code':code}}]
assert parsed['reasoning']=='I should check.' and parsed['content']==''
assert not gateway.parse_generated_tool_calls('<think><tool_call><function=python>do not execute',tools)['tool_calls']
assert not gateway.parse_generated_tool_calls('<tool_call><function=python><parameter=code>truncated',tools)['tool_calls']
legacy='<tool_call>'+json.dumps({'name':'python','arguments':{'code':code}})+'</tool_call>'
assert gateway.parse_generated_tool_calls(legacy,tools)['tool_calls'][0]['arguments']['code']==code
assert len(gateway.parse_generated_tool_calls(legacy+'\n'+legacy,tools)['tool_calls'])==2

class Parser:
    def parse(self,*args,**kwargs):return 'actual-prompt'
    def parse_completion(self,*args):raise AssertionError('XML must use the native parser')

sys.modules['rllm.parser']=types.SimpleNamespace(ChatTemplateParser=types.SimpleNamespace(get_parser=lambda *a,**k:Parser()))
class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['tools']==tools and kwargs['enable_thinking'] is False
        return 'actual-prompt'
    def encode(self,*args,**kwargs):return [101,102,103]
    def decode(self,ids,**kwargs):assert ids==[11,22,33];return raw
class Engine:
    tokenizer=Tokenizer();processor=None;chat_parser=types.SimpleNamespace(disable_thinking=True)
    accumulate_reasoning=False;weight_version=3
    config=types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(model=types.SimpleNamespace(path='test-model')))
    async def get_token_output_from_token_input(self,ids,**kwargs):
        assert ids==[101,102,103]
        return types.SimpleNamespace(token_ids=[11,22,33],log_probs=[-0.1,-0.2,-0.3],stop_reason='stop',extra_fields={'global_step':3,'model_version':'native-v3'})

answer=asyncio.run(gateway.create_rtp_handler(Engine())({'messages':[],'tools':tools,'model':'test-model'}))
choice=answer['choices'][0]
assert choice['token_ids']==[11,22,33]
assert [p['logprob'] for p in choice['logprobs']['content']]==[-0.1,-0.2,-0.3]
assert json.loads(choice['message']['tool_calls'][0]['function']['arguments'])=={'code':code}
assert choice['finish_reason']=='tool_calls' and answer['weight_version']==3
print('QWEN36_XML_GATEWAY_CPU_CHECK_PASSED')
