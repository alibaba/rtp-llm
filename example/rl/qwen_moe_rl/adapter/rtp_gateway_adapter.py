"""Token-preserving RTP gateway with native Qwen3.6 XML tool-call decoding."""
from __future__ import annotations

import json
import re
import time
import uuid


def normalize_tool_history(messages):
    """Canonical HF representation of OpenAI's JSON-string tool arguments."""
    output=[]
    for original in messages:
        message=dict(original)
        if message.get('tool_calls'):
            calls=[]
            for original_call in message['tool_calls']:
                call=dict(original_call)
                function=dict(call.get('function',call))
                if isinstance(function.get('arguments'),str):
                    function['arguments']=json.loads(function['arguments'])
                if 'function' in call:call['function']=function
                else:call=function
                calls.append(call)
            message['tool_calls']=calls
        output.append(message)
    return output


def _visible_completion(raw):
    """Keep tool syntax outside the generated reasoning block only."""
    reasoning = None
    if '</think>' in raw:
        before, raw = raw.split('</think>', 1)
        reasoning = before.split('<think>', 1)[-1].strip()
    elif '<think>' in raw:
        return '', raw.split('<think>', 1)[-1], False
    raw=raw.strip()
    if raw.startswith('<|im_start|>assistant\n'):
        raw=raw[len('<|im_start|>assistant\n'):]
    for marker in ('<|im_end|>', '<|endoftext|>'):
        while raw.endswith(marker):
            raw=raw[:-len(marker)].rstrip()
    return raw.strip(), reasoning, True


def parse_generated_tool_calls(raw, tools=()):
    """Decode complete generated XML (or legacy JSON) calls, without inventing any.

    Qwen3.6 uses <tool_call><function=python><parameter=code>… syntax.
    Text token IDs and log probabilities are handled separately and stay untouched.
    Incomplete calls from a length-truncated generation remain ordinary content.
    """
    visible, reasoning, outside_reasoning = _visible_completion(raw)
    schemas = {item.get('function', {}).get('name'): item.get('function', {}).get('parameters', {}).get('properties', {})
               for item in tools}
    calls = []
    spans = []
    if outside_reasoning:
        for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', visible, flags=re.S):
            body = match.group(1)
            function = re.fullmatch(r'\s*<function=([A-Za-z_][A-Za-z0-9_.-]*)>\s*(.*?)\s*</function>\s*', body, flags=re.S)
            if function:
                name, parameter_text = function.groups()
                arguments = {}
                consumed = []
                valid = True
                for parameter in re.finditer(r'<parameter=([A-Za-z_][A-Za-z0-9_.-]*)>(.*?)</parameter>', parameter_text, flags=re.S):
                    key, value = parameter.groups()
                    if key in arguments:
                        valid = False
                        break
                    value = value.strip()
                    kind = schemas.get(name, {}).get(key, {}).get('type', 'string')
                    if kind != 'string':
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            valid = False
                            break
                    arguments[key] = value
                    consumed.append(parameter.span())
                remainder = parameter_text
                for start, end in reversed(consumed):
                    remainder = remainder[:start]+remainder[end:]
                if not valid or remainder.strip():
                    continue
                calls.append({'name': name, 'arguments': arguments})
                spans.append(match.span())
            else:
                try:
                    call = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if isinstance(call, dict) and isinstance(call.get('name'), str) and isinstance(call.get('arguments'), dict):
                    calls.append({'name': call['name'], 'arguments': call['arguments']})
                    spans.append(match.span())
    content = visible
    for start, end in reversed(spans):
        content = content[:start]+content[end:]
    return {'content': content.strip(), 'reasoning': reasoning, 'tool_calls': calls}


def create_official_chat_parser(tokenizer, processor=None, disable_thinking=False):
    """Drop-in parser for RTPVerlEngine/completer; prompt syntax stays official."""
    from rllm.parser import ChatTemplateParser

    class OfficialQwen36ChatParser(ChatTemplateParser):
        def __init__(self):
            self.tokenizer=tokenizer
            self.processor=None  # this deployment intentionally accepts text only
            self.disable_thinking=disable_thinking
            self.bos_token=tokenizer.bos_token
            self.eos_token=tokenizer.eos_token
            self.eot_token='<|im_end|>\n'
            self.system_token='<|im_start|>system\n'
            self.user_token='<|im_start|>user\n'
            self.assistant_token='<|im_start|>assistant\n'
            self.image_token='<|image_pad|>'
            self.vision_start_token='<|vision_start|>'
            self.vision_end_token='<|vision_end|>'
            stops={getattr(tokenizer,'eos_token_id',None)}
            for token in ('<|im_end|>','<|endoftext|>'):
                value=tokenizer.convert_tokens_to_ids(token)
                if value is not None and value!=getattr(tokenizer,'unk_token_id',None):stops.add(int(value))
            self.stop_sequences=sorted(int(value) for value in stops if value is not None)
            probe=[{'role':'user','content':''}]
            prefix=tokenizer.apply_chat_template(probe,tokenize=False,add_generation_prompt=False,
                                                enable_thinking=not disable_thinking)
            with_prompt=tokenizer.apply_chat_template(probe,tokenize=False,add_generation_prompt=True,
                                                     enable_thinking=not disable_thinking)
            if not with_prompt.startswith(prefix):
                raise ValueError('Official Qwen chat template generation prefix is not composable')
            self.generation_prompt=with_prompt[len(prefix):]

        def parse(self,messages,add_generation_prompt=False,is_first_msg=False,tools=None,
                  accumulate_reasoning=False,**kwargs):
            schemas=[tool.json if hasattr(tool,'json') else tool for tool in (tools or [])]
            return self.tokenizer.apply_chat_template(normalize_tool_history(messages),tools=schemas or None,tokenize=False,
                add_generation_prompt=add_generation_prompt,enable_thinking=not self.disable_thinking,
                preserve_thinking=bool(accumulate_reasoning))

        def parse_completion(self,completion_ids):
            from rllm.tools.tool_base import ToolCall
            parsed=parse_generated_tool_calls(self.tokenizer.decode(completion_ids,skip_special_tokens=False))
            parsed['tool_calls']=[ToolCall(name=call['name'],arguments=call['arguments']) for call in parsed['tool_calls']]
            return parsed

    return OfficialQwen36ChatParser()


def create_rtp_handler(engine):
    async def handler(body):
        messages = body.get('messages', [])
        tools = body.get('tools', [])
        template_options = body.get('chat_template_kwargs') or {}
        thinking = template_options.get('enable_thinking')
        disable_thinking = not thinking if thinking is not None else bool(getattr(engine.chat_parser, 'disable_thinking', False))
        options=dict(template_options)
        options['enable_thinking']=not disable_thinking
        options.setdefault('preserve_thinking',bool(engine.accumulate_reasoning))
        prompt=engine.tokenizer.apply_chat_template(normalize_tool_history(messages),tools=tools or None,
            tokenize=False,add_generation_prompt=True,**options)
        prompt_ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
        kwargs = {key:body[key] for key in ('temperature','top_p','top_k','seed') if body.get(key) is not None}
        maximum = body.get('max_tokens', body.get('max_completion_tokens'))
        if maximum is not None:
            kwargs['max_tokens'] = maximum
        result = await engine.get_token_output_from_token_input(prompt_ids, **kwargs)
        ids = list(result.token_ids)
        probabilities = list(result.log_probs or [])
        if not ids or len(ids) != len(probabilities):
            raise RuntimeError('RTP gateway requires one actual logprob for every output token')
        raw = engine.tokenizer.decode(ids, skip_special_tokens=False)
        native_parsed = parse_generated_tool_calls(raw, tools)
        parsed = native_parsed
        message = {'role':'assistant','content':parsed.get('content') or ''}
        if parsed.get('reasoning'):
            message['reasoning'] = parsed['reasoning']
        calls = []
        for index, call in enumerate(parsed.get('tool_calls') or []):
            name = call.name if hasattr(call,'name') else call.get('name','')
            arguments = call.arguments if hasattr(call,'arguments') else call.get('arguments',{})
            calls.append({'id':f'call_{uuid.uuid4().hex[:12]}_{index}','type':'function',
                          'function':{'name':name,'arguments':json.dumps(arguments) if isinstance(arguments,dict) else str(arguments)}})
        finish = result.stop_reason or 'stop'
        if calls:
            message['tool_calls'] = calls
            finish = 'tool_calls'
        native = result.extra_fields or {}
        native_step = native.get('global_step')
        if type(native_step) is not int or native_step<0 or not native.get('model_version'):
            raise RuntimeError('RTP gateway requires the actual native model version and nonnegative global step')
        return {'id':f'chatcmpl-rtp-{uuid.uuid4().hex[:12]}','object':'chat.completion','created':int(time.time()),
                'model':body.get('model',engine.config.actor_rollout_ref.model.path),
                'choices':[{'index':0,'message':message,'finish_reason':finish,'token_ids':ids,
                            'logprobs':{'content':[{'logprob':value} for value in probabilities]}}],
                'usage':{'prompt_tokens':len(prompt_ids),'completion_tokens':len(ids),'total_tokens':len(prompt_ids)+len(ids)},
                'prompt_token_ids':prompt_ids,'weight_version':native_step,'engine_weight_version':engine.weight_version,
                'rtp_model_version':native.get('model_version'),'rtp_global_step':native_step,
                'rtp_stop_reason':result.stop_reason}

    return handler
