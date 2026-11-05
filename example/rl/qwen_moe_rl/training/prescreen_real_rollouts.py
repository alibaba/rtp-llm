"""Real Qwen/RTP/Python-Tool prescreen; never constructs or overrides rewards."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import time


async def main_async(args):
    from openai import AsyncOpenAI
    from python_math_agent import SYSTEM_PROMPT,TOOLS,execute_python,extract_boxed
    from rllm.rewards.math_utils.utils import grade_answer_mathd,grade_answer_sympy
    rows=[json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()][:args.questions]
    if len(rows)<args.batch_size:raise ValueError('Need at least batch_size real candidate questions')
    semaphore=asyncio.Semaphore(args.parallel)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    client=AsyncOpenAI(base_url=args.base_url,api_key='EMPTY',timeout=args.timeout,max_retries=0)
    async def episode(index,sample,row):
        async with semaphore:
            messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':row['question']}]
            result={'question_index':index,'sample_index':sample,'source_row_index':row.get('source_row_index'),
                    'ground_truth':str(row['ground_truth']),'tool_records':[],'segments':[],
                    'started_unix':time.time(),'error':None}
            try:
                final=''
                for turn in range(args.max_turns):
                    response=await client.chat.completions.create(model=args.model,messages=messages,
                        tools=TOOLS,tool_choice='auto',max_tokens=args.max_tokens,temperature=args.temperature,
                        top_p=1.0,extra_body={'top_k':0,'chat_template_kwargs':{'enable_thinking':False}})
                    if not response.choices:raise RuntimeError('No real completion choices')
                    choice=response.choices[0];message=choice.message
                    response_extra=response.model_extra or {};choice_extra=choice.model_extra or {}
                    ids=choice_extra.get('token_ids')
                    logprobs=choice.logprobs.content if choice.logprobs is not None else None
                    prompt_ids=response_extra.get('prompt_token_ids')
                    if ids and prompt_ids and logprobs:
                        if len(ids)!=len(logprobs):raise RuntimeError('Real token/logprob length mismatch')
                        result['segments'].append({'id':f'q{index}-s{sample}-t{turn}',
                            'prompt_token_ids':list(prompt_ids),'output_token_ids':list(ids),
                            'raw_log_probs':[float(x.logprob) for x in logprobs],
                            'temperature':args.temperature,'rtp_global_step':response_extra.get('rtp_global_step'),
                            'rtp_model_version':response_extra.get('rtp_model_version')})
                    content=message.content or ''
                    assistant={'role':'assistant','content':content}
                    if message.tool_calls:assistant['tool_calls']=[call.model_dump(exclude_none=True) for call in message.tool_calls]
                    messages.append(assistant)
                    if not message.tool_calls:
                        final=content
                        break
                    for call in message.tool_calls:
                        arguments=json.loads(call.function.arguments)
                        if call.function.name!='python':raise RuntimeError('Generated an unsupported tool: '+call.function.name)
                        code=str(arguments.get('code',''))
                        actual=await asyncio.to_thread(execute_python,code)
                        result['tool_records'].append({'turn':turn,'call_id':call.id,'code':code,'result':actual})
                        messages.append({'role':'tool','tool_call_id':call.id,'content':json.dumps(actual,ensure_ascii=False)})
                answer=extract_boxed(final)
                truth=result['ground_truth']
                correct=bool(answer and truth and (grade_answer_mathd(answer,truth) or grade_answer_sympy(answer,truth)))
                result.update(answer=answer,final_content=final,reward=float(correct))
            except Exception as error:
                result['error']=f'{type(error).__name__}: {error}'
                result['reward']=None
            result['finished_unix']=time.time()
            (args.output_dir/f'episode-q{index:03d}-s{sample:02d}.json').write_text(json.dumps(result,indent=2)+'\n')
            print('REAL_PRESCREEN_EPISODE',json.dumps({k:result[k] for k in ['question_index','sample_index','reward','error']}),flush=True)
            return result
    try:
        results=await asyncio.gather(*(episode(i,s,row) for i,row in enumerate(rows) for s in range(args.samples)))
    finally:
        await client.close()
    groups=[]
    for i,row in enumerate(rows):
        episodes=[r for r in results if r['question_index']==i]
        valid=all(r['error'] is None for r in episodes)
        rewards=[r['reward'] for r in episodes] if valid else []
        groups.append({'question_index':i,'source_row_index':row.get('source_row_index'),'valid':valid,
                       'rewards':rewards,'mixed':valid and min(rewards)!=max(rewards),
                       'python_tool_calls':sum(len(r['tool_records']) for r in episodes)})
    ranked=sorted([g for g in groups if g['valid']],key=lambda g:(not g['mixed'],g['python_tool_calls']==0,g['question_index']))
    selected=ranked[:args.batch_size]
    chosen=[rows[g['question_index']] for g in selected]
    mixed=sum(g['mixed'] for g in selected)
    calls=sum(g['python_tool_calls'] for g in selected)
    success=len(selected)==args.batch_size and mixed>=args.minimum_mixed_groups and calls>0
    summary={'schema':'qwen36-real-prescreen-v1','passed':success,'base_url':args.base_url,
             'model':args.model,'sampling':{'temperature':args.temperature,'top_p':1.0,'top_k':0,'max_tokens':args.max_tokens},
             'groups':groups,'selected_groups':selected,'selected_mixed_groups':mixed,
             'selected_python_tool_calls':calls,'error_episodes':sum(r['error'] is not None for r in results),
             'note':'Prescreen rewards are actual mathematical correctness of actual RTP completions. Training rerolls and computes new rewards; prescreen rewards are not injected.'}
    (args.output_dir/'prescreen-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if success:
        (args.output_dir/'train_nonuniform.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in chosen))
    cases=[segment for episode in results if episode['error'] is None for segment in episode['segments']]
    (args.output_dir/'hf-logprob-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    print('REAL_PRESCREEN_SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ['groups','selected_groups']}),flush=True)
    return 0 if success else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--dataset',type=Path,default=Path(__file__).parent/'data/deepscaler_train.jsonl')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--questions',type=int,default=12)
    ap.add_argument('--samples',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--minimum-mixed-groups',type=int,default=1)
    ap.add_argument('--parallel',type=int,default=8)
    ap.add_argument('--max-turns',type=int,default=3)
    ap.add_argument('--max-tokens',type=int,default=1024)
    ap.add_argument('--temperature',type=float,default=1.0)
    ap.add_argument('--timeout',type=float,default=300)
    args=ap.parse_args()
    if args.samples<2 or args.questions<args.batch_size:ap.error('Need at least two samples and batch_size candidate questions')
    return asyncio.run(main_async(args))


if __name__=='__main__':raise SystemExit(main())
