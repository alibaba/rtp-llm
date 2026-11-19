#!/usr/bin/env python3
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

ROOT = Path('results')
MODEL = ''
URL = 'http://127.0.0.1:29700/v1/chat/completions'
INPUT = 6656
OUTPUT = 347
PREFIX = 5200
WORDS = ('amber', 'bridge', 'copper', 'delta', 'ember', 'forest', 'garden', 'harbor',
         'island', 'jungle', 'kernel', 'lantern', 'meadow', 'nebula', 'orange',
         'planet', 'quartz', 'river', 'silver', 'tunnel', 'violet', 'window',
         'yellow', 'zenith')
TASKS = {
    'tutorial': 'Write a hands-on Python tutorial for building a tiny search engine. Include short code examples, careful explanations, common mistakes, and test cases. Continue for at least 500 words.',
    'reason': 'Solve a sequence of varied reasoning questions about probability, scheduling, and logic. Show each step in natural language with explanations, continuing for at least 500 words.',
}


def render_ids(tok, text):
    result = tok.apply_chat_template([{'role':'user','content':text}], tokenize=True,
                                     add_generation_prompt=True, enable_thinking=False)
    return result['input_ids'] if hasattr(result, 'keys') else result


def build_prompts(tok, seeds):
    single = [ids[0] for word in WORDS
              if len(ids := tok.encode(' ' + word, add_special_tokens=False)) == 1]
    common_ids = random.Random(1401).choices(single, k=PREFIX)
    common = tok.decode(common_ids, clean_up_tokenization_spaces=False)
    prompts = {}
    for kind, task in TASKS.items():
        for seed in seeds:
            rng = random.Random(seed)
            suffix_ids = rng.choices(single, k=INPUT-PREFIX+100)
            size = INPUT-PREFIX-100
            for _ in range(12):
                text = common + tok.decode(suffix_ids[:size], clean_up_tokenization_spaces=False) + '\n' + task
                n = len(render_ids(tok, text))
                if n == INPUT:
                    prompts[(kind,seed)] = text
                    break
                size += INPUT-n
            else:
                raise RuntimeError(f'{kind}/{seed}: cannot hit exact length, got {n}')
    return prompts


async def request(session, text, kind, seed, temperature):
    payload = {
        'model':'default', 'messages':[{'role':'user','content':text}],
        'temperature':temperature, 'top_k':1 if temperature == 0 else 0,
        'top_p':1.0 if temperature == 0 else 0.95,
        'max_completion_tokens':OUTPUT, 'ignore_eos':True, 'stream':True,
        'stream_options':{'include_usage':True},
        'chat_template_kwargs':{'enable_thinking':False}, 'enable_thinking':False,
        'extra_configs':{'min_new_tokens':OUTPUT,'ignore_eos':True},
    }
    start = time.perf_counter(); first = None; usage = None; aux = {}; error = None
    async with session.post(URL, json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {(await response.text())[:800]}')
        async for raw in response.content:
            for line in raw.decode('utf-8',errors='replace').splitlines():
                if not line.startswith('data: ') or line == 'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                error = event.get('error') or error
                usage = event.get('usage') or usage
                aux = event.get('aux_info') or aux
                for choice in event.get('choices') or []:
                    delta = choice.get('delta') or {}
                    if first is None and (delta.get('content') or delta.get('reasoning_content')):
                        first = time.perf_counter()
    end = time.perf_counter()
    if error or usage is None or first is None:
        raise RuntimeError(f'incomplete {kind}/{seed}: {error}, {usage}, {aux}')
    out = usage['completion_tokens']; rounds = aux.get('speculative_draft_rounds',0)
    accepted = aux.get('speculative_accepted_tokens_per_pos') or []
    if usage.get('prompt_tokens') != INPUT or out != OUTPUT or aux.get('pd_sep') is not True:
        raise RuntimeError(f'{kind}/{seed}: expected P/D input={INPUT}, output={OUTPUT}; got {usage}, {aux}')
    if kind != 'prime' and aux.get('reuse_len') != 5120:
        raise RuntimeError(f"{kind}/{seed}: expected 5120 reused tokens; got {aux.get('reuse_len')}")
    if (not isinstance(rounds, int) or rounds <= 0 or len(accepted) != 3
            or any(type(n) is not int for n in accepted)
            or not rounds >= accepted[0] >= accepted[1] >= accepted[2] >= 0):
        raise RuntimeError(f'{kind}/{seed}: expected three MTP proposal positions; got {rounds=}, {accepted=}')
    return {
        'kind':kind,'seed':seed,'temperature':temperature,
        'input_tokens':usage.get('prompt_tokens'),'output_tokens':out,
        'reuse_len':aux.get('reuse_len'),'pd_sep':aux.get('pd_sep'),
        'draft_rounds':rounds,'accepted_per_pos':accepted,
        'draft_accept_rate':sum(accepted)/(3*rounds),
        'ttft_ms':(first-start)*1000,'tpot_ms':(end-first)*1000/max(1,out-1),
        'latency_ms':(end-start)*1000,
    }


async def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    seeds = list(range(seed_start, seed_start + 21))
    prompts = build_prompts(tok, seeds)
    timeout = aiohttp.ClientTimeout(total=1200)
    rows = []
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False,
                                           connector=aiohttp.TCPConnector(force_close=True)) as session:
        prime = await request(session, prompts[('reason',seed_start)], 'prime', seed_start, 0)
        print(json.dumps({'prime':prime}), flush=True)
        for i, seed in enumerate(seeds[1:]):
            kind = 'reason' if i % 5 == 0 else 'tutorial'
            row = await request(session, prompts[(kind,seed)], kind, seed, 0)
            rows.append(row)
            print(json.dumps({'request':len(rows), **row}), flush=True)
    summary = {
        'requests':len(rows), 'input_tokens':INPUT, 'output_tokens':OUTPUT,
        'mean_reuse_len':statistics.mean(r['reuse_len'] for r in rows),
        'mean_draft_accept_rate':statistics.mean(r['draft_accept_rate'] for r in rows),
        'weighted_draft_accept_rate':sum(sum(r['accepted_per_pos']) for r in rows)/(3*sum(r['draft_rounds'] for r in rows)),
        'mean_ttft_ms':statistics.mean(r['ttft_ms'] for r in rows),
        'mean_tpot_ms':statistics.mean(r['tpot_ms'] for r in rows),
        'mean_e2e_ms':statistics.mean(r['latency_ms'] for r in rows),
        'p95_ttft_ms':sorted(r['ttft_ms'] for r in rows)[18],
        'p95_tpot_ms':sorted(r['tpot_ms'] for r in rows)[18],
    }
    (ROOT/'result.json').write_text(json.dumps({'prime':prime,'requests':rows,'summary':summary},indent=2))
    print(json.dumps({'summary':summary}),flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:29700')
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--seed-start', type=int, default=800)
    args = parser.parse_args()
    MODEL = args.model_dir
    URL = args.base_url.rstrip('/') + '/v1/chat/completions'
    ROOT = args.result_dir
    seed_start = args.seed_start
    asyncio.run(main())
