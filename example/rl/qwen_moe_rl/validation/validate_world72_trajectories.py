#!/usr/bin/env python3
"""Offline original reward/Python Tool/delivered-trace acceptance for world72.

Leaf checks are carried from the previously accepted trajectory validator.
The only rendering change is this model's required thinking=true/effort=low.
No model generation, GPU work, reward injection or training runs here.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,importlib,importlib.util,json,math,os
from pathlib import Path
import struct,sys,unicodedata

DEEPSCALER_SHA256='4e9e7b18248d982b4498a54a6bdb37b1bb653cf5f77bc309fa67a6c2533c8ba4'
DATASET18_SHA256='a1030fb5e9cf9b29517d918e4135d69d50357de3c7f615857e0aa3fc45bfd16d'
def sha(data):return hashlib.sha256(data).hexdigest()
def digest_ids(ids):return sha(json.dumps(ids).encode())
def fp32_bytes(values):return struct.pack(f'<{len(values)}f',*values)
def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def positive(value):return type(value) in (float,int) and math.isfinite(value) and value>0
def normalize(value):return ' '.join(unicodedata.normalize('NFKC',str(value)).split())
def steps(episode):
    for trajectory in episode.get('trajectories',[]):yield from trajectory.get('steps',[])
class Check:
    def __init__(self):self.errors=[];self.details={}
    def need(self,condition,message):
        if not condition:self.errors.append(message)
        return bool(condition)
class Validator:
    def __init__(self,args):
        self.args=args;self.root=args.evidence_root[0].resolve();self.rtp=self.root/'rtp'
        self.inventory={};self.input_errors=[];self.checks=[];self.episodes=[]
        self.training_episodes={};self.generation_receipts=[];self.raw_segments=[];self.episode_policy_steps={}
        self.syncs={};self._grader=None;self._tool_module=None;self._tokenizer=None
    def files(self,folder,pattern,recursive=False):
        folders=set()
        for root in self.args.evidence_root:
            if root.name==folder.name and root.is_dir():folders.add(root.resolve())
            if (root/folder.name).is_dir():folders.add((root/folder.name).resolve())
            folders.update(path.resolve() for path in root.rglob(folder.name) if path.is_dir())
        paths=set()
        for base in folders:
            iterator=base.rglob(pattern) if recursive else base.glob(pattern)
            paths.update(path.resolve() for path in iterator if path.is_file() and not path.name.startswith('._'))
        # Identical copies gathered from different nodes are one original file.
        seen=set();result=[]
        for path in sorted(paths):
            identity=(path.name,sha(path.read_bytes()))
            if identity not in seen:result.append(path);seen.add(identity)
        return result
    def records(self,folder,pattern,recursive=False):return [(path,self.read(path)) for path in self.files(folder,pattern,recursive)]
    def read(self, path, jsonl=False):
        path = Path(path)
        try:
            data = path.read_bytes()
            self.inventory[str(path.resolve())] = {'size_bytes': len(data), 'sha256': sha(data)}
            def reject(value):
                raise ValueError(f'Nonstandard JSON number: {value}')
            if jsonl:
                return [json.loads(line, parse_constant=reject) for line in data.decode().splitlines() if line.strip()]
            return json.loads(data, parse_constant=reject)
        except Exception as exc:
            self.input_errors.append(f'{path}: {type(exc).__name__}: {exc}')
            return [] if jsonl else {}


    def run_check(self, name, function, required=True):
        check = Check()
        try:
            function(check)
        except Exception as exc:
            check.errors.append(f'{type(exc).__name__}: {exc}')
        self.checks.append(dict(name=name, required=required,
                                passed=not check.errors, failures=check.errors,
                                details=check.details))


    def generation_logprobs(self, c):
        index = defaultdict(list)
        seen_steps = set()
        for path, record in self.generation_receipts:
            prefix = f'{path.name} request {record.get("request_id")}: '
            c.need(record.get('engine') == 'rtp-llm' and record.get('logprobs_mode') == 'raw_logprobs', prefix+'Native raw-logprob metadata is missing')
            c.need(positive(record.get('prompt_tokens')) and positive(record.get('response_tokens')) and
                   record.get('response_tokens') == record.get('logprobs'), prefix+'Generation token/logprob lengths disagree')
            key = tuple(record.get(name) for name in ('prompt_sha256', 'response_sha256', 'raw_logprobs_fp32_sha256'))
            c.need(all(valid_sha(x) for x in key), prefix+'Generation receipt has missing payload hashes')
            index[key].append(record)
        c.need(bool(index), 'No actual native model generations were recorded')
        for path, episode in self.episodes:
            policy_steps = set()
            for number, step in enumerate(steps(episode)):
                model_output = step.get('model_output') or {}
                prompt = step.get('prompt_ids') or model_output.get('prompt_ids') or []
                response = step.get('response_ids') or model_output.get('completion_ids') or []
                values = step.get('logprobs') or model_output.get('logprobs') or []
                prefix = f'{path.name} trajectory step {number}: '
                valid = bool(prompt) and bool(response) and len(response) == len(values)
                valid &= all(type(x) is int and x >= 0 for x in prompt+response)
                valid &= all(isinstance(x, (int, float)) and math.isfinite(x) and x <= 1e-5 for x in values)
                if not c.need(valid, prefix+'Missing or invalid raw generated token/logprob arrays'):
                    continue
                bits = fp32_bytes(values)
                c.need(list(struct.unpack(f'<{len(values)}f', bits)) == values, prefix+'Raw logprob values were rounded or converted away from native FP32')
                key = (digest_ids(prompt), digest_ids(response), sha(bits))
                receipts = index.get(key, [])
                if not c.need(bool(receipts), prefix+'Actual episode token/logprob payload is not linked to a native generation receipt'):
                    continue
                versions = {(x.get('global_step'), x.get('model_version')) for x in receipts}
                c.need(len(versions) == 1, prefix+'Identical payload is ambiguously associated with different policy versions')
                native_step, version = next(iter(versions))
                policy_steps.add(native_step)
                seen_steps.add(native_step)
                sync = self.syncs.get(native_step, (None, {}))[1]
                c.need(bool(sync) and version == sync.get('model_version'), prefix+'Generation model version is not a recorded committed policy')
                weight_version = step.get('weight_version', model_output.get('weight_version'))
                if weight_version is not None:
                    c.need(weight_version == native_step, prefix+'rLLM generation weight_version differs from the native policy step')
                self.raw_segments.append(dict(episode_id=episode.get('episode_id'), source_path=str(path),
                    prompt_token_ids=prompt, output_token_ids=response, raw_log_probs=values,
                    global_step=native_step, model_version=version,
                    agent_action=step.get('action'),
                    model_response=step.get('model_response', ''),
                    chat_completions=step.get('chat_completions', [])))
            c.need(len(policy_steps) == 1, f'{path.name}: One episode spans missing or multiple model versions')
            if len(policy_steps) == 1:
                self.episode_policy_steps[episode.get('episode_id')] = next(iter(policy_steps))
        for policy_step in range(self.args.minimum_updates):
            c.need(policy_step in seen_steps, f'No real sampled rollout from policy step {policy_step}; cannot establish the update/rollout loop')
        c.need(bool(self.raw_segments), 'No full token-level generated trajectories are available')
        c.details.update(native_receipts=len(self.generation_receipts), linked_trajectory_segments=len(self.raw_segments),
                         sampled_policy_steps=sorted(seen_steps), selected_tokens=sum(len(x['output_token_ids']) for x in self.raw_segments))


    def load_grader_and_tool(self):
        # Hide CUDA only inside this offline validation process before imports.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.path.insert(0, str(self.args.rllm_source))
        sys.path.insert(0, str(self.args.agent_source))
        if self._grader is None:
            module = importlib.import_module('rllm.rewards.math_utils.utils')
            self._grader = (module.grade_answer_mathd, module.grade_answer_sympy)
            self.read_python_source(Path(module.__file__))
        if self._tool_module is None:
            self._tool_module = importlib.import_module('python_math_agent')
            self.read_python_source(Path(self._tool_module.__file__))


    def read_python_source(self, path):
        data = path.read_bytes()
        self.inventory[str(path.resolve())] = {'size_bytes': len(data), 'sha256': sha(data)}


    def grounded_rewards(self, c):
        source = self.read(self.args.source_dataset)
        if not c.need(isinstance(source, list) and len(source) == 40315, 'Original 40,315-row DeepScaleR dataset is absent or replaced'):
            return
        c.need(self.inventory[str(self.args.source_dataset.resolve())]['sha256'] == DEEPSCALER_SHA256,
               'Original DeepScaleR source SHA256 differs from the previously deployed and preserved dataset')
        self.load_grader_and_tool()
        cached = {}
        groups = defaultdict(list)
        matched_training_steps = set()
        for path, episode in self.episodes:
            prefix = f'{path.name}: '
            task = episode.get('task') or {}
            row_number = task.get('source_row_index')
            if not c.need(type(row_number) is int and 0 <= row_number < len(source), prefix+'Missing original DeepScaleR row index'):
                continue
            original = source[row_number]
            question = task.get('original_question', task.get('question', ''))
            truth = str(task.get('ground_truth', task.get('answer', '')))
            c.need(task.get('data_source') == 'deepscaler_math' and normalize(question) == normalize(original['problem']), prefix+'Question is not grounded in the original DeepScaleR row')
            c.need(truth == str(original['answer']), prefix+'Ground-truth answer differs from the original dataset')
            artifacts = episode.get('artifacts') or {}
            answer = str(artifacts.get('answer', ''))
            c.need(answer == self._tool_module.extract_boxed(str(artifacts.get('final_content', ''))), prefix+'Graded answer does not come from the generated final boxed answer')
            key = (answer, truth)
            if key not in cached:
                cached[key] = bool(answer and truth and (self._grader[0](answer, truth) or self._grader[1](answer, truth)))
            correct = cached[key]
            c.need(episode.get('reward') == float(correct) and episode.get('is_correct') is correct, prefix+'Reward was not obtained by the original math correctness evaluator')
            episode_id = episode.get('episode_id')
            logged = self.training_episodes.get(episode_id)
            if c.need(logged is not None, prefix+'This evaluated episode is not linked to a logged real training batch'):
                logged_path, training = logged
                matched_training_steps.add(training.get('training_step'))
                c.need(training.get('is_correct') is correct, f'{logged_path.name}: Training correctness differs from the original evaluator')
            policy_step = self.episode_policy_steps.get(episode_id)
            if policy_step is not None:
                groups[(policy_step, row_number)].append(float(correct))
        mixed_by_step = defaultdict(list)
        group_summary = []
        for (policy_step, row), rewards in sorted(groups.items()):
            mixed = len(set(rewards)) > 1
            if mixed:
                mixed_by_step[policy_step].append(row)
            group_summary.append(dict(policy_step=policy_step, source_row_index=row, samples=len(rewards),
                                      reward_counts=dict(Counter(rewards)), mixed=mixed))
        for policy_step in range(self.args.minimum_updates):
            c.need(bool(mixed_by_step[policy_step]), f'Policy step {policy_step}: No same-question group has genuine nonuniform rewards')
        c.need(len(matched_training_steps) >= self.args.minimum_updates, 'Fewer than two distinct real training batches were logged')
        c.need(bool(self.episodes), 'No original PythonMathEvaluator enriched trajectories exist')
        c.details.update(source_dataset=str(self.args.source_dataset), source_rows=len(source),
                         source_sha256=self.inventory[str(self.args.source_dataset.resolve())]['sha256'],
                         real_episodes=len(self.episodes), logged_training_steps=sorted(x for x in matched_training_steps if x is not None),
                         groups=group_summary,
                         reward_definition='Original final boxed-answer correctness only. No reward or tool-call injection.')


    def python_tools(self, c):
        self.load_grader_and_tool()
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.args.model_snapshot, local_files_only=True)
        sys.path.insert(0, str(self.args.project_root/'adapter'))
        parser_module = importlib.import_module('rtp_gateway_adapter')
        self.read_python_source(Path(parser_module.__file__))
        parse = parser_module.parse_generated_tool_calls
        generated_by_episode = defaultdict(list)
        final_contents_by_episode = defaultdict(list)
        checked_prompts = 0
        for segment in self.raw_segments:
            raw = self._tokenizer.decode(segment['output_token_ids'], skip_special_tokens=False)
            parsed = parse(raw, self._tool_module.TOOLS)
            c.need(parsed['content'] == segment['model_response'],
                   f'{segment["source_path"]}: Visible model response is not decoded from the actual native generated tokens')
            c.need(parsed['content'] == segment['agent_action'],
                   f'{segment["source_path"]}: Original agent action does not match the delivered native token response')
            generated_by_episode[segment['episode_id']].extend(parsed['tool_calls'])
            if not parsed['tool_calls']:
                final_contents_by_episode[segment['episode_id']].append(parsed['content'])
            history = segment['chat_completions']
            if c.need(bool(history) and history[-1].get('role') == 'assistant',
                      f'{segment["source_path"]}: Actual agent prompt history is absent'):
                prompt = self._tokenizer.apply_chat_template(
                    parser_module.normalize_tool_history(history[:-1]), tools=self._tool_module.TOOLS,
                    tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort="low", preserve_thinking=False)
                prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
                c.need(prompt_ids == segment['prompt_token_ids'],
                       f'{segment["source_path"]}: Actual native prompt tokens differ from the official full Python Tool chat template')
                checked_prompts += 1
        successful = []
        counts = Counter()
        for path, episode in self.episodes:
            artifacts = episode.get('artifacts') or {}
            final_content = artifacts.get('final_content', '')
            if final_content:
                c.need(final_content in final_contents_by_episode[episode.get('episode_id')],
                       f'{path.name}: Graded final answer was not emitted as actual native model text')
            calls = artifacts.get('python_tools', [])
            messages = [message for step in steps(episode) for message in step.get('chat_completions', [])]
            generated = generated_by_episode[episode.get('episode_id')]
            metrics = episode.get('metrics') or {}
            c.need(metrics.get('python_tool_calls') == len(calls), f'{path.name}: Python-call metric disagrees with actual recorded executions')
            c.need(metrics.get('python_tool_successes') == sum(bool(x.get('result', {}).get('ok')) for x in calls), f'{path.name}: Python-success metric disagrees with actual results')
            for call in calls:
                code, call_id = call.get('code'), call.get('tool_call_id')
                result = call.get('result') or {}
                prefix = f'{path.name} call {call_id}: '
                counts['recorded_calls'] += 1
                c.need(isinstance(code, str) and bool(code.strip()) and bool(call_id), prefix+'Empty code or call ID')
                c.need(any(x.get('name') == 'python' and x.get('arguments', {}).get('code') == code for x in generated), prefix+'Executed Python code is not present in actual generated tool-call tokens')
                linked = False
                for message in messages:
                    for tool in message.get('tool_calls', []):
                        function = tool.get('function') or {}
                        arguments = function.get('arguments') or {}
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except ValueError:
                                arguments = {}
                        if tool.get('id') == call_id and function.get('name') == 'python' and arguments.get('code') == code:
                            linked = True
                c.need(linked, prefix+'Executed call does not match the assistant tool-call ID/code')
                returns = [x for x in messages if x.get('role') == 'tool' and x.get('tool_call_id') == call_id]
                for returned in returns:
                    c.need(json.loads(returned.get('content', '{}')) == result, prefix+'Returned tool message differs from the actual execution result')
                if returns:
                    counts['calls_with_model_visible_return'] += 1
                c.need(type(result.get('ok')) is bool and isinstance(result.get('stdout'), str), prefix+'Malformed Python sandbox result')
                if result.get('ok'):
                    c.need(positive(result.get('elapsed_s')), prefix+'Successful sandbox execution has no positive elapsed time')
                    counts['successful_calls'] += 1
                    successful.append((path, code, result))
        c.need(counts['successful_calls'] > 0 and counts['calls_with_model_visible_return'] > 0, 'No successful real Python Tool execution with a result returned to the model')
        replays = []
        unique = set()
        for path, code, expected in successful:
            if code in unique:
                continue
            unique.add(code)
            actual = self._tool_module.execute_python(code)
            match = actual.get('ok') == expected.get('ok') and actual.get('stdout') == expected.get('stdout')
            c.need(match, f'{path.name}: Original sandbox replay does not reproduce the recorded generated Python result')
            replays.append(dict(episode=str(path), code_sha256=sha(code.encode()),
                                stdout_sha256=sha(str(actual.get('stdout', '')).encode()), reproduced=match))
            if len(replays) >= self.args.tool_replay_limit:
                break
        c.need(bool(replays), 'No real generated Python call could be replayed with the original sandbox')
        c.details.update(counts=dict(counts), independently_replayed_calls=replays,
                         official_template_prompts_checked=checked_prompts,
                         scope='Original execution artifacts + native generated XML/JSON call tokens + matching tool messages. CPU replays verify reproducible results; they are not GPU-training evidence.')


    def delivered_trace_linkage(self, c):
        """Bind actual agent messages to selected native payloads after retries."""
        source = self.args.project_root/'adapter/qwen36_trace_enrichment.py'
        self.read_python_source(source)
        spec = importlib.util.spec_from_file_location('qwen36_linkage_validation', source)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        records = self.records(self.root/'trace-linkage', 'trace-linkage-*.json')
        episodes = {episode.get('episode_id'): episode for _, episode in self.episodes}
        seen = Counter()
        observed_payloads = Counter()
        orphaned_count = selected_count = 0
        for path, record in records:
            uid = record.get('uid')
            seen[uid] += 1
            prefix = path.name+': '
            c.need(record.get('schema') == 'qwen36-delivered-trace-linkage-v1' and
                   record.get('passed') is True and record.get('strict') is True,
                   prefix+'Strict delivered-message trace association did not succeed')
            c.need(record.get('tokens_logprobs_and_agent_artifacts_modified') is False,
                   prefix+'Trace selection modified native payloads or original agent artifacts')
            episode = episodes.get(uid)
            if not c.need(episode is not None, prefix+'Trace audit is not linked to an actual training episode'):
                continue
            actual_steps = list(steps(episode))
            selected = record.get('selected', [])
            orphaned = record.get('orphaned', [])
            observed = record.get('observed_traces', [])
            c.need(record.get('agent_steps') == record.get('selected_count') == len(selected) == len(actual_steps),
                   prefix+'Selected traces do not cover every delivered agent step')
            c.need(record.get('orphaned_count') == len(orphaned), prefix+'Orphan count is inconsistent')
            identifiers = [item.get('trace_id') for item in observed]
            c.need(all(isinstance(value, str) and value for value in identifiers) and
                   len(set(identifiers)) == len(identifiers), prefix+'Observed trace identifiers are missing/duplicated')
            by_id = {item.get('trace_id'): item for item in observed}
            partition = [item.get('trace_id') for item in selected+orphaned]
            c.need(Counter(partition) == Counter(identifiers), prefix+'Observed traces are not exactly partitioned into delivered and retry-only calls')
            request_hashes = set()
            policy_step = self.episode_policy_steps.get(uid)
            for number, item in enumerate(selected):
                if number >= len(actual_steps):
                    break
                step = actual_steps[number]
                history = helper._history(step.get('chat_completions', []))
                c.need(bool(history) and history[-1]['role'] == 'assistant', prefix+'Delivered assistant history is missing')
                if not history:
                    continue
                c.need(item.get('agent_step') == number and item.get('session_id') == uid,
                       prefix+'Selected trace order/session differs from the delivered step')
                c.need(item.get('delivered_history_sha256') == sha(helper._json(history).encode()) and
                       item.get('request_history_sha256') == sha(helper._json(history[:-1]).encode()) and
                       item.get('assistant_sha256') == sha(helper._json(history[-1]).encode()),
                       prefix+'Selected trace does not represent the exact delivered message/history')
                prompt, response, values = step.get('prompt_ids', []), step.get('response_ids', []), step.get('logprobs', [])
                c.need((item.get('prompt_sha256'), item.get('response_sha256'), item.get('raw_logprobs_fp32_sha256')) ==
                       (digest_ids(prompt), digest_ids(response), sha(fp32_bytes(values))),
                       prefix+'Selected native trace payload differs from the actual trained step')
                c.need(item.get('prompt_tokens') == len(prompt) and item.get('response_tokens') == item.get('logprobs') == len(response),
                       prefix+'Selected trace token/score lengths differ from the trained step')
                c.need(item.get('tool_call_ids') == [call['id'] for call in history[-1].get('tool_calls', [])],
                       prefix+'Delivered tool-call identity differs from its selected trace')
                request_hashes.add(item.get('request_history_sha256'))
            for item in selected+orphaned:
                original = by_id.get(item.get('trace_id'), {})
                c.need(bool(original) and all(item.get(key) == value for key, value in original.items()),
                       prefix+'Selected/orphan trace summary differs from its observed original')
                c.need(item.get('session_id') == uid and item.get('weight_version') == policy_step,
                       prefix+'Observed trace session or policy version is inconsistent')
                c.need(isinstance(item.get('response_id'), str) and bool(item.get('response_id')),
                       prefix+'Observed response ID is missing')
                observed_payloads[(item.get('prompt_sha256'), item.get('response_sha256'),
                                   item.get('raw_logprobs_fp32_sha256'), policy_step)] += 1
            c.need(all(item.get('request_history_sha256') in request_hashes for item in orphaned),
                   prefix+'Discarded extra trace is not a retry of a delivered request')
            selected_count += len(selected)
            orphaned_count += len(orphaned)
        c.need(set(seen) == set(episodes) and all(count == 1 for count in seen.values()),
               'Exactly one successful trace-linkage audit is required for every actual training episode')
        native_payloads = Counter((item.get('prompt_sha256'), item.get('response_sha256'),
                                   item.get('raw_logprobs_fp32_sha256'), item.get('global_step'))
                                  for _, item in self.generation_receipts)
        c.need(observed_payloads == native_payloads,
               'Delivered plus retry-only traces do not account for the actual native generation receipts')
        c.details.update(episodes=len(seen), selected_agent_steps=selected_count,
                         retry_only_calls_excluded_from_training=orphaned_count,
                         native_generation_calls=sum(native_payloads.values()),
                         scope='Every native generation is accounted for. Only exact agent-delivered histories select training token/logprob payloads; retry-only responses are excluded without changing the agent or rewards.')


    def load_actual_records(self,c):
        self.episodes=self.records(self.root/'tool-episodes','enriched_*.json')
        for path,row in self.records(self.root/'episodes','episode_*.json',recursive=True):
            uid=row.get('episode_id')
            if uid:
                c.need(uid not in self.training_episodes,f'{path}: duplicate actual training episode ID')
                self.training_episodes[uid]=(path,row)
        for path in self.files(self.rtp,'rtp_generations_*.jsonl'):
            self.generation_receipts.extend((path,row) for row in self.read(path,jsonl=True))
        for path,row in self.records(self.rtp,'rtp_sync_rank0_step*.json'):
            step=row.get('global_step');c.need(step not in self.syncs,f'{path}: duplicate actual source transaction')
            self.syncs[step]=(path,row)
        c.need(bool(self.episodes) and bool(self.training_episodes) and bool(self.generation_receipts),'Actual evaluated, logged and native-generated episodes are all required')
        c.details.update(enriched_episodes=len(self.episodes),logged_training_episodes=len(self.training_episodes),native_calls=len(self.generation_receipts))
    def exact_world72_groups(self,c):
        payload=self.args.training_jsonl.read_bytes()
        c.need(sha(payload)==DATASET18_SHA256,'The frozen original eighteen-question input changed')
        chosen=[json.loads(line) for line in payload.splitlines() if line.strip()]
        source_ids={row['source_row_index'] for row in chosen};c.need(len(chosen)==len(source_ids)==18,'Expected eighteen distinct genuine source rows')
        groups=defaultdict(list);successes=Counter();seen=set();dp_routes=defaultdict(set)
        for path,episode in self.episodes:
            uid=episode.get('episode_id');c.need(uid not in seen,f'{path}: duplicate real evaluated episode');seen.add(uid)
            step=self.episode_policy_steps.get(uid);task=episode.get('task') or {};row=task.get('source_row_index')
            c.need(step in (0,1) and row in source_ids,f'{path}: evaluated episode is not one of the two actual18x4 iterations')
            groups[(step,row)].append(episode.get('reward'))
            artifacts=episode.get('artifacts') or {};c.need(artifacts.get('thinking_enabled') is True,f'{path}: official thinking was disabled')
            successes[step]+=sum(call.get('result',{}).get('ok') is True for call in artifacts.get('python_tools',[]))
        for _,receipt in self.generation_receipts:dp_routes[receipt.get('global_step')].add(receipt.get('dp_rank'))
        summary=[]
        for step in (0,1):
            c.need({row for (version,row) in groups if version==step}==source_ids,f'Policy{step}: missing/extra source-question group')
            for row in source_ids:c.need(len(groups[(step,row)])==4,f'Policy{step} source{row}: require exactly four actual trajectories')
            c.need(successes[step]>0,f'Policy{step}: no original Python Tool succeeded in an actual trajectory')
            c.need(dp_routes[step]==set(range(9)),f'Policy{step}: actual generation did not reach all9 DP endpoints')
            summary.append(dict(policy_step=step,actual_trajectories=sum(len(groups[(step,row)]) for row in source_ids),source_prompts=18,successful_python_calls=successes[step],actual_dp_endpoints=sorted(dp_routes[step])))
        c.details['iterations']=summary
    def execute(self):
        self.run_check('actual_original_records',self.load_actual_records)
        self.run_check('actual_native_tokens_and_FP32_logprobs',self.generation_logprobs)
        self.run_check('original_correctness_rewards_and_nondegenerate_groups',self.grounded_rewards)
        self.run_check('actual_generated_PythonTool_and_sandbox_replay',self.python_tools)
        self.run_check('exact_delivered_history_trace_linkage',self.delivered_trace_linkage)
        self.run_check('two_real_18_prompt_times4_iterations',self.exact_world72_groups)
        passed=not self.input_errors and all(check['passed'] for check in self.checks)
        return dict(schema='q38-world72-original-trajectory-validation-v1',trajectory_checks_passed=passed,full_RL_acceptance=False,checks=self.checks,input_errors=self.input_errors,inputs=self.inventory,remaining_mandatory_checks=['Actual72-rank BF16 optimizer/FP32 gradients, policy0/1/2 HF/native numerical and source commits','All72 GPU peaks, actual18 cgroup800GiB peaks/events, no local-disk state','Pinned deployed source/native identities and actual final owned job/process/context exit'])
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',type=Path,action='append',required=True)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--model-snapshot',type=Path,required=True)
    parser.add_argument('--source-dataset',type=Path,required=True)
    parser.add_argument('--training-jsonl',type=Path,required=True)
    parser.add_argument('--rllm-source',type=Path,required=True)
    parser.add_argument('--agent-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tool-replay-limit',type=int,default=5)
    args=parser.parse_args();args.minimum_updates=2;args.world_size=72
    if not 1<=args.tool_replay_limit<=72:parser.error('Require1..72 actual original sandbox replays')
    result=Validator(args).execute();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(trajectory_checks_passed=result['trajectory_checks_passed'],full_RL_acceptance=False,failed_checks=[check['name'] for check in result['checks'] if not check['passed']],output=str(args.output))))
    sys.exit(0 if result['trajectory_checks_passed'] else 2)
