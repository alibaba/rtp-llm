#!/usr/bin/env python3
"""K3.1 service smoke with a separate Decode capacity check.

Reuse the target branch's shared request/modeling interfaces. Client results
require separate all-rank execution auditing before a complete smoke pass.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from math import lcm
from pathlib import Path
import time

if __package__:
    from . import text_smoke as legacy
    from .mtp_contract import synthetic_fixture, native_mla_draft
else:
    import text_smoke as legacy
    from mtp_contract import synthetic_fixture, native_mla_draft

K3_REFERENCE = '0a0b31a72b7e831c2499d6f3c9e77b3df70a33b1'
PREFILL_CHUNK_TOKENS = 65536
APPEND_SEED_TOKENS = 110 * 1024
CHUNK_CASE_STAGES = (
    'padding_tail_1_cold', 'padding_tail_1_hit',
    'padding_tail_7_cold', 'padding_tail_7_hit',
    'long_110k_seed', 'long_110k_append',
)


def decode_batch_sizes(profile, *, last5_geometry=False):
    if profile == 'smoke':
        return (1, 4, 8, 63) if last5_geometry else (1, 4, 8, 63, 64)
    if profile == 'capacity':
        return (64,)
    raise ValueError(f'unknown Decode batch profile: {profile}')


def request_budget(profile, pressure_requests=15, *, last5_geometry=False):
    batches = decode_batch_sizes(profile, last5_geometry=last5_geometry)
    if profile == 'capacity':
        return dict(total=64, formal=64, preparation=0)
    formal = 191 + sum(batches)
    if not last5_geometry:
        formal += len(CHUNK_CASE_STAGES)
    preparation = 22 + pressure_requests
    return dict(total=formal + preparation, formal=formal, preparation=preparation)


def cache_pressure_plan(device_pages, reuse_unit_tokens, *, scheduler_chunking=False):
    if type(device_pages) is not int or device_pages <= 0:
        raise ValueError('Prefill Device/full capacity must be positive')
    if type(reuse_unit_tokens) is not int or not 0 < reuse_unit_tokens < 65536:
        raise ValueError('Prefill checkpoint must fit in the 64K query limit')
    if scheduler_chunking:
        pages = (device_pages + 14) // 15
        return dict(device_pages=device_pages, pages_per_probe=pages,
                    input_tokens=pages * reuse_unit_tokens + 1, requests=15)
    # One suffix token lets the complete checkpoints be published as prefixes.
    pages = min((device_pages + 15) // 15, (65536 - 1) // reuse_unit_tokens)
    return dict(device_pages=device_pages, pages_per_probe=pages,
                input_tokens=pages * reuse_unit_tokens + 1,
                requests=max(15, (device_pages + pages) // pages))


def pending_host_load(path, request_id, *, timeout=5.0, clock=time.monotonic,
                      sleep=time.sleep):
    """Observe existing CPU events; never delay or synchronize the loader."""
    deadline = clock() + timeout
    observation = dict(request_id=request_id, load_started_ns=None,
                       load_done_ns=None, pending_observed=False)
    with path.open() as log:
        # The RPC has just been admitted. Include the recent tail in case
        # loading started before its admission response reached the client.
        log.seek(max(0, path.stat().st_size - 65536))
        incomplete = ''
        while True:
            lines = (incomplete + log.read()).splitlines(keepends=True)
            incomplete = lines.pop() if lines and not lines[-1].endswith('\n') else ''
            for line in lines:
                if '[K3_SMOKE_EVENT] ' not in line:
                    continue
                try:
                    event = json.loads(line.split('[K3_SMOKE_EVENT] ', 1)[1])
                except json.JSONDecodeError:
                    continue
                if event.get('request_id') != request_id:
                    continue
                if event.get('event') == 'host_cache_load_started':
                    observation['load_started_ns'] = event['time_ns']
                elif event.get('event') == 'host_cache_load_done':
                    observation['load_done_ns'] = event['time_ns']
            # Drain all currently visible events before deciding; a historical
            # start followed by done cannot prove that DMA remains pending.
            if observation['load_done_ns'] is not None:
                return observation
            if observation['load_started_ns'] is not None:
                observation['pending_observed'] = True
                return observation
            if clock() >= deadline:
                return observation
            sleep(0.001)


class K31FullRunner(legacy.Runner):
    def __init__(self, args):
        super().__init__(args)
        self.submitted_requests = 0

    def scheduler_chunk_profile(self):
        return getattr(getattr(self, 'args', None), 'last_n_checkpoints', None) == 5

    def prefill_case_tokens(self, label, index):
        if not self.scheduler_chunk_profile():
            return super().prefill_case_tokens(label, index)
        q, unit = self.args.chunk_tokens, self.reuse_unit_tokens
        return {('single', 0): q, ('batch', 0): 3 * q,
                ('batch', 1): 3 * q + 2 * unit + 1}[label, index]

    @property
    def cache_probe_tokens(self):
        return 12 * self.reuse_unit_tokens + 128 if self.scheduler_chunk_profile() else self.reuse_unit_tokens + 1

    @property
    def cache_partial_minimum(self):
        return self.reuse_unit_tokens

    def http_process_environment(self):
        env = dict(os.environ)
        env.pop('LD_LIBRARY_PATH', None)
        env.pop('LD_PRELOAD', None)
        return env

    def mtp_acceptance(self, aux, counters):
        # Use actual accepted draft counts instead of an iteration estimate.
        # The PD response carries D's counter; P's chunks are audited separately.
        if self.scheduler_chunk_profile() and (aux.get('speculative_metrics') or {}).get('version') == 1:
            return counters['accepted_draft_tokens'], 'native_speculative_metrics_v1'
        return super().mtp_acceptance(aux, counters)

    def prepare_case(self, case):
        case = replace(case, name=case.name.replace('+', 'plus'), thinking_disabled=True)
        if self.scheduler_chunk_profile() and case.reuse != 'miss':
            for pair in getattr(self, '_last_n_cache_pairs', {}).values():
                if case.prompt == pair['probe_prompt']:
                    return replace(case, expected_input_len=10*self.reuse_unit_tokens+128,
                                   expected_reuse_len=10*self.reuse_unit_tokens)
        return case

    def cache_seed_prompt(self, tag, value):
        if not self.scheduler_chunk_profile():
            return super().cache_seed_prompt(tag, value)
        if value == 'MEM-1' and getattr(self, '_memory_candidates', None):
            return self._memory_candidates[0][1].prompt
        head, tail = f'CACHE:{tag}:memory\n', f'\n只输出 JSON {{"value":"{value}"}}。'
        shared, target = 10*self.reuse_unit_tokens, 10*self.reuse_unit_tokens+128
        seed, seed_ids, probe, probe_ids = self.fit_linear_checkpoint_branch(
            head, tail, tail, self.cache_probe_tokens, target, shared)
        pairs = getattr(self, '_last_n_cache_pairs', {})
        pairs[seed] = dict(tag=tag, probe_prompt=probe, common_tokens=shared,
            seed_tokens=len(seed_ids), probe_tokens=len(probe_ids),
            seed_token_sha256=hashlib.sha256(json.dumps(seed_ids).encode()).hexdigest(),
            probe_token_sha256=hashlib.sha256(json.dumps(probe_ids).encode()).hexdigest())
        self._last_n_cache_pairs = pairs
        return seed

    def fit_linear_checkpoint_branch(self, head, seed_tail, probe_tail, seed_target, probe_target, shared):
        seed, seed_ids = self.fit_prompt(head, seed_tail, seed_target)
        x_count = shared - len(self.tokenize(head+probe_tail))
        z_count = probe_target-shared
        for _ in range(4):
            if min(x_count, z_count) <= 0:
                break
            probe = head + ' x'*x_count + ' z'*z_count + probe_tail
            probe_ids = self.tokenize(probe)
            common = next((i for i, pair in enumerate(zip(seed_ids, probe_ids)) if pair[0] != pair[1]),
                          min(len(seed_ids), len(probe_ids)))
            if common == shared and len(probe_ids) == probe_target:
                self.save_token_fixture(probe, probe_ids, enable_thinking=False)
                return seed, seed_ids, probe, probe_ids
            delta = shared-common
            x_count += delta
            z_count += probe_target-len(probe_ids)-delta
        raise legacy.SmokeFailure(f'cannot construct exact {shared}-token LINEAR branch')

    def cache_probe_case(self, seed):
        if not self.scheduler_chunk_profile():
            return seed
        pair = self._last_n_cache_pairs[seed.prompt]
        return replace(seed, prompt=pair['probe_prompt'], expected_input_len=pair['probe_tokens'],
                       expected_reuse_len=pair['common_tokens'])

    def _precision_reference(self, case):
        if not self._memory_candidates:
            raise legacy.SmokeFailure('precision reference needs the C10 seed fixture')
        probe = self.cache_probe_case(self._memory_candidates[0][1])
        return replace(probe, name=case.name, reuse='miss', expected_reuse_len=0,
                       preparation_only=case.preparation_only)

    def cache_seed_case(self, case, attempt):
        if self.scheduler_chunk_profile() and attempt == 1:
            return self._precision_reference(case)
        return case

    def cache_witness_case(self, case):
        return self._precision_reference(case) if self.scheduler_chunk_profile() else case

    def _orthogonal_cache_seed(self):
        super()._orthogonal_cache_seed()
        if self.scheduler_chunk_profile():
            # Keep the original seed request slots; first-attempt tier proof
            # is mandatory for the fixed 304-request acceptance contract.
            self._memory_candidates = self._memory_candidates[:1]

    def probe_cache_witness(self):
        if not self.scheduler_chunk_profile():
            return super().probe_cache_witness()
        # The existing witness probe runs after the existing long-KV traffic,
        # which supplies real HOST pressure and SSD demotion without more RPCs.

    def complete_last_n_ssd_probe(self):
        probe = replace(self._memory_witness, name='tier-memory-probe', reuse='hit',
            expected_cache_tier=None, preparation_only=True,
            expected_reuse_len=10*self.reuse_unit_tokens, max_tokens=64)
        self._required_stage(probe.name, [probe])
        tier = self.records[-1].get('prefill_cache_tiers') or {}
        if tier.get('disk', 0) <= 0:
            raise legacy.SmokeFailure('C10 SSD probe did not actually read back from disk')
        self.stages[-1].update(expected_linear_tier='DISK',
                               expected_linear_checkpoint_tokens=10*self.reuse_unit_tokens)

    def precision_case(self, case):
        return self.scheduler_chunk_profile() and case.name in {
            'tier-device-probe-0', 'tier-memory-seed-1', 'tier-witness-seed',
            'tier-mixed-0-memory', 'tier-memory-probe'}

    def precision_cold_case(self, case):
        return self.scheduler_chunk_profile() and case.name in {'tier-memory-seed-1', 'tier-witness-seed'}

    def run_prefix_branches(self):
        if not self.scheduler_chunk_profile():
            return super().run_prefix_branches()
        a, b = self.linear_c7_cases()
        unit = self.reuse_unit_tokens
        self.run_stage('prefix_A_seed', [a])
        self.run_stage('prefix_B_partial', [b])
        a_hit = replace(a, name='prefix-A-return', reuse='hit', expected_reuse_len=12*unit)
        b_hit = replace(b, name='prefix-B-hit', reuse='hit', expected_reuse_len=12*unit)
        self.run_stage('prefix_A_return', [a_hit])
        self.run_stage('prefix_AB_mixed', [replace(case, name=f'prefix-mixed-{i}',
            decode_owner_rank=(i+4) % len(self.decode_role_addrs))
            for i, case in enumerate((a_hit, b_hit, a_hit, b_hit))], concurrent=True)

    def linear_c7_cases(self):
        head = f'档案 {self.args.namespace}/prefix-branches。以下 x 为无关填充。\n'
        tail_a = '\n唯一有效记录：key=chosen; value=CEDAR-AMBER。只输出 JSON {"value":"CEDAR-AMBER"}，不要解释。'
        tail_b = '\n唯一有效记录：key=chosen; value=MAPLE-VIOLET。只输出 JSON {"value":"MAPLE-VIOLET"}，不要解释。'
        unit, target = self.reuse_unit_tokens, 12*self.reuse_unit_tokens+128
        prompt_a, tokens_a, prompt_b, tokens_b = self.fit_linear_checkpoint_branch(
            head, tail_a, tail_b, target, target, 7*unit)
        self._last_n_c7_fixture = dict(common_tokens=7*unit, input_tokens=target,
            seed_token_sha256=hashlib.sha256(json.dumps(tokens_a).encode()).hexdigest(),
            probe_token_sha256=hashlib.sha256(json.dumps(tokens_b).encode()).hexdigest())
        a = legacy.Case('prefix-A-seed', prompt_a, '', 'miss', expected_json={'value':'CEDAR-AMBER'},
                        expected_input_len=len(tokens_a), expected_reuse_len=0)
        b = legacy.Case('prefix-B-partial', prompt_b, '', 'miss', expected_json={'value':'MAPLE-VIOLET'},
                        expected_input_len=len(tokens_b), expected_reuse_len=0)
        return a, b

    def prepare_http_payload(self, payload, case):
        payload.update(seed=3101, enable_thinking=False, top_p=1.0,
                       user="k31-smoke:" + case.name)
        payload.setdefault('extra_configs', {}).update(
            return_input_ids=True, return_output_ids=True, force_sp_accept=False)
        if self.precision_case(case):
            payload['extra_configs']['trace_id'] = 'k31-last-n-precision:' + case.name
        if self.precision_cold_case(case):
            payload['extra_configs']['reuse_cache'] = False
        return payload

    def prepare_group_config(self, config, case):
        config.random_seed = 3101
        config.top_p = 1.0
        config.force_sp_accept = False
        if self.precision_case(case):
            config.trace_id = 'k31-last-n-precision:' + case.name
        if self.precision_cold_case(case):
            config.reuse_cache = False
        return config

    def max_grouped_prefill_requests(self):
        return max(decode_batch_sizes(
            getattr(getattr(self, 'args', None), 'decode_batch_profile', 'smoke'),
            last5_geometry=self.scheduler_chunk_profile()))

    def _request_cases_group(self, cases, *, decode_relay_ports=None):
        if len(cases) < 63 or decode_relay_ports is not None:
            return super()._request_cases_group(cases, decode_relay_ports=decode_relay_ports)
        if __package__:
            from .pd_group_relay import PdGroupRelay
        else:
            from pd_group_relay import PdGroupRelay
        endpoints = [f"{role['ip']}:{role['grpc_port']}" for role in self.decode_role_addrs]
        artifact = self.args.output.parent / f'pd-group-relay-{time.time_ns()}.jsonl'
        with PdGroupRelay(endpoints, len(cases), artifact) as relay:
            return super()._request_cases_group(cases, decode_relay_ports=relay.ports)

    def record_case(self, name, owner, *, words=1):
        # Batch transitions test membership and state, not long copying.
        structural_batch = name.startswith('batch-shape-')
        case = super().record_case(name, owner, words=1 if structural_batch else words)
        return replace(case, max_tokens=64) if structural_batch else case

    def group_tokenizer(self, factory):
        # Rebuilding the163K-vocabulary tokenizer for every grouped stage is
        # test setup, not inference. Reuse one pinned instance per runner;
        # actual wire tokens continue to be checked against serving tokens.
        path = self.args.long_prefix_checkpoint
        cached = getattr(self, '_group_tokenizer', None)
        if cached is None:
            started = time.monotonic()
            tokenizer = factory.from_pretrained(
                path, trust_remote_code=True, local_files_only=True)
            self._group_tokenizer = (path, tokenizer)
            self.group_tokenizer_setup_seconds = time.monotonic() - started
        elif cached[0] != path:
            raise legacy.SmokeFailure('grouped tokenizer checkpoint changed within smoke')
        return self._group_tokenizer[1]

    def tokenize(self, prompt, *, enable_thinking=None):
        return super().tokenize(prompt, enable_thinking=False)

    def validate(self, case, response, elapsed_s, request_max_tokens, *, verified_input_len=None):
        row = super().validate(self.prepare_case(case), response, elapsed_s,
                               request_max_tokens, verified_input_len=verified_input_len)
        debug = response.get('debug_info') or {}
        ids = debug.get('input_ids')
        if (not isinstance(ids, list) or len(ids) != row['input_len']
                or any(type(i) is not int for i in ids)):
            raise legacy.SmokeFailure(f'{case.name}: actual input token IDs missing')
        if debug.get('input_source') != 'group_rpc_wire':
            expected_ids = (self.tokenize(case.prompt) if case.messages is None else
                            self.tokenize_messages(case.messages, enable_thinking=False))
            if ids != expected_ids:
                raise legacy.SmokeFailure(f'{case.name}: actual input differs from tokenizer fixture')
        row.update(input_tokens=ids,
                   input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                   input_tokens_verified=True, target_only=False,
                   language_acceptance=not case.preparation_only,
                   owner_execution_verified=False)
        if not case.preparation_only and any(marker in row['content'] for marker in
                ('<|im_start|>', '<|im_end|>', '<|end_of_msg|>')):
            raise legacy.SmokeFailure(f'{case.name}: leaked special marker')
        return row

    def run_stage(self, name, cases, *args, **kwargs):
        if (name.startswith('batch_shape_') or name.startswith('cache_block_')
                or name in ('batch_all_miss', 'batch_all_hit', 'batch_mixed_hit_miss',
                            'batch_mixed_then_all_hit', 'cuda_graph_bucket_8', 'prefix_AB_mixed')):
            kwargs['grouped_pd'] = True
        self.submitted_requests += len(cases)
        super().run_stage(name, cases, *args, **kwargs)
        if self.stages[-1].get('skipped') or not self.stages[-1]['passed']:
            raise legacy.SmokeFailure(f'{name}: unfinished smoke stage')

    def cache_boundary_tokens(self):
        # K3's eight physical-page edges covered one checkpoint span. The
        # two-checkpoint edge coincided with its deferred64K+1 input. Preserve
        # those same eight families using K3.1's observed page/checkpoint ratio.
        page, unit = self.args.block_size, self.reuse_unit_tokens
        if unit // page != 8 or unit % page:
            raise legacy.SmokeFailure('service smoke needs the eight-page checkpoint geometry')
        if self.scheduler_chunk_profile():
            return (page, 2 * page, 4 * page, unit, 2 * unit, 4 * unit, 5 * unit, 6 * unit)
        return tuple(range(page, unit + 1, page))

    def decode_boundary_tokens(self):
        return (*self.cache_boundary_tokens(), self.args.chunk_tokens)

    def prefix_branch_target_tokens(self):
        # FIFO charges the full logical prefix, not only the uncached Q, and
        # requires a strictly sub-budget rectangle. Keep all four A/B members
        # in one forward under this suite's observed64K admission budget.
        page, unit = self.args.block_size, self.reuse_unit_tokens
        cap = ((65536 - 1) // 4 // page) * page
        target = min(super().prefix_branch_target_tokens(), cap)
        if target <= unit:
            raise legacy.SmokeFailure('prefix quartet cannot fit a reusable checkpoint and suffix')
        return target

    def _orthogonal_bounded_cache_pressure(self):
        # Demote the witness and all cancellation seeds by filling the observed
        # Device pool with newer prefixes. Larger pools need more preparations.
        plan = cache_pressure_plan(self._orthogonal_device_blocks(), self.reuse_unit_tokens,
                                   scheduler_chunking=self.scheduler_chunk_profile())
        self._cache_pressure_plan = plan
        self._cache_pressure_requests = plan['requests']
        for index in range(plan['requests']):
            prompt, _ = self.fit_prompt(f'CACHE-PRESSURE:{self.args.namespace}:{index}\n',
                                       '\nReply 1.', plan['input_tokens'])
            self._required_stage(f'tier-pressure-{index:02d}', [legacy.Case(
                f'tier-pressure-{index:02d}', prompt, r'.', 'miss',
                max_tokens=8, preparation_only=True,
                allow_long_history=plan['input_tokens'] > 65536,
                expected_input_len=plan['input_tokens'])])

    def validate_prefill_capacity(self):
        largest_batch = self.max_grouped_prefill_requests()
        # These short grouped requests each use one Prefill state slot.
        # Pure PD Prefill reserves one target token; proposal slots live on Decode.
        linear_required = largest_batch + 4
        linear = self._orthogonal_device_blocks('linear')
        full = self._orthogonal_device_blocks()
        full_required = largest_batch
        if self.args.decode_batch_profile == 'smoke':
            if self.scheduler_chunk_profile():
                full_required = max(full_required, 1048576 // self.reuse_unit_tokens + 1)
                # The existing cold reference disables reuse. Its short
                # 10I+128 prompt uses the compatibility allocation path.
                reference_slots = (10 * self.reuse_unit_tokens + 128 + self.args.block_size - 1) // self.args.block_size
                linear_required = max(linear_required, reference_slots + 4)
            else:
                full_required = max(full_required, 655360 // self.reuse_unit_tokens + 2)
                linear_required = max(linear_required, full_required)
        if linear < linear_required or full < full_required:
            raise legacy.SmokeFailure(
                f'Prefill cache capacity is too small before smoke admission: '
                f'LINEAR {linear}/{linear_required}, FULL {full}/{full_required} '
                f'(usable/required). Increase the Prefill KV budget while reserving activation memory.')

    def _seed_cancel_candidates(self):
        # Keep three reference seeds. A real64K cached prefix makes Host DMA
        # observable without injecting sleeps or synchronization into runtime.
        seeds = []
        for attempt in range(3):
            value = f'CANCEL-RECOVER-{attempt}'
            prompt, _ = self.fit_prompt(
                f'CACHE:{self.args.namespace}:cancel:{attempt}\n',
                f'\n只输出 JSON {{"value":"{value}"}}。', 65536,
                enable_thinking=False)
            case = self._orthogonal_answer(
                f'cancel-memory-seed-{attempt}', prompt, value, 'miss')
            self._required_stage(case.name, [case])
            seeds.append(case)
        return seeds

    def before_pending_cancel(self, request_id):
        return pending_host_load(self.args.prefill_engine_log, request_id)

    def _cancel_pending_request(self, seed, attempt):
        result = super()._cancel_pending_request(seed, attempt)
        result['cancel_accepted'] = (
            result['cancel_accepted']
            and result['before_cancel_observation']['pending_observed'])
        # Actual pending-load confirmation still comes from the native event,
        # not from absence of a load_done line in this client observation.
        return result

    def _orthogonal_page_boundaries(self):
        # K3 used physical-page/checkpoint +/-1 (six cold requests). Substitute
        # observed K3.1 geometry; do not add repeat/cross-owner permutations.
        owners = len(self.decode_role_addrs)
        g = self.args.geometry_value
        for boundary in dict.fromkeys((g['page_size'], g['checkpoint_span'])):
            for delta in (-1, 0, 1):
                value = f'PAGE-{boundary}-{delta:+d}'
                prompt, ids = self.fit_prompt(f'PAGE:{self.args.namespace}:{value}\n',
                    f'\n只输出 JSON {{"value":"{value}"}}。', boundary + delta)
                case = self._orthogonal_answer(f'orthogonal_page_{boundary}_{delta:+d}', prompt,
                    value, 'miss', expected_input_len=len(ids), decode_owner_rank=(delta + 1) % owners)
                self._required_stage(case.name, [case])

    def long_kv_seed_targets(self):
        targets = super().long_kv_seed_targets()
        if not self.scheduler_chunk_profile():
            return targets
        # Keep all twenty model request slots. One becomes the cold 1M case;
        # another retains C576 for the existing historical-KV final probe.
        targets = [1_048_576, *targets[:-1]]
        targets[targets.index(589_344)] = 589_824 + 128
        return targets

    def long_kv_seed_reuse(self, index):
        if self.scheduler_chunk_profile() and index <= 1:
            # Last-5 at 1M cannot satisfy the subsequent shorter prefix.
            return 'miss'
        return super().long_kv_seed_reuse(index)

    def long_kv_seed_prompt(self, base, overhead, target):
        if not self.scheduler_chunk_profile():
            return super().long_kv_seed_prompt(base, overhead, target)
        repeats = target - overhead
        for _ in range(3):
            prompt = base + ' x' * repeats
            tokens = self.tokenize(prompt, enable_thinking=False)
            if len(tokens) == target:
                self.save_token_fixture(prompt, tokens, enable_thinking=False)
                return prompt
            repeats += target - len(tokens)
        raise legacy.SmokeFailure(f'cannot construct exact long-KV seed length {target}')

    def _orthogonal_decode_batches(self):
        # BS63 exercises asymmetric owners and BS64 fills both owners.
        owners = len(self.decode_role_addrs)
        profile = getattr(self.args, 'decode_batch_profile', 'smoke')
        for stage_index, size in enumerate(decode_batch_sizes(
                profile, last5_geometry=self.scheduler_chunk_profile())):
            name = f'orthogonal_decode_{stage_index:02d}_batch_{size}'
            # Keep early requests decoding while the rest of the cohort joins.
            large_batch = size >= 63
            cases = [replace(
                self.record_case(f'{name}_{index:02d}', index % owners,
                                 words=8 if large_batch else 1),
                max_tokens=160 if large_batch else 64,
                require_mtp_draft=True, thinking_disabled=True)
                for index in range(size)]
            self._required_stage(name, cases, concurrent=size > 1, grouped_pd=size > 1)

    def run_110k_append(self):
        if __package__:
            from .k31_long_prefix_smoke import common_prefix
        else:
            from k31_long_prefix_smoke import common_prefix

        seed_name, append_name = CHUNK_CASE_STAGES[-2:]
        seed, seed_ids = self.fit_prompt(
            f'Archive {self.args.namespace}/110k.\nkey=archive; value=AMBER-110K\n',
            '\n只输出 JSON {"value":"READY-110K"}。',
            APPEND_SEED_TOKENS, enable_thinking=False,
        )
        self._required_stage(seed_name, [self._orthogonal_answer(
            seed_name, seed, 'READY-110K', 'miss', expected_input_len=len(seed_ids),
            expected_reuse_len=0, allow_long_history=True, scheduler_chunked=True,
        )])
        question = '检索 key=archive 的 value。只输出 JSON 对象，键为 value。'
        messages = [
            {'role': 'user', 'content': seed},
            {'role': 'assistant', 'content': self.records[-1]['content']},
            {'role': 'user', 'content': question},
        ]
        tokens = self.tokenize_messages(messages, enable_thinking=False)
        shared = min(common_prefix(seed_ids, tokens), len(seed_ids) - 1)
        reuse = shared // self.reuse_unit_tokens * self.reuse_unit_tokens
        if reuse <= self.args.chunk_tokens or len(tokens) <= len(seed_ids):
            raise legacy.SmokeFailure('110K append did not retain a cross-chunk prefix')
        self._required_stage(append_name, [self._orthogonal_answer(
            append_name, question, 'AMBER-110K', 'hit', messages=messages,
            expected_input_len=len(tokens), expected_reuse_len=reuse,
            allow_long_history=True, scheduler_chunked=True,
            decode_owner_rank=len(self.decode_role_addrs) - 1,
        )])

    def run_k3_full(self):
        if getattr(self.args, 'decode_batch_profile', 'smoke') == 'capacity':
            self._orthogonal_decode_batches()
            return
        # Preserve all reference requests. Run cache/cancel before long history
        # so their small witnesses are not coupled to a near-capacity80-page
        # recurrent-state allocation with different K3.1 checkpoint geometry.
        self.run_main_text()
        self._orthogonal_cache_seed()
        for phase in (self._orthogonal_page_boundaries, self._orthogonal_bounded_cache_pressure,
                      self._orthogonal_cache, self._orthogonal_cancel_recovery,
                      self._orthogonal_chunk_kv):
            phase()
        if self.scheduler_chunk_profile():
            self.complete_last_n_ssd_probe()
        self._orthogonal_decode_batches()
        if not self.scheduler_chunk_profile():
            self.run_padding_boundaries(scheduler_chunked=True)
            self.run_110k_append()

    def save(self, passed, error=None):
        formal = sum(row.get('phase') == 'formal' for row in self.records)
        preparation = sum(row.get('phase') == 'preparation' for row in self.records)
        profile = getattr(self.args, 'decode_batch_profile', 'smoke')
        budget = request_budget(profile, getattr(self, '_cache_pressure_requests', 15),
                                last5_geometry=self.scheduler_chunk_profile())
        complete_count = formal == budget['formal'] and preparation == budget['preparation']
        requested_pass = passed
        if passed and not complete_count:
            passed, error = False, f'full smoke count mismatch: formal={formal}, preparation={preparation}'
        super().save(passed, error)
        path = self.args.output
        value = json.loads(path.read_text())
        value.update(source_commit=self.args.source_commit, runtime_layers=93,
            geometry=self.args.geometry_value, target_only=False,
            legacy_reference_commit=K3_REFERENCE, complete_smoke_passed=False,
            execution_audit_pending=True, golden_alignment='pending',
            validation_profile=profile, request_budget=budget,
            legacy_reference_request_budget=dict(total=368, formal=331, preparation=37),
            actual_request_counts=dict(total=len(self.records), formal=formal, preparation=preparation),
            full_reference_count_matches=complete_count,
            cancellation_RPC_attempts_are_separate=True,
            native_K31_MTP=getattr(self.args, "draft_contract", None) is not None,
            draft_contract=getattr(self.args, "draft_contract", None), synthetic_draft=self.args.synthetic_draft,
            group_tokenizer_setup_seconds=getattr(self, 'group_tokenizer_setup_seconds', None),
            cache_pressure_plan=getattr(self, '_cache_pressure_plan', None))
        value['last_n_checkpoints'] = getattr(self.args, 'last_n_checkpoints', None)
        value['scheduler_chunk_profile'] = self.scheduler_chunk_profile()
        value['last_n_cache_fixtures'] = [{key: item for key, item in pair.items() if key != 'probe_prompt'}
            for pair in getattr(self, '_last_n_cache_pairs', {}).values()]
        value['last_n_c7_fixture'] = getattr(self, '_last_n_c7_fixture', None)
        if self.scheduler_chunk_profile():
            value['validation_profile'] = 'last5-chunked-' + profile
        elif profile == 'smoke':
            value['excluded_scenarios'] = []
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        if requested_pass and not complete_count:
            raise legacy.SmokeFailure(error)


def parse_args(argv=None, *, calibration=False):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base-url', 'decode-health-url', 'namespace', 'source-commit'):
        p.add_argument('--'+key, required=True)
    for key in ('output', 'geometry', 'prefill-engine-log', 'prefill-rpc-runfiles',
                'checkpoint'):
        p.add_argument('--'+key, required=True, type=Path)
    draft = p.add_mutually_exclusive_group(required=True)
    draft.add_argument('--draft-checkpoint', type=Path,
                       help='Trained native K3 MLA MTP checkpoint')
    draft.add_argument('--synthetic-draft-checkpoint', type=Path,
                       help='Historical fixture metadata only; no runtime fallback')
    p.add_argument('--prefill-grpc-port', required=True, type=int)
    p.add_argument('--decode-role-addr', action='append', dest='decode_role_addrs',
                   type=legacy.parse_decode_role_addr, required=True)
    p.add_argument('--decode-batch-profile', choices=('smoke', 'capacity'), default='smoke',
                   help='smoke: complete service suite; capacity: BS64 only')
    p.add_argument('--last-n-checkpoints', type=int, choices=(5,),
                   help='Exercise Last-5 with genuine scheduler chunks in the existing 304 request slots')
    args = p.parse_args(argv)
    args.geometry_value = json.loads(args.geometry.read_text())
    g = args.geometry_value
    keys = ('page_size','kernel_page_size','source_shards','checkpoint_span','cache_key_stride')
    if (g.get('runtime_verified') is not True or g.get('runtime_layers') != 93
            or g.get('source_shards') != 8
            or any(type(g.get(k)) is not int or g[k] <= 0 for k in keys)):
        p.error('full93 requires audited runtime page/checkpoint geometry and eight source shards')
    if g['page_size'] % g['kernel_page_size']:
        p.error('physical pages must contain complete kernel pages')
    if args.last_n_checkpoints is not None and (
            g['page_size'] not in (64, 128) or g['kernel_page_size'] != g['page_size']
            or g['checkpoint_span'] != g['page_size'] * 8 or g['cache_key_stride'] != g['checkpoint_span']
            or g.get('reuse_cache_tail_blocks') != (2147483647 if calibration else 5)
            or g.get('last_n_calibration', False) != calibration):
        p.error('Last-5 requires matching physical/kernel B64 or B128, CP8 and the requested runtime tail policy')
    if args.output.exists() or (args.output.parent/'requests').exists():
        p.error('existing evidence cannot be overwritten')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.draft_contract = (native_mla_draft(args.draft_checkpoint,
        json.loads((args.checkpoint/'config.json').read_text()))
        if args.draft_checkpoint is not None else None)
    args.synthetic_draft = (synthetic_fixture(args.synthetic_draft_checkpoint)
        if args.synthetic_draft_checkpoint is not None else None)
    args.block_size = g['page_size']
    args.reuse_unit_tokens = lcm(g['checkpoint_span'],g['cache_key_stride'])
    args.runtime_layers, args.profile = 93, 'k3-full'
    args.suite = 'main-text-64k-capped'
    if args.last_n_checkpoints is not None:
        args.suite = 'main-text-last5-chunked'
    args.run_kind = ('real93-native-mla-mtp-' if args.draft_contract is not None
                     else 'real93-synthetic-mtp-') + args.decode_batch_profile
    args.batch_size, args.chunk_tokens, args.max_tokens = 4, PREFILL_CHUNK_TOKENS, 128
    args.identity_max_tokens, args.single_exact_max_tokens, args.mtp_chunk_max_tokens = 256,128,128
    args.require_mtp, args.capture_rpc_frames = True,True
    args.timeout, args.case_deadline_s = 300,300
    args.rdma_prewarm_attempts, args.rdma_prewarm_timeout = 0,300
    args.rdma_prewarm_backoff_s, args.rdma_prewarm_settle_s = 0,0
    args.orthogonal_phases = ('cache','cancel','page','chunk','decode')
    args.long_prefix_checkpoint = args.checkpoint
    return args


def main(argv=None):
    args = parse_args(argv)
    runner = K31FullRunner(args)
    try:
        runner.validate_prefill_capacity()
        runner.run_k3_full()
        runner.save(True)
        return 0
    except BaseException as exc:
        runner.save(False, f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
