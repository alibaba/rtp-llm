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
    from .mtp_contract import synthetic_fixture
else:
    import text_smoke as legacy
    from mtp_contract import synthetic_fixture

K3_REFERENCE = '0a0b31a72b7e831c2499d6f3c9e77b3df70a33b1'


def decode_batch_sizes(profile):
    if profile == 'smoke':
        return (1, 4, 8, 63)
    if profile == 'capacity':
        return (64,)
    raise ValueError(f'unknown Decode batch profile: {profile}')


def request_budget(profile):
    decode_batch_sizes(profile)  # Reject an unsupported contract.
    return (dict(total=304, formal=267, preparation=37) if profile == 'smoke'
            else dict(total=64, formal=64, preparation=0))


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

    @property
    def cache_probe_tokens(self):
        return self.reuse_unit_tokens + 1

    @property
    def cache_partial_minimum(self):
        return self.reuse_unit_tokens

    def http_process_environment(self):
        env = dict(os.environ)
        env.pop('LD_LIBRARY_PATH', None)
        env.pop('LD_PRELOAD', None)
        return env

    def prepare_case(self, case):
        return replace(case, name=case.name.replace('+', 'plus'), thinking_disabled=True)

    def prepare_http_payload(self, payload, case):
        payload.update(seed=3101, enable_thinking=False, top_p=1.0,
                       user="k31-smoke:" + case.name)
        payload.setdefault('extra_configs', {}).update(
            return_input_ids=True, return_output_ids=True, force_sp_accept=False)
        return payload

    def prepare_group_config(self, config, case):
        config.random_seed = 3101
        config.top_p = 1.0
        config.force_sp_accept = False
        return config

    def max_grouped_prefill_requests(self):
        return max(decode_batch_sizes(
            getattr(getattr(self, 'args', None), 'decode_batch_profile', 'smoke')))

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
        if debug.get('input_source') != 'group_rpc_wire' and ids != self.tokenize(case.prompt):
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
        # Keep the reference15 requests, using the observed K3.1 checkpoint
        # and bounded pool. Fresh seeds plus the probes must exceed Device,
        # while fitting Device+Host. Long-history allocation runs afterwards:
        # it may legitimately drop unrelated small-prefix checkpoints.
        blocks = self._orthogonal_device_blocks()
        if blocks != 95:
            raise legacy.SmokeFailure('service smoke pressure plan requires the audited95-page pool')
        # Fill more than the entire Device pool with newer, distinct prefixes.
        # Subtracting the seeds only demotes the oldest few candidates and
        # leaves the later witness/cancel prefixes resident on Device.
        pages = max(1, (blocks + 1 + 14) // 15)
        tokens = pages * self.reuse_unit_tokens + 1
        if tokens > 65536:
            raise legacy.SmokeFailure('bounded cache pressure exceeds64K current-Q limit')
        self._cache_pressure_requests = 15
        seeds = len(self._memory_candidates) + 1 + len(self._cancel_host_candidates)
        self._cache_pressure_plan = dict(seed_pages=seeds, pages_per_probe=pages,
                                        input_tokens=tokens, device_pages=blocks)
        for index in range(15):
            prompt, _ = self.fit_prompt(f'CACHE-PRESSURE:{self.args.namespace}:{index}\n',
                                       '\nReply 1.', tokens)
            self._required_stage(f'tier-pressure-{index:02d}', [legacy.Case(
                f'tier-pressure-{index:02d}', prompt, r'.', 'miss',
                max_tokens=8, preparation_only=True)])

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

    def _orthogonal_decode_batches(self):
        # BS63 exercises a full owner and a padded owner together. BS64 is an
        # explicit capacity-only suite, not a repeated default-smoke gate.
        owners = len(self.decode_role_addrs)
        profile = getattr(self.args, 'decode_batch_profile', 'smoke')
        for stage_index, size in enumerate(decode_batch_sizes(profile)):
            name = f'orthogonal_decode_{stage_index:02d}_batch_{size}'
            cases = [replace(
                self.record_case(f'{name}_{index:02d}', index % owners,
                                 words=1),
                max_tokens=64,
                require_mtp_draft=True, thinking_disabled=True)
                for index in range(size)]
            self._required_stage(name, cases, concurrent=size > 1, grouped_pd=size > 1)

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
                      self._orthogonal_chunk_kv, self._orthogonal_decode_batches):
            phase()

    def save(self, passed, error=None):
        formal = sum(row.get('phase') == 'formal' for row in self.records)
        preparation = sum(row.get('phase') == 'preparation' for row in self.records)
        profile = getattr(self.args, 'decode_batch_profile', 'smoke')
        budget = request_budget(profile)
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
            native_K31_MTP=False, synthetic_draft=self.args.synthetic_draft,
            group_tokenizer_setup_seconds=getattr(self, 'group_tokenizer_setup_seconds', None),
            cache_pressure_plan=getattr(self, '_cache_pressure_plan', None))
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        if requested_pass and not complete_count:
            raise legacy.SmokeFailure(error)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base-url', 'decode-health-url', 'namespace', 'source-commit'):
        p.add_argument('--'+key, required=True)
    for key in ('output', 'geometry', 'prefill-engine-log', 'prefill-rpc-runfiles',
                'checkpoint', 'synthetic-draft-checkpoint'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--prefill-grpc-port', required=True, type=int)
    p.add_argument('--decode-role-addr', action='append', dest='decode_role_addrs',
                   type=legacy.parse_decode_role_addr, required=True)
    p.add_argument('--decode-batch-profile', choices=('smoke', 'capacity'), default='smoke',
                   help='smoke: complete service suite with BS63; capacity: BS64 only')
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
    if args.output.exists() or (args.output.parent/'requests').exists():
        p.error('existing evidence cannot be overwritten')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.synthetic_draft = synthetic_fixture(args.synthetic_draft_checkpoint)
    args.block_size = g['page_size']
    args.reuse_unit_tokens = lcm(g['checkpoint_span'],g['cache_key_stride'])
    args.runtime_layers, args.profile = 93, 'k3-full'
    args.suite = 'main-text-64k-capped'
    args.run_kind = 'real93-synthetic-mtp-' + args.decode_batch_profile
    args.batch_size, args.chunk_tokens, args.max_tokens = 4, 65536, 128
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
        runner.run_k3_full()
        runner.save(True)
        return 0
    except BaseException as exc:
        runner.save(False, f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
