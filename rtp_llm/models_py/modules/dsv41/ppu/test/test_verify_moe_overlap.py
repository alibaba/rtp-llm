"""Device DAG contracts using local analytic callbacks, NOT native MegaMoE.

These tests isolate fork/join, live buffers, telemetry and lifetimes. They do
not establish WORLD8 communication progress, MXFP4 math or overlap speedup.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.verify_moe_overlap import (
    VerifyMoEOverlap,
    VerifyMoEOverlapPlans,
)


class _LocalEngine:
    backend, precision, ep_size = "megamoe", "w4a4", 8
    hidden_size, inter_dim, num_routed_experts = 5120, 2304, 384

    def __init__(self, device):
        self._mega_device, self._mega_buffer = device, object()
        self._mega_capacity, self._mega_generation = 128, 1
        self.bias = torch.tensor(0.125, device=device)
        self._processed = {(layer, device): (self.bias,) for layer in range(40)}
        self._mega_recv_stats = torch.full((48,), 11, dtype=torch.int32, device=device)
        self.gpu_calls = torch.zeros((), dtype=torch.int64, device=device)
        self.streams, self.fail = [], False

    def forward_shared_graph(self, flat, ids, weights, *, layer_id, expert_load, fused_expert_load):
        assert not fused_expert_load
        self.streams.append(int(torch.cuda.current_stream().cuda_stream))
        self.gpu_calls.add_(1)
        increments = torch.zeros_like(self._mega_recv_stats)
        increments.scatter_add_(0, (ids.clamp_min(0) % 48).flatten(), (ids >= 0).int().flatten())
        before = self._mega_recv_stats.clone()
        self._mega_recv_stats.add_(increments)
        if expert_load is not None:
            expert_load.add_(self._mega_recv_stats - before)
        output = (flat.float() * weights.sum(-1, keepdim=True) + self.bias).bfloat16()
        if self.fail:
            raise RuntimeError("local routed failure")
        return output


class _LocalGate:
    def __init__(self, state, device):
        self.state = state
        self.bias = torch.tensor(3, dtype=torch.int64, device=device)
        self.streams, self.fail = [], False

    def __call__(self, flat, *, graph_routes, allow_top3_graph_routes):
        assert not allow_top3_graph_routes
        valid, rows = graph_routes
        self.streams.append(int(torch.cuda.current_stream().cuda_stream))
        positions = self.state.metadata.positions
        ids = (positions[:, None] + self.bias + torch.arange(6, device=flat.device)[None]) % 384
        ids = torch.where(valid[:, None], ids, -1)
        weights = (flat[:, :6].float().abs() + 1) / 16
        weights = torch.where(valid[:, None], weights, 0)
        rows.copy_((ids >= 0).any(-1).sum())
        if self.fail:
            raise RuntimeError("local gate failure")
        return ids, weights


class _LocalShared:
    def __init__(self, device):
        self.scale = torch.tensor(0.75, device=device)
        self.shift = torch.tensor(-0.03125, device=device)
        self.streams, self.options = [], []
        self.fail = False

    def __call__(self, flat, **kwargs):
        self.streams.append(int(torch.cuda.current_stream().cuda_stream))
        self.options.append(kwargs)
        output = (flat.float() * self.scale + self.shift).bfloat16()
        if self.fail:
            raise RuntimeError("local shared failure")
        return output


def _fixture(device):
    engine = _LocalEngine(device)
    state = SimpleNamespace(
        is_graph=True, verify=True, prefill=False, proposal=False,
        grouped_moe=None, packed_attention=object(), bulk_counters=False,
        verify_moe_overlap_allowed=True,
        metadata=SimpleNamespace(
            batch_size=8, q_len_per_req=4,
            positions=torch.arange(32, dtype=torch.int64, device=device),
            valid_mask=torch.arange(32, device=device) % 3 != 0,
        ),
        moe_calls=torch.full((40,), 5, dtype=torch.int64, device=device),
        routed_rows_device=torch.full((40,), -17, dtype=torch.int64, device=device),
    )
    owner = SimpleNamespace(
        layer_id=7, routed_enabled=True, ops=SimpleNamespace(moe_engine=engine),
        runtime=SimpleNamespace(current=state), shared_expert=_LocalShared(device),
    )
    owner.gate = _LocalGate(state, device)
    flat = ((torch.arange(32 * 5120, device=device).reshape(32, 5120) % 113 - 56).float() / 32).bfloat16()
    load = torch.full((48,), 19, dtype=torch.int32, device=device)
    return engine, state, owner, flat, load


def _serial(owner, flat, shape, state, load, *, return_shared=False):
    ids, weights = owner.gate(
        flat, graph_routes=(state.metadata.valid_mask, state.routed_rows_device[owner.layer_id]),
        allow_top3_graph_routes=False,
    )
    if not state.bulk_counters:
        state.moe_calls[owner.layer_id].add_(1)
    routed = owner.ops.moe_engine.forward_shared_graph(
        flat, ids, weights, layer_id=owner.layer_id, expert_load=load, fused_expert_load=False,
    )
    shared = owner.shared_expert(flat, prefill_exact=True)
    if return_shared:
        return routed.reshape(shape), shared.reshape(shape)
    return (routed + shared.to(routed.dtype)).reshape(shape)


def _copy_inputs(reference, actual):
    re, rs, ro, rx, rl = reference
    ae, ass, ao, ax, al = actual
    rx.copy_(ax)
    rs.metadata.positions.copy_(ass.metadata.positions)
    rs.metadata.valid_mask.copy_(ass.metadata.valid_mask)
    rs.moe_calls.copy_(ass.moe_calls)
    rs.routed_rows_device.copy_(ass.routed_rows_device)
    re.bias.copy_(ae.bias)
    re.gpu_calls.copy_(ae.gpu_calls)
    re._mega_recv_stats.copy_(ae._mega_recv_stats)
    ro.gate.bias.copy_(ao.gate.bias)
    ro.shared_expert.scale.copy_(ao.shared_expert.scale)
    ro.shared_expert.shift.copy_(ao.shared_expert.shift)
    rl.copy_(al)


class VerifyMoEOverlapContracts(unittest.TestCase):
    def test_cpu_constructor_rejects_before_resources(self):
        with self.assertRaisesRegex(ValueError, "PPU CUDA"):
            VerifyMoEOverlap(torch.device("cpu"), object())


@unittest.skipUnless(torch.cuda.is_available(), "PPU required")
class VerifyMoEOverlapDevice(unittest.TestCase):
    def setUp(self):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        self.device = torch.device("cuda", torch.cuda.current_device())
        flags = patch.dict("os.environ", {"RTP_V41_PPU_VERIFY_DIRECT_MOE_ROUTES": "1"})
        flags.start()
        self.addCleanup(flags.stop)

    def _assert_state(self, actual, reference):
        ae, ass, _, _, al = actual
        re, rs, _, _, rl = reference
        for a, b in ((ae._mega_recv_stats, re._mega_recv_stats), (ae.gpu_calls, re.gpu_calls),
                     (ass.moe_calls, rs.moe_calls), (ass.routed_rows_device, rs.routed_rows_device), (al, rl)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @torch.inference_mode()
    def test_nondefault_caller_guards_and_dynamic_fingerprint(self):
        engine, state, owner, flat, load = _fixture(self.device)
        plan = VerifyMoEOverlap(self.device, engine)
        self.assertFalse(plan.supported(state, flat, owner))
        initial = plan.storage_signature()
        caller, other = torch.cuda.Stream(), torch.cuda.Stream()
        plan.bind_caller(caller)
        plan.prepare_caller(caller)
        self.assertEqual(initial, plan.storage_signature())
        with torch.cuda.stream(caller):
            self.assertTrue(plan.supported(state, flat, owner))
            for field, bad in (("verify", False), ("prefill", True), ("proposal", True),
                               ("grouped_moe", object()), ("verify_moe_overlap_allowed", False)):
                original = getattr(state, field)
                setattr(state, field, bad)
                self.assertFalse(plan.supported(state, flat, owner), field)
                setattr(state, field, original)
            state.metadata.q_len_per_req = 3
            self.assertFalse(plan.supported(state, flat, owner))
            state.metadata.q_len_per_req = 4
            self.assertFalse(plan.supported(state, flat[:8], owner))
            with patch.dict("os.environ", {"RTP_V41_PPU_VERIFY_DIRECT_MOE_ROUTES": "0"}):
                self.assertFalse(plan.supported(state, flat, owner))
            with patch("torch.cuda.is_current_stream_capturing", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "before capture"):
                    plan.bind_caller(caller)
            plan._lock.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "not reentrant"):
                    plan.forward(owner, flat, flat.shape, state, load)
            finally:
                plan._lock.release()
        with torch.cuda.stream(other):
            self.assertFalse(plan.supported(state, flat, owner))
            with self.assertRaisesRegex(RuntimeError, "caller changed"):
                plan.bind_caller(other)
        original = engine._mega_buffer
        engine._mega_buffer = object()
        self.assertNotEqual(initial, plan.storage_signature())
        with self.assertRaisesRegex(RuntimeError, "recapture"):
            plan.validate()
        engine._mega_buffer = original
        weights = engine._processed[7, self.device]
        engine._processed[7, self.device] = (engine.bias.clone(),)
        with torch.cuda.stream(caller):
            self.assertFalse(plan.supported(state, flat, owner))
            with self.assertRaisesRegex(ValueError, "unqualified"):
                plan.forward(owner, flat, flat.shape, state, load)
        engine._processed[7, self.device] = weights
        plan.validate()
        plan.close()
        self.assertFalse(plan.supported(state, flat, owner))
        with self.assertRaisesRegex(RuntimeError, "recapture"):
            plan.bind_caller(caller)
        caller.synchronize()

    @torch.inference_mode()
    def test_local_callbacks_eager_math_counters_and_return_shared(self):
        actual, reference = _fixture(self.device), _fixture(self.device)
        engine, state, owner, flat, load = actual
        plan, caller = VerifyMoEOverlap(self.device, engine), torch.cuda.Stream()
        caller.wait_stream(torch.cuda.current_stream())
        plan.bind_caller(caller)
        with torch.cuda.stream(caller), patch.dict("os.environ", {"RTP_V41_PPU_VERIFY_EXACT_SWIGLU": "1"}):
            for separate in (False, True):
                _copy_inputs(reference, actual)
                expected = _serial(reference[2], reference[3], (8, 4, 5120), reference[1], reference[4], return_shared=separate)
                got = plan.forward(owner, flat, (8, 4, 5120), state, load, return_shared=separate)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                self._assert_state(actual, reference)
        caller.synchronize()
        self.assertEqual(owner.shared_expert.streams, [int(caller.cuda_stream)] * 2)
        self.assertEqual(owner.gate.streams, [int(plan.side_stream.cuda_stream)] * 2)
        self.assertEqual(engine.streams, [int(plan.side_stream.cuda_stream)] * 2)
        self.assertEqual(plan.receipt()["forks"], 2)
        self.assertEqual(plan.receipt()["joins"], 2)

    @torch.inference_mode()
    def test_two_independent_graphs_replay_changed_live_inputs(self):
        captured = []
        for separate in (False, True):
            actual, reference = _fixture(self.device), _fixture(self.device)
            engine, state, owner, flat, load = actual
            plan, caller = VerifyMoEOverlap(self.device, engine), torch.cuda.Stream()
            caller.wait_stream(torch.cuda.current_stream())
            plan.bind_caller(caller)
            with torch.cuda.stream(caller):
                plan.forward(owner, flat, (32, 5120), state, load, return_shared=separate)
            caller.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=caller):
                output = plan.forward(owner, flat, (32, 5120), state, load, return_shared=separate)
            captured.append((actual, reference, plan, caller, graph, output, separate))
        for replay in range(2):
            for actual, reference, plan, caller, graph, output, separate in reversed(captured):
                engine, state, owner, flat, load = actual
                with torch.cuda.stream(caller):
                    flat.neg_().add_(0.03125 * (replay + 1))
                    state.metadata.positions.mul_(3).add_(7)
                    state.metadata.valid_mask.logical_not_()
                    if replay:
                        state.metadata.valid_mask.zero_()
                    owner.gate.bias.add_(11)
                    owner.shared_expert.scale.mul_(0.5)
                    owner.shared_expert.shift.add_(0.125)
                    engine.bias.sub_(0.0625)
                    engine._mega_recv_stats.fill_(31 + replay)
                    engine.gpu_calls.fill_(17 + replay)
                    state.moe_calls.fill_(23 + replay)
                    state.routed_rows_device.fill_(-41)
                    load.fill_(37 + replay)
                    _copy_inputs(reference, actual)
                    graph.replay()
                    expected = _serial(reference[2], reference[3], (32, 5120), reference[1], reference[4], return_shared=separate)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    self._assert_state(actual, reference)
                    plan.validate()
        for _, _, _, caller, _, _, _ in captured:
            caller.synchronize()

    @torch.inference_mode()
    def test_delayed_tuple_consumer_and_side_allocator_churn(self):
        actual, reference = _fixture(self.device), _fixture(self.device)
        engine, state, owner, flat, load = actual
        plan, caller = VerifyMoEOverlap(self.device, engine), torch.cuda.Stream()
        caller.wait_stream(torch.cuda.current_stream())
        plan.bind_caller(caller)
        with torch.cuda.stream(caller):
            for _ in range(4):
                _copy_inputs(reference, actual)
                wanted = _serial(reference[2], reference[3], flat.shape, reference[1], reference[4], return_shared=True)
                expected = wanted[0].float() + wanted[1].float()
                # Do not synchronize or consume the side result immediately.
                transient = flat.clone()
                parts = plan.forward(owner, transient, flat.shape, state, load, return_shared=True)
                delay = flat.float()
                for _ in range(8):
                    delay = delay * 0.999
                del delay
                got = parts[0].float() + parts[1].float()
                del parts, transient
                # Same-size allocations on the producing stream can otherwise
                # recycle routed storage before the queued caller read runs.
                with torch.cuda.stream(plan.side_stream):
                    churn = [torch.empty_like(flat) for _ in range(8)]
                    for buffer in churn:
                        buffer.fill_(123)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                self._assert_state(actual, reference)
        caller.synchronize()
        plan.side_stream.synchronize()

    @torch.inference_mode()
    def test_pool_two_capture_callers_share_engine_serial_execution(self):
        first, second = _fixture(self.device), _fixture(self.device)
        engine = first[0]
        second[2].ops.moe_engine = engine
        second = (engine, *second[1:])
        pool = VerifyMoEOverlapPlans(self.device, engine)
        initial = pool.storage_signature()
        captures, plans = [], []
        for actual, separate in ((first, False), (second, True)):
            _, state, owner, flat, load = actual
            caller = torch.cuda.Stream()
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                plan = pool.prepare_current_stream()
                self.assertIs(plan, pool.prepare_current_stream())
                self.assertIs(plan, pool.lookup_current_stream(required=True))
                plan.forward(owner, flat, flat.shape, state, load, return_shared=separate)
            caller.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=caller):
                plan = pool.lookup_current_stream(required=True)
                output = plan.forward(owner, flat, flat.shape, state, load, return_shared=separate)
            captures.append((actual, _fixture(self.device), caller, graph, output, separate))
            plans.append(plan)
        self.assertIsNot(plans[0], plans[1])
        self.assertNotEqual(plans[0]._caller_key, plans[1]._caller_key)
        self.assertNotEqual(int(plans[0].side_stream.cuda_stream), int(plans[1].side_stream.cuda_stream))
        self.assertIsNot(plans[0].producer, plans[1].producer)
        self.assertIsNot(plans[0].done, plans[1].done)
        self.assertEqual(initial, pool.storage_signature())
        execution = torch.cuda.Stream()
        execution.wait_stream(torch.cuda.current_stream())
        for _, _, caller, _, _, _ in captures:
            execution.wait_stream(caller)
        checks = []
        with torch.cuda.stream(execution):
            self.assertIsNone(pool.lookup_current_stream())
            for step, index in enumerate((1, 0, 1, 0)):
                actual, reference, _, graph, output, separate = captures[index]
                _, state, owner, flat, load = actual
                flat.neg_().add_(0.03125 * (step + 1))
                state.metadata.positions.mul_(3).add_(step + 1)
                state.metadata.valid_mask.logical_not_()
                if step == 2:
                    state.metadata.valid_mask.zero_()
                owner.gate.bias.add_(7)
                owner.shared_expert.scale.mul_(0.5)
                owner.shared_expert.shift.add_(0.0625)
                engine.bias.add_(0.03125)
                load.fill_(13 + step)
                _copy_inputs(reference, actual)
                # No synchronization between graphs: the one execution stream
                # must order all captures' complete side joins/workspace uses.
                graph.replay()
                got = tuple(value.clone() for value in output) if separate else output.clone()
                expected = _serial(reference[2], reference[3], flat.shape, reference[1], reference[4], return_shared=separate)
                stats = (engine._mega_recv_stats.clone(), engine.gpu_calls.clone(),
                         state.moe_calls.clone(), state.routed_rows_device.clone(), load.clone())
                re, rs, _, _, rl = reference
                wanted = (re._mega_recv_stats.clone(), re.gpu_calls.clone(), rs.moe_calls.clone(),
                          rs.routed_rows_device.clone(), rl.clone())
                checks.append((got, expected, stats, wanted))
            pool.validate()
        execution.synchronize()
        for got, expected, stats, wanted in checks:
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
            torch.testing.assert_close(stats, wanted, rtol=0, atol=0)
        self.assertEqual(pool.receipt()["prepared_callers"], 2)

    @torch.inference_mode()
    def test_pool_unprepared_capture_and_workspace_replacement_reject(self):
        engine, _, _, _, _ = _fixture(self.device)
        pool = VerifyMoEOverlapPlans(self.device, engine)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.assertIsNone(pool.lookup_current_stream())
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                pool.lookup_current_stream(required=True)
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "before capture"):
            with torch.cuda.graph(graph, stream=stream):
                pool.lookup_current_stream()
        self.assertEqual(pool.receipt()["prepared_callers"], 0)
        with torch.cuda.stream(stream):
            pool.prepare_current_stream()
            with patch("torch.cuda.is_current_stream_capturing", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "before capture"):
                    pool.prepare_current_stream()
        original = engine._mega_buffer
        engine._mega_buffer = object()
        with self.assertRaisesRegex(RuntimeError, "recapture"):
            pool.validate()
        engine._mega_buffer = original
        pool.validate()
        pool.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.lookup_current_stream()
        stream.synchronize()

    @torch.inference_mode()
    def test_callback_failures_join_without_retry(self):
        for branch in ("shared", "gate", "routed"):
            with self.subTest(branch=branch):
                engine, state, owner, flat, load = _fixture(self.device)
                plan, caller = VerifyMoEOverlap(self.device, engine), torch.cuda.Stream()
                caller.wait_stream(torch.cuda.current_stream())
                plan.bind_caller(caller)
                owner.shared_expert.fail = branch == "shared"
                owner.gate.fail = branch == "gate"
                engine.fail = branch == "routed"
                with torch.cuda.stream(caller):
                    with self.assertRaisesRegex(RuntimeError, f"local {branch} failure"):
                        plan.forward(owner, flat, flat.shape, state, load)
                    observed = engine.gpu_calls.clone()
                    moe_calls = state.moe_calls.clone()
                caller.synchronize()
                self.assertEqual(observed.item(), int(branch == "routed"))
                self.assertEqual(moe_calls[owner.layer_id].item(), 5 + int(branch == "routed"))
                expected_counts = torch.full_like(moe_calls, 5)
                expected_counts[owner.layer_id] += int(branch == "routed")
                torch.testing.assert_close(moe_calls, expected_counts, rtol=0, atol=0)
                self.assertEqual(len(owner.shared_expert.streams), 1)
                self.assertEqual(len(engine.streams), int(branch == "routed"))
                self.assertEqual(plan.receipt()["forks"], 1)
                self.assertEqual(plan.receipt()["joins"], 1)
                self.assertFalse(plan._lock.locked())


if __name__ == "__main__":
    unittest.main()
