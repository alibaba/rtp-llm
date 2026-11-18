"""Public CP8 metadata construction with actual empty-stub consumers.

CPU tensors retain real CP births, native input getters and the public
aggregate/build/propagation/finally chain. The fixture stops at the first
Block, before attention or CP transport. It never advertises CPU as CUDA.
"""

from contextlib import ExitStack, contextmanager
from itertools import accumulate
import os
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer as attention
from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import _Meta
from rtp_llm.models_py.modules.dsv41.ppu.block import SharedPPUBlock
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.row_metadata import tensor_metadata_key
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import CP_ENV
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import _context
from rtp_llm.models_py.modules.dsv41.prefill import forward as public
from rtp_llm.ops.compute_ops import PyModelInputs


ENV = dict(CP_ENV, RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA="1",
           RTP_V41_PPU_OPTIMIZED_DIAGNOSTICS="0", RTP_V41_PPU_OPTIMIZED_LATE_FOCUS="0",
           DSV4_FORWARD_TENSOR_DEBUG="0", DSV4_DEBUG_HIDDEN_COMPARE="0")


@contextmanager
def _environment(values):
    # Restore only owned keys. Native heartbeat/getenv threads must never see
    # a temporarily cleared process environment, including on assertion exit.
    absent = object()
    previous = {name: os.environ.get(name, absent) for name in values}
    try:
        for name, value in values.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is absent:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _AtFirstBlock(RuntimeError):
    pass


def _fixture(*, consumers=3, native_inputs=False, inference=False,
             lengths=(32,) * 8, prefixes=(64,) * 8, rank=1):
    with torch.inference_mode(inference), torch.no_grad():
        cp, ids = _context(rank, lengths, prefixes)
        cp.first_position_host = prefixes[0] + min(rank * (cp.chunk_lengths_per_req[0] // 2), lengths[0] - 1)
        v4 = NS(layers=[], _cp_info=cp.cp_info, _cp_size=8, _cp_rank=rank,
                _kv_cache_sharded=True, capture_aux_hidden_layer_ids=(),
                fp8_kv_cache=True, hc_mult=4, args=NS(v41_config=object()),
                _mtp_hidden_buffer=None, _dsv4_prefill_fast_layer_calls=None,
                _prefill_ws_q_dim=2, _prefill_ws_main_w=2, _prefill_ws_idx_w=2,
                _prefill_ws_full_rows=cp.chunk_length)
        model = NS(v4=v4, _is_decode_role=False, _is_speculative=False, _prefill_cp_size=8,
                   parallelism_config=NS(tp_size=8))
        runtime = NS(model=model, policy=SharedPrefillPolicy(mode="compact"), current=None)
        for index in range(consumers):
            block = SharedPPUBlock.__new__(SharedPPUBlock)
            torch.nn.Module.__init__(block)
            attn = attention.SharedPPUAttention.__new__(attention.SharedPPUAttention)
            torch.nn.Module.__init__(attn)
            attn.runtime, attn.layer_id = runtime, index
            attn.compress_ratio, attn.window_size = index % 3, 128
            attn._cp_ctx = attn._prefill_meta_shared = None
            attn._kv_cache = attn._block_tables_by_type = attn._shared_attention = None
            block.attn, block.engram, block.engram_token_mask = attn, None, None
            v4.layers.append(block)
        cache = SharedNativeCache.__new__(SharedNativeCache)
        cache.sharded, cache._is_prefill = True, True
        cache.lengths = tuple(p + n for p, n in zip(prefixes, lengths))
        cache.bind_swa_fresh_lengths = lambda *args, **kwargs: None
        cache.shared_prefix_stripes = object()
        inputs = PyModelInputs() if native_inputs else NS()
        inputs.input_ids, inputs.combo_position_ids = ids, cp.global_positions
        attn_inputs = NS(
            cu_seqlens=torch.tensor((0, *accumulate(cp.chunk_lengths_per_req)), dtype=torch.int32),
            input_lengths=torch.tensor(cp.chunk_lengths_per_req, dtype=torch.int32),
            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32), is_cuda_graph=False,
        )
        state = SharedForwardState(runtime, inputs, attn_inputs, cache, prefill=True, idle=False,
                                   prefixes=prefixes, lengths=lengths)
        runtime.current, v4._ppu_runtime = state, runtime

        def propagate(context):
            state.bind_context(context)
            for layer in v4.layers:
                layer.attn._cp_ctx = context

        v4._propagate_cp_ctx = propagate
        propagate(cp)
        v4.embed = lambda value: torch.ones((value.numel(), 2), dtype=torch.bfloat16)
    return NS(state=state, runtime=runtime, v4=v4, cp=cp, ids=ids, workspace=object(), seen=[])


def _public_build(f, *, forbid_generic=False, prepare_hidden=False, recording=False):
    """Execute the actual public aggregate and metadata build, then stop."""
    def stop(*args, **kwargs):
        f.seen.extend(layer.attn._prefill_meta_shared for layer in f.v4.layers)
        raise _AtFirstBlock()

    for layer in f.v4.layers:
        layer.forward = stop
    original_to = torch.Tensor.to

    def to(value, *args, **kwargs):
        if forbid_generic and any(value is source for source in (
            f.cp.prefix_lengths, f.cp.req_id_per_token,
            f.state.attn.input_lengths, f.state.attn.prefix_lengths,
        )):
            raise AssertionError("unused generic metadata must not be cast/uploaded")
        return original_to(value, *args, **kwargs)

    original_metadata = public.device_attention_metadata
    propagated = False
    original_propagate = f.v4._propagate_cp_ctx

    def propagate(context):
        nonlocal propagated
        original_propagate(context)
        callback = getattr(f, "after_propagate", None)
        if callback is not None:
            callback()
        propagated = True

    def metadata(owner, name):
        if forbid_generic and propagated:
            raise AssertionError("unused generic metadata must not be requested")
        return original_metadata(owner, name)

    with ExitStack() as stack:
        stack.enter_context(patch.object(public, "PrefillWorkspace", return_value=f.workspace))
        stack.enter_context(patch.object(public, "build_cp_context_for_forward", return_value=f.cp))
        stack.enter_context(patch.object(f.v4, "_propagate_cp_ctx", side_effect=propagate))
        stack.enter_context(patch.object(public, "device_attention_metadata", side_effect=metadata))
        stack.enter_context(patch.object(torch.Tensor, "to", new=to))
        if forbid_generic:
            stack.enter_context(patch.object(torch.Tensor, "item", side_effect=AssertionError("no readback")))
            stack.enter_context(patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no readback")))
        stack.enter_context(patch.object(public._rt, "ENABLED", recording))
        if recording:
            stack.enter_context(patch.object(public._rt, "begin", return_value=None))
            stack.enter_context(patch.object(public._rt, "_get_buf", return_value=object()))
            stack.enter_context(patch.object(public._rt, "record", return_value=None))
        build = stack.enter_context(patch.object(public, "build_and_propagate_prefill_meta_fp8",
                                                wraps=public.build_and_propagate_prefill_meta_fp8))
        try:
            public.forward_layers(
                f.v4, None, f.ids, f.cp.global_positions, f.state.attn.cu_seqlens, None, None,
                attn_inputs=f.state.attn,
                prepare_hidden_fn=(lambda **kwargs: torch.ones((f.ids.numel(), 4, 2))) if prepare_hidden else None,
            )
        except _AtFirstBlock:
            pass
        else:
            raise AssertionError("public Prefill did not reach its first Block")
    return build


class CP8MetaStubContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(_environment(ENV))
        self.stack.enter_context(patch("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True))
        self.stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch.object(torch.cuda, "current_stream",
                                             side_effect=AssertionError("CPU fixture must not initialize CUDA")))

    def _assert_unused(self, f, build):
        build.assert_called_once()
        self.assertEqual(build.call_args.args[2], 0)
        for name in ("sp_per_req", "req_id_per_token", "input_lengths", "prefix_lengths"):
            self.assertIsNone(build.call_args.kwargs[name], name)
        self.assertEqual(build.call_args.kwargs["max_seqlen_q"], 0)
        self.assertEqual(len(f.seen), len(f.v4.layers))
        self.assertTrue(all(type(meta) is attention._Meta and meta.workspace is f.workspace for meta in f.seen))
        self.assertTrue(all(layer.attn._prefill_meta_shared is None for layer in f.v4.layers))
        self.assertEqual(f.state.shared["cp_meta_stub_admissions"], 1)
        self.assertEqual(f.state.shared["cp_meta_stub_consumers"], len(f.v4.layers))

    def _assert_generic(self, f, build):
        build.assert_called_once()
        self.assertEqual(build.call_args.args[2], f.cp.first_position_host)
        for name in ("sp_per_req", "req_id_per_token", "input_lengths", "prefix_lengths"):
            self.assertIsNotNone(build.call_args.kwargs[name], name)
        self.assertEqual(build.call_args.kwargs["max_seqlen_q"], max(f.cp.chunk_lengths_per_req))
        self.assertEqual(f.state.shared.get("cp_meta_stub_admissions", 0), 0)
        self.assertTrue(all(layer.attn._prefill_meta_shared is None for layer in f.v4.layers))

    def test_actual_public_full4096_all40_stubs_skip_unused_inputs_with_one_birth_check(self):
        f = _fixture(consumers=40, lengths=(4096,) * 8, prefixes=(126976,) * 8)
        self.assertIsNotNone(f.state._cp_full_rows_birth)
        # This wraps the real proof. No admission result is supplied by a mock.
        with patch.object(f.state, "_cp_full_rows_valid", wraps=f.state._cp_full_rows_valid) as valid:
            self._assert_unused(f, _public_build(f, forbid_generic=True))
        valid.assert_called_once()
        self.assertFalse(f.state._mega_all_live_phase(), "DP mask admission remains local")

    def test_native_pymodelinputs_getters_and_inference_birth_keep_exact_tensorimpl(self):
        for inference in (False, True):
            with self.subTest(inference=inference):
                f = _fixture(native_inputs=True, inference=inference)
                self.assertEqual(tensor_metadata_key(f.state.inputs.input_ids), tensor_metadata_key(f.ids))
                self._assert_unused(f, _public_build(f, forbid_generic=True))
        for change in ("clone", "view", "native_write", "native_rebind"):
            with self.subTest(change=change):
                f = _fixture(native_inputs=True)
                if change == "clone":
                    f.ids = f.ids.clone()
                elif change == "view":
                    f.ids = f.ids.view_as(f.ids)
                elif change == "native_write":
                    f.state.inputs.input_ids.add_(0)
                else:
                    f.state.inputs.input_ids = f.ids.clone()
                self._assert_generic(f, _public_build(f))

    def test_unknown_builder_capability_subclass_and_hooks_keep_public_fallback(self):
        original = attention.SharedPPUAttention._build_shared_prefill_meta

        def replacement(owner, *args, **kwargs):
            return original(owner, *args, **kwargs)

        class OtherAttention(attention.SharedPPUAttention):
            pass

        for change in ("instance_builder", "class_builder", "unknown_capability", "class_capability",
                       "subclass", "attn_hook", "block_hook", "global_hook", "unregistered"):
            with self.subTest(change=change), ExitStack() as stack:
                f = _fixture()
                layer, attn = f.v4.layers[-1], f.v4.layers[-1].attn
                if change == "instance_builder":
                    attn._build_shared_prefill_meta = MethodType(replacement, attn)
                elif change == "class_builder":
                    stack.enter_context(patch.object(attention.SharedPPUAttention, "_build_shared_prefill_meta", replacement))
                elif change == "unknown_capability":
                    attn._shared_prefill_meta_inputs_required = None
                elif change == "class_capability":
                    stack.enter_context(patch.object(attention.SharedPPUAttention,
                                                    "_shared_prefill_meta_inputs_required", lambda *args, **kwargs: True))
                elif change == "subclass":
                    attn.__class__ = OtherAttention
                elif change in ("attn_hook", "block_hook"):
                    owner = attn if change == "attn_hook" else layer
                    handle = owner.register_forward_pre_hook(lambda *args: None)
                    stack.callback(handle.remove)
                elif change == "global_hook":
                    handle = torch.nn.modules.module.register_module_forward_pre_hook(lambda *args: None)
                    stack.callback(handle.remove)
                else:
                    f.state.shared["layers"].pop(attn.layer_id)
                self._assert_generic(f, _public_build(f))

    def test_same_function_with_changed_stub_code_is_not_a_known_empty_builder(self):
        function = attention.SharedPPUAttention._build_shared_prefill_meta
        original = function.__code__

        def replacement(self, *args, **kwargs):
            return _Meta()

        # The replacement body runs in function.__globals__, which owns _Meta.
        # Restore the existing code attribute rather than asking patch to delete it.
        try:
            function.__code__ = replacement.__code__
            f = _fixture()
            self._assert_generic(f, _public_build(f))
        finally:
            function.__code__ = original

    def test_debug_recording_custom_prepare_and_domain_changes_do_not_opt_out(self):
        for change in ("debug", "recording", "prepare", "host_off", "graph", "verify", "proposal",
                       "idle", "decode", "compacted", "stale_runtime", "wrong_rank", "wrong_info",
                       "wrong_context", "cp_field_clone", "cp_field_write", "unknown_birth", "invalid_birth"):
            with self.subTest(change=change), ExitStack() as stack:
                f = _fixture()
                if change == "debug":
                    stack.enter_context(_environment({"DSV4_FORWARD_TENSOR_DEBUG": "1"}))
                elif change == "host_off":
                    stack.enter_context(_environment({"RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "0"}))
                elif change in ("graph", "verify", "proposal"):
                    setattr(f.state, "is_graph" if change == "graph" else change, True)
                elif change in ("idle", "compacted"):
                    setattr(f.state, change, True)
                elif change == "decode":
                    f.state.prefill = False
                elif change == "stale_runtime":
                    f.runtime.current = None
                elif change == "wrong_rank":
                    f.v4._cp_rank = 7
                elif change == "wrong_info":
                    f.v4._cp_info = object()
                elif change == "wrong_context":
                    f.after_propagate = lambda: setattr(f.v4.layers[-1].attn, "_cp_ctx", object())
                elif change == "cp_field_clone":
                    f.cp.local_is_real = f.cp.local_is_real.clone()
                elif change == "cp_field_write":
                    f.cp.req_id_per_token.add_(0)
                elif change == "unknown_birth":
                    f.state._cp_full_rows_birth = None
                elif change == "invalid_birth":
                    f.state._cp_full_rows_invalid = True
                self._assert_generic(f, _public_build(f, prepare_hidden=change == "prepare",
                                                      recording=change == "recording"))

    def test_non_ppu_aggregate_and_existing_dp_consumer_contract_are_unchanged(self):
        generic = NS(layers=[NS(attn=NS(_build_shared_prefill_meta=lambda: None))])
        with patch.object(attention, "_cp_prefill_meta_stub_inputs_unused",
                          side_effect=AssertionError("non-PPU must not invoke the optional probe")):
            self.assertTrue(public._shared_prefill_meta_inputs_required(generic, torch.arange(3), None))
        f = _fixture()
        attn = f.v4.layers[0].attn
        self.assertTrue(attn._shared_prefill_meta_inputs_required(
            builder=attn._build_shared_prefill_meta, v4=f.v4, input_ids=f.ids, cp_ctx=f.cp,
        ), "the existing per-consumer DP predicate still rejects CP8")


if __name__ == "__main__":
    unittest.main()
