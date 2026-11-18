"""Original CP birth before the public embedding broadcast and HC bit copies.

The public coordinator and SharedForwardState bind_context are real. CPU
cases stop at the selected embedding branch; device cases execute the
existing native expand_hc before stopping unrelated model work. No WORLD8
collective is introduced by this local copy operation.
"""

import os
import unittest
from contextlib import ExitStack
from itertools import accumulate
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer
from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_hc_expand as cp_expand
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.row_metadata import tensor_metadata_key
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import CP_ENV
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import (
    _context,
)
from rtp_llm.models_py.modules.dsv41.prefill import _hc_expand_triton
from rtp_llm.models_py.modules.dsv41.prefill import forward as public
from rtp_llm.ops.compute_ops import PyModelInputs


def _fixture(
    *,
    lengths=(17, 33, 145),
    prefixes=(17, 256, 8192),
    rank=0,
    device="cpu",
    source=None,
    native_inputs=False
):
    cp, ids = _context(rank, lengths, prefixes, device)
    if source is None:
        source = torch.empty(cp.chunk_length, 5120, dtype=torch.bfloat16, device=device)
    v4 = NS(
        layers=[],
        fp8_kv_cache=False,
        hc_mult=4,
        _cp_info=cp.cp_info,
        _cp_size=8,
        _cp_rank=rank,
        capture_aux_hidden_layer_ids=(),
        _mtp_hidden_buffer=None,
        _dsv4_prefill_fast_layer_calls=None,
        embed=lambda input_ids: source,
        _hc_head_reduce=lambda value: value,
        norm=lambda value: value,
    )
    model = NS(
        v4=v4,
        _is_decode_role=False,
        _is_speculative=False,
        _prefill_cp_size=8,
        parallelism_config=NS(tp_size=8),
    )
    runtime = NS(model=model, policy=SharedPrefillPolicy(mode="compact"), current=None)
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache.sharded, cache._is_prefill = True, True
    cache.lengths = tuple(p + n for p, n in zip(prefixes, lengths))
    cache.bind_swa_fresh_lengths = lambda *args, **kwargs: None
    cache.shared_prefix_stripes = object()
    inputs = PyModelInputs() if native_inputs else NS()
    inputs.input_ids, inputs.combo_position_ids = ids, cp.global_positions
    attn = NS(
        cu_seqlens=torch.tensor(
            (0, *accumulate(cp.chunk_lengths_per_req)), dtype=torch.int32, device=device
        ),
        input_lengths=torch.tensor(cp.chunk_lengths_per_req, dtype=torch.int32),
        prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
        is_cuda_graph=False,
    )
    state = SharedForwardState(
        runtime,
        inputs,
        attn,
        cache,
        prefill=True,
        idle=False,
        prefixes=prefixes,
        lengths=lengths,
    )
    runtime.current, v4._ppu_runtime = state, runtime
    events = []

    def propagate(context):
        events.append("propagate")
        # This is exactly the state bind invoked by the PPU transformer's
        # _propagate_cp_ctx override, without constructing model weights.
        state.bind_context(context)

    def embed(input_ids):
        events.append("embed")
        return source

    v4._propagate_cp_ctx, v4.embed = propagate, embed
    return NS(v4=v4, state=state, cp=cp, source=source, ids=ids, events=events)


class _Selected(RuntimeError):
    pass


def _forward(fixture, *, native=None, recording=False):
    """Stop after the actual public coordinator chooses its HC operation."""
    f = fixture
    selected = {}

    def expand(source):
        f.events.append("native")
        if source is not f.source:
            raise AssertionError("coordinator changed the embedding owner")
        selected["birth"] = f.state._cp_full_rows_birth
        if native is not None:
            selected["output"] = native(source)
        raise _Selected("native")

    def repeat(value, *args, **kwargs):
        f.events.append("repeat")
        raise _Selected("repeat")

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(public, "build_cp_context_for_forward", return_value=f.cp)
        )
        stack.enter_context(patch.object(public._rt, "ENABLED", recording))
        if recording:
            stack.enter_context(patch.object(public._rt, "begin", return_value=None))
            stack.enter_context(
                patch.object(public._rt, "_get_buf", return_value=object())
            )
            stack.enter_context(patch.object(public._rt, "record", return_value=None))
        stack.enter_context(
            patch.object(_hc_expand_triton, "expand_hc", side_effect=expand)
        )
        stack.enter_context(patch.object(torch.Tensor, "repeat", new=repeat))
        try:
            public.forward_layers(
                f.v4,
                None,
                f.ids,
                f.cp.global_positions,
                f.state.attn.cu_seqlens,
                None,
                None,
                attn_inputs=f.state.attn,
            )
        except _Selected as result:
            selected["route"] = str(result)
        else:
            raise AssertionError("coordinator did not reach embedding broadcast")
    return selected


class CP8HCExpandContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, CP_ENV))
        self.stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True
            )
        )
        self.stack.enter_context(
            patch.object(
                attention_layer, "optimized_diagnostics_enabled", return_value=False
            )
        )
        self.stack.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        # Some coordinator cases explicitly double Tensor.is_cuda so the
        # public branch can be tested on CPU storage. Keep the real birth's
        # stream bind, but supply that metadata boundary without a GPU init.
        self.cpu_stream = NS(device=torch.device("cpu"), cuda_stream=271)
        self.stack.enter_context(
            patch.object(torch.cuda, "current_stream", return_value=self.cpu_stream)
        )
        self.stack.enter_context(
            patch.object(
                torch.cuda,
                "_lazy_init",
                side_effect=AssertionError("CPU HC fixture must not initialize CUDA"),
            )
        )
        self.stack.enter_context(
            patch.object(cp_expand, "_device_supported", return_value=True)
        )
        self.stack.enter_context(
            patch.object(public._fwd_dbg, "enabled", return_value=False)
        )

    def test_actual_public_coordinator_binds_original_cp_before_native_broadcast(self):
        with patch.object(torch.Tensor, "is_cuda", new=property(lambda tensor: True)):
            for lengths, prefixes in (
                ((4096,) * 8, (126976,) * 8),
                ((17, 33, 145), (17, 256, 8192)),
            ):
                f = _fixture(lengths=lengths, prefixes=prefixes)
                self.assertIsNone(f.state._cp_full_rows_birth)
                selected = _forward(f)
                self.assertEqual(selected["route"], "native")
                self.assertEqual(f.events, ["propagate", "embed", "native"])
                self.assertIsNotNone(selected["birth"])
                self.assertEqual(
                    f.state._cp_metadata_stream[0], (torch.device("cpu"), 271)
                )
                self.assertIs(f.state._cp_metadata_stream[1], self.cpu_stream)
                self.assertTrue(cp_expand.enabled(f.v4, f.source, f.ids))
                self.assertFalse(
                    public._ppu_local_hc_expand_enabled(f.v4, f.source, f.ids),
                    "DP predicate stays local",
                )

    def test_direct_eligibility_rejects_unknown_unborn_stale_or_different_owners(self):
        f = _fixture()
        self.assertFalse(cp_expand.enabled(f.v4, f.source, f.ids))
        f.v4._propagate_cp_ctx(f.cp)
        self.assertTrue(cp_expand.enabled(f.v4, f.source, f.ids))
        self.assertFalse(cp_expand.enabled(f.v4, f.source, f.ids.clone()))
        f.v4._ppu_runtime.current = NS()
        self.assertFalse(cp_expand.enabled(f.v4, f.source, f.ids))
        f.v4._ppu_runtime.current = f.state
        f.cp.global_positions = f.cp.global_positions.clone()
        self.assertFalse(cp_expand.enabled(f.v4, f.source, f.ids))
        self.assertTrue(f.state._cp_full_rows_invalid)

    def test_native_cpp_input_ids_getter_uses_tensorimpl_identity_and_rejects_copies(
        self,
    ):
        f = _fixture(native_inputs=True)
        f.v4._propagate_cp_ctx(f.cp)
        native_ids = f.state.inputs.input_ids
        self.assertEqual(tensor_metadata_key(native_ids), tensor_metadata_key(f.ids))
        self.assertTrue(cp_expand.enabled(f.v4, f.source, native_ids))
        self.assertFalse(cp_expand.enabled(f.v4, f.source, native_ids.clone()))
        self.assertFalse(
            cp_expand.enabled(f.v4, f.source, native_ids.view_as(native_ids))
        )
        with patch.object(torch.Tensor, "is_cuda", new=property(lambda tensor: True)):
            self.assertEqual(_forward(f)["route"], "native")

    def test_actual_coordinator_preserves_graph_decode_debug_and_view_fallbacks(self):
        for change in (
            "graph",
            "verify",
            "decode",
            "idle",
            "compacted",
            "dtype",
            "negative",
            "strided",
            "recording",
            "debug",
        ):
            with self.subTest(change=change), patch.object(
                torch.Tensor, "is_cuda", new=property(lambda tensor: True)
            ):
                f = _fixture()
                if change == "graph":
                    f.state.is_graph = True
                elif change == "verify":
                    f.state.verify = True
                elif change == "decode":
                    f.state.prefill = False
                elif change == "idle":
                    f.state.idle = True
                elif change == "compacted":
                    f.state.compacted = True
                elif change == "dtype":
                    f.source = f.source.float()
                elif change == "negative":
                    f.source = torch._neg_view(f.source)
                elif change == "strided":
                    f.source = torch.empty(
                        5120, f.cp.chunk_length, dtype=torch.bfloat16
                    ).t()
                f.v4.embed = lambda ids: f.source
                with patch.object(
                    public._fwd_dbg, "enabled", return_value=change == "debug"
                ):
                    self.assertEqual(
                        _forward(f, recording=change == "recording")["route"], "repeat"
                    )

    def test_cp_metadata_rank_model_and_embedding_constraints_remain_narrow(self):
        for change in ("rank", "info", "model", "hc", "grad", "width", "rows", "cp1"):
            f = _fixture()
            f.v4._propagate_cp_ctx(f.cp)
            if change == "rank":
                f.v4._cp_rank = 7
            elif change == "info":
                f.v4._cp_info = object()
            elif change == "model":
                f.v4._ppu_runtime.model = NS(v4=object())
            elif change == "hc":
                f.v4.hc_mult = 2
            elif change == "grad":
                f.source.requires_grad_(True)
            elif change == "width":
                f.source = f.source[:, :-1].contiguous()
            elif change == "rows":
                f.source = f.source[:-1]
            elif change == "cp1":
                f.v4._cp_size = 1
            self.assertFalse(cp_expand.enabled(f.v4, f.source, f.ids), change)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1",
    "explicit PPU device test",
)
class CP8HCExpandDevice(unittest.TestCase):
    @torch.no_grad()
    def test_public_native_copy_matches_repeat_for_all_bf16_ieee_patterns_and_cp_padding(
        self,
    ):
        self.assertTrue(is_ppu())
        native = _hc_expand_triton.expand_hc
        with patch.dict(os.environ, CP_ENV):
            for rank, lengths, prefixes in (
                (0, (4096,) * 8, (126976,) * 8),
                (0, (17, 33, 145), (17, 256, 8192)),
                (7, (17, 33, 145), (17, 256, 8192)),
            ):
                cp, _ = _context(rank, lengths, prefixes, "cuda")
                rows = cp.chunk_length
                owner = torch.full(
                    (rows * 5120 + 9,), 17, dtype=torch.bfloat16, device="cuda"
                )
                source = owner[1 : 1 + rows * 5120].view(rows, 5120)
                bits = (torch.arange(source.numel(), device="cuda") % 65536).to(
                    torch.int16
                )
                source.view(torch.int16).copy_(bits.reshape_as(source))
                before = owner.clone()
                expected = source.unsqueeze(1).repeat(1, 4, 1)
                f = _fixture(
                    rank=rank,
                    lengths=lengths,
                    prefixes=prefixes,
                    device="cuda",
                    source=source,
                )
                result = _forward(f, native=native)
                self.assertEqual(result["route"], "native")
                actual = result["output"]
                self.assertEqual(actual.shape, (rows, 4, 5120))
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(owner.view(torch.uint8), before.view(torch.uint8))
                )
                self.assertFalse(torch._C._is_alias_of(actual, source))
                # Each lane is independently materialized, as with repeat.
                untouched = actual[:, 1:].clone()
                actual[:, 0].zero_()
                self.assertTrue(
                    torch.equal(
                        actual[:, 1:].contiguous().view(torch.uint8),
                        untouched.view(torch.uint8),
                    )
                )


if __name__ == "__main__":
    unittest.main()
