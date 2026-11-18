"""CP-specific Engram prefetch flag flow and the unchanged masked gate core.

CPU tests retain SharedPPUBlock, public _inject_engram and PPUSharedEngram;
only the unrelated HC body, projection and lookup are explicit fixtures.
Device cases run the actual native masked gate against its existing ATen
relative contract. Masked IEEE arithmetic follows x + 0 * shared, so a false
mask is not claimed to bit-copy hidden when shared is NaN/Inf or x is -zero.
"""

import os
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.block import Block
from rtp_llm.models_py.modules.dsv41.ppu import attention_layer, block
from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_engram_gate as cp_gate
from rtp_llm.models_py.modules.dsv41.ppu import engram, layers
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import (
    CP_ENV,
    _state,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import (
    _context,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_engram_gate import (
    assert_bits,
    assert_frozen_gate,
    cpu_oracle,
    module_fixture,
)

FLAGS = dict(CP_ENV, **{cp_gate.ENV: "1", "RTP_V41_PPU_CP8_YOCO_ENGRAM_RS": "1"})


def _fixture(
    *, rank=0, lengths=(17, 33, 145), prefixes=(17, 256, 8192), layer=1, device="cpu"
):
    cp, ids = _context(rank, lengths, prefixes, device)
    module, hidden, hashes, value, image_mask = module_fixture(
        cp.chunk_length, 5120, device
    )
    module.layer_id = layer
    unit = block.SharedPPUBlock.__new__(block.SharedPPUBlock)
    nn.Module.__init__(unit)
    unit.layer_id, unit.engram = layer, module
    unit.engram_hashes, unit.engram_token_mask = hashes, image_mask
    unit.attn = NS(runtime=None)
    model = NS(
        _is_decode_role=False,
        _is_speculative=False,
        _prefill_cp_size=8,
        parallelism_config=NS(tp_size=8),
        v4=NS(
            layers=[unit],
            _propagate_cp_ctx=lambda context: None,
            capture_aux_hidden_layer_ids=(),
        ),
    )
    runtime = NS(model=model, policy=SharedPrefillPolicy(mode="compact"), current=None)
    state = _state(cp, ids, runtime=runtime)
    unit.attn.runtime = runtime
    unit.bind_execution_context = Mock()
    state.before_block = Mock(
        return_value=(hidden, ids, cp.global_positions, state.attn.cu_seqlens)
    )
    state.publication = lambda layer: nullcontext()
    return NS(
        unit=unit,
        module=module,
        state=state,
        cp=cp,
        hidden=hidden,
        hashes=hashes,
        value=value,
        image_mask=image_mask,
        ids=ids,
    )


def _inject_public(unit, hidden, *args, **kwargs):
    return Block._inject_engram(unit, hidden)


def _run(fixture, *, native=False):
    f = fixture
    with ExitStack() as stack:
        native_call = None
        if native:
            kernels = engram._triton_kernels()
            if kernels is None:
                raise AssertionError("native PPU Engram gate is unavailable")
            native_call = stack.enter_context(
                patch.object(
                    kernels, "fused_engram_gate", wraps=kernels.fused_engram_gate
                )
            )
        stack.enter_context(
            patch.object(Block, "forward", autospec=True, side_effect=_inject_public)
        )
        stack.enter_context(patch.object(layers, "wkv_project", return_value=f.value))
        gate = stack.enter_context(
            patch.object(
                layers,
                "engram_gate_inject",
                **(
                    {"wraps": engram.engram_gate_inject}
                    if native
                    else {"return_value": f.hidden}
                )
            )
        )
        result = f.unit(f.hidden, f.ids, f.cp.global_positions, f.state.attn.cu_seqlens)
        if native_call is not None and native_call.call_count != 1:
            raise AssertionError(
                "CP Engram gate did not execute exactly one native call"
            )
    return result, gate


class CP8EngramGateContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, FLAGS))
        self.stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True
            )
        )
        self.stack.enter_context(patch.object(block, "is_ppu", return_value=True))
        self.stack.enter_context(
            patch.object(
                attention_layer, "optimized_diagnostics_enabled", return_value=False
            )
        )
        self.stack.enter_context(
            patch("torch.cuda.is_current_stream_capturing", return_value=False)
        )
        self.stack.enter_context(
            patch.object(cp_gate, "_device_supported", return_value=True)
        )

    def test_actual_block_to_engram_keeps_derived_image_and_padding_mask(self):
        for rank in range(8):
            f = _fixture(rank=rank, layer=1 if rank % 2 else 14)
            self.assertFalse(
                f.state._cp_full_rows_birth[2],
                "startup padding is a supported masked domain",
            )
            expected = f.image_mask & f.cp.local_is_real
            self.assertTrue(torch.equal(f.unit.engram_token_mask, expected))
            self.assertIsNot(f.unit.engram_token_mask, f.cp.local_is_real)
            self.assertFalse(
                block._local_prefill_engram_gate_enabled(f.state),
                "the DP/local guard stays local",
            )
            self.assertTrue(cp_gate.enabled(f.state, f.unit, f.hidden))
            f.state.engram_round_schedule = object()
            _, gate = _run(f)
            self.assertIs(gate.call_args.kwargs["token_mask"], f.unit.engram_token_mask)
            self.assertIs(gate.call_args.kwargs["fused_mask"], True)
            f.module.ops.engram_lookup.assert_called_once_with(
                f.hashes,
                layer_id=f.unit.layer_id,
                uniform_rank_rows=True,
                round_schedule=f.state.engram_round_schedule,
            )
            self.assertIsNone(f.module._prefetched)
            self.assertEqual(
                f.state.shared["cp_engram_gate_admissions"],
                [(f.unit.layer_id, f.cp.chunk_length, (f.cp.chunk_length,))],
            )

    def test_flag_off_keeps_lookup_options_and_one_use_native_choice_independent(self):
        f = _fixture()
        for flag in ("0", "1", "0"):
            f.module.ops.engram_lookup.reset_mock()
            with patch.dict(os.environ, {cp_gate.ENV: flag}):
                _, gate = _run(f)
            self.assertEqual(
                gate.call_args.kwargs.get("fused_mask", False), flag == "1"
            )
            f.module.ops.engram_lookup.assert_called_once_with(
                f.hashes, layer_id=1, uniform_rank_rows=True
            )
            self.assertIsNone(f.module._prefetched)

    def test_stale_birth_other_phases_and_observable_hooks_do_not_enter_cp_gate(self):
        for field, value in (
            ("prefill", False),
            ("idle", True),
            ("compacted", True),
            ("verify", True),
            ("is_graph", True),
        ):
            f = _fixture()
            setattr(f.state, field, value)
            self.assertFalse(cp_gate.enabled(f.state, f.unit, f.hidden), field)
        f = _fixture()
        f.cp.global_positions = f.cp.global_positions.clone()
        self.assertFalse(cp_gate.enabled(f.state, f.unit, f.hidden))
        self.assertTrue(f.state._cp_full_rows_invalid)
        for target in ("block", "engram"):
            f = _fixture()
            handle = (f.unit if target == "block" else f.module).register_forward_hook(
                lambda *args: None
            )
            self.addCleanup(handle.remove)
            self.assertFalse(cp_gate.enabled(f.state, f.unit, f.hidden), target)
            _, gate = _run(f)
            self.assertNotIn("fused_mask", gate.call_args.kwargs)
            handle.remove()

    def test_actual_mask_stride_is_supported_without_resolving_deferred_views(self):
        f = _fixture()
        storage = torch.zeros(2 * f.cp.chunk_length, dtype=torch.bool)
        storage[::2].copy_(f.unit.engram_token_mask)
        f.unit.engram_token_mask = storage[::2]
        self.assertTrue(cp_gate.enabled(f.state, f.unit, f.hidden))
        _, gate = _run(f)
        self.assertEqual(gate.call_args.kwargs["token_mask"].stride(), (2,))
        f.hidden = torch._neg_view(f.hidden)
        self.assertFalse(cp_gate.enabled(f.state, f.unit, f.hidden))

    def test_prefetch_operand_mutation_clears_pending_choice_before_public_injection(
        self,
    ):
        f = _fixture()
        embedded = f.module.ops.engram_lookup.return_value

        def lookup(*args, **kwargs):
            f.unit.engram_token_mask.logical_not_()
            return embedded

        f.module.ops.engram_lookup.side_effect = lookup
        with self.assertRaisesRegex(RuntimeError, "operands changed during prefetch"):
            _run(f)
        self.assertIsNone(f.module._prefetched)
        self.assertNotIn("cp_engram_gate_admissions", f.state.shared)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1",
    "explicit PPU device test",
)
class CP8EngramGateDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu())

    @torch.no_grad()
    def test_actual_cp_full_native_gate_keeps_existing_aten_relative_contract(self):
        with patch.dict(os.environ, FLAGS):
            for layer, rank, lengths, prefixes in (
                (1, 0, (4096,) * 8, (126976,) * 8),
                (14, 0, (17, 33, 145), (17, 256, 8192)),
            ):
                with self.subTest(layer=layer, lengths=lengths):
                    f = _fixture(
                        layer=layer,
                        rank=rank,
                        lengths=lengths,
                        prefixes=prefixes,
                        device="cuda",
                    )
                    mask = f.unit.engram_token_mask
                    frozen = [
                        value.clone()
                        for value in (
                            f.hidden,
                            f.module.q_weight,
                            f.module.k_weight,
                            f.value,
                            mask,
                        )
                    ]
                    expected = engram.engram_gate_inject(
                        f.hidden,
                        f.module.q_weight,
                        f.module.k_weight,
                        f.value,
                        token_mask=mask,
                        fused_mask=False,
                    )
                    actual, gate = _run(f, native=True)
                    self.assertIs(gate.call_args.kwargs["token_mask"], mask)
                    self.assertIs(gate.call_args.kwargs["fused_mask"], True)
                    assert_frozen_gate(self, actual, expected)
                    assert_bits(self, actual[~mask], expected[~mask])
                    for value, before in zip(
                        (f.hidden, f.module.q_weight, f.module.k_weight, f.value, mask),
                        frozen,
                    ):
                        assert_bits(self, value, before)
                    if f.cp.chunk_length < 100:
                        assert_frozen_gate(
                            self,
                            actual,
                            cpu_oracle(
                                f.hidden,
                                f.module.q_weight,
                                f.module.k_weight,
                                f.value,
                                mask,
                            ),
                        )

    @torch.no_grad()
    def test_false_mask_ieee_cells_preserve_existing_aten_result_not_a_copy_shortcut(
        self,
    ):
        with patch.dict(os.environ, FLAGS):
            f = _fixture(lengths=(16,) * 8, prefixes=(0,) * 8, device="cuda")
            f.unit.engram_token_mask.zero_()
            f.hidden.zero_()
            f.hidden[0].fill_(-0.0)
            f.hidden[1].fill_(0.25)
            f.hidden[2].fill_(0.5)
            f.value[:, 20480:] = 1
            f.value[1, 20480:] = torch.inf
            f.value[2, 20480:] = torch.nan
            expected = engram.engram_gate_inject(
                f.hidden,
                f.module.q_weight,
                f.module.k_weight,
                f.value,
                token_mask=f.unit.engram_token_mask,
                fused_mask=False,
            )
            actual, _ = _run(f, native=True)
            self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))
            self.assertTrue(
                torch.equal(torch.isposinf(actual), torch.isposinf(expected))
            )
            self.assertTrue(
                torch.equal(torch.isneginf(actual), torch.isneginf(expected))
            )
            finite = torch.isfinite(expected)
            assert_bits(self, actual[finite], expected[finite])
            self.assertTrue(torch.isnan(actual[1:3]).all())


if __name__ == "__main__":
    unittest.main()
