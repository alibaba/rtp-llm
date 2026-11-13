"""RoPE ownership/lookup contracts and explicitly enabled PPU byte/replay tests."""

import ast
import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import fused_ops, graph_swa_write, rope


def reference(value, frequencies, inverse=False, positions=None, valid=None):
    """Test-only complex arithmetic over the original immutable inputs."""
    if positions is not None:
        frequencies = frequencies[positions.long()]
    if valid is not None:
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        value = torch.where(valid.reshape(shape), value, torch.zeros_like(value))
    width = frequencies.shape[1] * 2
    real = value[..., -width::2].float()
    imag = value[..., -width + 1 :: 2].float()
    frequency_shape = (value.shape[0],) + (1,) * (value.ndim - 2) + (-1,)
    phase = frequencies.reshape(frequency_shape)
    rotated = torch.complex(real, imag) * (phase.conj() if inverse else phase)
    out = value.clone()
    out[..., -width::2] = rotated.real.to(value.dtype)
    out[..., -width + 1 :: 2] = rotated.imag.to(value.dtype)
    return out


def assert_bits(case, left, right):
    case.assertEqual((left.shape, left.dtype), (right.shape, right.dtype))
    case.assertTrue(
        torch.equal(
            left.resolve_conj().resolve_neg().contiguous().view(torch.uint8),
            right.resolve_conj().resolve_neg().contiguous().view(torch.uint8),
        )
    )


def inputs(rows=3, heads=None, dimension=128, *, device="cpu"):
    shape = (rows, dimension) if heads is None else (rows, heads, dimension)
    value = torch.randn(shape, device=device, dtype=torch.bfloat16)
    phase = torch.randn(rows, 32, device=device)
    return value, torch.polar(torch.ones_like(phase), phase).contiguous()


@contextmanager
def providers():
    """Real Torch storage/metadata, doubles only at native launch boundaries."""

    def inplace(
        value,
        frequencies,
        *,
        inverse=False,
        valid_mask=None,
        preserve_signed_zero=True,
        positions=None
    ):
        value.copy_(reference(value, frequencies, inverse, positions, valid_mask))
        return value

    with patch.object(rope, "_native_input", return_value=True), patch.object(
        graph_swa_write, "try_rope_", side_effect=inplace
    ) as overwrite, patch.object(
        fused_ops, "try_rope_tail", side_effect=reference
    ) as fresh:
        yield overwrite, fresh


class RoPEDispatchTest(unittest.TestCase):
    @torch.no_grad()
    def test_portable_known_quarter_turn_and_inverse_keep_prefix_and_inputs(self):
        for heads in (None, 2):
            value, frequencies = inputs(heads=heads)
            value[..., -64::2], value[..., -63::2] = 2, 3
            frequencies.fill_(1j)
            saved = value.clone()
            for inverse, pair in ((False, (-3, 2)), (True, (3, -2))):
                actual = rope.rotate(value, frequencies, inverse)
                self.assertNotEqual(
                    actual.untyped_storage().data_ptr(),
                    value.untyped_storage().data_ptr(),
                )
                self.assertTrue(torch.all(actual[..., -64::2] == pair[0]))
                self.assertTrue(torch.all(actual[..., -63::2] == pair[1]))
                assert_bits(self, actual[..., :-64], saved[..., :-64])
                assert_bits(self, value, saved)

    @torch.no_grad()
    def test_owned_offset_view_is_modified_without_touching_canaries(self):
        backing = torch.randn(5, 128, dtype=torch.bfloat16)
        value, frequencies = backing[1:4], inputs()[1]
        saved = backing.clone()
        with providers() as (overwrite, fresh):
            output = rope.rotate(value, frequencies, inplace=True)
            self.assertIs(output, value)
            self.assertEqual(output.storage_offset(), 128)
            overwrite.assert_called_once()
            self.assertTrue(overwrite.call_args.kwargs["preserve_signed_zero"])
            fresh.assert_not_called()
        assert_bits(self, value, reference(saved[1:4], frequencies))
        assert_bits(self, backing[0], saved[0])
        assert_bits(self, backing[4], saved[4])

    @torch.no_grad()
    def test_same_storage_live_aliases_decline_before_any_write(self):
        backing = torch.randn(6, 128, dtype=torch.bfloat16)
        value, frequencies = backing[:3], inputs()[1]
        for alias in (value, value.view_as(value), backing[3:]):
            saved = backing.clone()
            with providers() as (overwrite, fresh):
                output = rope.rotate(
                    value,
                    frequencies,
                    inplace=True,
                    live_inputs=(alias,),
                    allow_fused=True,
                )
                overwrite.assert_not_called()
                fresh.assert_called_once()
                self.assertNotEqual(
                    output.untyped_storage().data_ptr(),
                    backing.untyped_storage().data_ptr(),
                )
            assert_bits(self, backing, saved)
            assert_bits(self, output, reference(value, frequencies))

    @torch.no_grad()
    def test_frequency_storage_is_always_protected(self):
        value = torch.zeros(3, 128, dtype=torch.bfloat16)
        frequencies = value.view(torch.complex64).reshape(3, 32)
        frequencies.fill_(1 + 0j)
        saved = value.clone()
        with providers() as (overwrite, _):
            output = rope.rotate(value, frequencies, inplace=True, allow_fused=True)
            overwrite.assert_not_called()
        assert_bits(self, value, saved)
        assert_bits(self, output, reference(saved, frequencies))

    @torch.no_grad()
    def test_unknown_layout_and_lazy_views_cannot_grant_overwrite(self):
        value, frequencies = inputs()
        self.assertFalse(rope.can_overwrite(value, value.to_sparse()))
        for source, table in (
            (torch._neg_view(value), frequencies),
            (value.float()[:, ::2], frequencies[:, :16]),
            (value, frequencies.conj()),
        ):
            before = source.clone()
            output = rope.rotate(source, table, inplace=True, allow_fused=True)
            assert_bits(self, output, reference(before, table))
            assert_bits(self, source, before)

    @torch.no_grad()
    def test_masked_alias_gets_fresh_owned_storage_and_preserves_original(self):
        value, frequencies = inputs(heads=2)
        valid = torch.tensor([True, False, True])
        saved = value.clone()
        with providers() as (overwrite, _):
            output = rope.rotate(
                value,
                frequencies,
                inplace=True,
                live_inputs=(value.view_as(value),),
                valid_mask=valid,
            )
            overwrite.assert_called_once()
            self.assertNotEqual(output.data_ptr(), value.data_ptr())
        assert_bits(self, value, saved)
        assert_bits(self, output, reference(saved, frequencies, valid=valid))

    @torch.no_grad()
    def test_direct_lookup_uses_original_positions_and_table(self):
        value, _ = inputs()
        table = inputs(29)[1]
        positions = torch.tensor([17, 2, 23], dtype=torch.int64)
        for inplace in (False, True):
            work = value.clone()
            with providers() as (overwrite, fresh):
                output = rope.rotate(
                    work, table, positions=positions, inplace=inplace, allow_fused=True
                )
                operation = overwrite if inplace else fresh
                self.assertIs(operation.call_args.args[1], table)
                self.assertIs(operation.call_args.kwargs["positions"], positions)
            assert_bits(self, output, reference(value, table, positions=positions))

    @torch.no_grad()
    def test_declined_direct_lookup_gathers_current_positions_each_call(self):
        value, _ = inputs()
        table = inputs(29)[1]
        positions = torch.tensor([17, 2, 23], dtype=torch.int32)
        for inverse in (False, True):
            for inplace in (False, True):
                for ids in ([17, 2, 23], [1, 18, 0]):
                    positions.copy_(torch.tensor(ids, dtype=positions.dtype))
                    with providers() as (overwrite, fresh):
                        original = (
                            overwrite.side_effect if inplace else fresh.side_effect
                        )

                        def decline_direct(*args, **kwargs):
                            return (
                                None
                                if kwargs.get("positions") is not None
                                else original(*args, **kwargs)
                            )

                        operation = overwrite if inplace else fresh
                        operation.side_effect = decline_direct
                        if inplace:
                            fresh.return_value = None
                            fresh.side_effect = lambda *args, **kwargs: None
                        output = rope.rotate(
                            value.clone(),
                            table,
                            inverse,
                            positions=positions,
                            inplace=inplace,
                            allow_fused=True,
                        )
                    assert_bits(
                        self, output, reference(value, table, inverse, positions)
                    )

    @torch.no_grad()
    def test_joint_qk_outputs_are_disjoint_and_inputs_are_read_only(self):
        q, frequencies = inputs(heads=2, dimension=512)
        kv = torch.randn(3, 512, dtype=torch.bfloat16)
        saved = q.clone(), kv.clone()
        with patch.object(rope, "_native_input", return_value=True), patch.object(
            fused_ops, "is_supported", return_value=True
        ), patch.object(
            fused_ops,
            "qk_rope",
            side_effect=lambda q, kv, f: (reference(q, f), reference(kv, f)),
        ) as joint:
            q_out, kv_out = rope.rotate_qk(q, kv, frequencies)
            joint.assert_called_once_with(q, kv, frequencies)
        self.assertEqual(
            len({v.untyped_storage().data_ptr() for v in (q, kv, q_out, kv_out)}), 4
        )
        assert_bits(self, q, saved[0])
        assert_bits(self, kv, saved[1])

    @torch.no_grad()
    def test_provider_errors_propagate_without_retry(self):
        value, frequencies = inputs()
        with providers() as (overwrite, fresh):
            overwrite.side_effect = RuntimeError("owned launch")
            with self.assertRaisesRegex(RuntimeError, "owned launch"):
                rope.rotate(value, frequencies, inplace=True, allow_fused=True)
            fresh.assert_not_called()
        with providers() as (_, fresh):
            fresh.side_effect = RuntimeError("fresh launch")
            with self.assertRaisesRegex(RuntimeError, "fresh launch"):
                rope.rotate(value, frequencies, allow_fused=True)

    def test_grad_enabled_keeps_autograd_and_new_storage(self):
        value = torch.randn(3, 128, requires_grad=True)
        frequencies = inputs()[1]
        output = rope.rotate(value, frequencies, inplace=True, allow_fused=True)
        self.assertNotEqual(output.data_ptr(), value.data_ptr())
        output.sum().backward()
        self.assertIsNotNone(value.grad)

    @torch.no_grad()
    def test_wrappers_delegate_to_common_api(self):
        root = Path(__file__).resolve().parents[1]
        value, frequencies = inputs()
        for file, name in (
            ("prefill.py", "_rope_tail"),
            ("graph_attention_layer.py", "_rotate_graph_value"),
        ):
            tree = ast.parse((root / file).read_text())
            node = next(n for n in tree.body if getattr(n, "name", None) == name)
            operation = Mock(return_value=value)
            scope = {"torch": torch, "rotate": operation}
            exec(compile(ast.Module(body=[node], type_ignores=[]), file, "exec"), scope)
            self.assertIs(scope[name](value, frequencies, inverse=True), value)
            self.assertEqual(operation.call_count, 1)
            self.assertIs(operation.call_args.args[0], value)

    @torch.no_grad()
    def test_graph_cache_alias_declines_before_resolving_or_writing_cache(self):
        q, frequencies = inputs(heads=2, dimension=512)
        kv = q[:, 0]
        slots = torch.arange(3, dtype=torch.int64)
        with patch.object(
            graph_swa_write, "_native_pool", side_effect=AssertionError("cache access")
        ):
            self.assertIsNone(
                graph_swa_write.try_rotate_q_and_write_swa(
                    q, kv, frequencies, None, 0, slots
                )
            )
        assert_bits(self, q[:, 0], kv)

    @torch.no_grad()
    def test_graph_cache_output_pool_alias_declines_before_launch(self):
        q, frequencies = inputs(heads=2, dimension=512)
        kv = inputs(dimension=512)[0]
        slots = torch.arange(3, dtype=torch.int64)
        pool = q.view(torch.uint8).reshape(3, -1)
        before = q.clone()
        with patch.object(
            graph_swa_write, "_can_rotate_q_and_kv", return_value=True
        ), patch.object(
            graph_swa_write, "_native_pool", return_value=(pool, 1)
        ), patch.object(
            graph_swa_write,
            "_launch_rotate_q_and_store",
            side_effect=AssertionError("aliased cache launch"),
        ):
            self.assertIsNone(
                graph_swa_write.try_rotate_q_and_write_swa(
                    q, kv, frequencies, None, 0, slots
                )
            )
        assert_bits(self, q, before)

    def test_graph_cache_launch_always_preserves_signed_zero(self):
        class Kernel:
            def __init__(self):
                self.calls = []

            def __getitem__(self, grid):
                return lambda *args, **kwargs: self.calls.append((args, kwargs))

        kernel = Kernel()
        q, frequencies = inputs(heads=2, dimension=512)
        kv = inputs(dimension=512)[0]
        pool = torch.empty(3, 528, dtype=torch.uint8)
        with patch.object(
            graph_swa_write, "_kernels", return_value=(kernel, None, None, None, None)
        ):
            for journal in (False, True):
                graph_swa_write._launch_rotate_q_and_store(
                    q, kv, frequencies, torch.arange(3), pool, None, 1, journal
                )
                self.assertIs(kernel.calls[-1][0][-1], True)
                self.assertIs(kernel.calls[-1][0][-2], journal)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_ROPE_DEVICE") == "1", "explicit PPU device test"
)
class RoPEDeviceTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

        self.assertTrue(is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    @torch.no_grad()
    def test_all_bf16_bits_signed_zero_prefix_and_alias(self):
        value = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
            .reshape(512, 128)
        )
        frequencies = inputs(512, device="cuda")[1]
        parts = torch.view_as_real(frequencies)
        for bits in range(16):
            value[bits, -2:] = torch.tensor(
                [-0.0 if bits & 1 else 0.0, -0.0 if bits & 2 else 0.0],
                device="cuda",
                dtype=torch.bfloat16,
            )
            parts[bits, -1] = torch.tensor(
                [-0.0 if bits & 4 else 0.0, -0.0 if bits & 8 else 0.0], device="cuda"
            )
        for inverse in (False, True):
            expected = reference(value, frequencies, inverse)
            for inplace in (False, True):
                work = value.clone()
                actual = rope.rotate(
                    work, frequencies, inverse, inplace=inplace, allow_fused=True
                )
                assert_bits(self, actual, expected)
                assert_bits(self, actual[..., :-64], value[..., :-64])
            work = value.clone()
            actual = rope.rotate(
                work,
                frequencies,
                inverse,
                inplace=True,
                live_inputs=(work.view_as(work),),
                allow_fused=True,
            )
            assert_bits(self, work, value)
            assert_bits(self, actual, expected)

    @torch.no_grad()
    def test_graph_replay_uses_live_positions_and_valid_rows(self):
        value, _ = inputs(3, heads=2, device="cuda")
        frequencies = inputs(29, device="cuda")[1]
        positions = torch.tensor([17, 2, 23], device="cuda", dtype=torch.int64)
        valid = torch.tensor([True, False, True], device="cuda")
        for inplace in (False, True):
            work = value.clone()

            def operation():
                work.copy_(value)
                return rope.rotate(
                    work,
                    frequencies,
                    inverse=True,
                    positions=positions,
                    inplace=inplace,
                    valid_mask=valid,
                    allow_fused=True,
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    operation()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = operation()
            for ids, mask in (
                ([1, 18, 0], [False, True, True]),
                ([2, 27, 9], [True, True, False]),
            ):
                positions.copy_(torch.tensor(ids, device="cuda"))
                valid.copy_(torch.tensor(mask, device="cuda"))
                value.copy_(torch.randn_like(value))
                graph.replay()
                assert_bits(
                    self, output, reference(value, frequencies, True, positions, valid)
                )

    @torch.no_grad()
    def test_joint_graph_pack_matches_rope_then_codec(self):
        from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv

        q, frequencies = inputs(heads=2, dimension=512, device="cuda")
        kv = inputs(dimension=512, device="cuda")[0]
        saved = kv.clone()
        expected_q = reference(q, frequencies)
        expected = encode_kv(reference(kv, frequencies), pool_tag="swa_kv")
        actual_q, packed = graph_swa_write.try_rotate_q_and_pack_swa(q, kv, frequencies)
        self.assertIs(actual_q, q)
        assert_bits(self, q, expected_q)
        assert_bits(self, packed, expected)
        assert_bits(self, kv, saved)


if __name__ == "__main__":
    unittest.main()
