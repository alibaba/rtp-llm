"""Exercise production mega-attention call sites without launching GPU kernels."""

import sys
import unittest
from contextlib import nullcontext
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import attention_v41 as attention
from rtp_llm.models_py.modules.dsv41.fp8.decode import compute_qkv, rope_metadata


def _owner():
    owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
    torch.nn.Module.__init__(owner)
    owner.layer_id = 0
    owner._shared_attention = {"layers": {0: owner}}
    owner.dim, owner.n_heads, owner.head_dim, owner.rope_head_dim = 3, 2, 4, 2
    owner.wq_b, owner.wo_b = object(), object()
    owner.softmax_scale = 0.5
    owner.attn_sink = torch.zeros(2)
    owner.is_index_source = False
    owner._qb_weight_owner = None
    owner._mega_attention = None
    owner._prefill_output_all_reduce = Mock()
    return owner


def _workspace():
    def metadata(freqs, indices, lengths, rows, chunk):
        return (
            freqs.split(chunk),
            indices.split(chunk),
            (
                lengths.split(chunk)
                if lengths is not None
                else (None,) * len(freqs.split(chunk))
            ),
        )

    return SimpleNamespace(
        attention_metadata=metadata,
        prefill_q_head_views=lambda rows, heads, dim, rope: (
            torch.empty(rows, heads * dim, dtype=torch.bfloat16),
            None,
            None,
        ),
    )


class MegaAttentionWiringTest(unittest.TestCase):
    def test_prepare_runs_after_authoritative_rope_placement(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_mega_attention

        owner = _owner()
        owner._mega_attention = None
        ready = SimpleNamespace()
        order = []
        with patch.object(
            attention.AttentionFP8,
            "init_rope_cache",
            side_effect=lambda **kwargs: order.append("rope"),
        ), patch.object(
            _v41_mega_attention,
            "prepare",
            side_effect=lambda *args, **kwargs: (order.append("prepare"), ready)[1],
        ) as prepare:
            owner.init_rope_cache(device="cuda:0")
        self.assertEqual(order, ["rope", "prepare"])
        prepare.assert_called_once_with(owner, shared=owner._shared_attention)
        self.assertIs(owner._mega_attention, ready)

    def test_unavailable_mega_skips_merged_qb_and_keeps_weight_owner(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_mega_attention
        from rtp_llm.models_py.modules.dsv41.fp8._v41_merged_qb import MergedQB

        owner = _owner()
        weights = object()
        owner._qb_weight_owner = weights
        with patch.object(attention.AttentionFP8, "init_rope_cache"), patch.object(
            _v41_mega_attention, "prepare", return_value=None
        ), patch.object(MergedQB, "create") as merged:
            owner.init_rope_cache(device="cuda:0")
        self.assertIsNone(owner._mega_attention)
        self.assertIs(owner._qb_weight_owner, weights)
        merged.assert_not_called()

    def test_reinitialization_reloads_captured_buffers_in_place(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_mega_attention

        owner = _owner()
        original = SimpleNamespace(reload_weights=Mock())
        with patch.object(attention.AttentionFP8, "init_rope_cache"), patch.object(
            _v41_mega_attention, "prepare", return_value=original
        ) as prepare:
            owner.init_rope_cache(device="cuda:0")
            self.assertIs(owner._mega_attention, original)
            owner.init_rope_cache(device="cuda:0")
            self.assertIs(owner._mega_attention, original)
        self.assertEqual(prepare.call_count, 1)
        original.reload_weights.assert_called_once_with(owner)

    def test_prefill_chunks_use_absolute_positions_for_cp_and_non_cp(self):
        for cp in (False, True):
            with self.subTest(cp=cp):
                owner = _owner()
                rows = 8
                ordinary = torch.arange(28672, 28672 + rows)
                zigzag = torch.tensor([28672, 28673, 32766, 32767, 12, 13, 30, 31])
                expected = zigzag if cp else ordinary
                indices = torch.arange(rows * 128).reshape(rows, 1, 128).int()
                lengths = torch.arange(1, rows + 1).int()
                kv = torch.zeros(32, 1, 4, dtype=torch.bfloat16)
                qr = torch.arange(rows * 8).reshape(rows, 8).to(torch.bfloat16)
                common = SimpleNamespace(
                    workspace=_workspace(),
                    freqs_cis=torch.arange(rows * 2).reshape(rows, 2),
                    position_ids=ordinary,
                    cp_on=cp,
                    cp_ctx=SimpleNamespace(global_positions=zigzag),
                )
                mega = SimpleNamespace(
                    supports_query=Mock(return_value=True),
                    supports_prefill=Mock(return_value=True),
                    project_q=Mock(
                        side_effect=lambda linear, q, out: out.copy_(q).view(-1, 2, 4)
                    ),
                    prefill=Mock(
                        side_effect=lambda q, pool, idx, pos, lens: pos[:, None]
                        .expand(-1, 3)
                        .to(torch.bfloat16)
                    ),
                    project_output=Mock(
                        side_effect=lambda pair, linear, out: out.copy_(pair)
                    ),
                )
                owner._mega_attention = mega
                owner._project_prefill_q = Mock(
                    side_effect=AssertionError("separate Q/RoPE")
                )
                owner._prefill_output_proj_into = Mock(
                    side_effect=AssertionError("separate output projection")
                )
                legacy = Mock(side_effect=AssertionError("legacy attention"))
                with patch.object(
                    attention, "_FLASH_MLA_SPARSE_Q_CHUNK", 3
                ), patch.dict(
                    sys.modules,
                    {
                        "rtp_llm.models_py.modules.dsv41.flash_mla": SimpleNamespace(
                            flash_mla_sparse_fwd=legacy
                        )
                    },
                ):
                    result = owner._prefill_sparse_attention(
                        SimpleNamespace(qr=qr),
                        common,
                        kv=kv,
                        indices=indices,
                        topk_length=lengths,
                        profile_name="legacy",
                    )
                self.assertEqual(mega.prefill.call_count, 3)
                self.assertEqual(
                    [c.args[0].shape[0] for c in mega.prefill.call_args_list], [3, 3, 2]
                )
                for chunk, call in enumerate(mega.prefill.call_args_list):
                    start, end = chunk * 3, min(chunk * 3 + 3, rows)
                    self.assertIs(call.args[1], kv)
                    torch.testing.assert_close(call.args[2], indices[start:end])
                    torch.testing.assert_close(call.args[3], expected[start:end].int())
                    torch.testing.assert_close(call.args[4], lengths[start:end])
                torch.testing.assert_close(
                    result, expected[:, None].expand(-1, 3).to(torch.bfloat16)
                )
                owner._prefill_output_all_reduce.assert_called_once_with(result)

    def test_unsupported_prefill_preserves_original_q_rope_attention_and_projection(
        self,
    ):
        for rejected in ("query", "kv", "disabled"):
            with self.subTest(rejected=rejected):
                owner = _owner()
                mega = SimpleNamespace(
                    supports_query=Mock(return_value=rejected != "query"),
                    supports_prefill=Mock(return_value=rejected != "kv"),
                    project_q=Mock(side_effect=AssertionError("raw Q in fallback")),
                    prefill=Mock(side_effect=AssertionError("mega in fallback")),
                )
                owner._mega_attention = None if rejected == "disabled" else mega
                qr = torch.arange(56).reshape(7, 8).to(torch.bfloat16)
                freqs = torch.arange(14).reshape(7, 2)
                indices = torch.zeros(7, 1, 128, dtype=torch.int32)
                lengths = torch.ones(7, dtype=torch.int32)
                kv = torch.ones(16, 1, 4, dtype=torch.bfloat16)
                common = SimpleNamespace(workspace=_workspace(), freqs_cis=freqs)
                projected = []

                def legacy_q(q, freq, workspace):
                    projected.append((q.clone(), freq.clone()))
                    return q.view(-1, 2, 4) + 2

                owner._project_prefill_q = Mock(side_effect=legacy_q)
                owner._prefill_output_proj_into = Mock(
                    side_effect=lambda o, freq, out, workspace: out.copy_(
                        o[:, 0, :3] + 5
                    )
                )
                native = Mock(
                    side_effect=lambda **kwargs: (kwargs["q"] + 3, None, None)
                )
                with patch.object(
                    attention, "_FLASH_MLA_SPARSE_Q_CHUNK", 3
                ), patch.object(attention, "dispose_tensor"), patch.dict(
                    sys.modules,
                    {
                        "rtp_llm.models_py.modules.dsv41.flash_mla": SimpleNamespace(
                            flash_mla_sparse_fwd=native
                        )
                    },
                ):
                    actual = owner._prefill_sparse_attention(
                        SimpleNamespace(qr=qr),
                        common,
                        kv=kv,
                        indices=indices,
                        topk_length=lengths,
                        profile_name="legacy",
                    )
                torch.testing.assert_close(actual, qr.view(7, 2, 4)[:, 0, :3] + 10)
                self.assertEqual(native.call_count, 3)
                for chunk, (q, freq) in enumerate(projected):
                    torch.testing.assert_close(q, qr[chunk * 3 : chunk * 3 + 3])
                    torch.testing.assert_close(freq, freqs[chunk * 3 : chunk * 3 + 3])
                mega.project_q.assert_not_called()
                mega.prefill.assert_not_called()

    def test_position_cache_shared_by_layers_and_invalidated_for_reused_input_buffer(
        self,
    ):
        owner = _owner()
        other = _owner()
        other.layer_id = 1
        other._shared_attention = owner._shared_attention
        owner._shared_attention["layers"][1] = other
        positions = torch.tensor([100, 101, 500, 501, 0, 0])
        first = owner._mega_position_chunks(positions, 6, 4)
        second = other._mega_position_chunks(positions, 6, 4)
        self.assertIs(first, second)
        self.assertEqual([x.numel() for x in first], [4, 2])
        other._begin_forward()
        self.assertIs(other._mega_position_chunks(positions, 6, 4), first)
        positions.copy_(torch.tensor([8, 9, 28, 29, 30, 0]))
        owner._begin_forward()
        third = owner._mega_position_chunks(positions, 6, 4)
        self.assertIsNot(third, first)
        torch.testing.assert_close(torch.cat(third), positions.int())
        with self.assertRaisesRegex(ValueError, "one absolute position"):
            owner._mega_position_chunks(positions, 7, 4)

    def test_decode_admits_raw_q_only_after_complete_mega_support_checks(self):
        class StopAfterQKV(Exception):
            pass

        for reject in (None, "disabled", "input", "swa", "global_pool", "topk"):
            with self.subTest(reject=reject):
                owner = _owner()
                owner.compress_ratio = 2
                owner.is_kv_source = False
                owner.index_topk = 127 if reject == "topk" else 128
                owner.freqs_cis = torch.empty(128, 2)
                pool = torch.zeros(2, 2, 528, dtype=torch.uint8)
                global_pool = torch.zeros(2, 2, 288, dtype=torch.uint8)
                owner._pool_view_3d_fp8 = Mock(return_value=pool)
                owner._source_pool = Mock(return_value=global_pool)
                mega = SimpleNamespace(
                    supports_input=Mock(return_value=reject != "input"),
                    supports_decode=Mock(return_value=reject != "swa"),
                    supports_pool=Mock(return_value=reject != "global_pool"),
                )
                owner._mega_attention = None if reject == "disabled" else mega
                metadata = SimpleNamespace(
                    position_ids=torch.tensor([10, 11, 20, 21, 999]),
                    swa_global_slots=torch.zeros(5, 128, dtype=torch.int32),
                    pool_write_slot_mappings={attention.SWA_KV: torch.arange(5)},
                )
                with patch.object(
                    rope_metadata,
                    "decode_rope_metadata",
                    return_value=(metadata.position_ids[:4], torch.zeros(4, 2)),
                ), patch.object(
                    compute_qkv, "decode_compute_qkv", side_effect=StopAfterQKV
                ) as project:
                    with self.assertRaises(StopAfterQKV):
                        owner._forward_decode_body(torch.zeros(2, 2, 3), metadata)
                self.assertIs(
                    project.call_args.kwargs.get("mega_attention"),
                    mega if reject is None else None,
                )
                torch.testing.assert_close(
                    project.call_args.kwargs["swa_slots"], torch.arange(4)
                )

    def test_decode_qkv_keeps_raw_q_only_for_mega_and_still_rotates_kv(self):
        from rtp_llm.models_py.modules.dsv41 import _v41_query_quant
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_qkv_rope_cache_triton

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                qr = torch.ones(2, 2, 8, dtype=torch.bfloat16)
                raw_q = (
                    torch.arange(4 * 2 * 128).reshape(2, 2, 2, 128).to(torch.bfloat16)
                )
                raw_kv = torch.ones(2, 2, 128, dtype=torch.bfloat16)
                owner = SimpleNamespace(
                    rope_head_dim=64,
                    n_heads=2,
                    head_dim=128,
                    eps=1e-6,
                    skip_post_q_norm=True,
                    wq_b=object(),
                    kv_norm=torch.ones(128),
                    freqs_cis=torch.empty(128, 32),
                    _try_fused_qr_kv=Mock(return_value=(qr, raw_kv)),
                    _lin=Mock(return_value=raw_q.flatten(-2).clone()),
                )
                mega = SimpleNamespace(
                    project_q=Mock(return_value=raw_q.clone()), padded_heads=64
                )
                with patch.object(
                    _v41_query_quant, "try_project_quantized_qkv", return_value=None
                ), patch.object(
                    _v41_qkv_rope_cache_triton,
                    "try_fused_qkv_rope_cache",
                    return_value=None,
                ) as fused_cache, patch.object(
                    compute_qkv,
                    "_apply_v41_q_rope",
                    side_effect=lambda q, freq, dim: q.add_(2),
                ) as q_rope, patch.object(
                    compute_qkv,
                    "fused_rmsnorm_rope",
                    side_effect=lambda value, *args, **kwargs: value + 7,
                ) as kv_rope:
                    result = compute_qkv.decode_compute_qkv(
                        owner,
                        torch.ones(2, 2, 3),
                        torch.arange(4),
                        freqs_cis=torch.empty(4, 32),
                        swa_pool=torch.empty(2, 2, 528, dtype=torch.uint8),
                        swa_slots=torch.arange(4),
                        mega_attention=mega if enabled else None,
                    )
                torch.testing.assert_close(result.q, raw_q if enabled else raw_q + 2)
                torch.testing.assert_close(result.kv, raw_kv + 7)
                self.assertEqual(q_rope.call_count, 0 if enabled else 1)
                self.assertEqual(kv_rope.call_count, 1)
                if enabled:
                    self.assertIs(fused_cache.call_args.kwargs["apply_q_rope"], False)
                    self.assertEqual(
                        fused_cache.call_args.kwargs["pad_query_heads"], 64
                    )
                else:
                    self.assertNotIn("apply_q_rope", fused_cache.call_args.kwargs)

    def test_decode_executes_selected_path_and_preserves_swa_cache_write(self):
        for supported, swa_written in product((False, True), repeat=2):
            with self.subTest(supported=supported, swa_written=swa_written):
                owner = _owner()
                owner.compress_ratio = 0
                owner.freqs_cis = torch.empty(128, 2)
                pool = torch.zeros(2, 2, 528, dtype=torch.uint8)
                owner._pool_view_3d_fp8 = Mock(return_value=pool)
                owner._decode_write_swa_fp8 = Mock()
                owner._forward_decode_swa_only = Mock(
                    return_value=torch.ones(2, 3, 2, 4)
                )
                owner._project_output = Mock(return_value=torch.full((6, 3), -1.0))
                mega = SimpleNamespace(
                    supports_input=Mock(return_value=True),
                    supports_decode=Mock(return_value=supported),
                    decode=Mock(return_value=torch.full((6, 3), 2.0)),
                    project_output=Mock(side_effect=lambda pair, linear: pair),
                )
                owner._mega_attention = mega
                raw_q = torch.zeros(2, 3, 2, 4)
                kv = torch.ones(2, 3, 4)
                positions = torch.tensor([10, 11, 12, 20, 21, 22, 999])
                metadata = SimpleNamespace(
                    position_ids=positions,
                    swa_global_slots=torch.zeros(7, 128, dtype=torch.int32),
                    pool_write_slot_mappings={attention.SWA_KV: torch.arange(7)},
                )
                qkv = compute_qkv.DecodeQKV(
                    torch.ones(2, 3, 8),
                    raw_q,
                    kv,
                    torch.zeros(6, 2),
                    swa_written=swa_written,
                )
                with patch.object(
                    rope_metadata,
                    "decode_rope_metadata",
                    return_value=(positions[:6], qkv.freqs_cis),
                ), patch.object(compute_qkv, "decode_compute_qkv", return_value=qkv):
                    result = owner._forward_decode_body(torch.zeros(2, 3, 3), metadata)
                if swa_written:
                    owner._decode_write_swa_fp8.assert_not_called()
                else:
                    owner._decode_write_swa_fp8.assert_called_once_with(
                        kv, 2, 3, metadata
                    )
                self.assertEqual(result.shape, (2, 3, 3))
                torch.testing.assert_close(
                    result, torch.full_like(result, 2 if supported else -1)
                )
                self.assertEqual(owner._prefill_output_all_reduce.call_count, 1)
                if not supported:
                    mega.decode.assert_not_called()
                    owner._forward_decode_swa_only.assert_called_once_with(
                        raw_q, 2, 3, metadata
                    )
                    owner._project_output.assert_called_once_with(
                        owner._forward_decode_swa_only.return_value, qkv.freqs_cis
                    )
                    continue
                self.assertIs(mega.decode.call_args.args[0], raw_q)
                self.assertIs(mega.decode.call_args.args[1], pool)
                torch.testing.assert_close(
                    mega.decode.call_args.args[3], positions[:6].int()
                )
                self.assertIsNone(mega.decode.call_args.kwargs["extra_pool"])
                self.assertIsNone(mega.decode.call_args.kwargs["extra_indices"])
                owner._forward_decode_swa_only.assert_not_called()
                owner._project_output.assert_not_called()

    def test_global_decode_fallback_preserves_selected_slots_and_projection(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_fp4_decode_attn
        from rtp_llm.models_py.modules.dsv41.fp8.decode import decode_attn_metadata

        owner = _owner()
        owner.compress_ratio, owner.index_topk, owner.window_size = 1, 128, 128
        owner.is_kv_source = False
        owner.freqs_cis = torch.empty(128, 2)
        pool = torch.zeros(2, 32, 528, dtype=torch.uint8)
        extra = torch.zeros(2, 128, 288, dtype=torch.uint8)
        owner._pool_view_3d_fp8 = Mock(return_value=pool)
        owner._source_pool = Mock(return_value=extra)
        owner._decode_write_swa_fp8 = Mock()
        owner._global_region = Mock(return_value="global_kv_1")
        owner._get_fp8_decode_op = Mock(return_value=object())
        selected = torch.arange(128).repeat(6, 1).int()
        slots = selected.long() + 128
        owner._select_indices_decode = Mock(return_value=selected)
        owner._decode_global_slots = Mock(return_value=slots)
        owner._project_output = Mock(return_value=torch.ones(6, 3))
        positions = torch.arange(6)
        metadata = SimpleNamespace(
            position_ids=positions,
            swa_global_slots=torch.zeros(6, 128, dtype=torch.int32),
            req_id_per_token=torch.arange(2).repeat_interleave(3).int(),
            pool_write_slot_mappings={},
            pool_block_tables={attention.SWA_KV: torch.ones(2, 1).int()},
        )
        qkv = compute_qkv.DecodeQKV(
            torch.ones(2, 3, 8),
            torch.zeros(2, 3, 2, 4),
            torch.ones(2, 3, 4),
            torch.zeros(6, 2),
        )
        with patch.object(
            rope_metadata,
            "decode_rope_metadata",
            return_value=(positions, qkv.freqs_cis),
        ), patch.object(
            compute_qkv, "decode_compute_qkv", return_value=qkv
        ) as project, patch.object(
            decode_attn_metadata, "get_or_build_sched_meta", return_value=object()
        ), patch.object(
            _v41_fp4_decode_attn, "fp4_dual_decode_attention", return_value=qkv.q
        ) as native:
            result = owner._forward_decode_body(torch.zeros(2, 3, 3), metadata)
        self.assertNotIn("mega_attention", project.call_args.kwargs)
        self.assertIs(native.call_args.kwargs["swa_pool_3d"], pool)
        self.assertIs(native.call_args.kwargs["global_pool_3d"], extra)
        torch.testing.assert_close(
            native.call_args.kwargs["global_topk_3d"], slots.int().view(2, 3, 128)
        )
        owner._project_output.assert_called_once_with(qkv.q, qkv.freqs_cis)
        torch.testing.assert_close(result, torch.ones(2, 3, 3))


class DSparkMegaWiringTest(unittest.TestCase):
    def test_selects_layout_before_projection_and_preserves_noncausal_metadata(self):
        from rtp_llm.models_py.model_desc import (
            deepseek_v41_base_dspark_model as dspark,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_mega_attention import (
            _state,
        )

        for rejection in (None, "disabled", "input", "pool", "indices"):
            with self.subTest(rejection=rejection):
                batch, gamma, rows, dim = 2, 5, 10, 3
                state = _state()
                width = 127 if rejection == "indices" else 128
                dtype = torch.float32 if rejection == "input" else torch.bfloat16
                x = torch.ones(batch, gamma, dim, dtype=dtype)
                pool = torch.zeros(2, 32, 528, dtype=torch.uint8)
                if rejection == "pool":
                    pool = torch.empty(2, 17408, dtype=torch.uint8).as_strided(
                        (2, 32, 528), (17408, 528, 1)
                    )
                slots = torch.arange(rows)
                indices = torch.arange(width).repeat(rows, 1).int()
                lengths = torch.tensor([15, 0], dtype=torch.int32)
                positions = torch.arange(rows).view(batch, gamma) + 10
                qkv = SimpleNamespace(q=object(), kv=object(), freqs_cis=object())
                output = torch.arange(rows * dim).reshape(rows, dim).bfloat16()
                state.decode = Mock(return_value=object())
                state.project_output = Mock(return_value=output)
                legacy = SimpleNamespace(forward=Mock(return_value=object()))
                attn = SimpleNamespace(
                    _mega_attention=None if rejection == "disabled" else state,
                    n_heads=16,
                    head_dim=512,
                    wo_b=object(),
                    attn_sink=object(),
                    _get_fp8_decode_op=Mock(return_value=legacy),
                    _project_output=Mock(return_value=output),
                    _prefill_output_all_reduce=Mock(),
                )
                model = SimpleNamespace(
                    _swa_cache_bound=lambda *args: nullcontext((attn, 32, pool)),
                    _proposal_metadata=Mock(return_value=(slots, indices, lengths)),
                )
                scheduler = object()
                with patch.object(
                    dspark, "decode_compute_qkv", return_value=qkv
                ) as project, patch.object(
                    dspark, "_write_dspark_swa"
                ) as write, patch.object(
                    dspark, "get_or_build_sched_meta", return_value=scheduler
                ) as schedule:
                    result = dspark.DeepSeekV4DSparkModel._forward_dspark_attention(
                        model,
                        0,
                        x,
                        positions,
                        lengths,
                        lengths > 0,
                        None,
                        32,
                        object(),
                        {},
                    )
                torch.testing.assert_close(result, output.view(batch, gamma, dim))
                self.assertIs(
                    project.call_args.kwargs.get("mega_attention"),
                    state if rejection is None else None,
                )
                self.assertIs(write.call_args.kwargs["kv"], qkv.kv)
                self.assertIs(write.call_args.kwargs["slot_mapping"], slots)
                self.assertIs(write.call_args.kwargs["swa_pool_3d"], pool)
                attn._prefill_output_all_reduce.assert_called_once_with(output)
                if rejection is None:
                    legacy.forward.assert_not_called()
                    schedule.assert_not_called()
                    attn._project_output.assert_not_called()
                    torch.testing.assert_close(state.decode.call_args.args[2], indices)
                    torch.testing.assert_close(
                        state.decode.call_args.args[3], positions.flatten().int()
                    )
                    torch.testing.assert_close(
                        state.decode.call_args.kwargs["swa_lengths"],
                        lengths.repeat_interleave(gamma),
                    )
                else:
                    state.decode.assert_not_called()
                    self.assertIs(legacy.forward.call_args.kwargs["q"], qkv.q)
                    self.assertIs(legacy.forward.call_args.kwargs["kv_cache"], pool)
                    self.assertIs(
                        legacy.forward.call_args.kwargs["topk_length"], lengths
                    )
                    self.assertIs(
                        legacy.forward.call_args.kwargs["sched_meta"], scheduler
                    )
                    torch.testing.assert_close(
                        legacy.forward.call_args.kwargs["topk_idxs"],
                        indices.view(batch, gamma, width),
                    )
                    attn._project_output.assert_called_once_with(
                        legacy.forward.return_value, qkv.freqs_cis
                    )

    def test_mega_execution_failure_does_not_retry_with_split_attention(self):
        from rtp_llm.models_py.model_desc import (
            deepseek_v41_base_dspark_model as dspark,
        )

        mega = SimpleNamespace(
            supports_input=Mock(return_value=True),
            supports_decode=Mock(return_value=True),
            decode=Mock(side_effect=RuntimeError("mega launch failed")),
        )
        attn = SimpleNamespace(
            _mega_attention=mega, head_dim=512, _get_fp8_decode_op=Mock()
        )
        pool = torch.zeros(1, 32, 528, dtype=torch.uint8)
        positions = torch.arange(5).view(1, 5)
        model = SimpleNamespace(
            _swa_cache_bound=lambda *args: nullcontext((attn, 32, pool)),
            _proposal_metadata=lambda *args: (
                positions.flatten(),
                torch.zeros(5, 128).int(),
                torch.tensor([5]).int(),
            ),
        )
        with patch.object(
            dspark,
            "decode_compute_qkv",
            return_value=SimpleNamespace(q=object(), kv=object()),
        ), patch.object(dspark, "_write_dspark_swa"), self.assertRaisesRegex(
            RuntimeError, "mega launch failed"
        ):
            dspark.DeepSeekV4DSparkModel._forward_dspark_attention(
                model,
                0,
                torch.ones(1, 5, 3).bfloat16(),
                positions,
                None,
                None,
                None,
                32,
                None,
                {},
            )
        attn._get_fp8_decode_op.assert_not_called()


if __name__ == "__main__":
    unittest.main()
