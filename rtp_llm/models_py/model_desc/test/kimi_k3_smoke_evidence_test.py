"""Check runtime observations against real request metadata and Page-RR slots."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)
from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MlaImpl
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyCacheStoreInputs, PyModelInputs

_MODEL = "rtp_llm.models_py.model_desc.kimi_k3"
_MLA = "rtp_llm.models_py.modules.kimi_k3.mla"
_METADATA = (
    "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata"
)


def _target():
    hidden = torch.zeros((20, 2), dtype=torch.bfloat16)
    return SimpleNamespace(
        kv_cache=None,
        tp_size=4,
        tp_rank=2,
        _forward_layers=mock.Mock(return_value=hidden),
        norm=lambda value: value,
    )


class TargetPrefillObservationTest(unittest.TestCase):
    def test_real_cache_store_binding_is_opaque(self):
        # This is the actual native class, not a namespace that invents fields.
        cache_store = PyCacheStoreInputs()
        with self.assertRaises(AttributeError):
            _ = cache_store.request_id
        with self.assertRaises(AttributeError):
            cache_store.request_id = torch.tensor([31, 47, 59])

    def test_native_model_inputs_do_not_read_cache_store_ids_in_python(self):
        # Request-associated evidence belongs to the C++ successful-return
        # boundary, where GptModelInputs really exposes the context IDs.
        for prefill, verify in ((True, False), (True, True), (False, False)):
            with self.subTest(prefill=prefill, verify=verify):
                model = _target()
                attention = PyAttentionInputs()
                attention.is_prefill = prefill
                attention.is_target_verify = verify
                attention.logical_request_count = 3
                attention.physical_request_count = 4
                attention.logical_token_count = 15
                attention.physical_token_count = 20
                inputs = PyModelInputs(
                    input_ids=torch.arange(20), attention_inputs=attention
                )
                self.assertEqual(inputs.attention_inputs.logical_request_count, 3)
                self.assertIsNone(inputs.attention_inputs.cache_store_inputs)
                with (
                    mock.patch.dict(os.environ, {"RTP_LLM_SMOKE_EVIDENCE": "1"}),
                    mock.patch(
                        _MODEL + ".get_primary_attention_inputs",
                        side_effect=AssertionError(
                            "Python forward must not inspect opaque cache-store IDs"
                        ),
                    ),
                ):
                    result = KimiK3Model._forward_single(model, inputs)
                model._forward_layers.assert_called_once_with(None, inputs, None)
                self.assertEqual(result.hidden_states.shape, (20, 2))


class PagedContractObservationTest(unittest.TestCase):
    def test_contract_uses_actual_shape_and_preserves_ragged_update(self):
        metadata = SimpleNamespace(prepare_decode=mock.Mock())
        impl = SimpleNamespace(
            batch=8,
            tokens=32,
            graph_mode=False,
            fp8_compute=True,
            page_rr_metadata=metadata,
            _paged_workspace=torch.empty(1, dtype=torch.uint8),
        )
        inputs = SimpleNamespace(
            input_lengths=torch.ones(8, dtype=torch.int32),
            physical_token_count=32,
            logical_request_count=7,
            is_target_verify=True,
            is_mtp_draft_update=False,
        )
        with (
            mock.patch(_MLA + ".smoke_evidence_enabled", return_value=True),
            mock.patch(_MLA + ".emit_smoke_event") as emit,
        ):
            KimiK3MlaImpl._prepare_paged(impl, inputs)
            KimiK3MlaImpl._prepare_paged(impl, inputs)
            self.assertEqual(emit.call_count, 1)
            contract = emit.call_args.kwargs
            self.assertEqual(contract["logical_batch"], 7)
            self.assertEqual(contract["physical_batch"], 8)
            self.assertEqual(contract["q"], 4)
            self.assertEqual(contract["physical_tokens"], 32)
            self.assertEqual(contract["operand_dtype"], str(torch.float8_e4m3fn))
            self.assertEqual(contract["projection_dtype"], "bf16")
            self.assertEqual(
                contract["workspace"], hex(impl._paged_workspace.data_ptr())
            )
            inputs.is_target_verify = False
            inputs.is_mtp_draft_update = True
            inputs.physical_token_count = impl.tokens = 13
            KimiK3MlaImpl._prepare_paged(impl, inputs)
        update = emit.call_args.kwargs
        self.assertEqual(emit.call_count, 2)
        self.assertEqual(update["phase"], "mtp_update")
        self.assertEqual(update["physical_tokens"], 13)
        self.assertEqual(update["q"], 0)
        self.assertEqual(update["query_layout"], "ragged")
        self.assertEqual(metadata.prepare_decode.call_count, 3)


class PrefixExecutionObservationTest(unittest.TestCase):
    def test_only_a_dispatched_prefix_call_is_reported(self):
        for fails in (False, True):
            with self.subTest(backend_fails=fails):
                order = []
                output = torch.zeros((4, 1, 128), dtype=torch.bfloat16)
                lse = torch.zeros((4, 1))

                def run_partial(*args, **kwargs):
                    if not kwargs["causal"]:
                        order.append("prefix_call")
                        if fails:
                            raise RuntimeError("prefix backend failed")
                    return output, lse

                impl = SimpleNamespace(
                    token_per_block=4,
                    kv_lora_rank=2,
                    qk_rope_head_dim=1,
                    num_heads=1,
                    v_head_dim=128,
                    _make_kv_b_proj=mock.Mock(return_value=object()),
                    _project_kv=mock.Mock(return_value=(output, output)),
                    _prefix_plan=SimpleNamespace(
                        slices=[SimpleNamespace(owner=0, start=4096, length=8)]
                    ),
                    _prefix_q_offsets=(0, 4),
                    _page_rr_adapter=SimpleNamespace(
                        build_prefix_chunk_descriptor=mock.Mock(return_value=object()),
                        read_prefix_chunk=mock.Mock(
                            return_value=torch.zeros((8, 3), dtype=torch.bfloat16)
                        ),
                    ),
                    page_rr_metadata=SimpleNamespace(cp_rank=3),
                    block_table=torch.ones((1, 1), dtype=torch.int32),
                    qo_indptr=torch.tensor([0, 4], dtype=torch.int32),
                    fmha_params=SimpleNamespace(max_q=4),
                    prefill_wrapper=SimpleNamespace(run_partial=run_partial),
                )
                query = torch.zeros((4, 1, 192), dtype=torch.bfloat16)
                cache = SimpleNamespace(kv_cache_base=torch.zeros((4, 3)))
                with (
                    mock.patch(_MLA + ".smoke_evidence_enabled", return_value=True),
                    mock.patch(
                        _MLA + ".quantize_qkv_fp8", return_value=(query, output, output)
                    ),
                    mock.patch(
                        _MLA + ".quantize_kv_fp8", return_value=(output, output)
                    ),
                    mock.patch(
                        _MLA + ".emit_smoke_event",
                        side_effect=lambda *a, **k: order.append("event"),
                    ) as emit,
                    mock.patch(
                        _MLA + ".merge_attention_states_in_place",
                        side_effect=lambda *a: order.append("merge"),
                    ),
                ):
                    args = (
                        impl,
                        query,
                        torch.zeros((4, 2)),
                        torch.zeros((4, 1)),
                        cache,
                        7,
                    )
                    if fails:
                        with self.assertRaisesRegex(
                            RuntimeError, "prefix backend failed"
                        ):
                            KimiK3MlaImpl._forward_fp8_chunked(*args)
                    else:
                        KimiK3MlaImpl._forward_fp8_chunked(*args)
                self.assertEqual(
                    order,
                    ["prefix_call"] if fails else ["prefix_call", "event", "merge"],
                )
                if fails:
                    emit.assert_not_called()
                else:
                    emit.assert_called_once_with(
                        "mla_prefix_executed",
                        kind="mla_prefix_executed",
                        backend="fp8_tokenspeed",
                        layer_id=7,
                        rank=3,
                        owner=0,
                        start=4096,
                        length=8,
                        query_tokens=4,
                    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/Triton")
class PageRROwnershipObservationTest(unittest.TestCase):
    def test_observes_real_kernel_slots_across_page_boundaries_and_padding(self):
        # The five live tokens occupy pages [0,1,1,7,8]. With CP4 the
        # owner counts are [2,2,0,1]; the final three physical rows are padding.
        for rank, expected_owned in enumerate((2, 2, 0, 1)):
            with self.subTest(rank=rank):
                lengths = torch.tensor([3, 5], dtype=torch.int32)
                prefixes = torch.tensor([4095, 32767], dtype=torch.int32)
                inputs = SimpleNamespace(
                    input_lengths=lengths,
                    prefix_lengths=prefixes,
                    input_lengths_device=lengths.cuda(),
                    prefix_lengths_device=prefixes.cuda(),
                    cu_seqlens_device=torch.tensor(
                        [0, 3, 8], dtype=torch.int32, device="cuda"
                    ),
                    kv_cache_kernel_block_id_device=torch.arange(
                        1, 257, dtype=torch.int32, device="cuda"
                    ).view(2, 128),
                    padding_offset=torch.tensor(
                        [0, 0, 0, 2, 2, 2, 2, 2], dtype=torch.int32, device="cuda"
                    ),
                    logical_token_count=5,
                    physical_token_count=8,
                )
                metadata = PageRRMlaMetadata(4096, 128, 4, rank)
                with (
                    mock.patch(
                        _METADATA + ".smoke_evidence_enabled", return_value=True
                    ),
                    mock.patch(_METADATA + ".emit_smoke_tensor_event") as emit,
                ):
                    metadata.prepare_prefill(inputs)
                emit.assert_called_once()
                values = emit.call_args.kwargs
                self.assertEqual(emit.call_args.args, ("mla_page_rr_prefill",))
                self.assertEqual(values["logical_tokens"], 5)
                self.assertEqual(values["physical_tokens"], 8)
                self.assertEqual(values["padding_rows"], 3)
                observed = values["tensor_fields"]
                self.assertTrue(all(value.is_cuda for value in observed.values()))
                self.assertEqual(observed["owned_token_rows"].item(), expected_owned)
                self.assertEqual(observed["padding_owned_slots"].item(), 0)
                self.assertTrue(torch.all(metadata.slot_mapping[5:] == -1).item())


if __name__ == "__main__":
    unittest.main()
