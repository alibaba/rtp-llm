import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.qwen4_exp_config import Qwen4ModelConfig
from rtp_llm.models_py.kernels.cuda.qsa import paged_validation, prefill_validation
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.prefill_preflight import (
    MainGeometryReceipt,
    main_geometry_preflight,
    prefill_geometry_scope,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa import fmha_inputs, fmha_validation
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import SparseGqaFmhaImpl
from rtp_llm.models_py.modules.qwen4_exp.qsa.validation import DeviceChecks
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs


class PrefillGeometryScopeTest(unittest.TestCase):
    def test_hook_checks_only_apply_to_eager_qsa(self):
        model = torch.nn.Sequential(torch.nn.Identity())
        model.kv_cache = object()
        model.config = Qwen4ModelConfig()
        attn = PyAttentionInputs()
        attn.is_prefill = True
        inputs = PyModelInputs()
        inputs.attention_inputs = attn

        with patch.object(
            model, "modules", side_effect=AssertionError("unexpected module scan")
        ):
            with prefill_geometry_scope(model, inputs, None) as share:
                self.assertFalse(share)

        model.config._qwen4_indexer_head_dim = 128
        with prefill_geometry_scope(model, inputs, None) as share:
            self.assertTrue(share)
        with model[0].register_forward_pre_hook(lambda *_: None):
            with prefill_geometry_scope(model, inputs, None) as share:
                self.assertFalse(share)


class PrefillPreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")

    def fixture(self, *, padded=False):
        table = torch.tensor([[1, 2, 3, 0]], device="cuda", dtype=torch.int32)
        cache = torch.zeros(5, 2, 1, 4, 16, device="cuda", dtype=torch.bfloat16)
        bases = torch.tensor([4], dtype=torch.int32)
        lengths = torch.tensor([4], dtype=torch.int32)
        impl = SimpleNamespace(
            attn_inputs=SimpleNamespace(
                is_s_padded=padded, kv_cache_kernel_block_id_device=table
            ),
            attn_configs=SimpleNamespace(kernel_tokens_per_block=4),
            head_dim=16,
            kv_head_num=1,
            _model_prefill_geometry=None,
        )
        return impl, cache, bases, lengths

    def validate(self, entry):
        impl, cache, bases, lengths = entry
        return fmha_validation._validate_main_paged_geometry(
            impl,
            cache,
            impl.attn_inputs.kv_cache_kernel_block_id_device,
            bases,
            lengths,
            cache.device,
        )

    def test_preflight_and_fmha_share_current_host_uploads(self):
        impl, cache, _, _ = self.fixture()
        attn = impl.attn_inputs = PyAttentionInputs()
        attn.is_s_padded = False
        device = cache.device
        other_pool = cache.clone()
        table = torch.empty(2, 4, dtype=torch.int32, device=device)
        for query, prefix, pages in (
            ([4], [4], [[1, 2, 0, 0]]),
            ([2, 3], [8, 0], [[2, 3, 1, 0], [4, 0, 0, 0]]),
            ([1], [7], [[4, 2, 0, 0]]),
        ):
            with self.subTest(query=query, prefix=prefix):
                batch = len(query)
                lengths = attn.input_lengths = torch.tensor(query, dtype=torch.int32)
                bases = attn.prefix_lengths = torch.tensor(prefix, dtype=torch.int32)
                attn.kv_cache_kernel_block_id_device = table[:batch].copy_(
                    torch.tensor(pages, dtype=torch.int32)
                )
                attn.cu_seqlens_device = torch.cat(
                    (lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32))
                ).to(device)
                qkv = torch.empty(sum(query), 1, device=device)
                selected = torch.zeros(sum(query), 1, dtype=torch.int32, device=device)
                entries = [(impl, pool, bases, lengths) for pool in (cache, other_pool)]

                def prepare(pool=cache):
                    return fmha_inputs._try_prepare_host_incremental_prefill(
                        impl,
                        qkv,
                        SimpleNamespace(kv_cache_base=pool),
                        selected,
                        lengths,
                        bases,
                    )

                with patch.object(
                    paged_validation,
                    "geometry_flags",
                    wraps=paged_validation.geometry_flags,
                ) as geometry, patch.object(
                    prefill_validation,
                    "validate_prefill_inputs",
                    wraps=prefill_validation.validate_prefill_inputs,
                ) as packed, patch.object(
                    MainGeometryReceipt,
                    "matches",
                    autospec=True,
                    side_effect=MainGeometryReceipt.matches,
                ) as matches:
                    with main_geometry_preflight(entries):
                        plan = prepare()
                        other_plan = prepare(other_pool)
                    self.assertEqual(geometry.call_count, 1)
                    self.assertEqual(matches.call_count, 2)
                    ld, pd = geometry.call_args.args[:2]
                    torch.testing.assert_close(ld.cpu(), lengths)
                    torch.testing.assert_close(pd.cpu(), bases)
                    self.assertEqual(packed.call_count, 2)
                    for call in packed.call_args_list:
                        self.assertIs(call.args[2], ld)
                        self.assertIs(call.args[3], pd)
                        self.assertFalse(call.kwargs["check_pages"])
                    self.assertIs(plan.prefixes_device, pd)
                    self.assertIs(other_plan.prefixes_device, pd)

                with main_geometry_preflight(entries):
                    for index, error in (
                        (-2, "use -1"),
                        (prefix[0] + 1, "outside the row's visible KV"),
                    ):
                        selected[0, 0] = index
                        with self.assertRaisesRegex(ValueError, error):
                            prepare()
                    selected.zero_()
                    attn.cu_seqlens_device[0] = 1
                    with self.assertRaisesRegex(ValueError, "cu_seqlens"):
                        prepare()
                    attn.cu_seqlens_device[0] = 0
                    with patch.object(impl, "head_dim", 32):
                        with self.assertRaisesRegex(
                            RuntimeError, "main-cache geometry"
                        ):
                            prepare()
                table[0, (prefix[0] + query[0] - 1) // 4] = 0
                with self.assertRaisesRegex(RuntimeError, "unallocated"):
                    with main_geometry_preflight(entries):
                        self.fail("Invalid page passed main preflight")
                with self.assertRaisesRegex(RuntimeError, "unallocated"):
                    prepare()

    def test_shared_table_is_rechecked_for_smaller_pool(self):
        impl, cache, bases, lengths = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "out-of-range physical block"):
            with main_geometry_preflight(
                [(impl, cache, bases, lengths), (impl, cache[:2], bases, lengths)]
            ):
                self.fail("Page 2 is outside the second pool")
        self.assertIsNone(impl._model_prefill_geometry)
        with main_geometry_preflight([(impl, cache, bases, lengths)]):
            with self.assertRaisesRegex(RuntimeError, "out-of-range physical block"):
                self.validate((impl, cache[:2], bases, lengths))

    def test_current_host_lengths_win_over_stale_mirrors_and_receipts(self):
        impl, cache, bases, lengths = self.fixture()
        table = impl.attn_inputs.kv_cache_kernel_block_id_device
        attn = impl.attn_inputs = PyAttentionInputs()
        attn.is_prefill = True
        attn.input_lengths = lengths
        attn.prefix_lengths = bases
        attn.kv_cache_kernel_block_id_device = table
        stale_lengths = torch.tensor([3], dtype=torch.int32, device=cache.device)
        stale_prefixes = torch.tensor([16], dtype=torch.int32, device=cache.device)
        attn.cu_seqlens_device = torch.tensor(
            [0, 4], dtype=torch.int32, device=cache.device
        )
        qkv = torch.empty(4, 1, device=cache.device)
        selected = torch.zeros(4, 1, dtype=torch.int32, device=cache.device)

        def prepare():
            return fmha_inputs._try_prepare_host_incremental_prefill(
                impl,
                qkv,
                SimpleNamespace(kv_cache_base=cache),
                selected,
                attn.input_lengths,
                attn.prefix_lengths,
            )

        # Python can read the engine-owned mirrors but cannot assign them.
        with patch.object(
            PyAttentionInputs, "input_lengths_device", stale_lengths
        ), patch.object(PyAttentionInputs, "prefix_lengths_device", stale_prefixes):
            self.assertEqual(prepare().requests[0].kv_lens.tolist(), [[5, 6, 7, 8]])
            with main_geometry_preflight([(impl, cache, bases, lengths)]):
                self.assertEqual(prepare().requests[0].kv_lens.tolist(), [[5, 6, 7, 8]])
                bases.fill_(3)
                self.assertEqual(prepare().requests[0].kv_lens.tolist(), [[4, 5, 6, 7]])

    def test_preflight_keeps_int64_validation(self):
        impl, cache, _, lengths = self.fixture()
        bases = torch.tensor([2**32 + 4], dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "does not cover"):
            with main_geometry_preflight([(impl, cache, bases, lengths)]):
                self.fail("Int64 prefix was narrowed before validation")

    def test_receipts_are_forward_scoped_and_metadata_mutation_is_rechecked(self):
        entries = [self.fixture(), self.fixture()]
        with main_geometry_preflight(entries):
            with patch.object(
                fmha_validation,
                "_try_validate_main_paged_geometry",
                side_effect=AssertionError("redundant scan"),
            ):
                for entry in entries:
                    self.assertEqual(self.validate(entry), 4)
            entries[0][0].attn_inputs.kv_cache_kernel_block_id_device[0, 1] = 0
            with self.assertRaisesRegex(RuntimeError, "unallocated"):
                self.validate(entries[0])
        self.assertTrue(
            all(entry[0]._model_prefill_geometry is None for entry in entries)
        )
        with self.assertRaisesRegex(RuntimeError, "unallocated"):
            with main_geometry_preflight(entries):
                self.fail("Invalid live metadata entered the model body")

    def test_only_visible_pages_are_required_before_publishing_receipts(self):
        entries = [self.fixture(), self.fixture()]
        entries[1][0].attn_inputs.kv_cache_kernel_block_id_device[0, -1] = 5
        with main_geometry_preflight(entries):
            self.assertEqual(self.validate(entries[1]), 4)
        entries[1][0].attn_inputs.kv_cache_kernel_block_id_device[0, 1] = 5
        with self.assertRaisesRegex(RuntimeError, "out-of-range physical block"):
            with main_geometry_preflight(entries):
                self.fail("Invalid page entered the model body")
        for impl, *_ in entries:
            self.assertIsNone(impl._model_prefill_geometry)

    def test_only_matching_model_receipt_allows_shared_prefill_checks(self):
        entry = self.fixture()
        impl, cache, bases, lengths = entry

        def validate():
            checks = DeviceChecks()
            fmha_validation._validate_main_paged_geometry(
                impl,
                cache,
                impl.attn_inputs.kv_cache_kernel_block_id_device,
                bases,
                lengths,
                cache.device,
                checks=checks,
            )
            checks.verify()
            return checks.prefill_metadata_readonly

        self.assertFalse(validate())
        with main_geometry_preflight([entry]):
            self.assertTrue(validate())
            lengths.fill_(3)
            self.assertFalse(validate())
        self.assertFalse(validate())

    def test_decode_reuses_only_accepted_geometry_once_in_the_same_transaction(self):
        impl, cache, bases, lengths = self.fixture()
        impl.is_target_verify = True
        impl.attn_inputs.input_lengths = lengths
        table = impl.attn_inputs.kv_cache_kernel_block_id_device
        checks = DeviceChecks()
        self.assertEqual(
            fmha_validation._validate_main_paged_geometry(
                impl, cache, table, bases, lengths, cache.device, checks=checks
            ),
            4,
        )
        receipt = MainGeometryReceipt(impl, cache, table, bases, lengths, 4)

        def consume():
            return fmha_validation.consume_main_geometry(impl, cache, table, bases, 4)

        impl._qsa_main_geometry = (receipt, checks, 4)
        self.assertFalse(consume())
        checks.verify()
        impl._qsa_main_geometry = (receipt, checks, 4)
        self.assertTrue(consume())
        self.assertFalse(consume())

        impl._qsa_main_geometry = (receipt, checks, 4)
        SparseGqaFmhaImpl.begin_qsa_cache_transaction(impl)
        self.assertFalse(consume())
        impl._qsa_main_geometry = (receipt, checks, 4)
        table[0, 1] = 0
        self.assertFalse(consume())
        with self.assertRaisesRegex(RuntimeError, "unallocated"):
            self.validate((impl, cache, bases, lengths))

    def test_failed_or_extended_checks_cannot_authorize_a_main_writer(self):
        checks = DeviceChecks()
        invalid = torch.tensor(False, device="cuda")
        checks.add(invalid, "invalid metadata")
        checks.verify()
        self.assertTrue(checks.verified)
        invalid.fill_(True)
        with self.assertRaisesRegex(RuntimeError, "invalid metadata"):
            checks.verify()
        self.assertFalse(checks.verified)
        invalid.fill_(False)
        checks.verify()
        checks.add(torch.tensor(False, device="cuda"), "new check")
        self.assertFalse(checks.verified)

    def test_receipt_cleanup_after_model_failure_and_new_host_lengths(self):
        entry = self.fixture()
        with self.assertRaisesRegex(ValueError, "model failure"):
            with main_geometry_preflight([entry]):
                raise ValueError("model failure")
        self.assertIsNone(entry[0]._model_prefill_geometry)
        entry[2].fill_(16)
        with self.assertRaisesRegex(RuntimeError, "does not cover"):
            with main_geometry_preflight([entry]):
                self.fail("New host lengths must be validated")

    def test_empty_padded_rows_keep_the_existing_geometry_semantics(self):
        entry = self.fixture(padded=True)
        entry[0].attn_inputs.kv_cache_kernel_block_id_device.zero_()
        entry[2].fill_(128)
        with main_geometry_preflight([entry]):
            self.assertEqual(self.validate(entry), 4)

    def test_native_inference_tensors_are_validated_fresh_for_each_forward(self):
        with torch.inference_mode():
            entry = self.fixture()
            with main_geometry_preflight([entry]):
                with patch.object(
                    fmha_validation,
                    "_try_validate_main_paged_geometry",
                    side_effect=AssertionError("inference receipt was bypassed"),
                ):
                    self.assertEqual(self.validate(entry), 4)
            self.assertIsNone(entry[0]._model_prefill_geometry)
            entry[0].attn_inputs.kv_cache_kernel_block_id_device[0, 1] = 0
            with self.assertRaisesRegex(RuntimeError, "unallocated"):
                with main_geometry_preflight([entry]):
                    self.fail("Stale inference metadata survived a forward")


if __name__ == "__main__":
    unittest.main()
