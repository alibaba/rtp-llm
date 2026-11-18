import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.segmented_prefill import (
    QwenPrefillContext,
    QwenSegmentedPrefillAdapter,
)


class QwenSegmentedPrefillTest(unittest.TestCase):
    def setUp(self):
        self.adapter = QwenSegmentedPrefillAdapter.__new__(QwenSegmentedPrefillAdapter)
        self.adapter.device = torch.device("cpu")
        self.adapter.max_requests = 4
        self.adapter.buckets = [2048, 8192]
        self.adapter.linear_layers = [(0, None)]
        self.adapter.linear_tags = {"linear"}
        self.adapter.page_geometry = {"linear": (128, 1000), "full": (128, 1000)}
        self.adapter.model = SimpleNamespace(
            config=SimpleNamespace(max_seq_len=12288),
            parallelism_config=SimpleNamespace(role_type="PREFILL"),
            _attention_tag=lambda _: "linear",
            kv_cache=SimpleNamespace(get_kernel_seq_size_per_block=lambda _: 128),
        )

    def inputs(self, lengths=(2048, 3072), prefixes=(6144, 7168), width=96):
        groups = {}
        for tag in self.adapter.page_geometry:
            table = torch.arange(
                1, len(lengths) * width + 1, dtype=torch.int32
            ).reshape(len(lengths), width)
            groups[tag] = SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                is_cuda_graph=False,
                input_lengths=torch.tensor(lengths, dtype=torch.int32),
                prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
                kv_cache_kernel_block_id=table,
                kv_cache_kernel_block_id_device=table.clone(),
                kv_cache_block_id=table,
            )
        return SimpleNamespace(
            attention_inputs=groups,
            input_ids=torch.zeros(sum(lengths), dtype=torch.int32),
            input_hiddens=None,
            multimodal_inputs=SimpleNamespace(multimodal_features=[]),
            embedding_inputs=SimpleNamespace(combo_tokens_type_ids=None),
        )

    def test_reuse_admission_checks_consumed_pages_and_aliases(self):
        inputs = self.inputs()
        linear = inputs.attention_inputs["linear"].kv_cache_kernel_block_id
        linear[0, :47] = -1
        linear[1, :55] = -1
        linear[0, 49:63] = -1
        self.assertEqual(self.adapter.select(inputs), (8192, "replayed"))
        # No write can overlap a shared read-only prefix or another live write.
        linear[1, 55] = linear[0, 47]
        self.assertEqual(self.adapter.select(inputs), (8192, "replayed"))
        linear[1, 56] = linear[0, 48]
        self.assertEqual(self.adapter.select(inputs)[0], 0)
        linear[1, 56] = linear[0, 47]
        self.assertEqual(self.adapter.select(inputs)[0], 0)
        linear[0, 47] = -1
        with self.assertRaisesRegex(ValueError, "prefix page"):
            self.adapter.select(inputs)

    def test_short_tables_and_inconsistent_prefixes_fall_back(self):
        self.assertEqual(self.adapter.select(self.inputs(width=64))[0], 0)
        inputs = self.inputs()
        inputs.attention_inputs["full"].prefix_lengths[1] = 6144
        self.assertEqual(self.adapter.select(inputs), (0, "input_metadata_invalid"))
        for prefixes, reason in (
            ((6145, 7168), "prefix_cache_not_supported"),
            ((-128, 7168), "input_metadata_invalid"),
            ((6144, 12288), "request_not_supported"),
        ):
            self.assertEqual(
                self.adapter.select(self.inputs(prefixes=prefixes)), (0, reason)
            )

    def test_refresh_retains_addresses_and_clears_previous_requests(self):
        dummy = self.inputs(lengths=(8192,), prefixes=(0,), width=64)
        context = QwenPrefillContext(self.adapter, 8192, dummy)
        self.assertEqual(context.tables["linear"].shape, (4, 96))
        pointers = (
            context.tables["linear"].data_ptr(),
            context.prepared.prefix_lengths.data_ptr(),
        )
        live = self.inputs()
        context.refresh(live, [2048, 3072])
        self.assertEqual(context.prepared.prefix_lengths.tolist(), [6144, 7168, 0, 0])
        self.assertEqual(
            context.prepared.cu_seqlens.tolist(), [0, 2048, 5120, 5120, 5120]
        )
        self.assertTrue(
            torch.equal(
                context.tables["linear"][:2],
                live.attention_inputs["linear"].kv_cache_kernel_block_id_device,
            )
        )
        context.refresh(self.inputs(lengths=(2048,), prefixes=(0,), width=16), [2048])
        self.assertEqual(context.prepared.prefix_lengths.tolist(), [0, 0, 0, 0])
        self.assertEqual(
            context.prepared.request_live.tolist(), [True, False, False, False]
        )
        self.assertTrue(bool(torch.all(context.tables["linear"][1:] == -1)))
        self.assertTrue(bool(torch.all(context.tables["linear"][0, 16:] == -1)))
        self.assertEqual(
            pointers,
            (
                context.tables["linear"].data_ptr(),
                context.prepared.prefix_lengths.data_ptr(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
