import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    build_flashmla_device_params,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_forward_plan import (
    FlashMLAForwardRoute,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_forward_test_utils import (
    PAGE_SIZE,
    CaseInputs,
    DeterministicPackedProjection,
    make_case_inputs,
    make_op,
    output_and_lse,
)
from rtp_llm.utils.k3_model_trace import ModelTrace


class FlashMLADensePrefillPackedKVTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("K3 FlashMLA runtime test requires SM103a")

    @staticmethod
    def _make_inputs(
        q_lens: Sequence[int],
        prefix_lens: Sequence[int],
        *,
        strided_k_pe: bool = False,
    ) -> CaseInputs:
        inputs = make_case_inputs(q_lens, prefix_lens)
        q_lens = tuple(q_lens)
        prefix_lens = tuple(prefix_lens)
        max_blocks = max(
            (q_len + prefix_len + PAGE_SIZE - 1) // PAGE_SIZE
            for q_len, prefix_len in zip(q_lens, prefix_lens, strict=True)
        )
        block_table = torch.zeros(
            (len(q_lens), max_blocks), dtype=torch.int32, device="cuda"
        )
        page_cursor = 0
        for owner, prefix_len in enumerate(prefix_lens):
            page_count = (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE
            block_table[owner, :page_count] = torch.arange(
                page_cursor,
                page_cursor + page_count,
                dtype=torch.int32,
                device="cuda",
            )
            page_cursor += page_count

        max_q_len = max(q_lens)
        padding_offset = [
            owner * max_q_len - sum(q_lens[:owner])
            for owner, q_len in enumerate(q_lens)
            for _ in range(q_len)
        ]
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            total_tokens=sum(q_lens),
            input_lengths_host=torch.tensor(q_lens, dtype=torch.int32),
            prefix_lengths_host=torch.tensor(prefix_lens, dtype=torch.int32),
            input_lengths=torch.tensor(q_lens, dtype=torch.int32, device="cuda"),
            prefix_lengths=torch.tensor(prefix_lens, dtype=torch.int32, device="cuda"),
            cu_seqlens=inputs.params.qo_indptr_d,
            cu_kv_seqlens=inputs.params.prefill_ragged_kv_len_indptr_d,
            padding_offset=torch.tensor(
                padding_offset, dtype=torch.int32, device="cuda"
            ),
            kv_cache_kernel_block_id_device_by_group=[block_table],
            kv_cache_kernel_block_id_device=block_table,
        )
        params = build_flashmla_device_params(attn_inputs, PAGE_SIZE)
        k_pe = inputs.k_pe
        if strided_k_pe:
            k_pe_storage = torch.empty(
                (sum(q_lens), k_pe.shape[1] + 17),
                dtype=k_pe.dtype,
                device=k_pe.device,
            )
            strided_k_pe_view = k_pe_storage.narrow(1, 11, k_pe.shape[1])
            strided_k_pe_view.copy_(k_pe)
            k_pe = strided_k_pe_view
        return CaseInputs(
            params,
            inputs.q,
            inputs.compressed_kv,
            k_pe,
            inputs.kv_cache,
        )

    @staticmethod
    def _reference(inputs: CaseInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent paged-cache gather and bottom-right causal attention."""
        outputs, lses = [], []
        q_start = 0
        params = inputs.params
        for owner, (q_len, prefix_len) in enumerate(
            zip(params.q_lens_host, params.prefix_lens_host, strict=True)
        ):
            current = inputs.compressed_kv[q_start : q_start + q_len]
            current_kpe = inputs.k_pe[q_start : q_start + q_len]
            if prefix_len:
                pages = params.attn_inputs.kv_cache_kernel_block_id_device[
                    owner, : (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE
                ].long()
                prefix = inputs.kv_cache.kv_cache_base[pages].flatten(0, 1)[:prefix_len]
                latent = torch.cat((prefix[:, :512], current))
                kpe = torch.cat((prefix[:, 512:576], current_kpe))
            else:
                latent, kpe = current, current_kpe
            # DeterministicPackedProjection repeats the same K/V across heads.
            key = torch.cat((latent[:, :128], kpe), dim=-1).float()
            value = latent[:, 128:256].float()
            query = inputs.q[q_start : q_start + q_len].float()
            scores = torch.einsum("qhd,kd->hqk", query, key) * (192**-0.5)
            query_positions = prefix_len + torch.arange(q_len, device=query.device)
            key_positions = torch.arange(key.shape[0], device=query.device)
            scores.masked_fill_(
                key_positions[None, :] > query_positions[:, None], -torch.inf
            )
            outputs.append(torch.einsum("hqk,kd->qhd", scores.softmax(-1), value))
            lses.append(scores.logsumexp(-1).T)
            q_start += q_len
        return torch.cat(outputs), torch.cat(lses)

    def test_full_and_hybrid_match_causal_reference(self) -> None:
        cases = (
            ((2, 3), (0, 0), 256, False),
            ((2, 3), (128, 128), 384, False),
            ((2, 3), (128, 128), 256, False),
            ((2, 3, 1), (128, 0, 128), 256, False),
            ((1, 1, 1), (384, 0, 384), 256, False),
            ((2, 3), (0, 300), 128, True),
            ((1, 1), (127, 640), 256, False),
        )
        for q_lens, prefix_lens, capacity, strided in cases:
            with self.subTest(q=q_lens, prefix=prefix_lens, capacity=capacity):
                inputs = self._make_inputs(q_lens, prefix_lens, strided_k_pe=strided)
                expected, expected_lse = self._reference(inputs)
                for budget in (0, capacity):
                    op = make_op(expanded_kv_capacity_tokens=budget)
                    op.plan(inputs.params)
                    with mock.patch.object(
                        op,
                        "_create_kv_b_proj",
                        return_value=DeterministicPackedProjection(),
                    ):
                        output, lse = output_and_lse(op, inputs)
                    torch.testing.assert_close(
                        output.float(), expected, rtol=2e-2, atol=1e-3
                    )
                    torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-4)
                    if op._forward_plan.route is FlashMLAForwardRoute.HYBRID:
                        workspace = op._forward_workspace
                        self.assertLessEqual(workspace.packed_kv.shape[0], capacity)

    def test_trace_retains_context_results_across_workspace_reuse(self) -> None:
        inputs = self._make_inputs((2, 3, 1), (384, 0, 384))
        op = make_op(expanded_kv_capacity_tokens=256)
        op.plan(inputs.params)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(
                os.environ,
                {"K3_TRACE_ROOT": str(root), "K3_TRACE_RUN_ID": "mla-workspace"},
            ), mock.patch.object(
                op, "_create_kv_b_proj", return_value=DeterministicPackedProjection()
            ):
                trace = ModelTrace("mla", directory=root / "frames")
                try:
                    first, _ = trace.forward(lambda x: output_and_lse(op, x), inputs)
                    inputs.compressed_kv.add_(0.25)
                    second, _ = trace.forward(lambda x: output_and_lse(op, x), inputs)
                finally:
                    trace.close()

            frames = sorted((root / "frames").glob("frame-*.pt"))
            self.assertEqual(len(frames), 2)
            self.assertFalse(torch.equal(first, second))
            for path, expected in zip(frames, (first, second), strict=True):
                frame = torch.load(path, weights_only=True)
                events = frame["tensors"]
                outputs = [
                    event["value"]
                    for event in events
                    if event["name"] == "mla.layers.0.prefill.output"
                ]
                self.assertEqual(len(outputs), 1)
                torch.testing.assert_close(outputs[0], expected.cpu(), rtol=0, atol=0)
                chunks = []
                for event in events:
                    prefix = "mla.layers.0.prefill.context."
                    if not event["name"].startswith(prefix):
                        continue
                    key = event["name"][len(prefix) :]
                    if key == "q":
                        chunks.append({})
                    chunks[-1][key] = event["value"]
                self.assertGreater(len(chunks), 1)
                for chunk in chunks:
                    qo, kv = chunk["qo_indptr"], chunk["kv_indptr"]
                    for i in range(qo.numel() - 1):
                        q_slice = slice(int(qo[i]), int(qo[i + 1]))
                        kv_slice = slice(int(kv[i]), int(kv[i + 1]))
                        q = chunk["q"][q_slice].float()
                        k = chunk["k"][kv_slice].float()
                        v = chunk["v"][kv_slice].float()
                        scores = torch.einsum("qhd,khd->hqk", q, k) * (192**-0.5)
                        reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
                        torch.testing.assert_close(
                            chunk["result.output"][q_slice].float(),
                            reference,
                            rtol=2e-2,
                            atol=1e-3,
                        )
                        torch.testing.assert_close(
                            chunk["result.lse"][q_slice],
                            scores.logsumexp(-1).T,
                            rtol=2e-4,
                            atol=2e-4,
                        )

    def test_workspace_reuse_and_replan_preserve_results(self) -> None:
        inputs = self._make_inputs((2, 3, 1), (384, 0, 384))
        op = make_op(expanded_kv_capacity_tokens=256)
        op.plan(inputs.params)
        with mock.patch.object(
            op, "_create_kv_b_proj", return_value=DeterministicPackedProjection()
        ):
            first, _ = output_and_lse(op, inputs)
            workspace = op._forward_workspace
            # Change the next layer's values so stale outputs cannot pass.
            inputs.compressed_kv.add_(0.25)
            second, lse = output_and_lse(op, inputs)
            self.assertIs(op._forward_workspace, workspace)
            self.assertFalse(torch.equal(first, second))
            expected, expected_lse = self._reference(inputs)
            torch.testing.assert_close(second.float(), expected, rtol=2e-2, atol=1e-3)
            torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-4)

            next_inputs = self._make_inputs((4,), (0,))
            op.plan(next_inputs.params)
            self.assertIsNone(op._forward_workspace)
            self.assertEqual(op._prefix_runtime_launches, ())
            output, lse = output_and_lse(op, next_inputs)
            expected, expected_lse = self._reference(next_inputs)
            torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=1e-3)
            torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
