"""Single-GPU Prefill conv alias regression, independent of model weights."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    paged_short_conv_prefill as paged,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prefill_state_alias as aliasing,
)


class Inputs:
    pass


class PrefillConvAliasGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("Prefill state alias regression requires a CUDA GPU")

    def test_snapshot_matches_distinct_storage_and_independent_convolution(self):
        torch.manual_seed(3101)
        torch.backends.cudnn.allow_tf32 = False
        channels, page, prefix = 384, 128, 1024
        checked = 0
        for n in range(1, 6):
            for history_size in (3, 7):
                for length in (1, 2, 128, 2048, 65536):
                    with self.subTest(n=n, history=history_size, query=length):
                        x = (
                            torch.randn(
                                length, channels, device="cuda", dtype=torch.bfloat16
                            )
                            * 0.2
                        )
                        weight = (
                            torch.randn(
                                channels, history_size + 1, device="cuda", dtype=x.dtype
                            )
                            * 0.2
                        )
                        state = (
                            torch.randn(
                                n + 4,
                                history_size,
                                channels,
                                device="cuda",
                                dtype=x.dtype,
                            )
                            * 0.2
                        )
                        initial = state[1].clone()
                        end = prefix + length
                        row = [-1] * ((end + page - 1) // page)
                        row[(prefix - 1) // page] = 1
                        complete = end // 1024
                        first = max(1, complete - n + 1)
                        for checkpoint in range(first, complete + 1):
                            tokens = checkpoint * 1024
                            if tokens > prefix:
                                row[tokens // page - 1] = checkpoint - first + 2
                        row[-1] = 1
                        inputs = Inputs()
                        inputs.prefix_lengths = torch.tensor(
                            [prefix], dtype=torch.int32
                        )
                        inputs.cu_seqlens = torch.tensor([0, length], dtype=torch.int32)
                        inputs.kv_cache_kernel_block_id = torch.tensor(
                            [row], dtype=torch.int32
                        )
                        metadata = SimpleNamespace(prefill_cuda_inputs={})
                        snapshot, mask = aliasing.snapshot_prefill_conv_input(
                            state, inputs, page, metadata
                        )
                        self.assertIsNotNone(snapshot)
                        torch.testing.assert_close(snapshot[0], initial, atol=0, rtol=0)
                        conv_meta = paged.prepare_paged_short_conv_metadata(
                            inputs.cu_seqlens, x.device
                        )
                        prefix_gpu = inputs.prefix_lengths.to("cuda")
                        cu_gpu = inputs.cu_seqlens.to("cuda")
                        table = inputs.kv_cache_kernel_block_id.to("cuda")
                        result = paged.paged_short_conv_prefill(
                            x,
                            weight,
                            state,
                            table,
                            prefix_gpu,
                            cu_gpu,
                            page,
                            conv_meta,
                            current_conv_state=snapshot,
                            continuation_mask=mask,
                            return_final_state=True,
                        )

                        # Same operator with a separate immutable source. No snapshot
                        # is necessary; this detects corruption from in-place writes.
                        reference_state = state.clone()
                        reference_state[0] = initial
                        reference_row = list(row)
                        reference_row[(prefix - 1) // page] = n + 3
                        reference_state[n + 3] = initial
                        reference = paged.paged_short_conv_prefill(
                            x,
                            weight,
                            reference_state,
                            torch.tensor(
                                [reference_row], device="cuda", dtype=torch.int32
                            ),
                            prefix_gpu,
                            cu_gpu,
                            page,
                            conv_meta,
                            return_final_state=True,
                        )
                        for actual, expected in zip(result, reference):
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

                        raw = torch.cat((initial, x), dim=0)
                        golden = (
                            F.conv1d(
                                raw.float().T.unsqueeze(0),
                                weight.float().unsqueeze(1),
                                groups=channels,
                            )
                            .squeeze(0)
                            .T
                        )
                        golden = F.silu(golden).to(x.dtype)
                        actual = torch.cat(result[:3], dim=-1)
                        torch.testing.assert_close(
                            actual, golden, atol=0.002, rtol=0.02
                        )
                        expected_tail = raw[-history_size:]
                        torch.testing.assert_close(
                            result[3][0], expected_tail, atol=0, rtol=0
                        )
                        for position, physical in enumerate(row):
                            if physical > 0 and (position + 1) * page > prefix:
                                local_end = min((position + 1) * page, end) - prefix
                                expected = torch.cat((initial, x[:local_end]))[
                                    -history_size:
                                ]
                                torch.testing.assert_close(
                                    state[physical], expected, atol=0, rtol=0
                                )
                        checked += 1
        print(
            f"Prefill conv alias: {checked} cases, Last-1..5, including 64K and short history"
        )


if __name__ == "__main__":
    unittest.main()
