# type: ignore
import statistics
import sys
import time
from pathlib import Path
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter


def decode_scalar_from_bytes(data: bytes, dtype: torch.dtype):
    raw = torch.tensor(list(data), dtype=torch.uint8)
    scalar = raw.view(dtype)[0]
    return scalar.item()


class LinearCacheConverterTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is required for LinearCacheConverter tests")
        self.device = torch.device("cuda")

    def _assert_view_matches_bytes(
        self,
        base_tensor: torch.Tensor,
        view_tensor: torch.Tensor,
        stride_bytes: tuple[int, ...],
        storage_offset_bytes: int,
    ) -> None:
        raw = (
            torch.empty(0, device=base_tensor.device, dtype=torch.uint8)
            .set_(
                base_tensor.untyped_storage(),
                0,
                (base_tensor.untyped_storage().nbytes(),),
                (1,),
            )
            .cpu()
        )
        base_offset_bytes = (
            base_tensor.storage_offset()
            * LinearCacheConverter.dtype_size_bytes(base_tensor.dtype)
        )
        target_item_size = LinearCacheConverter.dtype_size_bytes(view_tensor.dtype)
        for index in torch.cartesian_prod(
            *[torch.arange(dim, dtype=torch.int64) for dim in view_tensor.shape]
        ):
            logical_index = tuple(int(x) for x in index.tolist())
            byte_offset = base_offset_bytes + storage_offset_bytes
            for dim, idx in enumerate(logical_index):
                byte_offset += idx * stride_bytes[dim]
            expected = decode_scalar_from_bytes(
                bytes(raw[byte_offset : byte_offset + target_item_size].tolist()),
                view_tensor.dtype,
            )
            self.assertEqual(view_tensor[logical_index].item(), expected)

    def test_gpu_linear_cache_converter(self) -> None:
        block_num = 4
        local_num_v_heads = 32
        head_v_dim = 128
        head_k_dim = 128
        linear_conv_kernel_dim = 3
        qkv_size = 256 * (32 + 2 + 2)

        ssm_state_size = local_num_v_heads * head_v_dim * head_k_dim
        ssm_state_size_bytes = ssm_state_size * LinearCacheConverter.dtype_size_bytes(
            torch.float32
        )
        conv_state_size_bytes = (
            (linear_conv_kernel_dim - 1)
            * qkv_size
            * LinearCacheConverter.dtype_size_bytes(torch.bfloat16)
        )
        block_size_bytes = (
            ssm_state_size_bytes + conv_state_size_bytes + 128
        )  # 128 for random padding
        base = torch.arange(
            block_num
            * (
                block_size_bytes
                // LinearCacheConverter.dtype_size_bytes(torch.bfloat16)
            ),
            dtype=torch.bfloat16,
            device=self.device,
        ).reshape(block_num, -1)

        converter = LinearCacheConverter(
            local_num_v_heads=local_num_v_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            ssm_state_dtype=torch.float32,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            qkv_size=qkv_size,
            conv_state_dtype=torch.bfloat16,
        )
        self.assertEqual(converter.ssm_state_size_bytes, ssm_state_size_bytes)
        self.assertEqual(converter.conv_state_size_bytes, conv_state_size_bytes)
        self.assertEqual(
            converter.block_size_bytes, ssm_state_size_bytes + conv_state_size_bytes
        )
        self.assertEqual(converter.get_block_size_bytes(base), block_size_bytes)
        conv_item_size = LinearCacheConverter.dtype_size_bytes(torch.bfloat16)
        q_size = (qkv_size - local_num_v_heads * head_v_dim) // 2
        expected_segments = (
            ssm_state_size_bytes,
            *(
                (
                    q_size * conv_item_size,
                    q_size * conv_item_size,
                    local_num_v_heads * head_v_dim * conv_item_size,
                )
                * (linear_conv_kernel_dim - 1)
            ),
        )
        self.assertEqual(converter.cache_store_segment_sizes(), expected_segments)
        self.assertEqual(sum(expected_segments), converter.block_size_bytes)

        ssm_view = converter.get_ssm_state_tensor(base)
        conv_view = converter.get_conv_state_tensor(base)

        self.assertEqual(
            ssm_view.shape, (block_num, local_num_v_heads, head_v_dim, head_k_dim)
        )
        self.assertEqual(
            ssm_view.stride(),
            (
                block_size_bytes
                // LinearCacheConverter.dtype_size_bytes(torch.float32),
                head_v_dim * head_k_dim,
                head_k_dim,
                1,
            ),
        )
        self.assertEqual(ssm_view.dtype, torch.float32)
        self.assertEqual(ssm_view.storage_offset(), 0)

        self.assertEqual(
            conv_view.shape, (block_num, linear_conv_kernel_dim - 1, qkv_size)
        )
        self.assertEqual(
            conv_view.stride(),
            (
                block_size_bytes
                // LinearCacheConverter.dtype_size_bytes(torch.bfloat16),
                qkv_size,
                1,
            ),
        )
        self.assertEqual(conv_view.dtype, torch.bfloat16)
        self.assertEqual(
            conv_view.storage_offset(),
            ssm_state_size_bytes
            // LinearCacheConverter.dtype_size_bytes(torch.bfloat16),
        )

        self._assert_view_matches_bytes(
            base,
            ssm_view,
            (
                block_size_bytes,
                head_v_dim
                * head_k_dim
                * LinearCacheConverter.dtype_size_bytes(torch.float32),
                head_k_dim * LinearCacheConverter.dtype_size_bytes(torch.float32),
                LinearCacheConverter.dtype_size_bytes(torch.float32),
            ),
            0,
        )
        self._assert_view_matches_bytes(
            base,
            conv_view,
            (
                block_size_bytes,
                qkv_size * LinearCacheConverter.dtype_size_bytes(torch.bfloat16),
                LinearCacheConverter.dtype_size_bytes(torch.bfloat16),
            ),
            ssm_state_size_bytes,
        )

        conv_view[1, 1, 3] = torch.tensor(-7, dtype=torch.bfloat16, device=self.device)
        self.assertEqual(conv_view[1, 1, 3].item(), -7)

    def test_cached_views_alias_invalidate_and_reduce_cpu_overhead(self) -> None:
        kwargs = dict(
            local_num_v_heads=4,
            head_v_dim=16,
            head_k_dim=16,
            ssm_state_dtype=torch.float32,
            linear_conv_kernel_dim=4,
            qkv_size=384,
            conv_state_dtype=torch.bfloat16,
        )
        cached = LinearCacheConverter(**kwargs, cache_views=True)
        uncached = LinearCacheConverter(**kwargs, cache_views=False)
        block_elements = (cached.block_size_bytes + 1) // 2 + 64
        base = torch.zeros(9, block_elements, dtype=torch.bfloat16, device=self.device)

        conv = cached.get_conv_state_tensor(base)
        ssm = cached.get_ssm_state_tensor(base.reshape(base.shape[0], -1))
        self.assertIs(conv, cached.get_conv_state_tensor(base.reshape_as(base)))
        self.assertIs(ssm, cached.get_ssm_state_tensor(base.reshape_as(base)))
        conv[2, 1, 7] = 11
        self.assertEqual(cached.get_conv_state_tensor(base)[2, 1, 7].item(), 11)

        sliced = base[1:]
        sliced_conv = cached.get_conv_state_tensor(sliced)
        sliced_ssm = cached.get_ssm_state_tensor(sliced)
        self.assertIsNot(sliced_conv, conv)
        self.assertIsNot(sliced_ssm, ssm)
        self.assertEqual(sliced_conv.shape[0], base.shape[0] - 1)
        sliced_conv[0, 0, 3] = -5
        self.assertEqual(
            uncached.get_conv_state_tensor(base)[1, 0, 3].item(), -5
        )

        other = torch.zeros_like(base)
        other_conv = cached.get_conv_state_tensor(other)
        self.assertIsNot(other_conv, sliced_conv)
        other_conv[0, 0, 0] = 17
        self.assertEqual(uncached.get_conv_state_tensor(base)[0, 0, 0].item(), 0)

        for _ in range(20):
            uncached.get_conv_state_tensor(base)
            uncached.get_ssm_state_tensor(base)
            cached.get_conv_state_tensor(base)
            cached.get_ssm_state_tensor(base)
        samples = {"uncached": [], "cached": []}
        for _ in range(50):
            for name, converter in (("uncached", uncached), ("cached", cached)):
                begin = time.thread_time_ns()
                for _ in range(100):
                    converter.get_conv_state_tensor(base)
                    converter.get_ssm_state_tensor(base)
                samples[name].append((time.thread_time_ns() - begin) / 100_000.0)
        uncached_us = statistics.median(samples["uncached"])
        cached_us = statistics.median(samples["cached"])
        print(
            "typed cache views CPU us/pair: "
            f"uncached={uncached_us:.3f}, cached={cached_us:.3f}"
        )
        self.assertLess(cached_us, uncached_us)


if __name__ == "__main__":
    main()
