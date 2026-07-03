from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.distributed_attention_buffer import (
    Dsv4CpAttentionBufferSpec,
    build_dsv4_cp_attention_buffer_spec,
    clear_dsv4_cp_attention_buffer_cache,
    get_or_create_dsv4_cp_attention_buffer,
)


class _FakeCommunicator:
    instances: list["_FakeCommunicator"] = []

    def __init__(self, group, local_rank: int, world_size: int, buffer_size: int):
        self.group = group
        self.local_rank = local_rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self.per_rank_buffer_size = buffer_size // world_size
        self._communicator_ptr = 0xCAFE0000 + local_rank
        self._ub_handle = 7
        self._gpu_ptr_handle = 3
        self._rank_offset_lists = [
            rank * self.per_rank_buffer_size for rank in range(world_size)
        ]
        self.__class__.instances.append(self)


class Dsv4CpAttentionBufferSpecTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_dsv4_cp_attention_buffer_cache()
        _FakeCommunicator.instances.clear()

    def test_legacy_spec_aligns_single_reused_stage_region(self) -> None:
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=8,
            max_tokens_per_rank=17,
            batch_cap=3,
            swa_bytes_per_token=1024,
            main_bytes_per_token=584,
            indexer_bytes_per_token=132,
            mega_compressed_topk_cap=0,
            protocol_bytes_per_rank=4096,
            align_bytes=256,
        )

        # SWA is the largest stage here: 17 * 1024 + 4096 = 21488, aligned.
        self.assertEqual(spec.stage_bytes_per_rank, 17 * 1024)
        self.assertEqual(spec.per_rank_bytes, 21504)
        self.assertEqual(spec.total_bytes, 21504 * 8)

    def test_default_capacity_covers_raw_bf16_projection_payloads(self) -> None:
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=8,
            max_tokens_per_rank=17,
            batch_cap=3,
            swa_bytes_per_token=1024,
            mega_compressed_topk_cap=0,
            protocol_bytes_per_rank=4096,
            align_bytes=256,
        )

        # Compressor payloads are gathered before being packed into 584B/132B
        # cache rows; size the shared stage for raw BF16 overlap=True tensors.
        self.assertEqual(spec.stage_bytes_per_rank, 17 * 2048)
        self.assertEqual(spec.per_rank_bytes, 38912)
        self.assertEqual(spec.total_bytes, 38912 * 8)

    def test_default_builder_covers_mega_side_effect_scratch(self) -> None:
        spec = build_dsv4_cp_attention_buffer_spec(
            cp_size=8,
            actual_tokens_per_rank=2,
            batch_size=1,
            swa_bytes_per_token=1024,
            page_rr=False,
        )

        # Mirrors the failing service shape at layer2/CSA: persistent fresh-K
        # plus semantic indexer-K starts scratch at 10496.  The mega side-effect
        # region then needs 21248B, so the old 14592B/rank allocation was too
        # small.  The default builder must cover this before any layer enters
        # the in-kernel CP protocol.
        self.assertGreaterEqual(spec.per_rank_bytes, 10496 + 21248)
        self.assertEqual(spec.per_rank_bytes, 4_226_048)

    def test_service_shape_can_disable_splitk_slack_for_exact_contract(self) -> None:
        with patch.dict(
            os.environ,
            {"DSV4_CP_DISTRIBUTED_PREFILL_ATTN_SPLITK_SCRATCH_BYTES": "0"},
        ):
            spec = build_dsv4_cp_attention_buffer_spec(
                cp_size=8,
                actual_tokens_per_rank=2,
                batch_size=1,
                swa_bytes_per_token=1024,
                page_rr=False,
            )

        self.assertEqual(spec.stage_bytes_per_rank, 27392)
        self.assertEqual(spec.per_rank_bytes, 31744)

    def test_spec_rejects_non_v0_geometry_before_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "cp_size=8"):
            Dsv4CpAttentionBufferSpec(
                cp_size=4,
                max_tokens_per_rank=16,
                batch_cap=1,
                swa_bytes_per_token=1024,
            )

        with self.assertRaisesRegex(ValueError, "page/RR"):
            Dsv4CpAttentionBufferSpec(
                cp_size=8,
                max_tokens_per_rank=16,
                batch_cap=1,
                swa_bytes_per_token=1024,
                page_rr=True,
            )

    def test_request_capacity_is_checked_before_barrier(self) -> None:
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=8,
            max_tokens_per_rank=16,
            batch_cap=2,
            swa_bytes_per_token=1024,
        )

        spec.validate_request(tokens_per_rank=0, batch_size=1)
        spec.validate_request(tokens_per_rank=16, batch_size=2)
        with self.assertRaisesRegex(ValueError, "tokens_per_rank=17"):
            spec.validate_request(tokens_per_rank=17, batch_size=1)
        with self.assertRaisesRegex(ValueError, "batch_size=3"):
            spec.validate_request(tokens_per_rank=1, batch_size=3)

    def test_env_token_cap_controls_request_acceptance(self) -> None:
        with patch.dict(
            os.environ, {"DSV4_CP_DISTRIBUTED_PREFILL_ATTN_MAX_TOKENS_PER_RANK": "32"}
        ):
            spec = build_dsv4_cp_attention_buffer_spec(
                cp_size=8,
                actual_tokens_per_rank=17,
                batch_size=2,
                swa_bytes_per_token=1024,
                page_rr=False,
            )
        self.assertEqual(spec.max_tokens_per_rank, 32)

        with patch.dict(
            os.environ, {"DSV4_CP_DISTRIBUTED_PREFILL_ATTN_MAX_TOKENS_PER_RANK": "16"}
        ):
            with self.assertRaisesRegex(ValueError, "tokens_per_rank=17"):
                build_dsv4_cp_attention_buffer_spec(
                    cp_size=8,
                    actual_tokens_per_rank=17,
                    batch_size=2,
                    swa_bytes_per_token=1024,
                    page_rr=False,
                )

    def test_cache_reuses_matching_communicator_and_exports_op_kwargs(self) -> None:
        group = object()
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=8,
            max_tokens_per_rank=16,
            batch_cap=2,
            swa_bytes_per_token=1024,
        )

        with (
            patch.dict(os.environ, {"LOCAL_RANK": ""}),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            first = get_or_create_dsv4_cp_attention_buffer(
                group=group,
                cp_rank=2,
                spec=spec,
                communicator_factory=_FakeCommunicator,
            )
            second = get_or_create_dsv4_cp_attention_buffer(
                group=group,
                cp_rank=2,
                spec=spec,
                communicator_factory=_FakeCommunicator,
            )

        self.assertIs(first, second)
        self.assertEqual(len(_FakeCommunicator.instances), 1)
        kwargs = first.op_kwargs(cp_rank=2)
        self.assertEqual(kwargs["cp_rank"], 2)
        self.assertEqual(kwargs["cp_size"], 8)
        self.assertEqual(kwargs["comm_ptr"], 0xCAFE0002)
        self.assertEqual(kwargs["buffer_handle"], 7)
        self.assertEqual(kwargs["signal_handle"], 3)
        self.assertEqual(kwargs["per_rank_buffer_bytes"], spec.per_rank_bytes)
        self.assertEqual(
            kwargs["rank_offsets"],
            [rank * spec.per_rank_bytes for rank in range(8)],
        )

    def test_communicator_uses_local_rank_env_but_op_kwargs_keep_cp_rank(self) -> None:
        group = object()
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=8,
            max_tokens_per_rank=16,
            batch_cap=2,
            swa_bytes_per_token=1024,
        )

        with patch.dict(os.environ, {"LOCAL_RANK": "1"}):
            buf = get_or_create_dsv4_cp_attention_buffer(
                group=group,
                cp_rank=5,
                spec=spec,
                communicator_factory=_FakeCommunicator,
            )

        self.assertEqual(_FakeCommunicator.instances[0].local_rank, 1)
        self.assertEqual(buf.op_kwargs(cp_rank=5)["cp_rank"], 5)
        self.assertEqual(buf.comm_ptr, 0xCAFE0001)


if __name__ == "__main__":
    unittest.main()
