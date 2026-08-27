"""CPU contract tests for the DeepEP low-latency row-scatter finalize.

Proves the algebra the fused path relies on: because the TP token slice is a
partition, summing each rank's ``[shared partial | routed slice scattered into
its own rows]`` buffer produces the same result as the previous
``all_gather(routed) + all_reduce(shared)`` pair.
"""

from functools import partial
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ROW_SCATTER_READY_ARG,
    ROW_SCATTER_TARGET_ARG,
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
    DeepEpLowLatencyRouter,
)

HIDDEN = 6
TP_SIZE = 4
# Five tokens over four ranks is the awkward case: a ragged tail plus a trailing
# rank that owns no tokens at all.
RAGGED_TOKENS = 5
_ROUTER_MODULE = (
    "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers."
    "deepep_low_latency_router"
)


def _stub_router(tp_size, tp_rank):
    """Minimal stand-in for ``self``: this path only reads the TP config."""
    router = SimpleNamespace(config=SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank))
    router._tp_token_slice = partial(DeepEpLowLatencyRouter._tp_token_slice, router)
    return router


def _tp_slice(tp_size, tp_rank, num_tokens):
    return _stub_router(tp_size, tp_rank)._tp_token_slice(num_tokens)


def _row_scatter(
    tp_size, tp_rank, combined_x, target, original_num_tokens, ready_event=None
):
    return DeepEpLowLatencyRouter._finalize_row_scatter(
        _stub_router(tp_size, tp_rank),
        combined_x,
        target,
        original_num_tokens,
        ready_event,
    )


def _finalize_stub(tp_size, tp_rank, combine_output):
    """Extend the stub with what ``finalize`` reads, minus the DeepEP buffer."""
    router = _stub_router(tp_size, tp_rank)
    router._handle = ("handle",)
    router._zero_copy = False
    router._async_finish = False
    router._return_recv_hook = False
    router._use_accl_ep = False
    router._normal_finalize = lambda combine_args: combine_output
    router._finalize_row_scatter = partial(
        DeepEpLowLatencyRouter._finalize_row_scatter, router
    )
    router._finalize_post_tp_gather = partial(
        DeepEpLowLatencyRouter._finalize_post_tp_gather, router
    )
    return router


def _finalize(router, extra_finalize_args):
    return DeepEpLowLatencyRouter.finalize(
        router,
        CombineForwardPayload(fused_expert_output=torch.zeros(1, HIDDEN)),
        torch.ones(1, 2),
        torch.zeros(1, 2, dtype=torch.int32),
        False,
        extra_finalize_args,
    )


class DeepEpLowLatencyRowScatterTest(TestCase):
    def test_scatter_writes_only_this_rank_rows(self):
        for tp_rank in range(TP_SIZE):
            with self.subTest(tp_rank=tp_rank):
                begin, size = _tp_slice(TP_SIZE, tp_rank, RAGGED_TOKENS)
                target = torch.zeros(RAGGED_TOKENS, HIDDEN)

                result = _row_scatter(
                    TP_SIZE,
                    tp_rank,
                    torch.full((size, HIDDEN), 7.0),
                    target,
                    RAGGED_TOKENS,
                )

                self.assertIs(result, target)
                expected = torch.zeros(RAGGED_TOKENS, dtype=torch.bool)
                expected[begin : begin + size] = True
                torch.testing.assert_close(target.abs().sum(dim=1) > 0, expected)

    def test_rejects_a_combine_output_that_is_not_the_dispatched_slice(self):
        # Pins the assumption that combine returns exactly the dispatched rows;
        # a padded result would scatter into the wrong tokens.
        _, size = _tp_slice(TP_SIZE, 0, RAGGED_TOKENS)
        with self.assertRaisesRegex(AssertionError, "expected the dispatched slice"):
            _row_scatter(
                TP_SIZE,
                0,
                torch.zeros(size + 1, HIDDEN),
                torch.zeros(RAGGED_TOKENS, HIDDEN),
                RAGGED_TOKENS,
            )

    def test_joins_the_producer_before_touching_the_buffer(self):
        # The trailing rank writes nothing, but the caller still reduces the
        # whole buffer, so it cannot skip the join either.
        for tp_rank in (0, TP_SIZE - 1):
            with self.subTest(tp_rank=tp_rank):
                _, size = _tp_slice(TP_SIZE, tp_rank, RAGGED_TOKENS)
                target = torch.zeros(RAGGED_TOKENS, HIDDEN)
                event = object()
                waited = []

                def wait_event(seen):
                    waited.append(seen)
                    self.assertEqual(
                        target.abs().sum().item(), 0.0, "wrote before joining"
                    )

                stream = SimpleNamespace(wait_event=wait_event)
                with patch("torch.cuda.current_stream", return_value=stream):
                    _row_scatter(
                        TP_SIZE,
                        tp_rank,
                        torch.ones(size, HIDDEN),
                        target,
                        RAGGED_TOKENS,
                        event,
                    )

                self.assertEqual(waited, [event])
                self.assertEqual(target.abs().sum().item() > 0.0, size > 0)

    def test_fused_reduction_matches_all_gather_plus_all_reduce(self):
        # Token counts covering: exactly divisible, one ragged tail, and enough
        # tokens missing that trailing ranks dispatch nothing at all.
        for original_num_tokens in (1, 3, RAGGED_TOKENS, 8, 9):
            with self.subTest(original_num_tokens=original_num_tokens):
                torch.manual_seed(original_num_tokens)
                gate = torch.sigmoid(torch.randn(original_num_tokens, 1))
                shared_partials = [
                    torch.randn(original_num_tokens, HIDDEN) for _ in range(TP_SIZE)
                ]
                routed_slices = [
                    torch.randn(
                        _tp_slice(TP_SIZE, rank, original_num_tokens)[1], HIDDEN
                    )
                    for rank in range(TP_SIZE)
                ]

                # Baseline: gather the routed slices, reduce the shared partials
                # separately, then add the two complete branches. Concatenating
                # to exactly the token count also proves the slices partition.
                routed_full = torch.cat(routed_slices, dim=0)
                self.assertEqual(routed_full.shape[0], original_num_tokens)
                baseline = routed_full + gate * sum(shared_partials)

                # Fused: every rank scatter-adds its slice into its own gated
                # shared partial, and one reduction sums the buffers.
                fused = sum(
                    _row_scatter(
                        TP_SIZE,
                        tp_rank,
                        routed_slices[tp_rank],
                        gate * shared_partials[tp_rank],
                        original_num_tokens,
                    )
                    for tp_rank in range(TP_SIZE)
                )

                torch.testing.assert_close(fused, baseline)


class DeepEpLowLatencyFinalizeDispatchTest(TestCase):
    def test_router_advertises_row_scatter_support(self):
        # Instantiating the router needs a DeepEP buffer; the property does not
        # read anything a stub cannot supply.
        self.assertTrue(
            DeepEpLowLatencyRouter.supports_row_scatter_finalize.fget(
                _stub_router(TP_SIZE, 0)
            )
        )

    def test_a_target_is_filled_in_place_without_gathering(self):
        for tp_rank in range(TP_SIZE):
            with self.subTest(tp_rank=tp_rank):
                _, size = _tp_slice(TP_SIZE, tp_rank, RAGGED_TOKENS)
                router = _finalize_stub(
                    TP_SIZE, tp_rank, torch.full((size, HIDDEN), 7.0)
                )
                target = torch.zeros(RAGGED_TOKENS, HIDDEN)
                event = object()
                waited = []
                stream = SimpleNamespace(wait_event=waited.append)

                with (
                    patch(f"{_ROUTER_MODULE}.all_gather") as all_gather,
                    patch("torch.cuda.current_stream", return_value=stream),
                ):
                    result = _finalize(
                        router,
                        {
                            "original_num_tokens": RAGGED_TOKENS,
                            ROW_SCATTER_TARGET_ARG: target,
                            ROW_SCATTER_READY_ARG: event,
                        },
                    )

                all_gather.assert_not_called()
                self.assertEqual(waited, [event])
                self.assertIs(result, target)
                self.assertIsNone(router._handle)

    def test_without_a_target_the_slices_are_gathered(self):
        _, size = _tp_slice(TP_SIZE, 0, RAGGED_TOKENS)
        router = _finalize_stub(TP_SIZE, 0, torch.full((size, HIDDEN), 7.0))

        with patch(
            f"{_ROUTER_MODULE}.all_gather",
            side_effect=lambda tensor, group: torch.cat([tensor] * TP_SIZE),
        ) as all_gather:
            result = _finalize(router, {"original_num_tokens": RAGGED_TOKENS})

        all_gather.assert_called_once()
        self.assertIs(all_gather.call_args.kwargs["group"], Group.TP)
        self.assertEqual(result.shape, (RAGGED_TOKENS, HIDDEN))
        self.assertIsNone(router._handle)


if __name__ == "__main__":
    main()
