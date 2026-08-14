import itertools
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.modules import GroupTopK


class GroupTopKTest(TestCase):
    NUM_EXPERTS = 256
    NUM_GROUPS = 8
    TOPK_GROUP = 4
    TOPK = 8

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    def _empty_outputs(
        self, tokens: int, index_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.empty(
            (tokens, self.TOPK), dtype=torch.float32, device="cuda"
        )
        indices = torch.empty(
            (tokens, self.TOPK), dtype=index_dtype, device="cuda"
        )
        return values, indices

    def _call(
        self,
        group_topk: GroupTopK,
        implementation: str,
        logits: torch.Tensor,
        correction_bias: torch.Tensor,
        index_dtype: torch.dtype,
        renormalize: bool,
        routed_scaling_factor: float,
        n_group: int | None = None,
        topk_group: int | None = None,
        topk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_group = self.NUM_GROUPS if n_group is None else n_group
        topk_group = self.TOPK_GROUP if topk_group is None else topk_group
        topk = self.TOPK if topk is None else topk
        values = torch.empty((logits.shape[0], topk), dtype=torch.float32, device="cuda")
        indices = torch.empty((logits.shape[0], topk), dtype=index_dtype, device="cuda")
        getattr(group_topk, implementation)(
            topk_weights=values,
            topk_ids=indices,
            scores=logits,
            correction_bias=correction_bias,
            n_group=n_group,
            topk_group=topk_group,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )
        return values, indices

    def _assert_strict_match(
        self,
        logits: torch.Tensor,
        correction_bias: torch.Tensor,
        index_dtype: torch.dtype,
        renormalize: bool,
        routed_scaling_factor: float,
    ) -> None:
        group_topk = GroupTopK(use_fused=True)
        legacy_values, legacy_indices = self._call(
            group_topk,
            "forward_legacy",
            logits,
            correction_bias,
            index_dtype,
            renormalize,
            routed_scaling_factor,
        )
        fused_values, fused_indices = self._call(
            group_topk,
            "forward_fused",
            logits,
            correction_bias,
            index_dtype,
            renormalize,
            routed_scaling_factor,
        )
        torch.testing.assert_close(
            fused_values,
            legacy_values,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        torch.testing.assert_close(fused_indices, legacy_indices, rtol=0, atol=0)

    def test_bf16_matches_legacy_bit_exact(self) -> None:
        token_counts = (0, 1, 7, 16, 17, 33, 64, 257, 6954, 8192)
        seeds = (0, 20260806)
        index_dtypes = (torch.int32, torch.int64)
        renormalize_values = (False, True)
        scaling_factors = (1.0, 2.5, 1.234567)

        for tokens, seed, index_dtype, renormalize, scale in itertools.product(
            token_counts,
            seeds,
            index_dtypes,
            renormalize_values,
            scaling_factors,
        ):
            with self.subTest(
                tokens=tokens,
                seed=seed,
                index_dtype=index_dtype,
                renormalize=renormalize,
                scale=scale,
            ):
                torch.manual_seed(seed)
                logits = torch.randn(
                    (tokens, self.NUM_EXPERTS),
                    dtype=torch.float32,
                    device="cuda",
                ).to(torch.bfloat16)
                correction_bias = torch.randn(
                    (self.NUM_EXPERTS,), dtype=torch.float32, device="cuda"
                )
                self._assert_strict_match(
                    logits,
                    correction_bias,
                    index_dtype,
                    renormalize,
                    scale,
                )

    def test_ties_nonfinite_and_fallback_match_legacy(self) -> None:
        equal_logits = torch.zeros(
            (33, self.NUM_EXPERTS), dtype=torch.bfloat16, device="cuda"
        )
        zero_bias = torch.zeros(
            (self.NUM_EXPERTS,), dtype=torch.float32, device="cuda"
        )
        self._assert_strict_match(
            equal_logits, zero_bias, torch.int64, True, 2.5
        )

        pattern = torch.tensor(
            [
                float("-inf"),
                -100.0,
                -0.0,
                0.0,
                100.0,
                float("inf"),
                float("nan"),
            ],
            dtype=torch.float32,
            device="cuda",
        )
        repeats = (4 * self.NUM_EXPERTS + pattern.numel() - 1) // pattern.numel()
        nonfinite_logits = pattern.repeat(repeats)[: 4 * self.NUM_EXPERTS]
        nonfinite_logits = nonfinite_logits.view(4, self.NUM_EXPERTS).to(
            torch.bfloat16
        )
        nonfinite_bias = torch.linspace(
            -1.0, 1.0, self.NUM_EXPERTS, dtype=torch.float32, device="cuda"
        )
        nonfinite_bias[::31] = float("nan")
        nonfinite_bias[1::37] = float("inf")
        nonfinite_bias[2::41] = float("-inf")
        self._assert_strict_match(
            nonfinite_logits, nonfinite_bias, torch.int32, False, 1.234567
        )

        fallback_logits = torch.full(
            (17, self.NUM_EXPERTS),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._assert_strict_match(
            fallback_logits, zero_bias, torch.int64, True, 2.5
        )
        fallback_values, fallback_indices = self._call(
            GroupTopK(use_fused=True),
            "forward_fused",
            fallback_logits,
            zero_bias,
            torch.int64,
            True,
            2.5,
        )
        torch.testing.assert_close(
            fallback_values,
            torch.full_like(fallback_values, 1.0 / self.TOPK),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            fallback_indices,
            torch.arange(self.TOPK, dtype=torch.int64, device="cuda")
            .unsqueeze(0)
            .expand_as(fallback_indices),
            rtol=0,
            atol=0,
        )

    def test_support_gate_and_fp32_fallback(self) -> None:
        group_topk = GroupTopK(use_fused=True)
        bf16_logits = torch.randn(
            (7, self.NUM_EXPERTS), dtype=torch.bfloat16, device="cuda"
        )
        fp32_logits = bf16_logits.float()
        correction_bias = torch.randn(
            (self.NUM_EXPERTS,), dtype=torch.float32, device="cuda"
        )
        values, indices = self._empty_outputs(7, torch.int64)

        self.assertTrue(
            group_topk.can_use_fused(
                values,
                indices,
                bf16_logits,
                correction_bias,
                self.NUM_GROUPS,
                self.TOPK_GROUP,
                self.TOPK,
            )
        )
        self.assertFalse(
            group_topk.can_use_fused(
                values,
                indices,
                bf16_logits,
                correction_bias,
                self.NUM_GROUPS,
                self.TOPK_GROUP,
                self.TOPK,
                use_fused=False,
            )
        )
        disabled_group_topk = GroupTopK(use_fused=False)
        self.assertFalse(
            disabled_group_topk.can_use_fused(
                values,
                indices,
                bf16_logits,
                correction_bias,
                self.NUM_GROUPS,
                self.TOPK_GROUP,
                self.TOPK,
                use_fused=True,
            )
        )
        self.assertFalse(
            group_topk.can_use_fused(
                values,
                indices,
                fp32_logits,
                correction_bias,
                self.NUM_GROUPS,
                self.TOPK_GROUP,
                self.TOPK,
            )
        )
        with self.assertRaises(ValueError):
            self._call(
                group_topk,
                "forward_fused",
                fp32_logits,
                correction_bias,
                torch.int64,
                True,
                1.0,
            )

        legacy_values, legacy_indices = self._call(
            group_topk,
            "forward_legacy",
            fp32_logits,
            correction_bias,
            torch.int64,
            True,
            1.0,
        )
        fallback_values, fallback_indices = self._call(
            group_topk,
            "forward",
            fp32_logits,
            correction_bias,
            torch.int64,
            True,
            1.0,
        )
        torch.testing.assert_close(
            fallback_values, legacy_values, rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            fallback_indices, legacy_indices, rtol=0, atol=0
        )

        noncontiguous_logits = torch.randn(
            (self.NUM_EXPERTS, 7), dtype=torch.bfloat16, device="cuda"
        ).transpose(0, 1)
        self.assertFalse(
            group_topk.can_use_fused(
                values,
                indices,
                noncontiguous_logits,
                correction_bias,
                self.NUM_GROUPS,
                self.TOPK_GROUP,
                self.TOPK,
            )
        )

    def test_cuda_graph_replay_matches_legacy(self) -> None:
        torch.manual_seed(20260806)
        logits = torch.randn(
            (33, self.NUM_EXPERTS), dtype=torch.bfloat16, device="cuda"
        )
        correction_bias = torch.randn(
            (self.NUM_EXPERTS,), dtype=torch.float32, device="cuda"
        )
        group_topk = GroupTopK(use_fused=True)
        legacy_values, legacy_indices = self._call(
            group_topk,
            "forward_legacy",
            logits,
            correction_bias,
            torch.int64,
            True,
            2.5,
        )
        graph_values, graph_indices = self._empty_outputs(33, torch.int64)

        group_topk.forward_fused(
            graph_values,
            graph_indices,
            logits,
            correction_bias,
            self.NUM_GROUPS,
            self.TOPK_GROUP,
            self.TOPK,
            True,
            2.5,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            group_topk.forward_fused(
                graph_values,
                graph_indices,
                logits,
                correction_bias,
                self.NUM_GROUPS,
                self.TOPK_GROUP,
                self.TOPK,
                True,
                2.5,
            )
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            graph_values, legacy_values, rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(graph_indices, legacy_indices, rtol=0, atol=0)


    def test_glm51_single_group_matches_legacy(self) -> None:
        """Official GLM-5.1 routing: n_group=1, topk_group=1, topk=8."""
        torch.manual_seed(20260814)
        group_topk = GroupTopK(use_fused=True)
        for tokens in (1, 7, 128, 6954):
            for index_dtype in (torch.int32, torch.int64):
                for renormalize in (False, True):
                    logits = torch.randn(
                        (tokens, self.NUM_EXPERTS), dtype=torch.bfloat16, device="cuda"
                    )
                    correction_bias = torch.randn(
                        (self.NUM_EXPERTS,), dtype=torch.float32, device="cuda"
                    )
                    legacy_values, legacy_indices = self._call(
                        group_topk,
                        "forward_legacy",
                        logits,
                        correction_bias,
                        index_dtype,
                        renormalize,
                        2.5,
                        n_group=1,
                        topk_group=1,
                        topk=8,
                    )
                    fused_values, fused_indices = self._call(
                        group_topk,
                        "forward_fused",
                        logits,
                        correction_bias,
                        index_dtype,
                        renormalize,
                        2.5,
                        n_group=1,
                        topk_group=1,
                        topk=8,
                    )
                    torch.testing.assert_close(
                        fused_values,
                        legacy_values,
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                    torch.testing.assert_close(fused_indices, legacy_indices, rtol=0, atol=0)
                    self.assertTrue(
                        group_topk.can_use_fused(
                            fused_values,
                            fused_indices,
                            logits,
                            correction_bias,
                            1,
                            1,
                            8,
                        )
                    )

if __name__ == "__main__":
    main()
