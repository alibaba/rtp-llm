"""T24 platform-neutral EP strategy and decode-graph contracts.

These tests deliberately do not initialize distributed, DeepEP, Mega, PPU, or
CUDA.  They prove only selection policy and ownership invariants; a passing
test is not DP8 acceptance.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch
from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
    DSv4DecodeFmhaImpl,
    DSv4DecodeFmhaImplConfig,
)
from rtp_llm.models_py.modules.dsv4.fp8 import indexer
from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    DeepEPStrategy,
    GroupedFP4Strategy,
    LocalLoopStrategy,
    MegaMoEStrategy,
    MoeCfg,
    select_strategy,
)


def _cfg(ep_size: int) -> MoeCfg:
    local = 256 // max(ep_size, 1)
    return MoeCfg(
        layer_id=0,
        dim=7168,
        moe_inter_dim=2048,
        n_routed_experts=256,
        n_activated_experts=6,
        swiglu_limit=10.0,
        ep_size=ep_size,
        ep_rank=0,
        n_local_experts=local,
        local_expert_start=0,
        local_expert_end=local,
        max_tokens_per_rank=32,
    )


class T24StrategyAndGraphContractTest(unittest.TestCase):
    _STRATEGY_ENV = (
        "DSV4_MOE_STRATEGY",
        "DSV4_USE_MEGA_MOE",
        "DSV4_USE_MEGA_MOE_SE",
        "DSV4_USE_MEGA_MOE_FUSED",
        "DSV4_USE_GROUPED_FP4",
    )

    def setUp(self):
        self._saved_env = {key: os.environ.get(key) for key in self._STRATEGY_ENV}
        for key in self._STRATEGY_ENV:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_ep1_strategy_contract_has_no_dp_selector(self):
        # DP replication is outside this API. The contract is structural:
        # MoeCfg has no dp field, so DP cannot silently select a strategy.
        self.assertNotIn("dp_size", MoeCfg.__dataclass_fields__)
        with mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=False
        ), mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            self.assertIs(select_strategy(_cfg(1)), LocalLoopStrategy)

    def test_ep_gt1_without_mega_fails_closed_for_each_ep_size(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            for ep_size in (2, 4, 8):
                with self.subTest(ep_size=ep_size):
                    with self.assertRaisesRegex(
                        RuntimeError, "requires MegaMoEStrategy"
                    ):
                        select_strategy(_cfg(ep_size))

    def test_deepep_is_never_an_automatic_ep_strategy(self):
        # Even if a test double says DeepEP can handle the config, the policy
        # rejects it as an EP>1 forced route and auto-pick still requires Mega.
        with mock.patch.object(
            DeepEPStrategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            with self.assertRaisesRegex(
                RuntimeError, "fallback to DeepEP/LocalLoop is disabled"
            ):
                select_strategy(_cfg(4))
            with self.assertRaisesRegex(RuntimeError, "bypass Mega"):
                select_strategy(_cfg(4), forced="deepep")

    def test_decode_topk_capture_miss_is_fail_closed(self):
        # A graph capture must never allocate its workspace lazily.
        saved_cache = dict(indexer._decode_topk_workspace_cache)
        try:
            indexer._decode_topk_workspace_cache.clear()
            with mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "warmed before graph capture"
                ):
                    indexer._get_decode_topk_workspace(torch.device("cuda"))
        finally:
            indexer._decode_topk_workspace_cache.clear()
            indexer._decode_topk_workspace_cache.update(saved_cache)

    def test_each_graph_instance_owns_distinct_metadata(self):
        cfg = DSv4DecodeFmhaImplConfig(
            max_batch_size=2,
            q_len=1,
            window_size=8,
            head_dim=32,
            max_seq_len=64,
            compress_ratios=[4, 128],
            index_topk=4,
        )
        instances = [
            DSv4DecodeFmhaImpl(cfg, device=torch.device("cpu")) for _ in range(4)
        ]
        pointers = [impl.metadata.start_pos.data_ptr() for impl in instances]
        self.assertEqual(len(set(pointers)), len(pointers))
        before = [impl.metadata.start_pos.clone() for impl in instances]
        instances[2].prepare_cuda_graph(
            type(
                "Attn",
                (),
                {"sequence_lengths": torch.tensor([11, 13], dtype=torch.int32)},
            )()
        )
        self.assertTrue(
            torch.equal(
                instances[2].metadata.start_pos,
                torch.tensor([11, 13], dtype=torch.int32),
            )
        )
        with mock.patch.object(
            instances[2], "prepare", wraps=instances[2].prepare
        ) as prepare:
            instances[2].prepare_cuda_graph(
                type(
                    "Attn",
                    (),
                    {"sequence_lengths": torch.tensor([11, 13], dtype=torch.int32)},
                )()
            )
            self.assertTrue(prepare.call_args.kwargs["forbid_realloc"])
        for i, impl in enumerate(instances):
            if i != 2:
                self.assertTrue(torch.equal(impl.metadata.start_pos, before[i]))


if __name__ == "__main__":
    unittest.main()
