"""Resident weights across the MoE and dsv41 initialization-time weight transforms.

Each case runs the production transform twice in one process: once on a cold start, which
publishes the result, and once against the attached record, which must hand the same bytes back
without running the transform again. The weight dicts the two runs see differ exactly as they
do in production -- the restart gets back only what was published -- so a transform that reads
something it consumed last time fails here rather than in a model-level test.
"""

import hashlib
import os
import unittest
import uuid
from types import SimpleNamespace

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import FP4_BLOCK, FP8_BLOCK
from rtp_llm.models_py.modules.dsv41.utils import merge_v41_qkv_weights
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4 import (
    GroupedFp4Executor,
    _has_fp8_fp4_grouped_kernel,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _mega_intermediate_size,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.grouped_moe_equivalence_test import (
    _make_layer_weights,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    ChunkedFp8Fp4MoeLayer as MoE,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.weight_adapter import (
    adapt_split_moe_weights,
)
from rtp_llm.resident_weights import (
    GLOBAL_SCOPE,
    ResidentWeights,
    WeightDict,
    resident_init_tensors,
    store,
)
from rtp_llm.utils.model_weight import W

_FINGERPRINT = {"model_type": "fake_moe", "checkpoint_path": "/fake"}


class _ModelWeights:
    """The two attributes of ModelWeights that resident weights use."""

    def __init__(self, layers, global_weights):
        self.weights = layers
        self.global_weights = global_weights


def _bytes(tensor: torch.Tensor) -> bytes:
    """Every element's bytes in order, for any dtype and any layout.

    The MegaMoE scale factors are UTCCP-transposed and some have a trailing dimension of one,
    which counts as contiguous at any stride, so neither ``contiguous()`` nor ``view(uint8)``
    alone can flatten them. A fresh flat copy has a storage that is exactly its elements.
    """
    flat = tensor.detach().reshape(-1).clone()
    storage = flat.untyped_storage()
    return (
        torch.empty(0, dtype=torch.uint8, device=flat.device)
        .set_(storage, 0, (storage.nbytes(),), (1,))
        .cpu()
        .numpy()
        .tobytes()
    )


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_bytes(tensor)).hexdigest()


class _ResidentCase(unittest.TestCase):
    """One private store per test, and the two halves of a restart in one process."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this dedicated GPU Bazel target")

    def setUp(self):
        os.environ["RTP_LLM_RESIDENT_STORE"] = (
            f"rtp_llm_resident_moe_test.{uuid.uuid4().hex}"
        )
        self.store_name = store.default_store_name()
        store.ensure_running(self.store_name, owner_pid=os.getpid())

    def tearDown(self):
        try:
            store.stop_store(self.store_name)
        except ConnectionRefusedError:
            pass

    def _manager(self) -> ResidentWeights:
        manager = ResidentWeights("target", 0, dict(_FINGERPRINT), False)
        self.assertTrue(manager.ensure_store_running())
        return manager

    def _cold(self, layer_weights: dict):
        """Relocate a layer dict into resident memory, as T1 does, and return it bound."""
        manager = self._manager()
        weights = _ModelWeights(
            [WeightDict(0, layer_weights)], WeightDict(GLOBAL_SCOPE, {})
        )
        manager.move_to_vmm(weights)
        return manager, weights.weights[0]

    def _restart(self):
        """Attach the published record and return the layer dict a restart would be given."""
        manager = self._manager()
        restored = manager.attach()
        self.assertIsNotNone(restored, "nothing was published")
        weights = _ModelWeights([{}], {})
        for item in restored:
            target = (
                weights.global_weights
                if item.scope == GLOBAL_SCOPE
                else weights.weights[item.scope]
            )
            target[item.name] = item.tensor
        manager.bind(weights)
        return manager, weights.weights[0]

    def _assert_published(self):
        self.assertIn(
            f"{torch.cuda.get_device_properties(0).uuid}/target",
            store.list_store(self.store_name).slots,
        )


_CHECKPOINT_NAMES = {
    key: f"checkpoint.{key}"
    for key in (
        "router",
        "router_bias",
        "routed_gate",
        "routed_up",
        "routed_gate_scale",
        "routed_up_scale",
        "routed_down",
        "routed_down_scale",
        "shared_gate_up",
        "shared_gate_up_scale",
        "shared_down",
        "shared_down_scale",
    )
}
_EXPERTS, _INTER, _HIDDEN = 2, 32, 64
_ROUTED_GROUP = "test.routed_executor"
_ROUTED_CONSUMES = (W.moe_w1, W.moe_s1, W.moe_w2, W.moe_s2)
_ROUTED_NAMES = tuple(f"{_ROUTED_GROUP}.{n}" for n in ("w13", "s13", "w2", "s2"))


def _split_moe_layer_weights() -> dict:
    """Checkpoint-named split gate/up MoE weights, as the dsv4/dsv41 blocks are given them."""

    def packed(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.randint(
            -10, 10, (_EXPERTS, out_dim, in_dim // 2), dtype=torch.int8, device="cuda"
        )

    def scale(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.randint(
            124,
            127,
            (_EXPERTS, out_dim, in_dim // 32),
            dtype=torch.uint8,
            device="cuda",
        ).view(torch.float8_e8m0fnu)

    named = {
        "router": torch.randn(_EXPERTS, _HIDDEN, dtype=torch.bfloat16, device="cuda"),
        "router_bias": torch.zeros(_EXPERTS, device="cuda"),
        "routed_gate": packed(_INTER, _HIDDEN),
        "routed_up": packed(_INTER, _HIDDEN),
        "routed_gate_scale": scale(_INTER, _HIDDEN),
        "routed_up_scale": scale(_INTER, _HIDDEN),
        "routed_down": packed(_HIDDEN, _INTER),
        "routed_down_scale": scale(_HIDDEN, _INTER),
        "shared_gate_up": torch.randn(
            2 * _INTER, _HIDDEN, dtype=torch.bfloat16, device="cuda"
        ),
        "shared_gate_up_scale": torch.ones(1, 1, device="cuda"),
        "shared_down": torch.randn(
            _HIDDEN, _INTER, dtype=torch.bfloat16, device="cuda"
        ),
        "shared_down_scale": torch.ones(1, 1, device="cuda"),
    }
    return {_CHECKPOINT_NAMES[key]: value for key, value in named.items()}


def _consume_routed(layer_weights) -> tuple:
    """Stand-in for a fused executor: consumes the packed routed weights and keeps the result.

    Only the resident contract matters here -- which names the executor takes and how many
    tensors it produces -- so this does what every CUDA MoE executor does around its kernel
    specific work, without needing the kernel itself.
    """
    return resident_init_tensors(
        layer_weights,
        transform_key=_ROUTED_GROUP,
        output_names=_ROUTED_NAMES,
        consumed_names=_ROUTED_CONSUMES,
        builder=lambda: tuple(layer_weights.pop(name) for name in _ROUTED_CONSUMES),
    )


class SplitMoeAdapterResidentTest(_ResidentCase):
    def test_packing_runs_once_and_the_executor_result_is_replayed(self):
        torch.manual_seed(20261112)
        manager, layer = self._cold(_split_moe_layer_weights())
        adapted = adapt_split_moe_weights(layer, _INTER, 1, _CHECKPOINT_NAMES)
        self.assertIs(adapted, layer)
        self.assertIn(W.moe_w1, layer)
        self.assertIn(W.moe_gate, layer)
        self.assertIn(W.ffn_w13, layer)
        # Held as the executor holds them on itself: a published output must still be alive.
        cold_outputs = _consume_routed(layer)
        cold = [_digest(tensor) for tensor in cold_outputs]
        cold_shared = _digest(layer[W.ffn_w13])
        manager.finalize()
        self._assert_published()

        restarted, hot_layer = self._restart()
        # The checkpoint names the packing consumed are gone; the ones it only renamed are back.
        self.assertNotIn(_CHECKPOINT_NAMES["routed_gate"], hot_layer)
        self.assertIn(_CHECKPOINT_NAMES["router"], hot_layer)
        self.assertIn(_CHECKPOINT_NAMES["shared_gate_up"], hot_layer)
        adapt_split_moe_weights(hot_layer, _INTER, 1, _CHECKPOINT_NAMES)
        # Packing must not have run: its output is the executor's to hand back.
        self.assertNotIn(W.moe_w1, hot_layer)
        self.assertIn(W.moe_gate, hot_layer)
        hot_outputs = _consume_routed(hot_layer)
        self.assertEqual([_digest(tensor) for tensor in hot_outputs], cold)
        self.assertEqual(_digest(hot_layer[W.ffn_w13]), cold_shared)
        restarted.finalize()  # every published result was asked for exactly once
        del cold_outputs, hot_outputs

    def test_a_plain_dict_runs_the_packing_unchanged(self):
        torch.manual_seed(20261112)
        layer = _split_moe_layer_weights()
        expected = _digest(layer[_CHECKPOINT_NAMES["routed_down"]])
        adapt_split_moe_weights(layer, _INTER, 1, _CHECKPOINT_NAMES)
        self.assertIn(W.moe_w1, layer)
        self.assertEqual(_digest(layer[W.moe_w2]), expected)


_QKV_KEYS = ("wq_a.weight", "wq_a.scale", "wkv.weight", "wkv.scale")
_Q_ROWS, _KV_ROWS, _QKV_DIM = 64, 96, 128


def _qkv_weights() -> dict:
    weights = {}
    for prefix, rows in (("wq_a", _Q_ROWS), ("wkv", _KV_ROWS)):
        weights[f"{prefix}.weight"] = (
            torch.randn(rows, _QKV_DIM, device="cuda") * 0.04
        ).to(torch.float8_e4m3fn)
        weights[f"{prefix}.scale"] = torch.ones(
            rows // 32, _QKV_DIM // 32, device="cuda"
        ).to(torch.float8_e8m0fnu)
    return weights


class MergeV41QkvResidentTest(_ResidentCase):
    def test_the_merged_output_and_its_slices_survive_a_restart(self):
        torch.manual_seed(20261112)
        manager, layer = self._cold(_qkv_weights())
        linear = merge_v41_qkv_weights(layer, *_QKV_KEYS)
        self.assertIsNotNone(linear)
        cold = {key: _digest(layer[key]) for key in _QKV_KEYS}
        cold["merged"] = _digest(linear.weight)
        self.assertEqual(layer["wq_a.weight"].shape[0], _Q_ROWS)
        manager.finalize()
        self._assert_published()

        restarted, hot_layer = self._restart()
        # The loader names come back as the slices of the merged output they were re-pointed to,
        # which is what lets the merge read the same row counts again.
        self.assertEqual(hot_layer["wq_a.weight"].shape[0], _Q_ROWS)
        self.assertEqual(hot_layer["wkv.weight"].shape[0], _KV_ROWS)
        hot_linear = merge_v41_qkv_weights(hot_layer, *_QKV_KEYS)
        self.assertIsNotNone(hot_linear)
        hot = {key: _digest(hot_layer[key]) for key in _QKV_KEYS}
        hot["merged"] = _digest(hot_linear.weight)
        self.assertEqual(hot, cold)
        self.assertEqual(
            hot_linear.weight.untyped_storage().data_ptr(),
            hot_layer["wq_a.weight"].untyped_storage().data_ptr(),
        )
        restarted.finalize()

    def test_an_unsupported_merge_keeps_its_inputs_as_loaded_weights(self):
        layer = {key: torch.ones(32, 32, device="cuda") for key in _QKV_KEYS}
        manager, layer = self._cold(layer)
        self.assertIsNone(merge_v41_qkv_weights(layer, *_QKV_KEYS))
        cold = {key: _digest(layer[key]) for key in _QKV_KEYS}
        manager.finalize()
        self._assert_published()

        restarted, hot_layer = self._restart()
        self.assertEqual({key: _digest(hot_layer[key]) for key in _QKV_KEYS}, cold)
        self.assertIsNone(merge_v41_qkv_weights(hot_layer, *_QKV_KEYS))
        restarted.finalize()

    def test_a_plain_dict_merges_unchanged(self):
        torch.manual_seed(20261112)
        layer = _qkv_weights()
        linear = merge_v41_qkv_weights(layer, *_QKV_KEYS)
        self.assertIsNotNone(linear)
        self.assertEqual(layer["wq_a.weight"].shape[0], _Q_ROWS)
        self.assertEqual(
            linear.weight.untyped_storage().data_ptr(),
            layer["wq_a.weight"].untyped_storage().data_ptr(),
        )


class MoeExecutorResidentTest(_ResidentCase):
    """The production MoE layer, built twice from the same weights: cold, then from the record."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _has_fp8_fp4_grouped_kernel():
            raise AssertionError(
                "the SM100 grouped FP8xFP4 DeepGEMM kernel is required by this target"
            )

    def _build_moe(self, layer_weights, strategy: str) -> MoE:
        return MoE(
            moe_w1_layout="gate_up",
            layer_id=3,
            dim=512,
            moe_inter_dim=256,
            n_routed_experts=16,
            n_activated_experts=4,
            n_shared_experts=1,
            score_func="sqrtsoftplus",
            route_scale=1.0,
            swiglu_limit=10.0,
            n_hash_layers=0,
            vocab_size=32,
            layer_weights=layer_weights,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=128,
            strategy=strategy,
        )

    def _round_trip(self, strategy: str, executor_type) -> None:
        torch.manual_seed(20261112)
        weights = _make_layer_weights(16, 512, 256, "cuda")
        x = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        input_ids = torch.arange(8, dtype=torch.long, device="cuda")

        manager, layer = self._cold(dict(weights))
        moe = self._build_moe(layer, strategy)
        self.assertIsInstance(moe.fused_moe.fused_experts, executor_type)
        with torch.inference_mode():
            cold_out = moe(x, input_ids).clone()
        self.assertGreater(cold_out.float().abs().mean().item(), 1.0e-3)
        manager.finalize()
        self._assert_published()
        del moe, layer

        restarted, hot_layer = self._restart()
        hot_moe = self._build_moe(hot_layer, strategy)
        self.assertIsInstance(hot_moe.fused_moe.fused_experts, executor_type)
        with torch.inference_mode():
            hot_out = hot_moe(x, input_ids)
        restarted.finalize()  # every published result was asked for exactly once
        self.assertEqual(_bytes(hot_out), _bytes(cold_out))

    def test_grouped_fp4_executor_replays_its_transformed_weights(self):
        self._round_trip("grouped_fp4", GroupedFp4Executor)

    def test_local_loop_executor_replays_its_transformed_weights(self):
        self._round_trip("local_loop", LocalLoopExecutor)

    def test_a_plain_dict_builds_the_layer_unchanged(self):
        torch.manual_seed(20261112)
        weights = _make_layer_weights(16, 512, 256, "cuda")
        x = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        input_ids = torch.arange(8, dtype=torch.long, device="cuda")
        moe = self._build_moe(dict(weights), "grouped_fp4")
        with torch.inference_mode():
            out = moe(x, input_ids)
        self.assertEqual(out.shape, x.shape)
        self.assertGreater(out.float().abs().mean().item(), 1.0e-3)


_MEGA_DIM, _MEGA_INTER, _MEGA_EXPERTS = 512, 512, 4


def _mega_cfg() -> SimpleNamespace:
    """The configuration fields MegaMoE's two weight transforms read.

    Everything EP-related belongs to the second half of ``setup_weights``, which allocates the
    symmetric buffer and needs more than one GPU; the transforms below take ``deep_gemm`` as an
    argument and touch no distributed state at all.
    """
    return SimpleNamespace(
        dim=_MEGA_DIM,
        moe_inter_dim=_MEGA_INTER,
        moe_w1_layout="gate_up",
        n_local_experts=_MEGA_EXPERTS,
        n_routed_experts=_MEGA_EXPERTS,
        n_activated_experts=2,
        ep_size=1,
        n_shared_experts=1,
        shared_fp8_block_size=FP8_BLOCK,
    )


def _mega_layer_weights(inter: int) -> dict:
    """Checkpoint-named weights shaped for MegaMoE-SE: FP4 routed experts, FP8 shared expert."""

    def fp4(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.randint(
            -10,
            10,
            (_MEGA_EXPERTS, out_dim, in_dim // 2),
            dtype=torch.int8,
            device="cuda",
        )

    def fp4_scale(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.randint(
            124,
            127,
            (_MEGA_EXPERTS, out_dim, in_dim // FP4_BLOCK),
            dtype=torch.uint8,
            device="cuda",
        ).view(torch.float8_e8m0fnu)

    def fp8(rows: int, cols: int) -> torch.Tensor:
        return (torch.randn(rows, cols, device="cuda") * 0.04).to(torch.float8_e4m3fn)

    def fp8_scale(rows: int, cols: int) -> torch.Tensor:
        blocks = (
            (rows + FP8_BLOCK - 1) // FP8_BLOCK,
            (cols + FP8_BLOCK - 1) // FP8_BLOCK,
        )
        return torch.randint(124, 127, blocks, dtype=torch.uint8, device="cuda").view(
            torch.float8_e8m0fnu
        )

    named = {
        "router": torch.randn(
            _MEGA_EXPERTS, _MEGA_DIM, dtype=torch.bfloat16, device="cuda"
        ),
        "routed_gate": fp4(inter, _MEGA_DIM),
        "routed_up": fp4(inter, _MEGA_DIM),
        "routed_gate_scale": fp4_scale(inter, _MEGA_DIM),
        "routed_up_scale": fp4_scale(inter, _MEGA_DIM),
        "routed_down": fp4(_MEGA_DIM, inter),
        "routed_down_scale": fp4_scale(_MEGA_DIM, inter),
        "shared_gate_up": fp8(2 * inter, _MEGA_DIM),
        "shared_gate_up_scale": fp8_scale(2 * inter, _MEGA_DIM),
        "shared_down": fp8(_MEGA_DIM, inter),
        "shared_down_scale": fp8_scale(_MEGA_DIM, inter),
    }
    return {_CHECKPOINT_NAMES[key]: value for key, value in named.items()}


def _mega_executor(cfg: SimpleNamespace) -> MegaMoeSEExecutor:
    executor = MegaMoeSEExecutor.__new__(MegaMoeSEExecutor)
    torch.nn.Module.__init__(executor)
    executor.cfg = cfg
    return executor


def _run_mega_transforms(executor, layer_weights, deep_gemm, inter: int) -> list:
    """The weight-transforming half of MegaMoE-SE's setup_weights, unchanged between the paths."""
    adapt_split_moe_weights(
        layer_weights, _MEGA_INTER, executor.cfg.n_shared_experts, _CHECKPOINT_NAMES
    )
    executor._transform_routed_weights(layer_weights, deep_gemm, inter)
    executor._setup_shared_expert_weights(layer_weights, deep_gemm, _MEGA_DIM, inter)
    return [
        executor._mega_l1_w,
        executor._mega_l1_sf,
        executor._mega_l2_w,
        executor._mega_l2_sf,
        executor._se_l1_w,
        executor._se_l1_sf,
        executor._se_l2_w,
        executor._se_l2_sf,
    ]


class MegaMoeSEResidentTest(_ResidentCase):
    """MegaMoE-SE is the only path where a renamed loader weight is consumed by a transform."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import deep_gemm

        for name in (
            "transform_weights_for_mega_moe",
            "transform_sf_into_required_layout",
        ):
            if not hasattr(deep_gemm, name):
                raise AssertionError(f"deep_gemm.{name} is required by this target")

    def test_routed_and_shared_transforms_replay_after_a_restart(self):
        import deep_gemm

        cfg = _mega_cfg()
        inter = _mega_intermediate_size(cfg)
        self.assertEqual(inter, _MEGA_INTER)  # the shapes below assume no padding
        torch.manual_seed(20261112)

        manager, layer = self._cold(_mega_layer_weights(inter))
        cold_executor = _mega_executor(cfg)
        cold = [
            _digest(tensor)
            for tensor in _run_mega_transforms(cold_executor, layer, deep_gemm, inter)
        ]
        manager.finalize()
        self._assert_published()

        restarted, hot_layer = self._restart()
        # The shared expert was consumed under its canonical name, so the checkpoint name it was
        # renamed from is not in the record either: only the view signature linked the two.
        self.assertNotIn(_CHECKPOINT_NAMES["shared_gate_up"], hot_layer)
        self.assertNotIn(_CHECKPOINT_NAMES["routed_gate"], hot_layer)
        self.assertIn(_CHECKPOINT_NAMES["router"], hot_layer)
        hot_executor = _mega_executor(cfg)
        hot = [
            _digest(tensor)
            for tensor in _run_mega_transforms(
                hot_executor, hot_layer, deep_gemm, inter
            )
        ]
        restarted.finalize()  # every published result was asked for exactly once
        self.assertEqual(hot, cold)
        del cold_executor, hot_executor

    def test_a_plain_dict_runs_both_transforms_unchanged(self):
        import deep_gemm

        cfg = _mega_cfg()
        inter = _mega_intermediate_size(cfg)
        torch.manual_seed(20261112)
        executor = _mega_executor(cfg)
        outputs = _run_mega_transforms(
            executor, _mega_layer_weights(inter), deep_gemm, inter
        )
        self.assertEqual(len(outputs), 8)
        self.assertTrue(all(tensor.is_cuda for tensor in outputs))


if __name__ == "__main__":
    unittest.main()
