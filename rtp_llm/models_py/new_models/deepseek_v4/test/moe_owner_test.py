"""Native-byte checkpoint ownership tests; no GPU kernel is substituted."""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4.moe import OwnedFp8Fp4MoE
from rtp_llm.models_py.registry import MODEL_REGISTRY


def _owner(**kwargs):
    defaults = dict(
        layer_id=1,
        dim=128,
        moe_inter_dim=128,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        score_func="softmax",
        vocab_size=17,
        prefix="ffn",
    )
    defaults.update(kwargs)
    return OwnedFp8Fp4MoE(**defaults)


class _Model(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.ffn = _owner(**config.args)


def _scales(shape, offset=0):
    return (
        (
            torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 5
            + 124
            + offset
        )
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )


def _weights(shared=True, shared_block=128, hash_gate=False):
    weights = {}
    for expert in range(4):
        for column, projection in enumerate(("w1", "w2", "w3")):
            stem = f"ffn.experts.{expert}.{projection}."
            weights[stem + "weight"] = (
                (torch.arange(128 * 64).reshape(128, 64) + expert + 7 * column)
                .to(torch.uint8)
                .view(torch.int8)
            )
            weights[stem + "scale"] = _scales((128, 4), expert + column)
    if shared:
        for projection in ("w1", "w2", "w3"):
            weights[f"ffn.shared_experts.{projection}.weight"] = (
                (torch.arange(128 * 128).reshape(128, 128) % 7 - 3)
                .float()
                .to(torch.float8_e4m3fn)
            )
            weights[f"ffn.shared_experts.{projection}.scale"] = _scales(
                (128 // shared_block, 128 // shared_block)
            )
    weights["ffn.gate.weight"] = torch.arange(4 * 128).reshape(4, 128).bfloat16() / 512
    if hash_gate:
        weights["ffn.gate.tid2eid"] = (torch.arange(34).reshape(17, 2) % 4).int()
    else:
        weights["ffn.gate.bias"] = torch.arange(4).float()
    return weights


class Fp8Fp4OwnerTest(unittest.TestCase):
    def test_stats_follow_actual_v4_and_v41_routes_across_chunks(self):
        from rtp_llm.models_py.eplb_controller import create_stage_expert_balancer
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
            ChunkedFp8Fp4MoeLayer,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import FusedMoeFactory
        from rtp_llm.models_py.new_models.deepseek_v41.moe import OwnedV41MoE

        class ExpertOutput(torch.nn.Module):
            # Keep the production hash gate, chunker, and selector. Substitute
            # only the SM100 expert GEMMs, and expose the IDs they consumed.
            strategy_name = "local_loop"
            includes_shared_expert = False
            supports_gate_pack = True

            def forward(self, hidden_states, topk_weights, topk_ids, activation):
                return topk_ids.sum(dim=1, keepdim=True).expand_as(hidden_states)

        for cls in (OwnedFp8Fp4MoE, OwnedV41MoE):
            with self.subTest(cls=cls):
                model = RtpModule()
                layer = model.ffn = cls(
                    1, 128, 128, 4, 2, 0, score_func="softmax", n_hash_layers=2,
                    vocab_size=17, ep_size=2, ep_rank=1, max_tokens_per_rank=2, prefix="ffn",
                )
                layer.load_weights(_weights(shared=False, hash_gate=True))
                layer.validate_weights_loaded()
                config = types.SimpleNamespace(
                    num_layers=2, moe_n_group=0, is_mtp=False, moe_layer_index=[1],
                    eplb_config=types.SimpleNamespace(enable_eplb=lambda: True, eplb_mode="STATS"),
                )
                loader = types.SimpleNamespace(
                    load_config=types.SimpleNamespace(parallelism_config=types.SimpleNamespace(pp_size=1))
                )
                controller = create_stage_expert_balancer(loader, model, config)
                self.assertTrue(controller.eplb_stats_only)
                with mock.patch.object(FusedMoeFactory, "create_fused_moe", return_value=ExpertOutput()):
                    layer._runtime = ChunkedFp8Fp4MoeLayer(weight_owner=layer, **layer._runtime_kwargs)
                logical = torch.zeros(2, 4, dtype=torch.int32)
                gpu_loads = torch.zeros(2, 2, dtype=torch.int32)
                model._newloader_eplb_bind_stats(logical, gpu_loads)
                # Chunk capacity 2 => routes [0,1],[2,3] then [2,3]. Both
                # independent counters use global layer ID 1; layer 0 is dense.
                with mock.patch.object(
                    layer.gate, "can_prepare_gate_payload",
                    side_effect=AssertionError("stats mode must observe the executed routes"),
                ):
                    output = layer(torch.zeros(3, 128, dtype=torch.bfloat16), torch.tensor([0, 1, 1]))
                self.assertEqual(output[:, 0].tolist(), [1, 5, 5])
                self.assertEqual(logical.tolist(), [[0, 0, 0, 0], [1, 1, 2, 2]])
                self.assertEqual(gpu_loads.tolist(), [[0, 0], [2, 4]])
                config.eplb_config.eplb_mode = "ALL"
                with self.assertRaisesRegex(ValueError, "not weight rebalancing"):
                    create_stage_expert_balancer(loader, model, config)

    def _load(self, weights=None, load_method="scratch", **kwargs):
        weights = _weights() if weights is None else weights
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            MODEL_REGISTRY, {"fp8_fp4_owner_fixture": _Model}
        ):
            save_file(weights, os.path.join(directory, "model.safetensors"))
            loader = NewModelLoader(
                types.SimpleNamespace(model_type="fp8_fp4_owner_fixture", args=kwargs),
                NewLoaderConfig(device="cpu", load_method=load_method),
                model_path=directory,
            )
            model = loader.load()
            self.assertEqual(loader._selected_load_method.value, load_method)
            return model.ffn

    def _assert_bytes(self, layer, checkpoint):
        for name in layer._expected:
            actual = layer._target(name)
            expected = checkpoint["ffn." + name]
            self.assertEqual(actual.dtype, expected.dtype)
            torch.testing.assert_close(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        self.assertTrue(layer.checkpoint_load_complete())
        self.assertIsNone(layer._runtime)

    def test_native_checkpoint_bytes_and_parameter_ownership(self):
        layer = self._load()
        self._assert_bytes(layer, _weights())
        self.assertIsInstance(layer.routed_w13, torch.nn.Parameter)
        self.assertEqual(layer.routed_w13.dtype, torch.int8)
        self.assertEqual(layer.routed_s13.dtype, torch.float8_e8m0fnu)
        self.assertEqual(layer.routed_w13.shape, (4, 256, 64))
        self.assertEqual(layer.routed_s13.shape, (4, 256, 4))

    def test_attention_tp_replicates_moe_without_incorrect_slicing(self):
        checkpoint = _weights()
        for tp_size in (1, 2, 4):
            for rank in range(tp_size):
                layer = self._load(checkpoint, tp_size=tp_size, tp_rank=rank)
                self._assert_bytes(layer, checkpoint)
                self.assertEqual(layer.routed_w13.shape, (4, 256, 64))

    def test_main_executor_gate_up_row_order_on_nonzero_ep_rank(self):
        checkpoint = _weights()
        owner = self._load(checkpoint, ep_size=2, ep_rank=1, tp_size=2, tp_rank=1)
        # Independent of the loader's target selector, inspect the storage the
        # existing kernel consumes: contiguous gate rows followed by up rows.
        for local, expert in enumerate((2, 3)):
            for rows, projection in ((slice(0, 128), "w1"), (slice(128, 256), "w3")):
                stem = f"ffn.experts.{expert}.{projection}."
                torch.testing.assert_close(
                    owner.routed_w13[local, rows], checkpoint[stem + "weight"]
                )
                torch.testing.assert_close(
                    owner.routed_s13[local, rows].view(torch.uint8),
                    checkpoint[stem + "scale"].view(torch.uint8),
                )

    def test_mega_weight_transform_required_before_gpu_initialization(self):
        backend = types.SimpleNamespace(
            transform_sf_into_required_layout=lambda: None,
            fp8_fp4_mega_moe=lambda: None,
            get_symm_buffer_for_mega_moe=lambda: None,
        )
        with mock.patch("torch.version.hip", None), mock.patch(
            "torch.cuda.get_device_capability", return_value=(10, 3)
        ), mock.patch("importlib.import_module", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "transform_weights_for_mega_moe"):
                _owner(ep_size=4, ep_rank=2).validate_runtime_device("cuda:0")

    def test_ep_slices_whole_experts_before_materialization(self):
        checkpoint = _weights()
        for ep_size in (1, 2, 4):
            for rank in range(ep_size):
                layer = self._load(checkpoint, ep_size=ep_size, ep_rank=rank)
                self._assert_bytes(layer, checkpoint)
                self.assertEqual(layer.routed_w13.shape[0], 4 // ep_size)
                for expert in range(4):
                    spec = layer.checkpoint_read_spec(
                        f"experts.{expert}.w1.weight", (128, 64)
                    )
                    self.assertEqual(
                        spec.load,
                        layer.expert_start
                        <= expert
                        < layer.expert_start + layer.n_local_experts,
                    )

    def test_native_shared_mxfp8_block32_bytes(self):
        checkpoint = _weights(shared_block=32)
        layer = self._load(checkpoint, shared_fp8_block_size=32)
        self._assert_bytes(layer, checkpoint)
        self.assertEqual(layer.shared_s13.shape, (8, 4))

    def test_native_fst_mixed_fp4_ue8m0_expert_checkpoint(self):
        checkpoint = _weights()
        if os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0":
            with self.assertRaisesRegex(
                RuntimeError, "fastsafetensors prerequisite failed"
            ):
                self._load(checkpoint, load_method="fastsafetensors")
            return
        layer = self._load(checkpoint, load_method="fastsafetensors")
        # The file and its FST reader have gone away; the owner must still
        # contain the exact packed bytes, including non-representable scales.
        self._assert_bytes(layer, checkpoint)

    def test_hash_router_int32_and_no_shared(self):
        checkpoint = _weights(shared=False, hash_gate=True)
        layer = self._load(checkpoint, n_shared_experts=0, n_hash_layers=2)
        self._assert_bytes(layer, checkpoint)
        self.assertEqual(layer.router_tid2eid.dtype, torch.int32)
        self.assertFalse(hasattr(layer, "router_bias"))

    def test_all_executor_weights_preserve_old_and_direct_transfer_semantics(self):
        from rtp_llm.models_py.modules.fp8_fp4_weight_source import (
            OwnerWeightSource,
            take_weight,
        )
        from rtp_llm.utils.model_weight import W

        owner = self._load()
        source = OwnerWeightSource(owner)
        pairs = (
            ("routed_w13", W.moe_w1),
            ("routed_s13", W.moe_s1),
            ("routed_w2", W.moe_w2),
            ("routed_s2", W.moe_s2),
            ("shared_w13", W.ffn_w13),
            ("shared_s13", W.ffn_s13),
            ("shared_w2", W.ffn_w2),
            ("shared_s2", W.ffn_s2),
        )
        legacy = {key: getattr(owner, name) for name, key in pairs}
        for name, key in pairs:
            old = take_weight(legacy, name)
            self.assertNotIn(key, legacy)
            self.assertIs(old, take_weight(source, name))
            self.assertIsNone(getattr(owner, name))
        self.assertEqual(legacy, {})
        self.assertIsNotNone(owner.router_weight)

    def test_release_int64_hash_table_both_loaders_and_invalid_index(self):
        ckpt = _weights(hash_gate=True)
        table = ckpt["ffn.gate.tid2eid"].long()
        ckpt["ffn.gate.tid2eid"] = table
        methods = (
            ("scratch",)
            if os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0"
            else ("scratch", "fastsafetensors")
        )
        for method in methods:
            with self.subTest(method=method):
                layer = self._load(ckpt, load_method=method, n_hash_layers=2)
                torch.testing.assert_close(layer.router_tid2eid, table.int())
        table[0, 0] = 1 << 32
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self._load(ckpt, n_hash_layers=2)

    def test_missing_scale_fails(self):
        checkpoint = _weights()
        del checkpoint["ffn.experts.1.w1.scale"]
        with self.assertRaisesRegex(RuntimeError, "Missing FP8/FP4"):
            self._load(checkpoint)

    def test_executor_transfers_raw_storage_exactly_once(self):
        import gc
        import weakref

        from rtp_llm.models_py.modules.fp8_fp4_weight_source import (
            OwnerWeightSource,
            take_weight,
        )

        layer = self._load()
        ref = weakref.ref(layer.routed_w13)
        address = layer.routed_w13.data_ptr()
        source = OwnerWeightSource(layer)
        tensor = take_weight(source, "routed_w13")
        self.assertEqual(tensor.data_ptr(), address)
        self.assertIsNone(layer.routed_w13)
        self.assertIs(ref(), tensor)
        with self.assertRaisesRegex(RuntimeError, "already took"):
            take_weight(source, "routed_w13")
        with self.assertRaisesRegex(RuntimeError, "reload the model"):
            layer.initialize_runtime()
        del tensor
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNotNone(layer.routed_w2)
        self.assertIsNotNone(layer.router_weight)

    def test_wrong_packed_dtype_and_scale_geometry_fail(self):
        checkpoint = _weights()
        checkpoint["ffn.experts.0.w1.weight"] = checkpoint[
            "ffn.experts.0.w1.weight"
        ].view(torch.uint8)
        with self.assertRaisesRegex(TypeError, "Dtype mismatch"):
            self._load(checkpoint)
        checkpoint = _weights()
        checkpoint["ffn.experts.0.w1.scale"] = _scales((128, 8))
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            self._load(checkpoint)

    def test_duplicate_tensor_and_cpu_forward_rejected(self):
        layer = _owner()
        value = {"gate.weight": _weights()["ffn.gate.weight"]}
        layer.load_weights(value)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            layer.load_weights(value)
        with self.assertRaisesRegex(RuntimeError, "accelerator post-load"):
            loaded = self._load()
            loaded.configure_runtime(is_decode_role=False, max_tokens_per_rank=16)
            loaded.forward(torch.zeros(1, 128))

    def test_runtime_configuration_is_delayed_until_model_initialize(self):
        layer = self._load()
        self.assertFalse(layer._runtime_configured)
        with self.assertRaisesRegex(RuntimeError, "configure_runtime"):
            layer.forward(torch.zeros(1, 128))
        layer.configure_runtime(is_decode_role=True, max_tokens_per_rank=8)
        self.assertTrue(layer._runtime_kwargs["is_decode_role"])
        self.assertEqual(layer._runtime_kwargs["max_tokens_per_rank"], 8)
        self.assertIsNone(layer._runtime)

    def test_grouped_backend_checks_dense_fallback_before_loading(self):
        layer = _owner(strategy="grouped_fp4")
        backend = types.SimpleNamespace(
            transform_sf_into_required_layout=lambda *args: None,
            m_grouped_fp8_fp4_gemm_nt_contiguous=lambda *args: None,
            get_mk_alignment_for_contiguous_layout=lambda *args: None,
        )
        with mock.patch("torch.version.hip", None), mock.patch(
            "torch.cuda.get_device_capability", return_value=(10, 0)
        ), mock.patch("importlib.import_module", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "fp8_fp4_gemm_nt"):
                layer.validate_runtime_device("cuda:0")

    def test_backend_import_error_is_load_time_error(self):
        with mock.patch("torch.version.hip", None), mock.patch(
            "torch.cuda.get_device_capability", return_value=(10, 0)
        ), mock.patch("importlib.import_module", side_effect=OSError("ABI mismatch")):
            with self.assertRaisesRegex(RuntimeError, "cannot import DeepGEMM"):
                _owner().validate_runtime_device("cuda:0")

    def test_h20_is_not_claimed_as_fp8_fp4_kernel_verification(self):
        with mock.patch("torch.version.hip", None), mock.patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ):
            with self.assertRaisesRegex(RuntimeError, "SM100"):
                _owner().validate_runtime_device("cuda:0")


if __name__ == "__main__":
    unittest.main()
