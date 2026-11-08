"""One checkpoint-derived ModelWeights -> inherited Block -> PPU smoke.

This is a local layer-component computation, not a 40-layer/PD acceptance.
The independent CPU IEEE reference and existing strict row/global SSE-energy
criterion (5e-5) are selected before launch. No CUDA model/kernel is an oracle.
"""

import hashlib
import json
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch
import torch.nn.functional as F

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.models.deepseek_v41 import DeepSeekV41
from rtp_llm.models_py.model_desc.deepseek_v41_cuda_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv41.block import Block
from rtp_llm.models_py.modules.dsv41.ppu.backend import get_backend, get_ops
from rtp_llm.models_py.modules.dsv41.ppu.block import SharedPPUBlock
from rtp_llm.models_py.modules.dsv41.ppu.model import DeepSeekV41PPUModel
from rtp_llm.models_py.modules.dsv41.ppu.transformer import SharedPPUTransformer
from rtp_llm.models_py.modules.dsv41.ppu.model_weights import DeepSeekV41PPUWeight
from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_probe import (
    OUTPUTS,
    _qualification_metric,
    independent_transition_reference,
)
from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer
from rtp_llm.ops import HWKernelConfig, KVCacheConfig, ParallelismConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W

CONFIG_SHA = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
INDEX_SHA = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"


class RecordingCheckpointSource(DatabaseTensorSource):
    """Real framework checkpoint source; retain only the selected raw tensors."""

    def __init__(self, database):
        super().__init__(database)
        self.raw = {}
        self.calls = []

    def load_tensor(self, name, data_type=None):
        self.calls.append(name)
        tensors = super().load_tensor(name, data_type)
        if len(tensors) != 1 or name in self.raw:
            raise AssertionError(f"checkpoint key missing or loaded twice: {name}")
        self.raw[name] = tensors[0]
        return tensors


def _norm_reference(x, weight, eps):
    x = x.float()
    return (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).to(torch.bfloat16)


def _dense_reference(raw, prefix):
    # Decode exponent bytes independently of the production PPU linear helper.
    value = raw[prefix + ".weight"].view(torch.float8_e4m3fn).float()
    scale = torch.exp2(raw[prefix + ".scale"].view(torch.uint8).float() - 127)
    scale = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
    return (value * scale[: value.shape[0], : value.shape[1]]).to(torch.bfloat16)


def _pre_reference(residual, fn, scale, base, config):
    # Independent full-width FP32 projection, sinkhorn and entry lane readout.
    hc = config.hc_mult
    flat = residual.flatten(1).float()
    mixes = (flat @ fn.float().T) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + config.rms_eps
    )
    scale, base = scale.flatten().float(), base.flatten().float()
    pre = (mixes[:, :hc] * scale[0] + base[:hc]).sigmoid() + config.hc_eps
    post = 2 * (mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]).sigmoid()
    comb = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)
    comb = comb.softmax(-1) + config.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    for _ in range(config.sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + config.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    return residual[:, 0].contiguous(), pre, post.unsqueeze(-1), comb


def _tensor_record(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "data_ptr": value.data_ptr(),
    }


def _assert_shared_dependency_adapter(test):
    """Check the actual live package/cache/alias before checkpoint loading."""
    import importlib.metadata as metadata

    import deep_gemm
    from deep_gemm import deep_gemm_cpp
    from deep_gemm.utils import layout

    expected_layout = "9211d06752fd1e5a915198227f9c1651d678894b311b3a48a45cf91e48f23381"
    expected_native = "816ff22272d7636c8ee57b960189ada869910af74016f35d53638248fcdc3942"
    variant = "pip_v41_ppu_shared_deep_gemm_" + expected_layout[:16]
    cache_key = f"deep_gemm_python-0.0.0__{variant}__801e73833647717c"
    root = Path(deep_gemm.__file__).resolve().parent
    native_path = Path(deep_gemm_cpp.__file__).resolve()
    layout_path = Path(layout.__file__).resolve()
    test.assertEqual(root.parent.name, "site-packages")
    test.assertEqual(root.parent.parent.name, cache_key, "old package cache selected")
    test.assertEqual(native_path.parent, root)
    test.assertEqual(layout_path, root / "utils/layout.py")
    test.assertEqual(native_path.stat().st_mode & 0o777, 0o755)
    test.assertEqual(native_path.stat().st_size, 2741136)
    test.assertEqual(layout_path.stat().st_mode & 0o777, 0o644)
    test.assertEqual(layout_path.stat().st_size, 666)
    test.assertIs(
        layout.get_mn_major_tma_aligned_packed_ue8m0_tensor,
        deep_gemm_cpp.get_mn_major_tma_aligned_packed_ue8m0_tensor,
    )
    test.assertTrue(callable(layout.get_mn_major_tma_aligned_packed_ue8m0_tensor))
    # Do not invoke the unsupported packing operation. The same native callable
    # retains its original error semantics; this path uses resident BF16.
    marker = root.parent.parent / ".copy_complete"
    test.assertEqual(marker.read_text(), "deep_gemm==0.0.0\n")
    manifest_path = root / "rtp_v41_ppu_shared_adapter.json"
    receipt = json.loads(manifest_path.read_text())
    test.assertEqual(receipt["schema"], "v41-ppu-shared-deepgemm-host-alias-v1")
    test.assertEqual(receipt["source_commit"], "07674e672082c2e06ea9f1ec1993d79684444a1c")
    test.assertEqual(receipt["version"], "0.0.0")
    test.assertEqual(receipt["variant"], variant)
    test.assertEqual(receipt["cache_key"], cache_key)
    test.assertEqual(receipt["layout_after_sha256"], expected_layout)
    test.assertEqual(receipt["native_sha256"], expected_native)
    measured = {}
    for path, expected in (
        (layout_path, expected_layout),
        (native_path, expected_native),
        (
            root / "__init__.py",
            "8d0883a94786920ec2588aa73fd25a947506daa424b92e85c8daa80123abfb97",
        ),
        (
            root / "utils/__init__.py",
            "bf31f6e9fc30482ef83a73d63fdb6ed4d46deec847b35c0f9d39816d9b923bb4",
        ),
    ):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        test.assertEqual(digest, expected, str(path))
        measured[str(path)] = digest
    header_root = root / "include/deep_gemm"
    test.assertTrue(header_root.is_dir())
    headers = hashlib.sha256()
    for header in sorted(header_root.rglob("*")):
        if header.is_file():
            headers.update(header.relative_to(header_root).as_posix().encode() + b"\0")
            headers.update(header.read_bytes() + b"\0")
    test.assertEqual(headers.hexdigest(), receipt["header_sha256"])
    test.assertTrue(headers.hexdigest().startswith("801e73833647717c"))
    distribution = metadata.distribution("deep_gemm")
    metadata_package = Path(distribution.locate_file("deep_gemm")).resolve()
    test.assertEqual(distribution.version, "0.0.0")
    test.assertEqual(
        next((part for part in metadata_package.parts if part.startswith("pip_")), None),
        variant,
        "package copier metadata selected the original distribution",
    )
    return {
        "actual_package": str(root),
        "actual_layout": str(layout_path),
        "actual_native": str(native_path),
        "actual_distribution_package": str(metadata_package),
        "cache_key": cache_key,
        "variant": variant,
        "measured_sha256": measured,
        "header_sha256": headers.hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "alias_is_existing_native_callable": True,
        "unsupported_packing_invoked": False,
    }


class TestSharedModelBoundaries(unittest.TestCase):
    def test_inherited_methods(self):
        self.assertTrue(issubclass(DeepSeekV41PPUModel, DeepSeekV41Model))
        self.assertIs(SharedPPUTransformer.forward, V4Transformer.forward)
        self.assertIs(SharedPPUTransformer.forward_decode, V4Transformer.forward_decode)
        self.assertIs(SharedPPUBlock.prefill_fast_attn_pre, Block.prefill_fast_attn_pre)
        self.assertIs(SharedPPUBlock.prefill_fast_ffn_body, Block.prefill_fast_ffn_body)

    def test_model_dispatch_and_block_delegation(self):
        from rtp_llm.models_py.modules.dsv41.ppu.decode_graph import SharedPPUDecodeGraphImpl

        runtime = Mock()
        graph_runtime = Mock()
        model = SimpleNamespace(shared_runtime=runtime, shared_graph_runtime=graph_runtime)
        forward_impl = DeepSeekV41PPUModel._forward_impl
        self.assertIs(forward_impl(model, sentinel.inputs), runtime.dispatch.return_value)
        runtime.dispatch.assert_called_once_with(sentinel.inputs, None)
        graph_meta = object.__new__(SharedPPUDecodeGraphImpl)
        self.assertIs(
            forward_impl(model, sentinel.inputs, graph_meta),
            graph_runtime.forward.return_value,
        )
        graph_runtime.forward.assert_called_once_with(sentinel.inputs, graph_meta)
        with self.assertRaises(TypeError):
            forward_impl(model, sentinel.inputs, sentinel.foreign_metadata)

        block = SharedPPUBlock.__new__(SharedPPUBlock)
        torch.nn.Module.__init__(block)
        block.layer_id, block.engram = 3, None
        adapted = (sentinel.rows, sentinel.ids, sentinel.positions, sentinel.cu)
        before = Mock(return_value=adapted)
        publication = Mock(return_value=nullcontext())
        state = SimpleNamespace(
            before_block=before,
            publication=publication,
            runtime=SimpleNamespace(policy=SimpleNamespace(mode="compact")),
        )
        block.attn = SimpleNamespace(runtime=SimpleNamespace(current=state))
        inputs = (sentinel.raw_rows, sentinel.raw_ids, sentinel.raw_positions, sentinel.raw_cu)
        with patch.object(Block, "forward", autospec=True, return_value=sentinel.output) as common:
            self.assertIs(block.forward(*inputs, expert_load=sentinel.load), sentinel.output)
        before.assert_called_once_with(block, *inputs)
        publication.assert_called_once_with(block.layer_id)
        common.assert_called_once_with(block, *adapted, expert_load=sentinel.load)
        self.assertEqual(block._mhc_phase, "ced")


class TestSharedModelWeightsDevice(unittest.TestCase):
    @torch.inference_mode()
    def test_real_checkpoint_shared_consumer(self):
        from rtp_llm.device.device_type import is_ppu

        self.assertTrue(is_ppu(), "this smoke requires a real PPU")
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 1, "bind physical GPU0 via gpu_lock")
        checkpoint = Path(os.environ["RTP_V41_TEST_CHECKPOINT"])
        for filename, digest in (
            ("config.json", CONFIG_SHA),
            ("model.safetensors.index.json", INDEX_SHA),
        ):
            self.assertEqual(
                hashlib.sha256((checkpoint / filename).read_bytes()).hexdigest(), digest
            )
        report = {
            "SMOKE": "NOT_PASSED",
            "FULL_VALIDATION": "DEFERRED",
            "scope": "layer0 mHC/norm/query/shared-expert, no routed-MoE or PD",
            "checkpoint": str(checkpoint),
            "config_sha256": CONFIG_SHA,
            "index_sha256": INDEX_SHA,
            "rows": 4,
            "oracle": "independent CPU FP32 equations + BF16 landing points",
            "criterion": "existing row/global SSE-energy strictly < 5e-5",
        }
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "shared-model.json"
        self.addCleanup(lambda: output.write_text(json.dumps(report, indent=2) + "\n"))
        report["dependency_adapter"] = _assert_shared_dependency_adapter(self)
        backend = get_backend()
        backend.check_capabilities(None)
        self.assertIs(backend.model_cls, DeepSeekV41PPUModel)
        self.assertIs(backend.weight_cls, DeepSeekV41PPUWeight)
        config = DeepSeekV41._create_config(str(checkpoint))
        config.ckpt_path = str(checkpoint)
        parallel = ParallelismConfig()
        self.assertEqual((parallel.ep_size, parallel.get_attn_tp_size()), (1, 1))
        descriptor = backend.weight_cls(
            config, parallel, HWKernelConfig(), KVCacheConfig()
        )
        database = CkptDatabase(str(checkpoint))
        descriptor.process_meta_from_ckpt(database.pretrain_file_list)
        info = descriptor.get_weight_info()
        self.assertEqual(len(info.layer_weights), 40)
        self.assertTrue(all(info.layer_weights), "shared descriptor graph was cleared")
        needed = {
            W.v4_attn_norm,
            W.v4_ffn_norm,
            W.v4_attn_q_norm,
            W.v4_attn_wq_a_w,
            W.v4_attn_wq_a_s,
            W.v4_shared_w13_w,
            W.v4_shared_w13_s,
            W.v4_shared_w2_w,
            W.v4_shared_w2_s,
            *(
                getattr(W, f"v4_hc_{which}_{part}")
                for which in ("attn", "ffn")
                for part in ("fn", "base", "scale")
            ),
        }
        source = RecordingCheckpointSource(database)
        load_config = descriptor.create_load_config(torch.bfloat16, database)
        mw = ModelWeights(40, "cuda:0", torch.bfloat16)
        for entry in info.layer_weights[0]:
            if entry.name in needed:
                for key, value in entry.load(source, 0, "cuda:0", load_config).items():
                    mw.set_layer_weight(0, key, value)
        self.assertEqual(set(mw.weights[0]), needed)
        self.assertTrue(all(not layer for layer in mw.weights[1:]))
        self.assertFalse(mw.global_weights, "local smoke must not load embedding/head")
        model = backend.model_cls(config, parallel, mw, None, max_generate_batch_size=1)
        views = model.shared_weight_views
        g = views.geometry
        self.assertEqual((g.num_layers, g.hidden_size, g.hc_mult), (40, 5120, 4))
        ops = get_ops(
            ep_size=1,
            num_routed_experts=g.n_routed_experts,
            num_local_experts=g.n_routed_experts,
            hidden_size=g.hidden_size,
            inter_dim=g.moe_inter_dim,
            mhc_config=g.mhc_config(),
        )
        self.assertEqual(
            (ops.moe_engine.backend, ops.moe_engine.precision), ("grouped", "w4a8")
        )
        self.assertEqual(ops.mhc_execution_stats()["backend"], "mega")
        report["strategy"] = {
            "model_path": "shared",
            "moe_backend": "grouped",
            "moe_precision": "w4a8",
            "mhc_backend": "mega",
        }
        block = model.build_weight_consumer(views, 0, ops, query_only=True)
        self.assertIs(type(block).prefill_fast_attn_pre, Block.prefill_fast_attn_pre)
        block.bind_execution_context("prefill")
        self.assertIs(block.attn_norm.weight, mw.weights[0][W.v4_attn_norm])
        self.assertEqual(
            block.attn_hc.fn.data_ptr(), mw.weights[0][W.v4_hc_attn_fn].data_ptr()
        )
        self.assertIs(views.linear(0, W.v4_attn_wq_a_w), block.attn.wq_a)
        self.assertIs(
            views.linear(0, W.v4_attn_wq_a_w, output_dtype=torch.float32).weight,
            block.attn.wq_a.weight,
        )
        self.assertTrue(all(count == 1 for count in views.conversions.values()))
        # References are computed before the first device consumer launch.
        raw = source.raw
        residual_cpu = (
            torch.sin(torch.arange(4 * 4 * 5120).float() * 0.013)
            .reshape(4, 4, 5120)
            .to(torch.bfloat16)
        )
        branch_cpu = (
            torch.cos(torch.arange(4 * 5120).float() * 0.017)
            .reshape(4, 5120)
            .to(torch.bfloat16)
        )
        ref_hidden, ref_pre, ref_post, ref_comb = _pre_reference(
            residual_cpu,
            raw["layers.0.hc_attn_fn"],
            raw["layers.0.hc_attn_scale"],
            raw["layers.0.hc_attn_base"],
            g.mhc_config(),
        )
        ref_norm = _norm_reference(ref_hidden, raw["layers.0.attn_norm.weight"], g.norm_eps)
        q_weight = _dense_reference(raw, "layers.0.attn.wq_a")
        ref_q = (ref_norm.float() @ q_weight.float().T).to(torch.bfloat16)
        ref_q = _norm_reference(ref_q, raw["layers.0.attn.q_norm.weight"], g.norm_eps)
        transition_inputs = {
            "x": branch_cpu,
            "residual": residual_cpu,
            "shifted_prev_mix": ref_pre.unsqueeze(-1),
            "post_mix": ref_post,
            "comb_res_mix": ref_comb,
            "fn": raw["layers.0.hc_ffn_fn"],
            "mix_scales": raw["layers.0.hc_ffn_scale"].flatten(),
            "mix_bases": raw["layers.0.hc_ffn_base"].flatten(),
            "rmsnorm_weight": raw["layers.0.ffn_norm.weight"],
        }
        ref_transition = independent_transition_reference(
            torch,
            transition_inputs,
            dict(
                hc_norm_eps=g.norm_eps,
                hc_pre_eps=g.hc_eps,
                hc_post_scale=2.0,
                sinkhorn_eps=g.hc_eps,
                num_sinkhorn_iters=g.hc_sinkhorn_iters,
                rmsnorm_eps=g.norm_eps,
                rmsnorm_scale=1.0,
            ),
        )
        shared_w13 = torch.cat(
            [
                _dense_reference(raw, f"layers.0.ffn.shared_experts.{sub}")
                for sub in ("w1", "w3")
            ]
        )
        shared_w2 = _dense_reference(raw, "layers.0.ffn.shared_experts.w2")
        ref_gate_up = (ref_transition["y_bf16"].float() @ shared_w13.float().T).to(
            torch.bfloat16
        )
        gate, up = ref_gate_up.chunk(2, -1)
        if g.swiglu_limit > 0:
            gate = gate.clamp(max=g.swiglu_limit)
            up = up.clamp(min=-g.swiglu_limit, max=g.swiglu_limit)
        ref_shared = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        ref_shared = (ref_shared.float() @ shared_w2.float().T).to(torch.bfloat16)
        torch.testing.assert_close(block.attn.wq_a.weight.cpu(), q_weight, rtol=0, atol=0)
        torch.testing.assert_close(
            block.ffn.shared_expert.w13.weight.cpu(), shared_w13, rtol=0, atol=0
        )
        torch.testing.assert_close(
            block.ffn.shared_expert.w2.weight.cpu(), shared_w2, rtol=0, atol=0
        )
        residual = residual_cpu.cuda()
        branch = branch_cpu.cuda()
        original = residual.clone()
        returned_residual, normed, post, comb = block.prefill_fast_attn_pre(residual)
        self.assertIs(returned_residual, residual)
        query = block.attn.query_latent(normed)
        fused = block._try_mega_mhc(branch, residual, post, comb)
        self.assertIsNotNone(fused, ops.mhc_execution_stats())
        consumed = []
        hook = block.ffn.shared_expert.register_forward_pre_hook(
            lambda module, args: consumed.append(args[0])
        )
        try:
            shared = block.ffn.shared_expert(fused[1])
        finally:
            hook.remove()
        self.assertEqual(len(consumed), 1)
        self.assertIs(
            consumed[0], fused[1], "normalized buffer copied or normalized again"
        )
        self.assertIs(block.ffn_hc.state.post_mix, fused[2])
        self.assertNotEqual(fused[0].data_ptr(), residual.data_ptr())
        self.assertNotEqual(fused[1].data_ptr(), normed.data_ptr())
        torch.testing.assert_close(residual, original, rtol=0, atol=0)
        actual = {
            "attn_pre": (block.attn_hc.pre_mix_out, ref_pre),
            "attn_post": (post, ref_post),
            "attn_comb": (comb, ref_comb),
            "attn_norm": (normed, ref_norm),
            "query_latent": (query, ref_q),
            "shared_expert": (shared, ref_shared),
        }
        for name, value in zip(
            OUTPUTS,
            (
                fused[0],
                block.ffn_hc.pre_mix_out.unsqueeze(-1),
                fused[2],
                fused[3],
                fused[1],
            ),
        ):
            actual[name] = (value, ref_transition[name])
        torch.cuda.synchronize()
        report["metrics"] = {
            name: _qualification_metric(torch, value, expected)
            for name, (value, expected) in actual.items()
        }
        report["loaded_keys"] = source.calls
        self.assertEqual(len(source.calls), 17)
        report["raw_tensors"] = {
            key: _tensor_record(value) for key, value in source.raw.items()
        }
        report["tensors"] = {
            key: _tensor_record(value) for key, value in mw.weights[0].items()
        }
        report["resident_tensors"] = {
            "wq_a": _tensor_record(block.attn.wq_a.weight),
            "shared_w13": _tensor_record(block.ffn.shared_expert.w13.weight),
            "shared_w2": _tensor_record(block.ffn.shared_expert.w2.weight),
        }
        report["aliases"] = {
            "attn_norm_is_ModelWeights_tensor": True,
            "hc_fn_is_ModelWeights_storage": True,
            "resident_rebind_reuses_storage_across_output_dtypes": True,
            "normalized_is_shared_expert_input": True,
            "residual_input_unchanged": True,
        }
        report["outputs"] = {key: _tensor_record(value[0]) for key, value in actual.items()}
        report["conversions"] = dict(views.conversions)
        report["mhc"] = ops.mhc_execution_stats()
        report["normalized_consumptions"] = len(consumed)
        report["norm_weight_offset_transforms"] = 0
        report["routed_moe_executed"] = False
        report["full_model_materialized"] = model.v4 is not None
        report["caller"] = (
            "SharedPPUModel.build_weight_consumer -> inherited "
            "Block.prefill_fast_attn_pre -> PPU HC/norm -> query_latent; "
            "Block mHC seam -> shared FFN"
        )
        self.assertEqual(report["mhc"]["calls"], {"prefill": 1})
        for name, metric in report["metrics"].items():
            self.assertEqual(metric["qualified_status"], "PASS", (name, metric))
        report["SMOKE"] = "PASS_LOCAL_REAL_WEIGHT_COMPONENTS"
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    unittest.main()
