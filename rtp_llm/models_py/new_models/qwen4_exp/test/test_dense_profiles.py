"""Real checkpoint regressions for Qwen4's dense role-specific recipe."""
import os
from pathlib import Path
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import MODEL_REGISTRY
from rtp_llm.models_py.new_models.qwen4_exp.dense_weights import Qwen4BlockProjection
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import Qwen4ProjectionWeight
from rtp_llm.models_py.new_models.qwen4_exp.target_weights import Qwen4SharedExpertWeights, Qwen4VocabWeight
from rtp_llm.models_py.new_models.qwen4_exp.gdn_weights import Qwen4GDNWeights


class _DenseOwners(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.block = Qwen4BlockProjection((256, 256))
        self.scalar = Qwen4ProjectionWeight((128, 256), fp8_storage=True)
        self.unaligned = Qwen4ProjectionWeight((1, 256), fp8_storage=True)
        self.shared = Qwen4SharedExpertWeights(256, 256, torch.bfloat16,
            tp_size=load.ffn_tp_size, tp_rank=load.ffn_tp_rank, fp8_storage=True)
        self.vocab = Qwen4VocabWeight(37, 256, embedding=True,
            tp_size=load.attn_tp_size, tp_rank=load.attn_tp_rank,
            dtype=torch.bfloat16, fp8_storage=True)
        la = SimpleNamespace(linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=8, linear_num_value_heads=16, linear_conv_kernel_dim=3)
        self.gdn = Qwen4GDNWeights(la,256,1e-6,load.attn_tp_size,load.attn_tp_rank,
            torch.bfloat16,None,"gdn",fp8_storage=True)


def _scalar_reference(value):
    scale = value.float().abs().amax().clamp_min(1e-12).reshape(1) / 448.0
    return (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


class DenseProfileTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("RUN_Q4_DIRECT_STORAGE_GPU") == "1", "manual SM120 native quantization")
    def test_sm120_uses_original_direct_packed_quantization(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(),(12,0))
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import quant_weight_ue8m0_packed
        from rtp_llm.models_py.new_models.qwen4_exp.dense_weights import bind_projection
        owner = Qwen4BlockProjection((640,2560)).cuda()
        original = torch.randn(640,2560,device="cuda",dtype=torch.bfloat16)
        owner.weight.data.copy_(original)
        reference, scale = quant_weight_ue8m0_packed(original)
        owner.process_weights_after_loading()
        torch.testing.assert_close(owner.weight.view(torch.uint8),reference.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(owner.weight_scale,scale,rtol=0,atol=0)
        before = owner.weight.data_ptr(),owner.weight_scale.data_ptr()
        bind_projection(owner)
        self.assertEqual(before,(owner.weight.data_ptr(),owner.weight_scale.data_ptr()))

    def test_real_scratch_and_fst_role_storage_after_rank_selection(self):
        self._run_profiles()

    @unittest.skipUnless(os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "1", "native FST opt-in")
    def test_real_two_rank_fst_role_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(_fst_worker,
                args=(str(Path(directory)/"gloo"),), nprocs=2, join=True, start_method="spawn")

    def _run_profiles(self, methods=("scratch",), ranks=(0, 1)):
        torch.manual_seed(1234)
        values = {name + ".weight": torch.randn(shape).bfloat16() for name, shape in (
            ("block", (256, 256)), ("scalar", (128, 256)), ("unaligned", (1, 256)),
            ("shared.gate_proj", (256, 256)), ("shared.up_proj", (256, 256)),
            ("shared.down_proj", (256, 256)), ("vocab", (37, 256)),
        )}
        # Force distinct rank scales and a merged gate/up scale domain.
        values["vocab.weight"][:, 128:] *= 5
        values["shared.up_proj.weight"] *= 7
        for name, shape in {
            "in_proj_qkv.weight":(512,256), "in_proj_z.weight":(256,256),
            "in_proj_b.weight":(16,256), "in_proj_a.weight":(16,256),
            "conv1d.weight":(512,1,3), "dt_bias":(16,), "A_log":(16,),
            "norm.weight":(16,), "out_proj.weight":(256,256),
        }.items():
            values["gdn."+name] = torch.randn(shape).bfloat16()
        for method in methods:
            for rank in ranks:
                with self.subTest(method=method, rank=rank), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LOAD_METHOD": ""}), patch.dict(MODEL_REGISTRY, {"_q4_dense_roles": _DenseOwners}):
                    save_file(values, str(Path(directory) / "model.safetensors"))
                    load = NewLoaderConfig(device="cpu", compute_dtype=torch.bfloat16,
                        load_method=method, tp_size=2, tp_rank=rank,
                        attn_tp_size=2, attn_tp_rank=rank, ffn_tp_size=2, ffn_tp_rank=rank)
                    loader = NewModelLoader({"model_type": "_q4_dense_roles", "ckpt_path": directory}, load, model_path=directory)
                    owner = loader.load()
                    self.assertEqual(loader.loaded_method.value, method)
                    from rtp_llm.models_py.quant_methods.fp8 import per_block_quant_like_legacy
                    block, scale = per_block_quant_like_legacy(values["block.weight"], 128)
                    torch.testing.assert_close(owner.block.weight.float(), block.float(), rtol=0, atol=0)
                    torch.testing.assert_close(owner.block.weight_scale, scale, rtol=0, atol=0)
                    for leaf, source in (
                        (owner.scalar, values["scalar.weight"]),
                        (owner.vocab, values["vocab.weight"][:, rank*128:(rank+1)*128]),
                        (owner.shared.down_proj, values["shared.down_proj.weight"][:, rank*128:(rank+1)*128]),
                    ):
                        expected, scale = _scalar_reference(source)
                        torch.testing.assert_close(leaf.weight.float(), expected.float(), rtol=0, atol=0)
                        torch.testing.assert_close(leaf.storage_scale, scale, rtol=0, atol=0)
                    merged = torch.cat([values["shared." + name + ".weight"][rank*128:(rank+1)*128] for name in ("gate_proj", "up_proj")])
                    expected, scale = _scalar_reference(merged)
                    actual = torch.cat((owner.shared.gate_proj.weight, owner.shared.up_proj.weight))
                    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
                    torch.testing.assert_close(owner.shared.up_storage_scale, scale, rtol=0, atol=0)
                    self.assertEqual(owner.unaligned.weight.dtype, torch.bfloat16)
                    torch.testing.assert_close(owner.unaligned.weight, values["unaligned.weight"], rtol=0, atol=0)
                    local_qkvz = torch.cat([
                        values["gdn.in_proj_qkv.weight"][offset+rank*count:offset+(rank+1)*count]
                        for offset,count in ((0,64),(128,64),(256,128))
                    ]+[values["gdn.in_proj_z.weight"][rank*128:(rank+1)*128]])
                    expected, scale = per_block_quant_like_legacy(local_qkvz,128)
                    torch.testing.assert_close(owner.gdn.in_proj_qkvz.weight.float(),expected.float(),rtol=0,atol=0)
                    torch.testing.assert_close(owner.gdn.in_proj_qkvz.weight_scale,scale,rtol=0,atol=0)
                    local_ba = torch.cat([values["gdn.in_proj_"+part+".weight"][rank*8:(rank+1)*8] for part in ("b","a")]).t()
                    expected, scale = _scalar_reference(local_ba)
                    torch.testing.assert_close(owner.gdn.in_proj_ba_w.float(),expected.float(),rtol=0,atol=0)
                    torch.testing.assert_close(owner.gdn.in_proj_ba_storage_scale,scale,rtol=0,atol=0)
                    before = {name:(id(p), p.data_ptr()) for name,p in owner.named_parameters()}
                    for child in reversed(tuple(owner.modules())):
                        hook = getattr(child, "process_weights_after_loading", None)
                        if hook is not None:
                            hook()
                    self.assertEqual(before, {name:(id(p), p.data_ptr()) for name,p in owner.named_parameters()})


def _fst_worker(rank, rendezvous):
    torch.distributed.init_process_group("gloo", init_method="file://" + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        DenseProfileTest()._run_profiles(("fastsafetensors",), (rank,))
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
