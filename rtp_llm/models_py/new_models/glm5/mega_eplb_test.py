"""Packed GLM5 checkpoint -> staged loader -> kernel launch / CMP transaction.

The CPU reference replaces the Blackwell-only DeepGEMM calls. The loader, FP4
owner, executor, dispatch buffers, graph clone and install/rollback are real.
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.mega_moe import MegaMoeExecutionPolicy
from rtp_llm.models_py.expert_placement import ExpertPlacement
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import mega_moe
from rtp_llm.models_py.new_models.glm5.language import _PackedGlm5Experts, _PackedGlm5MoE
from rtp_llm.utils.model_weight import W


class _Model(RtpModule):
    def resolve_checkpoint_load_plan(self, name, shape):
        return None, None


def _reference_kernel(y, l1, l2, buf, **kwargs):
    # Every FP4 byte in this checkpoint has identical low and high nibbles.
    # Decode positive E2M1 weights and group-32 scales into dense matrices.
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])

    def dequant(pair):
        w, s = pair
        return lut[w.long() & 7].repeat_interleave(2, dim=-1) * s

    gate, up = dequant(l1).chunk(2, dim=1)
    down = dequant(l2)
    for row, (x, slot) in enumerate(zip(buf.x, buf.topk_idx.flatten())):
        h = torch.nn.functional.silu(gate[slot] @ x) * (up[slot] @ x)
        y[row].copy_((down[slot] @ h) * buf.topk_weights[row])


class PackedMegaEplbTest(unittest.TestCase):
    @torch.inference_mode()
    def test_physical_weights_cmp_dispatch_and_rollback(self):
        config = SimpleNamespace(
            num_layers=1, model_type="glm_5", activation_type="SiGLU",
            eplb_config=SimpleNamespace(enable_eplb=True, redundant_expert=0),
        )
        parallel = SimpleNamespace(
            world_size=1, local_world_size=1, tp_size=1, tp_rank=0, ep_size=1, ep_rank=0
        )
        model = _Model()
        leaf = model.experts = _PackedGlm5Experts(
            num_experts=2, top_k=1, hidden_size=32, moe_intermediate_size=32,
            tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, params_dtype=torch.bfloat16,
            model_config=config, parallelism_config=parallel, quant_config=None,
            moe_config=SimpleNamespace(moe_strategy="mega_moe"), layer_idx=0,
        )
        ckpt = {}
        for expert in range(2):
            for j, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
                key = f"{expert}.{proj}"
                ckpt[key + ".weight"] = torch.full(
                    (32, 16), 0x11 * (expert + j + 1), dtype=torch.int8
                )
                ckpt[key + ".scale"] = torch.full((32, 1), 2.0 ** (expert - j - 3))
        leaf.load_weights(ckpt)
        raw = leaf.prepare_eplb_reload_weights()
        kernel = mega_moe.MegaMoeExecutor.__new__(mega_moe.MegaMoeExecutor)
        torch.nn.Module.__init__(kernel)
        kernel.cfg = SimpleNamespace(
            mega_moe_policy=MegaMoeExecutionPolicy(), moe_strategy="mega_moe",
            moe_w1_layout="up_gate", moe_inter_dim=32, dim=32, n_local_experts=2,
            layer_id=0,
        )
        x = torch.ones(2, 32)
        ids = torch.tensor([[0], [1]])
        buf = kernel._mega_buf = SimpleNamespace(
            x=x.clone(), x_sf=torch.ones_like(x), topk_idx=ids.clone(),
            topk_weights=torch.ones(2, 1), num_max_tokens_per_rank=2,
        )
        kernel._mega_y = torch.empty_like(x)
        kernel._mega_group = object()
        backend = SimpleNamespace(
            transform_sf_into_required_layout=lambda s, *args: s,
            # The down weight is an exact alias; copying it twice is invalid.
            transform_weights_for_mega_moe=lambda l1, l2: (
                (l1[0].clone(), l1[1].clone()), (l2[0], l2[1].clone())
            ),
            fp8_fp4_mega_moe=_reference_kernel,
        )
        block = _PackedGlm5MoE.__new__(_PackedGlm5MoE)
        torch.nn.Module.__init__(block)
        block.experts = leaf
        block.fake_balance_expert = block.shared_experts = None
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(sys.modules, {"deep_gemm": backend}),
            patch.object(kernel, "_setup_buffer_and_warmup") as warmup,
            patch.object(kernel, "_new_input_packer", return_value=None),
        ):
            save_file({leaf.prefix + "." + k: v for k, v in ckpt.items()},
                      os.path.join(folder, "model.safetensors"))
            kernel.setup_weights_from_fp4(
                raw[W.moe_w1], raw[W.moe_s1], raw[W.moe_w2], raw[W.moe_s2],
                w1_layout="up_gate",
            )
            leaf.fused_moe = FusedMoe(None, kernel, 2, "mega_moe")
            graph_block = block.clone_for_cuda_graph(share_mega_buf=True)
            loader = NewModelLoader(
                config, NewLoaderConfig(device="cpu", load_method="scratch",
                                        parallelism_config=parallel), model_path=folder,
            )
            staged = loader.load_moe_layer_weights(
                model, layer_id=0, physical_to_logical=[1, 0], ep_rank=0, ep_size=1,
            )
            self.assertEqual({k for k in staged if k.startswith("mega.")},
                             {"mega.l1_w", "mega.l1_sf", "mega.l2_sf"})
            phy = torch.tensor([1, 0], dtype=torch.int32)
            routing = leaf._routing_tensors(ExpertPlacement(2, 1, 0, (1, 0)))
            original = {k: v.data_ptr() for k, v in leaf.eplb_runtime_weight_view().items()}

            # Independent oracle for a constant input and homogeneous matrices.
            expected = []
            for e in range(2):
                gate, up, down = [
                    32 * (e + j + 1) * 0.5 * 2 ** (e - j - 3) for j in range(3)
                ]
                expected.append(torch.nn.functional.silu(torch.tensor(gate)) * up * down)
            oracle = torch.stack(expected)[:, None].expand_as(x)

            def check(swapped):
                # Raw physical launch must actually change as the plan changes.
                buf.topk_idx.copy_(ids)
                y = kernel.forward_prepacked(x)
                torch.testing.assert_close(y, oracle.flip(0) if swapped else oracle)
                for owner in (block, graph_block):
                    _, _, arena_ids, weights = owner.fused_moe.prepacked_input_views(2)
                    arena_ids.copy_(ids)
                    y = owner.forward_prepacked(x, arena_ids, weights)
                    torch.testing.assert_close(arena_ids, ids.flip(0) if swapped else ids)
                    torch.testing.assert_close(y, oracle)
                self.assertEqual(original, {
                    k: v.data_ptr() for k, v in leaf.eplb_runtime_weight_view().items()
                })

            check(False)
            with self.assertRaisesRegex(ValueError, "Missing EPLB runtime tensor"):
                leaf.apply_eplb_plan(phy, *routing, {k: v for k, v in staged.items() if k != "mega.l1_sf"})
            check(False)
            leaf.apply_eplb_plan(phy, *routing, staged, defer_commit=True)
            check(True)
            leaf.finalize_eplb_install(commit=False)
            check(False)
            leaf.apply_eplb_plan(phy, *routing, staged)
            check(True)
            warmup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
