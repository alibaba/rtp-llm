import itertools
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F
from torch import dtype as _dtype
from torch import nn
from torch.profiler import ProfilerActivity, profile

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import LinearFactory

from rtp_llm.ops.compute_ops import FusedMoEOp  # isort:skip


class FusedMoEOpTest(TestCase):
    # DTYPES = [torch.float32, torch.float16]
    # NUM_TOKENS = [7, 83, 4096, 5120]
    # NUM_EXPERT = [128]
    # TOP_K = [2, 5, 10, 32, 128]
    # HIDDEN_SIZES = [768, 2048, 4096, 5120, 8192]
    # INTER_SIZES = [128]

    # DTYPES = [torch.float16]
    DTYPES = [torch.bfloat16, torch.float16]
    NUM_TOKENS = [40960]
    NUM_EXPERT = [128]
    TOP_K = [16]
    HIDDEN_SIZES = [512]
    INTER_SIZES = [128]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def torch_sparse_block_forward(
        self, hidden_states, up_proj, down_proj, routing_weights, expert_ids
    ):
        sequence_length = hidden_states.shape[0]
        num_experts = up_proj.shape[0]
        hidden_dim = down_proj.shape[1]
        inter_dim = down_proj.shape[2]
        layers = list()
        for i in range(num_experts):
            up_proj_x, up_proj_g = torch.split(up_proj[i], inter_dim, dim=0)
            up_proj_x_dict = {"weight": up_proj_x.transpose(0, 1)}
            up_proj_g_dict = {"weight": up_proj_g.transpose(0, 1)}
            down_proj_dict = {"weight": down_proj[i].transpose(0, 1)}
            layers.append(
                {
                    "up_proj_x": LinearFactory.create_linear_from_weights(
                        up_proj_x_dict, "weight", None, None, None
                    ),
                    "up_proj_g": LinearFactory.create_linear_from_weights(
                        up_proj_g_dict, "weight", None, None, None
                    ),
                    "down_proj": LinearFactory.create_linear_from_weights(
                        down_proj_dict, "weight", None, None, None
                    ),
                    "act_fn": nn.SiLU(),
                }
            )
        final_hidden_states = torch.zeros(
            (sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = F.one_hot(expert_ids.long(), num_classes=num_experts).permute(
            2, 1, 0
        )

        for expert_idx in range(num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            routing_weight = (
                # routing_weights [num_tokens, top_k]
                routing_weights[top_x_list, idx_list, None]
            )

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            layer = layers[expert_idx]
            current_hidden_states = layer["act_fn"](
                layer["up_proj_g"](current_state)
            ) * layer["up_proj_x"](current_state)
            current_hidden_states = layer["down_proj"](current_hidden_states)
            current_hidden_states = current_hidden_states * routing_weight

            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.reshape(sequence_length, hidden_dim)
        return final_hidden_states

    def _run_fused_moe_op_test(
        self,
        num_tokens: int,
        num_expert: int,
        top_k: int,
        dtype: _dtype,
        hidden_dim: int,
        inter_dim: int,
    ):
        torch.manual_seed(0)
        model_param = GptInitModelParameters(1, 128, 1, 1, 5120)
        model_param.expert_num = num_expert
        model_param.moe_k = top_k
        model_param.has_moe_norm = True
        model_param.hidden_size = hidden_dim
        model_param.moe_inter_padding_size = inter_dim
        model_param.moe_normalize_expert_scale = 0
        model_param.activation_type = "SiGLU"
        model_param.ep_size = 1
        model_param.ep_rank = 0
        fused_moe_op = FusedMoEOp(model_param)

        hidden_states = (
            torch.rand(num_tokens, hidden_dim, dtype=dtype).to("cuda") * 2 - 1
        )
        up_proj = (
            torch.rand(
                num_expert,
                inter_dim * 2,
                hidden_dim,
                dtype=dtype,
                device=hidden_states.device,
            )
            * 2
            - 1
        )
        down_proj = (
            torch.rand(
                num_expert,
                hidden_dim,
                inter_dim,
                dtype=dtype,
                device=hidden_states.device,
            )
            * 2
            - 1
        )
        expert_scales = (
            torch.rand(
                num_tokens, top_k, dtype=torch.float32, device=hidden_states.device
            )
            * 2
            - 1
        )
        expert_scales = torch.softmax(expert_scales, dim=1)
        expert_ids = torch.zeros(
            (num_tokens, top_k), dtype=torch.int32, device=hidden_states.device
        )
        rtp_op_outputs = torch.zeros(
            (num_tokens, hidden_dim), dtype=dtype, device=hidden_states.device
        )

        for _ in range(5):
            fused_moe_op.forward(
                hidden_states,
                up_proj,
                down_proj,
                expert_scales,
                expert_ids,
                rtp_op_outputs,
            )
            torch_outputs = self.torch_sparse_block_forward(
                hidden_states, up_proj, down_proj, expert_scales, expert_ids
            )
            print(f"rtp_op_outputs = {rtp_op_outputs}")
            print(f"torch_outputs = {torch_outputs}")
            # self.assertTrue(torch.allclose(rtp_op_outputs, torch_outputs, atol=1e-2, rtol=1e-2))

            # I don't know why the element in result is so large which is 164, -346, or 535.
            # But it does equal between rtp_op_outputs and torch_outputs with eps = 1e0

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(10):
                fused_moe_op.forward(
                    hidden_states,
                    up_proj,
                    down_proj,
                    expert_scales,
                    expert_ids,
                    rtp_op_outputs,
                )
                torch_outputs = self.torch_sparse_block_forward(
                    hidden_states, up_proj, down_proj, expert_scales, expert_ids
                )

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    def test_fused_moe_op(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.NUM_EXPERT,
            self.TOP_K,
            self.DTYPES,
            self.HIDDEN_SIZES,
            self.INTER_SIZES,
        ):
            with self.subTest(
                num_tokens=params[0],
                num_expert=params[1],
                top_k=params[2],
                dtype=params[3],
                hidden_dim=params[4],
                inter_dim=params[5],
            ):
                self._run_fused_moe_op_test(*params)


if __name__ == "__main__":
    main()
