import multiprocessing as mp
import os
import tempfile
import unittest

import torch
from aiter.fused_moe import fused_topk
from aiter.test_common import checkAllclose

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


torch.classes.load_library(
    os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/librocm_test_ops.so"
)

block_scale_n = 128
block_scale_k = 128
unit_size = 32


def shuffle_moe_weight(x: torch.Tensor, is_gate, do_shuffle) -> torch.Tensor:
    def _padding_to_multiply_512(x_, is_gate):
        align = [0, 512, 0] if is_gate else [0, 0, 512]
        shape_tmp = list(
            x_.shape
        )  # due to gate+up, need temporarily seperate them for padding
        if is_gate:
            shape_tmp[1] = shape_tmp[1] // 2
        # align and padding to multiply of 512
        padding = [0 for i in range(len(align) * 2)]
        for i in range(len(align)):
            if (align[i] > 0) and (shape_tmp[i] % align[i] > 0):
                padding[-(i * 2 + 1)] = align[i] - (shape_tmp[i] % align[i])
        if sum(padding):
            if is_gate:
                x_ = torch.cat(
                    [
                        torch.nn.functional.pad(
                            x_[:, : x_.shape[1] // 2, :],
                            padding,
                            mode="constant",
                            value=0,
                        ),
                        torch.nn.functional.pad(
                            x_[:, x_.shape[1] // 2 :, :],
                            padding,
                            mode="constant",
                            value=0,
                        ),
                    ],
                    dim=1,
                )
            else:
                x_ = torch.nn.functional.pad(
                    x_, tuple(padding), mode="constant", value=0
                )
            # logging.info(f'Moe padding shape {[ele for ele in x.shape]} with {padding} to {[ele for ele in x_.shape]}')
        return x_

    def _shuffle_weight(x_, layout=(16, 16), use_int4=False):
        # Hardcode BLOCK_K and BLOCK_N
        IN, IK = layout
        BK = IK * 2
        K = 16 // x_.element_size() if not use_int4 else 32
        BN = IN
        assert x_.shape[-2] % BN == 0, f"{x_.shape[-2]} % {BN} == {x_.shape[-2] % BN }"
        assert x_.shape[-1] % BK == 0, f"{x_.shape[-1]} % {BK} == {x_.shape[-1] % BK }"
        x__ = x_.view(-1, x_.shape[-2] // BN, BN, x_.shape[-1] // BK, BK // K, K)
        x__ = x__.permute(0, 1, 3, 4, 2, 5)
        x__ = x__.contiguous()
        x__ = x__.view(*x_.shape)
        return x__

    # x_ = torch.cat([x[:, x.shape[1] // 2:, :], x[:, :x.shape[1]//2, :]], dim =1) if is_gate else x #swap from [up, gate] to [gate, up]
    x_ = x
    if do_shuffle:
        # for now we use ck_moe for dtype is not fp8, so we need to pad to multiply of 512
        if x_.dtype not in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
            x_ = _padding_to_multiply_512(x_, is_gate)
        x_ = _shuffle_weight(x_)
    return x_


def torch_moe(input, w1, w2, gating_weight, topk, num_expert_group, topk_group):
    dtype = torch.bfloat16
    compute_dtype = torch.bfloat16

    token, model_dim = input.shape
    num_expert, _, inter_dim = w2.shape

    # calculate gating score and get topk weights, topk ids
    score = torch.nn.functional.linear(
        input.type(compute_dtype), gating_weight.type(compute_dtype), None
    )

    topk_weights, topk_ids = fused_topk(input, score, topk, False)

    input = input.to(compute_dtype)
    input = input.view(token, 1, model_dim).repeat(1, topk, 1)
    w1 = w1.to(compute_dtype)
    w2 = w2.to(compute_dtype)

    torch_ref_output = torch.zeros(
        (token, topk, model_dim), dtype=compute_dtype, device="cuda"
    )
    torch_ref_a2 = torch.zeros(
        (token, topk, inter_dim * 2), dtype=compute_dtype, device="cuda"
    )
    torch_ref_a2_activated = torch.zeros(
        (token, topk, inter_dim), dtype=compute_dtype, device="cuda"
    )

    # calculate ffn
    for e_id in range(num_expert):
        mask = topk_ids == e_id
        if mask.sum():
            sub_a = input[mask]
            act_a = sub_a @ (w1[e_id].transpose(0, 1))
            torch_ref_a2[mask] = act_a
            gate, up = act_a.split([inter_dim, inter_dim], dim=-1)
            act_o = torch.nn.functional.silu(gate) * up
            torch_ref_a2_activated[mask] = act_o
            torch_ref_output[mask] = act_o @ (w2[e_id].transpose(0, 1))
    ref_out = (torch_ref_output * topk_weights.view(token, -1, 1)).sum(dim=1).to(dtype)
    return ref_out


def subprocess_moe(
    input_path,
    w1_q_path,
    w2_q_path,
    w1_scale_path,
    w2_scale_path,
    gating_weight_path,
    correction_bias_path,
    topk,
    num_expert_group,
    topk_group,
    ep_rank,
    ep_size,
):
    dtype = torch.bfloat16

    # load inputs/weights
    input = torch.load(input_path, weights_only=True).cuda()
    w1_q = torch.load(w1_q_path, weights_only=True).cuda()
    w2_q = torch.load(w2_q_path, weights_only=True).cuda()
    w1_scale = (
        torch.load(w1_scale_path, weights_only=True).cuda() if w1_scale_path else None
    )
    w2_scale = (
        torch.load(w2_scale_path, weights_only=True).cuda() if w2_scale_path else None
    )
    gating_weight = torch.load(gating_weight_path, weights_only=True).cuda()
    correction_bias = (
        torch.load(correction_bias_path, weights_only=True).cuda()
        if correction_bias_path
        else None
    )

    num_expert_per_rank = gating_weight.shape[0] // ep_size
    token_per_rank = input.shape[0] // ep_size

    input = input[ep_rank * token_per_rank : (ep_rank + 1) * token_per_rank]

    torch_ref_output = torch_moe(
        input.clone(),  # input should be split first, weights not
        w1_q.clone(),
        w2_q.clone(),
        gating_weight.clone(),
        topk,
        num_expert_group,
        topk_group,
    )

    # for aiter op, do shuffle, it's ok shuffle first then split
    w1_q = shuffle_moe_weight(w1_q, True, True)
    w2_q = shuffle_moe_weight(w2_q, False, True)

    # split weights to each rank
    w1_q = w1_q[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]
    w2_q = w2_q[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]
    if w1_scale:
        w1_scale = w1_scale[
            ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank
        ]
    if w2_scale:
        w2_scale = w2_scale[
            ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank
        ]

    # init device
    rocm_ffn_moe_op = torch.classes.unittest.ROCmFfnMoeBf16Op(ep_rank, ep_size)

    # invoke rtp (c++ aiter) op
    output = torch.empty_like(input, dtype=dtype, device="cuda")
    rocm_ffn_moe_op.forward(
        input,
        w1_q,
        w2_q,
        gating_weight,
        correction_bias,
        topk,
        num_expert_group,
        topk_group,
        output,
    )

    checkAllclose(
        torch_ref_output,
        output,
        rtol=0.05,
        atol=0.05,
        msg=f"[ep_size={ep_size}, ep_rank={ep_rank}]: python torch vs rtp",
    )


class TestROCmFfnMoe(unittest.TestCase):

    def _test_moe(
        self,
        token,
        model_dim,
        inter_dim,
        num_expert,
        num_shared_expert,
        topk,
        num_expert_group,
        topk_group,
        norm_topk_prob,
        ep_size,
        dtype,
    ):
        assert dtype == torch.bfloat16
        # assert quant_dtype == torch.float8_e4m3fnuz

        print(
            f"token={token}, model_dim={model_dim}, inter_dim={inter_dim}, num_expert={num_expert}, num_shared_expert={num_shared_expert}, topk={topk}, ep_size={ep_size}"
        )

        # Note: num_expert is global, token is per rank

        # input
        input = torch.randn((token * ep_size, model_dim), dtype=dtype)

        # w1 gate + up -> w1
        w1_gate = torch.randn((num_expert, inter_dim, model_dim), dtype=dtype) / 10
        w1_up = torch.randn((num_expert, inter_dim, model_dim), dtype=dtype) / 10
        w1 = torch.cat((w1_gate, w1_up), dim=1)

        # w2
        w2 = torch.randn((num_expert, model_dim, inter_dim), dtype=dtype) / 10

        # gating weight, correction bias, num expert group, topk group
        gating_weight = torch.randn((num_expert, model_dim), dtype=dtype)
        correction_bias = torch.randn((num_expert,), dtype=torch.float32)

        # save all inputs/weights to disk file and load in new process
        input_file = tempfile.NamedTemporaryFile(
            prefix="rocm_ffn_moe_test_input_", suffix=".pt"
        )
        w1_file = tempfile.NamedTemporaryFile(
            prefix="rocm_ffn_moe_test_w1_q_", suffix=".pt"
        )
        w2_file = tempfile.NamedTemporaryFile(
            prefix="rocm_ffn_moe_test_w2_q_", suffix=".pt"
        )
        gating_weight_file = tempfile.NamedTemporaryFile(
            prefix="rocm_ffn_moe_test_gating_weight_", suffix=".pt"
        )
        # correction_bias_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_test_correction_bias_', suffix='.pt')

        torch.save(input, input_file)
        torch.save(w1, w1_file)
        torch.save(w2, w2_file)
        torch.save(gating_weight, gating_weight_file)
        # torch.save(correction_bias, correction_bias_file)

        # start a new process to invoke rtp ffn layer
        procs = list()
        for ep_rank in range(ep_size):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ep_rank)
            proc = mp.Process(
                target=subprocess_moe,
                args=(
                    input_file.name,
                    w1_file.name,
                    w2_file.name,
                    None,
                    None,
                    gating_weight_file.name,
                    None,
                    topk,
                    num_expert_group,
                    topk_group,
                    ep_rank,
                    ep_size,
                ),
            )
            proc.start()
            procs.append(proc)
        try:
            [p.join() for p in procs]
        except Exception:
            [p.terminate() for p in procs]
            [p.join() for p in procs]

    # blockscale quant, for qwen3
    def test_moe(self):
        # for ep_size in [1, 2]:
        for ep_size in [1]:
            for dtype in [torch.bfloat16]:
                for token in [1, 2, 5, 16, 32]:
                    for model_dim in [2048]:
                        for inter_dim in [1408]:
                            self._test_moe(
                                token,
                                model_dim,
                                inter_dim,
                                60,
                                0,
                                4,
                                1,
                                1,
                                False,
                                ep_size,
                                dtype,
                            )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    unittest.main()
