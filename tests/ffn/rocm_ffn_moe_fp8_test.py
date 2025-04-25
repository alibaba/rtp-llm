import aiter
from aiter.ops.moe_op import fmoe_fp8_blockscale_g1u1
from aiter.ops.moe_sorting import moe_sorting_fwd
from aiter.ops.quant import dynamic_per_token_scaled_fp8_quant, pertoken_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.topk import biased_grouped_topk, biased_grouped_topk_torch
from aiter.test_common import checkAllclose
from einops import rearrange
import multiprocessing as mp
import os
import torch
import tempfile
import unittest


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/librocm_test_ops.so")

block_scale_n = 128
block_scale_k = 128
unit_size = 32


def aiter_moe_fp8(input, w1_q, w2_q, w1_scale, w2_scale, gating_weight, correction_bias, topk, num_expert_group, topk_group):
    dtype = torch.bfloat16
    quant_dtype = torch.float8_e4m3fnuz

    token, model_dim = input.shape
    num_expert, _, _ = w2_q.shape

    # calculate gating score and get topk weights, topk ids
    score = torch.nn.functional.linear(input.type(torch.float32), gating_weight.type(torch.float32), None)

    topk_weights = torch.empty((token, topk), dtype=torch.float32, device='cuda')
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device='cuda')

    biased_grouped_topk(
        score,
        correction_bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        True,
        1.0
    )

    a_q = torch.empty((token, model_dim), dtype=quant_dtype, device='cuda')
    a_scale = torch.empty((token, model_dim//block_scale_k), dtype=torch.float32, device='cuda')

    dynamic_per_token_scaled_fp8_quant(
        a_q.view(token, model_dim//block_scale_k, block_scale_k),
        input.view(token, model_dim//block_scale_k, block_scale_k),
        a_scale
    )
    a_q = a_q.view(-1, model_dim)
    a_scale = a_scale.t().contiguous()

    w1_scale = w1_scale.view(num_expert, -1)
    w2_scale = w2_scale.view(num_expert, -1)

    max_num_tokens_padded = topk_ids.numel() + num_expert * unit_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + unit_size - 1) // unit_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device='cuda')
    sorted_weights = torch.empty((max_num_tokens_padded,), dtype=torch.float32, device='cuda')
    sorted_expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device='cuda')
    num_valid_ids = torch.empty((1), dtype=torch.int32, device='cuda')
    aiter_ref_output = torch.empty((token, model_dim), dtype=dtype, device='cuda')

    moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        aiter_ref_output,
        num_expert,
        unit_size,
        None
    )

    fmoe_fp8_blockscale_g1u1(
        aiter_ref_output,
        a_q,
        w1_q,
        w2_q,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        w1_scale,
        w2_scale,
        a_scale,
        block_scale_n,
        block_scale_k,
        None
    )

    return aiter_ref_output


def torch_moe_fp8(input, w1_q, w2_q, w1_scale, w2_scale, gating_weight, correction_bias, topk, num_expert_group, topk_group):
    dtype = torch.bfloat16
    quant_dtype = torch.float8_e4m3fnuz
    compute_dtype = torch.float32

    token, model_dim = input.shape
    num_expert, _, inter_dim = w2_q.shape

    # calculate gating score and get topk weights, topk ids
    score = torch.nn.functional.linear(input.type(torch.float32), gating_weight.type(torch.float32), None)

    topk_weights, topk_ids = biased_grouped_topk_torch(
        score,
        correction_bias,
        topk,
        True,
        num_expert_group,
        topk_group
    )

    # blockscale quant a
    a_q, a_scale = pertoken_quant(
        input.view(-1, model_dim//block_scale_k, block_scale_k), quant_dtype=quant_dtype)

    a_q = a_q.to(compute_dtype)
    w1_q = w1_q.to(compute_dtype)
    w2_q = w2_q.to(compute_dtype)

    # de-quant a, w1 and w2
    a = a_q * a_scale
    a = a.view(token, 1, model_dim).repeat(1, topk, 1)

    w1_scale = rearrange(w1_scale.view(-1, 1).repeat(1, block_scale_n*block_scale_k).view(-1, w1_q.shape[1]//block_scale_n, w1_q.shape[2]//block_scale_k, block_scale_n, block_scale_k),
                            'e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)')
    w2_scale = rearrange(w2_scale.view(-1, 1).repeat(1, block_scale_n*block_scale_k).view(-1, w2_q.shape[1]//block_scale_n, w2_q.shape[2]//block_scale_k, block_scale_n, block_scale_k),
                            'e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)')
    w1 = w1_q * w1_scale
    w2 = w2_q * w2_scale

    torch_ref_output = torch.zeros((token, topk, model_dim), dtype=compute_dtype, device='cuda')

    # calculate ffn
    for e_id in range(num_expert):
        mask = topk_ids == e_id
        if mask.sum():
            sub_a = a[mask]
            act_a = sub_a @ (w1[e_id].transpose(0, 1))
            gate, up = act_a.split([inter_dim, inter_dim], dim=-1)
            act_o = torch.nn.functional.silu(gate) * up
            torch_ref_output[mask] = act_o @ (w2[e_id].transpose(0, 1))

    return (torch_ref_output * topk_weights.view(token, -1, 1)).sum(dim=1).to(dtype)


def subprocess_moe_fp8(input_path, w1_q_path, w2_q_path, w1_scale_path, w2_scale_path, gating_weight_path, correction_bias_path, topk, num_expert_group, topk_group, ep_rank, ep_size):
    dtype = torch.bfloat16

    # load inputs/weights
    input = torch.load(input_path, weights_only=True).cuda()
    w1_q = torch.load(w1_q_path, weights_only=True).cuda()
    w2_q = torch.load(w2_q_path, weights_only=True).cuda()
    w1_scale = torch.load(w1_scale_path, weights_only=True).cuda()
    w2_scale = torch.load(w2_scale_path, weights_only=True).cuda()
    gating_weight = torch.load(gating_weight_path, weights_only=True).cuda()
    correction_bias = torch.load(correction_bias_path, weights_only=True).cuda()

    num_expert_per_rank = gating_weight.shape[0] // ep_size
    token_per_rank = input.shape[0] // ep_size

    input = input[ep_rank * token_per_rank : (ep_rank + 1) * token_per_rank]

    # before we split weights to each rank, calculate reference first

    # invoke torch ref op
    torch_ref_output = torch_moe_fp8(
        input.clone(), # input should be split first, weights not
        w1_q.clone(),
        w2_q.clone(),
        w1_scale.clone(),
        w2_scale.clone(),
        gating_weight.clone(),
        correction_bias.clone(),
        topk,
        num_expert_group,
        topk_group
    )

    # for aiter op, do shuffle, it's ok shuffle first then split
    w1_q = shuffle_weight(w1_q)
    w2_q = shuffle_weight(w2_q)

    # invoke (python) aiter ref op
    aiter_ref_output = aiter_moe_fp8(
        input.clone(), # input should be split first, weights not
        w1_q.clone(),
        w2_q.clone(),
        w1_scale.clone(),
        w2_scale.clone(),
        gating_weight.clone(),
        correction_bias.clone(),
        topk,
        num_expert_group,
        topk_group
    )

    # split weights to each rank
    w1_q = w1_q[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]
    w2_q = w2_q[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]
    w1_scale = w1_scale[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]
    w2_scale = w2_scale[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]

    # init device
    rocm_ffn_moe_fp8_op = torch.classes.unittest.ROCmFfnMoeFp8Op(ep_rank, ep_size)

    # invoke rtp (c++ aiter) op
    output = torch.empty_like(input, dtype=dtype, device='cuda')
    rocm_ffn_moe_fp8_op.forward(
        input,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        gating_weight,
        correction_bias,
        topk,
        num_expert_group,
        topk_group,
        output
    )

    checkAllclose(torch_ref_output, output, rtol=0.05, atol=0.05, msg=f'[ep_size={ep_size}, ep_rank={ep_rank}]: python torch vs rtp')
    checkAllclose(aiter_ref_output, output, rtol=0.05, atol=0.05, msg=f'[ep_size={ep_size}, ep_rank={ep_rank}]: python aiter vs rtp')


class TestROCmFfnMoeFp8(unittest.TestCase):

    def _test_moe_fp8(self, token, model_dim, inter_dim, num_expert, num_shared_expert, topk, num_expert_group, topk_group, ep_size, dtype, quant_dtype):
        assert dtype == torch.bfloat16
        assert quant_dtype == torch.float8_e4m3fnuz

        print(f'token={token}, model_dim={model_dim}, inter_dim={inter_dim}, num_expert={num_expert}, num_shared_expert={num_shared_expert}, topk={topk}, ep_size={ep_size}')

        # Note: num_expert is global, token is per rank

        # input
        input = torch.randn((token * ep_size, model_dim), dtype=dtype)

        # w1 gate + up -> w1
        w1_gate = torch.randn((num_expert, inter_dim, model_dim), dtype=dtype) / 10
        w1_up = torch.randn((num_expert, inter_dim, model_dim), dtype=dtype) / 10
        w1 = torch.cat((w1_gate, w1_up), dim=1)

        # w2
        w2 = torch.randn((num_expert, model_dim, inter_dim), dtype=dtype) / 10

        # blockscale quant w1
        tmp = rearrange(w1.view(-1,
                                w1.shape[1]//block_scale_n, block_scale_n,
                                w1.shape[2]//block_scale_k, block_scale_k),
                        'e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)').contiguous()
        w1_q, w1_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
        w1_q = rearrange(w1_q.view(-1,
                                   w1.shape[1]//block_scale_n, w1.shape[2]//block_scale_k,
                                   block_scale_n, block_scale_k),
                        'e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)').contiguous()
        w1_scale = w1_scale.squeeze(-1)

        # blockscale quant w2
        tmp = rearrange(w2.view(-1,
                                w2.shape[1]//block_scale_n, block_scale_n,
                                w2.shape[2]//block_scale_k, block_scale_k),
                        'e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)').contiguous()
        w2_q, w2_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
        w2_q = rearrange(w2_q.view(-1,
                                   w2.shape[1]//block_scale_n, w2.shape[2]//block_scale_k,
                                   block_scale_n, block_scale_k),
                        'e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)').contiguous()
        w2_scale = w2_scale.squeeze(-1)

        # gating weight, correction bias, num expert group, topk group
        gating_weight = torch.randn((num_expert, model_dim), dtype=dtype)
        correction_bias = torch.randn((num_expert,), dtype=torch.float32)

        # save all inputs/weights to disk file and load in new process
        input_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_test_input_', suffix='.pt')
        w1_q_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_test_w1_q_', suffix='.pt')
        w2_q_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_test_w2_q_', suffix='.pt')
        w1_scale_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_test_w1_scale_', suffix='.pt')
        w2_scale_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_test_w2_scale_', suffix='.pt')
        gating_weight_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_test_gating_weight_', suffix='.pt')
        correction_bias_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_test_correction_bias_', suffix='.pt')

        torch.save(input, input_file)
        torch.save(w1_q, w1_q_file)
        torch.save(w2_q, w2_q_file)
        torch.save(w1_scale, w1_scale_file)
        torch.save(w2_scale, w2_scale_file)
        torch.save(gating_weight, gating_weight_file)
        torch.save(correction_bias, correction_bias_file)

        # start a new process to invoke rtp ffn layer
        procs = list()
        for ep_rank in range(ep_size):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(ep_rank)
            proc = mp.Process(target=subprocess_moe_fp8, args=(
                input_file.name,
                w1_q_file.name,
                w2_q_file.name,
                w1_scale_file.name,
                w2_scale_file.name,
                gating_weight_file.name,
                correction_bias_file.name,
                topk,
                num_expert_group,
                topk_group,
                ep_rank,
                ep_size
            ))
            proc.start()
            procs.append(proc)
        try:
            [p.join() for p in procs]
        except Exception:
            [p.terminate() for p in procs]
            [p.join() for p in procs]


    # blockscale quant
    def test_moe_fp8(self):
        # for ep_size in [1, 2]:
        for ep_size in [2]:
            for dtype in [torch.bfloat16]:
                # for token in [1, 2, 5, 16, 32]:
                for token in [2]:
                    for model_dim in [7168]:
                        for inter_dim in [256]:
                            self._test_moe_fp8(token, model_dim, inter_dim, 256, 0, 8, 8, 4, ep_size, dtype, torch.float8_e4m3fnuz)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    unittest.main()