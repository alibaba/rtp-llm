import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_topk, get_block_size_M, torch_moe_stage1, torch_moe_stage2
from aiter.ops.moe_sorting import moe_sorting_fwd
from aiter.ops.quant import dynamic_per_token_scaled_quant, pertoken_quant
from aiter.ops.topk import biased_grouped_topk, biased_grouped_topk_torch
from aiter.test_common import checkAllclose
import multiprocessing as mp
import os
import torch
import tempfile
import unittest

os.environ['DEVICE_RESERVE_MEMORY_BYTES'] = '128000000'

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/rtp_llm/tests/librocm_test_ops.so")

unit_size = 32

def aiter_moe_fp8_ptpc(input, w1_q, w2_q, w1_scale, w2_scale, gating_weight, correction_bias, topk, num_expert_group, topk_group):
    dtype = torch.bfloat16
    quant_dtype = torch.float8_e4m3fnuz

    token, model_dim = input.shape
    num_expert, _, inter_dim = w2_q.shape

    # calculate gating score and get topk weights, topk ids
    score = torch.nn.functional.linear(input.type(torch.float32), gating_weight.type(torch.float32), None)

    topk_weights = torch.empty((token, topk), dtype=torch.float32, device='cuda')
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device='cuda')

    if num_expert_group > 1: # for deepseek-r1
        if correction_bias is not None: # use bias
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
        else: # not implemented now
            pass
    else: # for qwen3
       topk_weights, topk_ids = fused_topk(input, score, topk, True)

    a_q = torch.empty((token, model_dim), dtype=quant_dtype, device='cuda')
    a_scale = torch.empty((token, 1), dtype=torch.float32, device='cuda')
    dynamic_per_token_scaled_quant(a_q, input, a_scale)

    unit_size = get_block_size_M(token, topk, num_expert, inter_dim)

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

    a2 = torch.empty((token, topk, inter_dim), dtype=dtype, device='cuda')

    aiter.moe_stage1_g1u1(
        a_q,
        w1_q,
        w2_q,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        a2,
        inter_dim,
        "", # empty kernelName
        unit_size,
        0, # 0 ksplit
        ActivationType.Silu,
        QuantType.per_Token,
        a_scale,
        w1_scale,
        None, # doweight_stage1 is false
    )

    a2_q = torch.empty((token, topk, inter_dim), dtype=quant_dtype, device='cuda')
    a2_scale = torch.empty((token, 1), dtype=torch.float32, device='cuda')
    dynamic_per_token_scaled_quant(a2_q, a2, a2_scale)

    aiter.ck_moe_stage2(
        a2_q,
        w1_q,
        w2_q,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        aiter_ref_output,
        topk,
        "",
        w2_scale,
        a2_scale,
        unit_size,
        sorted_weights,
        2,
    )

    return aiter_ref_output


def torch_moe_fp8(input, w1_q, w2_q, w1_scale, w2_scale, gating_weight, correction_bias, topk, num_expert_group, topk_group):
    dtype = torch.bfloat16
    quant_dtype = torch.float8_e4m3fnuz

    token, _ = input.shape

    # calculate gating score and get topk weights, topk ids
    score = torch.nn.functional.linear(input.type(torch.float32), gating_weight.type(torch.float32), None)

    if num_expert_group > 1: # for deepseek-r1
        if correction_bias is not None: # use bias
            topk_weights, topk_ids = biased_grouped_topk_torch(
                score,
                correction_bias,
                topk,
                True,
                num_expert_group,
                topk_group
            )
        else: # not implemented now
            pass
    else: # for qwen3
       topk_weights, topk_ids = fused_topk(input, score, topk, True)

    a_q, a_scale = pertoken_quant(input, quant_dtype=quant_dtype)

    a2 = torch_moe_stage1(
        a_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype,
        ActivationType.Silu,
        QuantType.per_Token,
        a_scale,
        w1_scale,
        False
    )

    a2_q, a2_scale = pertoken_quant(a2, quant_dtype=quant_dtype)

    torch_out = torch_moe_stage2(
        a2_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype,
        QuantType.per_Token,
        w2_scale,
        a2_scale,
        True
    )

    return torch_out


def subprocess_moe_fp8_ptpc(input_path, w1_q_path, w2_q_path, w1_scale_path, w2_scale_path, gating_weight_path, correction_bias_path, topk, num_expert_group, topk_group, ep_rank, ep_size):
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
    aiter_ref_output = aiter_moe_fp8_ptpc(
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
    rocm_ffn_moe_fp8_ptpc_op = torch.classes.unittest.ROCmFfnMoeFp8PTPCOp(ep_rank, ep_size)

    # invoke rtp (c++ aiter) op
    output = torch.empty_like(input, dtype=dtype, device='cuda')
    rocm_ffn_moe_fp8_ptpc_op.forward(
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
    checkAllclose(torch_ref_output, aiter_ref_output, rtol=0.05, atol=0.05, msg=f'[ep_size={ep_size}, ep_rank={ep_rank}]: python torch vs python aiter')


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

        # ptpc quant w1 and w2
        w1_q, w1_scale = pertoken_quant(w1, quant_dtype=quant_dtype)
        w2_q, w2_scale = pertoken_quant(w2, quant_dtype=quant_dtype)

        # gating weight, correction bias, num expert group, topk group
        gating_weight = torch.randn((num_expert, model_dim), dtype=dtype)
        correction_bias = torch.randn((num_expert,), dtype=torch.float32)

        # save all inputs/weights to disk file and load in new process
        input_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_ptpc_test_input_', suffix='.pt')
        w1_q_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_ptpc_test_w1_q_', suffix='.pt')
        w2_q_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_ptpc_test_w2_q_', suffix='.pt')
        w1_scale_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_ptpc_test_w1_scale_', suffix='.pt')
        w2_scale_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_ptpc_test_w2_scale_', suffix='.pt')
        gating_weight_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_ptpc_test_gating_weight_', suffix='.pt')
        correction_bias_file = tempfile.NamedTemporaryFile(prefix='rocm_ffn_moe_fp8_ptpc_test_correction_bias_', suffix='.pt')

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
            proc = mp.Process(target=subprocess_moe_fp8_ptpc, args=(
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


    # fp8 ptpc quant, for qwen3
    def test_moe_fp8_ptpc(self):
        for ep_size in [1, 2]:
            for dtype in [torch.bfloat16]:
                for token in [1, 2, 5, 16, 32]:
                    for model_dim in [4096]:
                        for inter_dim in [1536]:
                            self._test_moe_fp8(token, model_dim, inter_dim, 128, 0, 8, 1, 1, ep_size, dtype, torch.float8_e4m3fnuz)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    unittest.main()
