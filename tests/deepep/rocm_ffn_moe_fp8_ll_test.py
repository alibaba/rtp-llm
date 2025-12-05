import aiter
from aiter.fused_moe import fused_topk
from aiter.ops.moe_op import fmoe_fp8_blockscale_g1u1
from aiter.ops.moe_sorting import moe_sorting_fwd
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.quant import dynamic_per_token_scaled_quant, pertoken_quant
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


torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/rtp_llm/tests/librocm_test_ops.so")

block_scale_n = 128
block_scale_k = 128
unit_size = 32

OPEN_PROFILER = True

def torch_moe_fp8(input, w1_q, w2_q, w1_scale, w2_scale, gating_weight, correction_bias, topk, num_expert_group, topk_group):
    dtype = torch.bfloat16
    quant_dtype = torch.float8_e4m3fnuz
    compute_dtype = torch.float32

    token, model_dim = input.shape
    num_expert, _, inter_dim = w2_q.shape

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
    #    topk_weights, topk_ids = fused_topk(input, score, topk, True)
        score_softmax = torch.nn.functional.softmax(
            score.float(),
            dim=-1,
        )
        topk_weights, topk_ids = score_softmax.topk(
            k=topk,
            dim=-1,
            largest=True,
            sorted=True,
        )
    print(f"++++++++++ topk_ids : {topk_ids.shape}, topk_ids content:")
    for i in range(min(5, topk_ids.shape[0])):
        print(f"  Row {i}: {topk_ids[i][:10]}...")
    # per-token quant
    a_q, a_scale = pertoken_quant(input, quant_dtype=quant_dtype)
    print(f"a_q.shape: {a_q.shape}, a_scale.shape: {a_scale.shape}")
    
    a_q = a_q.to(compute_dtype)
    w1_q = w1_q.to(compute_dtype)
    w2_q = w2_q.to(compute_dtype)

    # de-quant a, w1, w2
    a = a_q * a_scale
    a = a.view(token, 1, model_dim).repeat(1, topk, 1)
    w1 = w1_q * w1_scale
    w2 = w2_q * w2_scale


    torch_ref_output = torch.zeros((token, topk, model_dim), dtype=compute_dtype, device='cuda')
    torch_ref_a2 = torch.zeros((token, topk, inter_dim*2), dtype=compute_dtype, device='cuda')
    torch_ref_a2_activated = torch.zeros((token, topk, inter_dim), dtype=compute_dtype, device='cuda')

    # calculate ffn
    for e_id in range(num_expert):
        mask = topk_ids == e_id
        if mask.sum():
            sub_a = a[mask]
            act_a = sub_a @ (w1[e_id].transpose(0, 1))
            torch_ref_a2[mask] = act_a
            gate, up = act_a.split([inter_dim, inter_dim], dim=-1)
            act_o = torch.nn.functional.silu(gate) * up
            torch_ref_a2_activated[mask] = act_o
            torch_ref_output[mask] = act_o @ (w2[e_id].transpose(0, 1))
    ref_out = (torch_ref_output * topk_weights.view(token, -1, 1)).sum(dim=1).to(dtype)
    return ref_out

def torch_moe_fp8_2(input, w1_q, w2_q, w1_scale, w2_scale, gating_weight, correction_bias, topk, num_expert_group, topk_group):
    dtype = torch.bfloat16
    quant_dtype = torch.float8_e4m3fnuz
    compute_dtype = torch.float32

    token, model_dim = input.shape
    num_expert, _, inter_dim = w2_q.shape

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

    print(f"++++++++++ topk_ids : {topk_ids.shape}, topk_ids content:")
    for i in range(min(5, topk_ids.shape[0])):
        print(f"  Row {i}: {topk_ids[i][:10]}...")
    # per-token quant
    a_q, a_scale = pertoken_quant(input, quant_dtype=quant_dtype)

    print(f"========== a_q shape: {a_q.shape}, a_scale shape: {a_scale.shape}")
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
    print(f"torch_ref_output shape : {torch_ref_output.shape}")

    # for aiter op, do shuffle, it's ok shuffle first then split
    w1_q = shuffle_weight(w1_q)
    w2_q = shuffle_weight(w2_q)


    # split weights to each rank
    w1_q = w1_q[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]
    w2_q = w2_q[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]
    w1_scale = w1_scale[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]
    w2_scale = w2_scale[ep_rank * num_expert_per_rank : (ep_rank + 1) * num_expert_per_rank]

    # init device
    rocm_ffn_moe_fp8_test = torch.classes.unittest.ROCmFfnMoeFp8Test(ep_rank, ep_size)
    output = torch.empty_like(input, dtype=dtype, device='cuda')
    if OPEN_PROFILER:
        num_runs = 20
        # 使用 PyTorch Profiler 进行性能分析
        with torch.profiler.profile(
            profile_memory=False,
            record_shapes=False,
            schedule=torch.profiler.schedule(wait=0, warmup=10, active=50),
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            for _ in range(num_runs):
                # invoke rtp (c++ aiter) op
                rocm_ffn_moe_fp8_test.forward(
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
                prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        trace_file = f"trace_rank_{ep_rank}.json"
        if os.path.exists(trace_file):
            os.remove(trace_file)
        prof.export_chrome_trace(trace_file)
    else:
        # invoke rtp (c++ aiter) op
        rocm_ffn_moe_fp8_test.forward(
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
    print("rocm_ffn_moe_fp8_ll_test output: ", output)

    checkAllclose(torch_ref_output, output, rtol=0.05, atol=0.05, msg=f'[ep_size={ep_size}, ep_rank={ep_rank}]: python torch vs rtp')


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

        w1_q, w1_scale = pertoken_quant(w1, quant_dtype=quant_dtype)
        w2_q, w2_scale = pertoken_quant(w2, quant_dtype=quant_dtype)
        print(f"w1_q.shape: {w1_q.shape}, w1_scale.shape: {w1_scale.shape}")
        print(f"w2_q.shape: {w2_q.shape}, w2_scale.shape: {w2_scale.shape}")

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


    # blockscale quant, for deepseek-r1
    # def test_moe_fp8(self):
    #     # for ep_size in [1, 2]:
    #     for ep_size in [2]:
    #         for dtype in [torch.bfloat16]:
    #             # for token in [1, 2, 5, 16, 32]:
    #             for token in [2]:
    #                 for model_dim in [7168]:
    #                     for inter_dim in [256]:
    #                         self._test_moe_fp8(token, model_dim, inter_dim, 256, 0, 8, 8, 4, ep_size, dtype, torch.float8_e4m3fnuz)


    # blockscale quant, for qwen3
    def test_moe_fp8(self):
        # for ep_size in [1, 2]:
        for ep_size in [4]:
            for dtype in [torch.bfloat16]:
                # for token in [1, 2, 5, 16, 32]:
                for token in [1]:
                    for model_dim in [4096]:
                        for inter_dim in [1536]:
                            self._test_moe_fp8(token, model_dim, inter_dim, 128, 0, 8, 1, 1, ep_size, dtype, torch.float8_e4m3fnuz)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    unittest.main()