"""cuLA adapter output, FP32 publication, prefixes and virtual request rows."""

import json

import torch

from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path


def run_case(lengths, prefixes, tables, block_size=64, heads=2):
    setup_cutlass_import_path()
    from cula.kda import chunk_kda

    from rtp_llm.models_py.modules.factory.attention.cuda_kda_impl import cula_kda_paged_prefill

    torch.manual_seed(3101)
    total = sum(lengths)
    q, k, v, g = [
        (0.1 * torch.randn((total, heads, 128), device="cuda")).bfloat16()
        for _ in range(4)
    ]
    beta = (0.1 * torch.randn((total, heads), device="cuda")).bfloat16()
    a_log = torch.zeros((heads,), device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros((heads, 128), device="cuda", dtype=torch.float32)
    count = 1 + max(block for row in tables for block in row)
    cache = 0.01 * torch.randn(
        (count, heads, 128, 128), device="cuda", dtype=torch.float32
    )
    original_cache = cache.clone()
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    expected = torch.zeros_like(v)
    expected_final = {}
    with torch.inference_mode():
        for request, (length, prefix, table) in enumerate(
            zip(lengths, prefixes, tables)
        ):
            if not any(table):
                continue
            start, end = cu[request : request + 2]
            initial = (
                original_cache[table[(prefix - 1) // block_size]]
                .unsqueeze(0)
                .contiguous()
                if prefix
                else torch.zeros((1, heads, 128, 128), device="cuda")
            )
            inputs = [
                t[start:end].unsqueeze(0).contiguous() for t in (q, k, v, g, beta)
            ]
            if inputs[-1].data_ptr() % 16:
                inputs[-1] = inputs[-1].clone()
            values, final = chunk_kda(
                *inputs,
                scale=128**-0.5,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=-5.0,
                disable_recompute=False,
                use_intracard_cp=False,
                A_log=a_log,
                dt_bias=dt_bias,
            )
            expected[start:end] = values[0]
            expected_final[table[(prefix + length - 1) // block_size]] = final[0]
        actual = cula_kda_paged_prefill(
            q,
            k,
            v,
            g,
            beta,
            a_log,
            dt_bias,
            -5.0,
            cache,
            cu,
            prefixes,
            tables,
            block_size,
        )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.0005)
    max_state_error = 0.0
    for block, state in expected_final.items():
        torch.testing.assert_close(cache[block], state, rtol=0.03, atol=0.0005)
        max_state_error = max(
            max_state_error, (cache[block] - state).abs().max().item()
        )
    used = {block for row in tables for block in row if block > 0}
    for block in set(range(count)) - used:
        torch.testing.assert_close(cache[block], original_cache[block], rtol=0, atol=0)
    for request, table in enumerate(tables):
        if not any(table):
            assert actual[cu[request] : cu[request + 1]].count_nonzero().item() == 0
    print(
        json.dumps(
            dict(
                passed=True,
                lengths=lengths,
                prefixes=prefixes,
                output_max_abs=(actual.float() - expected.float()).abs().max().item(),
                state_max_abs=max_state_error,
                state_dtype=str(cache.dtype),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    run_case([333], [0], [[1, 2, 3, 4, 5, 6]])
    run_case([137, 159, 64], [0, 37, 0], [[1, 2, 3], [4, 5, 6, 7], [0]])
    run_case([3203], [0], [list(range(1, 52))], heads=12)
