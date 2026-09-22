"""Check exact block-boundary state preservation, independently of model text."""
import json
import torch
from rtp_llm.models_py.triton_kernels.kimi_kda import chunk_kda

def test_fp32_cache_checkpoints():
    torch.manual_seed(917)
    results = []
    for boundary in (128, 4096):
        length, heads, dim = boundary + 64, 2, 128
        args = [torch.randn(1, length, heads, dim, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
        beta = torch.randn(1, length, heads, device='cuda').sigmoid()
        alog = torch.randn(heads, device='cuda') * .1
        bias = torch.randn(heads * dim, device='cuda') * .01
        initial = torch.randn(1, heads, dim, dim, device='cuda') * .1
        def run(xs, bs, state, fp32):
            return chunk_kda(*xs, bs, initial_state=state, output_final_state=True,
                             use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                             A_log=alog, dt_bias=bias, lower_bound=-5.0,
                             return_intermediate_states=True,
                             intermediate_states_in_fp32=fp32)
        legacy_o, legacy_final, legacy_h = run(args, beta, initial, False)
        fp_o, fp_final, fp_h = run(args, beta, initial, True)
        assert legacy_h.dtype == torch.bfloat16
        assert fp_h.dtype == torch.float32
        torch.testing.assert_close(fp_o, legacy_o, rtol=0, atol=0)
        torch.testing.assert_close(fp_final, legacy_final, rtol=0, atol=0)
        _, prefix_final, _ = run([x[:, :boundary].contiguous() for x in args], beta[:, :boundary].contiguous(), initial, True)
        checkpoint = fp_h[:, boundary // 64]
        # Both paths execute the identical recurrence over identical full 64-token tiles.
        torch.testing.assert_close(checkpoint, prefix_final, rtol=0, atol=0)
        loss = (legacy_h[:, boundary // 64].float() - prefix_final).abs().max().item()
        assert loss > 0, 'Fixture must detect BF16 checkpoint rounding'
        tail_o, tail_final, _ = run([x[:, boundary:].contiguous() for x in args], beta[:, boundary:].contiguous(), checkpoint.contiguous(), True)
        torch.testing.assert_close(tail_final, fp_final, rtol=0, atol=0)
        torch.testing.assert_close(tail_o, fp_o[:, boundary:], rtol=0, atol=0)
        results.append({'boundary': boundary, 'legacy_checkpoint_max_abs': loss,
                        'fp32_checkpoint_max_abs': 0, 'cached_tail_exact': True,
                        'default_output_unchanged': True})
    print(json.dumps({'passed': True, 'cases': results}, indent=2))


if __name__ == "__main__":
    test_fp32_cache_checkpoints()
