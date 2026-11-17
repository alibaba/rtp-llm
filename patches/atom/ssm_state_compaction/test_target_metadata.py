"""ROCm oracle test for fused SSM/conv metadata; run on an idle GPU."""
import torch
import triton
from atom.plugin.rtpllm.utils.forward_context import _target_verify_state_slots_graph_kernel as original
from atom.plugin.rtpllm.utils.split_ssm_storage import _split_target_state_slots_kernel as fused


def main():
    cases = 0
    for batch in (1, 4, 16, 129):
        for width in (2, 3, 4):
            cols, tile = 12, 128
            grid = (triton.cdiv(batch, tile),)
            state = torch.arange(batch * cols, device="cuda", dtype=torch.int32).reshape(batch, cols) + 1
            conv = state + 500
            lengths = torch.tensor(([0, 1, 2, 63, 64, 65, 127, 128, 129, 768] * 20)[:batch], device="cuda", dtype=torch.int32)
            prefixes = (lengths > 1).to(torch.int32)
            outputs = [torch.empty((batch, width), device="cuda", dtype=torch.int32) for _ in range(8)]
            flags = [torch.empty(grid, device="cuda", dtype=torch.int32) for _ in range(3)]
            def run():
                original[grid](state, lengths, prefixes, outputs[0], outputs[1], flags[0], batch, *state.stride(), cols, 64, width, tile)
                original[grid](conv, lengths, prefixes, outputs[2], outputs[3], flags[1], batch, *conv.stride(), cols, 64, width, tile)
                fused[grid](state, conv, lengths, prefixes, *outputs[4:], flags[2], batch, *state.stride(), *conv.stride(), cols, cols, 64, width, tile)
            def check():
                torch.cuda.synchronize()
                for index in range(4):
                    assert torch.equal(outputs[index], outputs[index + 4]), (batch, width, index)
                assert torch.equal(flags[2] > 0, (flags[0] > 0) | (flags[1] > 0))
            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for bad in (False, True):
                if bad:
                    state[:, 2] = -1
                    conv[:, 3] = -1
                graph.replay()
                check()
                cases += 1
    print(f"PASS {cases} graph replays: exact indices, padding, boundaries, invalid-table checks")


if __name__ == "__main__":
    main()
