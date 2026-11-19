import torch
import triton
import aiter
from atom.plugin.rtpllm.utils.topk512 import run
torch.manual_seed(123)
for m in [1, 2, 3, 4, 8, 64, 1024]:
    for shared in [0, 1]:
        g = torch.randn(m, 520, device='cuda', dtype=torch.bfloat16)[:, :512 + shared]
        w = torch.zeros(m, 12, device='cuda')
        ids = torch.full((m, 12), 777, device='cuda', dtype=torch.int32)
        tei = torch.empty(m, 10, device='cuda', dtype=torch.int32)
        valid = torch.tensor([m], device='cuda', dtype=torch.int32)
        w2 = w.clone()
        i2 = ids.clone()

        def base():
            aiter.topk_softmax(w, ids[:, :10], tei, g, True, shared, 'sigmoid' if shared else '', valid)

        def cand():
            run(g, w2, i2, valid)
        for seed in range(20):
            g.normal_()
            if seed == 0:
                g.zero_()
            valid.fill_(seed % (m + 1))
            base()
            cand()
            assert torch.equal(ids, i2), (m, shared, seed, ids, i2)
            torch.testing.assert_close(w, w2, rtol=1e-05, atol=1e-07)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            cand()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            cand()
        for live in (range(m + 1) if m <= 8 else [0, 1, m // 2, m - 1, m]):
            valid.fill_(live)
            g.normal_()
            graph.replay()
            base()
            assert torch.equal(ids, i2)
            torch.testing.assert_close(w, w2, rtol=1e-05, atol=1e-07)
        print('PASS eager and graph replay dynamic padding', m, shared, flush=True)
