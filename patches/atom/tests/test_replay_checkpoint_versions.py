"""GPU regression checks for the installed ReplaySSM dependency patch."""
from types import SimpleNamespace
import torch
from atom.plugin.rtpllm.utils import replayssm_rtp_fused_copy as replay


def main():
    names = ('v_cache', 'replay_buf_k', 'replay_buf_u', 'replay_buf_g')
    caches = [SimpleNamespace(
        v_cache=torch.randn(8, 2, 64, 64, device='cuda'),
        replay_buf_k=torch.randn(8, 2, 16, 64, device='cuda', dtype=torch.bfloat16),
        replay_buf_u=torch.randn(8, 2, 16, 64, device='cuda', dtype=torch.bfloat16),
        replay_buf_g=torch.randn(8, 2, 16, device='cuda'),
    ) for _ in range(3)]
    tables, sizes = replay.make_layer_pointer_tables(caches, torch.device('cuda'))
    src = torch.tensor([0, 1], device='cuda')
    candidates = torch.tensor([[2, 3, 4], [5, 6, 7]], device='cuda')
    cursor = torch.full((8,), 12, device='cuda', dtype=torch.int32)
    version = torch.zeros(8, device='cuda', dtype=torch.int64)
    buffers = [torch.empty(2, device='cuda', dtype=dtype) for dtype in
               (torch.int64, torch.int32, torch.int64, torch.int32)]
    for accepted in (1, 2, 3):
        cursor.fill_(12)
        version.zero_()
        accepts = torch.full((2,), accepted, device='cuda', dtype=torch.int32)
        replay.commit_copy_versioned(tables, sizes, cursor, version, src, candidates,
                                     accepts, *buffers, 16, epoch_base=1000 + accepted * 8)
        dst = candidates[:, accepted - 1]
        assert version[dst[0]].item() != version[dst[1]].item()
        for cache in caches:
            torch.testing.assert_close(cache.v_cache[dst], cache.v_cache[src], rtol=0, atol=0)
        replay.copy_replay_slot_with_version(tables, sizes, cursor, version, dst[:1], dst[1:])
        for cache in caches:
            for name in names:
                tensor = getattr(cache, name)
                torch.testing.assert_close(tensor[dst[0]], tensor[dst[1]], rtol=0, atol=0)
    # Captured copies must use current metadata on each replay, not capture-time decisions.
    reads = torch.tensor([0, -1], device='cuda')
    writes = torch.tensor([2, -1], device='cuda')
    flags = torch.empty(2, device='cuda', dtype=torch.int32)
    def copy():
        replay.copy_replay_slot_with_version(tables, sizes, cursor, version, reads, writes, flags)
    for _ in range(3):
        copy()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy()
    for equal in (False, True, False):
        version[0], version[2] = 100, 100 if equal else 200
        for cache in caches:
            cache.v_cache[0].normal_()
            if equal:
                cache.v_cache[2].copy_(cache.v_cache[0])
            else:
                cache.v_cache[2].zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert flags[0].item() == (0 if equal else 1)
        assert flags[1].item() == 0
        for cache in caches:
            for name in names:
                tensor = getattr(cache, name)
                torch.testing.assert_close(tensor[0], tensor[2], rtol=0, atol=0)
    # Identity and empty rows are valid, including a zero-capacity scratch buffer.
    replay.copy_replay_slot_with_version(tables, sizes, cursor, version, src, src, flags)
    replay.copy_replay_slot_with_version(tables, sizes, cursor, version, src[:0], src[:0], flags[:0])
    torch.cuda.synchronize()
    print('PASS: fork flush accepts 1/2/3; dynamic CUDA Graph versions; invalid/identity/empty slots')


if __name__ == '__main__':
    main()
