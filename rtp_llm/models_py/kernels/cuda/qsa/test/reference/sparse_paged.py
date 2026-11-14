import torch


def _reference(q, cache, block_table, kv_lens, selected):
    batch, q_heads, query_len, head_dim = q.shape
    page_size = cache.shape[3]
    kv_heads = cache.shape[2]
    out = torch.zeros_like(q)
    for b in range(batch):
        for h in range(q_heads):
            kv_head = h // (q_heads // kv_heads)
            for s in range(query_len):
                indices = selected[b, s]
                indices = indices[(indices >= 0) & (indices < kv_lens[b, s])]
                if not indices.numel():
                    continue
                keys, values = [], []
                for index in indices.tolist():
                    page = int(block_table[b, index // page_size])
                    offset = index % page_size
                    keys.append(cache[page, 0, kv_head, offset])
                    values.append(cache[page, 1, kv_head, offset])
                keys = torch.stack(keys).float()
                values = torch.stack(values).float()
                score = (q[b, h, s].float() @ keys.T) / head_dim**0.5
                out[b, h, s] = (score.softmax(dim=-1) @ values).to(out.dtype)
    return out
