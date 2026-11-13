"""FlashInfer token-CSR attention for cold QSA prefill."""

import torch


class ColdPrefillBSR:
    """Exact token CSR adapter for eager BF16 prefill."""

    def __init__(self, device):
        from flashinfer.sparse import BlockSparseAttentionWrapper

        self.stream = torch.cuda.current_stream(device)

        self.workspace = torch.empty(
            128 * 1024 * 1024, device=device, dtype=torch.uint8
        )
        self.wrapper = BlockSparseAttentionWrapper(self.workspace, backend="auto")

    def __call__(self, q, k, v, selected):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("token CSR compaction and host planning are eager only")
        # Production cold Q/K/V are strided slices of packed QKV. Include
        # their normalization cost in this adapter, rather than timing only
        # the contiguous synthetic operands of the earlier core benchmark.
        q, k, v = (tensor.contiguous() for tensor in (q, k, v))
        tokens, heads, dim = q.shape
        valid = selected >= 0
        count = valid.sum(-1, dtype=torch.int32)
        indptr = torch.cat((count.new_zeros(1), count.cumsum(0, dtype=torch.int32)))
        indices = selected[valid].int()
        self.wrapper.plan(
            indptr,
            indices,
            M=tokens,
            N=k.shape[0],
            R=1,
            C=1,
            num_qo_heads=heads,
            num_kv_heads=k.shape[1],
            head_dim=dim,
            causal=False,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            o_data_type=q.dtype,
        )
        return self.wrapper.run(q, k, v)


def try_flashinfer_cold_prefill(impl, q, k, v, selected, lengths, *, trusted_selected):
    if not (
        trusted_selected
        and not impl._is_mtp_draft
        and not impl.attn_inputs.is_cuda_graph
        and impl.head_dim == 256
        and q.is_cuda
        and torch.version.hip is None
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and selected.dtype == torch.int32
        and q.device == k.device == v.device == selected.device
        and torch.cuda.get_device_capability(q.device)[0] == 10
        and not torch.cuda.is_current_stream_capturing()
    ):
        return None
    adapter = getattr(impl, "_flashinfer_cold_prefill_adapter", None)
    if adapter is None:
        adapter = ColdPrefillBSR(q.device)
        impl._flashinfer_cold_prefill_adapter = adapter
    elif adapter.stream != torch.cuda.current_stream(q.device):
        return None

    # Cap the temporary selection/CSR footprint per sequence. Query IDs in
    # every chunk still refer to this request's unchanged K/V prefix.
    outputs = []
    offset = 0
    for length in lengths:
        end = offset + length
        keys = (
            k[offset:end].reshape(length, impl.kv_head_num, impl.head_dim).contiguous()
        )
        values = v[offset:end].reshape_as(keys).contiguous()
        for start in range(offset, end, 8192):
            stop = min(start + 8192, end)
            query = q[start:stop].reshape(stop - start, impl.head_num, impl.head_dim)
            outputs.append(
                adapter(query, keys, values, selected[start:stop]).reshape(
                    -1, impl.q_width
                )
            )
        offset = end
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)
