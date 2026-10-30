"""Select one causal token from request-major speculative metadata."""

from dataclasses import replace


def slice_compressor_query(meta, batch_size, query_length, query_index):
    if batch_size < 0 or query_length < 1 or not 0 <= query_index < query_length:
        raise ValueError("Invalid speculative query geometry")

    def select(tensor):
        if tensor.numel() != batch_size * query_length:
            raise ValueError("Speculative compressor metadata row counts differ")
        return tensor.reshape(batch_size, query_length)[:, query_index].contiguous()

    values = {
        name: select(getattr(meta, name))
        for name in ("positions", "b_idx", "state_slots", "kv_slots", "token_to_req")
    }
    if meta.compressed_lens_per_token is not None:
        values["compressed_lens_per_token"] = select(meta.compressed_lens_per_token)
    return replace(
        meta,
        **values,
        is_batched=False,
        seq_start_per_req=values["positions"],
        cu_seq_per_req=None,
    )
