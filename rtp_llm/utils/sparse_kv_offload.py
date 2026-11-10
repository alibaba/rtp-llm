def sparse_kv_offload_max_selected_rows(
    hot_rows: int, top_k: int, batch_size: int, gen_num_per_cycle: int
) -> int:
    """Size hot KV for all requests and queries without changing model top-k."""
    if hot_rows <= 0 or top_k <= 0 or hot_rows < top_k:
        raise ValueError(
            "sparse_kv_offload_hot_rows must be positive and at least "
            f"indexer_topk ({top_k}), got {hot_rows}"
        )
    selected = batch_size * max(1, gen_num_per_cycle + 1) * hot_rows
    if not 0 < selected <= 2**31 - 1:
        raise ValueError("sparse KV offload max_selected_rows must fit positive int32")
    return selected
