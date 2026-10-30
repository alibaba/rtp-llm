"""Resolve tagged sparse MLA routes independently of model cache tag names."""

from collections.abc import Mapping

from rtp_llm.models_py.utils.module_profiler import profile_scope


def resolve_mla_routes(fmha, cache, *, sparse: bool, attention_tag="default"):
    if not sparse:
        if isinstance(fmha, Mapping) or isinstance(cache, Mapping):
            raise RuntimeError("dense MLA does not accept tagged cache routes")
        return fmha, cache, None, None
    required = {attention_tag, "indexer_kv"}
    if not isinstance(fmha, Mapping) or set(fmha) != required:
        raise RuntimeError(
            f"sparse MLA requires exactly the {sorted(required)} FMHA routes"
        )
    if cache is None:
        return fmha[attention_tag], None, fmha["indexer_kv"], None
    if not isinstance(cache, Mapping) or set(cache) != required:
        raise RuntimeError(
            f"sparse MLA requires exactly the {sorted(required)} cache routes"
        )
    return (
        fmha[attention_tag],
        cache[attention_tag],
        fmha["indexer_kv"],
        cache["indexer_kv"],
    )


def run_sparse_indexer(
    indexer,
    hidden,
    query_latent,
    cache,
    fmha,
    *,
    encoded_q_lora=None,
    prepared_key=None,
):
    if indexer is None or cache is None:
        return None
    topk = indexer(
        hidden,
        query_latent,
        cache,
        fmha.fmha_params,
        fmha.attn_inputs,
        use_fast_path=not fmha.is_sparse(),
        cp_params=fmha.cp_params,
        **({"encoded_q_lora": encoded_q_lora} if encoded_q_lora is not None else {}),
        **({"prepared_key": prepared_key} if prepared_key is not None else {}),
    )
    _publish_indexer_cache(fmha.attn_inputs, cache)
    return topk


def _publish_indexer_cache(inputs, cache):
    if (
        inputs.is_prefill
        and inputs.cache_store_inputs is not None
        and inputs.cache_store_writer is not None
    ):
        inputs.cache_store_writer.write(inputs.cache_store_inputs, cache)


def run_prepared_sparse_indexer(indexer, query, weights, cache, fmha):
    """Complete history once after natural K ingress and query projection.

    The caller has already written K. Warmup/non-sparse routes still publish
    their cache exactly once, matching run_sparse_indexer's fast path.
    """
    if indexer is None or cache is None:
        return None
    topk = None
    if fmha.is_sparse():
        with profile_scope("RTP::attention.dsa.indexer_core"):
            topk = indexer._compute_topk(
                query,
                weights,
                cache,
                fmha.fmha_params,
                fmha.attn_inputs,
                fmha.cp_params,
            )
    _publish_indexer_cache(fmha.attn_inputs, cache)
    return topk
