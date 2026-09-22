"""K3 target verification and draft updates through RTP's paged MLA planner."""

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    MlaFlashInferDecodeOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferImplBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)


class KimiK3MlaVerifyImpl(MlaFlashInferImplBase):
    """Causal multi-query MLA with fixed storage for Graph replay.

    Main's ordinary prefill backend materializes ragged K/V and is not graph
    safe. Verification instead uses the existing paged MLA operator, whose
    qo_indptr supports several consecutive queries per request.
    """

    def __init__(
        self, config, parallelism, weights, inputs, fmha_config, is_cuda_graph
    ):
        attention = config.getAttentionConfigs(parallelism.get_attn_tp_size())
        batch = inputs.input_lengths.numel()
        tokens = inputs.physical_token_count
        if batch <= 0 or tokens <= 0:
            raise ValueError("K3 paged MLA requires nonempty physical rows")
        # Target verification has a fixed query width per request. Draft
        # prefill instead packs accepted prefixes, including zero-length slots
        # in smaller graph buckets; fill_params builds its ragged qo_indptr.
        if inputs.is_target_verify and tokens % batch:
            raise ValueError("K3 MLA verification requires rectangular physical rows")
        op = MlaFlashInferDecodeOp(
            attention.head_num,
            attention.kv_lora_rank,
            attention.rope_head_dim,
            attention.nope_head_dim,
            attention.kernel_tokens_per_block,
            attention.softmax_extra_scale,
            attention.use_mla,
            False,
            weights.weights,
            max_bs=batch,
            max_context_len=config.max_seq_len,
            num_tokens=tokens,
            is_cuda_graph=is_cuda_graph,
        )
        super().__init__(
            op,
            None,
            MlaKVCacheWriteOp(kv_cache_dtype=attention.kv_cache_dtype),
            inputs,
            attention.kernel_tokens_per_block,
            attention,
            weights.weights,
            None,
            fmha_config,
            quant_config=None,
            max_seq_len=config.max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism,
        )

    def prepare_cuda_graph(self, inputs):
        self.attn_inputs = inputs
        self.prepare(inputs, forbid_realloc=True)
