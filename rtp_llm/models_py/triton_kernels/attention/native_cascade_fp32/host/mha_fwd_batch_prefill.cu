#include "mha_fwd.h"
#include <string>

namespace aiter {
mha_batch_prefill_traits
get_mha_batch_prefill_traits(int head_size_q,
                             int head_size_v,
                             std::string dtype,
                             bool is_group_mode,
                             bool has_logits_soft_cap,
                             mask_enum mask_type,
                             bias_enum bias_type,
                             bool has_lse,
                             bool has_dropout,
                             quant_scale_enum qscale_type,
                             ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout,
                             ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table,
                             int page_size,
                             bool skip_min_seqlen_q = false,
                             bool has_sink         = false)
{
    return mha_batch_prefill_traits(head_size_q,
                                    head_size_v,
                                    dtype,
                                    is_group_mode,
                                    has_logits_soft_cap,
                                    mask_type,
                                    bias_type,
                                    has_lse,
                                    has_dropout,
                                    qscale_type,
                                    skip_min_seqlen_q,
                                    has_sink,
                                    kv_memory_layout,
                                    kv_lookup_table,
                                    page_size);
}

float mha_batch_prefill(mha_batch_prefill_args args,
                        const ck_tile::stream_config& stream_config,
                        std::string q_dtype_str,
                        bool is_group_mode,
                        mask_enum mask_type,
                        bias_enum bias_type,
                        bool has_lse,
                        quant_scale_enum qscale_type,
                        bool use_ext_asm)
{
    int head_size_q  = args.hdim_q;
    int head_size_v  = args.hdim_v;
    bool has_dropout = args.p_drop > 0.f;
    bool has_sink    = args.sink_size > 0;

    // The kUseGlobalLoad decision (>2GB KV cache → use `global_load_lds_*`
    // instead of SRD `buffer_load_*`) is made per-arm inside the auto-generated
    // dispatcher in fmha_batch_prefill_api.cpp, where each arm knows its own
    // compile-time bn0 and dtype element size. The wrapper just forwards args;
    // no runtime trait field for it.
    auto traits      = get_mha_batch_prefill_traits(head_size_q,
                                               head_size_v,
                                               q_dtype_str,
                                               is_group_mode,
                                               args.logits_soft_cap > 0.f,
                                               mask_type,
                                               bias_type,
                                               has_lse,
                                               has_dropout,
                                               qscale_type,
                                               args.kv_memory_layout,
                                               args.kv_lookup_table,
                                               args.page_block_size,
                                               /*skip_min_seqlen_q=*/false,
                                               has_sink);
    return fmha_batch_prefill(traits, args, stream_config);
}

} // namespace aiter
