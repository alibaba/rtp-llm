#pragma once
#include "rtp_llm/cpp/cache/CacheStoreTypes.h"
#include "rtp_llm/cpp/comm/CollectiveTypes.h"
#include "rtp_llm/cpp/core/CopyTypes.h"
#include "rtp_llm/cpp/models/SamplingTypes.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/models/models_weight/Weights.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"
#include <cstddef>
#include <optional>
#include <string>
#include <memory>
#include <torch/extension.h>
#include <torch/python.h>
#include <type_traits>

namespace rtp_llm {

// A batch includes two parts: context batch and decoder batch.
// context batch is request for initial word, decoder batch is request for incremental word.
// ids and lengths are int32_t
struct GptModelInputs {
    // input_lengths holds original input length for requests,
    // shape [decoder_batch_size + context_batch_size], int32
    // sequence_lengths holds current sequence length for incremental decoding requests,
    // shape [decoder_batch_size], int32
    mutable torch::Tensor combo_tokens;       // [cumulated_seq_len]
    torch::Tensor         input_lengths;      // [batch_size]
    torch::Tensor         sequence_lengths;   // [decoder_batch_size]
    torch::Tensor         lm_output_indexes;  // selected output rows
    // Kept for ModelInputsLogger/legacy micro-batch consumers; the async
    // scheduling redesign no longer populates it (stays undefined).
    torch::Tensor lm_output_lengths;        // [total_batch_size]
    torch::Tensor prefix_lengths;           // [context_batch_size]
    torch::Tensor sequence_lengths_plus_1;  // optional CUDA mirror for target-verify linear attention

    torch::Tensor combo_tokens_type_ids;  // [cumulated_seq_len]
    torch::Tensor combo_position_ids;     // [cumulated_seq_len]

    // for mtp model
    torch::Tensor last_hidden_states;

    torch::Tensor attention_mask;  // [batch_size, seq_len, seq_len]

    // - single-type cache: [batch_size, block_nums]
    // - hybrid cache: [group_nums, batch_size, block_nums]
    torch::Tensor kv_cache_block_id;
    torch::Tensor kv_cache_kernel_block_id;  // [group, batch, kernel_blocks], int32

    torch::Tensor kv_cache_group_types;     // [group_num], int32, Convention: 0 -> LINEAR, 1 -> FULL.
    torch::Tensor kv_cache_update_mapping;  // [block_copy_num, 3]: group_id, src block, dst block

    std::optional<std::vector<torch::Tensor>> multimodal_features;  // all features in gathered stream stored here
    torch::Tensor text_tokens_mask;  // text part in multimodal input tokens [cumulated_seq_len]
    torch::Tensor mm_features_locs;  // features index
    std::optional<std::vector<torch::Tensor>>
        mm_extra_input;  // model-specific extra input (opaque flat 1-D, e.g. deepstack)

    std::optional<std::vector<torch::Tensor>> input_embeddings;  // all input embeddings in gathered stream stored here
    torch::Tensor                             input_embeddings_locs;  // input embeddings index

    torch::Tensor request_id;             // int64, [context_batch_size]
    torch::Tensor request_pd_separation;  // bool, [context_batch_size]
    torch::Tensor cache_keys;             // [context_batch_size]
    // Physical KV-manager block strides. These are independent of any kernel-block view exposed to attention ops.
    size_t kv_block_stride_bytes;
    size_t kv_scale_stride_bytes;
    size_t seq_size_per_block;
    size_t kernel_seq_size_per_block = 0;  // 0 means same as seq_size_per_block
    bool   pd_separation             = false;
    bool   decode_entrance           = false;
    bool   use_opaque_kv_cache_store = false;

    bool need_all_logits        = false;
    // Set when any stream requests return_all_hidden_states. Gates whether the
    // CP prefill exit must materialize the full [seq, hidden] all_hidden_states
    // (true) or may gather only the last-token rows lm_head needs (false).
    bool need_all_hidden_states = false;
    bool need_moe_gating        = false;
    bool warmup                 = false;
    bool skip_run               = false;
    bool is_fake_stream         = false;

    // Linear attention target verify should write draft tokens mamba states
    // to extra kv_cache blocks when normal inference only write last token mamba state.
    // So, the model has different inference logic for target verify and normal inference.
    // To select correct inference mode, we need to set this flag manually.
    bool is_target_verify = false;

    // not sync to other tp rank
    std::vector<std::string> trace_ids;

public:
    std::string debugString(bool force = false) const;
};

struct GptModelOutputs {
    torch::Tensor logits;
    torch::Tensor hidden_states;
    torch::Tensor all_hidden_states;
    torch::Tensor all_logits;
    torch::Tensor softmax_result;

    std::vector<torch::Tensor> moe_gating;
};

struct AttentionCommonInputs {
    // see detailed comments at GptModelInputs
    torch::Tensor input_lengths;     // int32_t, [decoder_batch_size + context_batch_size]
    torch::Tensor sequence_lengths;  // int32_t, [decoder_batch_size]

    std::optional<KvCacheInfo> kv_cache;

    torch::Tensor cu_seqlens;
    torch::Tensor cu_kv_seqlens;
    torch::Tensor kv_seqlens;
    torch::Tensor padding_offset;

    size_t context_batch_size      = 0;
    size_t decoder_batch_size      = 0;
    size_t context_max_seq_len     = 0;
    size_t decoder_max_seq_len     = 0;
    size_t context_token_num       = 0;
    size_t context_total_kv_length = 0;

    torch::Tensor position_ids;
    torch::Tensor attention_mask;
    torch::Tensor linear_bias_slopes;
    torch::Tensor prefix_prompt_lengths;
    int32_t       max_prefix_length = 0;

    ParamsPtr prefill_flash_infer_attn;
    ParamsPtr decode_flash_infer_attn;
};

struct AttentionModuleParams {
    const int32_t layer_id;
    // qkv shape[h_token_num, (head_num + 2 * kv_head_num) * size_per_head]
    const torch::Tensor& input;
    torch::Tensor&       output;  // shape [token_num, size_per_head]

    AttentionCommonInputs&       common;
    const AttentionLayerWeights& weights;
    const AttentionConfigs&      configs;
    const QScheme                qscheme;
    const DataType               compute_type = DataType::TYPE_INVALID;
};

struct MoeConfigs {
    size_t expert_num;
    size_t extra_expert_num = 0;
    size_t top_k;

    bool   normalize_expert_scale = false;
    bool   has_moe_norm           = false;
    bool   use_all_gather         = false;
    size_t ep_rank                = 0;
    size_t ep_size                = 1;
    size_t tp_rank                = 0;
    size_t tp_size                = 1;
    size_t dp_rank                = 0;
    size_t dp_size                = 1;

    int    scoring_func          = 0;  // 0: softmax, 1: sigmoid
    int    topk_group            = 1;
    int    n_group               = 1;
    double routed_scaling_factor = 1.0;  // used in deepseek v2 and glm4 moe

    bool enable_eplb = false;
    // NOTE(yinzhi): not used yet
    EplbBalanceMethod balance_method = EplbBalanceMethod::EQUAL;
};

struct FfnConfigs {
    ActivationType            activation_type;
    std::optional<MoeConfigs> moe_configs = std::nullopt;
};

}  // namespace rtp_llm
