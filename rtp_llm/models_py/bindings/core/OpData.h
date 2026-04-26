#pragma once
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/models/models_weight/Weights.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include <cstddef>
#include <optional>
#include <memory>
#include <torch/extension.h>
#include <torch/python.h>
#include <ATen/Generator.h>
#include <type_traits>

namespace rtp_llm {

enum class ParallelMode {
    TP        = 0,
    DP        = 1,
    DP_AND_TP = 2,
    FFN_TP    = 3,
    EP        = 4,
    EPLB      = 5,
};

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
    torch::Tensor         lm_output_indexes;  // [sum(lm_output_lengths)]
    torch::Tensor         lm_output_lengths;  // [total_batch_size]
    torch::Tensor         prefix_lengths;     // [context_batch_size]

    torch::Tensor combo_tokens_type_ids;  // [cumulated_seq_len]
    torch::Tensor combo_position_ids;     // [cumulated_seq_len]

    // for mtp model
    torch::Tensor last_hidden_states;

    torch::Tensor attention_mask;  // [batch_size, seq_len, seq_len]

    // - single-type cache: [batch_size, block_nums]
    // - hybrid cache: [group_nums, batch_size, block_nums]
    torch::Tensor kv_cache_block_id;
    torch::Tensor kv_cache_kernel_block_id;  // [group, batch, kernel_blocks], int32

    torch::Tensor kv_cache_layer_to_group;  // [layer_num], int32
    torch::Tensor kv_cache_group_types;     // [group_num], int32, Convention: 0 -> LINEAR, 1 -> FULL.
    torch::Tensor kv_cache_update_mapping;  // [block_copy_num, 2] kv cache update mapping

    std::optional<std::vector<torch::Tensor>> multimodal_features;  // all features in gathered stream stored here
    torch::Tensor text_tokens_mask;  // text part in multimodal input tokens [cumulated_seq_len]
    torch::Tensor mm_features_locs;  // features index

    std::optional<std::vector<torch::Tensor>> input_embeddings;  // all input embeddings in gathered stream stored here
    torch::Tensor                             input_embeddings_locs;  // input embeddings index

    torch::Tensor request_id;             // int64, [context_batch_size]
    torch::Tensor request_pd_separation;  // bool, [context_batch_size]
    torch::Tensor cache_keys;             // [context_batch_size]
    size_t        kv_block_stride_bytes;
    size_t        kv_scale_stride_bytes;
    size_t        seq_size_per_block;
    size_t        kernel_seq_size_per_block = 0;  // 0 means same as seq_size_per_block
    bool          pd_separation             = false;
    bool          decode_entrance           = false;

    bool need_all_logits = false;
    bool need_moe_gating = false;
    bool warmup          = false;
    bool skip_run        = false;
    bool is_fake_stream  = false;

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

struct CopyParams {
    const torch::Tensor& dst;
    const torch::Tensor& src;
    bool                 overlapped = false;
    bool                 async      = true;

    void check() const {
        RTP_LLM_CHECK_WITH_INFO(src.scalar_type() == dst.scalar_type(), "copy dst and src need has same type.");
        RTP_LLM_CHECK_WITH_INFO(
            src.nbytes() == dst.nbytes(), "src and dst copy size mismatch: %zu vs %zu", src.nbytes(), dst.nbytes());
    }
};

struct MultiMergeCopyParams {
    void*               dst_ptr;
    std::vector<void*>  src_ptrs;
    std::vector<size_t> copy_size;
    std::vector<size_t> dst_offsets;
};

struct BatchCopyParams {
    enum CopyType : uint32_t {
        D2H = 0,
        H2D = 1,
        D2D = 2,
        H2H = 3,

        // dummy enum indicating number of copy types
        TYPE_SIZE
    };

    struct BatchCopyBuffers {
        std::vector<void*>       dst_ptr;
        std::vector<const void*> src_ptr;
        std::vector<uint64_t>    sizes;
    };

    BatchCopyBuffers copy_buffers[TYPE_SIZE];

    bool overlapped = false;

    BatchCopyParams& set_overlapped(bool overlapped) {
        this->overlapped = overlapped;
        return *this;
    }

    static CopyType  get_copy_type(MemoryType dst_type, MemoryType src_type);
    BatchCopyParams& reserve(CopyType copy_type, size_t size);
    BatchCopyParams& add(void* dst, const void* src, size_t size, CopyType copy_type);
};

struct KvCacheInfo {
    int           layer_num;
    torch::Tensor kv_cache_block_id;  // [batch_size, block_nums], kv cache block offset
    // only meaningful for hybrid cache, per-group block tables, each is [batch_size, block_nums]
    std::vector<torch::Tensor> kv_cache_block_ids_by_group;
    // Base buffer for kv cache blocks. For current cache layout, this represents the base (K) address of kv blocks.
    // V address can be derived by offset/stride when needed.
    torch::Tensor kv_cache_buffer;
    // Optional scale buffer for kv cache quantization (int8/fp8). If set, it should match kv_cache_buffer layout.
    torch::Tensor kv_scale_buffer;
};

struct CacheStoreInputs {
    torch::Tensor input_lengths_host;
    torch::Tensor prefix_lengths_host;
    torch::Tensor host_kv_cache_offset;

    torch::Tensor kv_cache_layer_to_group_host;
    torch::Tensor kv_cache_group_types_host;  // 0 -> LINEAR, 1 -> FULL.

    size_t context_batch_size = 0;
    size_t decoder_batch_size = 0;

    torch::Tensor            request_id;             // [context_batch_size]
    torch::Tensor            request_pd_separation;  // [context_batch_size]
    std::vector<std::string> cache_keys;             // [context_batch_size]
    size_t                   tokens_per_block;
    size_t                   kv_block_stride_bytes = 0;
    size_t                   kv_scale_stride_bytes = 0;
    bool                     pd_separation         = false;
    size_t                   model_id              = 0;
    bool                     decode_entrance       = false;
    bool                     warmup;

    int layer_id = 0;

    // Pre-created event from the main thread to avoid cudaEventRecord
    // contention on background threads. nullptr means writeCacheStore will
    // create an event on the spot (single-threaded / C++ path).
    std::shared_ptr<torch::Event> pre_created_event = nullptr;
};

struct AttentionCommonInputs {
    // see detailed comments at GptModelInputs
    torch::Tensor input_lengths;     // int32_t, [decoder_batch_size + context_batch_size]
    torch::Tensor sequence_lengths;  // int32_t, [decoder_batch_size]

    std::optional<KvCacheInfo>      kv_cache;
    std::optional<CacheStoreInputs> cache_store_inputs;

    // Hybrid cache helper: layer_id -> kv cache group id (host-side).
    // When kv_cache->kv_cache_block_ids_by_group is non-empty, model will select the right group per layer
    // and set kv_cache->kv_cache_block_id before calling attention ops.
    std::vector<int32_t> kv_cache_layer_to_group_id;

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
    ParamsPtr prefill_trt_attn;
    ParamsPtr decode_trt_attn;
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

struct GreedyParams {
    torch::Tensor logits;            // [batch_size, vocab_size_padded], mutable for in-place penalty
    torch::Tensor input_lengths;     // [batch_size]
    torch::Tensor sequence_lengths;  // [batch_size]
    torch::Tensor token_ids;         // [batch_size, max_input_length + 1]
    size_t        step;

    torch::Tensor top_k;
    torch::Tensor top_p;
    torch::Tensor temperature;

    std::optional<torch::Tensor> repetition_penalty;
    std::optional<torch::Tensor> no_repeat_ngram_size;

    std::optional<torch::Tensor> cum_log_probs;
    std::optional<torch::Tensor> output_log_probs;

    std::optional<torch::Tensor> output_all_probs;
    std::optional<torch::Tensor> presence_penalty;
    std::optional<torch::Tensor> frequency_penalty;
    std::optional<torch::Tensor> do_sample;

    std::vector<at::Generator> generator;
};

struct GreedyOutput {
    torch::Tensor success;
};

struct BeamSearchParams {
    const torch::Tensor& logits;            // [batch_size, num_beams_in, vocab_size]
    torch::Tensor        token_ids;         // [batch_size, num_beams_in, max_seq_len]
    torch::Tensor        input_lengths;     // [batch_size, num_beams_in]
    torch::Tensor        sequence_lengths;  // [batch_size, num_beams_in]
    torch::Tensor        cum_log_probs;     // [batch_size, num_beams_in]
    size_t               num_beams_out = 0;
};

struct BeamSearchOutput {
    torch::Tensor token_ids;         // [batch_size, num_beams_out, max_seq_len]
    torch::Tensor input_lengths;     // [batch_size, num_beams_out]
    torch::Tensor sequence_lengths;  // [batch_size, num_beams_out]
    torch::Tensor cum_log_probs;     // [batch_size, num_beams_out]
    torch::Tensor beam_indices;      // [batch_size, num_beams_out]
};

struct BroadcastParams {
    const std::vector<torch::Tensor>& buffers;
    const int64_t                     root;
    ParallelMode                      mode       = ParallelMode::TP;
    bool                              overlapped = false;
};

enum class ReduceOp {
    Sum  = 0,
    Prod = 1,
    Max  = 2,
    Min  = 3,
    Avg  = 4,
};

struct AllReduceParams {
    torch::Tensor  buffer;
    const ReduceOp op;
    bool           overlapped = false;
    ParallelMode   mode       = ParallelMode::TP;
    torch::Tensor  dest;  // undefined = no separate dest
};

struct AllReduceOutput {
    torch::Tensor buffer;
};

struct AllGatherParams {
    const std::vector<torch::Tensor>& recv_buffers;
    ParallelMode                      mode = ParallelMode::TP;
    std::vector<torch::Tensor>        send_buffers;
    bool                              inplace    = true;
    bool                              overlapped = false;
};

struct SpeculativeSamplingParams {
    torch::Tensor& draft_probs_d;
    torch::Tensor& draft_token_ids_d;
    torch::Tensor& uniform_samples_d;
    torch::Tensor& target_probs_d;
    torch::Tensor& output_token_ids_d;
    torch::Tensor& output_accepted_token_num_d;
    torch::Tensor& output_emitted_token_num_d;

    SpeculativeSamplingParams(torch::Tensor& draft_probs_d,
                              torch::Tensor& draft_token_ids_d,
                              torch::Tensor& uniform_samples_d,
                              torch::Tensor& target_probs_d,
                              torch::Tensor& output_token_ids_d,
                              torch::Tensor& output_accepted_token_num_d,
                              torch::Tensor& output_emitted_token_num_d):
        draft_probs_d(draft_probs_d),
        draft_token_ids_d(draft_token_ids_d),
        uniform_samples_d(uniform_samples_d),
        target_probs_d(target_probs_d),
        output_token_ids_d(output_token_ids_d),
        output_accepted_token_num_d(output_accepted_token_num_d),
        output_emitted_token_num_d(output_emitted_token_num_d) {}
};

}  // namespace rtp_llm
