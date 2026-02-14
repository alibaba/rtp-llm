#pragma once
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/devices/LoraWeights.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "rtp_llm/cpp/model_utils/layernorm_types.h"
#include "rtp_llm/cpp/utils/StackTrace.h"
#include <cstddef>
#include <optional>
#include <functional>
#include <sstream>
#include <memory>
#include <torch/extension.h>
#include <torch/python.h>
#include <ATen/Generator.h>
#include <type_traits>

namespace rtp_llm {

class DeviceBase;

enum class OpErrorType {
    ERROR_NONE,
    ERROR_INVALID_ARGS,
    ERROR_RESOURCE_EXHAUSTED,
    ERROR_UNIMPLEMENTED,
    ERROR_INTERNAL,
    ERROR_UNKNOWN,
};

enum class ParallelMode {
    TP        = 0,
    DP        = 1,
    DP_AND_TP = 2,
    FFN_TP    = 3,
    EP        = 4,
    EPLB      = 5
};

enum class DeviceStream {
    DEFAULT = 0,
};

class OpStatus {
public:
    OpStatus(OpErrorType, const std::string& message = ""):
        error_type(OpErrorType::ERROR_NONE), error_message(message) {}

    static OpStatus make(OpErrorType error_type, const std::string& error_message = "") {
        return OpStatus(error_type, error_message);
    }
    static OpStatus OK() {
        return OpStatus(OpErrorType::ERROR_NONE);
    }

    bool ok() const {
        return error_type == OpErrorType::ERROR_NONE;
    }

public:
    OpErrorType error_type;
    std::string error_message;
};

class OpException: public std::exception {
public:
    OpException(const OpStatus& status);

    const char* what() const noexcept override {
        return detail_str_.c_str();
    }

    const OpStatus& status() const {
        return status_;
    }

private:
    OpStatus            status_;
    mutable std::string detail_str_;
};

using OptionalConstBufferRef       = std::optional<std::reference_wrapper<const Buffer>>;
using OptionalBufferRef            = std::optional<std::reference_wrapper<Buffer>>;
using OptionalConstVecBufferPtrRef = std::optional<std::reference_wrapper<const std::vector<BufferPtr>>>;

using CloneOutput = BufferPtr;

// A batch includes two parts: context batch and decoder batch.
// context batch is request for initial word, decoder batch is request for incremental word.
// ids and lengths are int32_t
struct GptModelInputs {
    // input_lengths holds original input length for requests,
    // shape [decoder_batch_size + context_batch_size], int32
    // sequence_lengths holds current sequence length for incremental decoding requests,
    // shape [decoder_batch_size], int32
    mutable rtp_llm::BufferPtr combo_tokens;       // [cumulated_seq_len]
    rtp_llm::BufferPtr         input_lengths;      // [batch_size]
    rtp_llm::BufferPtr         sequence_lengths;   // [decoder_batch_size]
    rtp_llm::BufferPtr         lm_output_indexes;  // [sum(lm_output_lengths)]
    rtp_llm::BufferPtr         lm_output_lengths;  // [total_batch_size]
    rtp_llm::BufferPtr         prefix_lengths;     // [context_batch_size]

    rtp_llm::BufferPtr combo_tokens_type_ids;  // [cumulated_seq_len]
    rtp_llm::BufferPtr combo_position_ids;     // [cumulated_seq_len]

    // for mtp model
    rtp_llm::BufferPtr last_hidden_states;

    // for tp sync
    rtp_llm::BufferPtr lora_ids;            // [batch_size]
    rtp_llm::BufferPtr lora_input_lengths;  // [batch_size]

    // no need tp sync
    rtp_llm::lora::LoraModelInputPtr lora_model_input;

    rtp_llm::BufferPtr attention_mask;  // [batch_size, seq_len, seq_len]

    // - single-type cache: [batch_size, block_nums]
    // - hybrid cache: [group_nums, batch_size, block_nums]
    rtp_llm::BufferPtr kv_cache_block_id;

    rtp_llm::BufferPtr kv_cache_layer_to_group;  // [layer_num], int32
    rtp_llm::BufferPtr kv_cache_group_types;     // [group_num], int32, Convention: 0 -> LINEAR, 1 -> FULL.
    rtp_llm::BufferPtr kv_cache_update_mapping;  // [block_copy_num, 2] kv cache update mapping

    std::optional<std::vector<rtp_llm::BufferPtr>> multimodal_features;  // all features in gathered stream stored here
    rtp_llm::BufferPtr text_tokens_mask;  // text part in multimodal input tokens [cumulated_seq_len]
    rtp_llm::BufferPtr mm_features_locs;  // features index

    std::optional<std::vector<rtp_llm::BufferPtr>>
                       input_embeddings;       // all input embeddings in gathered stream stored here
    rtp_llm::BufferPtr input_embeddings_locs;  // input embeddings index

    rtp_llm::BufferPtr request_id;             // int64, [context_batch_size]
    rtp_llm::BufferPtr request_pd_separation;  // bool, [context_batch_size]
    rtp_llm::BufferPtr cache_keys;             // [context_batch_size]
    size_t             kv_block_stride_bytes;
    size_t             kv_scale_stride_bytes;
    size_t             seq_size_per_block;
    bool               pd_separation   = false;
    bool               decode_entrance = false;

    bool need_all_logits = false;
    bool need_moe_gating = false;
    bool warmup          = false;
    bool skip_run        = false;
    bool is_fake_stream  = false;

    // not sync to other tp rank
    std::vector<std::string> trace_ids;

public:
    std::string debugString(bool force = false) const;
};

struct GptModelOutputs {
    rtp_llm::BufferPtr logits;
    rtp_llm::BufferPtr hidden_states;
    rtp_llm::BufferPtr all_hidden_states;
    rtp_llm::BufferPtr all_logits;
    rtp_llm::BufferPtr softmax_result;

    std::vector<rtp_llm::BufferPtr> moe_gating;

    mutable rtp_llm::BufferPtr scatter_logits;
    mutable rtp_llm::BufferPtr scatter_hidden_states;
    std::shared_ptr<void>      captured_values;
};

struct CloneParams {
    CloneParams(const Buffer&        input,
                const AllocationType alloc_type = AllocationType::DEVICE,
                const BufferHints&   hints      = BufferHints(),
                bool                 overlapped = false,
                bool                 async      = true):
        input(input), alloc_type(alloc_type), hints(hints), overlapped(overlapped), async(async) {}

    const Buffer&        input;
    const AllocationType alloc_type;
    const BufferHints&   hints;
    bool                 overlapped = false;
    bool                 async      = true;
};

struct SliceParams {
    const Buffer& input;
    int64_t       dim;
    int64_t       start;
    int64_t       end;
    int64_t       step = 1;
};

using SliceOutput = BufferPtr;

struct CopyParams {
    const Buffer&      dst;
    const Buffer&      src;
    bool               overlapped = false;
    const DeviceStream stream     = DeviceStream::DEFAULT;
    bool               async      = true;

    void check() const {
        RTP_LLM_CHECK_WITH_INFO(
            src.type() == dst.type(), "copy dst[%d] and src[%d] need has same type.", src.type(), dst.type());
        RTP_LLM_CHECK_WITH_INFO(src.sizeBytes() == dst.sizeBytes(),
                                "src and dst copy size mismatch: [%s] vs [%s]",
                                src.debugString().c_str(),
                                dst.debugString().c_str());
    }
};

struct MultiMergeCopyParams {
    void*               dst_ptr;
    std::vector<void*>  src_ptrs;
    std::vector<size_t> copy_size;
    std::vector<size_t> dst_offsets;
};

struct MultiCopyParams {
    std::vector<BufferPtr> multi_dst;
    std::vector<BufferPtr> multi_src;
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

    bool         overlapped = false;
    DeviceStream stream     = DeviceStream::DEFAULT;

    BatchCopyParams& set_overlapped(bool overlapped) {
        this->overlapped = overlapped;
        return *this;
    }
    BatchCopyParams& set_stream(const DeviceStream stream) {
        this->stream = stream;
        return *this;
    }

    static CopyType  get_copy_type(MemoryType dst_type, MemoryType src_type);
    BatchCopyParams& reserve(CopyType copy_type, size_t size);
    BatchCopyParams& add(const Buffer& dst, const Buffer& src);
    BatchCopyParams& add(void* dst, const void* src, size_t size, CopyType copy_type);
};

using SelectOutput = BufferPtr;

enum SelectType {
    LAST  = 0,
    FIRST = 1,
};

struct SelectParams {
    const Buffer& input;
    const Buffer& index;
    size_t        dim = 0;
};

using TransposeOutput = BufferPtr;

struct TransposeParams {
    const Buffer& input;
    bool          overlapped = false;
};

using ConvertOutput = BufferPtr;

struct ConvertParams {
    const BufferPtr input;
    const DataType  type;
};

using ConcatOutput = BufferPtr;

struct ConcatParams {
    const std::vector<BufferPtr>& inputs;
    const size_t                  dim = 0;
};

struct SplitOutput {
    std::vector<BufferPtr> outputs;
};

struct SplitParams {
    const Buffer&              input;
    const std::vector<size_t>& split_sizes;
    const size_t               dim        = 0;
    bool                       overlapped = false;
};

struct LayernormOutput {
    BufferPtr output;
    BufferPtr before_norm_output;
};

struct AddBiasOutput {
    BufferPtr output;
};

struct QkRmsNormParams {
    BufferPtr                                                           input;
    const std::optional<std::reference_wrapper<const LayerNormWeights>> q_norm_weight;
    const std::optional<std::reference_wrapper<const LayerNormWeights>> k_norm_weight;
    double                                                              eps;
    size_t                                                              q_group_num;
    size_t                                                              k_group_num;
    size_t                                                              norm_size;
};

using QkRmsNormOutput = BufferPtr;

struct LayernormWithStrideParams {
    BufferPtr                                                           input;
    const std::optional<std::reference_wrapper<const LayerNormWeights>> norm_weight;
    double                                                              eps;
    NormType                                                            norm_type;
    size_t                                                              offset;
    // do normalize for each group in norm_group
    size_t  norm_group_size;
    QScheme qscheme  = QScheme::NoQuantize;
    bool    in_place = true;
};

struct LayernormParams {
    LayernormParams(BufferPtr                                                           input,
                    BufferPtr                                                           before_norm_output,
                    const std::optional<std::reference_wrapper<const LayerNormWeights>> norm_weight,
                    OptionalConstBufferRef                                              residual1  = std::nullopt,
                    OptionalConstBufferRef                                              residual2  = std::nullopt,
                    OptionalConstBufferRef                                              bias       = std::nullopt,
                    double                                                              alpha      = 1.0f,
                    double                                                              eps        = 1e-5,
                    bool                                                                is_inplace = true,
                    bool                                                                return_normed_output = false,
                    NormType                                                            norm_type = NormType::layernorm,
                    QScheme                                                             qscheme   = QScheme::NoQuantize,
                    bool                                                                attn_swap_comm_buffer = false,
                    bool                                                                ffn_swap_comm_buffer  = false):
        input(std::move(input)),
        before_norm_output(std::move(before_norm_output)),
        norm_weight(norm_weight),
        residual1(residual1),
        residual2(residual2),
        bias(bias),
        norm_type(norm_type),
        alpha(alpha),
        eps(eps),
        return_normed_output(return_normed_output),
        is_inplace(is_inplace),
        qscheme(qscheme),
        attn_swap_comm_buffer(attn_swap_comm_buffer),
        ffn_swap_comm_buffer(ffn_swap_comm_buffer) {};

    BufferPtr input;
    BufferPtr before_norm_output;

    const std::optional<std::reference_wrapper<const LayerNormWeights>> norm_weight;

    const OptionalConstBufferRef residual1;
    const OptionalConstBufferRef residual2;
    const OptionalConstBufferRef bias;

    const NormType norm_type;

    const double alpha;
    const double eps;

    const bool    return_normed_output;
    const bool    is_inplace;
    const QScheme qscheme;

    bool attn_swap_comm_buffer = false;
    bool ffn_swap_comm_buffer  = false;
};

enum GemmType : size_t {
    InvalidGemm = 0,

    BufferA_BufferB_BufferC_2DGemm,
    BufferA_BufferB_BufferC_3DGemm,

    QBufferA_BufferB_BufferC_2DGemm,
    BufferA_QBufferB_BufferC_2DGemm,
    QBufferA_QBufferB_BufferC_2DGemm,
};

struct AddBiasParams {
    BufferPtr     input;
    const Buffer& bias;
    bool          inplace = true;
};

enum TransposeOperation {
    NONE,
    TRANSPOSE,
};

std::string inline enumToString(TransposeOperation type) {
    if (type == NONE) {
        return "NONE";
    } else {
        return "TRANSPOSE";
    }
};

// D = alpha * op(A) * op(B) + beta * C
// shapes of A, B, C, D have two options: [m, k], [k, n], [m, n] / [1, n], [m, n]
// or [bs, m, k], [bs, k, n], [bs, m, n], [bs, m, n] where bs is batch_size
// D is optional, if not passed, it will be allocated by the op
struct GemmParams {
    GemmParams(const Buffer&          A,
               const Buffer&          B,
               OptionalConstBufferRef C              = std::nullopt,
               BufferPtr              D              = nullptr,
               const DataType         compute_type   = DataType::TYPE_INVALID,
               const DataType         D_type         = DataType::TYPE_INVALID,
               TransposeOperation     transA         = TransposeOperation::NONE,
               TransposeOperation     transB         = TransposeOperation::NONE,
               const ActivationType   activationType = ActivationType::Identity,
               const float            alpha          = 1.0f,
               const float            beta           = 0.0f,
               int                    math_sm_count  = 0,
               void*                  stream         = nullptr):
        A(A),
        B(B),
        C(C),
        D(D),
        compute_type(compute_type),
        D_type(D != nullptr ? D->type() : D_type),
        transA(transA),
        transB(transB),
        activationType(activationType),
        alpha(alpha),
        beta(beta),
        math_sm_count(math_sm_count),
        stream(stream) {}

    const Buffer&          A;
    const Buffer&          B;
    OptionalConstBufferRef C;
    BufferPtr              D;
    const DataType         compute_type = DataType::TYPE_INVALID;  // If passed invalid type, op should infer type
    const DataType         D_type =
        DataType::TYPE_INVALID;  // for D is null, you can pass output_type to set output_buffer type

    const TransposeOperation transA = TransposeOperation::NONE;
    const TransposeOperation transB = TransposeOperation::NONE;

    ActivationType activationType = ActivationType::Identity;

    const float alpha         = 1.0f;
    const float beta          = 0.0f;
    mutable int math_sm_count = 0;
    void*       stream        = nullptr;
    // tmp param, should use qscheme instead of this bool
    QScheme qscheme = QScheme::NoQuantize;

    void     check() const;
    GemmType dispatch() const;
};

struct GroupedGemmOutput {
    std::vector<BufferPtr> output;
};

// C = alpha * op(A) * op(B) + beta * C
// shapes of each A, B, C needs to be [m, k], [k, n], [m, n]
struct GroupedGemmParams {

    const std::vector<BufferPtr>&         A;
    const std::vector<BufferPtr>&         B;
    std::optional<std::vector<BufferPtr>> C     = std::nullopt;
    const float                           alpha = 1.0f;
    const float                           beta  = 1.0f;

    void check() const;
};

using MultiplyOutput = BufferPtr;

// output = A * B
// A: [m], B: [m] or [m, dim_1, ..., dim_n]
struct MultiplyParams {
    const Buffer& A;
    const Buffer& B;
    BufferPtr     output = nullptr;
};

struct EmbeddingLookupParams {
    const Buffer& combo_tokens;
    const Buffer& embedding_table;
    double        input_embedding_scalar = 1;

    OptionalConstBufferRef text_tokens_mask;

    OptionalConstBufferRef position_ids;
    OptionalConstBufferRef position_table;

    OptionalConstBufferRef token_types;
    OptionalConstBufferRef token_type_table;
};

struct KvCacheInfo {
    int       layer_num;
    BufferPtr kv_cache_block_id;  // [batch_size, block_nums], kv cache block offset
    // only meaningful for hybrid cache, per-group block tables, each is [batch_size, block_nums]
    std::vector<BufferPtr> kv_cache_block_ids_by_group;
    // Base buffer for kv cache blocks. For current cache layout, this represents the base (K) address of kv blocks.
    // V address can be derived by offset/stride when needed.
    BufferPtr kv_cache_buffer;
    // Optional scale buffer for kv cache quantization (int8/fp8). If set, it should match kv_cache_buffer layout.
    BufferPtr kv_scale_buffer;
};

struct MultimodalEmbeddingParams {
    const BufferPtr&             word_embeddings;
    OptionalConstVecBufferPtrRef multimodal_features;
    OptionalConstBufferRef       multimodal_locs;
};

struct InputEmbeddingParams {
    const BufferPtr&             word_embeddings;
    OptionalConstVecBufferPtrRef input_embeddings;
    OptionalConstBufferRef       input_embeddings_locs;
};

using MultimodalEmbeddingOutput = BufferPtr;

struct CacheStoreInputs {
    BufferPtr input_lengths_host;
    BufferPtr prefix_lengths_host;
    BufferPtr host_kv_cache_offset;

    BufferPtr kv_cache_layer_to_group_host;
    BufferPtr kv_cache_group_types_host;  // 0 -> LINEAR, 1 -> FULL.

    size_t context_batch_size = 0;
    size_t decoder_batch_size = 0;

    BufferPtr                request_id;             // [context_batch_size]
    BufferPtr                request_pd_separation;  // [context_batch_size]
    std::vector<std::string> cache_keys;             // [context_batch_size]
    size_t                   tokens_per_block;
    size_t                   kv_block_stride_bytes = 0;
    size_t                   kv_scale_stride_bytes = 0;
    bool                     pd_separation         = false;
    size_t                   model_id              = 0;
    bool                     decode_entrance       = false;
    bool                     warmup;

    int layer_id = 0;
};

struct AttentionCommonInputs {
    // see detailed comments at GptModelInputs
    ConstBufferPtr input_lengths;     // int32_t, [decoder_batch_size + context_batch_size]
    ConstBufferPtr sequence_lengths;  // int32_t, [decoder_batch_size]

    std::optional<KvCacheInfo>      kv_cache;
    std::optional<CacheStoreInputs> cache_store_inputs;

    // Hybrid cache helper: layer_id -> kv cache group id (host-side).
    // When kv_cache->kv_cache_block_ids_by_group is non-empty, model will select the right group per layer
    // and set kv_cache->kv_cache_block_id before calling attention ops.
    std::vector<int32_t> kv_cache_layer_to_group_id;

    ConstBufferPtr cu_seqlens;
    ConstBufferPtr cu_kv_seqlens;
    ConstBufferPtr kv_seqlens;
    ConstBufferPtr padding_offset;

    size_t context_batch_size      = 0;
    size_t decoder_batch_size      = 0;
    size_t context_max_seq_len     = 0;
    size_t decoder_max_seq_len     = 0;
    size_t context_token_num       = 0;
    size_t context_total_kv_length = 0;

    BufferPtr      position_ids;
    BufferPtr      attention_mask;
    ConstBufferPtr linear_bias_slopes;
    BufferPtr      prefix_prompt_lengths;
    int32_t        max_prefix_length = 0;

    lora::AttentionLayerLoraInput lora_input;

    ParamsPtr prefill_flash_infer_attn;
    ParamsPtr decode_flash_infer_attn;
    ParamsPtr prefill_trt_attn;
    ParamsPtr decode_trt_attn;

    // rocm
    ParamsPtr decode_aiter_attn;
};

using AttentionModuleOutput = void;

struct AttentionModuleParams {
    const int32_t layer_id;
    // qkv shape[h_token_num, (head_num + 2 * kv_head_num) * size_per_head]
    const Buffer& input;
    Buffer&       output;  // shape [token_num, size_per_head]

    AttentionCommonInputs&       common;
    const AttentionLayerWeights& weights;
    const AttentionConfigs&      configs;
    const QScheme                qscheme;
    const DataType               compute_type = DataType::TYPE_INVALID;
};

struct MlaRotaryWriteKVCacheParams {
    const Buffer& q;
    BufferPtr     fused_dest_q;
    const Buffer& fused_qkv;
    const int64_t kv_offset;
    ParamsPtr     flash_infer;  // prefill or decode

    AttentionCommonInputs&       common;
    const AttentionLayerWeights& weights;
    const AttentionConfigs&      configs;
    const QScheme                qscheme;
    bool                         is_decode = false;
};

struct MlaAttentionModuleParams {
    const int32_t layer_id;
    const Buffer& q;
    const Buffer& fused_qkv;
    const int64_t kv_offset;
    BufferPtr     qkv_output;  // shape [token_num, hidden_size]

    AttentionCommonInputs&       common;
    const AttentionLayerWeights& weights;
    const AttentionConfigs&      configs;
    const QScheme                qscheme;
    const DataType               compute_type = DataType::TYPE_INVALID;
    bool                         is_prefill   = false;
};

enum class SendRecvType {
    kSend,
    kRecv,
};

struct P2pSendRecvParams {
    SendRecvType type;
    BufferPtr    buffer;
    int          dest_rank;
};

struct BatchSendRecvParams {
    std::vector<P2pSendRecvParams> p2p_params;
    ParallelMode                   mode = ParallelMode::DP_AND_TP;
};

struct WriteCacheParams {

    std::optional<CacheStoreInputs>& cache_store_inputs;
    std::optional<KvCacheInfo>&      kv_cache;
    bool                             mla_kvcache = false;

    WriteCacheParams(const AttentionModuleParams& params):
        cache_store_inputs(params.common.cache_store_inputs), kv_cache(params.common.kv_cache) {}

    WriteCacheParams(const MlaAttentionModuleParams& params):
        cache_store_inputs(params.common.cache_store_inputs), kv_cache(params.common.kv_cache), mla_kvcache(true) {}
};

using WriteCacheParamsPtr = std::shared_ptr<WriteCacheParams>;

struct AttentionLayerOutput {
    BufferPtr hidden_states;
};

struct LayerNormConfig {
    double   eps;
    NormType norm_type;
};

struct AttentionLayerParams {
    int32_t                      layer_id;
    const Buffer&                input;
    BufferPtr                    output;
    const AttentionConfigs&      configs;
    const AttentionLayerWeights& weights;
    AttentionCommonInputs&       common;
    const OptionalConstBufferRef residual;  // for intel xft
    const LayerNormConfig        ln_params;
    const QScheme                qscheme;
    const DataType               compute_type;
    bool                         enable_sp;
    size_t                       pad_token_num;
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

struct DeepEPDispatchOutput;
struct DeepEPDispatchOutputLowLatency;
struct MoeGateSelectOutput;

struct MoeCombineParams {
    BufferPtr                                       input;
    BufferPtr                                       indices;
    BufferPtr                                       output;
    std::vector<size_t>                             input_split_sizes;
    std::vector<size_t>                             output_split_sizes;
    MoeConfigs                                      moe_configs;
    size_t                                          origin_token_num;
    bool                                            overlapped = false;
    std::shared_ptr<DeepEPDispatchOutput>           deep_ep_output;
    std::shared_ptr<DeepEPDispatchOutputLowLatency> deep_ep_ll_output;
    std::shared_ptr<MoeGateSelectOutput>            select_output;
    BufferPtr                                       expert_ids;
    BufferPtr                                       expert_scales;
    DeviceEventPtr                                  compute_stream_event = nullptr;
};

struct MoeCombineOutput {
    BufferPtr        all_output;
    BufferPtr        scatter_output;
    MoeCombineParams params;
    DeviceHookPtr    comm_barrier_hook;
};

struct FfnLayerOutput {
    BufferPtr                       hidden_states;
    BufferPtr                       moe_gating;
    DeviceHookPtr                   comm_barrier_hook;
    std::optional<MoeCombineOutput> moe_combine_output;
};

struct FfnLayerParams {
    FfnLayerParams(const Buffer&                input,
                   const FfnConfigs&            configs,
                   const FfnLayerWeights&       weights,
                   const OptionalConstBufferRef residual        = std::nullopt,
                   const QScheme                qscheme         = QScheme::NoQuantize,
                   const DataType               compute_type    = DataType::TYPE_INVALID,
                   BufferPtr                    output          = nullptr,
                   bool                         enable_sp       = false,
                   bool                         need_moe_gating = false):
        input(input),
        configs(configs),
        weights(weights),
        residual(residual),
        qscheme(qscheme),
        compute_type(compute_type),
        output(std::move(output)),
        enable_sp(enable_sp) {}

    const Buffer&          input;
    const FfnConfigs&      configs;
    const FfnLayerWeights& weights;

    const OptionalConstBufferRef residual;  // for intel xft

    OptionalExpertStats expert_stats = std::nullopt;

    const QScheme  qscheme;
    const DataType compute_type;
    BufferPtr      output;

    lora::FfnLayerLoraInput lora_input;
    bool                    enable_sp;
    bool                    need_moe_gating;
};

struct MoeDispatchOutput {
    BufferPtr                    hidden;
    BufferPtr                    expert_ids;
    BufferPtr                    expert_scales;
    BufferPtr                    indices;
    const std::vector<size_t>    input_split_sizes;
    const std::vector<size_t>    output_split_sizes;
    const std::vector<BufferPtr> dispatch_src_buffers;  // to make them outlive async sendrecv
    const BufferPtr              concated_src_buffers;  // to make them outlive async sendrecv
    const BufferPtr              split_dst_buffers;     // to make them outlive async sendrecv
    DeviceHookPtr                comm_barrier_hook;

    std::shared_ptr<DeepEPDispatchOutput>           deep_ep_output;
    std::shared_ptr<DeepEPDispatchOutputLowLatency> deep_ep_ll_output;
};

struct MoeGateSelectOutput {
    BufferPtr                                       expert_ids;
    BufferPtr                                       expert_scales;
    BufferPtr                                       moe_gating        = nullptr;
    std::shared_ptr<DeepEPDispatchOutputLowLatency> deep_ep_ll_output = nullptr;
};

struct MoeDispatchParams {
    const Buffer&       input;
    const Buffer&       expert_ids;
    const Buffer&       expert_scales;
    const MoeConfigs&   moe_configs;
    bool                overlapped = false;
    const QScheme       qscheme;
    OptionalExpertStats expert_stats         = std::nullopt;
    DeviceEventPtr      compute_stream_event = nullptr;
};

struct MoeEpPlanParams {
    BufferPtr             expert_ids;
    BufferPtr             expert_scales;
    const FfnLayerParams& params;
    bool                  overlapped = false;
};

struct MoeEpPlanOutput {
    BufferPtr           all_token_indices;
    BufferPtr           balanced_expert_ids;
    std::vector<size_t> input_split_sizes;
    std::vector<size_t> output_split_sizes;
};

struct MoeBalanceOutput {
    BufferPtr balance_expert_ids;
};

struct MoeBalanceParams {
    const BufferPtr       experts_ids_host;
    const FfnLayerParams& params;
};

// for deepseek decode micro batch
struct MoEInsertionParams {
    MoEInsertionParams(const MoeDispatchOutput&             dispatched_output,
                       const FfnLayerParams&                ffn_params,
                       std::shared_ptr<MoeGateSelectOutput> gate_output,
                       size_t                               origin_token_num):
        dispatched_output(dispatched_output),
        ffn_params(ffn_params),
        gate_output(std::move(gate_output)),
        origin_token_num(origin_token_num) {}

    MoeDispatchOutput                    dispatched_output;
    FfnLayerParams                       ffn_params;
    std::shared_ptr<MoeGateSelectOutput> gate_output;
    size_t                               origin_token_num;
};

struct MoEInsertionReturns {
    MoeCombineOutput combine_output;
};

struct GreedyParams {
    const Buffer& logits;            // [batch_size, vocab_size_padded]
    const Buffer& input_lengths;     // [batch_size]
    const Buffer& sequence_lengths;  // [batch_size]
    Buffer&       token_ids;         // [batch_size, max_input_length + 1]
    const size_t  step;

    const Buffer& top_k;
    const Buffer& top_p;
    const Buffer& temperature;

    OptionalBufferRef repetition_penalty;
    OptionalBufferRef no_repeat_ngram_size;

    OptionalBufferRef cum_log_probs;
    OptionalBufferRef output_log_probs;

    OptionalBufferRef output_all_probs;
    OptionalBufferRef presence_penalty;
    OptionalBufferRef frequency_penalty;
    OptionalBufferRef do_sample;

    std::vector<at::Generator> generator;
};

struct GreedyOutput {
    BufferPtr success;
    // BufferPtr new_tokens;
};

struct BeamSearchParams {
    const Buffer& logits;            // [batch_size, num_beams_in, vocab_size]
    BufferPtr     token_ids;         // [batch_size, num_beams_in, max_seq_len]
    BufferPtr     input_lengths;     // [batch_size, num_beams_in]
    BufferPtr     sequence_lengths;  // [batch_size, num_beams_in]
    BufferPtr     cum_log_probs;     // [batch_size, num_beams_in]
    size_t        num_beams_out = 0;
};

struct BeamSearchOutput {
    BufferPtr token_ids;         // [batch_size, num_beams_out, max_seq_len]
    BufferPtr input_lengths;     // [batch_size, num_beams_out]
    BufferPtr sequence_lengths;  // [batch_size, num_beams_out]
    BufferPtr cum_log_probs;     // [batch_size, num_beams_out]
    BufferPtr beam_indices;      // [batch_size, num_beams_out]
};

struct BroadcastParams {
    const std::vector<BufferPtr>& buffers;
    const int64_t                 root;
    ParallelMode                  mode       = ParallelMode::TP;
    bool                          overlapped = false;
};

enum class ReduceOp {
    Sum  = 0,
    Prod = 1,
    Max  = 2,
    Min  = 3,
    Avg  = 4,
};

struct PrepareAllReduceParams {
    const BufferPtr buffer;
    const ReduceOp  op;
    ParallelMode    mode = ParallelMode::TP;
};

struct PrepareAllReduceOutput {
    const BufferPtr buffer;
};

struct AllReduceParams {
    const BufferPtr buffer;
    const ReduceOp  op;
    bool            overlapped = false;
    ParallelMode    mode       = ParallelMode::TP;
    const BufferPtr dest       = nullptr;
};

struct AllReduceOutput {
    const BufferPtr buffer;
};

struct AllGatherParams {
    const std::vector<BufferPtr>& recv_buffers;
    ParallelMode                  mode = ParallelMode::TP;
    std::vector<BufferPtr>        send_buffers;
    bool                          inplace    = true;
    bool                          overlapped = false;
};

struct ReduceScatterParams {
    const BufferPtr send_buffer;
    const BufferPtr recv_buffer;
    const ReduceOp  op;
    ParallelMode    mode       = ParallelMode::TP;
    bool            overlapped = false;
};

struct AllToAllParams {
    const std::vector<BufferPtr> buffers;
    const std::vector<size_t>    input_split_sizes;
    const std::vector<size_t>    output_split_sizes;
    bool                         overlapped           = false;
    ParallelMode                 mode                 = ParallelMode::DP_AND_TP;
    DeviceEventPtr               compute_stream_event = nullptr;
};

struct AllToAllOutput {
    std::vector<BufferPtr> outputs;
    BufferPtr              concated_input;
    BufferPtr              output_to_split;
    DeviceHookPtr          comm_barrier_hook;
};

// output = act(input) + bias
struct ActivationParams {
    ActivationType atype;
    // can be nullptr for fuse gemm with activation
    BufferPtr states;

    const OptionalConstBufferRef bias          = std::nullopt;
    const OptionalConstBufferRef gate          = std::nullopt;
    const OptionalConstBufferRef gate_bias     = std::nullopt;
    const OptionalConstBufferRef act_scale     = std::nullopt;
    BufferPtr                    output_buffer = nullptr;
    bool                         fuse_gate_up  = false;
    QScheme                      qscheme       = QScheme::NoQuantize;

    ActivationParams(ActivationType               atype,
                     BufferPtr                    states,
                     const OptionalConstBufferRef bias,
                     const OptionalConstBufferRef gate,
                     const OptionalConstBufferRef gate_bias,
                     const OptionalConstBufferRef act_scale,
                     BufferPtr                    output_buffer = nullptr,
                     bool                         fuse_gate_up  = false,
                     QScheme                      qscheme       = QScheme::NoQuantize):
        atype(atype),
        states(states),
        bias(bias),
        gate(gate),
        gate_bias(gate_bias),
        act_scale(act_scale),
        output_buffer(output_buffer),
        fuse_gate_up(fuse_gate_up),
        qscheme(qscheme) {}

    ActivationParams(ActivationType atype, BufferPtr states):
        atype(atype),
        states(states),
        bias(std::nullopt),
        gate(std::nullopt),
        gate_bias(std::nullopt),
        act_scale(std::nullopt) {};
};

// softmax op is inplace-update, thus output buffer is same as input
struct SoftmaxParams {
    const BufferPtr              input;
    const OptionalConstBufferRef mask               = std::nullopt;
    const OptionalConstBufferRef bias               = std::nullopt;
    float                        scale              = 1.0f;
    const DataType               output_t           = DataType::TYPE_INVALID;
    const OptionalConstBufferRef linear_bias_slopes = std::nullopt;
};

struct LossParams {
    const Buffer& logits;
    const Buffer& labels;
};

using LossOutput = BufferPtr;

struct MaskParams {
public:
    const Buffer& input_lengths;
    const Buffer& prefix_lengths;
    DataType      dtype;
    bool          is_causal;
};

using MaskOutput = BufferPtr;

struct DevicePrepParams {
    const AttentionConfigs& configs;

    const BufferPtr& prefix_lengths;
    const BufferPtr& sequence_lengths;
    const BufferPtr& input_lengths;
    const BufferPtr& kv_cache_block_id;
    const BufferPtr& kv_cache_block_id_d;
    const BufferPtr& kv_cache;

    DataType attn_dtype         = DataType::TYPE_INVALID;
    size_t   context_batch_size = 0;
    size_t   decoder_batch_size = 0;
    bool     diff_qkv_len       = false;
    bool     has_alibi_slopes   = false;
};

struct DevicePrepOutput {
    bool      need_mask = true;
    ParamsPtr decode_flash_infer_attn;
    ParamsPtr prefill_flash_infer_attn;
    ParamsPtr decode_trt_attn;
    ParamsPtr prefill_trt_attn;

    // rocm
    ParamsPtr decode_aiter_attn;
};

struct LoraLinearOutput {
    BufferPtr output;
};

struct AllGatherLoraLinearOutput {
    BufferPtr output;
    BufferPtr all_gather_recv_buffer;
};

struct ReduceScatterLoraLinearOutput {
    BufferPtr output;
    BufferPtr reduce_scatter_recv_buffer;
};

struct LoraLinearParams {

    LoraLinearParams(GemmParams& gemm_params, lora::LoraOpInputPtr lora_input = nullptr):
        gemm_params(gemm_params), lora_input(lora_input) {}

    GemmParams&          gemm_params;
    lora::LoraOpInputPtr lora_input;
};

struct LoraLinearReduceScatterParams {
    const LoraLinearParams& lora_linear_params;
    const BufferPtr&        rs_recv_buffer;
    QScheme                 qscheme;
    DataType                output_type;
    ParallelMode            mode = ParallelMode::TP;
    LoraLinearReduceScatterParams(const LoraLinearParams& lora_linear_params,
                                  const BufferPtr&        rs_recv_buffer,
                                  QScheme                 qscheme,
                                  DataType                output_type,
                                  ParallelMode            mode = ParallelMode::TP):
        lora_linear_params(lora_linear_params),
        rs_recv_buffer(rs_recv_buffer),
        qscheme(qscheme),
        output_type(output_type),
        mode(mode) {}
};

struct AllGatherLoraLinearParams {
    const LoraLinearParams& lora_linear_params;
    const BufferPtr&        ag_send_buffer;
    BufferPtr               ag_recv_buffer;
    QScheme                 qscheme;
    DataType                output_type;
    ParallelMode            mode = ParallelMode::TP;
    AllGatherLoraLinearParams(const LoraLinearParams& lora_linear_params,
                              const BufferPtr&        ag_send_buffer,
                              BufferPtr               ag_recv_buffer,
                              QScheme                 qscheme,
                              DataType                output_type,
                              ParallelMode            mode = ParallelMode::TP):
        lora_linear_params(lora_linear_params),
        ag_send_buffer(ag_send_buffer),
        ag_recv_buffer(ag_recv_buffer),
        qscheme(qscheme),
        output_type(output_type),
        mode(mode) {}
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

struct PrepareCommBufferParams {
    const size_t max_batch_seq_len;
    const size_t attn_rs_hidden;
    const size_t ffn_rs_hidden;
    const size_t attn_ag_hidden;
    const size_t ffn_ag_hidden;
    DataType     rs_output_type;
    DataType     ag_input_type;
    bool         enable_per_token_scale = false;
    bool         enable_ffn_tp          = false;
    PrepareCommBufferParams(size_t   max_batch_seq_len,
                            size_t   attn_rs_hidden,
                            size_t   ffn_rs_hidden,
                            size_t   attn_ag_hidden,
                            size_t   ffn_ag_hidden,
                            DataType rs_output_type,
                            DataType ag_input_type,
                            bool     enable_per_token_scale = false,
                            bool     enable_ffn_tp          = false):
        max_batch_seq_len(max_batch_seq_len),
        attn_rs_hidden(attn_rs_hidden),
        ffn_rs_hidden(ffn_rs_hidden),
        attn_ag_hidden(attn_ag_hidden),
        ffn_ag_hidden(ffn_ag_hidden),
        rs_output_type(rs_output_type),
        ag_input_type(ag_input_type),
        enable_per_token_scale(enable_per_token_scale),
        enable_ffn_tp(enable_ffn_tp) {}
};

struct LoraLinearWithActivationParams {
    const LoraLinearParams& lora_linear_params;
    const ActivationParams& activation_params;
    LoraLinearWithActivationParams(const LoraLinearParams& lora_linear_params,
                                   const ActivationParams& activation_params):
        lora_linear_params(lora_linear_params), activation_params(activation_params) {}
};
struct QuantizeParams {
    const Buffer& input;
    DataType      qtype;
    size_t        axis;
    QScheme       qscheme;

    // for soomth quantize
    OptionalConstBufferRef smoother;
    OptionalConstBufferRef shift;

    // for static quantize
    OptionalConstBufferRef static_scale;
    OptionalConstBufferRef static_scale_reciprocal;

    // for groupwise quantize
    int64_t groupSize;
    int64_t paddingSize = 0;

    QuantizeParams(const Buffer&          input,
                   DataType               qtype,
                   size_t                 axis,
                   QScheme                qscheme,
                   OptionalConstBufferRef smoother,
                   OptionalConstBufferRef shift,
                   OptionalConstBufferRef static_scale,
                   OptionalConstBufferRef static_scale_reciprocal):
        input(input),
        qtype(qtype),
        axis(axis),
        qscheme(qscheme),
        smoother(smoother),
        shift(shift),
        static_scale(static_scale),
        static_scale_reciprocal(static_scale_reciprocal),
        groupSize(64) {}
    QuantizeParams(const Buffer& input, DataType qtype, size_t axis):
        input(input), qtype(qtype), axis(axis), qscheme(QScheme::Qint8PerToken), groupSize(64) {}
    QuantizeParams(const Buffer& input, DataType qtype, size_t axis, int64_t groupSize):
        input(input), qtype(qtype), axis(axis), qscheme(QScheme::Qint8PerToken), groupSize(groupSize) {}
    QuantizeParams(const Buffer& input, DataType qtype, size_t axis, QScheme qscheme, int64_t paddingSize = 0):
        input(input), qtype(qtype), axis(axis), qscheme(qscheme), paddingSize(paddingSize) {}
    QuantizeParams(
        const Buffer& input, DataType qtype, size_t axis, QScheme qscheme, int64_t groupSize, int64_t paddingSize):
        input(input), qtype(qtype), axis(axis), qscheme(qscheme), groupSize(groupSize), paddingSize(paddingSize) {}
};

}  // namespace rtp_llm
