#pragma once
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace rtp_llm {

using namespace torch_ext;

// Current state of CUDA graph execution (used when calling canRun/forward with graph runner)
struct CudaGraphState {
    int current_batch_size{1};
    int current_seq_len{1};
    int current_real_graph_bs{1};       // for decode
    int current_real_graph_seq_len{1};  // for prefill
    int seq_len_sum{0};
};

struct CacheBlockTableCapacity {
    int64_t physical_block_table_capacity = 0;
    int64_t kernel_block_table_capacity   = 0;

    static CacheBlockTableCapacity fromBlockSizes(int64_t          max_seq_len,
                                                  int64_t          physical_tokens_per_block,
                                                  int64_t          kernel_tokens_per_block,
                                                  int64_t          sp_steps,
                                                  std::string_view context = {}) {
        const std::string context_text(context);
        RTP_LLM_CHECK_WITH_INFO(max_seq_len >= 0,
                                "CUDA graph cache capacity context=%s max_seq_len must be non-negative, got %ld",
                                context_text.c_str(),
                                max_seq_len);
        RTP_LLM_CHECK_WITH_INFO(sp_steps >= 0,
                                "CUDA graph cache capacity context=%s sp_steps must be non-negative, got %ld",
                                context_text.c_str(),
                                sp_steps);
        RTP_LLM_CHECK_WITH_INFO(
            physical_tokens_per_block > 0,
            "CUDA graph cache capacity context=%s physical tokens per block must be positive, got %ld",
            context_text.c_str(),
            physical_tokens_per_block);
        RTP_LLM_CHECK_WITH_INFO(
            kernel_tokens_per_block > 0,
            "CUDA graph cache capacity context=%s kernel tokens per block must be positive, got %ld",
            context_text.c_str(),
            kernel_tokens_per_block);
        RTP_LLM_CHECK_WITH_INFO(
            physical_tokens_per_block % kernel_tokens_per_block == 0,
            "CUDA graph cache capacity context=%s physical tokens per block=%ld must be divisible by kernel tokens "
            "per block=%ld",
            context_text.c_str(),
            physical_tokens_per_block,
            kernel_tokens_per_block);

        const int64_t sequence_blocks = max_seq_len / physical_tokens_per_block
                                        + static_cast<int64_t>(max_seq_len % physical_tokens_per_block != 0);
        RTP_LLM_CHECK_WITH_INFO(sequence_blocks <= std::numeric_limits<int64_t>::max() - sp_steps,
                                "CUDA graph cache capacity context=%s physical capacity overflow",
                                context_text.c_str());
        const int64_t physical_capacity          = sequence_blocks + sp_steps;
        const int64_t kernel_blocks_per_physical = physical_tokens_per_block / kernel_tokens_per_block;
        RTP_LLM_CHECK_WITH_INFO(physical_capacity == 0
                                    || kernel_blocks_per_physical
                                           <= std::numeric_limits<int64_t>::max() / physical_capacity,
                                "CUDA graph cache capacity context=%s kernel capacity overflow",
                                context_text.c_str());
        return {physical_capacity, physical_capacity * kernel_blocks_per_physical};
    }
};

struct GraphParams {
    bool             enable_cuda_graph            = false;
    bool             enable_cuda_graph_debug_mode = false;
    bool             is_prefill_cuda_graph_mode   = false;
    bool             is_target_verify             = false;
    int              max_seq_len                  = 0;
    int              tokens_per_block             = 0;  // physical kv block size
    int              kernel_tokens_per_block      = 0;  // must be explicitly configured
    int              num_tokens_per_bs      = 1;  // Number of tokens per batch (1 for decode, max_seq_len for prefill)
    int              sp_steps               = 0;
    size_t           max_context_batch_size = 128;
    std::size_t      hidden_size            = 0;
    c10::ScalarType  model_data_type        = c10::ScalarType::Float;
    std::vector<int> prefill_capture_seq_lens;
    std::vector<int> decode_capture_batch_sizes;
    int64_t          hc_mult = 1;
    // Per-group block-table capacities used to allocate fixed capture buffers.
    std::map<std::string, CacheBlockTableCapacity> kv_cache_block_table_capacities;
    // Per-token position-id factor for combo_position_ids capture buffer.
    // 0 = model does not use combo_position_ids (no buffer allocated, capture skips it).
    // >0 = factor (e.g. Mrope = rope_config.index_factor). Sourced from
    //     description_.attention_conf.rope_config in the model wrapper, not Python reflection.
    int position_id_len_factor = 0;
};

class GraphBase {
public:
    GraphBase(py::object py_instance): py_instance_(std::move(py_instance)) {}
    virtual ~GraphBase() {}
    virtual void           initCapture()                                                = 0;
    virtual PyModelOutputs forward(const PyModelInputs& inputs, CudaGraphState& state)  = 0;
    virtual void           setPositionEncoding(torch::Tensor position_encoding)         = 0;
    virtual void           setTokenTypeEmbedding(torch::Tensor token_type_embedding)    = 0;
    virtual void           setInputEmbeddingScalar(float input_embedding_scalar)        = 0;
    virtual bool           canRun(const PyModelInputs& inputs, CudaGraphState& state)   = 0;
    virtual void           prepareAttentionInputs(const PyModelInputs& inputs,
                                                  CudaGraphState&      state,
                                                  bool                 skip_forward_event_sync = false) = 0;

    // Refresh only captured kv_cache_kernel_block_id state and FlashInfer plan
    // buffers after page-table changes. Other captured fields stay untouched.
    virtual void updateKVCacheKernelBlockId(const PyModelInputs& inputs, CudaGraphState& state) {}

    py::object py_instance_;
};
}  // namespace rtp_llm
