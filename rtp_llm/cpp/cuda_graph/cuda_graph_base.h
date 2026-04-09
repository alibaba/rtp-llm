#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

using namespace torch_ext;

// Resolved batch / graph-key dimensions for one forward (decode BS, prefill seq len, etc.).
struct BatchDescriptor {
    int current_batch_size{1};
    int current_seq_len{1};
    int current_real_graph_bs{1};       // for decode
    int current_real_graph_seq_len{1};  // for prefill
    int seq_len_sum{0};
};

struct GraphParams {
    bool                 enable_cuda_graph            = false;
    bool                 enable_cuda_graph_debug_mode = false;
    bool                 is_prefill_cuda_graph_mode   = false;
    bool                 is_target_verify             = false;
    int                  max_seq_len                  = 0;
    int                  tokens_per_block             = 0;  // physical kv block size
    int                  kernel_tokens_per_block      = 0;  // must be explicitly configured
    int                  num_tokens_per_bs = 1;  // Number of tokens per batch (1 for decode, max_seq_len for prefill)
    int                  sp_steps          = 0;
    size_t               max_context_batch_size = 1;    // for prefill mode
    size_t               concurrency_limit      = 128;  // for decode mode
    std::size_t          hidden_size            = 0;
    c10::ScalarType      model_data_type        = c10::ScalarType::Float;
    std::vector<int>     prefill_capture_seq_lens;
    std::vector<int>     decode_capture_batch_sizes;
    std::vector<int32_t> kv_cache_layer_to_group;  // layer index -> group id for hybrid kv cache
    int32_t              kv_cache_group_num = 0;   // number of kv cache groups
};

}  // namespace rtp_llm
