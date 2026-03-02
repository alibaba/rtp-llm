#pragma once
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <cstddef>
#include <cstdint>
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

struct GraphParams {
    bool                 enable_cuda_graph            = false;
    bool                 enable_cuda_graph_debug_mode = false;
    bool                 is_prefill_cuda_graph_mode   = false;
    int                  max_seq_len                  = 0;
    int                  tokens_per_block             = 0;
    int                  num_tokens_per_bs = 1;  // Number of tokens per batch (1 for decode, max_seq_len for prefill)
    size_t               max_context_batch_size = 1;    // for prefill mode
    size_t               concurrency_limit      = 128;  // for decode mode
    std::size_t          hidden_size            = 0;
    c10::ScalarType      model_data_type        = c10::ScalarType::Float;
    std::vector<int>     prefill_capture_seq_lens;
    std::vector<int>     decode_capture_batch_sizes;
    int                  max_prefill_cuda_graph_len = 0;  // for prefill mode only
    std::vector<int32_t> kv_cache_layer_to_group;         // layer index -> group id for hybrid kv cache
    int32_t              kv_cache_group_num = 0;          // number of kv cache groups
};

class GraphBase {
public:
    GraphBase(py::object py_instance): py_instance_(std::move(py_instance)) {}
    virtual ~GraphBase() {}
    virtual void           initCapture()                                               = 0;
    virtual PyModelOutputs forward(const PyModelInputs& inputs, CudaGraphState& state) = 0;
    virtual void           setPositionEncoding(torch::Tensor position_encoding)        = 0;
    virtual void           setTokenTypeEmbedding(torch::Tensor token_type_embedding)   = 0;
    virtual void           setInputEmbeddingScalar(float input_embedding_scalar)       = 0;
    virtual bool           canRun(const PyModelInputs& inputs, CudaGraphState& state)  = 0;
    py::object             py_instance_;
};
}  // namespace rtp_llm
