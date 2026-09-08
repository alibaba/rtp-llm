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
    int lazy_capture_key{-1};
};

// Decision returned by GraphBase::plan for the current request.
// - Replay:            a captured graph for the selected bucket is ready; replay it.
// - CaptureAfterEager: the selected bucket is not captured yet; run eager this time,
//                      then capture the bucket so future requests can replay.
// - Eager:             no bucket can serve this request (out of range / disabled / failed);
//                      run eager and do not attempt capture.
enum class GraphRunDecision {
    Replay,
    CaptureAfterEager,
    Eager,
};

enum class GraphMode {
    Decode,
    GenerationPrefill,
    EmbeddingPrefill,
    MtpPrefill,
};

struct GraphParams {
    bool      enable_cuda_graph            = false;
    bool      enable_cuda_graph_debug_mode = false;
    bool      is_prefill_cuda_graph_mode   = false;
    bool      is_target_verify             = false;
    GraphMode graph_mode                   = GraphMode::Decode;
    // When true, initCapture only initializes shared storage/state and captures each bucket
    // lazily on first demand (first request eager, later requests replay). When false (default),
    // all configured buckets are captured eagerly during initialization (embedding / MTP / tests).
    bool                 lazy_capture            = false;
    int                  max_seq_len             = 0;
    int                  tokens_per_block        = 0;  // physical kv block size
    int                  kernel_tokens_per_block = 0;  // must be explicitly configured
    int                  num_tokens_per_bs = 1;  // Number of tokens per batch (1 for decode, max_seq_len for prefill)
    int                  sp_steps          = 0;
    int                  mori_max_tokens   = 0;
    size_t               max_context_batch_size = 128;
    std::size_t          hidden_size            = 0;
    c10::ScalarType      model_data_type        = c10::ScalarType::Float;
    std::vector<int>     prefill_capture_seq_lens;
    std::vector<int>     decode_capture_batch_sizes;
    std::vector<int32_t> kv_cache_layer_to_group;  // layer index -> group id for hybrid kv cache
    int32_t              kv_cache_group_num = 0;   // number of kv cache groups
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
    // Lazy-capture entry points (used when GraphParams::lazy_capture is true):
    // plan() selects a bucket for the request and returns the run decision;
    // captureCurrentBucket() captures the bucket selected by the most recent plan() call,
    // and must be invoked only after the eager forward for that request has fully completed.
    virtual GraphRunDecision plan(const PyModelInputs& inputs, CudaGraphState& state) = 0;
    virtual bool             captureCurrentBucket(const CudaGraphState& state)        = 0;
    py::object               py_instance_;
};
}  // namespace rtp_llm
