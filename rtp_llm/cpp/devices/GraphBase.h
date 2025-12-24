#pragma once
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <cstddef>
#include <vector>

namespace rtp_llm {

using namespace torch_ext;

struct GraphParams {
    bool             enable_cuda_graph            = false;
    bool             enable_cuda_graph_debug_mode = false;
    bool             is_prefill_cuda_graph_mode   = false;
    int              max_seq_len                  = 0;
    int              tokens_per_block             = 0;
    size_t           max_context_batch_size       = 1;    // for prefill mode
    size_t           concurrency_limit            = 128;  // for decode mode
    std::size_t      hidden_size                  = 0;
    std::vector<int> prefill_capture_seq_lens;
    std::vector<int> decode_capture_batch_sizes;
};

class GraphBase {
public:
    GraphBase(py::object py_instance): py_instance_(std::move(py_instance)) {}
    virtual ~GraphBase() {}
    virtual void           initCapture()                                             = 0;
    virtual PyModelOutputs forward(PyModelInputs& inputs)            = 0;
    virtual void           setPositionEncoding(torch::Tensor position_encoding)      = 0;
    virtual void           setTokenTypeEmbedding(torch::Tensor token_type_embedding) = 0;
    virtual void           setInputEmbeddingScalar(float input_embedding_scalar)     = 0;
    virtual bool           canRun(PyModelInputs& inputs)                             = 0;
    py::object             py_instance_;
};
}  // namespace rtp_llm
