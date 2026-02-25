#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <memory>
#include "rtp_llm/cpp/devices/GraphBase.h"
#include "rtp_llm/cpp/devices/graph_common/GraphBaseRunner.h"
namespace py = pybind11;
namespace rtp_llm {

struct HipGraphNcclCaptureContext {
    int64_t comm_handle{0};
    int     rank{0};
    int     world_size{1};
};

class HipGraphRunner: public GraphBase {
public:
    HipGraphRunner(const DeviceInitParams& params,
                   py::object              py_instance,
                   c10::ScalarType         model_data_type,
                   int                     num_tokens_per_bs,
                   bool                    is_prefill_hip_graph_mode = false);

    ~HipGraphRunner() override;
    void           captureDecode();
    void           capturePrefill();
    void           captureDecodeOneBatchSize(int bs);
    void           capturePrefillOneSeqLen(int seq_len);
    void           prepareInputs(PyModelInputs& inputs);
    bool           canRun(PyModelInputs& inputs) override;
    void           replayGraph(int key);
    void           replayDecode(int bs);
    void           replayPrefill(int seq_len);
    void           setMaxPrefillHipGraphLen(int max_prefill_hip_graph_len);
    py::object     normalForward(PyModelInputs& inputs);
    int            getCurrentRealGraphBs();
    PyModelOutputs forward(PyModelInputs& inputs) override;
    void           initCapture() override;

    // Set the existing C++ NCCL communicator handle for use during HIP Graph capture.
    // This avoids creating a new communicator - we reuse the one from ROCmDevice.
    void setNcclCommHandle(void* nccl_comm, size_t rank, size_t world_size) {
        nccl_capture_ctx_->comm_handle = reinterpret_cast<int64_t>(nccl_comm);
        nccl_capture_ctx_->rank        = static_cast<int>(rank);
        nccl_capture_ctx_->world_size  = static_cast<int>(world_size);
    }

    void setPositionEncoding(torch::Tensor position_encoding) override;
    void setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void setInputEmbeddingScalar(float input_embedding_scalar) override;

private:
    std::shared_ptr<HipGraphNcclCaptureContext> nccl_capture_ctx_;
    std::unique_ptr<GraphBaseRunner>            runner_;
};
}  // namespace rtp_llm
