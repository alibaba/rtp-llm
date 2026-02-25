#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <memory>
#include "rtp_llm/cpp/devices/GraphBase.h"
#include "rtp_llm/cpp/devices/graph_common/GraphBaseRunner.h"

namespace py = pybind11;
namespace rtp_llm {
class CudaGraphRunner: public GraphBase {
public:
    CudaGraphRunner(const DeviceInitParams& params,
                    py::object              py_instance,
                    c10::ScalarType         model_data_type,
                    int                     num_tokens_per_bs,
                    bool                    is_prefill_cuda_graph_mode = false);

    ~CudaGraphRunner() override;

    void           captureDecode();
    void           capturePrefill();
    void           captureDecodeOneBatchSize(int bs);
    void           capturePrefillOneSeqLen(int seq_len);
    void           prepareInputs(PyModelInputs& inputs);
    bool           canRun(PyModelInputs& inputs) override;
    void           replayGraph(int key);
    void           replayDecode(int bs);
    void           replayPrefill(int seq_len);
    void           setMaxPrefillCudaGraphLen(int max_prefill_cuda_graph_len);
    py::object     normalForward(PyModelInputs& inputs);
    int            getCurrentRealGraphBs();
    PyModelOutputs forward(PyModelInputs& inputs) override;
    void           initCapture() override;
    void           setPositionEncoding(torch::Tensor position_encoding) override;
    void           setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void           setInputEmbeddingScalar(float input_embedding_scalar) override;

private:
    std::unique_ptr<GraphBaseRunner> runner_;
};
}  // namespace rtp_llm
