#pragma once

#include "rtp_llm/cpp/cuda_graph/cuda_graph_prefill_capture_dispatcher.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner_base.h"

namespace rtp_llm {

class CudaGraphPrefillRunner: public CudaGraphRunnerBase {
public:
    explicit CudaGraphPrefillRunner(GraphParams graph_params, py::object py_instance);

    void           capturePrefill();
    void           capturePrefillOneSeqLen(int seq_len);
    void           replayPrefill(int seq_len);
    void           prepareInputs(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor);
    bool           canRun(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    PyModelOutputs forward(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    void           initCapture() override;
    int            getCurrentRealGraphSize(const BatchDescriptor& batch_descriptor) const override;

protected:
    void buildCaptureDispatcher() override;

    cuda_graph::CudaGraphPrefillCaptureDispatcher prefill_capture_dispatcher_;
};

}  // namespace rtp_llm
