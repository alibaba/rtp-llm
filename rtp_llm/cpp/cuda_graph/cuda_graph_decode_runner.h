#pragma once

#include "rtp_llm/cpp/cuda_graph/cuda_graph_decode_capture_dispatcher.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner_base.h"

namespace rtp_llm {

class CudaGraphDecodeRunner: public CudaGraphRunnerBase {
public:
    explicit CudaGraphDecodeRunner(GraphParams graph_params, py::object py_instance);

    void           captureDecode();
    void           captureDecodeOneBatchSize(int bs);
    void           replayDecode(int bs);
    void           prepareInputs(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor);
    bool           canRun(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    PyModelOutputs forward(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) override;
    void           initCapture() override;
    int            getCurrentRealGraphSize(const BatchDescriptor& batch_descriptor) const override;

protected:
    void buildCaptureDispatcher() override;

    cuda_graph::CudaGraphDecodeCaptureDispatcher decode_capture_dispatcher_;
};

}  // namespace rtp_llm
