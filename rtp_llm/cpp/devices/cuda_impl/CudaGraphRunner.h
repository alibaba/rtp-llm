#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphUtils.h"
#include "rtp_llm/cpp/devices/GraphBase.h"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>

namespace py = pybind11;
namespace rtp_llm {

class CudaGraphRunner: public GraphBase {
public:
    CudaGraphRunner(const DeviceInitParams& params,
                    py::object              py_instance,
                    int                     kv_cache_block_offset,
                    DeviceBase*             device,
                    bool                    is_prefill_cuda_graph_mode = false);

    ~CudaGraphRunner();

    void initCapture() override;
    void deviceSpecificSync() override {
        CudaGraphUtils::deviceSynchronize();
    }

    void capture();

    // CUDA特定的方法
    void setPositionEncoding(torch::Tensor position_encoding) override;
    void setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void setInputEmbeddingScalar(float input_embedding_scalar) override;
    void setModelDataType(caffe2::TypeMeta data_type) override;

    std::unique_ptr<void, std::function<void(void*)>> createStreamLife(void* capture_stream) override;
    void                                              setParamsPtr(int bs, const PyModelOutputs& outputs) override;
    void*                                             getDeviceStream() override {
        return &capture_stream_;
    }

    void initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs);
    void initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token);

    py::object normalForward(PyModelInputs& inputs) override;

    at::cuda::CUDAStream capture_stream_;
};
}  // namespace rtp_llm