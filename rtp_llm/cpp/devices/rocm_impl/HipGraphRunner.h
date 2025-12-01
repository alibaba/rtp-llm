#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/devices/rocm_impl/HipGraphUtils.h"
#include "rtp_llm/cpp/devices/GraphBase.h"
#include <ATen/hip/HIPGraph.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

namespace py = pybind11;
namespace rtp_llm {

// 前向声明
class ROCmDevice;

class HipGraphRunner: public GraphBase {
public:
    HipGraphRunner(const DeviceInitParams& params,
                   py::object              py_instance,
                   int                     kv_cache_block_offset,
                   DeviceBase*             device,
                   bool                    is_prefill_hip_graph_mode = false);

    ~HipGraphRunner();

    void initCapture() override;
    void initKernelInternalMemory();
    void capture();
    void deviceSpecificSync() override {
        HipGraphUtils::deviceSynchronize();
    }

    // ROCm特定的方法
    void setPositionEncoding(torch::Tensor position_encoding) override {
        position_encoding_ = position_encoding;
    }

    void setTokenTypeEmbedding(torch::Tensor token_type_embedding) override {
        token_type_embedding_ = token_type_embedding;
    }

    void setInputEmbeddingScalar(float input_embedding_scalar) override {
        input_embedding_scalar_ = input_embedding_scalar;
    }

    void setModelDataType(caffe2::TypeMeta data_type) override {
        model_data_type_ = data_type;
    }

    std::unique_ptr<void, std::function<void(void*)>> createStreamLife(void* capture_stream) override;
    void                                              setParamsPtr(int bs, const PyModelOutputs& outputs) override;
    void*                                             getDeviceStream() override {
        return &capture_stream_;
    }

    std::string sizesToString(const torch::Tensor& tensor);
    std::string getDTypeName(const torch::Tensor& tensor);

    at::hip::HIPStream capture_stream_;
};

}  // namespace rtp_llm