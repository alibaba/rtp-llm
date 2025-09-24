#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"

#include "torch/extension.h"
#include <cstdio>
#include <iostream>
#include <memory>

namespace torch_ext {

// For now the DeviceExporter only export single device as there is no pipeline parallelism
// It may need to hold multiple devices in the future.
class DeviceExporter {
public:
    DeviceExporter(const rtp_llm::DeviceInitParams& params): device_params_(params) {};
    virtual ~DeviceExporter() {};

    rtp_llm::DeviceType getDeviceType();
    int64_t             getDeviceId();

    void updateCurrentTorchStream();

    virtual torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool = false) = 0;
    virtual torch::Tensor packInt8TensorToPackedInt4(torch::Tensor weight)                                      = 0;
    virtual torch::Tensor
    preprocessWeightsForMixedGemm(torch::Tensor weight, py::object quant_type, const std::string& arch) = 0;
    virtual std::vector<torch::Tensor>
    symmetricQuantizeLastAxisOfBatchedMatrix(torch::Tensor weight, py::object quant_type, const std::string& arch) = 0;
    virtual torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale)                         = 0;

protected:
    rtp_llm::DeviceInitParams device_params_;
};

template<class Device>
class DeviceExporterImpl: public DeviceExporter {
public:
    DeviceExporterImpl(const rtp_llm::DeviceInitParams& params): DeviceExporter(params) {};
    ~DeviceExporterImpl() {};

    torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool user_arm_gemm_use_kai) {
        return Device::preprocessGemmWeightByKey(key, weight, user_arm_gemm_use_kai);
    }

    torch::Tensor packInt8TensorToPackedInt4(torch::Tensor weight) {
        return Device::packInt8TensorToPackedInt4(weight);
    }

    torch::Tensor preprocessWeightsForMixedGemm(torch::Tensor weight, py::object quant_type, const std::string& arch) {
        const auto dtype = torch::python::detail::py_object_to_dtype(quant_type);
        return Device::preprocessWeightsForMixedGemm(weight, dtype, arch);
    }

    std::vector<torch::Tensor>
    symmetricQuantizeLastAxisOfBatchedMatrix(torch::Tensor weight, py::object quant_type, const std::string& arch) {
        const auto dtype = torch::python::detail::py_object_to_dtype(quant_type);
        return Device::symmetricQuantizeLastAxisOfBatchedMatrix(weight, dtype, arch);
    }
    torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale) {
        return Device::preprocessWeightScale(weight, scale);
    }
};

}  // namespace torch_ext
