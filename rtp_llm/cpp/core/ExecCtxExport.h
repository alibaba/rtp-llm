#pragma once

#include "rtp_llm/cpp/core/DeviceData.h"

#include "torch/extension.h"
#include <cstdio>
#include <iostream>
#include <memory>

namespace torch_ext {

class ExecCtxExporter {
public:
    ExecCtxExporter(const rtp_llm::ExecInitParams& params): exec_params_(params) {};
    virtual ~ExecCtxExporter() {};

    rtp_llm::DeviceType getDeviceType();
    int64_t             getDeviceId();

    virtual torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool = false) = 0;
    virtual torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale)                      = 0;

protected:
    rtp_llm::ExecInitParams exec_params_;
};

template<class Device>
class ExecCtxExporterImpl: public ExecCtxExporter {
public:
    ExecCtxExporterImpl(const rtp_llm::ExecInitParams& params): ExecCtxExporter(params) {};
    ~ExecCtxExporterImpl() {};

    torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool user_arm_gemm_use_kai) {
        return Device::preprocessGemmWeightByKey(key, weight, user_arm_gemm_use_kai);
    }

    torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale) {
        return Device::preprocessWeightScale(weight, scale);
    }
};

}  // namespace torch_ext

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);
}  // namespace rtp_llm
