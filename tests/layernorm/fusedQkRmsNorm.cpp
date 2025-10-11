#ifdef USING_ROCM
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#else
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#endif
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

using namespace rtp_llm;

namespace unittest {

class FusedQkRmsNormOp: public torch::jit::CustomClassHolder {
public:
    FusedQkRmsNormOp(double eps): eps(eps) {};

    void forward(torch::Tensor                  input,
                 torch::Tensor                  q_gamma,
                 torch::optional<torch::Tensor> q_bias,
                 torch::Tensor                  k_gamma,
                 torch::optional<torch::Tensor> k_bias,
                 int64_t                        q_group_num,
                 int64_t                        k_group_num,
                 int64_t                        norm_size);

private:
    DeviceBase* device_;
    double      eps;
};

void FusedQkRmsNormOp::forward(torch::Tensor                  input,
                               torch::Tensor                  q_gamma,
                               torch::optional<torch::Tensor> q_bias,
                               torch::Tensor                  k_gamma,
                               torch::optional<torch::Tensor> k_bias,
                               int64_t                        q_group_num,
                               int64_t                        k_group_num,
                               int64_t                        norm_size) {
    auto gpt_params = GptInitParameter();
    rtp_llm::DeviceFactory::initDevices(gpt_params);
    device_ = rtp_llm::DeviceFactory::getDefaultDevice();
    ;
    int  m                       = input.size(0);
    int  n                       = input.size(1);
    auto input_buffer            = torchTensor2Buffer(input);
    auto q_gamma_buffer          = torchTensor2Buffer(q_gamma);
    auto q_bias_buffer           = q_bias ? torchTensor2Buffer(q_bias.value()) : nullptr;
    auto q_norm_weight           = std::make_shared<const LayerNormWeights>(q_gamma_buffer, q_bias_buffer);
    auto k_gamma_buffer          = torchTensor2Buffer(k_gamma);
    auto k_bias_buffer           = k_bias ? torchTensor2Buffer(k_bias.value()) : nullptr;
    auto k_norm_weight           = std::make_shared<const LayerNormWeights>(k_gamma_buffer, k_bias_buffer);
    auto fused_qk_rmsnorm_params = QkRmsNormParams({input_buffer,
                                                    *q_norm_weight,
                                                    *k_norm_weight,
                                                    eps,
                                                    (size_t)q_group_num,
                                                    (size_t)k_group_num,
                                                    (size_t)norm_size});
    auto out                     = device_->qkRmsNorm(fused_qk_rmsnorm_params);
}

}  // namespace unittest

static auto FusedQkRmsNormTHS = torch::jit::class_<unittest::FusedQkRmsNormOp>("unittest", "FusedQkRmsNormOp")
                                    .def(torch::jit::init<double>())
                                    .def("forward", &unittest::FusedQkRmsNormOp::forward);
