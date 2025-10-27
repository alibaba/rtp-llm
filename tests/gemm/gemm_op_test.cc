
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/OpData.h"

using namespace rtp_llm;
namespace unittest {

class GemmOp: public torch::jit::CustomClassHolder {
public:
    GemmOp();

    void forward(torch::Tensor input, 
                 torch::Tensor weight, 
                 torch::Tensor weight_scale, 
                 torch::Tensor output);
    void forward_with_input_scale(torch::Tensor input,
                                  torch::Tensor input_scale,
                                  torch::Tensor weight,
                                  torch::Tensor weight_scale,
                                  torch::Tensor output,
                                  std::optional<torch::Tensor> bias);
private:
    DeviceBase* device = nullptr;

    BufferPtr make_qbuffer(const BufferPtr& kernel_buf, const BufferPtr& scale_buf) {
        auto zeros = BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr));
        auto kernel = BufferPtr(new Buffer(kernel_buf->where(), kernel_buf->type(), kernel_buf->shape(), kernel_buf->data()));
        auto scales = BufferPtr(new Buffer(scale_buf->where(), scale_buf->type(), scale_buf->shape(), scale_buf->data()));
        return BufferPtr(new QBuffer(std::move(kernel), std::move(scales), std::move(zeros)));
    }
};

GemmOp::GemmOp() {
    rtp_llm::initLogger();
    DeviceFactory::initDevices(GptInitParameter());
    device = DeviceFactory::getDefaultDevice();
}

void GemmOp::forward(torch::Tensor input, torch::Tensor weight, torch::Tensor weight_scale, torch::Tensor output) {
    auto hidden        = torchTensor2Buffer(input);
    auto kernel        = torchTensor2Buffer(weight);
    auto scales_buffer = torchTensor2Buffer(weight_scale);
    auto output_buffer = torchTensor2Buffer(output);

    auto weight_buffer =make_qbuffer(kernel, scales_buffer);
    auto gemm_result = device->gemm({*hidden, *weight_buffer});
    device->copy({*output_buffer, *gemm_result});
}

void GemmOp::forward_with_input_scale(torch::Tensor input,
                                      torch::Tensor input_scale,
                                      torch::Tensor weight,
                                      torch::Tensor weight_scale,
                                      torch::Tensor output,
                                      std::optional<torch::Tensor> bias) {
    auto a_buf       = torchTensor2Buffer(input);
    auto a_scale_buf = torchTensor2Buffer(input_scale);
    auto b_buf       = torchTensor2Buffer(weight);
    auto b_scale_buf = torchTensor2Buffer(weight_scale);
    auto out_buf     = torchTensor2Buffer(output);

    auto a_qptr = make_qbuffer(a_buf, a_scale_buf);
    auto b_qptr = make_qbuffer(b_buf, b_scale_buf);
    BufferPtr gemm_result;

    if (bias != std::nullopt){
        torch::Tensor bias_t = bias.value().to(output.dtype()).contiguous();
        const int64_t n = output.size(-1);

        // Normalization to 1D and GemmParams::check() will broadcast [n] to [m, n]
        if (bias_t.dim() == 2 && bias_t.size(0) == 1 && bias_t.size(1) == n) {
            bias_t = bias_t.view({n});
        }
        auto c_buf     = torchTensor2Buffer(bias_t);
        gemm_result = device->gemm({*a_qptr, *b_qptr, *c_buf});
    } else {
        gemm_result = device->gemm({*a_qptr, *b_qptr});
    }
    device->copy({*out_buf, *gemm_result});
}

}  // namespace unittest

static auto GemmOp = torch::jit::class_<unittest::GemmOp>("unittest", "GemmOp")
                         .def(torch::jit::init<>())
                         .def("forward", &unittest::GemmOp::forward)
                         .def("forward_with_input_scale", &unittest::GemmOp::forward_with_input_scale);
