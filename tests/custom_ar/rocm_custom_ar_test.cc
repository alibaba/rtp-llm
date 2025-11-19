#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"

#define private public
#include "rtp_llm/cpp/devices/DeviceFactory.h"

using namespace rtp_llm;

namespace unittest {

class ROCmCustomAROp : public torch::jit::CustomClassHolder {

public:
    ROCmCustomAROp(int64_t tp_rank, int64_t tp_size);

    void forward(torch::Tensor input, torch::Tensor output);

private:
    DeviceBase* device_ = nullptr;
};

ROCmCustomAROp::ROCmCustomAROp(int64_t tp_rank, int64_t tp_size) {
    auto             device_creator = DeviceFactory::getRegistrationMap().at(DeviceType::ROCm);
    DeviceInitParams params;
    params.device_id      = tp_rank;
    params.tp_rank        = tp_rank;
    params.tp_size        = tp_size;
    params.master_ip      = "localhost";
    params.tp_master_port = 68120;
    device_ = device_creator.create(params);
}

void ROCmCustomAROp::forward(torch::Tensor input, torch::Tensor output) {
    BufferPtr input_buffer = torchTensor2Buffer(input);
    BufferPtr output_buffer = torchTensor2Buffer(output);

    AllReduceParams allreduce_params(input_buffer, ReduceOp::Sum);

    AllReduceOutput allreduce_output = device_->allReduce(allreduce_params);

    device_->copy({*output_buffer, *(allreduce_output.buffer)});
}
    
} // namespace unittest

static auto ROCmCustomAROp = torch::jit::class_<unittest::ROCmCustomAROp>("unittest", "ROCmCustomAROp")
    .def(torch::jit::init<int64_t, int64_t>())
    .def("forward", &unittest::ROCmCustomAROp::forward);
