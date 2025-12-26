#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

#define private public
#include "rtp_llm/cpp/devices/DeviceFactory.h"

using namespace rtp_llm;

namespace unittest {

class ROCmCustomAGOp : public torch::jit::CustomClassHolder {

public:
    ROCmCustomAGOp(int64_t tp_rank, int64_t tp_size);

    void forward(torch::Tensor all_gather_input, torch::Tensor all_gather_output, bool inplace, torch::Tensor all2all_input, torch::Tensor all2all_output);

private:
    DeviceBase* device_ = nullptr;
    DeviceInitParams params_;
};

ROCmCustomAGOp::ROCmCustomAGOp(int64_t tp_rank, int64_t tp_size) {
    auto             device_creator = DeviceFactory::getRegistrationMap().at(DeviceType::ROCm);
    params_.device_id               = tp_rank;
    // ep = tp, dp = 1, used to debug custom ag + all2all issue
    params_.tp_rank                 = tp_rank;
    params_.tp_size                 = tp_size;
    params_.ep_rank                 = tp_rank;
    params_.ep_size                 = tp_size;
    params_.master_ip               = "localhost";
    params_.tp_master_port          = 68120;
    params_.dp_tp_master_port       = 56789;
    device_ = device_creator.create(params_);
}

void ROCmCustomAGOp::forward(torch::Tensor all_gather_input, torch::Tensor all_gather_output, bool inplace, torch::Tensor all2all_input, torch::Tensor all2all_output) {
    BufferPtr all_gather_input_buffer = torchTensor2Buffer(all_gather_input);   // input shape:  [xx, yy, ...]
    BufferPtr all_gather_output_buffer = torchTensor2Buffer(all_gather_output); // output shape: [tp_size, xx, yy, ...]

    // try custom all gather
    if (inplace) {
        auto all_gather_input_size = all_gather_input_buffer->size();
        Buffer all_gather_input_flatten = all_gather_input_buffer->reshape({all_gather_input_size});
        Buffer all_gather_output_flatten = all_gather_output_buffer->reshape({all_gather_output_buffer->size()});

        device_->copy({all_gather_output_flatten.view(params_.tp_rank * all_gather_input_size, all_gather_input_size), all_gather_input_flatten});

        device_->allGather({{all_gather_output_buffer}, ParallelMode::TP, {}, inplace, false});
    } else {
        device_->allGather({{all_gather_output_buffer}, ParallelMode::TP, {all_gather_input_buffer}, inplace, false});
    }

    // try all2all
    BufferPtr all2all_input_buffer = torchTensor2Buffer(all2all_input);
    BufferPtr all2all_res = device_->allToAll({{all2all_input_buffer}}).outputs[0];

    BufferPtr all2all_output_buffer = torchTensor2Buffer(all2all_output);
    device_->copy({*all2all_output_buffer, all2all_res->reshape({all2all_input_buffer->size()})});
}
    
} // namespace unittest

static auto ROCmCustomAGOp = torch::jit::class_<unittest::ROCmCustomAGOp>("unittest", "ROCmCustomAGOp")
    .def(torch::jit::init<int64_t, int64_t>())
    .def("forward", &unittest::ROCmCustomAGOp::forward);
