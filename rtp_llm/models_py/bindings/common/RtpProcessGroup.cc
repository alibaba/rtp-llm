#include "rtp_llm/models_py/bindings/common/RtpProcessGroup.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

namespace rtp_llm {

RtpProcessGroup::RtpProcessGroup(RtpProcessGroupType type) {
    device_ = dynamic_cast<DefaultDeviceType*>(DeviceFactory::getDefaultDevice());
    if (!device_) {
        RTP_LLM_LOG_ERROR("DeviceFactory::getDefaultDevice() return type is not CudaDevice !");
        throw std::runtime_error("DeviceFactory::getDefaultDevice() return type is not CudaDevice !");
    }
    auto device_properties = device_->getDeviceProperties();
    if (type == RtpProcessGroupType::DP_GROUP) {
        rank_       = device_properties.dp_rank;
        world_size_ = device_properties.dp_size;
        RTP_LLM_LOG_INFO("Create dp group: rank = %d, world_size = %d", rank_, world_size_);
        mode_ = ParallelMode::DP;
    } else if (type == RtpProcessGroupType::TP_GROUP) {
        rank_       = device_properties.tp_rank;
        world_size_ = device_properties.tp_size;
        RTP_LLM_LOG_INFO("Create tp group: rank = %d, world_size = %d", rank_, world_size_);
        mode_ = ParallelMode::TP;
    } else if (type == RtpProcessGroupType::DP_AND_TP_GROUP) {
        rank_       = device_properties.dp_rank * device_properties.tp_size + device_properties.tp_rank;
        world_size_ = device_properties.dp_size * device_properties.tp_size;
        RTP_LLM_LOG_INFO("Create dp and tp group: rank = %d, world_size = %d", rank_, world_size_);
        mode_ = ParallelMode::DP_AND_TP;
    } else {
        RTP_LLM_LOG_ERROR("Invalid RtpProcessGroupType !");
        throw std::runtime_error("Invalid RtpProcessGroupType !");
    }
}

void RtpProcessGroup::broadcast(std::vector<torch::Tensor>& input, int rootRank) {
    std::vector<BufferPtr> buffers;
    for (auto& tensor : input) {
        buffers.push_back(torchTensor2Buffer(tensor));
    }

    device_->broadcast({buffers, rootRank, mode_, false});
    check_cuda_error();
}

ReduceOp getReduceOp(c10d::ReduceOp reduce_op) {
    switch (reduce_op) {
        case c10d::ReduceOp::SUM:
            return ReduceOp::Sum;
        case c10d::ReduceOp::PRODUCT:
            return ReduceOp::Prod;
        case c10d::ReduceOp::MAX:
            return ReduceOp::Max;
        case c10d::ReduceOp::MIN:
            return ReduceOp::Min;
        case c10d::ReduceOp::AVG:
            return ReduceOp::Avg;
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "Invalid reduce op !");
            return ReduceOp::Sum;  // This line should never be reached
    }
}

std::vector<torch::Tensor> RtpProcessGroup::all_reduce(std::vector<torch::Tensor>& input) {
    RTP_LLM_CHECK_WITH_INFO(input.size() == 1, "AllReduce input size must be 1 , but got %d", input.size());
    auto     tensor      = input[0];
    auto     dest_tensor = torch::empty_like(tensor);
    ReduceOp reduce_op   = ReduceOp::Sum;
    device_->allReduce({torchTensor2Buffer(tensor), reduce_op, false, mode_, torchTensor2Buffer(dest_tensor)});
    check_cuda_error();
    return {dest_tensor};
}

void RtpProcessGroup::send(std::vector<torch::Tensor>& input, int dst_rank) {
    RTP_LLM_CHECK_WITH_INFO(input.size() == 1, "Send input size must be 1 , but got %d", input.size());
    BatchSendRecvParams params;
    params.p2p_params.push_back({SendRecvType::kSend, torchTensor2Buffer(input[0]), dst_rank});
    device_->batchSendRecv(params, mode_);
    check_cuda_error();
}

void RtpProcessGroup::recv(std::vector<torch::Tensor>& input, int src_rank) {
    RTP_LLM_CHECK_WITH_INFO(input.size() == 1, "Send input size must be 1 , but got %d", input.size());
    BatchSendRecvParams params;
    params.p2p_params.push_back({SendRecvType::kRecv, torchTensor2Buffer(input[0]), src_rank});
    device_->batchSendRecv(params, mode_);
    check_cuda_error();
}

std::vector<torch::Tensor> RtpProcessGroup::all_gather(std::vector<torch::Tensor>& input) {
    RTP_LLM_CHECK_WITH_INFO(input.size() == 1, "AllGather input size must be 1 , but got %d", input.size());
    auto output = torch::empty({input[0].size(0), input[0].size(1) * world_size_}, input[0].options());
    device_->allGather({{torchTensor2Buffer(output)}, mode_, {torchTensor2Buffer(input[0])}, false});
    check_cuda_error();
    return {output};
}

void registerRtpProcessGroup(const py::module& m) {
    py::enum_<RtpProcessGroupType>(m, "RtpProcessGroupType")
        .value("DP_GROUP", RtpProcessGroupType::DP_GROUP)
        .value("TP_GROUP", RtpProcessGroupType::TP_GROUP)
        .value("DP_AND_TP_GROUP", RtpProcessGroupType::DP_AND_TP_GROUP);

    // 注册RtpProcessGroup类
    py::class_<RtpProcessGroup>(m, "RtpProcessGroup")
        .def(py::init<RtpProcessGroupType>())
        .def("broadcast", &RtpProcessGroup::broadcast)
        .def("all_reduce", &RtpProcessGroup::all_reduce)
        .def("send", &RtpProcessGroup::send)
        .def("recv", &RtpProcessGroup::recv)
        .def("all_gather", &RtpProcessGroup::all_gather);
}

}  // namespace rtp_llm
