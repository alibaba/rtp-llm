#pragma once
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"

namespace rtp_llm {

enum class RtpProcessGroupType {
    DP_GROUP        = 0,
    TP_GROUP        = 1,
    DP_AND_TP_GROUP = 2,
};

class RtpProcessGroup {
public:
    RtpProcessGroup(RtpProcessGroupType type);
    ~RtpProcessGroup() = default;

    void broadcast(std::vector<torch::Tensor>& input, const c10d::BroadcastOptions& opts = c10d::BroadcastOptions());
    std::vector<torch::Tensor> all_reduce(std::vector<torch::Tensor>& input);

    void                       send(std::vector<torch::Tensor>& input, int dst_rank);
    std::vector<torch::Tensor> all_gather(std::vector<torch::Tensor>& input);
    void                       recv(std::vector<torch::Tensor>& input, int src_rank);

private:
    CudaDevice*  device_;
    ParallelMode mode_;
    int          rank_;
    int          world_size_;
};

void registerRtpProcessGroup(const py::module& m);

}  // namespace rtp_llm