#pragma once

#include "rtp_llm/cpp/pybind/th_utils.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"

namespace th = torch;
namespace torch_ext {

class NcclOp: public th::jit::CustomClassHolder {
public:
    NcclOp(const int64_t     tensor_para_size,
           const int64_t     pipeline_para_size,
           const int64_t     world_size,
           const int64_t     world_rank,
           const std::string master_ip,
           const int64_t     master_port);

    ~NcclOp();

    void broadcast_tp(std::vector<th::Tensor> tensors, int64_t root, bool timeout = true);

private:
    size_t             tensor_para_size_;
    size_t             pipeline_para_size_;
    rtp_llm::NcclParam tensor_para_;
    rtp_llm::NcclParam pipeline_para_;
};

}  // namespace torch_ext
