#pragma once

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include "rtp_llm/cpp/kernels/moe_kernels.h"

namespace rtp_llm {

class FakeBalanceExpertOp {
public:
    FakeBalanceExpertOp(int64_t expert_num, int64_t moe_k, int64_t dp_rank, int64_t dp_size, int64_t ep_size);
    void forward(torch::Tensor expert_ids, torch::Tensor expert_scales);

private:
    int64_t expert_num_;
    int64_t moe_k_;
    int64_t dp_rank_;
    int64_t dp_size_;
    int64_t ep_size_;
};

void registerFakeBalanceExpertOp(pybind11::module& m);
}  // namespace rtp_llm
