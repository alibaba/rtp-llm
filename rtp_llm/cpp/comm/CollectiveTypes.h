#pragma once

#include <cstdint>
#include <vector>
#include <torch/torch.h>

namespace rtp_llm {

enum class ParallelMode {
    TP        = 0,
    DP        = 1,
    DP_AND_TP = 2,
    FFN_TP    = 3,
    EP        = 4,
    EPLB      = 5,
};

struct BroadcastParams {
    const std::vector<torch::Tensor>& buffers;
    const int64_t                     root;
    ParallelMode                      mode       = ParallelMode::TP;
    bool                              overlapped = false;
};

enum class ReduceOp {
    Sum  = 0,
    Prod = 1,
    Max  = 2,
    Min  = 3,
    Avg  = 4,
};

struct AllReduceParams {
    torch::Tensor  buffer;
    const ReduceOp op;
    bool           overlapped = false;
    ParallelMode   mode       = ParallelMode::TP;
    torch::Tensor  dest;  // Undefined when no separate destination is requested.
};

struct AllReduceOutput {
    torch::Tensor buffer;
};

struct AllGatherParams {
    const std::vector<torch::Tensor>& recv_buffers;
    ParallelMode                      mode = ParallelMode::TP;
    std::vector<torch::Tensor>        send_buffers;
    bool                              inplace    = true;
    bool                              overlapped = false;
};

}  // namespace rtp_llm
