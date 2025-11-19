#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/all.h>

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"

// aiter quick all reduce kernel
#include "quick_all_reduce.h"

namespace rtp_llm {

class QuickAllReduceComm {

public:
    QuickAllReduceComm(const std::vector<size_t>& tp_ranks, size_t rank, size_t rank_index);

    ~QuickAllReduceComm();

    void init(const NcclParam& nccl_para, hipStream_t stream);

    void allReduce(torch::Tensor& input_tensor, torch::Tensor& output_tensor);

    bool checkAllReduceAvailable(size_t elts_total_num, DataType data_type, size_t world_size);

    static bool shouldQuickAR(const std::vector<size_t>& tp_ranks, size_t rank);

private:
    std::vector<std::vector<char>> all_gather(void* addr, size_t size, hipStream_t stream);

    const size_t        rank_               = 0;
    const size_t        rank_index_         = 0;
    const size_t        world_size_         = 0;
    bool                support_nv_link_    = false;
    std::vector<size_t> tp_ranks_;
    int64_t             ptr_;
    bool                use_fp16_kernels_;
    int64_t             qr_quant_level_;
    int64_t             qr_max_size_;
    NcclParam           nccl_para_;

    inline static std::map<std::string, int64_t> QuickReduceRegime = {
        {"FP",   0},
        {"FP8",  1},
        {"INT6", 2},
        {"INT4", 3},
        {"NONE", 4},
    };

    // following data is based on kernel tests, order: [FP, FP8, INT6, INT4], unit: MB
    // NOTE(liyangcheng.lyc): for MI308X, TP2 and TP4 gains improvement in 0~2048MB range, TP8 only gains improvement in 4~64MB range
    inline static std::map<std::pair<uint8_t, size_t>, std::vector<std::pair<int64_t, int64_t>>> QR_SIZE_RANGE = {
        {{DataType::TYPE_FP16, 2}, {{1, 2048},  {2, 2048},    {2, 2048},    {1, 2048}}},
        {{DataType::TYPE_FP16, 4}, {{1, 2048},  {16, 2048},   {4, 2048},    {2, 2048}}},
        {{DataType::TYPE_FP16, 8}, {{16, 2048}, {4, 2048},    {4, 2048},    {4, 64}}},
        {{DataType::TYPE_BF16, 2}, {{2, 2048},  {8, 2048},    {8, 2048},    {8, 2048}}},
        {{DataType::TYPE_BF16, 4}, {{8, 2048},  {64, 2048},   {64, 2048},   {16, 2048}}},
        {{DataType::TYPE_BF16, 8}, {{16, 2048}, {2048, 2048}, {2048, 2048}, {2048, 2048}}},
    };
};

std::unique_ptr<QuickAllReduceComm>
initQuickAllReduceComm(const NcclParam& nccl_para, const std::vector<size_t>& tp_ranks, hipStream_t stream);

} // namespace rtp_llm
