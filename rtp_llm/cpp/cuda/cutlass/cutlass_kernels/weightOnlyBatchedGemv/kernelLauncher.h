/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/weightOnlyBatchedGemv/common.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/weightOnlyBatchedGemv/details.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace weight_only
{
template <bool isGroupwise, typename Details>
void select_gs(Params& params, cudaStream_t s);

inline void kernel_launcher(int arch, Params& params, cudaStream_t s)
{
#define EXEC(KType, A, B, Layout)                                                                                      \
    if (params.type == KType) {                                                                                        \
        select_gs<kernel_type_traits<KType>::isGroupwise, KernelDetails<A, B, Layout, 64>>(params, s);                 \
        return;                                                                                                        \
    }
    if (arch >= 70 && arch < 75)
    {
        EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajor);
    }
    else if (arch >= 75 && arch < 80)
    {
        EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved);
        EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved);
    }
    else if (arch >= 80 && arch < 90)
    {
        EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved);
        EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved);
        EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved);
        EXEC(KernelType::BF16Int8PerChannel, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved);
    }
    else if (arch >= 90)
    {
        EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajor);
        EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajor);
        EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajor);
        EXEC(KernelType::BF16Int8PerChannel, BF16DetailsA, Int8DetailsW, ColumnMajor);
    }
#undef EXEC
}

inline bool is_supported(int arch, KernelType kernel_type)
{
#define SUPPORT(Type)                                                                                                  \
    if (kernel_type == Type)                                                                                           \
        return true;
    if (arch >= 70 && arch < 75)
    {
        return false;
        // SUPPORT(KernelType::FP16Int8PerChannel);
    }
    else if (arch >= 75 && arch < 80)
    {
        SUPPORT(KernelType::FP16Int4Groupwise);
        SUPPORT(KernelType::FP16Int8PerChannel);
    }
    else if (arch >= 80 && arch < 90)
    {
        SUPPORT(KernelType::FP16Int4Groupwise);
        SUPPORT(KernelType::BF16Int4Groupwise);
        SUPPORT(KernelType::FP16Int8PerChannel);
        SUPPORT(KernelType::BF16Int8PerChannel);
    }
    else if (arch >= 90)
    {
        return false;
        // SUPPORT(KernelType::FP16Int4Groupwise);
        // SUPPORT(KernelType::BF16Int4Groupwise);
        // SUPPORT(KernelType::FP16Int8PerChannel);
        // SUPPORT(KernelType::BF16Int8PerChannel);
    }
    return false;
#undef SUPPORT
}
} // namespace weight_only
} // namespace kernels
} // namespace tensorrt_llm
