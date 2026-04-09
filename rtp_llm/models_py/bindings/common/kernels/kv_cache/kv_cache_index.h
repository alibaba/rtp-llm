/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>
#include "rtp_llm/cpp/utils/AssertUtils.h"

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

class KVCacheIndex {
public:
    using UnderlyingType = std::int32_t;

    // Flag indicating KVCacheIndex refers to secondary pool
    static constexpr UnderlyingType kSecondaryPoolFlag = static_cast<UnderlyingType>(1)
                                                         << (8 * sizeof(UnderlyingType) - 1);

    explicit KVCacheIndex(UnderlyingType value, bool isSecondary = false):
        value{isSecondary ? value | kSecondaryPoolFlag : value} {
        RTP_LLM_CHECK_WITH_INFO(value >= 0, "value < 0 is not allowed.");
    }

    __host__ __device__ UnderlyingType get() const {
        return value & (~kSecondaryPoolFlag);
    }

    __host__ __device__ bool isPrimary() const {
        return (value & kSecondaryPoolFlag) == 0;
    }

private:
    UnderlyingType value;
};

}  // namespace rtp_llm
