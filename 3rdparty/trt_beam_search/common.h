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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

namespace tensorrt_llm::common {

template <typename T1, typename T2>
constexpr inline size_t divUp(T1 const& a, T2 const& b)
{
    auto const tmp_a = static_cast<size_t>(a);
    auto const tmp_b = static_cast<size_t>(b);
    return (tmp_a + tmp_b - 1) / tmp_b;
}

constexpr inline size_t roundUp(size_t a, size_t b)
{
    return divUp(a, b) * b;
}

inline int getMultiProcessorCount()
{
    int nSM{0};
    int deviceID{0};
    check_cuda_value(cudaGetDevice(&deviceID));
    check_cuda_value(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
    return nSM;
}

inline int getMaxSharedMemoryPerSM()
{
    int nByteMaxSharedMemoryPerSM{0};
    int deviceID{0};
    check_cuda_value(cudaGetDevice(&deviceID));
    check_cuda_value(
        cudaDeviceGetAttribute(&nByteMaxSharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, deviceID));
    return nByteMaxSharedMemoryPerSM;
}

inline int getMaxSharedMemoryPerBlockOptin()
{
    int nByteMaxSharedMemoryPerBlockOptin{0};
    int deviceID{0};
    check_cuda_value(cudaGetDevice(&deviceID));
    check_cuda_value(
        cudaDeviceGetAttribute(&nByteMaxSharedMemoryPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID));
    return nByteMaxSharedMemoryPerBlockOptin;
}

} // tensorrt_llm::common

namespace tensorrt_llm::runtime
{

#define FMT_DIM "%ld"

// typedefs
// Note that we use signed size types as recommended by TensorRT:
// https://github.com/NVIDIA/TensorRT/blob/main/CODING-GUIDELINES.md#signed-vs-unsigned-integers
using SizeType32 = std::int32_t;
using SizeType64 = std::int64_t;

enum class RequestType : std::int32_t
{
    kCONTEXT = 0,
    kGENERATION = 1
};

// Token ID type
using TokenIdType = std::int32_t;

using LoraTaskIdType = std::uint64_t;
using TokenExtraIdType = std::uint64_t;
using VecTokenExtraIds = std::vector<TokenExtraIdType>;

struct UniqueToken
{
    TokenIdType tokenId;
    TokenExtraIdType tokenExtraId;

    bool operator==(UniqueToken const& other) const noexcept
    {
        return (tokenId == other.tokenId && tokenExtraId == other.tokenExtraId);
    }
};

using VecUniqueTokens = std::vector<UniqueToken>;

template <typename T>
using StringPtrMap = std::unordered_map<std::string, std::shared_ptr<T>>;

} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::executor {

/// @brief The reason why the model stopped generating tokens for a request.
enum class FinishReason
{
    /// @brief The request is not finished.
    kNOT_FINISHED = 0,

    /// @brief The request finished because the end id was generated.
    kEND_ID = 1,

    /// @brief The request finished because a stop word was generated.
    kSTOP_WORDS = 2,

    /// @brief The request finished because the maximum number of tokens was reached.
    kLENGTH = 3,

    /// @brief The request finished because it got timed out (via the mAllotedTime parameter)
    kTIMED_OUT = 4,

    /// @brief The request was cancelled by calling cancelRequest.
    kCANCELLED = 5
};

} // tensorrt_llm::executor
