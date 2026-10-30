/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace rtp_llm {

template<typename T>
inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

inline int div_up(int a, int b) {
    return ceil_div<int>(a, b);
}

template<typename T,
         typename U,
         typename = std::enable_if_t<std::is_integral<T>::value>,
         typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator) {
    return (numerator + denominator - 1) / denominator;
}

inline size_t pad(const size_t& input, const size_t& alignment) {
    return alignment * ((input + alignment - 1) / alignment);
}

// Keep complete Q-token request rows while making their total divisible by SP.
inline size_t padSequenceParallelRequests(size_t requests, size_t query_width, size_t sequence_parallel_size) {
    return pad(requests, sequence_parallel_size / std::gcd(sequence_parallel_size, query_width));
}

inline int mtpDraftPrefillRequestCapacity(
    int token_capacity, int max_requests, int query_width, int sequence_parallel_size) {
    if (token_capacity <= 0 || max_requests <= 0 || query_width <= 0 || sequence_parallel_size <= 0) {
        throw std::invalid_argument("MTP draft prefill capacities, query width and SP size must be positive");
    }
    // Every MTP draft update retains Q physical tokens per request.
    const int64_t requests = ceilDiv(static_cast<int64_t>(token_capacity), query_width);
    const int64_t rounded  = padSequenceParallelRequests(requests, query_width, sequence_parallel_size);
    if (rounded > std::numeric_limits<int>::max() || rounded * query_width > std::numeric_limits<int>::max()) {
        throw std::overflow_error("MTP draft prefill rectangular capacity exceeds int range");
    }
    if (rounded > max_requests) {
        throw std::invalid_argument("MTP draft prefill bucket exceeds request capacity");
    }
    return static_cast<int>(rounded);
}

inline size_t pad_to_multiple_of_16(const size_t& input) {
    return pad(input, 16);
}

inline size_t pad_to_multiple_of_64(const size_t& input) {
    return pad(input, 64);
}

inline size_t pad_to_multiple_of_128(const size_t& input) {
    return pad(input, 128);
}

}  // namespace rtp_llm
