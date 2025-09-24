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

inline size_t pad_to_multiple_of_16(const size_t& input) {
    return pad(input, 16);
}

inline size_t pad_to_multiple_of_128(const size_t& input) {
    return pad(input, 128);
}

}  // namespace rtp_llm
