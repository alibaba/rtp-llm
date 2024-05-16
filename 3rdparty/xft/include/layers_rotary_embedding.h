// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include <cmath>
#include <cstring>

#include "dtype.h"

namespace xft {
void invokeRotaryEmbedding(DataType dt,
        const int64_t *positionIds, // [num_tokens]
        void *query, // [num_tokens, head_num, head_size]
        void *key, // [num_tokens, head_num, head_size]
        const void *embCos, // [max_position, dim]
        const void *embSin, // [max_position, dim]
        const int dim, // rot_dim
        const int qStride, const int kStride, const int numTokens, const int headNum, const int headSize,
        const int numKvHeads);
} // namespace xft