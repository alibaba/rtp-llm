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

#include <optional>
#include "common.h"

namespace tensorrt_llm
{
namespace kernels
{

// force_path: 0 = auto route, 1 = force multi-block radix, 2 = force one-block radix.
// Any nonzero force_path clamps back to one-block when grid_dim < 2: multi-block is
// not valid below 2 blocks (the auto route has never selected it there).
// Primarily a testing/benchmarking knob; beam search also wires it to the
// RTP_LLM_BEAM_TOPK_FORCE_PATH env var as a runtime rollback switch (see
// beamSearchKernels.h). Workspace size depends on the path, so queries must pass the
// same force_path as the run.
// On ROCm, k <= 256 is gated to the WarpSort kernel for both the workspace query and the
// run: force_path has no effect there, and WarpSort always emits value-sorted output
// regardless of `sorted` (callers passing sorted=false still receive sorted results).
template <typename T>
size_t invokeComputeTopkLastDimWorkspaceSize(
    runtime::SizeType32 batchSize, runtime::SizeType32 inputLength, runtime::SizeType32 k, bool is_largest,
    int force_path = 0);

// sorted = false skips the trailing StableSortPairsDescending; out_val/out_idx then come out
// ordered by original index instead of by value. (Exception: the ROCm k<=256 WarpSort gate
// above always returns value-sorted output.)
template <typename T>
void invokeTopkLastDim(runtime::SizeType32 batchSize, runtime::SizeType32 inputLength, runtime::SizeType32 k, bool is_largest,
    std::optional<T> mask_val, void const* __restrict__ input, void* __restrict__ out_val, void* __restrict__ out_ind,
    void* workspace, cudaStream_t stream, bool sorted = true, int force_path = 0);

} // namespace kernels
} // namespace tensorrt_llm
