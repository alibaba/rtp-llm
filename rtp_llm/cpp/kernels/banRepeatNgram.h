/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace runtime {
using TokenIdType = int32_t;
using SizeType32  = int32_t;
};  // namespace runtime
using FinishedState = void;

namespace tensorrt_llm {
namespace kernels {

template<typename T>
void invokeBanRepeatNgram(T*                           logits,
                          runtime::TokenIdType const** output_ids_buf,
                          FinishedState const*         finished_buf,
                          runtime::SizeType32 const**  parent_ids_buf,
                          runtime::SizeType32 const*   batch_slot,
                          runtime::SizeType32 const*   sequence_lengths,
                          runtime::SizeType32          batch_size,
                          runtime::SizeType32          beam_width,
                          runtime::SizeType32          max_seq_len,
                          runtime::SizeType32 const*   no_repeat_ngram_size_buf,
                          runtime::SizeType32          vocab_size_padded,
                          runtime::SizeType32          max_step,
                          cudaStream_t                 stream);

}  // namespace kernels
}  // namespace tensorrt_llm
