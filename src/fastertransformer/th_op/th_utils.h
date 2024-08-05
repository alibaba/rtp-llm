/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/core/Tensor.h"
#include "src/fastertransformer/core/allocator.h"

#include "torch/extension.h"
#include <cstdio>
#include <iostream>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>

#if USING_CUDA
#include "src/fastertransformer/cuda/allocator_torch.h"
#include "torch/csrc/cuda/Stream.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "src/fastertransformer/rocm/hip_utils.h"
#endif

#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)                                                                                             \
    CHECK_TH_CUDA(x);                                                                                                  \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_CPU_INPUT(x, st)                                                                                         \
    CHECK_CPU(x);                                                                                                      \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_OPTIONAL_INPUT(x, st)                                                                                    \
    if (x.has_value()) {                                                                                               \
        CHECK_INPUT(x.value(), st);                                                                                    \
    }
#define CHECK_OPTIONAL_CPU_INPUT(x, st)                                                                                \
    if (x.has_value()) {                                                                                               \
        CHECK_CPU_INPUT(x.value(), st);                                                                                \
    }
#define PRINT_TENSOR(x) std::cout << #x << ":\n" << x << std::endl
#define PRINT_TENSOR_SIZE(x) std::cout << "size of " << #x << ": " << x.sizes() << std::endl

namespace torch_ext {

template<typename T>
inline T* get_ptr(const torch::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

template<typename T>
inline T* get_ptr(const fastertransformer::Tensor& t) {
    return reinterpret_cast<T*>(t.data());
}

std::vector<size_t> convert_shape(torch::Tensor tensor);

template<typename T>
fastertransformer::Tensor convert_tensor(torch::Tensor tensor);

template<typename T>
fastertransformer::Tensor convert_tensor(torch::Tensor tensor, fastertransformer::MemoryType memory_type);

size_t sizeBytes(torch::Tensor tensor);

}  // namespace torch_ext
