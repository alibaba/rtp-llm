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

#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/cuda/memory_utils.h"

namespace torch_ext {
namespace ft = fastertransformer;
using torch::Tensor;

// copy tensor h2d
std::vector<Tensor> async_auto_copy_to_gpu(std::vector<Tensor> inputs)
{
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    std::vector<Tensor> results;
    for (auto &input: inputs) {
        Tensor res = torch::empty_like(input, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
        ft::cudaAutoCpy(get_ptr<int8_t>(res), get_ptr<int8_t>(input), input.nbytes(), stream);
        results.emplace_back(res);
    }
    return results;
}

// copy tensor h2d
Tensor async_copy_to_gpu(const Tensor& input)
{
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    Tensor res = torch::empty_like(input, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
    ft::cudaAutoCpy(get_ptr<int8_t>(res), get_ptr<int8_t>(input), input.nbytes(), stream);
    return res;
}

// copy tensor d2h
Tensor async_copy_to_cpu(const Tensor& input)
{
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    Tensor res = torch::empty_like(input, torch::dtype(input.dtype()).device(torch::kCPU).requires_grad(false));
    ft::cudaAutoCpy(get_ptr<int8_t>(res), get_ptr<int8_t>(input), input.nbytes(), stream);
    return res;
}

// auto copy
void async_auto_copy(std::vector<Tensor> tgt, std::vector<Tensor> src)
{
    TORCH_CHECK(tgt.size() == src.size(), "src tensor size is not euqal to src tensor size ");
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (size_t i = 0; i < src.size(); ++i) {
        TORCH_CHECK(tgt[i].nbytes() >= src[i].nbytes(), "src tensor size is not euqal to src tensor size ");
        ft::cudaAutoCpy(get_ptr<int8_t>(tgt[i]), get_ptr<int8_t>(src[i]), src[i].nbytes(), stream);
    }
}


// Results a tensor of {batch_to_compact_idx, compact_to_batch_idx}
std::vector<Tensor> find_context_duplications(Tensor input_ids)
{
    CHECK_INPUT(input_ids, torch::kInt32);
    TORCH_CHECK(input_ids.dim() == 2, "Invalid dim. Input ids must be a matrix [batch, seq_len]");

    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    const int batch_size = input_ids.size(0);
    const int seq_len    = input_ids.size(1);

    Tensor shared_contexts =
        torch::empty({batch_size}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    Tensor batch_to_compact     = torch::empty_like(shared_contexts);
    Tensor compact_to_batch_tmp = torch::empty_like(shared_contexts);

    Tensor compact_size_tensor =
        torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    ft::invokeFindContextDups(get_ptr<int>(shared_contexts),
                              get_ptr<int>(batch_to_compact),
                              get_ptr<int>(compact_to_batch_tmp),
                              get_ptr<int>(compact_size_tensor),
                              get_ptr<const int>(input_ids),
                              batch_size,
                              seq_len,
                              stream);

    Tensor    compact_size_cpu_tensor = compact_size_tensor.to(torch::kCPU);
    const int compact_size            = compact_size_cpu_tensor.item<int>();

    Tensor compact_to_batch =
        torch::empty({compact_size}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    ft::cudaD2Dcpy(get_ptr<int>(compact_to_batch), get_ptr<const int>(compact_to_batch_tmp), compact_size);
    return {batch_to_compact, compact_to_batch};
}

}  // namespace torch_ext

// Utility methods that may be useful for preprocessing weights in torch.
static auto find_context_duplications =
    torch::RegisterOperators("fastertransformer::find_context_duplications", &torch_ext::find_context_duplications);

// maybe faster than torch copy
static auto async_auto_copy =
    torch::RegisterOperators("fastertransformer::async_auto_copy", &torch_ext::async_auto_copy);

// maybe faster than torch copy
static auto async_copy_to_gpu =
    torch::RegisterOperators("fastertransformer::async_copy_to_gpu", &torch_ext::async_copy_to_gpu);

// maybe faster than torch copy
static auto async_copy_to_cpu =
    torch::RegisterOperators("fastertransformer::async_copy_to_cpu", &torch_ext::async_copy_to_cpu);
