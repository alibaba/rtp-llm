// SPDX-License-Identifier: Apache-2.0
// RTP-LLM binding for the vLLM-derived bounded alignment candidate.
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> sm90_moe_align_cuda(
    const at::Tensor& ids, int64_t num_experts, int64_t block_size,
    int64_t local_start);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("align", &sm90_moe_align_cuda);
}
