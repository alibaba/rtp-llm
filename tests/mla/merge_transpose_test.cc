
#include "rtp_llm/cpp/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"

using namespace rtp_llm;

namespace unittest {
class MergeTransposeOP: public torch::jit::CustomClassHolder {
public:
    void forward(torch::Tensor q, torch::Tensor k_nope, torch::Tensor k_rope, torch::Tensor v, torch::Tensor qkv);
};

void MergeTransposeOP::forward(
    torch::Tensor q, torch::Tensor k_nope, torch::Tensor k_rope, torch::Tensor v, torch::Tensor qkv) {
    auto   stream      = at::cuda::getCurrentCUDAStream().stream();
    float* q_data      = (float*)q.data_ptr();
    float* k_nope_data = (float*)k_nope.data_ptr();
    float* k_rope_data = (float*)k_rope.data_ptr();
    float* v_data      = (float*)v.data_ptr();
    float* qkv_data    = (float*)qkv.data_ptr();

    int token_num = q.size(0);
    int head_num  = q.size(1);
    int nope_dim  = k_nope.size(2);
    int rope_dim  = k_rope.size(2);
    int vhead_dim = v.size(2);

    invokeMlaMergeTranspose<float>(
        q_data, k_nope_data, k_rope_data, v_data, qkv_data, token_num, head_num, nope_dim, rope_dim, vhead_dim, stream);
}

}  // namespace unittest

static auto MergeTransposeTHS = torch::jit::class_<unittest::MergeTransposeOP>("unittest", "MergeTransposeOP")
                                    .def(torch::jit::init<>())
                                    .def("forward", &unittest::MergeTransposeOP::forward);
