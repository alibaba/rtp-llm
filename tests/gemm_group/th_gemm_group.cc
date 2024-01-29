#include <cublas_v2.h>
#include <iostream>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/fastertransformer/cutlass/cutlass_kernels/group_gemm/group_gemm.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;

namespace torch_ext {

namespace ft = fastertransformer;

template<typename T>
void group_gemm_helper(T** As, T** Bs, T** Cs, int* m, int* n, int* k, int count) {
    
    int timing_iterations = 1;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fastertransformer::CutlassGroupGemmRunner<T> group_gemm_runner;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    group_gemm_runner.gemm(As, Bs, Cs, m, n, k, 1.0f, 0.0f, count, stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    FT_LOG_INFO("group gemm avg_time is % d\n", total_time_ms);

    return;
}

std::vector<Tensor> group_gemm(std::vector<Tensor> As, std::vector<Tensor> Bs) {

    TORCH_CHECK(As.size() == Bs.size(), "As and Bs must be equal");
    int count = As.size();
    for (auto A : As) {
        TORCH_CHECK(A.dim() == 2, "Invalid rank for A");
        
    }
    for (auto B : Bs) {
        TORCH_CHECK(B.dim() == 2, "Invalid rank for B");
    }
    std::vector<int> ms(count);
    std::vector<int> ns(count);
    std::vector<int> ks(count);

    for (int i = 0; i < count; i++) {
        ms[i] = As[i].size(0);
        ks[i] = As[i].size(1);
    }
    for (int i = 0; i < count; i++) {
        TORCH_CHECK((ks[i] == Bs[i].size(0)), "Invalid rank for B");
        ns[i] = Bs[i].size(1);
    }

    std::vector<Tensor> Cs(count);
    for (int i = 0; i < As.size(); i++) {
        Cs[i] = torch::zeros({ms[i], ns[i]}).to(As[i].dtype()).to(As[i].device());
    }
    switch (As[0].scalar_type()) {
        case at::ScalarType::Half: {
            std::vector<half*> A_ptrs(count);
            std::vector<half*> B_ptrs(count);
            std::vector<half*> C_ptrs(count);
            for (int i = 0; i < count; i++) {
                A_ptrs[i] = get_ptr<half>(As[i]);
                B_ptrs[i] = get_ptr<half>(Bs[i]);
                C_ptrs[i] = get_ptr<half>(Cs[i]);
            }
            group_gemm_helper<half>(A_ptrs.data(), B_ptrs.data(), C_ptrs.data(),
                                    ms.data(), ns.data(), ks.data(), count);
        }
        break;

        case at::ScalarType::Float: {
            std::vector<float*> A_ptrs(count);
            std::vector<float*> B_ptrs(count);
            std::vector<float*> C_ptrs(count);
            std::vector<float*> D_ptrs(count);
            for (int i = 0; i < count; i++) {
                A_ptrs[i] = get_ptr<float>(As[i]);
                B_ptrs[i] = get_ptr<float>(Bs[i]);
                C_ptrs[i] = get_ptr<float>(Cs[i]);
            }
            group_gemm_helper<float>(A_ptrs.data(), B_ptrs.data(), C_ptrs.data(),
                                    ms.data(), ns.data(), ks.data(), count);

        }
        break;
    }
    return Cs;
}


TORCH_LIBRARY(group_gemm_ops, m)
{
    m.def("group_gemm", group_gemm);
} 

} // namespace torch_ext