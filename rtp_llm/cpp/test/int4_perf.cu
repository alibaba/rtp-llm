#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <vector>

#include "rtp_llm/cpp/cuda/cublas/cublasMMWrapper.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/memory_utils.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_lut_utils.h"
#include "rtp_llm/cpp/cuda/cutlass/interface.h"

#include "rtp_llm/cpp/cuda/allocator_cuda.h"
#include "rtp_llm/cpp/cuda/cublas/cublas.h"

namespace tc  = tensorrt_llm::cutlass_extensions;
namespace tkc = tensorrt_llm::kernels::cutlass_kernels;

struct Dim2 {
    int k;
    int n;
};

void gemm_test(int m, Dim2 dim2, cudaStream_t stream) {
    int n          = dim2.n;
    int k          = dim2.k;
    int group_size = 128;

    half* in_ptr1 = nullptr;
    deviceMalloc(&in_ptr1, m * k);

    half* w_ptr1 = nullptr;
    deviceMalloc(&w_ptr1, n * k);

    half* s_ptr1 = nullptr;
    deviceMalloc(&s_ptr1, n);

    half* z_ptr1 = nullptr;
    deviceMalloc(&z_ptr1, n * k / group_size);

    half* out_ptr1 = nullptr;
    deviceMalloc(&out_ptr1, m * n, false);
    check_cuda_value(cudaMemset(out_ptr1, 0xdc, m * n * sizeof(half)));

    half* in_ptr2 = nullptr;
    deviceMalloc(&in_ptr2, m * k);

    half* w_ptr2 = nullptr;
    deviceMalloc(&w_ptr2, n * k);

    half* s_ptr2 = nullptr;
    deviceMalloc(&s_ptr2, n * k / group_size);

    half* z_ptr2 = nullptr;
    deviceMalloc(&z_ptr2, n * k / group_size);

    half* out_ptr2 = nullptr;
    deviceMalloc(&out_ptr2, m * n, false);
    check_cuda_value(cudaMemset(out_ptr2, 0xdc, m * n * sizeof(half)));

    tkc::CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>
        int4_runner;

    char*     ws_ptr2  = nullptr;
    const int ws_size2 = int4_runner.getWorkspaceSize(m, n, k);
    deviceMalloc(&ws_ptr2, ws_size2);

    tc::CutlassGemmConfig int4_config = int4_runner.getChosenConfig(
        in_ptr2, w_ptr2, s_ptr2, z_ptr2, nullptr, out_ptr2, m, n, k, group_size, ws_ptr2, ws_size2, stream);

    int  mArch              = rtp_llm::get_sm();
    bool mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
        mArch, tensorrt_llm::kernels::weight_only::KernelType::FP16Int4Groupwise);
    tensorrt_llm::kernels::weight_only::KernelType mCudaKernelType =
        tensorrt_llm::kernels::weight_only::KernelType::FP16Int4Groupwise;

    tensorrt_llm::kernels::weight_only::Params params{
        in_ptr1, nullptr, w_ptr1, s_ptr1, z_ptr1, nullptr, out_ptr1, 1.0f, m, n, k, group_size, mCudaKernelType, false};

    int  timing_iterations     = 100;
    auto int4_runner_operation = [&](cudaStream_t stream) {
        int4_runner.gemm(in_ptr2,
                         w_ptr2,
                         s_ptr2,
                         z_ptr2,
                         nullptr,
                         out_ptr2,
                         m,
                         n,
                         k,
                         group_size,
                         int4_config,
                         ws_ptr2,
                         ws_size2,
                         stream);
    };

    float int4_time = timing_function(int4_runner_operation, timing_iterations, stream);
    // printf("m=%d n=%d k=%d fpa_int8_time=%.6f\n", m, n, k, int4_time);
    // tkc::print_config(int4_config);

    auto int4_gemv = [&](cudaStream_t stream) {
        tensorrt_llm::kernels::weight_only::kernel_launcher(mArch, params, stream);
    };
    float gemv_time = timing_function(int4_gemv, timing_iterations, stream);
    float ratio     = gemv_time / int4_time;
    printf("m=%d n=%d k=%d gemv_time=%.6f fpa_int4_time=%.6f ratio=%.6f\n", m, n, k, gemv_time, int4_time, ratio);

    deviceFree(in_ptr1);
    deviceFree(w_ptr1);
    deviceFree(s_ptr1);
    deviceFree(out_ptr1);
    deviceFree(z_ptr1);
    deviceFree(in_ptr2);
    deviceFree(w_ptr2);
    deviceFree(s_ptr2);
    deviceFree(z_ptr2);
    deviceFree(out_ptr2);
    deviceFree(ws_ptr2);
}

int main() {
    std::vector<int>  M_list{1, 2, 3, 4};
    std::vector<Dim2> dim_list;
    dim_list.push_back({4096, 4096});
    dim_list.push_back({4096, 11008});
    dim_list.push_back({4096, 12288});
    dim_list.push_back({4096, 16384});
    dim_list.push_back({5120, 5120});
    dim_list.push_back({5120, 13696});
    dim_list.push_back({5120, 15360});
    dim_list.push_back({5120, 20480});
    dim_list.push_back({6144, 6144});
    dim_list.push_back({6144, 6400});
    dim_list.push_back({6144, 24576});
    dim_list.push_back({11008, 4096});
    dim_list.push_back({13696, 5120});
    dim_list.push_back({16384, 4096});
    dim_list.push_back({20480, 5120});
    dim_list.push_back({24576, 6144});

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (auto dim : dim_list) {
        for (auto m : M_list) {
            gemm_test(m, dim, stream);
        }
    }
    return 0;
}
