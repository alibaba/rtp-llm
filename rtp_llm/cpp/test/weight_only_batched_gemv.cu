#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <vector>

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/memory_utils.h"
#include "rtp_llm/cpp/cuda/cutlass/interface.h"

struct Dim2 {
    int k;
    int n;
};

void gemm_test(int m, Dim2 dim2, cudaStream_t stream) {
    int n = dim2.n;
    int k = dim2.k;

    half* in_ptr1 = nullptr;
    deviceMalloc(&in_ptr1, m * k);

    uint8_t* w_ptr1 = nullptr;
    deviceMalloc(&w_ptr1, n * k);

    half* s_ptr1 = nullptr;
    deviceMalloc(&s_ptr1, n);

    half* out_ptr1 = nullptr;
    deviceMalloc(&out_ptr1, m * n, false);
    check_cuda_value(cudaMemset(out_ptr1, 0xdc, m * n * sizeof(half)));

    half* in_ptr2 = nullptr;
    deviceMalloc(&in_ptr2, m * k);

    uint8_t* w_ptr2 = nullptr;
    deviceMalloc(&w_ptr2, n * k);

    half* s_ptr2 = nullptr;
    deviceMalloc(&s_ptr2, n);

    half* out_ptr2 = nullptr;
    deviceMalloc(&out_ptr2, m * n, false);
    check_cuda_value(cudaMemset(out_ptr2, 0xdc, m * n * sizeof(half)));

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    rtp_llm::kernels::WeightOnlyActivationType weight_only_act_type = rtp_llm::kernels::WeightOnlyActivationType::FP16;
    rtp_llm::kernels::WeightOnlyParams         params{reinterpret_cast<const uint8_t*>(w_ptr1),
                                              s_ptr1,
                                              nullptr,
                                              in_ptr1,
                                              nullptr,
                                              out_ptr1,
                                              m,
                                              n,
                                              k,
                                              0,
                                              rtp_llm::kernels::WeightOnlyQuantType::Int8b,
                                              rtp_llm::kernels::WeightOnlyType::PerChannel,
                                              rtp_llm::kernels::WeightOnlyActivationFunctionType::Identity,
                                              weight_only_act_type};
    tensorrt_llm::kernels::cutlass_kernels::
        CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>
            runner;

    char* ws_ptr = nullptr;
    deviceMalloc(&ws_ptr, runner.getWorkspaceSize(m, n, k));
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig config =
        runner.getChosenConfig(in_ptr2,
                               w_ptr2,
                               s_ptr2,
                               nullptr,
                               nullptr,
                               out_ptr2,
                               m,
                               n,
                               k,
                               k,
                               ws_ptr,
                               runner.getWorkspaceSize(m, n, k),
                               stream);

    // warm up

    int   iterations         = 100;
    float total_time_gemv    = 0;
    float total_time_fpaintb = 0;

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaDeviceSynchronize();

    cudaEventSynchronize(start1);
    cudaEventRecord(start1, stream);
    for (int iter = 0; iter < iterations; iter++) {
        weight_only_batched_gemv_launcher(params, stream);
    }
    cudaEventRecord(stop1, stream);
    cudaEventSynchronize(stop1);

    cudaEventElapsedTime(&total_time_gemv, start1, stop1);

    cudaDeviceSynchronize();
    cudaEventSynchronize(start2);
    cudaEventRecord(start2, stream);

    for (int iter = 0; iter < iterations; iter++) {
        runner.gemm(
            in_ptr2, w_ptr2, s_ptr2, out_ptr2, m, n, k, config, ws_ptr, runner.getWorkspaceSize(m, n, k), stream);
    }
    cudaEventRecord(stop2, stream);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&total_time_fpaintb, start2, stop2);

    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    float avg_time_gemv    = total_time_gemv / iterations;
    float avg_time_fpaintb = total_time_fpaintb / iterations;
    printf("m=%d n=%d k=%d batched_gemv=%.6f fpa_intb=%.6f ratio=%f\n",
           m,
           n,
           k,
           avg_time_gemv,
           avg_time_fpaintb,
           avg_time_gemv / avg_time_fpaintb);

    check_cuda_value(status);

    deviceFree(in_ptr1);
    deviceFree(w_ptr1);
    deviceFree(s_ptr1);
    deviceFree(out_ptr1);
    deviceFree(in_ptr2);
    deviceFree(w_ptr2);
    deviceFree(s_ptr2);
    deviceFree(out_ptr2);
    deviceFree(ws_ptr);
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
    dim_list.push_back({3072, 3072});
    dim_list.push_back({3072, 8192});
    dim_list.push_back({3072, 9216});
    dim_list.push_back({8192, 3072});

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (auto m : M_list) {
        for (auto dim : dim_list) {
            gemm_test(m, dim, stream);
        }
    }
    return 0;
}
