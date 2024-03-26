#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <vector>

#include "src/fastertransformer/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/cuda/memory_utils.h"

using namespace fastertransformer;

struct Dim2 {
    int k;
    int n;
};

void gemm_test(int m, Dim2 dim2, cudaStream_t stream)
{
    int n = dim2.n;
    int k = dim2.k;

    half* in_ptr = nullptr;
    deviceMalloc(&in_ptr, m * k);

    uint8_t* w_ptr = nullptr;
    deviceMalloc(&w_ptr, n * k);

    half* s_ptr = nullptr;
    deviceMalloc(&s_ptr, n);

    half* out_ptr = nullptr;
    deviceMalloc(&out_ptr, m * n, false);
    check_cuda_error(cudaMemset(out_ptr, 0xdc, m * n * sizeof(half)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    tensorrt_llm::kernels::cutlass_kernels::
        CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>
            runner;
    char*   ws_ptr = nullptr;

    const int ws_size = runner.getWorkspaceSize(m, n, k);
    deviceMalloc(&ws_ptr, ws_size);
    const auto bestTactic = runner.getChosenConfig(reinterpret_cast<const void*>(in_ptr),
                                                   reinterpret_cast<const void*>(w_ptr),
                                                   reinterpret_cast<const void*>(s_ptr),
                                                   nullptr,
                                                   nullptr,
                                                   reinterpret_cast<void*>(out_ptr),
                                                   m,
                                                   n,
                                                   k,
                                                   k,
                                                   reinterpret_cast<char*>(ws_ptr),
                                                   ws_size,
                                                   stream);

    FT_CHECK_WITH_INFO(
        &bestTactic,
        "No valid weight only per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
        "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
        "engine.)");

    runner.gemm(reinterpret_cast<const void*>(in_ptr),
                reinterpret_cast<const void*>(w_ptr),
                reinterpret_cast<const void*>(s_ptr),
                reinterpret_cast<void*>(out_ptr),
                m,
                n,
                k,
                bestTactic,
                reinterpret_cast<char*>(ws_ptr),
                ws_size,
                stream);

    cudaEventRecord(start, stream);

    int iterations = 10;
    for (int iter = 0; iter < iterations; iter++) {
            runner.gemm(reinterpret_cast<const void*>(in_ptr),
                reinterpret_cast<const void*>(w_ptr),
                reinterpret_cast<const void*>(s_ptr),
                reinterpret_cast<void*>(out_ptr),
                m,
                n,
                k,
                bestTactic,
                reinterpret_cast<char*>(ws_ptr),
                ws_size,
                stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    float avg_time = total_time_ms / float(iterations);
    printf("m=%d n=%d k=%d time=%.6f\n", m, n, k, avg_time);

    sync_check_cuda_error();

    deviceFree(in_ptr);
    deviceFree(w_ptr);
    deviceFree(s_ptr);
    deviceFree(out_ptr);
    deviceFree(ws_ptr);
}

int main()
{
    std::vector<int>  M_list{1, 4, 8, 16, 32, 64, 128};
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

    for (auto m : M_list) {
        for (auto dim : dim_list) {
            gemm_test(m, dim, stream);
        }
    }
    return 0;
}
