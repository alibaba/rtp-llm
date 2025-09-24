#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <vector>

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/memory_utils.h"
#include "rtp_llm/cpp/cuda/cublas/cublasMMWrapper.h"
#include "rtp_llm/cpp/cuda/cutlass/interface.h"

#include "rtp_llm/cpp/cuda/allocator_cuda.h"
#include "rtp_llm/cpp/cuda/cublas/cublas.h"

struct Dim2 {
    int k;
    int n;
};

void gemm_test(int m, Dim2 dim2, cudaStream_t stream) {
    int n = dim2.n;
    int k = dim2.k;

    half* in_ptr1 = nullptr;
    deviceMalloc(&in_ptr1, m * k);

    half* w_ptr1 = nullptr;
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

    Allocator<AllocatorType::CUDA>* allocator_ = nullptr;
    std::mutex*                     mutex_     = nullptr;
    mutex_                                     = new std::mutex();  // mutex per process

    cublasAlgoMap* cublas_algo_map_ = nullptr;
    cublas_algo_map_                = new cublasAlgoMap(GEMM_CONFIG);

    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator_->setStream(stream);

    cublasHandle_t cublas_handle_;
    check_cuda_value(cublasCreate(&cublas_handle_));
    cublasLtHandle_t cublaslt_handle_;
    check_cuda_value(cublasLtCreate(&cublaslt_handle_));
    check_cuda_value(cublasSetStream(cublas_handle_, stream));

    cublasMMWrapper* cublas_wrapper_ =
        new cublasMMWrapper(cublas_handle_, cublaslt_handle_, stream, cublas_algo_map_, mutex_, allocator_);

    cublas_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);

    tensorrt_llm::kernels::cutlass_kernels::
        CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>
            runner;
    char*   ws_ptr = nullptr;

    const int ws_size = runner.getWorkspaceSize(m, n, k);
    deviceMalloc(&ws_ptr, ws_size);
    const auto bestTactic = runner.getChosenConfig(reinterpret_cast<const void*>(in_ptr2),
                                                   reinterpret_cast<const void*>(w_ptr2),
                                                   reinterpret_cast<const void*>(s_ptr2),
                                                   nullptr,
                                                   nullptr,
                                                   reinterpret_cast<void*>(out_ptr2),
                                                   m,
                                                   n,
                                                   k,
                                                   k,
                                                   reinterpret_cast<char*>(ws_ptr),
                                                   ws_size,
                                                   stream);

    RTP_LLM_CHECK_WITH_INFO(
        &bestTactic,
        "No valid weight only per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
        "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
        "engine.)");

    int   iterations         = 100;
    float total_time_fp16    = 0;
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
        cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, w_ptr1, n, in_ptr1, k, out_ptr1, n);
    }
    cudaEventRecord(stop1, stream);
    cudaEventSynchronize(stop1);

    cudaEventElapsedTime(&total_time_fp16, start1, stop1);

    cudaDeviceSynchronize();
    cudaEventSynchronize(start2);
    cudaEventRecord(start2, stream);

    for (int iter = 0; iter < iterations; iter++) {
        runner.gemm(reinterpret_cast<const void*>(in_ptr2),
                    reinterpret_cast<const void*>(w_ptr2),
                    reinterpret_cast<const void*>(s_ptr2),
                    reinterpret_cast<void*>(out_ptr2),
                    m,
                    n,
                    k,
                    bestTactic,
                    reinterpret_cast<char*>(ws_ptr),
                    ws_size,
                    stream);
    }
    cudaEventRecord(stop2, stream);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&total_time_fpaintb, start2, stop2);

    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    float avg_time_fp16    = total_time_fp16 / iterations;
    float avg_time_fpaintb = total_time_fpaintb / iterations;
    printf("m=%d n=%d k=%d cublas=%.6f fpa_intb=%.6f ratio=%f\n",
           m,
           n,
           k,
           avg_time_fp16,
           avg_time_fpaintb,
           avg_time_fp16 / avg_time_fpaintb);

    delete cublas_algo_map_;
    delete cublas_wrapper_;
    delete allocator_;
    delete mutex_;

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
    std::vector<int>  M_list{1, 2, 3, 4, 1024, 2048, 4096, 16384};
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
