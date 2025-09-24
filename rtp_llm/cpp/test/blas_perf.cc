#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <vector>

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/memory_utils.h"
#include "rtp_llm/cpp/cuda/allocator_cuda.h"
#include "rtp_llm/cpp/cuda/cublas/cublasMMWrapper.h"

using namespace rtp_llm;

struct Dim2 {
    int k;
    int n;
};

template<typename T>
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

    cublas_wrapper_->setGemmConfig(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F);

    int   iterations      = 3;
    float total_time_fp16 = 0;

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, w_ptr1, n, in_ptr1, k, out_ptr1, n);

    cudaDeviceSynchronize();
    cudaEventSynchronize(start1);
    cudaEventRecord(start1, stream);
    for (int iter = 0; iter < iterations; iter++) {
        cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, w_ptr1, n, in_ptr1, k, out_ptr1, n);
    }
    cudaEventRecord(stop1, stream);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&total_time_fp16, start1, stop1);

    float avg_time_fp16 = total_time_fp16 / float(iterations);
    printf("m=%d n=%d k=%d cublas=%.6f tflops=%.2f\n", m, n, k, avg_time_fp16, 2.0 * m * n * k / 1000 / 1000 / 1000);

    delete cublas_algo_map_;
    delete cublas_wrapper_;
    delete allocator_;
    delete mutex_;

    deviceFree(in_ptr1);
    deviceFree(w_ptr1);
    deviceFree(s_ptr1);
    deviceFree(out_ptr1);
}

int main() {
    std::vector<int>  M_list{27000};
    std::vector<Dim2> dim_list;
    dim_list.push_back({5120, 5120});

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    printf("type = fp16 \n");
    for (auto m : M_list) {
        for (auto dim : dim_list) {
            gemm_test<half>(m, dim, stream);
        }
    }

    return 0;
}
