#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <vector>

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/memory_utils.h"
#include "rtp_llm/cpp/cuda/allocator_cuda.h"
#include "rtp_llm/cpp/cuda/cublas/cublas.h"

#include "rtp_llm/cpp/cuda/cublas/cublasMMWrapper.h"
#include "rtp_llm/cpp/cuda/cutlass/interface.h"

namespace tk = tensorrt_llm::common;
namespace tc = tensorrt_llm::cutlass_extensions;
//

struct Dim2 {
    int k;
    int n;
};

template<typename T>
void gemm_test(int m, Dim2 dim2, cudaStream_t stream) {
    int n = dim2.n;
    int k = dim2.k;
    // quantizeWeights quantizeActivations perToken perChannel useInt4Weights useInt8KvCache useFp8KvCache useFp8Qdq
    tk::QuantMode quant_mode = tk::QuantMode::fromDescription(true, true, true, true, false, false, false, false);

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

    cublas_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);

    uint8_t* in_ptr2 = nullptr;
    deviceMalloc(&in_ptr2, m * k);

    uint8_t* w_ptr2 = nullptr;
    deviceMalloc(&w_ptr2, n * k);

    T* out_ptr2 = nullptr;
    deviceMalloc(&out_ptr2, m * n, false);
    check_cuda_value(cudaMemset(out_ptr2, 0xdc, m * n * sizeof(T)));

    float* alphaCol = nullptr;
    check_cuda_value(cudaMalloc((void**)(&alphaCol), sizeof(float)));

    float* alphaRow = nullptr;
    check_cuda_value(cudaMalloc((void**)(&alphaRow), sizeof(float)));

    float Colscale = 1.0f;
    float Rowscale = 1.0f;
    cudaMemcpy(alphaCol, &Colscale, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(alphaRow, &Rowscale, sizeof(float), cudaMemcpyHostToDevice);

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    int   iterations         = 100;
    float total_time_fp16    = 0;
    float total_time_ms_int8 = 0;

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    tensorrt_llm::kernels::cutlass_kernels::CutlassInt8GemmRunner<T> runner;
    char*                                                            ws_ptr = nullptr;
    const int                                                        wsSize = runner.getWorkspaceSize(m, n, k);
    deviceMalloc(&ws_ptr, wsSize);

    // warm up
    const auto gemmConfig = runner.getChosenConfig(
        in_ptr2, w_ptr2, quant_mode, alphaCol, alphaRow, out_ptr2, m, n, k, ws_ptr, wsSize, stream);
    runner.gemm(in_ptr2, w_ptr2, quant_mode, alphaCol, alphaRow, out_ptr2, m, n, k, gemmConfig, ws_ptr, wsSize, stream);
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

    cudaDeviceSynchronize();
    cudaEventSynchronize(start2);
    cudaEventRecord(start2, stream);

    for (int iter = 0; iter < iterations; iter++) {
        runner.gemm(
            in_ptr2, w_ptr2, quant_mode, alphaCol, alphaRow, out_ptr2, m, n, k, gemmConfig, ws_ptr, wsSize, stream);
    }
    cudaEventRecord(stop2, stream);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&total_time_ms_int8, start2, stop2);

    float avg_time_int8 = total_time_ms_int8 / float(iterations);
    float avg_time_fp16 = total_time_fp16 / float(iterations);
    printf("m=%d n=%d k=%d cublas=%.6f w8a8=%.6f ratio=%f\n",
           m,
           n,
           k,
           avg_time_fp16,
           avg_time_int8,
           avg_time_fp16 / avg_time_int8);

    check_cuda_value(status);

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
    deviceFree(out_ptr2);
    deviceFree(ws_ptr);
    cudaFree(alphaRow);
    cudaFree(alphaCol);
}

int main() {
    std::vector<int>  M_list{1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768};
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

    // printf("type = int32 \n");
    // for (auto m : M_list) {
    //     for (auto dim : dim_list) {
    //         gemm_test<int32_t>(m, dim, stream);
    //     }
    // }
    // printf("type = bf16 \n");
    // for (auto m : M_list) {
    //     for (auto dim : dim_list) {
    //         gemm_test<__nv_bfloat16>(m, dim, stream);
    //     }
    // }
    printf("type = fp16 \n");
    for (auto m : M_list) {
        for (auto dim : dim_list) {
            gemm_test<half>(m, dim, stream);
        }
    }
    // printf("type = fp32 \n");
    // for (auto m : M_list) {
    //     for (auto dim : dim_list) {
    //         gemm_test<float>(m, dim, stream);
    //     }
    // }

    return 0;
}
