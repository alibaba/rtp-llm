#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <array>

// Model-local ctypes shim. No torch bindings or framework native ABI changes.
extern "C" int v41_grouped_index_gemm(void* handle,
                                      void* stream,
                                      int count,
                                      const int* rows,
                                      const void* const* weights,
                                      const void* const* inputs,
                                      void* const* outputs) {
    constexpr int max_segments = 1024;
    if (count <= 0 || count > max_segments || !rows || !weights || !inputs || !outputs)
        return CUBLAS_STATUS_INVALID_VALUE;

    std::array<cublasOperation_t, max_segments> transa{}, transb{};
    std::array<int, max_segments> m{}, k{}, lda{}, ldb{}, ldc{}, sizes{};
    std::array<float, max_segments> alpha{}, beta{};
    int total_rows = 0;
    for (int i = 0; i < count; ++i) {
        if (rows[i] <= 0 || rows[i] > 65536 - total_rows)
            return CUBLAS_STATUS_INVALID_VALUE;
        total_rows += rows[i];
        transa[i] = CUBLAS_OP_T;
        transb[i] = CUBLAS_OP_N;
        m[i] = ldc[i] = 128;
        k[i] = lda[i] = ldb[i] = 512;
        sizes[i] = 1;
        alpha[i] = 1.0f;
    }
    auto blas = static_cast<cublasHandle_t>(handle);
    auto status = cublasSetStream(blas, static_cast<cudaStream_t>(stream));
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;
    return cublasGemmGroupedBatchedEx(blas,
                                      transa.data(),
                                      transb.data(),
                                      m.data(),
                                      rows,
                                      k.data(),
                                      alpha.data(),
                                      weights,
                                      CUDA_R_16BF,
                                      lda.data(),
                                      inputs,
                                      CUDA_R_16BF,
                                      ldb.data(),
                                      beta.data(),
                                      outputs,
                                      CUDA_R_16BF,
                                      ldc.data(),
                                      count,
                                      sizes.data(),
                                      CUBLAS_COMPUTE_32F);
}
