#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/common/kernels/debug_kernel.h"

#include <cstdio>

namespace rtp_llm {

// Helper function to convert to float (specialized for each type)
#ifdef ENABLE_BF16
__device__ float convert_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}
#endif

__device__ float convert_to_float(__half val) {
    return __half2float(val);
}

__device__ float convert_to_float(float val) {
    return val;
}

__device__ float convert_to_float(int val) {
    return float(val);
}

template<typename T>
__global__ void debug_kernel2(T* data, int start_row, int start_col, int m, int n, int row_len, int info_id) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("debug_kernel2 start: %d\n", info_id);
        for (int i = start_row; i < start_row + m; i++) {
            for (int j = start_col; j < start_col + n; j++) {
                int   index = i * row_len + j;
                float value = convert_to_float(data[index]);
                printf("%f ", value);
            }
            printf("\n");
        }
        printf("debug_kernel2 end: %d\n", info_id);
    }
}

template<typename T>
void invoke_debug_kernel2(
    T* data, int start_row, int start_col, int m, int n, int row_len, int info_id, cudaStream_t stream) {
    debug_kernel2<<<1, 1, 0, stream>>>(data, start_row, start_col, m, n, row_len, info_id);
}

#define INSTANTIATEDEBUGKERNEL2(T)                                                                                     \
    template void invoke_debug_kernel2(                                                                                \
        T* data, int start_row, int start_col, int m, int n, int row_len, int info_id, cudaStream_t stream)
INSTANTIATEDEBUGKERNEL2(float);
INSTANTIATEDEBUGKERNEL2(half);
INSTANTIATEDEBUGKERNEL2(int);
#ifdef ENABLE_BF16
INSTANTIATEDEBUGKERNEL2(__nv_bfloat16);
#endif

}  // namespace rtp_llm
