#include "src/fastertransformer/utils/serialize_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

void fwrite_gpu(const void *ptr, size_t size, FILE *stream) {
    void *tmp = malloc(size);
    cudaMemcpy(tmp, ptr, size, cudaMemcpyDeviceToHost);
    fwrite(tmp, size, 1, stream);
    free(tmp);
}

void *fread_gpu(size_t size, FILE *stream, size_t allocation_size) {
    if (allocation_size <= 0) {
        allocation_size = size;
    }
    void *tmp = malloc(allocation_size);
    size_t ret_code = fread(tmp, size, 1, stream);
    if (ret_code != size) {
        perror("Error fread gpu memory");
    }
    void *ptr;
    cudaMalloc(&ptr, allocation_size);
    cudaMemcpy(ptr, tmp, size, cudaMemcpyHostToDevice);
    sync_check_cuda_error();

    free(tmp);
    return ptr;
}

};
