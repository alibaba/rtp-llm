#include <thrust/device_vector.h>
#include <cstddef>
#include <vector>
#include "rtp_llm/cpp/kernels/copy_utils.h"

namespace rtp_llm {

__global__ void multiCopyKernel(const char** src_ptrs, char** dst_ptrs, const size_t* copy_sizes, int num_blocks) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks)
        return;

    const char*  src  = src_ptrs[block_idx];
    char*        dst  = dst_ptrs[block_idx];
    const size_t size = copy_sizes[block_idx];

    for (size_t offset = threadIdx.x; offset < size; offset += blockDim.x) {
        dst[offset] = src[offset];
    }
}

void InvokeMultiCopyKernel(const std::vector<void*>&  h_src_ptrs,
                           const std::vector<void*>&  h_dst_ptrs,
                           const std::vector<size_t>& h_copy_sizes,
                           cudaStream_t               stream) {
    const int num_blocks = h_copy_sizes.size();

    const int threads_per_block = 256;
    dim3      grid(num_blocks);
    dim3      block(threads_per_block);

    thrust::device_vector<void*>  d_src_ptrs(h_src_ptrs.begin(), h_src_ptrs.end());
    thrust::device_vector<void*>  d_dst_ptrs(h_dst_ptrs.begin(), h_dst_ptrs.end());
    thrust::device_vector<size_t> d_copy_sizes(h_copy_sizes.begin(), h_copy_sizes.end());

    multiCopyKernel<<<grid, block, 0, stream>>>((const char**)thrust::raw_pointer_cast(d_src_ptrs.data()),
                                                (char**)thrust::raw_pointer_cast(d_dst_ptrs.data()),
                                                thrust::raw_pointer_cast(d_copy_sizes.data()),
                                                num_blocks);
}

void InvokeMultiMergeCopyKernel(const void*                h_dst_ptr,
                                const std::vector<void*>&  h_src_ptrs,
                                const std::vector<size_t>& h_copy_sizes,
                                const std::vector<size_t>& h_begin_offsets,
                                cudaStream_t               stream) {
    const int num_blocks = h_src_ptrs.size();

    const int threads_per_block = 256;
    dim3      grid(num_blocks);
    dim3      block(threads_per_block);

    std::vector<void*> h_dst_ptrs;
    for (size_t i = 0; i < h_src_ptrs.size(); i++) {
        h_dst_ptrs.push_back((char*)h_dst_ptr + h_begin_offsets[i]);
    }

    thrust::device_vector<void*>  d_src_ptrs(h_src_ptrs.begin(), h_src_ptrs.end());
    thrust::device_vector<void*>  d_dst_ptrs(h_dst_ptrs.begin(), h_dst_ptrs.end());
    thrust::device_vector<size_t> d_copy_sizes(h_copy_sizes.begin(), h_copy_sizes.end());

    multiCopyKernel<<<grid, block, 0, stream>>>((const char**)thrust::raw_pointer_cast(d_src_ptrs.data()),
                                                (char**)thrust::raw_pointer_cast(d_dst_ptrs.data()),
                                                thrust::raw_pointer_cast(d_copy_sizes.data()),
                                                num_blocks);
}

}  // namespace rtp_llm
