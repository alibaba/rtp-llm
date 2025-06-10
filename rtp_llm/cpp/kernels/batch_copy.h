#pragma once

namespace rtp_llm
{
namespace kernels
{

struct BatchCopyConfig {
    bool aligned_copy;
};

BatchCopyConfig getBatchCopyConfig(const size_t * bytes_host, size_t batch_size);

void invokeBatchCopy(void * const* dst, 
                     void const* const* src, 
                     size_t * bytes, 
                     size_t batch_size, 
                     const BatchCopyConfig &config,
                     cudaStream_t stream);

}
}