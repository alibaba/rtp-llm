#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostMemoryRegistration.h"

#include <cuda_runtime.h>

namespace rtp_llm {

bool hostMemoryRegistrationSupported() {
    return true;
}

bool registerHostMemory(void* ptr, size_t size_bytes, std::string& error_message) {
    const auto error = cudaHostRegister(ptr, size_bytes, cudaHostRegisterDefault);
    if (error == cudaSuccess) {
        return true;
    }
    error_message = cudaGetErrorString(error);
    return false;
}

bool unregisterHostMemory(void* ptr, std::string& error_message) {
    const auto error = cudaHostUnregister(ptr);
    if (error == cudaSuccess) {
        return true;
    }
    error_message = cudaGetErrorString(error);
    return false;
}

}  // namespace rtp_llm
