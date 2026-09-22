#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostMemoryRegistration.h"

namespace rtp_llm {

bool hostMemoryRegistrationSupported() {
    return false;
}

bool registerHostMemory(void*, size_t, std::string& error_message) {
    error_message = "host memory registration is not implemented for this device backend";
    return false;
}

bool unregisterHostMemory(void*, std::string& error_message) {
    error_message = "host memory registration is not implemented for this device backend";
    return false;
}

}  // namespace rtp_llm
