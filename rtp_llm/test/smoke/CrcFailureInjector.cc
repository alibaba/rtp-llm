#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <unistd.h>

namespace rtp_llm {

// Test-only LD_PRELOAD interposer. Claiming the arm file corrupts exactly one
// backing across all workers/ranks; the production load still checks and dumps it.
CrcCopyStatus CrcBlockCopyBatch::load(const std::vector<CrcCopyItem>& items, std::vector<CrcCopyFailure>* failures) {
    using Load = CrcCopyStatus (*)(CrcBlockCopyBatch*, const std::vector<CrcCopyItem>&, std::vector<CrcCopyFailure>*);
    void*             caller_address = __builtin_return_address(0);
    static const auto original       = [&] {
        Dl_info caller{};
        if (!dladdr(caller_address, &caller))
            std::abort();
        // Python can load the engine with RTLD_LOCAL, outside RTLD_NEXT's scope.
        void* library = dlopen(caller.dli_fname, RTLD_NOW | RTLD_NOLOAD);
        if (!library)
            std::abort();
        const auto load = reinterpret_cast<Load>(dlsym(
            library,
            "_ZN7rtp_llm17CrcBlockCopyBatch4loadERKSt6vectorINS_11CrcCopyItemESaIS2_EEPS1_INS_14CrcCopyFailureESaIS7_EE"));
        if (!load) {
            std::fprintf(stderr, "CRC smoke cannot resolve production load: %s\n", dlerror());
            std::abort();
        }
        return load;
    }();
    const char* arm = std::getenv("CRC_SMOKE_ARM_FILE");
    if (arm && !items.empty() && std::rename(arm, (std::string(arm) + ".claimed").c_str()) == 0) {
        const auto&    item   = items.front();
        auto*          bytes  = static_cast<uint8_t*>(item.host);
        const unsigned before = bytes[0];
        bytes[0] ^= 1;
        std::ofstream evidence(std::string(arm) + ".injected.json");
        evidence << "{\"pid\":" << getpid() << ",\"address\":" << reinterpret_cast<uintptr_t>(item.host)
                 << ",\"payload_bytes\":" << item.payload_bytes << ",\"before\":" << before
                 << ",\"after\":" << unsigned(bytes[0]) << "}\n";
    }
    return original(this, items, failures);
}

}  // namespace rtp_llm
