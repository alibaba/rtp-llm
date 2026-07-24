#include <cstdio>
#include <cstdlib>
#include <string>

#include <dlfcn.h>

namespace {

[[noreturn]] void fail(const char* message, const char* detail = nullptr) {
    if (detail == nullptr) {
        std::fprintf(stderr, "%s\n", message);
    } else {
        std::fprintf(stderr, "%s: %s\n", message, detail);
    }
    // These production Python extensions own process-lifetime static state.
    // Avoid unloading them from this minimal loader-only test process.
    std::_Exit(EXIT_FAILURE);
}

std::string runfilePath(const char* name) {
    const char* test_srcdir = std::getenv("TEST_SRCDIR");
    const char* workspace   = std::getenv("TEST_WORKSPACE");
    if (test_srcdir == nullptr || workspace == nullptr) {
        fail("Bazel runfiles environment is unavailable");
    }
    return std::string(test_srcdir) + "/" + workspace + "/" + name;
}

void requireSymbol(void* handle, const char* symbol, const char* description) {
    ::dlerror();
    void*       address = ::dlsym(handle, symbol);
    const char* error   = ::dlerror();
    if (address == nullptr || error != nullptr) {
        fail(description, error == nullptr ? "symbol address is null" : error);
    }
}

}  // namespace

int main() {
    const std::string transformer_path = runfilePath("libth_transformer.so");
    void*             transformer      = ::dlopen(transformer_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (transformer == nullptr) {
        fail("failed to load production libth_transformer.so", ::dlerror());
    }

    // tpSyncModelInputs is defined by libth_transformer.so. Its call to
    // execBroadcastCpu crosses into the DT_NEEDED librtp_compute_ops.so; the
    // RTLD_NOW load above fails if that production definition was discarded.
    requireSymbol(transformer,
                  "_ZN7rtp_llm17tpSyncModelInputsERNS_14GptModelInputsERKNS_17ParallelismConfigE",
                  "production tpSyncModelInputs symbol is unavailable");
    requireSymbol(transformer,
                  "_ZN7rtp_llm16execBroadcastCpuERKNS_15BroadcastParamsEb",
                  "production execBroadcastCpu dependency is unavailable");

    std::_Exit(EXIT_SUCCESS);
}
