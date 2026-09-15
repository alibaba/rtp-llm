#include "rtp_llm/models_py/bindings/cuda/SleepMemoryAllocator.h"

#include <dlfcn.h>
#include <sys/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {
struct RegionApi {
    using SetTag = void (*)(const char*);
    using SetBool = void (*)(bool);
    using GetBool = bool (*)();
    SetTag set_tag = reinterpret_cast<SetTag>(dlsym(RTLD_DEFAULT, "tms_set_current_tag"));
    SetBool set_region = reinterpret_cast<SetBool>(dlsym(RTLD_DEFAULT, "tms_set_interesting_region"));
    GetBool get_region = reinterpret_cast<GetBool>(dlsym(RTLD_DEFAULT, "tms_get_interesting_region"));
    SetBool set_backup = reinterpret_cast<SetBool>(dlsym(RTLD_DEFAULT, "tms_set_enable_cpu_backup"));
    GetBool get_backup = reinterpret_cast<GetBool>(dlsym(RTLD_DEFAULT, "tms_get_enable_cpu_backup"));

    bool available() const {
        return set_tag && set_region && get_region && set_backup && get_backup;
    }
};

const RegionApi& regionApi() {
    static const RegionApi api;
    return api;
}

class ScratchRegion {
public:
    explicit ScratchRegion(bool backup): api_(regionApi()) {
        TORCH_CHECK(api_.available(), "sleep scratch allocator requires the torch_memory_saver preload shim");
        // Preserve an enclosing weights or CUDA-graph capture region verbatim.
        active_ = !api_.get_region();
        if (active_) {
            old_backup_ = api_.get_backup();
            api_.set_tag("weights");
            api_.set_backup(backup);
            api_.set_region(true);
        }
    }
    ~ScratchRegion() {
        if (active_) {
            api_.set_region(false);
            api_.set_tag("default");
            api_.set_backup(old_backup_);
        }
    }
private:
    const RegionApi& api_;
    bool active_{false};
    bool old_backup_{false};
};

void* allocateScratch(ssize_t size, int device, bool backup) {
    const c10::cuda::CUDAGuard guard(device);
    ScratchRegion region(backup);
    void* pointer = nullptr;
    // The preload shim supplies fixed-VA, pausable backing. PyTorch's enclosing
    // custom MemPool still owns caching, stream recording and delayed frees;
    // its allocations do not use the default pool's expandable segments.
    const auto error = cudaMalloc(&pointer, size);
    if (error == cudaErrorMemoryAllocation) {
        // Let the caching allocator release cached blocks and retry as usual.
        (void)cudaGetLastError();
        return nullptr;
    }
    C10_CUDA_CHECK(error);
    return pointer;
}
}  // namespace

namespace rtp_llm {
bool sleepMemoryAllocatorAvailable() {
    return regionApi().available();
}
}  // namespace rtp_llm

// CUDAPluggableAllocator ABI. These symbols are resolved from librtp_compute_ops
// once; there is no Python callback on allocation or free.
extern "C" __attribute__((visibility("default"))) void* rtp_sleep_scratch_malloc(ssize_t size, int device, cudaStream_t) {
    return allocateScratch(size, device, false);
}
extern "C" __attribute__((visibility("default"))) void* rtp_sleep_scratch_malloc_backup(ssize_t size, int device, cudaStream_t) {
    return allocateScratch(size, device, true);
}
extern "C" __attribute__((visibility("default"))) void rtp_sleep_scratch_free(void* pointer, ssize_t, int device, cudaStream_t) {
    const c10::cuda::CUDAGuard guard(device);
    C10_CUDA_CHECK(cudaFree(pointer));
}
