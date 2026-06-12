#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>

namespace rtp_llm {

// Abstraction over the physical-memory pause/resume mechanism (freeze/resume M5).
//
// Contract for all implementations:
// - pause(tag) releases the physical GPU pages of every allocation tracked under `tag`
//   while keeping the virtual address range reserved (constraint C2: VA must stay stable,
//   CUDA graphs and C++/Python aliases bake the pointers in).
// - resume(tag) re-maps fresh physical pages at the very same virtual addresses.
//   KV cache uses discard mode: the content after resume is undefined (constraint C3),
//   metadata reset is handled by the caller.
class PhysicalMemoryBackend {
public:
    virtual ~PhysicalMemoryBackend() = default;

    virtual bool        isAvailable() const = 0;
    virtual std::string name() const        = 0;

    virtual bool pause(const std::string& tag)  = 0;
    virtual bool resume(const std::string& tag) = 0;
};

// Backend backed by torch_memory_saver's LD_PRELOAD hook shim
// (torch_memory_saver_hook_mode_preload*.so, validated by spike S1).
//
// The shim intercepts cudaMalloc/cudaFree process-wide and exports a C API. We do not link
// against it: the symbols are probed at runtime via dlsym(RTLD_DEFAULT). If the process was
// not started with the shim preloaded, the backend reports isAvailable() == false.
//
// Probed symbols (torch_memory_saver >= 0.0.9 C API):
//   void tms_pause(const char* tag);                 // tag == nullptr -> all tags
//   void tms_resume(const char* tag);
//   void tms_set_current_tag(const char* tag);
//   void tms_set_interesting_region(bool);
//   bool tms_get_interesting_region();
//   void tms_set_enable_cpu_backup(bool);           // required when enable_cpu_backup=true
// pause/resume are mandatory for availability; the region/tag symbols enable tagging
// future allocations (see beginAllocationRegion/endAllocationRegion).
class TmsBackend: public PhysicalMemoryBackend {
public:
    TmsBackend();

    bool        isAvailable() const override;
    std::string name() const override;

    bool pause(const std::string& tag) override;
    bool resume(const std::string& tag) override;

    // Allocation-region scoping: cudaMalloc calls issued between begin/end are tracked by
    // torch_memory_saver under `tag` and become pause/resume-able. Integration code wraps the
    // BlockPool big-buffer allocation (torch::empty(kCUDA)) with these so the KV buffer is hooked.
    bool beginAllocationRegion(const std::string& tag, bool enable_cpu_backup = false);
    bool endAllocationRegion();

private:
    using TmsTagFn     = void (*)(const char*);
    using TmsSetBoolFn = void (*)(bool);

    TmsTagFn     pause_fn_                  = nullptr;
    TmsTagFn     resume_fn_                 = nullptr;
    TmsTagFn     set_current_tag_fn_        = nullptr;
    TmsSetBoolFn set_interesting_region_fn_ = nullptr;
    TmsSetBoolFn set_enable_cpu_backup_fn_  = nullptr;
};

// Controls the physical memory backing the KV cache big buffer (freeze/resume M5).
//
// Attach mode: the buffer itself is allocated by BlockPool via torch::empty(kCUDA) (which the
// preload shim intercepts at allocation time); the controller only records base_ptr/size and
// drives pause/resume through the injected backend, by tag.
//
// Invariants:
// - basePtr() never changes across pause/resume (VA stability, constraint C2).
// - pause/resume are idempotent: re-pausing while paused (or re-resuming while running) is a
//   no-op that does not hit the backend and returns true.
// - The caller (M1 FreezeLifecycleController sequence) guarantees the engine is drained before
//   pausePhysicalMemory(); after resumePhysicalMemory() the KV content is garbage and the
//   caller must reset KV metadata (BlockPool::resetMetadata + BlockCache::clear).
class KVCachePhysicalMemoryController {
public:
    static constexpr const char* kDefaultTag = "kv_cache";

    explicit KVCachePhysicalMemoryController(std::shared_ptr<PhysicalMemoryBackend> backend,
                                             std::string                            tag = kDefaultTag);

    // Attach an externally allocated buffer; returns the stable base pointer (VA).
    // Re-attaching the same {ptr, size} is a no-op; attaching while paused or attaching a
    // different buffer over an existing one fails (returns nullptr).
    void* allocateOrAttach(void* base_ptr, size_t size_bytes);

    // Release physical pages, keep VA. Requires an attached buffer and an available backend.
    bool pausePhysicalMemory();
    // Re-map physical pages at the same VA. Content is undefined afterwards (discard mode).
    bool resumePhysicalMemory();

    bool   isPaused() const;
    void*  basePtr() const;
    size_t totalSizeBytes() const;

    bool               backendAvailable() const;
    const std::string& tag() const {
        return tag_;
    }

private:
    std::shared_ptr<PhysicalMemoryBackend> backend_;
    std::string                            tag_;

    mutable std::mutex mutex_;
    void*              base_ptr_         = nullptr;
    size_t             total_size_bytes_ = 0;
    bool               paused_           = false;
};

using KVCachePhysicalMemoryControllerPtr = std::shared_ptr<KVCachePhysicalMemoryController>;

}  // namespace rtp_llm
