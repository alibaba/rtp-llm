#include "rtp_llm/cpp/cache/KVCachePhysicalMemoryController.h"

#include <dlfcn.h>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// ---------------------------------------------------------------------------
// VmmBackend
// ---------------------------------------------------------------------------

namespace {

template<typename FnType>
FnType probeSymbol(const char* symbol_name) {
    // The torch_memory_saver shim is injected via LD_PRELOAD, so its exports live in the
    // global symbol namespace and are reachable through RTLD_DEFAULT without linking.
    return reinterpret_cast<FnType>(dlsym(RTLD_DEFAULT, symbol_name));
}

}  // namespace

VmmBackend::VmmBackend() {
    pause_fn_                  = probeSymbol<ShimTagFn>("tms_pause");
    resume_fn_                 = probeSymbol<ShimTagFn>("tms_resume");
    set_current_tag_fn_        = probeSymbol<ShimTagFn>("tms_set_current_tag");
    set_interesting_region_fn_ = probeSymbol<ShimSetBoolFn>("tms_set_interesting_region");
    set_enable_cpu_backup_fn_  = probeSymbol<ShimSetBoolFn>("tms_set_enable_cpu_backup");

    if (isAvailable()) {
        RTP_LLM_LOG_INFO("VmmBackend: torch_memory_saver VMM preload shim detected "
                         "(tms_pause/tms_resume resolved, region scoping %s)",
                         (set_current_tag_fn_ && set_interesting_region_fn_) ? "available" : "unavailable");
    } else {
        RTP_LLM_LOG_INFO("VmmBackend: torch_memory_saver VMM preload shim not detected, backend unavailable");
    }
}

bool VmmBackend::isAvailable() const {
    return pause_fn_ != nullptr && resume_fn_ != nullptr;
}

std::string VmmBackend::name() const {
    return "vmm";
}

bool VmmBackend::pause(const std::string& tag) {
    if (!isAvailable()) {
        RTP_LLM_LOG_ERROR("VmmBackend::pause(tag=%s) failed: shim not available", tag.c_str());
        return false;
    }
    pause_fn_(tag.empty() ? nullptr : tag.c_str());
    return true;
}

bool VmmBackend::resume(const std::string& tag) {
    if (!isAvailable()) {
        RTP_LLM_LOG_ERROR("VmmBackend::resume(tag=%s) failed: shim not available", tag.c_str());
        return false;
    }
    resume_fn_(tag.empty() ? nullptr : tag.c_str());
    return true;
}

bool VmmBackend::beginAllocationRegion(const std::string& tag, bool enable_cpu_backup) {
    if (!set_current_tag_fn_ || !set_interesting_region_fn_) {
        RTP_LLM_LOG_ERROR("VmmBackend::beginAllocationRegion(tag=%s) failed: region symbols not available",
                          tag.c_str());
        return false;
    }
    if (enable_cpu_backup && !set_enable_cpu_backup_fn_) {
        RTP_LLM_LOG_ERROR("VmmBackend::beginAllocationRegion(tag=%s) failed: cpu backup symbol not available",
                          tag.c_str());
        return false;
    }
    set_current_tag_fn_(tag.c_str());
    if (set_enable_cpu_backup_fn_) {
        set_enable_cpu_backup_fn_(enable_cpu_backup);
    }
    set_interesting_region_fn_(true);
    return true;
}

bool VmmBackend::endAllocationRegion() {
    if (!set_interesting_region_fn_) {
        return false;
    }
    set_interesting_region_fn_(false);
    if (set_current_tag_fn_) {
        set_current_tag_fn_("default");
    }
    if (set_enable_cpu_backup_fn_) {
        set_enable_cpu_backup_fn_(false);
    }
    return true;
}

// ---------------------------------------------------------------------------
// KVCachePhysicalMemoryController
// ---------------------------------------------------------------------------

KVCachePhysicalMemoryController::KVCachePhysicalMemoryController(std::shared_ptr<PhysicalMemoryBackend> backend,
                                                                 std::string                            tag):
    backend_(std::move(backend)), tag_(std::move(tag)) {}

void* KVCachePhysicalMemoryController::allocateOrAttach(void* base_ptr, size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (base_ptr == nullptr || size_bytes == 0) {
        RTP_LLM_LOG_ERROR(
            "KVCachePhysicalMemoryController: attach rejected, invalid buffer ptr=%p size=%zu", base_ptr, size_bytes);
        return nullptr;
    }
    if (paused_) {
        RTP_LLM_LOG_ERROR("KVCachePhysicalMemoryController: attach rejected while paused");
        return nullptr;
    }
    if (base_ptr_ != nullptr) {
        if (base_ptr_ == base_ptr && total_size_bytes_ == size_bytes) {
            return base_ptr_;  // idempotent re-attach
        }
        RTP_LLM_LOG_ERROR("KVCachePhysicalMemoryController: attach rejected, already attached to ptr=%p size=%zu, "
                          "got ptr=%p size=%zu",
                          base_ptr_,
                          total_size_bytes_,
                          base_ptr,
                          size_bytes);
        return nullptr;
    }
    base_ptr_         = base_ptr;
    total_size_bytes_ = size_bytes;
    RTP_LLM_LOG_INFO("KVCachePhysicalMemoryController: attached kv buffer ptr=%p size=%zu tag=%s backend=%s(%s)",
                     base_ptr_,
                     total_size_bytes_,
                     tag_.c_str(),
                     backend_ ? backend_->name().c_str() : "none",
                     backendAvailable() ? "available" : "unavailable");
    return base_ptr_;
}

bool KVCachePhysicalMemoryController::pausePhysicalMemory() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (paused_) {
        RTP_LLM_LOG_INFO("KVCachePhysicalMemoryController: pause skipped, already paused");
        return true;
    }
    if (base_ptr_ == nullptr) {
        RTP_LLM_LOG_ERROR("KVCachePhysicalMemoryController: pause failed, no kv buffer attached");
        return false;
    }
    if (!backend_ || !backend_->isAvailable()) {
        RTP_LLM_LOG_ERROR("KVCachePhysicalMemoryController: pause failed, backend unavailable");
        return false;
    }
    if (!backend_->pause(tag_)) {
        RTP_LLM_LOG_ERROR("KVCachePhysicalMemoryController: backend pause(tag=%s) failed", tag_.c_str());
        return false;
    }
    paused_ = true;
    RTP_LLM_LOG_INFO(
        "KVCachePhysicalMemoryController: kv physical memory paused, ptr=%p size=%zu", base_ptr_, total_size_bytes_);
    return true;
}

bool KVCachePhysicalMemoryController::resumePhysicalMemory() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!paused_) {
        RTP_LLM_LOG_INFO("KVCachePhysicalMemoryController: resume skipped, not paused");
        return true;
    }
    if (!backend_ || !backend_->isAvailable()) {
        RTP_LLM_LOG_ERROR("KVCachePhysicalMemoryController: resume failed, backend unavailable");
        return false;
    }
    if (!backend_->resume(tag_)) {
        RTP_LLM_LOG_ERROR("KVCachePhysicalMemoryController: backend resume(tag=%s) failed", tag_.c_str());
        return false;
    }
    paused_ = false;
    RTP_LLM_LOG_INFO(
        "KVCachePhysicalMemoryController: kv physical memory resumed, ptr=%p size=%zu", base_ptr_, total_size_bytes_);
    return true;
}

bool KVCachePhysicalMemoryController::isPaused() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return paused_;
}

void* KVCachePhysicalMemoryController::basePtr() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return base_ptr_;
}

size_t KVCachePhysicalMemoryController::totalSizeBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_size_bytes_;
}

bool KVCachePhysicalMemoryController::backendAvailable() const {
    return backend_ && backend_->isAvailable();
}

}  // namespace rtp_llm
