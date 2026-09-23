#pragma once

#include <cstddef>
#include <string>

namespace rtp_llm {

// Experimental and opt-in. Non-SCR allocations must keep their existing path.
bool useScrHostCacheShm();

// A named /dev/shm mapping. CUDA registration belongs to the tensor owner and
// MUST be released before this object. Keep the name linked until destruction:
// unlinking a live mapping would make CRIU treat it as a ghost file.
class HostCacheShm {
public:
    explicit HostCacheShm(size_t size_bytes);
    ~HostCacheShm();
    HostCacheShm(const HostCacheShm&)            = delete;
    HostCacheShm& operator=(const HostCacheShm&) = delete;

    void* data() const {
        return data_;
    }
    size_t size() const {
        return size_bytes_;
    }
    const std::string& name() const {
        return name_;
    }
    std::string path() const {
        return "/dev/shm" + name_;
    }

private:
    std::string name_;
    size_t      size_bytes_;
    void*       data_ = nullptr;
};

}  // namespace rtp_llm
