#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <torch/torch.h>

namespace rtp_llm {

class AlignedHostMemory {
public:
    AlignedHostMemory(size_t usable_bytes, size_t alignment, const std::string& allocation_name);
    ~AlignedHostMemory();

    uint8_t* data() const;
    bool     isRegistered() const;
    size_t   backingBytes() const;

private:
    torch::Tensor backing_;
    uint8_t*      data_{nullptr};
    int           shared_memory_fd_{-1};
    size_t        backing_bytes_{0};
    bool          registered_{false};
};

}  // namespace rtp_llm
