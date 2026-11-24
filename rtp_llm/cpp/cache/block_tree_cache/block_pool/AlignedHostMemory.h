#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#ifndef RTP_BTC_RANK_STANDALONE
#include <torch/torch.h>
#endif

namespace rtp_llm {

class AlignedHostMemory {
public:
    AlignedHostMemory(size_t usable_bytes, size_t alignment, const std::string& allocation_name);
    ~AlignedHostMemory();

    uint8_t* data() const;
    bool     isRegistered() const;
    size_t   backingBytes() const;

private:
    std::shared_ptr<void> pinned_allocation_;
#ifndef RTP_BTC_RANK_STANDALONE
    torch::Tensor         backing_;
#endif
    uint8_t*              data_{nullptr};
    int                   shared_memory_fd_{-1};
    size_t                backing_bytes_{0};
    bool                  registered_{false};
};

}  // namespace rtp_llm
