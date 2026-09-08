#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <torch/torch.h>

namespace rtp_llm {

class AlignedHostMemory {
public:
    AlignedHostMemory(size_t usable_bytes, size_t alignment, const std::string& allocation_name);

    uint8_t* data() const;

private:
    torch::Tensor backing_;
    uint8_t*      data_{nullptr};
};

}  // namespace rtp_llm
