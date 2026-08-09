#pragma once

namespace rtp_llm {

constexpr bool groupedCacheReplaySupported(bool using_cuda_backend) {
    return using_cuda_backend;
}

}  // namespace rtp_llm
