#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BufferTypes.h"

namespace rtp_llm {

// Native planes keep their maximum shape/stride for attention and CUDA Graph.
// A segment consists of a permanent formal-KV prefix followed by hot pages.
struct OffloadDeviceSegment {
    size_t offset;
    size_t fixed_bytes;
    size_t page_bytes;
};

class OffloadDeviceBuffer final: public OffloadHotCache,
                               public std::enable_shared_from_this<OffloadDeviceBuffer> {
public:
    OffloadDeviceBuffer(size_t bytes,
                        uint32_t max_rows,
                        uint32_t rows_per_page,
                        std::vector<OffloadDeviceSegment> segments);
    ~OffloadDeviceBuffer() override;
    torch::Tensor tensor();
    void resize(uint32_t rows, bool grow_only = false) override;
    size_t committedBytes() const override;
    const torch::Tensor& activeRows() const;
    const torch::Tensor& epoch() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
