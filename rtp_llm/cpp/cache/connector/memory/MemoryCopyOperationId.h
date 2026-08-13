#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace rtp_llm {

class MemoryCopyOperationIdGenerator {
public:
    MemoryCopyOperationIdGenerator();

    MemoryCopyOperationIdGenerator(const MemoryCopyOperationIdGenerator&)            = delete;
    MemoryCopyOperationIdGenerator& operator=(const MemoryCopyOperationIdGenerator&) = delete;

    std::string next();

private:
    const std::string    epoch_;
    std::atomic<uint64_t> counter_{0};
};

}  // namespace rtp_llm
