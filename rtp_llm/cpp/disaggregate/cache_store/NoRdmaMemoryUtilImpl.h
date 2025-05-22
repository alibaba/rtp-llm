#pragma once
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

class NoRdmaMemoryUtilImpl: public MemoryUtil {
public:
    bool regUserMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size) override;
    bool deregUserMr(void* buf, bool gpu) override;
    bool isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted) override;
    bool findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) override;
    bool isRdmaMode() override;
};

}  // namespace rtp_llm