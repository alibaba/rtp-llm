#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

class NoRdmaMemoryUtilImpl: public MemoryUtilBase {
public:
    bool regUserMr(void* buf, uint64_t size, bool gpu) override;
    bool deregUserMr(void* buf, bool gpu) override;
    bool isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted) override;
    bool findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) override;
    bool isRdmaMode() override;

private:
    AUTIL_LOG_DECLARE();
};

}