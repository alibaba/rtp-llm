#include "maga_transformer/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"

#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {
/**************************** NoRdmaMemoryUtilImpl *******************************/

bool NoRdmaMemoryUtilImpl::regUserMr(void* buf, uint64_t size, bool gpu) {
    return true;
}

bool NoRdmaMemoryUtilImpl::deregUserMr(void* buf, bool gpu) {
    return true;
}

bool NoRdmaMemoryUtilImpl::isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted) {
    FT_LOG_DEBUG("tcp mode, no memory actualy regist mr");
    return false;
}

bool NoRdmaMemoryUtilImpl::findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) {
    FT_LOG_DEBUG("tcp mode, no memory actualy regist mr");
    return false;
}

bool NoRdmaMemoryUtilImpl::isRdmaMode() {
    return false;
}

}  // namespace rtp_llm