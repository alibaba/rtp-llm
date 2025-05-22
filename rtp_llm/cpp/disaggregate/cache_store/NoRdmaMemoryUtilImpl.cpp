#include "rtp_llm/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
/**************************** NoRdmaMemoryUtilImpl *******************************/

bool NoRdmaMemoryUtilImpl::regUserMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size) {
    return true;
}

bool NoRdmaMemoryUtilImpl::deregUserMr(void* buf, bool gpu) {
    return true;
}

bool NoRdmaMemoryUtilImpl::isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted) {
    RTP_LLM_LOG_DEBUG("tcp mode, no memory actualy regist mr");
    return false;
}

bool NoRdmaMemoryUtilImpl::findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) {
    RTP_LLM_LOG_DEBUG("tcp mode, no memory actualy regist mr");
    return false;
}

bool NoRdmaMemoryUtilImpl::isRdmaMode() {
    return false;
}

}  // namespace rtp_llm