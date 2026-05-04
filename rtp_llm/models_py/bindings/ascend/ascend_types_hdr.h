#pragma once

#if USING_ASCEND
#include <acl/acl.h>

namespace rtp_llm {
namespace ascend {

using ascendStream_t = aclrtStream;
using ascendEvent_t  = aclrtEvent;

template<typename T>
void check(T result, const char* const file, int const line);

void syncAndCheckInDebug(const char* const file, int const line);

}  // namespace ascend
}  // namespace rtp_llm

#define ASCEND_CHECK(val) rtp_llm::ascend::check((val), __FILE__, __LINE__)
#define ASCEND_CHECK_ERROR() rtp_llm::ascend::syncAndCheckInDebug(__FILE__, __LINE__)

#endif  // USING_ASCEND
