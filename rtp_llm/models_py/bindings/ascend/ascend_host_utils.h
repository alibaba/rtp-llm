#pragma once

#if USING_ASCEND
#include <acl/acl.h>
#include "ascend_types_hdr.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <cstddef>
#include <tuple>
#include <string>

namespace rtp_llm {
namespace ascend {

int  getDevice();
int  getDeviceCount();
int  currentDeviceId();
std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm);
int  getMultiProcessorCount(int device_id = -1);
void setAscendGraphCaptureEnabled(bool enabled);
bool isAscendGraphCaptureEnabled();

}  // namespace ascend
}  // namespace rtp_llm

#endif  // USING_ASCEND
