#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/RdmaMessage.h"
#include "autil/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

// 支持从RdmaMessage 和 accl::barex::XMessage之间的相互转换
class SimpleMessageConverter {
public:
    bool ConvertFromRdmaMessage(RdmaMessage* message, ::accl::barex::XMessage& xmessage);
    bool ConvertFromXMessage(RdmaMessage* message, const ::accl::barex::XMessage& xmessage);

private:
    bool ConvertUnMrCPUMemory(RdmaMessage& message, ::accl::barex::XMessage& xmessage, size_t index);
    bool ConvertUnMrGPUMemory(RdmaMessage& message, ::accl::barex::XMessage& xmessage, size_t index);

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);