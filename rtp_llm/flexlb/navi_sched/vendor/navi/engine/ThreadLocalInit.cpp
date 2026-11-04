#include "navi/engine/ThreadLocalInit.h"
#include "navi/log/NaviLogger.h"
#include "navi/config/LogConfig.h"
#include <multi_call/common/common.h>


namespace navi {

void ThreadLocalInit::init() {
    {
        NaviLogger::getTlsBuffer(LogConfig().maxMessageLength);
    }
    {
        multi_call::convertProtocolTypeToStr(multi_call::MC_PROTOCOL_GRPC_STREAM);
        multi_call::convertRequestTypeToStr(multi_call::RT_NORMAL);
    }
}

}
