#pragma once

#include "navi/engine/Data.h"
#include <memory>
#include <string>

namespace grpc {
class ChannelInterface;
class ClientContext;
}

namespace navi_sched {

inline const std::string FLEXLB_REQUEST_TYPE = "NaviSchedFlexlbRequest";
inline const std::string FLEXLB_RESPONSE_TYPE = "NaviSchedFlexlbResponse";
inline const std::string FLEXLB_REQUEST_NAME = "navi_sched.flexlb_request";

class FlexlbScheduleRequestData : public navi::Data {
public:
    FlexlbScheduleRequestData();
    std::string payload;
    std::shared_ptr<grpc::ChannelInterface> channel;
    std::shared_ptr<grpc::ClientContext> context;
};

class FlexlbScheduleResponseData : public navi::Data {
public:
    FlexlbScheduleResponseData();
    std::string payload;
    int32_t statusCode = 0;
    std::string errorMessage;
};

}
