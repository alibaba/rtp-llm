#include "navi/rpc_server/NaviArpcResponseData.h"
#include "navi/rpc_server/NaviArpcRequestData.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

const std::string NaviArpcResponseData::TYPE_ID = "navi.arpc_response";

NaviArpcResponseData::NaviArpcResponseData(google::protobuf::Message *response)
    : Data(TYPE_ID)
    , _response(response)
{
}

NaviArpcResponseData::~NaviArpcResponseData() {
    multi_call::freeProtoMessage(_response);
}

google::protobuf::Message *NaviArpcResponseData::stealResponse() {
    auto tmp = _response;
    _response = nullptr;
    return tmp;
}

SummaryErrorCode NaviArpcResponseData::getSummary(const std::string &query, SummaryBuilder &builder) const {
    std::string summaryStr;
    if (!_response) {
        summaryStr = "empty response";
    } else if ("1" == query) {
        summaryStr = _response->GetDescriptor()->DebugString();
    } else {
        summaryStr = _response->DebugString();
    }
    builder.setDesc(summaryStr);
    return SEC_OK;
}

NaviArpcResponseType::NaviArpcResponseType()
    : Type(NAVI_TYPE_PARAM, NaviArpcResponseData::TYPE_ID)
{
}

NaviArpcResponseType::~NaviArpcResponseType() {
}

navi::TypeErrorCode
NaviArpcResponseType::serialize(navi::TypeContext &ctx,
                                const navi::DataPtr &data) const {
    auto *responseData = dynamic_cast<NaviArpcResponseData *>(data.get());
    if (!responseData) {
        NAVI_KERNEL_LOG(ERROR, "navi arpc response data is null");
        return navi::TEC_NULL_DATA;
    }
    auto &dataBuffer = ctx.getDataBuffer();
    const auto *responsePb = responseData->getResponse();
    if (responsePb) {
        dataBuffer.write(true);
        dataBuffer.write(responsePb->GetTypeName());
        dataBuffer.write(responsePb->SerializeAsString());
    } else {
        dataBuffer.write(false);
    }
    return navi::TEC_NONE;
}

navi::TypeErrorCode
NaviArpcResponseType::deserialize(navi::TypeContext &ctx,
                                  navi::DataPtr &data) const {
    auto &dataBuffer = ctx.getDataBuffer();
    bool hasResponse = false;
    dataBuffer.read(hasResponse);
    google::protobuf::Message *responsePb = nullptr;
    if (hasResponse) {
        std::string typeName;
        dataBuffer.read(typeName);
        autil::StringView valueStr;
        dataBuffer.read(valueStr);
        auto message =
            NaviArpcRequestType::parseProtobufMesasge(typeName, valueStr);
        if (!message) {
            return TEC_FAILED;
        }
        responsePb = message;
    }
    data = std::make_shared<NaviArpcResponseData>(responsePb);
    return navi::TEC_NONE;
}

REGISTER_BUILDIN_TYPE(NaviArpcResponseType);

}

