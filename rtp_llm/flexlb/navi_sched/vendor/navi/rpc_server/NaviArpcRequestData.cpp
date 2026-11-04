#include "navi/rpc_server/NaviArpcRequestData.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

const std::string NaviArpcRequestData::TYPE_ID = "navi.arpc_request";

NaviArpcRequestData::NaviArpcRequestData(
        const google::protobuf::Message *request,
        const std::shared_ptr<google::protobuf::Arena> &arena,
        google::protobuf::Message *needDeleteRequest,
        const std::string &methodName,
        const std::string &clientIp,
        RpcProtocolType type)
    : Data(TYPE_ID)
    , _arena(arena)
    , _request(request)
    , _needDeleteRequest(needDeleteRequest)
    , _methodName(methodName)
    , _clientIp(clientIp)
    , _rpcProtocolType(type)
{
}

NaviArpcRequestData::~NaviArpcRequestData() {
    if (_needDeleteRequest) {
        multi_call::freeProtoMessage(_needDeleteRequest);
    }
}

const google::protobuf::Message *NaviArpcRequestData::getRequest() const {
    if (_request) {
        return _request;
    } else {
        return _needDeleteRequest;
    }
}

NaviArpcRequestType::NaviArpcRequestType()
    : Type(NAVI_TYPE_PARAM, NaviArpcRequestData::TYPE_ID)
{
}

NaviArpcRequestType::~NaviArpcRequestType() {
}

navi::TypeErrorCode
NaviArpcRequestType::serialize(navi::TypeContext &ctx,
                               const navi::DataPtr &data) const {
    auto *requestData = dynamic_cast<NaviArpcRequestData *>(data.get());
    if (!requestData) {
        NAVI_KERNEL_LOG(ERROR, "navi arpc request data is null");
        return navi::TEC_NULL_DATA;
    }
    auto &dataBuffer = ctx.getDataBuffer();
    auto requestPb = requestData->getRequest();
    if (requestPb) {
        dataBuffer.write(true);
        dataBuffer.write(requestPb->GetTypeName());
        dataBuffer.write(requestPb->SerializeAsString());
    } else {
        dataBuffer.write(false);
    }
    dataBuffer.write(requestData->getMethodName());
    dataBuffer.write(requestData->getClientIp());
    dataBuffer.write(requestData->isAiosDebug());
    dataBuffer.write((int)requestData->getRpcProtocolType());
    return navi::TEC_NONE;
}

navi::TypeErrorCode
NaviArpcRequestType::deserialize(navi::TypeContext &ctx,
                                 navi::DataPtr &data) const {
    auto &dataBuffer = ctx.getDataBuffer();
    bool hasRequest = false;
    dataBuffer.read(hasRequest);
    google::protobuf::Message *requestPb = nullptr;
    if (hasRequest) {
        std::string typeName;
        dataBuffer.read(typeName);
        autil::StringView valueStr;
        dataBuffer.read(valueStr);
        auto message = parseProtobufMesasge(typeName, valueStr);
        if (!message) {
            return TEC_FAILED;
        }
        requestPb = message;
    }
    std::string methodName;
    std::string clientIp;
    bool aiosDebug = false;
    int rpcProtocolType = 0;
    dataBuffer.read(methodName);
    dataBuffer.read(clientIp);
    dataBuffer.read(aiosDebug);
    dataBuffer.read(rpcProtocolType);
    auto requestData = std::make_shared<NaviArpcRequestData>(
        nullptr, nullptr, requestPb, methodName, clientIp, (RpcProtocolType)rpcProtocolType);
    requestData->setAiosDebug(aiosDebug);
    data = requestData;
    return navi::TEC_NONE;
}

google::protobuf::Message *
NaviArpcRequestType::parseProtobufMesasge(const std::string &typeName,
                                          const autil::StringView &valueStr) {
    auto desc = google::protobuf::DescriptorPool::generated_pool()
                    ->FindMessageTypeByName(typeName);
    if (!desc) {
        NAVI_KERNEL_LOG(ERROR, "find protobuf type [%s] descriptor failed",
                        typeName.c_str());
        return nullptr;
    }
    auto message = google::protobuf::MessageFactory::generated_factory()
                       ->GetPrototype(desc)
                       ->New();
    if (!message) {
        NAVI_KERNEL_LOG(ERROR, "new protobuf message failed, type [%s]",
                        typeName.c_str());
        return nullptr;
    }
    if (!message->ParseFromArray(valueStr.data(), valueStr.size())) {
        NAVI_KERNEL_LOG(ERROR, "parse protobuf message failed, type [%s]",
                        typeName.c_str());
        delete message;
        return nullptr;
    }
    return message;
}

REGISTER_BUILDIN_TYPE(NaviArpcRequestType);

}
