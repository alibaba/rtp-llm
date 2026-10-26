#include "aios/network/rdma/accl/unittest/FakeRdmaRequest.h"

ARPC_BEGIN_NAMESPACE(arpc)

std::shared_ptr<RdmaMessage> FakeRdmaRequest::Encode() {
    if (_value.empty()) {
        return std::shared_ptr<RdmaMessage>();
    }
    auto message = std::make_shared<RdmaMessage>();
    message->SetRequestId(_requestId);
    message->SetConnection(_connection);
    message->SetMempool(_mempool);

    auto buffer = message->AllocateSharedSendBuffer(_value.size());
    if (!buffer) {
        return std::shared_ptr<RdmaMessage>();
    }
    memcpy(buffer.get(), _value.c_str(), _value.size());

    message->AddIov(buffer, _value.size(), RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    return message;
}

bool FakeRdmaRequest::Decode(const std::shared_ptr<RdmaMessage>& message) {
    auto buf    = message->GetIovBuffer(0);
    auto bufLen = message->GetIovLen(0);
    if (!buf || bufLen == 0) {
        return false;
    }
    _value      = std::string((const char*)buf.get(), bufLen);
    _connection = message->GetConnection();
    _requestId  = message->GetRequestId();
    _mempool    = message->GetMempool();
    return true;
}

ARPC_END_NAMESPACE(arpc)