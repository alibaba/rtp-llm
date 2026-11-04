#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCMessageSerializable.h"

ARPC_BEGIN_NAMESPACE(arpc);

// 用于处理请求和响应已经是序列化之后的字节流格式的消息的序列化和反序列化, 如 jgig 请求
class RPCRawMessageSerializable : public RPCMessageSerializable {
public:
    RPCRawMessageSerializable(RPCMessage *header, std::string *body);
    ~RPCRawMessageSerializable() override {}

private:
    RPCRawMessageSerializable(const RPCRawMessageSerializable &);
    RPCRawMessageSerializable &operator=(const RPCRawMessageSerializable &);

public:
    bool serialize(anet::DataBuffer *outputBuffer) const override;
    bool deserialize(anet::DataBuffer *inputBuffer, int length = 0) override;
    std::string *getBody() const { return _body; }

private:
    bool serializeRawMessage(const std::string *message, anet::DataBuffer *outputBuffer) const;
    bool deserializeRawMessage(anet::DataBuffer *inputBuffer, std::string *message, int32_t &leftPacketDataLen) const;

private:
    std::string *_body{nullptr};
};

ARPC_END_NAMESPACE(arpc);
