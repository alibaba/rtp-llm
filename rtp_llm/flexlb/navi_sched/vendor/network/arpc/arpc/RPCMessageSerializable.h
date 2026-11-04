#pragma once
#include <memory>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databufferserializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"
#include "google/protobuf/arena.h"

ARPC_BEGIN_NAMESPACE(arpc);

class RPCMessageSerializable : public anet::DataBufferSerializable {
public:
    RPCMessageSerializable(RPCMessage *header, RPCMessage *body, const std::shared_ptr<google::protobuf::Arena> &arena);
    ~RPCMessageSerializable();

public:
    RPCMessage *getHeader() { return _header; }
    RPCMessage *getBody() { return _body; }

    // override anet::DataBufferSerializable
    bool serialize(anet::DataBuffer *outputBuffer) const override;
    bool deserialize(anet::DataBuffer *inputBuffer, int length = 0) override;
    int64_t getSpaceUsed() override { return _header->SpaceUsed() + _body->SpaceUsed(); }
    size_t getSerializedSize() const override { return 4 + _headerSerializedSize + 4 + _bodySerializedSize; }

protected:
    bool serializeMessage(const RPCMessage *message, size_t messageSize, anet::DataBuffer *outputBuffer) const;
    bool deserializeMessage(anet::DataBuffer *inputBuffer, RPCMessage *message, int32_t &leftPacketDataLen) const;

protected:
    std::shared_ptr<google::protobuf::Arena> _arena;
    RPCMessage *_header{nullptr};
    size_t _headerSerializedSize{0};
    size_t _bodySerializedSize{0};

private:
    RPCMessage *_body{nullptr};
};

ARPC_END_NAMESPACE(arpc);

