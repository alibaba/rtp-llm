#pragma once

#include <memory>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databufferserializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "google/protobuf/arena.h"

ARPC_BEGIN_NAMESPACE(arpc);

class MessageSerializable : public anet::DataBufferSerializable {
public:
    MessageSerializable(RPCMessage *message, const std::shared_ptr<google::protobuf::Arena> &arena);
    virtual ~MessageSerializable();

public:
    RPCMessage *getMessage() const { return _message; }

    // override anet::DataBufferSerializable
    bool serialize(anet::DataBuffer *outputBuffer) const override;
    bool deserialize(anet::DataBuffer *inputBuffer, int length = 0) override;
    int64_t getSpaceUsed() override { return _message->SpaceUsed(); }
    size_t getSerializedSize() const override { return _messageSerializedSize; }

private:
    std::shared_ptr<google::protobuf::Arena> _arena;
    RPCMessage *_message;
    size_t _messageSerializedSize{0};
};

ARPC_END_NAMESPACE(arpc);

