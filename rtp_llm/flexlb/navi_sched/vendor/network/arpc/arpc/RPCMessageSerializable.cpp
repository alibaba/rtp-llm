#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCMessageSerializable.h"

#include <google/protobuf/descriptor.h>
#include <stddef.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/DataBufferInputStream.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/DataBufferOutputStream.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

using namespace anet;
using namespace google::protobuf::io;
ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(RPCMessageSerializable);

static const int32_t kEnableSerializationDeterministicSizeLimit = 16 * 1024;

RPCMessageSerializable::RPCMessageSerializable(RPCMessage *header,
                                               RPCMessage *body,
                                               const std::shared_ptr<google::protobuf::Arena> &arena) {
    _arena = arena;
    _header = header;
    _body = body;
    if (_header) {
        _headerSerializedSize = _header->ByteSizeLong();
    }
    if (_body) {
        _bodySerializedSize = _body->ByteSizeLong();
    }
}

RPCMessageSerializable::~RPCMessageSerializable() {}

bool RPCMessageSerializable::serialize(DataBuffer *outputBuffer) const {
    if (_header == NULL || _body == NULL || outputBuffer == NULL) {
        return false;
    }

    outputBuffer->ensureFree(getSerializedSize());
    if (!serializeMessage(_header, _headerSerializedSize, outputBuffer)) {
        ARPC_LOG(ERROR, "serialize header error");
        return false;
    }

    if (!serializeMessage(_body, _bodySerializedSize, outputBuffer)) {
        ARPC_LOG(ERROR, "serialize body error");
        return false;
    }

    return true;
}

bool RPCMessageSerializable::deserialize(anet::DataBuffer *inputBuffer, int length) {
    if (length == 0) {
        return true;
    }

    if (_header == NULL || _body == NULL || inputBuffer == NULL) {
        return false;
    }

    int leftPacketDataLen = length;

    if (!deserializeMessage(inputBuffer, _header, leftPacketDataLen)) {
        inputBuffer->drainData(leftPacketDataLen);
        ARPC_LOG(ERROR, "deserialize header error");
        return false;
    }

    if (!deserializeMessage(inputBuffer, _body, leftPacketDataLen)) {
        ARPC_LOG(ERROR, "deserialize body error");
        inputBuffer->drainData(leftPacketDataLen);
        return false;
    }

    return true;
}

bool RPCMessageSerializable::serializeMessage(const RPCMessage *message,
                                              size_t messageSize,
                                              DataBuffer *outputBuffer) const {
    int oldLen = outputBuffer->getDataLen();
    if (messageSize > MAX_RPC_MSG_BYTE_SIZE) {
        ARPC_LOG(ERROR,
                 "failed to serialize message. ByteSize %lu should be between 0 and %d",
                 messageSize,
                 MAX_RPC_MSG_BYTE_SIZE);
        return false;
    }

    bool ret = message->IsInitialized();
    if (!ret) {
        ARPC_LOG(ERROR,
                 "failed to serialize message: required fields"
                 " may not be initialized");
        return false;
    }

    int32_t sizeOffset = outputBuffer->getDataLen();
    outputBuffer->writeInt32(messageSize);
    outputBuffer->ensureFree(messageSize);
    DataBufferOutputStream outputStream(outputBuffer);
    ::google::protobuf::io::CodedOutputStream encoder(&outputStream);
    // SerializationDeterministic 功能会影响 std::RawString 的序列化，需要在这个功能开启的情况下才会触发 WriteRawString
    // 回调，从而避免拷贝。 然而报文大小小于一定程度时，启用 serialization deterministic
    // 会导致性能受到一定影响，因此根据报文大小判断是否开启这个功能。
    encoder.SetSerializationDeterministic(messageSize > kEnableSerializationDeterministicSizeLimit);
    bool success = message->SerializeToCodedStream(&encoder);
    encoder.Trim();
    int appendLen = outputBuffer->getDataLen() - oldLen;
    if (!success) {
        ARPC_LOG(ERROR, "serialize the message failed");
        outputBuffer->stripData(appendLen);
        return false;
    }

    if (appendLen - sizeof(int32_t) != messageSize) {
        outputBuffer->fillInt32(sizeOffset, appendLen - sizeof(int32_t));
    }
    return true;
}

bool RPCMessageSerializable::deserializeMessage(DataBuffer *inputBuffer,
                                                RPCMessage *message,
                                                int32_t &leftPacketDataLen) const {
    int32_t len = inputBuffer->readInt32();
    leftPacketDataLen -= sizeof(int32_t);

    if (len == 0 && leftPacketDataLen == 0) {
        // All fileds in current message are optional
        return true;
    }

    bool ret = false;

    if (len > 0 && len <= leftPacketDataLen) {
        DataBufferInputStream input(inputBuffer, len);
        ret = (message->ParseFromZeroCopyStream(&input));
        leftPacketDataLen -= len;
    }

    return ret;
}

TYPEDEF_PTR(RPCMessageSerializable);
ARPC_END_NAMESPACE(arpc);
