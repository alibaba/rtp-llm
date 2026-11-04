#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCRawMessageSerializable.h"

using namespace arpc;
using namespace anet;

ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(RPCRawMessageSerializable);

RPCRawMessageSerializable::RPCRawMessageSerializable(RPCMessage *header, std::string *body)
    : RPCMessageSerializable(header, nullptr, nullptr), _body(body) {
    if (_header) {
        _headerSerializedSize = _header->ByteSizeLong();
    }
    if (_body) {
        _bodySerializedSize = _body->length();
    }
}

bool RPCRawMessageSerializable::serialize(anet::DataBuffer *outputBuffer) const {
    if (_header == nullptr || _body == nullptr || outputBuffer == nullptr) {
        ARPC_LOG(ERROR, "serialize failed, header or body or outputBuffer is nullptr");
        return false;
    }

    if (!serializeMessage(_header, _headerSerializedSize, outputBuffer)) {
        ARPC_LOG(ERROR, "serialize header error");
        return false;
    }

    if (!serializeRawMessage(_body, outputBuffer)) {
        ARPC_LOG(ERROR, "serialize body error");
        return false;
    }

    return true;
}

bool RPCRawMessageSerializable::deserialize(anet::DataBuffer *inputBuffer, int length) {
    if (length == 0) {
        return true;
    }

    if (_header == nullptr || _body == nullptr || inputBuffer == nullptr) {
        ARPC_LOG(ERROR, "deserialize failed, header or body or inputBuffer is nullptr");
        return false;
    }

    int leftPacketDataLen = length;

    if (!deserializeMessage(inputBuffer, _header, leftPacketDataLen)) {
        inputBuffer->drainData(leftPacketDataLen);
        ARPC_LOG(ERROR, "deserialize header error");
        return false;
    }

    if (!deserializeRawMessage(inputBuffer, _body, leftPacketDataLen)) {
        ARPC_LOG(ERROR, "deserialize body error");
        inputBuffer->drainData(leftPacketDataLen);
        return false;
    }

    return true;
}

bool RPCRawMessageSerializable::serializeRawMessage(const std::string *message, DataBuffer *outputBuffer) const {
    int oldLen = outputBuffer->getDataLen();
    int size = message->length();
    if (size < 0 || size > MAX_RPC_MSG_BYTE_SIZE) {
        ARPC_LOG(
            ERROR, "failed to serialize message. ByteSize %d should be between 0 and %d", size, MAX_RPC_MSG_BYTE_SIZE);
        return false;
    }

    outputBuffer->writeInt32(size);
    outputBuffer->ensureFree(size);
    outputBuffer->writeBytes(message->c_str(), size);
    int appendLen = outputBuffer->getDataLen() - oldLen;

    unsigned char *ptr = (unsigned char *)(outputBuffer->getData() + oldLen);
    outputBuffer->fillInt32(ptr, appendLen - sizeof(int32_t));
    return true;
}

bool RPCRawMessageSerializable::deserializeRawMessage(anet::DataBuffer *inputBuffer,
                                                      std::string *message,
                                                      int32_t &leftPacketDataLen) const {
    int32_t len = inputBuffer->readInt32();
    leftPacketDataLen -= sizeof(int32_t);

    if (len == 0 && leftPacketDataLen == 0) {
        // All fileds in current message are optional
        return true;
    }

    if (len > 0 && len <= leftPacketDataLen) {
        char *pData = inputBuffer->getData();
        message->assign(pData, len);
        inputBuffer->drainData(len);
        leftPacketDataLen -= len;
    }

    return true;
}

ARPC_END_NAMESPACE(arpc);
