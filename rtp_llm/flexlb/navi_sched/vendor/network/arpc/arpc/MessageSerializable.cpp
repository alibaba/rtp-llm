#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageSerializable.h"

#include <cassert>
#include <cstddef>
#include <google/protobuf/descriptor.h>
#include <memory>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/DataBufferInputStream.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/DataBufferOutputStream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/io/coded_stream.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/DataBufferOutputStream.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

using namespace std;
using namespace anet;
using namespace google::protobuf::io;

ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(MessageSerializable);

MessageSerializable::MessageSerializable(RPCMessage *message, const std::shared_ptr<google::protobuf::Arena> &arena) {
    _arena = arena;
    _message = message;
    if (_message) {
        _messageSerializedSize = _message->ByteSizeLong();
    }
}

MessageSerializable::~MessageSerializable() {}

bool MessageSerializable::serialize(DataBuffer *outputBuffer) const {
    if (_message == NULL) {
        return true;
    }

    if (!_message->IsInitialized()) {
        ARPC_LOG(ERROR,
                 "failed to serialize: required fields may "
                 "not be initialized");
        return false;
    }

    if (_messageSerializedSize > MAX_RPC_MSG_BYTE_SIZE) {
        ARPC_LOG(ERROR,
                 "failed to serialize message. ByteSize %lu should be between 0 and %d",
                 _messageSerializedSize,
                 MAX_RPC_MSG_BYTE_SIZE);
        return false;
    }
    outputBuffer->ensureFree(_messageSerializedSize);
    DataBufferOutputStream outputStream(outputBuffer);
    ::google::protobuf::io::CodedOutputStream encoder(&outputStream);
    encoder.SetSerializationDeterministic(true);
    bool success = _message->SerializeToCodedStream(&encoder);
    encoder.Trim();

    return success;
}

bool MessageSerializable::deserialize(DataBuffer *inputBuffer, int length)
{
    if (inputBuffer == nullptr || length < 0 || inputBuffer->getDataLen() < length) {
        return false;
    }

    if (length == 0) {
        return true;
    }

    int oldDataLen = inputBuffer->getDataLen();
    DataBufferInputStream inputStream(inputBuffer, length);
    google::protobuf::io::CodedInputStream decoder(&inputStream);
    decoder.SetTotalBytesLimit(MAX_RPC_MSG_BYTE_SIZE, MAX_RPC_MSG_BYTE_SIZE);
    bool success = _message->ParseFromCodedStream(&decoder) && decoder.ConsumedEntireMessage();

    // DataBufferInputStream 保证无论序列化是否成功，DataBuffer 中的数据都被消费
    assert(oldDataLen - inputBuffer->getDataLen() == length);
    (void)oldDataLen;

    return success;
}

ARPC_END_NAMESPACE(arpc);
