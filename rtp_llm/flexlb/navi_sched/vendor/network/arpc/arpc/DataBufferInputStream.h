#pragma once

#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_ext.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"

ARPC_BEGIN_NAMESPACE(arpc)

class DataBufferInputStream : public ::google::protobuf::io::ZeroCopyInputStream4RawStr {
public:
    DataBufferInputStream(anet::DataBuffer *dataBuffer, int32_t messageSize) : _messageSize(messageSize) {
        if (dataBuffer && messageSize > 0 && dataBuffer->getDataLen() >= _messageSize) {
            dataBuffer->readBytes(_messageData, _messageSize);
        }
    };
    virtual ~DataBufferInputStream() = default;

    // override ZeroCopyInputStream4RawStr
    bool Next(const void **data, int *size) override;
    void BackUp(int count) override;
    bool Skip(int count) override;
    int64_t ByteCount() const override;
    bool FillRawString(std::RawString *str, size_t str_size, const char *data, size_t data_size) override;

private:
    std::shared_ptr<const char> _messageData;
    int32_t _messageSize{0};
    int32_t _offset{0};
};

ARPC_END_NAMESPACE(arpc)
