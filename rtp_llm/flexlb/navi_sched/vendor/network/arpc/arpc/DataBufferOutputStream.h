#pragma once
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_ext.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "google/protobuf/io/zero_copy_stream.h"

ARPC_BEGIN_NAMESPACE(arpc)

class DataBufferOutputStream : public ::google::protobuf::io::ZeroCopyOutputStream4RawStr {
public:
    DataBufferOutputStream(anet::DataBuffer *dataBuffer, int blockSize = BLOCK_SIZE);
    virtual ~DataBufferOutputStream();
public:
    // override ZeroCopyInputStream4RawString
    bool Next(void **data, int *size) override;
    void BackUp(int count) override;
    RPCint64 ByteCount() const override;
    bool WriteRawString(std::RawString *rstr) override;
    bool SupportRawString() const override { return true; }

private:
    anet::DataBuffer *_dataBuffer;
    int _blockSize;
    RPCint64 _byteCount;
    // using value of kDefaultBlockSize in protocolbuffer 2.3.0
    static const int BLOCK_SIZE = 8192;
};

ARPC_END_NAMESPACE(arpc)

