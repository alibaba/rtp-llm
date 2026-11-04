#pragma once

#include <grpc++/impl/codegen/byte_buffer.h>
#include <grpc++/impl/codegen/proto_utils.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class GrpcByteBufferSource : public grpc::protobuf::io::ZeroCopyInputStream
{
public:
    GrpcByteBufferSource();
    bool Init(const grpc::ByteBuffer &src); // Can be called multiple times.
    bool Next(const void **data, int *size) override;
    void BackUp(int count) override;
    bool Skip(int count) override;
    grpc::protobuf::int64 ByteCount() const override;
    void setLimit(int limit);
    int size() const;

private:
    std::vector<grpc::Slice> slices_;
    int size_;
    int limit_;
    int cur_;         // Current slice index.
    int left_;        // Number of bytes in slices_[cur_] left to yield.
    const char *ptr_; // Address of next byte in slices_[cur_] to yield.
    grpc::protobuf::int64 byte_count_;
};

class GrpcByteBufferSourceByteReader
{
public:
    GrpcByteBufferSourceByteReader(GrpcByteBufferSource &source);
    ~GrpcByteBufferSourceByteReader();

public:
    bool next(uint8_t *byte);

private:
    GrpcByteBufferSource &_source;
    const void *_data;
    int32_t _size;
};

// compatible util
class ProtobufCompatibleUtil
{
public:
    static bool getErrorCodeField(const google::protobuf::Message *message,
                                  const std::string &fieldName, MultiCallErrorCode &ec);
    static void setErrorCodeField(google::protobuf::Message *message, const std::string &fieldName,
                                  MultiCallErrorCode ec);
    static bool getGigMetaField(const google::protobuf::Message *message,
                                const std::string &fieldName, std::string &meta);
    static void setGigMetaField(google::protobuf::Message *message, const std::string &fieldName,
                                const std::string &meta);
    static bool getStringField(const google::protobuf::Message *message,
                               const std::string &fieldName, std::string &value);


private:
    static bool getFieldAndReflection(const google::protobuf::Message *message,
                                      const std::string &fieldName,
                                      google::protobuf::FieldDescriptor::Type type,
                                      const google::protobuf::Reflection *&reflection,
                                      const google::protobuf::FieldDescriptor *&field);
};

} // namespace multi_call

