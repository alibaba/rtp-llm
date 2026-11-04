#pragma once

#include <grpc++/impl/codegen/byte_buffer.h>
#include <grpc++/impl/codegen/proto_utils.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/ProtobufUtil.h"

namespace multi_call {

class ProtobufByteBufferUtil
{
public:
    static void serializeToBuffer(const google::protobuf::Message &src, grpc::ByteBuffer *dst);
    static bool deserializeFromBuffer(const grpc::ByteBuffer &src, google::protobuf::Message *dst);
    static void serializeToBuffer(const google::protobuf::Message *header, const grpc::Slice *msg,
                                  grpc::ByteBuffer *dst);
    static bool deserializeFromBuffer(const grpc::ByteBuffer &src,
                                      google::protobuf::Message *header,
                                      google::protobuf::Message *msg);
    static bool getHeaderLength(GrpcByteBufferSource &stream, size_t &headerLength);

public:
    static constexpr size_t LENGTH_SIZE = 8;
};

} // namespace multi_call

