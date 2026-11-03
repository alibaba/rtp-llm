#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcError.h"

namespace rtp_llm {
namespace {

std::string details(int64_t code, const std::string& message = "worker error") {
    ErrorDetailsPB error;
    error.set_error_code(code);
    error.set_error_message(message);
    return error.SerializeAsString();
}

TEST(MMGrpcErrorTest, preservesStructuredCodeAndMessageAheadOfGenericStatusAndText) {
    // Include Python-only and future codes, not just the C++ multimodal enum.
    for (int code : {409, 429, 499, 603, 605, 606, 901, 902, 903, 906, 908, 8502, 123456}) {
        SCOPED_TRACE(code);
        auto error = parseMultimodalGrpcError(
            grpc::Status(grpc::StatusCode::INTERNAL, "[MM_PROCESS_ERROR] generic", details(code)));
        EXPECT_EQ(static_cast<int>(error.code()), code);
        EXPECT_EQ(error.ToString(), "worker error");
    }
}

TEST(MMGrpcErrorTest, emptyStructuredMessageUsesGrpcMessage) {
    auto error =
        parseMultimodalGrpcError(grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid input", details(605, "")));
    EXPECT_EQ(error.code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(error.ToString(), "invalid input");
}

TEST(MMGrpcErrorTest, malformedOrInvalidDetailsFallBackToLegacyPrefix) {
    for (const auto& wire : {std::string(),
                             std::string("malformed"),
                             details(0),
                             details(-1),
                             details(static_cast<int64_t>(std::numeric_limits<int>::max()) + 1)}) {
        auto error = parseMultimodalGrpcError(
            grpc::Status(grpc::StatusCode::INTERNAL, "[MM_WRONG_FORMAT_ERROR] bad image", wire));
        EXPECT_EQ(error.code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
        EXPECT_EQ(error.ToString(), "bad image");
    }
}

TEST(MMGrpcErrorTest, transportFailuresKeepTheirExistingFallback) {
    const std::pair<grpc::StatusCode, ErrorCode> cases[] = {
        {grpc::StatusCode::CANCELLED, ErrorCode::CANCELLED},
        {grpc::StatusCode::RESOURCE_EXHAUSTED, ErrorCode::CONCURRENCY_LIMIT_ERROR},
        {grpc::StatusCode::UNAVAILABLE, ErrorCode::MM_REMOTE_RPC_FAILED},
        {grpc::StatusCode::DEADLINE_EXCEEDED, ErrorCode::MM_REMOTE_RPC_FAILED},
        {grpc::StatusCode::INTERNAL, ErrorCode::UNKNOWN_ERROR},
    };
    for (const auto& [status, expected] : cases) {
        SCOPED_TRACE(status);
        auto error = parseMultimodalGrpcError(grpc::Status(status, "transport failure"));
        EXPECT_EQ(error.code(), expected);
        EXPECT_EQ(error.ToString(), "transport failure");
    }
}

}  // namespace
}  // namespace rtp_llm
