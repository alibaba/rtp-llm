#include "gtest/gtest.h"

#include "grpcpp/impl/codegen/time.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"

namespace rtp_llm {

TEST(RpcErrorCodeTest, DecodeDeadlineDetailsSurvivePrefillBridge) {
    ErrorDetailsPB details;
    details.set_error_code(static_cast<int>(ErrorCode::GENERATE_TIMEOUT));
    details.set_error_message("request deadline expired before decode allocation");
    std::string serialized;
    ASSERT_TRUE(details.SerializeToString(&serialized));
    const grpc::Status status(grpc::StatusCode::DEADLINE_EXCEEDED, "remote request failed", serialized);

    const auto error = transGrpcStatusToErrorInfo(status, ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED);

    EXPECT_EQ(error.code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(error.ToString(), "request deadline expired before decode allocation");
}

TEST(RpcErrorCodeTest, GrpcDeadlineWithoutDetailsMapsToGenerateTimeout) {
    const grpc::Status status(grpc::StatusCode::DEADLINE_EXCEEDED, "deadline from transport");

    const auto error = transGrpcStatusToErrorInfo(status, ErrorCode::REMOTE_GENERATE_FAILED);

    EXPECT_EQ(error.code(), ErrorCode::GENERATE_TIMEOUT);
}

TEST(RpcErrorCodeTest, CanonicalDeadlineWinsOverConflictingCancelDetails) {
    ErrorDetailsPB details;
    details.set_error_code(static_cast<int>(ErrorCode::CANCELLED));
    details.set_error_message("server observed cancellation after its deadline");
    std::string serialized;
    ASSERT_TRUE(details.SerializeToString(&serialized));
    const grpc::Status status(grpc::StatusCode::DEADLINE_EXCEEDED, "transport deadline", serialized);

    const auto error = transGrpcStatusToErrorInfo(status, ErrorCode::REMOTE_GENERATE_FAILED);

    EXPECT_EQ(error.code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(error.ToString(), "server observed cancellation after its deadline");
}

TEST(RpcErrorCodeTest, TerminalRequestFailuresAreNotRetried) {
    EXPECT_FALSE(shouldRetryGenerateFailure(ErrorCode::GENERATE_TIMEOUT, grpc::StatusCode::INTERNAL));
    EXPECT_FALSE(shouldRetryGenerateFailure(ErrorCode::CANCELLED, grpc::StatusCode::INTERNAL));
    EXPECT_FALSE(shouldRetryGenerateFailure(ErrorCode::UNKNOWN_ERROR, grpc::StatusCode::INVALID_ARGUMENT));
    EXPECT_TRUE(shouldRetryGenerateFailure(ErrorCode::MALLOC_FAILED, grpc::StatusCode::RESOURCE_EXHAUSTED));
    EXPECT_TRUE(shouldRetryGenerateFailure(ErrorCode::CONNECT_FAILED, grpc::StatusCode::UNAVAILABLE));
}

TEST(RpcErrorCodeTest, RemoteLoadTransportPreservesDeadlineAndCancellation) {
    EXPECT_EQ(transRemoteLoadGrpcStatus(grpc::StatusCode::DEADLINE_EXCEEDED, false),
              ErrorCode::LOAD_CACHE_TIMEOUT);
    EXPECT_EQ(transRemoteLoadGrpcStatus(grpc::StatusCode::CANCELLED, true), ErrorCode::LOAD_CACHE_TIMEOUT);
    EXPECT_EQ(transRemoteLoadGrpcStatus(grpc::StatusCode::CANCELLED, false), ErrorCode::CANCELLED);
    EXPECT_EQ(transRemoteLoadGrpcStatus(grpc::StatusCode::UNAVAILABLE, false),
              ErrorCode::LOAD_KV_CACHE_FAILED);
}

TEST(RpcErrorCodeTest, RemoteLoadAggregationCannotDowngradeTerminalFailure) {
    EXPECT_EQ(mergeRemoteLoadErrorCode(ErrorCode::LOAD_CACHE_TIMEOUT, ErrorCode::LOAD_KV_CACHE_FAILED),
              ErrorCode::LOAD_CACHE_TIMEOUT);
    EXPECT_EQ(mergeRemoteLoadErrorCode(ErrorCode::CANCELLED, ErrorCode::LOAD_KV_CACHE_FAILED),
              ErrorCode::CANCELLED);
    EXPECT_EQ(mergeRemoteLoadErrorCode(ErrorCode::CANCELLED, ErrorCode::LOAD_CACHE_TIMEOUT),
              ErrorCode::LOAD_CACHE_TIMEOUT);
    EXPECT_EQ(mergeRemoteLoadErrorCode(ErrorCode::NONE_ERROR, ErrorCode::LOAD_KV_CACHE_FAILED),
              ErrorCode::LOAD_KV_CACHE_FAILED);
}

TEST(RpcErrorCodeTest, SixteenWorkerCompletionOrderCannotHideDeadline) {
    for (const bool reverse : {false, true}) {
        ErrorCode aggregate = ErrorCode::NONE_ERROR;
        for (int offset = 0; offset < 16; ++offset) {
            const int rank = reverse ? 15 - offset : offset;
            const auto candidate = rank == 7 ? ErrorCode::LOAD_CACHE_TIMEOUT :
                                                ErrorCode::LOAD_KV_CACHE_FAILED;
            aggregate = mergeRemoteLoadErrorCode(aggregate, candidate);
        }
        EXPECT_EQ(aggregate, ErrorCode::LOAD_CACHE_TIMEOUT);
    }
}

TEST(RpcErrorCodeTest, ServerDeadlineRejectsAtExactBoundaryWithoutCancelFlag) {
    grpc::ServerContext context;
    constexpr int64_t   deadline_unix_us = 10'000'000;
    const auto deadline =
        std::chrono::system_clock::time_point(std::chrono::microseconds(deadline_unix_us));
    grpc::Timepoint2Timespec(deadline, &context.deadline_);

    EXPECT_TRUE(serverContextStopError(&context, "admission", deadline_unix_us - 1).ok());
    const auto error = serverContextStopError(&context, "admission", deadline_unix_us);
    EXPECT_EQ(error.code(), ErrorCode::GENERATE_TIMEOUT);
}

}  // namespace rtp_llm
