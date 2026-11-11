#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(RpcErrorCodeTest, RemoteLoadTransportFailureSurvivesTheDecodePrefillWire) {
    // Decode's RemoteLoad completion reports 8212 for a failed transport, e.g.
    // "Socket closed". Preserve it through GenerateOutputsPB and the Prefill
    // conversion instead of falling back to UNKNOWN_ERROR (514).
    GenerateOutputsPB decode_response;
    decode_response.mutable_error_info()->set_error_code(transErrorCodeToRPC(ErrorCode::LOAD_KV_CACHE_FAILED));
    decode_response.mutable_error_info()->set_error_message("failed to load kv cache in rank: 0: Socket closed");

    GenerateOutputsPB prefill_response;
    ASSERT_TRUE(prefill_response.ParseFromString(decode_response.SerializeAsString()));
    EXPECT_NE(prefill_response.error_info().error_code(), ErrorCodePB::UNKNOWN_ERROR);
    EXPECT_EQ(transRPCErrorCode(prefill_response.error_info().error_code()), ErrorCode::LOAD_KV_CACHE_FAILED);
    EXPECT_EQ(prefill_response.error_info().error_message(), decode_response.error_info().error_message());
}

TEST(RpcErrorCodeTest, CacheTimeoutAndCancellationKeepTheirExistingCodes) {
    for (auto error : {ErrorCode::NONE_ERROR, ErrorCode::LOAD_CACHE_TIMEOUT, ErrorCode::CANCELLED}) {
        EXPECT_EQ(transRPCErrorCode(transErrorCodeToRPC(error)), error);
    }
    EXPECT_EQ(transRPCErrorCode(static_cast<ErrorCodePB>(9999)), ErrorCode::UNKNOWN_ERROR);
}

}  // namespace
}  // namespace rtp_llm
