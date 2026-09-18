#include <cstdint>
#include <limits>
#include <string>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorStatus.h"

namespace rtp_llm {

TEST(RpcErrorStatusTest, StructuredDetailsExplainEveryUnavailableState) {
    for (const std::string state : {"DRAINING", "SLEEPING", "WAKING_UP", "ERROR"}) {
        ErrorDetailsPB details;
        details.set_error_code(8600);
        details.set_error_code_str("ENGINE_UNAVAILABLE");
        details.set_state(state);
        details.set_sleep_epoch(7);
        details.set_error_message("engine unavailable");
        const grpc::Status status(grpc::StatusCode::UNAVAILABLE, "engine unavailable", details.SerializeAsString());
        EXPECT_EQ(formatGrpcErrorStatus(status),
                  "grpc_code=14, business_code=8600, symbol=ENGINE_UNAVAILABLE, state=" + state
                      + ", sleep_epoch=7, business_message=engine unavailable");
    }
}

TEST(RpcErrorStatusTest, OldNumericOnlyDetailsRecoverSymbolWithoutInventingState) {
    ErrorDetailsPB details;
    details.set_error_code(8300);
    details.set_error_message("load deadline expired");
    const grpc::Status status(
        grpc::StatusCode::DEADLINE_EXCEEDED, "load deadline expired", details.SerializeAsString());
    EXPECT_EQ(formatGrpcErrorStatus(status),
              "grpc_code=4, business_code=8300, symbol=LOAD_CACHE_TIMEOUT, business_message=load deadline expired");
}

TEST(RpcErrorStatusTest, UnknownBinaryAndZeroCodeProtobufsUseBoundedHex) {
    ErrorDetailsPB zero_code;
    zero_code.set_error_message("not a business error");
    for (const auto& bytes : {std::string{},
                              std::string("\xff\0", 2),
                              std::string("\x38\x01", 2),  // unknown protobuf field only
                              zero_code.SerializeAsString(),
                              std::string(100000, '\xff')}) {
        const auto result = formatGrpcErrorStatus(grpc::Status(grpc::StatusCode::INTERNAL, "transport failed", bytes));
        EXPECT_EQ(result,
                  "grpc_code=13, grpc_details_hex=" + rpcErrorDetailsHex(bytes) + ", grpc_message=transport failed");
        EXPECT_EQ(result.find("business_code="), std::string::npos);
        EXPECT_LT(result.size(), 650);
    }
}

TEST(RpcErrorStatusTest, StructuredCausePrecedesBoundedFreeFormMessages) {
    ErrorDetailsPB details;
    details.set_error_code(8600);
    details.set_error_code_str("ENGINE_UNAVAILABLE");
    details.set_state("DRAINING");
    details.set_sleep_epoch(23);
    details.set_error_message(std::string(100000, 'x'));
    const auto result = formatGrpcErrorStatus(
        grpc::Status(grpc::StatusCode::UNAVAILABLE, std::string(100000, 'y'), details.SerializeAsString()));
    const std::string prefix =
        "grpc_code=14, business_code=8600, symbol=ENGINE_UNAVAILABLE, state=DRAINING, sleep_epoch=23";
    EXPECT_EQ(result.substr(0, prefix.size()), prefix);
    EXPECT_NE(result.find("business_message=" + std::string(256, 'x') + "...[truncated]"), std::string::npos);
    EXPECT_NE(result.find("grpc_message=" + std::string(256, 'y') + "...[truncated]"), std::string::npos);
    EXPECT_LT(result.size(), 750);
    EXPECT_TRUE(google::protobuf::internal::IsStructurallyValidUTF8(result));
}

TEST(RpcErrorStatusTest, FutureBusinessCodesRemainNumericAndPreserveSuppliedSymbol) {
    ErrorDetailsPB details;
    details.set_error_code(std::numeric_limits<int64_t>::max());
    details.set_error_code_str("FUTURE_ERROR");
    auto result =
        formatGrpcErrorStatus(grpc::Status(grpc::StatusCode::INTERNAL, "future", details.SerializeAsString()));
    EXPECT_NE(result.find("business_code=9223372036854775807, symbol=FUTURE_ERROR"), std::string::npos);
    details.clear_error_code_str();
    result = formatGrpcErrorStatus(grpc::Status(grpc::StatusCode::INTERNAL, "future", details.SerializeAsString()));
    EXPECT_NE(result.find("business_code=9223372036854775807, symbol=UNRECOGNIZED"), std::string::npos);
}

}  // namespace rtp_llm
