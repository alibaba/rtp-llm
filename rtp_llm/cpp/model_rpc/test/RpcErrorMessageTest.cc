#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorMessage.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

TEST(RpcErrorMessageTest, ValidTextIncludingChineseEmojiAndNulIsPreserved) {
    for (const auto& input :
         {std::string{}, std::string("plain ASCII"), std::string(u8"休眠失败 💤"), std::string("a\0b\n", 4)}) {
        EXPECT_EQ(safeRpcErrorMessage(input), input);
    }
}

TEST(RpcErrorMessageTest, MalformedUtf8NeverLeaksIntoProtobufString) {
    const std::vector<std::string> invalid = {
        std::string("\x08\x98\x43", 3),  // serialized error_code=8600
        std::string("\xff", 1),
        std::string("\x80", 1),
        std::string("\xc0\xaf", 2),          // overlong sequence
        std::string("\xed\xa0\x80", 3),      // surrogate
        std::string("\xf4\x90\x80\x80", 4),  // beyond Unicode
        std::string("\xe4\xb8", 2),          // incomplete code point
    };
    for (const auto& input : invalid) {
        SCOPED_TRACE(rpcErrorDetailsHex(input));
        const auto output = safeRpcErrorMessage(input);
        EXPECT_TRUE(google::protobuf::internal::IsStructurallyValidUTF8(output));
        EXPECT_NE(output.find("invalid UTF-8"), std::string::npos);
        ErrorDetailsPB pb;
        pb.set_error_code(8600);
        setSafeRpcErrorMessage(&pb, input);
        ErrorDetailsPB round_trip;
        ASSERT_TRUE(round_trip.ParseFromString(pb.SerializeAsString()));
        EXPECT_EQ(round_trip.error_code(), 8600);
        EXPECT_EQ(round_trip.error_message(), output);
    }
}

TEST(RpcErrorMessageTest, TruncationDoesNotSplitAnyUtf8CodePoint) {
    for (const std::string codepoint : {u8"é", u8"中", u8"💤"}) {
        for (size_t cut = 1; cut < codepoint.size(); ++cut) {
            const std::string prefix(4096 - cut, 'a');
            const auto        result = safeRpcErrorMessage(prefix + codepoint + "suffix");
            EXPECT_EQ(result, prefix + "...[truncated]");
            EXPECT_TRUE(google::protobuf::internal::IsStructurallyValidUTF8(result));
        }
    }
    EXPECT_EQ(safeRpcErrorMessage(std::string(4096, 'x')), std::string(4096, 'x'));
    EXPECT_EQ(safeRpcErrorMessage(std::string(4097, 'x')), std::string(4096, 'x') + "...[truncated]");
}

TEST(RpcErrorMessageTest, ArbitraryDetailsHaveBoundedAsciiHexRepresentation) {
    std::string all_bytes;
    for (int i = 0; i < 256; ++i) {
        all_bytes.push_back(static_cast<char>(i));
    }
    const auto hex = rpcErrorDetailsHex(all_bytes);
    EXPECT_EQ(hex.size(), 512);
    EXPECT_EQ(hex.substr(0, 8), "00010203");
    EXPECT_EQ(hex.substr(504), "fcfdfeff");
    EXPECT_EQ(rpcErrorDetailsHex(""), "");
    const auto bounded = rpcErrorDetailsHex(all_bytes + std::string(100000, 'x'));
    EXPECT_EQ(bounded, hex + "...[truncated,total_bytes=100256]");
    EXPECT_LT(safeRpcErrorMessage(all_bytes + std::string(100000, 'x')).size(), 600);
}

}  // namespace rtp_llm
