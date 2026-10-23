#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Exception.h"

#include <exception>
#include <string>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(RTPExceptionTest, PreservesMessageLocationAndStack) {
    RTPException      error("exception_test.cc", 42, "literal 100% and %s");
    const std::string text = error.what();
    EXPECT_NE(text.find("literal 100% and %s (exception_test.cc:42)\n"), std::string::npos);
    EXPECT_GT(text.size(), text.find('\n') + 1);
    EXPECT_FALSE(error.getTrace().empty());
    EXPECT_STREQ(error.what(), text.c_str());
}

TEST(RTPExceptionTest, EmptyUserMessageStillHasLocationAndTrace) {
    RTPException error("empty.cc", 7, "");
    EXPECT_NE(std::string(error.what()).find(" (empty.cc:7)\n"), std::string::npos);
    EXPECT_FALSE(error.getTrace().empty());
}

TEST(RTPExceptionTest, CopiedExceptionSurvivesOriginalAndCanBeRethrown) {
    std::exception_ptr saved;
    std::string        text;
    try {
        RTPException original("copy.cc", 9, "retained message");
        RTPException copied(original);
        text = copied.what();
        throw copied;
    } catch (const RTPException& error) {
        EXPECT_EQ(std::string(error.what()), text);
        saved = std::current_exception();
    }
    ASSERT_TRUE(saved);
    try {
        std::rethrow_exception(saved);
        FAIL() << "Expected an RTPException";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), text);
    }
}

TEST(RTPExceptionTest, RuntimeAssertionIsCatchableRepeatedly) {
    for (int i = 0; i < 8; ++i) {
        try {
            throwRuntimeError("assertion.cc", 23, "invalid configuration");
        } catch (const RTPException& error) {
            EXPECT_NE(std::string(error.what()).find("invalid configuration"), std::string::npos);
            EXPECT_NE(std::string(error.what()).find("assertion.cc:23"), std::string::npos);
            EXPECT_FALSE(error.getTrace().empty());
        }
    }
}

}  // namespace
}  // namespace rtp_llm
