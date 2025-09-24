
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/models/logits_processor/DFAUtil.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class DFAUtilTest: public ::testing::Test {
protected:
};

TEST_F(DFAUtilTest, testSimple) {
    std::vector<int>              re = {3};
    StringContainDFA<size_t, int> dfa(re);

    ASSERT_FALSE(dfa.isFinished());
    ASSERT_EQ(0, dfa.next(1));
    ASSERT_EQ(0, dfa.next(2));
    ASSERT_EQ(1, dfa.next(3));
    ASSERT_EQ(1, dfa.next(1));
    ASSERT_EQ(1, dfa.next(1));
    ASSERT_EQ(1, dfa.next(3));
    ASSERT_TRUE(dfa.isFinished());
}

TEST_F(DFAUtilTest, testComplex) {
    std::vector<int>              re = {1, 1, 2};
    StringContainDFA<size_t, int> dfa(re);

    ASSERT_FALSE(dfa.isFinished());
    ASSERT_EQ(1, dfa.next(1));
    ASSERT_EQ(2, dfa.next(1));
    ASSERT_EQ(2, dfa.next(1));
    ASSERT_EQ(0, dfa.next(3));
    ASSERT_EQ(1, dfa.next(1));
    ASSERT_EQ(2, dfa.next(1));
    ASSERT_EQ(3, dfa.next(2));
    ASSERT_TRUE(dfa.isFinished());
}

}  // namespace rtp_llm
