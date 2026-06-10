#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/models/logits_processor/DFAUtil.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"

#include <chrono>
#include <fstream>
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

class TreeDFATest: public ::testing::Test {
protected:
    void SetUp() override {
        std::string   json = R"({
            "start_token_id": 100, "end_token_id": 999,
            "prefix_dict": {
                "100":     [10, 20],
                "100_10":  [30, 40],
                "100_20":  [50],
                "100_10_30": [999]
            }
        })";
        std::string   path = "/tmp/test_tree_dfa.json";
        std::ofstream f(path);
        f << json;
        f.close();
        prefix_ = PrefixToCandidateTokens::instance();
        prefix_->reloadPrefixDict(path);
    }

    std::shared_ptr<PrefixToCandidateTokens> prefix_;
};

TEST_F(TreeDFATest, InvalidTokenEntersErrorState) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    std::string s = dfa.next(10);
    ASSERT_EQ("100_10", s);
    ASSERT_FALSE(dfa.hasError());

    // Invalid transition: "100_10" -> input 99 → error state
    s = dfa.next(99);
    ASSERT_TRUE(dfa.hasError());
    // status_ stays at last valid state
    ASSERT_EQ("100_10", s);
}

TEST_F(TreeDFATest, ErrorStateOnlyAllowsEOS) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    dfa.next(10);
    dfa.next(99);  // invalid → error
    ASSERT_TRUE(dfa.hasError());

    auto candidates = dfa.getCandidateTokenIds();
    ASSERT_EQ(1u, candidates.size());
    ASSERT_EQ(prefix_->endTokenId(), candidates[0]);
}

TEST_F(TreeDFATest, ErrorStateFreezesFurtherInput) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    dfa.next(10);
    std::string before_error = dfa.status();
    dfa.next(99);  // invalid → error
    ASSERT_TRUE(dfa.hasError());

    // Further inputs are ignored, status doesn't change
    std::string s = dfa.next(20);
    ASSERT_EQ(before_error, s);
    ASSERT_TRUE(dfa.hasError());
}

TEST_F(TreeDFATest, ValidPathToEnd) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    ASSERT_FALSE(dfa.isFinished());
    dfa.next(10);  // "100_10"
    ASSERT_FALSE(dfa.isFinished());
    dfa.next(30);  // "100_10_30"
    ASSERT_FALSE(dfa.isFinished());
    dfa.next(999);  // end token
    ASSERT_TRUE(dfa.isFinished());
}

TEST_F(TreeDFATest, FinishedDFAIgnoresFurtherInput) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    dfa.next(10);
    dfa.next(30);
    dfa.next(999);
    ASSERT_TRUE(dfa.isFinished());

    // Further input on a finished DFA is ignored: the status must stay unchanged.
    std::string finished_status = dfa.status();
    ASSERT_EQ(finished_status, dfa.next(10));
    ASSERT_TRUE(dfa.isFinished());
}

TEST_F(TreeDFATest, GetCandidateTokenIds) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    auto                       candidates = dfa.getCandidateTokenIds();
    std::unordered_set<size_t> candidate_set(candidates.begin(), candidates.end());
    // At start "100", candidates should be {10, 20}
    ASSERT_TRUE(candidate_set.count(10));
    ASSERT_TRUE(candidate_set.count(20));

    dfa.next(10);  // "100_10"
    candidates    = dfa.getCandidateTokenIds();
    candidate_set = std::unordered_set<size_t>(candidates.begin(), candidates.end());
    // At "100_10", candidates should be {30, 40}
    ASSERT_TRUE(candidate_set.count(30));
    ASSERT_TRUE(candidate_set.count(40));
}

TEST_F(TreeDFATest, ForceSetStatus) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    dfa.forceSetStatus("100_20");
    ASSERT_EQ("100_20", dfa.status());

    auto                       candidates = dfa.getCandidateTokenIds();
    std::unordered_set<size_t> candidate_set(candidates.begin(), candidates.end());
    ASSERT_TRUE(candidate_set.count(50));
}

// Runtime coverage for the negative-candidate fix: negative ids are skipped at load, so
// getCandidateTokenIds() must not hit its negative-id throw path during logits processing.
TEST_F(TreeDFATest, NegativeCandidateTokensSkippedAtRuntime) {
    std::string   json = R"({
        "start_token_id": 100, "end_token_id": 999,
        "prefix_dict": {"100": [-5, 10, -1, 20]}
    })";
    std::string   path = "/tmp/test_tree_dfa_neg.json";
    std::ofstream f(path);
    f << json;
    f.close();

    auto prefix = PrefixToCandidateTokens::instance();
    prefix->reloadPrefixDict(path);
    ASSERT_TRUE(prefix->initSuccess());

    TreeDFA<std::string, size_t> dfa(prefix);

    std::vector<size_t> candidates;
    ASSERT_NO_THROW(candidates = dfa.getCandidateTokenIds());

    std::unordered_set<size_t> candidate_set(candidates.begin(), candidates.end());
    ASSERT_TRUE(candidate_set.count(10));
    ASSERT_TRUE(candidate_set.count(20));
    ASSERT_EQ(2u, candidate_set.size()) << "negative candidate ids must not survive to runtime";
}

}  // namespace rtp_llm
