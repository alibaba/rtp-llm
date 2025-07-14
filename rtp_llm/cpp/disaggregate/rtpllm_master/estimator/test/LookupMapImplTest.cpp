#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/LookupMapImpl.h"

namespace rtp_llm {
namespace rtp_llm_master {
class LookupMapImplTest: public EngineBaseTest {
public:
    void check(const LookupMapImpl& estimator, const TaskInfo& task_info, int64_t expect_value) {
        auto result = estimator.estimate(task_info);
        ASSERT_TRUE(result.ok()) << result.status().message() << task_info.prefix_length << " "
                                 << task_info.input_length;
        ASSERT_EQ(result.value(), expect_value);
    }
    void check_fail(const LookupMapImpl& estimator, const TaskInfo& task_info, const std::string& error) {
        auto result = estimator.estimate(task_info);
        ASSERT_FALSE(result.ok()) << result.status().message();
        std::cout << result.status().message() << std::endl;
        ASSERT_TRUE(result.status().message() == error);
    }

protected:
    std::string config_path = "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/test/testdata/config.json";
};

TEST_F(LookupMapImplTest, testInit) {
    auto estimator = LookupMapImpl();
    ASSERT_TRUE(estimator.init(config_path));
    ASSERT_EQ(estimator.scopes_.size(), 2);
    ASSERT_EQ(estimator.scopes_[0].map_items.size(), 12);
    ASSERT_EQ(estimator.scopes_[1].map_items.size(), 12);
    ASSERT_EQ(estimator.max_input_length_, 8);
    ASSERT_EQ(estimator.max_prefix_length_, 6);
}

TEST_F(LookupMapImplTest, testEstimate) {
    auto estimator = LookupMapImpl();
    ASSERT_TRUE(estimator.init(config_path));
    // get the point
    check(estimator, {1, 1}, 11);
    // // boundary of scope0
    check(estimator, {2, 4}, 42);
    check(estimator, {6, 8}, 86);
    // // get mean
    check(estimator, {4, 5}, 54);
    check(estimator, {5, 5}, 55);
    check(estimator, {0, 5}, 50);
    // out of range
    check_fail(estimator, {6, 9}, "illegal node: (6, 9) with boundary: (0, 0) - [6, 8]");
    check_fail(estimator, {7, 8}, "illegal node: (7, 8) with boundary: (0, 0) - [6, 8]");
    check_fail(estimator, {1, 0}, "illegal node: (1, 0) with boundary: (0, 0) - [6, 8]");
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm
