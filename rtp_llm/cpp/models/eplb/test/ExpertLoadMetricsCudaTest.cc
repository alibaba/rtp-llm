#include "rtp_llm/cpp/models/eplb/ExpertMetricUtils.h"
#include "rtp_llm/cpp/models/eplb/test/ExpertLoadMetricsReference.h"

#include <gtest/gtest.h>
#include <torch/cuda.h>

namespace rtp_llm {
namespace {

class ExpertLoadMetricsCudaTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "This test requires the GPU assigned by gpu_lock";
        host_ratio_ =
            torch::empty({}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU).pinned_memory(true));
    }

    void compare(const std::vector<int64_t>& packed, int64_t samples, int64_t layers, int64_t world, int64_t ep) {
        const auto reference = reduceExpertLoadWindowReference(packed, samples, layers, world, ep);
        const auto window    = torch::tensor(packed, torch::kInt64).reshape({world, samples, layers}).to(torch::kCUDA);
        const auto ratio     = reduceExpertLoadWindow(window, ep);
        ASSERT_TRUE(ratio.is_cuda());
        ASSERT_EQ(ratio.scalar_type(), torch::kDouble);
        ASSERT_EQ(ratio.dim(), 0);
        ASSERT_EQ(ratio.numel(), 1);
        ASSERT_TRUE(host_ratio_.is_pinned());
        host_ratio_.copy_(ratio, false);
        EXPECT_DOUBLE_EQ(host_ratio_.data_ptr<double>()[0], reference.aggregate.moe_lb);
    }

    torch::Tensor host_ratio_;
};

TEST_F(ExpertLoadMetricsCudaTest, DividesSumOfStepMaximaBySumOfStepMeans) {
    compare({4, 2, 4, 0}, 2, 1, 2, 2);
}

TEST_F(ExpertLoadMetricsCudaTest, KeepsAlternatingHotspotsAcrossStepsAndLayers) {
    compare({8, 0, 0, 8}, 2, 1, 2, 2);
    compare({8, 2, 0, 0, 0, 0, 0, 2, 0, 8, 0, 0}, 2, 3, 2, 2);
}

TEST_F(ExpertLoadMetricsCudaTest, MatchesReferenceForWorldAndReplicatedTpLayouts) {
    for (const int64_t world : {1, 2, 4, 8}) {
        constexpr int64_t    samples = 5;
        constexpr int64_t    layers  = 3;
        std::vector<int64_t> packed(world * samples * layers);
        for (int64_t rank = 0; rank < world; ++rank) {
            for (int64_t sample = 0; sample < samples; ++sample) {
                for (int64_t layer = 0; layer < layers; ++layer) {
                    packed[(rank * samples + sample) * layers + layer] = (rank * 13 + sample * 7 + layer * 5) % 19;
                }
            }
        }
        for (const auto ep : {int64_t{1}, world}) {
            SCOPED_TRACE(::testing::Message() << "world=" << world << ", ep=" << ep);
            compare(packed, samples, layers, world, ep);
        }
    }
}

TEST_F(ExpertLoadMetricsCudaTest, ZeroWorkAndEmptyWindowsYieldOne) {
    for (const int64_t world : {1, 2, 8}) {
        for (const auto ep : {int64_t{1}, world}) {
            SCOPED_TRACE(::testing::Message() << "world=" << world << ", ep=" << ep);
            compare(std::vector<int64_t>(world * 2 * 3, 0), 2, 3, world, ep);
            compare({}, 0, 3, world, ep);
        }
    }
}

TEST_F(ExpertLoadMetricsCudaTest, CountsStayInt64ThroughReduction) {
    compare({3000000000LL, 1000000000LL}, 1, 1, 2, 2);
    compare({3000000000LL, 1000000000LL, 2000000000LL, 4000000000LL}, 2, 1, 2, 2);
    compare({3000000000LL, 1000000000LL, 2000000000LL, 4000000000LL}, 2, 1, 2, 1);
}

}  // namespace
}  // namespace rtp_llm
