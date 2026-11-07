#include <algorithm>
#include <vector>

#include <gtest/gtest.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/Sleep.h>
#include <cuda_runtime.h>

#include "rtp_llm/models_py/bindings/common/FusedCopyHostSource.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

namespace rtp_llm {
namespace {

TEST(FusedCopyHostSourceTest, KeepsPinnedViewsAliveBeyondTwoReleaseRounds) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto storage = torch::arange(32, options);
    auto source = storage.narrow(0, 3, 16);
    auto device = torch::empty({16}, options.device(torch::kCUDA).pinned_memory(false));
    std::vector<torch::Tensor> warm;
    for (int i = 0; i < 129; ++i) warm.push_back(torch::empty({32}, options));
    warm.clear();
    auto stream = at::cuda::getCurrentCUDAStream();
    FusedD2DCopyParams copies;
    copies.add(source.data_ptr(), device.data_ptr(), source.nbytes());
    // Warm lazy CUDA module loading before placing a delay in the stream.
    at::cuda::sleep(1);
    fusedCopy(copies);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    // Leave the device read queued while the CPU reclaims holder rounds.
    at::cuda::sleep(300'000'000);
    auto owned = prepareFusedCopyHostSource(source);
    EXPECT_EQ(owned.data_ptr(), source.data_ptr());
    const auto pointer = storage.data_ptr();
    TensorHolder holder;
    holder.hold_host(owned);
    FusedD2DCopyParams delayed_copy;
    delayed_copy.add(owned.data_ptr(), device.data_ptr(), owned.nbytes());
    fusedCopy(delayed_copy);
    at::cuda::CUDAEvent complete;
    complete.record(stream);
    storage = source = owned = torch::Tensor();
    for (int round = 0; round < 3; ++round) holder.release();
    std::vector<torch::Tensor> churn;
    for (int i = 0; i < 128; ++i) {
        churn.push_back(torch::full({32}, -999, options));
        EXPECT_NE(churn.back().data_ptr(), pointer);
    }
    EXPECT_FALSE(complete.query()) << "CPU churn finished after the delayed copy";
    complete.synchronize();
    EXPECT_TRUE(torch::equal(device.cpu(), torch::arange(3, 19, torch::kInt32)));
}

TEST(FusedCopyHostSourceTest, CopiesUntrackedHostStorage) {
    std::vector<int32_t> data{3, 1, 4, 1, 5};
    auto source = torch::from_blob(data.data(), {5}, torch::kInt32);
    auto owned = prepareFusedCopyHostSource(source);
    EXPECT_TRUE(owned.is_pinned());
    EXPECT_NE(source.data_ptr(), owned.data_ptr());
    std::fill(data.begin(), data.end(), -1);
    EXPECT_TRUE(torch::equal(owned, torch::tensor({3, 1, 4, 1, 5}, torch::kInt32)));
}

}  // namespace
}  // namespace rtp_llm
