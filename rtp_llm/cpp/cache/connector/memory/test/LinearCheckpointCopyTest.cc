#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm::test {

TEST(LinearCheckpointCopyTest, QuantizesOncePerChannelAndPreservesConvolutionBytes) {
    ASSERT_TRUE(torch::cuda::is_available());
    torch::manual_seed(42);
    const float      limit        = 127.f;
    const size_t     scalar_bytes = sizeof(int8_t);
    constexpr int    count = 7, heads = 3, values = 128, keys = 64, conv_bytes = 1536;
    constexpr size_t elements     = heads * values * keys;
    const size_t     packed_bytes = elements * scalar_bytes + heads * keys * sizeof(float);
    auto             device       = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto             host         = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true);
    auto             original     = torch::randn({count, heads, keys, values}, device);
    original[0].zero_();
    original[1].mul_(1.e-30);
    original[2][0][0].mul_(1000);
    auto state            = original.clone();
    auto history          = torch::randint(0, 256, {count, conv_bytes}, device.dtype(torch::kUInt8));
    auto expected_history = history.clone();
    auto packed           = torch::empty({count, static_cast<int64_t>(packed_bytes + conv_bytes)}, host);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    BatchedMemoryCopyParams copies;
    copies.device_index = state.get_device();
    std::vector<LinearCheckpointCopyTile> checkpoints;
    for (int i = 0; i < count; ++i) {
        auto* out = packed.data_ptr<uint8_t>() + i * (packed_bytes + conv_bytes);
        checkpoints.push_back(
            {state.data_ptr<float>() + i * elements, out, heads, values, keys, LinearCheckpointDType::INT8});
        copies.tiles.push_back({out + packed_bytes, history.data_ptr<uint8_t>() + i * conv_bytes, conv_bytes});
    }
    ASSERT_TRUE(execLinearCheckpointCopy(copies, checkpoints, false));
    auto cpu_original = original.cpu();
    for (int i = 0; i < count; ++i) {
        auto* bytes = packed.data_ptr<uint8_t>() + i * (packed_bytes + conv_bytes);
        auto  quant = torch::from_blob(bytes, {heads, keys, values}, torch::kInt8).to(torch::kFloat32);
        auto  scale = torch::from_blob(bytes + elements * scalar_bytes, {heads, keys, 1}, torch::kFloat32);
        auto  expected_scale =
            (cpu_original[i].abs().amax(2, true) / limit).clamp_min(std::numeric_limits<float>::min());
        ASSERT_TRUE(torch::allclose(scale, expected_scale, 1.e-6, 0));
        auto error = (quant * scale - cpu_original[i]).abs();
        ASSERT_TRUE((error <= scale * 0.501 + cpu_original[i].abs() * 1.e-6).all().item<bool>());
    }
    state.fill_(NAN);
    history.zero_();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (auto& tile : copies.tiles) {
        auto* old_dst = tile.dst;
        tile.dst      = const_cast<void*>(tile.src);
        tile.src      = old_dst;
    }
    ASSERT_TRUE(execLinearCheckpointCopy(copies, checkpoints, true));
    ASSERT_TRUE(torch::equal(history, expected_history));
    ASSERT_TRUE(torch::isfinite(state).all().item<bool>());
    auto scale = (original.abs().amax(3, true) / limit).clamp_min(std::numeric_limits<float>::min());
    ASSERT_TRUE(((state - original).abs() <= scale * 0.501 + original.abs() * 1.e-6).all().item<bool>());
    ASSERT_TRUE(torch::equal(state[0], original[0]));
}

TEST(LinearCheckpointCopyTest, BFloat16RetainsSmallChannelsAndRounding) {
    ASSERT_TRUE(torch::cuda::is_available());
    torch::manual_seed(43);
    constexpr int    count = 3, heads = 2, keys = 64, values = 128;
    constexpr size_t elements = heads * keys * values;
    auto             options  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto             state    = torch::randn({count, heads, keys, values}, options);
    state[0].mul_(1.e-20);
    state[1].mul_(1.e10);
    auto                    expected = state.to(torch::kBFloat16).to(torch::kFloat32);
    auto                    host     = torch::empty({count, static_cast<int64_t>(elements * 2)},
                             torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
    BatchedMemoryCopyParams params;
    params.device_index = state.get_device();
    std::vector<LinearCheckpointCopyTile> checkpoints;
    for (int i = 0; i < count; ++i) {
        checkpoints.push_back({state.data_ptr<float>() + i * elements,
                               host.data_ptr<uint8_t>() + i * elements * 2,
                               heads,
                               values,
                               keys,
                               LinearCheckpointDType::BF16});
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(execLinearCheckpointCopy(params, checkpoints, false));
    state.fill_(NAN);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(execLinearCheckpointCopy(params, checkpoints, true));
    ASSERT_TRUE(torch::equal(state, expected));
}

TEST(LinearCheckpointCopyTest, BothDirectionsRoundTripSparseTiles) {
    ASSERT_TRUE(torch::cuda::is_available());
    auto data     = torch::arange(0, 16384, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    auto expected = data.clone();
    auto host     = torch::zeros({16384}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    BatchedMemoryCopyParams params;
    params.device_index = data.get_device();
    for (int i = 0; i < 32; ++i) {
        params.tiles.push_back(
            {host.data_ptr<int32_t>() + i * 512, data.data_ptr<int32_t>() + i * 512, 511 * sizeof(int32_t)});
    }
    ASSERT_TRUE(execBatchedMemoryCopy(params));
    data.zero_();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (auto& tile : params.tiles) {
        auto* dst = tile.dst;
        tile.dst  = const_cast<void*>(tile.src);
        tile.src  = dst;
    }
    ASSERT_TRUE(execBatchedMemoryCopy(params));
    for (int i = 0; i < 32; ++i) {
        ASSERT_TRUE(torch::equal(data.slice(0, i * 512, i * 512 + 511), expected.slice(0, i * 512, i * 512 + 511)));
        ASSERT_EQ(data[i * 512 + 511].item<int32_t>(), 0);
    }
}

}  // namespace rtp_llm::test
