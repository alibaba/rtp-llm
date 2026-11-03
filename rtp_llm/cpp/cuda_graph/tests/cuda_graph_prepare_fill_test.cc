#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#define hipSuccess cudaSuccess
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipDeviceSynchronize cudaDeviceSynchronize
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

namespace rtp_llm {
namespace {

TEST(CudaGraphPrepareFillTest, FusedSpPaddingPreservesGroupsInt64AndPartialWords) {
    SequenceParallelPadParams         params;
    std::vector<uint8_t*>             allocations;
    std::vector<std::vector<uint8_t>> expected;
    const auto                        add = [&](const void* source,
                         const void* result,
                         int64_t     old_group_bytes,
                         int64_t     new_group_bytes,
                         int64_t     groups,
                         int         element_bytes,
                         uint64_t    fill_bits) {
        uint8_t* src = nullptr;
        uint8_t* dst = nullptr;
        ASSERT_EQ(hipSuccess, hipMalloc(&src, old_group_bytes * groups));
        ASSERT_EQ(hipSuccess, hipMalloc(&dst, new_group_bytes * groups));
        ASSERT_EQ(hipSuccess, hipMemcpy(src, source, old_group_bytes * groups, hipMemcpyHostToDevice));
        allocations.push_back(src);
        allocations.push_back(dst);
        auto& region                   = params.regions[params.region_count++];
        region.source                  = src;
        region.destination             = dst;
        region.source_group_bytes      = old_group_bytes;
        region.destination_group_bytes = new_group_bytes;
        region.groups                  = groups;
        region.element_bytes           = element_bytes;
        region.fill_bits               = fill_bits;
        expected.emplace_back(new_group_bytes * groups);
        std::memcpy(expected.back().data(), result, expected.back().size());
    };
    const int32_t ids[]        = {8, 9, 10, 11, 12, 13};
    const int32_t ids_padded[] = {8, 9, 10, -1, -1, 11, 12, 13, -1, -1};
    add(ids, ids_padded, 12, 20, 2, 4, uint64_t(-1));
    const int64_t epochs[]        = {(int64_t(1) << 35) + 3, (int64_t(1) << 36) + 7};
    const int64_t epochs_padded[] = {epochs[0], epochs[1], -1, -1};
    add(epochs, epochs_padded, 16, 32, 1, 8, uint64_t(-1));
    const uint8_t mask[]        = {1, 0, 1, 0, 1, 0};
    const uint8_t mask_padded[] = {1, 0, 1, 1, 1, 0, 1, 0, 1, 1};
    add(mask, mask_padded, 3, 5, 2, 1, 1);
    const uint16_t hidden[]        = {0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0};
    const uint16_t hidden_padded[] = {0x3f80, 0x4000, 0x4040, 0, 0, 0x4080, 0x40a0, 0x40c0, 0, 0};
    add(hidden, hidden_padded, 6, 10, 2, 2, 0);
    invokeSequenceParallelPad(params, nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    for (int i = 0; i < params.region_count; ++i) {
        auto actual = expected[i];
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(actual.data(), params.regions[i].destination, actual.size(), hipMemcpyDeviceToHost));
        EXPECT_EQ(actual, expected[i]);
    }
    for (auto* pointer : allocations)
        EXPECT_EQ(hipSuccess, hipFree(pointer));
}

TEST(CudaGraphPrepareFillTest, FusesMaskHiddenTailAndBatchScalarWithoutTouchingLiveRows) {
    uint8_t*             hidden      = nullptr;
    bool*                source_mask = nullptr;
    bool*                mask        = nullptr;
    int32_t*             batch       = nullptr;
    const bool           source[]    = {true, false, true, true};
    std::vector<uint8_t> bytes(23, 91);
    ASSERT_EQ(hipSuccess, hipMalloc(&hidden, bytes.size()));
    ASSERT_EQ(hipSuccess, hipMalloc(&source_mask, sizeof(source)));
    ASSERT_EQ(hipSuccess, hipMalloc(&mask, 8 * sizeof(bool)));
    ASSERT_EQ(hipSuccess, hipMalloc(&batch, sizeof(int32_t)));
    ASSERT_EQ(hipSuccess, hipMemcpy(hidden, bytes.data(), bytes.size(), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(source_mask, source, sizeof(source), hipMemcpyHostToDevice));
    CudaGraphPrepareFillParams params;
    params.region_count            = 1;
    params.regions[0].ptr          = batch;
    params.regions[0].count        = 1;
    params.regions[0].value        = 14;
    params.zero_bytes              = hidden + 8;
    params.zero_byte_count         = 15;
    params.valid_token_mask        = mask;
    params.source_valid_token_mask = source_mask;
    params.mask_count              = 8;
    params.source_mask_count       = 4;
    for (const int first : {8, 9}) {
        std::fill(bytes.begin(), bytes.end(), 91);
        ASSERT_EQ(hipSuccess, hipMemcpy(hidden, bytes.data(), bytes.size(), hipMemcpyHostToDevice));
        params.zero_bytes      = hidden + first;
        params.zero_byte_count = bytes.size() - first;
        invokeCudaGraphPrepareFill(params, nullptr);
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
        bool    actual_mask[8];
        int32_t actual_batch = -1;
        ASSERT_EQ(hipSuccess, hipMemcpy(bytes.data(), hidden, bytes.size(), hipMemcpyDeviceToHost));
        ASSERT_EQ(hipSuccess, hipMemcpy(actual_mask, mask, sizeof(actual_mask), hipMemcpyDeviceToHost));
        ASSERT_EQ(hipSuccess, hipMemcpy(&actual_batch, batch, sizeof(actual_batch), hipMemcpyDeviceToHost));
        EXPECT_EQ(actual_batch, 14);
        for (int i = 0; i < 8; ++i)
            EXPECT_EQ(actual_mask[i], i < 4 && source[i]);
        for (int i = 0; i < 23; ++i)
            EXPECT_EQ(bytes[i], i < first ? 91 : 0);
    }
    EXPECT_EQ(hipSuccess, hipFree(batch));
    EXPECT_EQ(hipSuccess, hipFree(mask));
    EXPECT_EQ(hipSuccess, hipFree(source_mask));
    EXPECT_EQ(hipSuccess, hipFree(hidden));
}

TEST(CudaGraphPrepareFillTest, FillsLiteralAndDeviceBasedIotaRegions) {
    std::vector<int32_t> initial(10, -1);
    int32_t              source_value = 100;
    int32_t*             output       = nullptr;
    int32_t*             source       = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&output, initial.size() * sizeof(int32_t)));
    ASSERT_EQ(hipSuccess, hipMalloc(&source, sizeof(int32_t)));
    ASSERT_EQ(hipSuccess,
              hipMemcpy(output, initial.data(), initial.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(source, &source_value, sizeof(int32_t), hipMemcpyHostToDevice));

    CudaGraphPrepareFillParams params;
    auto&                      literal = params.regions[params.region_count++];
    literal.ptr                        = output + 2;
    literal.count                      = 4;
    literal.value                      = 10;
    literal.step                       = 3;

    auto& from_device        = params.regions[params.region_count++];
    from_device.ptr          = output + 6;
    from_device.value_ptr    = source;
    from_device.count        = 3;
    from_device.value_offset = 5;
    from_device.step         = 2;

    invokeCudaGraphPrepareFill(params, nullptr);

    std::vector<int32_t> actual(initial.size());
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    ASSERT_EQ(hipSuccess,
              hipMemcpy(actual.data(), output, actual.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    EXPECT_EQ(actual, (std::vector<int32_t>{-1, -1, 10, 13, 16, 19, 105, 107, 109, -1}));

    EXPECT_EQ(hipSuccess, hipFree(source));
    EXPECT_EQ(hipSuccess, hipFree(output));
}

TEST(CudaGraphPrepareFillTest, ClearsGraphEightTargetVerifyTailForSixLiveRequests) {
    constexpr int32_t graph_batch = 8;
    constexpr int32_t live_batch  = 6;
    constexpr int32_t query_len   = 8;
    constexpr int32_t mrope_dims  = 3;

    std::vector<int32_t> prefix(graph_batch, 17);
    std::vector<int32_t> lengths(graph_batch, query_len);
    std::vector<int32_t> cu_query(graph_batch + 1);
    std::vector<int32_t> cu_kv(graph_batch + 1);
    std::vector<int32_t> sequence_lengths(graph_batch, 18);
    std::vector<int32_t> input_ids(graph_batch * query_len, 91);
    std::vector<int32_t> position_ids(graph_batch * query_len * mrope_dims, 73);
    for (int32_t row = 0; row <= graph_batch; ++row) {
        cu_query[row] = row * query_len;
        cu_kv[row]    = row * (17 + query_len);
    }

    auto allocate_and_copy = [](const std::vector<int32_t>& host) {
        int32_t* device = nullptr;
        EXPECT_EQ(hipSuccess, hipMalloc(&device, host.size() * sizeof(int32_t)));
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(device, host.data(), host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        return device;
    };
    auto copy_back = [](std::vector<int32_t>& host, int32_t* device) {
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(host.data(), device, host.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    };

    int32_t* prefix_device           = allocate_and_copy(prefix);
    int32_t* lengths_device          = allocate_and_copy(lengths);
    int32_t* cu_query_device         = allocate_and_copy(cu_query);
    int32_t* cu_kv_device            = allocate_and_copy(cu_kv);
    int32_t* sequence_lengths_device = allocate_and_copy(sequence_lengths);
    int32_t* input_ids_device        = allocate_and_copy(input_ids);
    int32_t* position_ids_device     = allocate_and_copy(position_ids);

    CudaGraphPrepareFillParams params;
    auto add_region = [&params](int32_t* ptr, int64_t count, int32_t value, int32_t step = 0) {
        auto& region = params.regions[params.region_count++];
        region.ptr   = ptr;
        region.count = count;
        region.value = value;
        region.step  = step;
    };
    add_region(prefix_device + live_batch, graph_batch - live_batch, 0);
    add_region(lengths_device + live_batch, graph_batch - live_batch, query_len);
    add_region(cu_query_device + live_batch + 1,
               graph_batch - live_batch,
               (live_batch + 1) * query_len,
               query_len);
    add_region(cu_kv_device + live_batch + 1, graph_batch - live_batch, 0, query_len);
    params.regions[params.region_count - 1].value_ptr    = cu_kv_device + live_batch;
    params.regions[params.region_count - 1].value_offset = query_len;
    add_region(sequence_lengths_device + live_batch, graph_batch - live_batch, query_len);
    add_region(input_ids_device + live_batch * query_len, (graph_batch - live_batch) * query_len, 0);
    add_region(position_ids_device + live_batch * query_len * mrope_dims,
               (graph_batch - live_batch) * query_len * mrope_dims,
               0);

    invokeCudaGraphPrepareFill(params, nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    copy_back(prefix, prefix_device);
    copy_back(lengths, lengths_device);
    copy_back(cu_query, cu_query_device);
    copy_back(cu_kv, cu_kv_device);
    copy_back(sequence_lengths, sequence_lengths_device);
    copy_back(input_ids, input_ids_device);
    copy_back(position_ids, position_ids_device);

    EXPECT_EQ(prefix, (std::vector<int32_t>{17, 17, 17, 17, 17, 17, 0, 0}));
    EXPECT_EQ(lengths, (std::vector<int32_t>{8, 8, 8, 8, 8, 8, 8, 8}));
    EXPECT_EQ(cu_query, (std::vector<int32_t>{0, 8, 16, 24, 32, 40, 48, 56, 64}));
    EXPECT_EQ(cu_kv, (std::vector<int32_t>{0, 25, 50, 75, 100, 125, 150, 158, 166}));
    EXPECT_EQ(sequence_lengths, (std::vector<int32_t>{18, 18, 18, 18, 18, 18, 8, 8}));
    EXPECT_TRUE(std::all_of(input_ids.begin() + live_batch * query_len, input_ids.end(), [](int32_t value) {
        return value == 0;
    }));
    EXPECT_TRUE(std::all_of(position_ids.begin() + live_batch * query_len * mrope_dims,
                            position_ids.end(),
                            [](int32_t value) { return value == 0; }));

    EXPECT_EQ(hipSuccess, hipFree(position_ids_device));
    EXPECT_EQ(hipSuccess, hipFree(input_ids_device));
    EXPECT_EQ(hipSuccess, hipFree(sequence_lengths_device));
    EXPECT_EQ(hipSuccess, hipFree(cu_kv_device));
    EXPECT_EQ(hipSuccess, hipFree(cu_query_device));
    EXPECT_EQ(hipSuccess, hipFree(lengths_device));
    EXPECT_EQ(hipSuccess, hipFree(prefix_device));
}

}  // namespace
}  // namespace rtp_llm
