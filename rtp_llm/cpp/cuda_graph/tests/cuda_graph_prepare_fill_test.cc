#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

#if !USING_ROCM
#error "cuda_graph_prepare_fill_rocm_test must be built with --config=rocm"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

namespace rtp_llm {
namespace {

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
