#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <numeric>
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

TEST(CudaGraphPrepareFillTest, GeneratesQ1MetadataForShrinkingPaddedAndFakeBatches) {
    constexpr int capacity = 19;
    int32_t*      lengths  = nullptr;
    int32_t*      output   = nullptr;
    bool*         mask     = nullptr;
    constexpr int words    = 3 * (capacity + 1) + capacity;
    ASSERT_EQ(hipSuccess, hipMalloc(&lengths, capacity * sizeof(int32_t)));
    ASSERT_EQ(hipSuccess, hipMalloc(&output, words * sizeof(int32_t)));
    ASSERT_EQ(hipSuccess, hipMalloc(&mask, capacity * sizeof(bool)));
    std::vector<int32_t> source(capacity);
    for (int row = 0; row < capacity; ++row) {
        source[row] = row * 65537;
    }
    ASSERT_EQ(hipSuccess, hipMemcpy(lengths, source.data(), source.size() * sizeof(int32_t), hipMemcpyHostToDevice));

    for (const int batch : {capacity, 13, 1, 17}) {
        for (const bool fake : {false, true}) {
            std::vector<int32_t> initial(words, -77);
            ASSERT_EQ(hipSuccess, hipMemcpy(output, initial.data(), words * sizeof(int32_t), hipMemcpyHostToDevice));
            CudaGraphPrepareFillParams params;
            const auto add_region = [&](int32_t* destination, int64_t count, int32_t value, int32_t step = 0) {
                auto& region = params.regions[params.region_count++];
                region.ptr   = destination;
                region.count = count;
                region.value = value;
                region.step  = step;
            };
            auto* cu_query        = output;
            auto* cu_kv           = cu_query + capacity + 1;
            auto* decode_cu       = cu_kv + capacity + 1;
            auto* sequence_plus_1 = decode_cu + capacity + 1;
            add_region(cu_query, batch + 1, 0);
            add_region(cu_kv, batch + 1, 0);
            add_region(decode_cu, batch + 1, 0, 1);
            add_region(sequence_plus_1 + batch, capacity - batch, 1);
            params.decode_sequence_lengths        = lengths;
            params.decode_sequence_lengths_plus_1 = sequence_plus_1;
            params.decode_batch_size              = batch;
            params.valid_token_mask               = mask;
            params.mask_count                     = capacity;
            params.source_mask_count              = fake ? 0 : std::max(0, batch - 2);
            params.generate_valid_token_mask      = true;
            invokeCudaGraphPrepareFill(params, nullptr);
            ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
            std::vector<int32_t> actual(words);
            bool                 actual_mask[capacity];
            ASSERT_EQ(hipSuccess, hipMemcpy(actual.data(), output, words * sizeof(int32_t), hipMemcpyDeviceToHost));
            ASSERT_EQ(hipSuccess, hipMemcpy(actual_mask, mask, capacity * sizeof(bool), hipMemcpyDeviceToHost));
            for (int row = 0; row <= capacity; ++row) {
                EXPECT_EQ(actual[row], row <= batch ? 0 : -77);
                EXPECT_EQ(actual[capacity + 1 + row], row <= batch ? 0 : -77);
                EXPECT_EQ(actual[2 * (capacity + 1) + row], row <= batch ? row : -77);
            }
            for (int row = 0; row < capacity; ++row) {
                EXPECT_EQ(actual[3 * (capacity + 1) + row], row < batch ? source[row] + 1 : 1);
                EXPECT_EQ(actual_mask[row], !fake && row < batch - 2);
            }
        }
    }
    EXPECT_EQ(hipSuccess, hipFree(mask));
    EXPECT_EQ(hipSuccess, hipFree(output));
    EXPECT_EQ(hipSuccess, hipFree(lengths));
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

#if USING_CUDA
TEST(CudaGraphPrepareFillTest, FusedSpInputStagingOrdersProducerModelReplayAndSourceRetirement) {
    constexpr int    capacity = 128 * 4;
    constexpr size_t bytes    = capacity * sizeof(int32_t);
    cudaStream_t     producer, consumer;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&producer));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&consumer));
    int32_t *output = nullptr, *model_output = nullptr, *results = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&output, bytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&model_output, bytes));
    cudaGraph_t     graph;
    cudaGraphExec_t executable;
    ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(consumer, cudaStreamCaptureModeThreadLocal));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(model_output, output, bytes, cudaMemcpyDeviceToDevice, consumer));
    ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(consumer, &graph));
    ASSERT_EQ(cudaSuccess, cudaGraphInstantiateWithFlags(&executable, graph, 0));
    const std::vector<int>   batches = {111, 113, 112, 111, 128, 1};
    std::vector<cudaEvent_t> ready_events(batches.size());
    ASSERT_EQ(cudaSuccess, cudaMalloc(&results, batches.size() * bytes));
    for (auto& event : ready_events) {
        ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    for (size_t i = 0; i < batches.size(); ++i) {
        const int live     = batches[i] * 4;
        const int physical = (batches[i] + 3) / 4 * 4 * 4;
        int32_t*  source   = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMallocAsync(&source, live * sizeof(int32_t), producer));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(source, 33 + i, live * sizeof(int32_t), producer));
        ASSERT_EQ(cudaSuccess, cudaEventRecord(ready_events[i], producer));
        ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(consumer, ready_events[i], 0));
        // Existing metadata preparation owns the captured-bucket tail.
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(output, 0, bytes, consumer));
        if (live < physical) {
            CudaGraphInputStagingSources sources{
                reinterpret_cast<const uint8_t*>(source), nullptr, live * static_cast<int64_t>(sizeof(int32_t)), 0};
            invokeCudaGraphInputStaging(
                sources, reinterpret_cast<uint8_t*>(output), physical * sizeof(int32_t), nullptr, 0, consumer);
        } else {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpyAsync(output, source, live * sizeof(int32_t), cudaMemcpyDeviceToDevice, consumer));
        }
        ASSERT_EQ(cudaSuccess, cudaGraphLaunch(executable, consumer));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpyAsync(results + i * capacity, model_output, bytes, cudaMemcpyDeviceToDevice, consumer));
        ASSERT_EQ(cudaSuccess, cudaFreeAsync(source, consumer));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(consumer));
    for (size_t i = 0; i < batches.size(); ++i) {
        std::vector<int32_t> actual(capacity);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), results + i * capacity, bytes, cudaMemcpyDeviceToHost));
        const int32_t value = static_cast<int32_t>(0x01010101u * (33 + i));
        for (int token = 0; token < capacity; ++token) {
            EXPECT_EQ(actual[token], token < batches[i] * 4 ? value : 0);
        }
        ASSERT_EQ(cudaSuccess, cudaEventDestroy(ready_events[i]));
    }
    ASSERT_EQ(cudaSuccess, cudaFree(results));
    ASSERT_EQ(cudaSuccess, cudaGraphExecDestroy(executable));
    ASSERT_EQ(cudaSuccess, cudaGraphDestroy(graph));
    ASSERT_EQ(cudaSuccess, cudaFree(model_output));
    ASSERT_EQ(cudaSuccess, cudaFree(output));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(consumer));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(producer));
}

TEST(CudaGraphPrepareFillTest, FusedSpInputStagingPadsPhysicalRowsAndLeavesGraphTailUntouched) {
    constexpr int bucket = 128, max_query = 4, width = 7;
    cudaStream_t  stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int32_t* tokens = nullptr;
    uint8_t* hidden = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&tokens, bucket * max_query * sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&hidden, bucket * max_query * width * sizeof(uint64_t)));
    for (const int query : {1, 4}) {
        for (const int element_bytes : {2, 4, 8}) {
            for (const bool with_hidden : {false, true}) {
                const int64_t token_capacity  = bucket * query * sizeof(int32_t);
                const int64_t hidden_capacity = with_hidden ? bucket * query * width * element_bytes : 0;
                for (const int batch : {113, 111, 112, 1, 128, 111}) {
                    const int            alignment       = 16 / std::gcd(16, query);
                    const int            physical        = (batch + alignment - 1) / alignment * alignment;
                    const int            live_tokens     = batch * query;
                    const int            physical_tokens = physical * query;
                    std::vector<int32_t> expected(bucket * query, 0x7f7f7f7f);
                    std::fill(expected.begin() + live_tokens, expected.begin() + physical_tokens, 0);
                    for (int i = 0; i < live_tokens; ++i)
                        expected[i] = batch * 65537 + i;
                    std::vector<uint8_t> expected_hidden(hidden_capacity, 0x6e);
                    const int64_t        live_hidden     = with_hidden ? live_tokens * width * element_bytes : 0;
                    const int64_t        physical_hidden = with_hidden ? physical_tokens * width * element_bytes : 0;
                    if (with_hidden)
                        std::fill(expected_hidden.begin() + live_hidden, expected_hidden.begin() + physical_hidden, 0);
                    for (int64_t i = 0; i < live_hidden; ++i)
                        expected_hidden[i] = static_cast<uint8_t>((i * 17 + batch) % 251);
                    int32_t* source_tokens = nullptr;
                    uint8_t* source_hidden = nullptr;
                    ASSERT_EQ(cudaSuccess, cudaMalloc(&source_tokens, live_tokens * sizeof(int32_t)));
                    ASSERT_EQ(
                        cudaSuccess,
                        cudaMemcpy(
                            source_tokens, expected.data(), live_tokens * sizeof(int32_t), cudaMemcpyHostToDevice));
                    if (with_hidden) {
                        ASSERT_EQ(cudaSuccess, cudaMalloc(&source_hidden, live_hidden));
                        ASSERT_EQ(
                            cudaSuccess,
                            cudaMemcpy(source_hidden, expected_hidden.data(), live_hidden, cudaMemcpyHostToDevice));
                    }
                    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(tokens, 0x7f, token_capacity, stream));
                    ASSERT_EQ(cudaSuccess,
                              cudaMemsetAsync(hidden, 0x6e, bucket * query * width * sizeof(uint64_t), stream));
                    if (batch < physical) {
                        CudaGraphInputStagingSources sources{reinterpret_cast<const uint8_t*>(source_tokens),
                                                             source_hidden,
                                                             live_tokens * static_cast<int64_t>(sizeof(int32_t)),
                                                             live_hidden};
                        invokeCudaGraphInputStaging(sources,
                                                    reinterpret_cast<uint8_t*>(tokens),
                                                    physical_tokens * sizeof(int32_t),
                                                    hidden,
                                                    physical_hidden,
                                                    stream);
                    } else {
                        ASSERT_EQ(cudaSuccess,
                                  cudaMemcpyAsync(tokens,
                                                  source_tokens,
                                                  live_tokens * sizeof(int32_t),
                                                  cudaMemcpyDeviceToDevice,
                                                  stream));
                        if (with_hidden) {
                            ASSERT_EQ(
                                cudaSuccess,
                                cudaMemcpyAsync(hidden, source_hidden, live_hidden, cudaMemcpyDeviceToDevice, stream));
                        }
                    }
                    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
                    std::vector<int32_t> actual(expected.size());
                    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), tokens, token_capacity, cudaMemcpyDeviceToHost));
                    EXPECT_EQ(actual, expected);
                    if (with_hidden) {
                        std::vector<uint8_t> actual_hidden(hidden_capacity);
                        ASSERT_EQ(cudaSuccess,
                                  cudaMemcpy(actual_hidden.data(), hidden, hidden_capacity, cudaMemcpyDeviceToHost));
                        EXPECT_EQ(actual_hidden, expected_hidden);
                    } else {
                        uint8_t untouched = 0;
                        ASSERT_EQ(cudaSuccess, cudaMemcpy(&untouched, hidden, 1, cudaMemcpyDeviceToHost));
                        EXPECT_EQ(untouched, 0x6e);
                    }
                    ASSERT_EQ(cudaSuccess, cudaFree(source_tokens));
                    if (source_hidden) {
                        ASSERT_EQ(cudaSuccess, cudaFree(source_hidden));
                    }
                }
            }
        }
    }
    ASSERT_EQ(cudaSuccess, cudaFree(hidden));
    ASSERT_EQ(cudaSuccess, cudaFree(tokens));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
#endif

}  // namespace
}  // namespace rtp_llm
