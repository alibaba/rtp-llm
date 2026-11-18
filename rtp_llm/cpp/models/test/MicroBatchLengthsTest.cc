#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include <ATen/record_function.h>
#include <numeric>
#include <string>
#include <chrono>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace rtp_llm {
namespace {

thread_local int length_d2h_count = 0;

class CountLengthCopies {
public:
    CountLengthCopies() {
        length_d2h_count = 0;
        handle_ = at::addThreadLocalCallback(
            at::RecordFunctionCallback([](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
                if (std::string(fn.name()) == "aten::copy_" && fn.inputs().size() >= 2
                    && fn.inputs()[0].isTensor() && fn.inputs()[1].isTensor()) {
                    const auto dst = fn.inputs()[0].toTensor();
                    const auto src = fn.inputs()[1].toTensor();
                    if (dst.device().is_cpu() && src.is_cuda()) ++length_d2h_count;
                }
                return nullptr;
            }).needsInputs(true));
    }
    ~CountLengthCopies() { at::removeCallback(handle_); }

private:
    at::CallbackHandle handle_;
};

GptModelInputs makeMicroInputs(const std::vector<int32_t>& lengths, int64_t decode) {
    GptModelInputs inputs;
    const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto batch = static_cast<int64_t>(lengths.size());
    const auto tokens = std::accumulate(lengths.begin() + decode, lengths.end(), int64_t{decode});
    inputs.input_lengths = torch::tensor(lengths, options);
    inputs.sequence_lengths = torch::full({decode}, 11, options);
    inputs.prefix_lengths = torch::arange(batch - decode, options);
    inputs.combo_tokens = torch::arange(tokens, options);
    inputs.lm_output_indexes = torch::arange(batch, options);
    inputs.kv_cache_block_id = torch::arange(batch * 4, options).view({batch, 4});
    inputs.kv_cache_kernel_block_id = inputs.kv_cache_block_id;
    return inputs;
}

class MicroBatchLengthsTest: public DeviceTestBase {};

TEST_F(MicroBatchLengthsTest, LazyPlannerAndSplitPreserveSlicesWithFewerReadbacks) {
    struct Case {
        std::vector<int32_t> lengths;
        int64_t decode;
        size_t layers;
        std::vector<int64_t> counts;
        int copies;
    };
    const std::vector<Case> cases = {
        {{1}, 1, 2, {}, 0},
        {{1, 1, 1}, 3, 2, {2, 1}, 0},
        {{2, 5, 3}, 0, 2, {7, 3}, 1},
        {{1, 1, 2, 6, 4}, 2, 1, {4, 10}, 1},
        {{1, 1, 2, 6, 4}, 2, 2, {}, 0},
    };
    for (const auto& item : cases) {
        auto inputs = makeMicroInputs(item.lengths, item.decode);
        auto legacy_plan = PyWrappedModel::planMicroBatches(inputs, true, item.layers);
        auto legacy = PyWrappedModel::splitInputsIntoMicroBatches(inputs, legacy_plan);
        torch::Tensor snapshot;
        std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>> actual;
        MicroBatchPlan plan;
        {
            CountLengthCopies counter;
            plan = PyWrappedModel::planMicroBatches(inputs, true, item.layers, &snapshot);
            actual = PyWrappedModel::splitInputsIntoMicroBatches(inputs, plan, &snapshot);
            EXPECT_EQ(length_d2h_count, item.copies);
        }
        EXPECT_EQ(plan.enable, legacy_plan.enable);
        EXPECT_EQ(snapshot.defined(), item.copies != 0);
        ASSERT_EQ(actual.first.size(), legacy.first.size());
        ASSERT_EQ(actual.second.size(), item.counts.size());
        size_t offset = 0;
        for (size_t i = 0; i < item.counts.size(); ++i) {
            EXPECT_EQ(actual.second[i].offset, offset);
            EXPECT_EQ(actual.second[i].count, static_cast<size_t>(item.counts[i]));
            offset += item.counts[i];
        }
        for (size_t i = 0; i < actual.first.size(); ++i) {
            EXPECT_TRUE(torch::equal(actual.first[i].combo_tokens, legacy.first[i].combo_tokens));
            EXPECT_TRUE(torch::equal(actual.first[i].input_lengths, legacy.first[i].input_lengths));
            EXPECT_TRUE(torch::equal(actual.first[i].sequence_lengths, legacy.first[i].sequence_lengths));
            EXPECT_TRUE(torch::equal(actual.first[i].prefix_lengths, legacy.first[i].prefix_lengths));
            if (plan.enable) {
                EXPECT_TRUE(torch::equal(actual.first[i].lm_output_indexes, legacy.first[i].lm_output_indexes));
                EXPECT_TRUE(torch::equal(actual.first[i].kv_cache_kernel_block_id,
                                        legacy.first[i].kv_cache_kernel_block_id));
            }
        }
    }
}

TEST_F(MicroBatchLengthsTest, NewForwardUsesNewValuesAtSameDeviceAddress) {
    auto inputs = makeMicroInputs({1, 1, 2, 6, 4}, 2);
    auto address = inputs.input_lengths.data_ptr();
    for (const auto& lengths : {std::vector<int32_t>{1, 1, 2, 6, 4}, std::vector<int32_t>{1, 1, 8, 2, 2}}) {
        inputs.input_lengths.copy_(torch::tensor(lengths, torch::kInt32));
        torch::Tensor snapshot;
        auto plan = PyWrappedModel::planMicroBatches(inputs, true, 1, &snapshot);
        ASSERT_TRUE(snapshot.defined());
        auto snapshot_address = snapshot.data_ptr();
        auto split = PyWrappedModel::splitInputsIntoMicroBatches(inputs, plan, &snapshot);
        EXPECT_EQ(snapshot.data_ptr(), snapshot_address);
        EXPECT_EQ(inputs.input_lengths.data_ptr(), address);
        EXPECT_TRUE(torch::equal(snapshot, torch::tensor(lengths, torch::kInt32)));
        ASSERT_EQ(split.second.size(), 2);
        EXPECT_EQ(split.second[0].count, lengths[2] == 2 ? 4u : 2u);
    }
}

TEST_F(MicroBatchLengthsTest, DisabledPlannerDoesNotReadLengths) {
    auto inputs = makeMicroInputs({1, 3}, 1);
    torch::Tensor snapshot;
    CountLengthCopies counter;
    auto plan = PyWrappedModel::planMicroBatches(inputs, false, 1, &snapshot);
    auto split = PyWrappedModel::splitInputsIntoMicroBatches(inputs, plan, &snapshot);
    EXPECT_FALSE(plan.enable);
    EXPECT_FALSE(snapshot.defined());
    EXPECT_EQ(length_d2h_count, 0);
    EXPECT_EQ(split.first.front().combo_tokens.data_ptr(), inputs.combo_tokens.data_ptr());
}

TEST_F(MicroBatchLengthsTest, DISABLED_benchmarkLocalMetadataAB) {
    auto measure = [](auto&& operation, int iterations) {
        for (int i = 0; i < 20; ++i)
            operation();
        TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            operation();
        TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / iterations;
    };
    for (int batch : {1, 31, 127}) {
        for (bool decode : {false, true}) {
            auto inputs = makeMicroInputs(std::vector<int32_t>(batch, decode ? 1 : 17), decode ? batch : 0);
            auto legacy = [&] {
                auto plan = PyWrappedModel::planMicroBatches(inputs, true, 1);
                return PyWrappedModel::splitInputsIntoMicroBatches(inputs, plan);
            };
            auto shared = [&] {
                torch::Tensor snapshot;
                auto          plan = PyWrappedModel::planMicroBatches(inputs, true, 1, &snapshot);
                return PyWrappedModel::splitInputsIntoMicroBatches(inputs, plan, &snapshot);
            };
            auto expected = legacy();
            auto actual   = shared();
            ASSERT_EQ(actual.first.size(), expected.first.size());
            for (size_t i = 0; i < actual.first.size(); ++i) {
                ASSERT_TRUE(torch::equal(actual.first[i].combo_tokens, expected.first[i].combo_tokens));
            }
            int baseline_copies, candidate_copies;
            {
                CountLengthCopies count;
                legacy();
                baseline_copies = length_d2h_count;
            }
            {
                CountLengthCopies count;
                shared();
                candidate_copies = length_d2h_count;
            }
            for (int round = 0; round < 10; ++round) {
                double before, after;
                if (round % 2) {
                    after  = measure(shared, 100);
                    before = measure(legacy, 100);
                } else {
                    before = measure(legacy, 100);
                    after  = measure(shared, 100);
                }
                std::cout << "LOCAL_METADATA_AB {\"case\":\"microbatch\",\"batch\":" << batch
                          << ",\"decode\":" << decode << ",\"round\":" << round
                          << ",\"baseline_d2h\":" << baseline_copies << ",\"candidate_d2h\":" << candidate_copies
                          << ",\"baseline_us\":" << before << ",\"candidate_us\":" << after << "}" << std::endl;
            }
        }
        auto storage = torch::arange((batch + 2) * 41, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32))
                           .view({batch + 2, 41})
                           .narrow(0, 1, batch)
                           .slice(1, 2, 39, 2);
        auto accepted    = torch::arange(batch, torch::kInt32).remainder(6);
        auto old_metrics = [&] {
            auto rows  = accepted.to(torch::kInt64).flatten().contiguous();
            auto total = accepted.to(torch::kInt64).sum().reshape({1}).pin_memory();
            return std::make_pair(rows, total);
        };
        auto reused_metrics = [&] {
            auto rows  = accepted.to(torch::kInt64).flatten().contiguous();
            auto total = rows.sum().reshape({1}).pin_memory();
            return std::make_pair(rows, total);
        };
        ASSERT_TRUE(torch::equal(old_metrics().second, reused_metrics().second));
        for (int round = 0; round < 10; ++round) {
            double before, after;
            if (round % 2) {
                after  = measure(reused_metrics, 1000);
                before = measure(old_metrics, 1000);
            } else {
                before = measure(old_metrics, 1000);
                after  = measure(reused_metrics, 1000);
            }
            std::cout << "LOCAL_METADATA_AB {\"case\":\"cpu_metrics_primitive\",\"batch\":" << batch
                      << ",\"round\":" << round << ",\"baseline_us\":" << before << ",\"candidate_us\":" << after << "}"
                      << std::endl;
        }
        auto stream        = at::cuda::getStreamFromPool();
        auto legacy_record = [&] { storage.record_stream(stream); };
        auto direct_record = [&] {
            c10::cuda::CUDACachingAllocator::recordStream(storage.storage().data_ptr(), stream);
        };
        auto legacy_view = [&] { return storage.narrow(0, batch - 1, 1); };
        auto direct_view = [&] {
            auto sizes = storage.sizes().vec();
            sizes[0]   = 1;
            return storage.as_strided(
                sizes, storage.strides(), storage.storage_offset() + (batch - 1) * storage.stride(0));
        };
        ASSERT_TRUE(torch::equal(legacy_view(), direct_view()));
        ASSERT_EQ(legacy_view().data_ptr(), direct_view().data_ptr());
        for (int round = 0; round < 10; ++round) {
            double old_record, new_record, old_view, new_view;
            if (round % 2) {
                new_record = measure(direct_record, 10000);
                old_record = measure(legacy_record, 10000);
                new_view   = measure(direct_view, 10000);
                old_view   = measure(legacy_view, 10000);
            } else {
                old_record = measure(legacy_record, 10000);
                new_record = measure(direct_record, 10000);
                old_view   = measure(legacy_view, 10000);
                new_view   = measure(direct_view, 10000);
            }
            std::cout << "LOCAL_METADATA_AB {\"case\":\"record_stream_primitive\",\"batch\":" << batch
                      << ",\"round\":" << round << ",\"baseline_us\":" << old_record
                      << ",\"candidate_us\":" << new_record << "}" << std::endl;
            std::cout << "LOCAL_METADATA_AB {\"case\":\"row_view_primitive\",\"batch\":" << batch
                      << ",\"round\":" << round << ",\"baseline_us\":" << old_view << ",\"candidate_us\":" << new_view
                      << "}" << std::endl;
        }
        for (int width : {257, 8193}) {
            auto history          = torch::arange(batch * width, storage.options()).view({batch, width});
            auto published        = history.select(1, width - 1).clone();
            auto full_readback    = [&] { return history.cpu(); };
            auto compact_readback = [&] { return published.cpu(); };
            ASSERT_TRUE(torch::equal(full_readback().select(1, width - 1), compact_readback()));
            for (int round = 0; round < 10; ++round) {
                double before, after;
                if (round % 2) {
                    after  = measure(compact_readback, 100);
                    before = measure(full_readback, 100);
                } else {
                    before = measure(full_readback, 100);
                    after  = measure(compact_readback, 100);
                }
                std::cout << "LOCAL_METADATA_AB {\"case\":\"prefill_readback_primitive\",\"batch\":" << batch
                          << ",\"width\":" << width << ",\"round\":" << round
                          << ",\"baseline_bytes\":" << batch * width * 4 << ",\"candidate_bytes\":" << batch * 4
                          << ",\"baseline_us\":" << before << ",\"candidate_us\":" << after << "}" << std::endl;
            }
        }
    }
}
}
}
