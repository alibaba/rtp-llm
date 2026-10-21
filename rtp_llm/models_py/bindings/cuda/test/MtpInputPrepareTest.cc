#include <ATen/cuda/CUDAContext.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/profiler_kineto.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <set>

#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

namespace rtp_llm {
namespace {

const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

struct Rows {
    std::vector<torch::Tensor> values, lengths, value_owners, length_owners;
};

Rows makeRows(int64_t batch, int width) {
    Rows result;
    // Reordered views from separate owners model requests carried across batches.
    for (int parent = 0; parent < 3; ++parent) {
        auto values = torch::arange((batch + 3) * width, i32).reshape({batch + 3, width}) + parent * 10000;
        result.value_owners.push_back(values);
        result.length_owners.push_back(torch::arange(batch + 3, i32).remainder(width) + 1);
    }
    for (int64_t row = 0; row < batch; ++row) {
        const int     parent = row % 3;
        const int64_t source = (row * 37 + 1) % (batch + 3);
        auto          values = result.value_owners[parent].narrow(0, source, 1);
        result.values.push_back(row % 2 ? values.select(0, 0) : values);
        result.lengths.push_back(result.length_owners[parent].narrow(0, source, 1));
    }
    return result;
}

// Match the observed scalar dtype composition without assuming D22's owners
// or the origin of its INT32 rows. Publication/narrow-view construction is
// outside the consumer timing; source-vector copies and collection remain in it.
Rows makeProposalRows(int batch, int i32_rows, bool normalized) {
    Rows result;
    auto raw             = torch::arange(batch, i32.dtype(torch::kInt64)).reshape({batch, 1});
    auto values          = normalized ? raw.to(torch::kInt32) : raw;
    auto initial         = raw.narrow(0, batch - i32_rows, i32_rows).to(torch::kInt32);
    result.value_owners  = {values, initial};
    result.length_owners = {torch::arange(batch, i32) + 500};
    for (int row = 0; row < batch; ++row) {
        result.values.push_back(row < batch - i32_rows ? values.narrow(0, row, 1) :
                                                         initial.narrow(0, row - batch + i32_rows, 1));
        result.lengths.push_back(result.length_owners[0].narrow(0, row, 1));
    }
    return result;
}

std::vector<torch::Tensor> columnsFor(int64_t batch, int width) {
    std::vector<torch::Tensor> columns(width);
    for (int column = 1; column < width; ++column) {
        columns[column] = torch::arange(batch, i32) + column * 100;
    }
    return columns;
}

torch::Tensor referenceFirstColumn(const Rows& rows) {
    std::vector<torch::Tensor> values, lengths;
    for (size_t row = 0; row < rows.values.size(); ++row) {
        values.push_back(rows.values[row].reshape({1, -1}));
        lengths.push_back(rows.lengths[row].reshape({1}));
    }
    auto indexes = (torch::cat(lengths).to(torch::kInt64) - 1).reshape({-1, 1});
    return torch::cat(values).gather(1, indexes).flatten().to(torch::kInt32);
}

void referenceDraftInput(const Rows& rows, torch::Tensor& tokens, torch::Tensor& lengths, torch::Tensor& indexes) {
    std::vector<torch::Tensor> values, next_lengths;
    for (size_t row = 0; row < rows.values.size(); ++row) {
        const auto& value = rows.values[row];
        values.push_back(value.select(-1, value.size(-1) - 1).reshape({1}));
        next_lengths.push_back(rows.lengths[row].reshape({1}));
    }
    tokens  = torch::cat(values).to(torch::kInt32);
    lengths = torch::cat(next_lengths).to(torch::kInt32) - 1;
    indexes = torch::arange(static_cast<int64_t>(rows.values.size()), i32);
}

void checkStatePack(const Rows&                       rows,
                    const std::vector<torch::Tensor>& columns,
                    const torch::Tensor&              tokens,
                    const torch::Tensor&              lengths,
                    const torch::Tensor&              indexes) {
    auto reference = columns;
    reference[0]   = referenceFirstColumn(rows);
    EXPECT_TRUE(torch::equal(tokens, torch::stack(reference, 1)));
    EXPECT_TRUE(torch::equal(lengths, torch::full_like(lengths, static_cast<int64_t>(columns.size()))));
    EXPECT_TRUE(torch::equal(indexes, torch::arange(indexes.numel(), i32)));
}

TEST(MtpInputPrepareTest, ReorderedRowsAndChunkBoundaries) {
    c10::InferenceMode         inference;
    const auto                 stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    for (int64_t batch : {1, 7, 109, 127, 128, 160, 257}) {
        for (int width : {4, 8}) {
            auto rows           = makeRows(batch, width);
            auto columns        = columnsFor(batch, width);
            auto tokens         = torch::empty({batch, width}, i32);
            auto lengths        = torch::empty({batch}, i32);
            auto indexes        = torch::empty({batch * width}, i32);
            auto before_values  = torch::stack(rows.value_owners).clone();
            auto before_lengths = torch::stack(rows.length_owners).clone();
            ASSERT_TRUE(tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(
                rows.values, rows.lengths, columns, tokens, lengths, indexes, width, stream.stream()));
            checkStatePack(rows, columns, tokens, lengths, indexes);
            EXPECT_TRUE(torch::equal(torch::stack(rows.value_owners), before_values));
            EXPECT_TRUE(torch::equal(torch::stack(rows.length_owners), before_lengths));
        }
    }
}

TEST(MtpInputPrepareTest, UnsupportedLateRowsDoNotWrite) {
    c10::InferenceMode inference;
    const auto         stream       = at::cuda::getCurrentCUDAStream().stream();
    auto               rows         = makeRows(160, 4);
    auto               columns      = columnsFor(160, 4);
    auto               tokens       = torch::full({160, 4}, -7, i32);
    auto               lengths      = torch::full({160}, -7, i32);
    auto               indexes      = torch::full({640}, -7, i32);
    const auto         original_row = rows.values.back();
    rows.values.back()              = original_row.to(torch::kInt64);
    EXPECT_FALSE(tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(
        rows.values, rows.lengths, columns, tokens, lengths, indexes, 4, stream));
    EXPECT_TRUE(torch::equal(tokens, torch::full_like(tokens, -7)));
    EXPECT_TRUE(torch::equal(lengths, torch::full_like(lengths, -7)));
    EXPECT_TRUE(torch::equal(indexes, torch::full_like(indexes, -7)));
    rows.values.back()  = original_row;
    rows.lengths.back() = rows.lengths.back().cpu();
    EXPECT_FALSE(tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(
        rows.values, rows.lengths, columns, tokens, lengths, indexes, 4, stream));
    EXPECT_FALSE(tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(
        rows.values, rows.lengths, columns, tokens, lengths, indexes, 9, stream));
    EXPECT_TRUE(torch::equal(tokens, torch::full_like(tokens, -7)));
}

TEST(MtpInputPrepareTest, AliasedStorageDoesNotWrite) {
    c10::InferenceMode inference;
    const auto         stream   = at::cuda::getCurrentCUDAStream().stream();
    auto               accepted = makeRows(7, 4);
    auto               columns  = columnsFor(7, 4);
    auto               storage  = torch::full({56}, -7, i32);
    auto               tokens   = storage.narrow(0, 0, 28).reshape({7, 4});
    auto               indexes  = storage.narrow(0, 28, 28);
    auto               lengths  = torch::full({7}, -7, i32);
    EXPECT_FALSE(tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(
        accepted.values, accepted.lengths, columns, tokens, lengths, indexes, 4, stream));
    EXPECT_TRUE(torch::equal(storage, torch::full_like(storage, -7)));
    EXPECT_TRUE(torch::equal(lengths, torch::full_like(lengths, -7)));
}

struct Graph {
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
    }
};

TEST(MtpInputPrepareTest, ReplayUsesChangedContentsAtTheSameAddresses) {
    c10::InferenceMode         inference;
    const auto                 stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto                       accepted = makeRows(257, 8);
    auto                       columns  = columnsFor(257, 8);
    auto                       tokens   = torch::empty({257, 8}, i32);
    auto                       lengths  = torch::empty({257}, i32);
    auto                       indexes  = torch::empty({257 * 8}, i32);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    Graph graph;
    C10_CUDA_CHECK(cudaStreamBeginCapture(stream.stream(), cudaStreamCaptureModeThreadLocal));
    const bool packed = tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(
        accepted.values, accepted.lengths, columns, tokens, lengths, indexes, 8, stream.stream());
    C10_CUDA_CHECK(cudaStreamEndCapture(stream.stream(), &graph.graph));
    ASSERT_TRUE(packed);
    C10_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph.executable, graph.graph, 0));
    for (int replay = 0; replay < 3; ++replay) {
        for (auto& owner : accepted.value_owners)
            owner.add_(101);
        for (auto& owner : accepted.length_owners)
            owner.copy_(owner.remainder(8) + 1);
        for (size_t column = 1; column < columns.size(); ++column)
            columns[column].add_(17);
        C10_CUDA_CHECK(cudaGraphLaunch(graph.executable, stream.stream()));
        checkStatePack(accepted, columns, tokens, lengths, indexes);
    }
}

// CUDA events measure the submitted multi-op path, including gaps caused by
// host submission. They are not a pure-kernel measurement. Both paths include
// output/intermediate allocations and copying the source Tensor vectors; the
// GenerateStream mutex/lookup is outside this fixture.
void measure(const std::string&           path,
             int                          batch,
             int                          width,
             int                          i32_rows,
             const std::function<void()>& baseline,
             const std::function<void()>& candidate) {
    constexpr int warmup = 30, iterations = 100;
    const auto    stream = at::cuda::getCurrentCUDAStream().stream();
    for (int round = 0; round < 2; ++round) {
        for (int slot = 0; slot < 2; ++slot) {
            const bool  use_candidate = (round == 0 ? slot : 1 - slot) != 0;
            const auto& operation     = use_candidate ? candidate : baseline;
            for (int i = 0; i < warmup; ++i)
                operation();
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaEvent_t begin, end;
            C10_CUDA_CHECK(cudaEventCreate(&begin));
            C10_CUDA_CHECK(cudaEventCreate(&end));
            const auto wall_begin = std::chrono::system_clock::now();
            C10_CUDA_CHECK(cudaEventRecord(begin, stream));
            const auto cpu_begin = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i)
                operation();
            const auto cpu_end = std::chrono::steady_clock::now();
            C10_CUDA_CHECK(cudaEventRecord(end, stream));
            C10_CUDA_CHECK(cudaEventSynchronize(end));
            const auto wall_end = std::chrono::system_clock::now();
            float      gpu_ms   = 0;
            C10_CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, begin, end));
            const double cpu_us = std::chrono::duration<double, std::micro>(cpu_end - cpu_begin).count() / iterations;
            std::cout << "{\"benchmark\":\"mtp_input_prepare\",\"path\":\"" << path << "\",\"mode\":\""
                      << (use_candidate ? "after" : "before") << "\",\"batch\":" << batch << ",\"width\":" << width
                      << ",\"i32_rows_before\":" << i32_rows << ",\"round\":" << round << ",\"warmup\":" << warmup
                      << ",\"iterations\":" << iterations << ",\"allocations_included\":true"
                      << ",\"source_owner_width\":" << width
                      << ",\"producer_included\":" << (path == "proposal_publish_i32" ? "true" : "false")
                      << ",\"measure_wall_begin_unix_ns\":"
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(wall_begin.time_since_epoch()).count()
                      << ",\"measure_wall_end_unix_ns\":"
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end.time_since_epoch()).count()
                      << ",\"source_vector_copy_included\":" << (path == "proposal_publish_i32" ? "false" : "true")
                      << ",\"cpu_submit_us\":" << cpu_us << ",\"cuda_event_path_us\":" << gpu_ms * 1000.0 / iterations
                      << "}" << std::endl;
            C10_CUDA_CHECK(cudaEventDestroy(begin));
            C10_CUDA_CHECK(cudaEventDestroy(end));
        }
    }
}

TEST(MtpInputPrepareTest, OptionalPathBenchmark) {
    const char* enabled = std::getenv("RTP_LLM_MTP_INPUT_PREPARE_BENCH");
    if (enabled == nullptr || std::string(enabled) != "1")
        GTEST_SKIP() << "opt-in path benchmark";
    c10::InferenceMode         inference;
    const auto                 stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    const char*                selected_batch = std::getenv("RTP_LLM_MTP_INPUT_PREPARE_BENCH_BATCH");
    for (int batch : {1, 7, 32, 109, 127, 128, 160, 257}) {
        if (selected_batch != nullptr && batch != std::atoi(selected_batch))
            continue;
        for (int width : {4, 8}) {
            const auto rows    = makeRows(batch, width);
            const auto columns = columnsFor(batch, width);
            auto       run     = [&](bool fused) {
                Rows collected;
                collected.values    = rows.values;
                collected.lengths   = rows.lengths;
                auto packed_columns = columns;
                auto tokens         = torch::empty({batch, width}, i32);
                auto lengths        = torch::empty({batch}, i32);
                auto indexes        = torch::empty({batch * width}, i32);
                if (fused) {
                    ASSERT_TRUE(tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(collected.values,
                                                                                     collected.lengths,
                                                                                     packed_columns,
                                                                                     tokens,
                                                                                     lengths,
                                                                                     indexes,
                                                                                     width,
                                                                                     stream.stream()));
                } else {
                    packed_columns[0] = referenceFirstColumn(collected);
                    invokeMtpSpecDecodeTokensMetadataPrepare(
                        packed_columns, tokens, lengths, indexes, width, stream.stream());
                }
            };
            measure("verify_pack", batch, width, 0, [&] { run(false); }, [&] { run(true); });
        }
        for (int i32_rows : {0, std::min(2, batch)}) {
            const auto before = makeProposalRows(batch, i32_rows, false);
            const auto after  = makeProposalRows(batch, i32_rows, true);

            torch::Tensor tokens_before, lengths_before, indexes_before;
            torch::Tensor tokens_after, lengths_after, indexes_after;
            referenceDraftInput(before, tokens_before, lengths_before, indexes_before);
            referenceDraftInput(after, tokens_after, lengths_after, indexes_after);
            ASSERT_TRUE(torch::equal(tokens_before, tokens_after));
            ASSERT_TRUE(torch::equal(lengths_before, lengths_after));
            ASSERT_TRUE(torch::equal(indexes_before, indexes_after));

            auto run = [&](const Rows& rows) {
                Rows collected;
                collected.values  = rows.values;
                collected.lengths = rows.lengths;
                torch::Tensor tokens, lengths, indexes;
                referenceDraftInput(collected, tokens, lengths, indexes);
            };
            measure("proposal_consume", batch, 1, i32_rows, [&] { run(before); }, [&] { run(after); });
        }
        // Separate producer cost; it is excluded from proposal_consume above.
        const auto raw = torch::arange(batch, i32.dtype(torch::kInt64)).reshape({batch, 1});
        measure(
            "proposal_publish_i32",
            batch,
            1,
            0,
            [&] {
                auto published = raw;
                (void)published;
            },
            [&] {
                auto published = raw.to(torch::kInt32);
                (void)published;
            });
    }
}

// A separate instrumented run supplies kernel durations/counts, not benchmark timing.
TEST(MtpInputPrepareTest, OptionalKernelProfile) {
    const char* path = std::getenv("RTP_MTP_INPUT_PROFILE");
    if (path == nullptr || path[0] != '/')
        GTEST_SKIP() << "set an absolute RTP_MTP_INPUT_PROFILE path";
    c10::InferenceMode         inference;
    const auto                 stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    const auto                 accepted = makeRows(109, 4);
    const auto                 before   = makeProposalRows(109, 2, false);
    const auto                 after    = makeProposalRows(109, 2, true);
    const auto                 raw      = torch::arange(109, i32.dtype(torch::kInt64)).reshape({109, 1});
    auto                       columns  = columnsFor(109, 4);
    auto                       tokens   = torch::empty({109, 4}, i32);
    auto                       lengths  = torch::empty({109}, i32);
    auto                       indexes  = torch::empty({436}, i32);
    auto                       run      = [&] {
        {
            RECORD_FUNCTION("mtp_input.verify.before", std::vector<c10::IValue>());
            columns[0] = referenceFirstColumn(accepted);
            invokeMtpSpecDecodeTokensMetadataPrepare(columns, tokens, lengths, indexes, 4, stream.stream());
        }
        {
            RECORD_FUNCTION("mtp_input.verify.after", std::vector<c10::IValue>());
            ASSERT_TRUE(tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(
                accepted.values, accepted.lengths, columns, tokens, lengths, indexes, 4, stream.stream()));
        }
        {
            RECORD_FUNCTION("mtp_input.draft.before", std::vector<c10::IValue>());
            torch::Tensor draft_tokens, draft_lengths, draft_indexes;
            referenceDraftInput(before, draft_tokens, draft_lengths, draft_indexes);
        }
        {
            RECORD_FUNCTION("mtp_input.draft.after", std::vector<c10::IValue>());
            torch::Tensor draft_tokens, draft_lengths, draft_indexes;
            referenceDraftInput(after, draft_tokens, draft_lengths, draft_indexes);
        }
        {
            RECORD_FUNCTION("mtp_input.proposal.normalize", std::vector<c10::IValue>());
            auto published = raw.to(torch::kInt32);
            (void)published;
        }
    };
    for (int i = 0; i < 10; ++i)
        run();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    namespace tap = torch::autograd::profiler;
    namespace tpi = torch::profiler::impl;
    tpi::ProfilerConfig               config(tpi::ProfilerState::KINETO, /*report_input_shapes=*/false);
    const std::set<tpi::ActivityType> activities{tpi::ActivityType::CPU, tpi::ActivityType::CUDA};
    tap::prepareProfiler(config, activities);
    tap::enableProfiler(config, activities);
    try {
        for (int i = 0; i < 5; ++i)
            run();
        C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    } catch (...) {
        tap::disableProfiler();
        throw;
    }
    auto result = tap::disableProfiler();
    ASSERT_NE(result, nullptr);
    result->save(path);
}

}  // namespace
}  // namespace rtp_llm
