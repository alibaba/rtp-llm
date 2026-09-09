#include <gtest/gtest.h>

#include <torch/cuda.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

namespace rtp_llm {
namespace {

class StubSpecProcessor: public SpecLogitsProcessor {
public:
    StubSpecProcessor(bool eligible, int cap, int masked_token): eligible_(eligible), cap_(cap), masked_token_(masked_token) {}

    bool isSpecVerifyEligible() const override {
        ++eligibility_calls_;
        return eligible_;
    }

    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        ++fill_calls_;
        draft_tokens_.assign(request.draft_tokens, request.draft_tokens + request.propose_step);
        if (masked_token_ >= 0) {
            for (int row = 0; row <= request.propose_step; ++row) {
                int32_t* row_ptr = request.bitmask_cpu_out + row * request.bitmask_size_int32;
                row_ptr[masked_token_ / 32] &= ~(1 << (masked_token_ % 32));
            }
        }
        return cap_;
    }

    int fillCalls() const {
        return fill_calls_;
    }

    int eligibilityCalls() const {
        return eligibility_calls_;
    }

    const std::vector<int32_t>& draftTokens() const {
        return draft_tokens_;
    }

private:
    bool eligible_;
    int  cap_;
    int  masked_token_;
    int  fill_calls_ = 0;
    mutable int eligibility_calls_ = 0;
    std::vector<int32_t> draft_tokens_;
};

SpecLogitsVerifyRunner::LaunchTask makeTask(const std::vector<SpecLogitsProcessorPtr>& processors,
                                            int                                        propose_step,
                                            size_t                                     vocab_size) {
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = processors.size();
    task.propose_step  = propose_step;
    task.vocab_size    = vocab_size;
    task.draft_tokens =
        torch::zeros({static_cast<int64_t>(processors.size()), propose_step}, torch::kInt32);
    for (size_t i = 0; i < processors.size(); ++i) {
        task.active.push_back({processors[i], /*stream_idx=*/i, /*processor_idx=*/0,
                               /*stream_id=*/100 + i, /*base_seq_len=*/4, /*base_output_len=*/2});
    }
    return task;
}

}  // namespace

TEST(SpecLogitsVerifyRunnerTest, MixedBatchSkipsIneligibleWithoutDroppingArtifact) {
    const int    P = 2;
    const size_t V = 64;

    auto eligible   = std::make_shared<StubSpecProcessor>(/*eligible=*/true, /*cap=*/1, /*masked_token=*/7);
    auto ineligible = std::make_shared<StubSpecProcessor>(/*eligible=*/false, /*cap=*/0, /*masked_token=*/3);

    SpecLogitsVerifyRunner runner;
    auto result = runner.buildInline(makeTask({eligible, ineligible}, P, V));

    EXPECT_TRUE(result.has_active_processor);
    EXPECT_EQ(result.skipped_ineligible_processors, 1u);
    ASSERT_EQ(result.applied_processors.size(), 1u);
    EXPECT_EQ(result.applied_processors[0].stream_id, 100u);
    EXPECT_EQ(eligible->fillCalls(), 1);
    EXPECT_EQ(ineligible->fillCalls(), 0);

    ASSERT_TRUE(result.spec_vocab_mask_gpu.defined());
    ASSERT_TRUE(result.spec_cap_gpu.defined());
    if (result.ready_event) {
        result.ready_event->synchronize();
    }
    auto mask = result.spec_vocab_mask_gpu.cpu();
    auto cap  = result.spec_cap_gpu.cpu();

    ASSERT_EQ(mask.sizes(), (torch::IntArrayRef{2 * (P + 1), static_cast<int64_t>(V)}));
    // Stream 0 (eligible): exactly token 7 masked in each of its P+1 rows.
    for (int row = 0; row <= P; ++row) {
        for (size_t tok = 0; tok < V; ++tok) {
            EXPECT_EQ(mask[row][static_cast<int64_t>(tok)].item<bool>(), tok == 7)
                << "stream0 row=" << row << " tok=" << tok;
        }
    }
    // Stream 1 (skipped): rows stay all-allow.
    for (int row = P + 1; row < 2 * (P + 1); ++row) {
        EXPECT_FALSE(mask[row].any().item<bool>()) << "stream1 row=" << row;
    }
    EXPECT_EQ(cap[0].item<int32_t>(), 1);
    EXPECT_EQ(cap[1].item<int32_t>(), P);
}

TEST(SpecLogitsVerifyRunnerTest, AllIneligibleReturnsNoArtifactWithSkipCount) {
    auto a = std::make_shared<StubSpecProcessor>(/*eligible=*/false, /*cap=*/0, /*masked_token=*/-1);
    auto b = std::make_shared<StubSpecProcessor>(/*eligible=*/false, /*cap=*/0, /*masked_token=*/-1);

    SpecLogitsVerifyRunner runner;
    auto result = runner.buildInline(makeTask({a, b}, /*propose_step=*/2, /*vocab_size=*/64));

    EXPECT_FALSE(result.has_active_processor);
    EXPECT_EQ(result.skipped_ineligible_processors, 2u);
    EXPECT_TRUE(result.applied_processors.empty());
    EXPECT_FALSE(result.spec_vocab_mask_gpu.defined());
    EXPECT_FALSE(result.spec_cap_gpu.defined());
    EXPECT_EQ(a->fillCalls(), 0);
    EXPECT_EQ(b->fillCalls(), 0);
}

TEST(SpecLogitsVerifyRunnerTest, EarlyCudaTransfersPackStridedInt32AndInt64BeforeSourceReuse) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "requires CUDA streams and pinned D2H";
    }
    const int P = 2;
    SpecLogitsVerifyRunner runner;
    std::vector<SpecLogitsVerifyRunner::LaunchTask> tasks;
    std::vector<std::vector<std::shared_ptr<StubSpecProcessor>>> processors;
    std::vector<std::shared_ptr<torch::Event>> mutation_done;

    for (int round = 0; round < 2; ++round) {
        const auto dtype = round == 0 ? torch::kInt32 : torch::kInt64;
        const int offset = round * 100;
        auto first  = std::make_shared<StubSpecProcessor>(true, P, -1);
        auto second = std::make_shared<StubSpecProcessor>(true, P, -1);
        auto task   = makeTask({first, second}, P, 64);
        const auto active = task.active;
        // Enqueue must work before the worker publishes/inspects processors.
        task.active.clear();
        torch::Stream producer = cuda_graph::graphGetStreamFromPool(false);
        task.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        {
            cuda_graph::GraphStreamGuard guard(cuda_graph::toGraphStream(producer));
            auto host = torch::tensor({{91, 92}, {offset + 10, offset + 20}, {offset + 11, offset + 21}},
                                      torch::TensorOptions().dtype(dtype));
            // Shape [B,P+1], with non-contiguous strides. Column zero is the
            // already-known token and must not reach the spec processor.
            task.draft_tokens = host.to(torch::kCUDA).transpose(0, 1);
            // The final values are produced by a GPU op on a non-copy stream;
            // the early transfer must honor the event recorded after it.
            task.draft_tokens.add_(1);
            task.draft_tokens_ready_event->record(producer);
        }
        ASSERT_FALSE(task.draft_tokens.is_contiguous());
        runner.enqueueDraftTokensToCpu(task);
        ASSERT_TRUE(task.draft_transfer);
        ASSERT_TRUE(task.draft_transfer->ready_event);
        EXPECT_EQ(task.draft_transfer->total_streams, 2u);
        EXPECT_EQ(task.draft_transfer->propose_step, P);
        EXPECT_EQ(task.draft_transfer->cpu_tokens.scalar_type(), torch::kInt32);
        EXPECT_TRUE(task.draft_transfer->cpu_tokens.is_contiguous());
        EXPECT_TRUE(task.draft_transfer->cpu_tokens.is_pinned());
        EXPECT_TRUE(task.draft_transfer->packed_tokens.is_contiguous());
        EXPECT_EQ(task.draft_transfer->packed_tokens.scalar_type(), torch::kInt32);
        EXPECT_EQ(first->eligibilityCalls(), 0);
        EXPECT_EQ(second->eligibilityCalls(), 0);

        auto mutated = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        {
            cuda_graph::GraphStreamGuard guard(cuda_graph::toGraphStream(producer));
            task.draft_transfer->ready_event->block(producer);
            task.draft_tokens.fill_(-999);
            mutated->record(producer);
        }
        task.active = active;
        tasks.push_back(std::move(task));
        processors.push_back({first, second});
        mutation_done.push_back(std::move(mutated));
    }

    ASSERT_NE(tasks[0].draft_transfer.get(), tasks[1].draft_transfer.get());
    ASSERT_NE(tasks[0].draft_transfer->cpu_tokens.data_ptr<int32_t>(),
              tasks[1].draft_transfer->cpu_tokens.data_ptr<int32_t>());
    // Test-only waits make the later source overwrite deterministic. Enqueue
    // itself must not perform these waits on the caller.
    for (const auto& done : mutation_done) {
        done->synchronize();
    }
    // Consume out of order: neither round may read the runner's latest scratch
    // or rematerialize the now-overwritten source tensor.
    for (int round = 1; round >= 0; --round) {
        SCOPED_TRACE(round);
        const int offset = round * 100;
        auto result = runner.buildInline(tasks[round]);
        EXPECT_TRUE(result.has_active_processor);
        EXPECT_EQ(processors[round][0]->draftTokens(), (std::vector<int32_t>{offset + 11, offset + 12}));
        EXPECT_EQ(processors[round][1]->draftTokens(), (std::vector<int32_t>{offset + 21, offset + 22}));
        ASSERT_TRUE(result.ready_event);
        result.ready_event->synchronize();
    }
}

TEST(SpecLogitsVerifyRunnerTest, EarlyPinnedSlotReusesStorageAcrossRoundsAndResizesSafely) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "requires pinned CUDA transfers";
    }
    SpecLogitsVerifyRunner runner;
    void* first_storage = nullptr;
    for (int round = 0; round < 5; ++round) {
        auto processor = std::make_shared<StubSpecProcessor>(true, 3, -1);
        const int P = round == 1 ? 2 : (round >= 3 ? 6 : 3);
        auto task = makeTask({processor}, P, 64);
        task.draft_tokens.fill_(10 + round);
        task.draft_tokens = task.draft_tokens.to(torch::kCUDA);
        task.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        task.draft_tokens_ready_event->record(cuda_graph::graphGetCurrentStream());
        runner.enqueueDraftTokensToCpu(task);
        ASSERT_EQ(task.draft_transfer->cpu_tokens.sizes(), (torch::IntArrayRef{1, P}));
        auto storage = task.draft_transfer->cpu_tokens.data_ptr<int32_t>();
        if (round == 0 || round == 3) {
            first_storage = storage;
        } else {
            EXPECT_EQ(storage, first_storage);
        }
        auto result = runner.buildInline(task);
        EXPECT_EQ(processor->draftTokens(), (std::vector<int32_t>(P, 10 + round)));
        result.ready_event->synchronize();
        // task is destroyed here, releasing the lease after CPU consumption.
    }
}

TEST(SpecLogitsVerifyRunnerTest, EarlyPinnedSlotsDoNotOverwriteRetainedTransfersOrCpuAliases) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "requires pinned CUDA transfers";
    }
    SpecLogitsVerifyRunner runner;
    std::vector<SpecLogitsVerifyRunner::LaunchTask> tasks;
    for (int round = 0; round < 3; ++round) {
        auto processor = std::make_shared<StubSpecProcessor>(true, 2, -1);
        auto task = makeTask({processor}, 2, 64);
        task.draft_tokens.fill_(20 + round);
        task.draft_tokens = task.draft_tokens.to(torch::kCUDA);
        task.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        task.draft_tokens_ready_event->record(cuda_graph::graphGetCurrentStream());
        runner.enqueueDraftTokensToCpu(task);
        // Complete each copy/CPU read, but keep all transfer leases alive.
        auto result = runner.buildInline(task);
        result.ready_event->synchronize();
        for (const auto& earlier : tasks) {
            EXPECT_NE(earlier.draft_transfer->cpu_tokens.data_ptr<int32_t>(),
                      task.draft_transfer->cpu_tokens.data_ptr<int32_t>());
        }
        tasks.push_back(std::move(task));
    }
    // Keep a Tensor alias while dropping its transfer owner.
    auto alias = tasks[0].draft_transfer->cpu_tokens;
    tasks[0] = {};
    auto processor = std::make_shared<StubSpecProcessor>(true, 2, -1);
    auto task = makeTask({processor}, 2, 64);
    task.draft_tokens.fill_(99);
    task.draft_tokens = task.draft_tokens.to(torch::kCUDA);
    task.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    task.draft_tokens_ready_event->record(cuda_graph::graphGetCurrentStream());
    runner.enqueueDraftTokensToCpu(task);
    EXPECT_NE(alias.data_ptr<int32_t>(), task.draft_transfer->cpu_tokens.data_ptr<int32_t>());
    auto result = runner.buildInline(task);
    result.ready_event->synchronize();
    EXPECT_TRUE(alias.eq(20).all().item<bool>());
    EXPECT_TRUE(tasks[1].draft_transfer->cpu_tokens.eq(21).all().item<bool>());
    EXPECT_TRUE(tasks[2].draft_transfer->cpu_tokens.eq(22).all().item<bool>());
}

TEST(SpecLogitsVerifyRunnerTest, DroppedEarlyTransferIsNotReusedWithoutWorkerCompletion) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "requires pinned CUDA transfers";
    }
    SpecLogitsVerifyRunner runner;
    auto processor = std::make_shared<StubSpecProcessor>(true, 2, -1);
    auto task = makeTask({processor}, 2, 64);
    task.draft_tokens.fill_(31);
    task.draft_tokens = task.draft_tokens.to(torch::kCUDA);
    task.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    task.draft_tokens_ready_event->record(cuda_graph::graphGetCurrentStream());
    runner.enqueueDraftTokensToCpu(task);
    auto abandoned_cpu = task.draft_transfer->cpu_tokens;
    auto abandoned_done = task.draft_transfer->ready_event;
    task = {};  // No buildInline, just as for a dropped/skipped task.

    auto next = makeTask({processor}, 2, 64);
    next.draft_tokens.fill_(42);
    next.draft_tokens = next.draft_tokens.to(torch::kCUDA);
    next.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    next.draft_tokens_ready_event->record(cuda_graph::graphGetCurrentStream());
    runner.enqueueDraftTokensToCpu(next);
    EXPECT_NE(abandoned_cpu.data_ptr<int32_t>(), next.draft_transfer->cpu_tokens.data_ptr<int32_t>());
    auto result = runner.buildInline(next);
    result.ready_event->synchronize();
    abandoned_done->synchronize();
    EXPECT_TRUE(abandoned_cpu.eq(31).all().item<bool>());
    EXPECT_EQ(processor->draftTokens(), (std::vector<int32_t>(2, 42)));
}

TEST(SpecLogitsVerifyRunnerTest, CpuTransferOwnsSnapshotAndLegacyPathStillReadsCurrentTokens) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "runner artifacts require CUDA even with CPU token input";
    }
    SpecLogitsVerifyRunner runner;
    auto first  = std::make_shared<StubSpecProcessor>(true, 2, -1);
    auto second = std::make_shared<StubSpecProcessor>(true, 2, -1);
    auto old_path = std::make_shared<StubSpecProcessor>(true, 2, -1);
    auto first_task = makeTask({first}, 2, 64);
    first_task.draft_tokens = torch::tensor({{90, 11, 12}}, torch::kInt64);
    runner.enqueueDraftTokensToCpu(first_task);
    ASSERT_TRUE(first_task.draft_transfer);
    EXPECT_FALSE(first_task.draft_transfer->ready_event);
    EXPECT_TRUE(first_task.draft_transfer->cpu_tokens.device().is_cpu());
    EXPECT_EQ(first_task.draft_transfer->cpu_tokens.scalar_type(), torch::kInt32);

    first_task.draft_tokens.fill_(31);
    auto second_task = makeTask({second}, 2, 64);
    second_task.draft_tokens = first_task.draft_tokens;
    runner.enqueueDraftTokensToCpu(second_task);
    ASSERT_TRUE(second_task.draft_transfer);
    ASSERT_NE(first_task.draft_transfer->cpu_tokens.data_ptr<int32_t>(),
              second_task.draft_transfer->cpu_tokens.data_ptr<int32_t>());
    first_task.draft_tokens.fill_(44);

    auto second_result = runner.buildInline(second_task);
    EXPECT_EQ(second->draftTokens(), (std::vector<int32_t>{31, 31}));
    auto first_result = runner.buildInline(first_task);
    EXPECT_EQ(first->draftTokens(), (std::vector<int32_t>{11, 12}));
    auto legacy_task = makeTask({old_path}, 2, 64);
    legacy_task.draft_tokens = first_task.draft_tokens;
    ASSERT_FALSE(legacy_task.draft_transfer);
    auto legacy_result = runner.buildInline(legacy_task);
    EXPECT_EQ(old_path->draftTokens(), (std::vector<int32_t>{44, 44}));
    for (const auto* result : {&first_result, &second_result, &legacy_result}) {
        ASSERT_TRUE(result->ready_event);
        result->ready_event->synchronize();
    }
}

TEST(SpecLogitsVerifyRunnerTest, EarlyTransferDoesNotInspectAllIneligibleProcessors) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "runner owns a CUDA copy stream";
    }
    auto first  = std::make_shared<StubSpecProcessor>(false, 0, -1);
    auto second = std::make_shared<StubSpecProcessor>(false, 0, -1);
    SpecLogitsVerifyRunner runner;
    auto task = makeTask({first, second}, 2, 64);
    runner.enqueueDraftTokensToCpu(task);
    ASSERT_TRUE(task.draft_transfer);
    EXPECT_EQ(first->eligibilityCalls(), 0);
    EXPECT_EQ(second->eligibilityCalls(), 0);
    auto result = runner.buildInline(task);
    EXPECT_FALSE(result.has_active_processor);
    EXPECT_EQ(result.skipped_ineligible_processors, 2u);
    EXPECT_TRUE(result.applied_processors.empty());
    EXPECT_FALSE(result.spec_vocab_mask_gpu.defined());
    EXPECT_FALSE(result.spec_cap_gpu.defined());
    EXPECT_EQ(first->fillCalls(), 0);
    EXPECT_EQ(second->fillCalls(), 0);
}

TEST(SpecLogitsVerifyRunnerTest, EarlyTransferRejectsInvalidLayoutDtypeMissingReadyAndDuplicateEnqueue) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "runner owns a CUDA copy stream";
    }
    SpecLogitsVerifyRunner runner;
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = 2;
    task.propose_step = 2;
    task.draft_tokens = torch::zeros({5}, torch::kInt32);
    EXPECT_ANY_THROW(runner.enqueueDraftTokensToCpu(task));
    EXPECT_FALSE(task.draft_transfer);
    task.draft_tokens = torch::zeros({2, 1}, torch::kInt32);
    EXPECT_ANY_THROW(runner.enqueueDraftTokensToCpu(task));
    EXPECT_FALSE(task.draft_transfer);
    task.draft_tokens = torch::zeros({2, 2}, torch::kFloat32);
    EXPECT_ANY_THROW(runner.enqueueDraftTokensToCpu(task));
    EXPECT_FALSE(task.draft_transfer);
    task.draft_tokens = torch::zeros({2, 2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    EXPECT_ANY_THROW(runner.enqueueDraftTokensToCpu(task));
    EXPECT_FALSE(task.draft_transfer);
    task.draft_tokens = torch::zeros({2, 2}, torch::kInt32);
    task.propose_step = -1;
    EXPECT_ANY_THROW(runner.enqueueDraftTokensToCpu(task));
    EXPECT_FALSE(task.draft_transfer);
    task.propose_step = 2;
    runner.enqueueDraftTokensToCpu(task);
    ASSERT_TRUE(task.draft_transfer);
    auto original = task.draft_transfer;
    EXPECT_ANY_THROW(runner.enqueueDraftTokensToCpu(task));
    EXPECT_EQ(task.draft_transfer, original);
}

}  // namespace rtp_llm
