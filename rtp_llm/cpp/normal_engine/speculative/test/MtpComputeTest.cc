#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

template<typename T>
std::vector<T> toVec(const torch::Tensor& t) {
    auto c = t.is_cuda() ? t.cpu().contiguous() : t.contiguous();
    return std::vector<T>(c.data_ptr<T>(), c.data_ptr<T>() + c.numel());
}

class MtpComputeTest: public DeviceTestBase {};

#if USING_CUDA
namespace {

/** Keep GPU work pending until the test releases it; timeout also drains a failing test. */
class CudaStreamGate {
public:
    explicit CudaStreamGate(const torch::Stream& stream): stream_(stream) {
        const auto error = cudaLaunchHostFunc(cuda_graph::toGraphStream(stream).stream(), [](void* arg) {
            auto& gate = *static_cast<CudaStreamGate*>(arg);
            std::unique_lock<std::mutex> lock(gate.mutex_);
            gate.entered_ = true;
            gate.cv_.notify_all();
            gate.timed_out_ = !gate.cv_.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released_; });
        }, this);
        TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
    }

    ~CudaStreamGate() {
        release();
        stream_.synchronize();
    }

    bool waitUntilEntered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [&] { return entered_; });
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

    bool timedOut() const {
        return timed_out_.load();
    }

private:
    torch::Stream           stream_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    entered_ = false;
    bool                    released_ = false;
    std::atomic<bool>       timed_out_{false};
};

/** Sampling values are fixed so only the real cap, transfer and event protocol is under test. */
class CapInputSampler: public speculative::SpeculativeSampler {
public:
    CapInputSampler(): speculative::SpeculativeSampler({}, 2) {
        tokens_  = torch::tensor({{1, 2, 10}}, torch::kInt32).cuda();
        lengths_ = torch::tensor({3}, torch::kInt32).cuda();
    }

    speculative::SpeculativeSamplerOutput forward(const speculative::SpeculativeSamplingParams&,
                                                   SamplerOutput&,
                                                   SamplerOutput&) override {
        speculative::SpeculativeSamplerOutput output;
        output.accept_tokens = tokens_;
        output.accept_len = lengths_;
        return output;
    }

private:
    torch::Tensor tokens_;
    torch::Tensor lengths_;
};

class RoundCapProcessor: public BaseLogitsProcessor {
public:
    explicit RoundCapProcessor(int cap): cap_(cap) {}
    std::optional<ErrorInfo> process(const SamplerInputs&, size_t, size_t) override {
        return std::nullopt;
    }
    void updateMultiSeqStatus(const std::vector<int>&) override {}
    std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t) override {
        return std::nullopt;
    }
    MtpProcessorCapability mtpCapability() const override {
        return {MtpProcessorMode::SPEC_VERIFY, "round cap test"};
    }
    ErrorResult<int> prepareSpeculative(const SpecLogitsProcessorRequest&) override {
        ++calls;
        return ErrorResult<int>(static_cast<int>(cap_));
    }
    std::atomic<int> calls{0};

private:
    int cap_;
};

speculative::SpeculativeSamplerOutput consumeCap(const torch::Stream& stream,
                                                 CapInputSampler&     sampler,
                                                 SamplerOutput&       target,
                                                 const SpecLogitsVerifyRunner::LaunchResult& result) {
    cuda_graph::GraphStreamGuard guard(cuda_graph::toGraphStream(stream));
    SamplerOutput draft;
    speculative::SpeculativeSamplerOutput output;
    mtp::runRejectionSampling(sampler, {}, draft, target, result, output);
    return output;
}

void expectCappedOutput(const speculative::SpeculativeSamplerOutput& output, int cap) {
    output.transfer_done_event->synchronize();
    const std::vector<int32_t> tokens = cap == 0 ? std::vector<int32_t>{8, 2, 10} :
                                                 std::vector<int32_t>{1, 9, 10};
    EXPECT_EQ(toVec<int32_t>(output.accept_len_cpu), (std::vector<int32_t>{cap + 1}));
    EXPECT_EQ(toVec<int32_t>(output.accept_len), (std::vector<int32_t>{cap + 1}));
    EXPECT_EQ(toVec<int32_t>(output.accept_tokens_cpu), tokens);
    EXPECT_EQ(toVec<int32_t>(output.accept_tokens), tokens);
}

}

TEST_F(MtpComputeTest, RejectionWaitsForCapProducedOnAnotherStream) {
    const auto producer = cuda_graph::graphGetStreamFromPool(false);
    const auto consumer = cuda_graph::graphGetStreamFromPool(false);
    ASSERT_NE(producer, consumer);
    CapInputSampler sampler;
    SamplerOutput target;
    target.token_ids = torch::tensor({{8}, {9}, {10}}, torch::kInt32).cuda();
    SpecLogitsVerifyRunner::LaunchResult result;
    result.spec_cap_gpu = torch::tensor({1}, torch::kInt32).cuda();
    result.ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    result.consumed_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    {
        cuda_graph::GraphStreamGuard guard(cuda_graph::toGraphStream(producer));
        /** Load the producer kernel and initialize its event before holding GPU work pending. */
        result.spec_cap_gpu.fill_(1);
        result.ready_event->record(producer);
    }
    cuda_graph::graphDeviceSynchronize();
    /** Warm allocator/kernel paths before blocking a stream; setup sync is outside the tested interval. */
    consumeCap(consumer, sampler, target, result).transfer_done_event->synchronize();

    CudaStreamGate gate(producer);
    ASSERT_TRUE(gate.waitUntilEntered());
    {
        cuda_graph::GraphStreamGuard guard(cuda_graph::toGraphStream(producer));
        result.spec_cap_gpu.fill_(0);
        result.ready_event->record(producer);
    }
    std::promise<void> started;
    auto completed = std::async(std::launch::async, [&] {
        started.set_value();
        auto output = consumeCap(consumer, sampler, target, result);
        output.transfer_done_event->synchronize();
        return output;
    });
    started.get_future().wait();
    EXPECT_EQ(completed.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    gate.release();
    expectCappedOutput(completed.get(), 0);
    EXPECT_FALSE(gate.timedOut());
}

TEST_F(MtpComputeTest, RejectionConsumptionProtectsRunnerBuffersAcrossRounds) {
    const auto producer = cuda_graph::graphGetStreamFromPool(false);
    const auto consumer = cuda_graph::graphGetStreamFromPool(false);
    ASSERT_NE(producer, consumer);
    CapInputSampler sampler;
    SamplerOutput target;
    target.token_ids = torch::tensor({{8}, {9}, {10}}, torch::kInt32).cuda();
    auto first_processor = std::make_shared<RoundCapProcessor>(1);
    auto next_processor = std::make_shared<RoundCapProcessor>(0);
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = 1;
    task.propose_step = 2;
    task.vocab_size = 16;
    task.draft_tokens = torch::tensor({{1, 2}}, torch::kInt32);
    task.active = {{first_processor, 0}};
    SpecLogitsVerifyRunner runner;
    auto produce = [&](const SpecLogitsVerifyRunner::LaunchTask& input) {
        cuda_graph::GraphStreamGuard guard(cuda_graph::toGraphStream(producer));
        return runner.run(input);
    };
    cuda_graph::graphDeviceSynchronize();
    auto warmup = produce(task);
    consumeCap(consumer, sampler, target, warmup).transfer_done_event->synchronize();
    auto first = produce(task);
    first.ready_event->synchronize();
    const auto* cpu_storage = first.spec_cap_cpu.data_ptr();
    const auto* gpu_storage = first.spec_cap_gpu.data_ptr();

    CudaStreamGate gate(consumer);
    ASSERT_TRUE(gate.waitUntilEntered());
    auto first_output = consumeCap(consumer, sampler, target, first);
    EXPECT_FALSE(first.consumed_event->query());
    auto next_task = task;
    next_task.active = {{next_processor, 0}};
    std::promise<void> started;
    auto next = std::async(std::launch::async, [&] {
        started.set_value();
        return produce(next_task);
    });
    started.get_future().wait();
    EXPECT_EQ(next.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    EXPECT_EQ(next_processor->calls.load(), 0);
    gate.release();
    auto second = next.get();
    expectCappedOutput(first_output, 1);
    EXPECT_EQ(second.spec_cap_cpu.data_ptr(), cpu_storage);
    EXPECT_EQ(second.spec_cap_gpu.data_ptr(), gpu_storage);
    expectCappedOutput(consumeCap(consumer, sampler, target, second), 0);
    EXPECT_EQ(first_processor->calls.load(), 2);
    EXPECT_EQ(next_processor->calls.load(), 1);
    EXPECT_FALSE(gate.timedOut());
}
#endif

TEST_F(MtpComputeTest, DSparkPublicationReportsPeerFailureAfterBothLocalWaits) {
    struct PublicationModel: public ModelBase {
        GptModelOutputs forward(const GptModelInputs&) override {
            return {};
        }

        std::string waitCacheStorePublication() override {
            ++wait_count;
            return {};
        }

        int wait_count = 0;
    };

    PublicationModel  target;
    PublicationModel  draft;
    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 2;
    torch::Tensor status;
    int           reductions = 0;
    const auto error = mtp::finishDSparkCachePublication(
        &target, &draft, parallelism_config, cuda_graph::graphGetCurrentStream(), status, [&](bool local_ok) {
            EXPECT_EQ(target.wait_count, 1);
            EXPECT_EQ(draft.wait_count, 1);
            EXPECT_TRUE(local_ok);
            ++reductions;
            /** Local publication succeeded, but another TP rank failed. */
            return false;
        });

    EXPECT_EQ(reductions, 1);
    EXPECT_FALSE(error.empty());
}

TEST_F(MtpComputeTest, SharedHiddenOverridePreservesViewsAndFallbacks) {
    struct RecordingHiddenModel: public ModelBase {
        GptModelOutputs forward(const GptModelInputs&) override {
            return {};
        }

        torch::Tensor getMtpTargetHiddenStates(int64_t rows) override {
            requests.push_back(rows);
            if (throw_on_access) {
                throw std::runtime_error("MTP hidden accessor failed");
            }
            if (!buffer.defined() || buffer.numel() == 0) {
                return buffer;
            }
            return buffer.narrow(0, 0, rows < 0 ? local_rows : rows);
        }

        torch::Tensor        buffer;
        std::vector<int64_t> requests;
        int64_t              local_rows      = 2;
        bool                 throw_on_access = false;
    };

    for (bool device_input : {false, true}) {
        SCOPED_TRACE(device_input);
        const auto device  = device_input ? torch::kCUDA : torch::kCPU;
        auto       options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        RecordingHiddenModel model;
        model.buffer = torch::arange(48, options).reshape({8, 6}).slice(1, 1, 5, 2);
        auto fallback = torch::arange(24, options.dtype(torch::kFloat64)).reshape({4, 6}).slice(1, 1, 5, 2);
        const auto fallback_before = fallback.clone();

        GptModelOutputs output;
        output.all_hidden_states = fallback;
        mtp::maybeOverrideLastHiddenWithMtpBuffer(output, model, 3);
        auto resolved = output.all_hidden_states;
        EXPECT_EQ(model.requests, (std::vector<int64_t>{3}));
        EXPECT_EQ(resolved.sizes().vec(), (std::vector<int64_t>{3, 2}));
        EXPECT_EQ(resolved.strides().vec(), model.buffer.strides().vec());
        EXPECT_EQ(resolved.storage_offset(), model.buffer.storage_offset());
        EXPECT_EQ(resolved.scalar_type(), model.buffer.scalar_type());
        EXPECT_EQ(resolved.device(), model.buffer.device());
        EXPECT_EQ(resolved.data_ptr(), model.buffer.data_ptr());
        EXPECT_TRUE(torch::equal(resolved, model.buffer.narrow(0, 0, 3)));

        /** Fake draft execution writes in place and must retain the model buffer alias. */
        resolved.zero_();
        EXPECT_TRUE(torch::equal(model.buffer.narrow(0, 0, 3), torch::zeros_like(resolved)));
        EXPECT_TRUE(torch::equal(fallback, fallback_before));

        GptModelOutputs cp_output;
        mtp::maybeOverrideLastHiddenWithMtpBuffer(cp_output, model, -1);
        const auto& cp_hidden = cp_output.all_hidden_states;
        EXPECT_EQ(cp_hidden.sizes().vec(), (std::vector<int64_t>{2, 2}));
        EXPECT_EQ(cp_hidden.data_ptr(), model.buffer.data_ptr());
        EXPECT_EQ(model.requests, (std::vector<int64_t>{3, -1}));

        /* hidden_rows=0 takes the row count from a nonempty fallback and skips the getter otherwise. */
        model.requests.clear();
        GptModelOutputs inferred;
        mtp::maybeOverrideLastHiddenWithMtpBuffer(inferred, model, 0);
        EXPECT_TRUE(model.requests.empty());
        EXPECT_FALSE(inferred.all_hidden_states.defined());

        inferred.all_hidden_states = torch::empty({0, 2}, options);
        mtp::maybeOverrideLastHiddenWithMtpBuffer(inferred, model, 0);
        EXPECT_TRUE(model.requests.empty());
        EXPECT_EQ(inferred.all_hidden_states.numel(), 0);

        inferred.all_hidden_states = torch::zeros({3, 2}, options);
        mtp::maybeOverrideLastHiddenWithMtpBuffer(inferred, model, 0);
        EXPECT_EQ(model.requests, (std::vector<int64_t>{3}));
        EXPECT_EQ(inferred.all_hidden_states.sizes().vec(), (std::vector<int64_t>{3, 2}));
        EXPECT_EQ(inferred.all_hidden_states.data_ptr(), model.buffer.data_ptr());

        /* An explicit row count wins over a smaller fallback. */
        model.requests.clear();
        GptModelOutputs counted;
        counted.all_hidden_states = torch::zeros({2, 2}, options);
        mtp::maybeOverrideLastHiddenWithMtpBuffer(counted, model, 4);
        EXPECT_EQ(model.requests, (std::vector<int64_t>{4}));
        EXPECT_EQ(counted.all_hidden_states.sizes().vec(), (std::vector<int64_t>{4, 2}));
        EXPECT_EQ(counted.all_hidden_states.data_ptr(), model.buffer.data_ptr());

        for (const auto& missing : {torch::Tensor(), torch::empty({0, 2}, options)}) {
            model.buffer = missing;
            output.all_hidden_states = fallback;
            mtp::maybeOverrideLastHiddenWithMtpBuffer(output, model, 3);
            EXPECT_EQ(output.all_hidden_states.unsafeGetTensorImpl(), fallback.unsafeGetTensorImpl());
            EXPECT_TRUE(torch::equal(output.all_hidden_states, fallback_before));
            cp_output.all_hidden_states = torch::Tensor();
            mtp::maybeOverrideLastHiddenWithMtpBuffer(cp_output, model, -1);
            EXPECT_FALSE(cp_output.all_hidden_states.defined());

            /* The default row count still queries the getter with the fallback's rows. */
            model.requests.clear();
            GptModelOutputs defaulted;
            defaulted.all_hidden_states = fallback;
            mtp::maybeOverrideLastHiddenWithMtpBuffer(defaulted, model);
            EXPECT_EQ(model.requests, (std::vector<int64_t>{4}));
            EXPECT_EQ(defaulted.all_hidden_states.unsafeGetTensorImpl(), fallback.unsafeGetTensorImpl());
        }

        model.throw_on_access = true;
        output.all_hidden_states = fallback;
        EXPECT_THROW(mtp::maybeOverrideLastHiddenWithMtpBuffer(output, model, 3), std::runtime_error);
        EXPECT_EQ(output.all_hidden_states.unsafeGetTensorImpl(), fallback.unsafeGetTensorImpl());
        EXPECT_TRUE(torch::equal(fallback, fallback_before));

        /** A graph replay's explicit output must win even when the model-side getter is stale or unavailable. */
        model.requests.clear();
        const auto explicit_hidden = fallback.narrow(0, 1, 3);
        for (int64_t rows : {3, -1}) {
            output.all_hidden_states        = torch::Tensor();
            output.mtp_target_hidden_states = explicit_hidden;
            mtp::maybeOverrideLastHiddenWithMtpBuffer(output, model, rows);
            EXPECT_EQ(output.all_hidden_states.unsafeGetTensorImpl(), explicit_hidden.unsafeGetTensorImpl());
            EXPECT_TRUE(model.requests.empty());
        }
        EXPECT_ANY_THROW(mtp::maybeOverrideLastHiddenWithMtpBuffer(output, model, 2));
        EXPECT_TRUE(model.requests.empty());
    }
}

TEST_F(MtpComputeTest, PrefillShiftsTokensAndPositions) {
    const auto host_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    for (const auto& options : {host_options, host_options.pinned_memory(true), host_options.device(torch::kCUDA)}) {
        const auto device           = options.device();
        const bool device_input     = device.is_cuda();
        const auto position_options = host_options.pinned_memory(device_input || options.pinned_memory());
        SCOPED_TRACE(::testing::Message() << "device=" << device << " pinned=" << options.pinned_memory());
        GptModelInputs draft_input;
        draft_input.combo_tokens       = torch::tensor({11, 21, 22}, options);
        draft_input.input_lengths      = torch::tensor({1, 2}, options);
        draft_input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11, 12, 11, 12, 13}, position_options);
        const auto tokens_alias    = draft_input.combo_tokens;
        const auto lengths_alias   = draft_input.input_lengths;
        const auto positions_alias = draft_input.combo_position_ids;
        auto sampled_tokens        = torch::tensor({{-1, 12}, {-1, 23}}, torch::kInt32).to(device);
        auto next_positions        = torch::tensor({6, 7, 8, 12, 13, 14}, position_options);
        auto hidden_states         = torch::arange(6, torch::kFloat32).reshape({3, 2}).cuda();
        TensorHolder holder;

        mtp::prepareDraftPrefillAfterTargetPrefill(draft_input, hidden_states, sampled_tokens, next_positions, 3, holder);

        EXPECT_EQ(toVec<int32_t>(draft_input.combo_tokens), (std::vector<int32_t>{12, 22, 23}));
        EXPECT_EQ(toVec<int32_t>(draft_input.input_lengths), (std::vector<int32_t>{1, 2}));
        EXPECT_EQ(toVec<int32_t>(draft_input.combo_position_ids),
                  (std::vector<int32_t>{6, 7, 8, 11, 12, 13, 12, 13, 14}));
        EXPECT_TRUE(draft_input.combo_tokens.is_cuda());
        EXPECT_TRUE(draft_input.input_lengths.is_cuda());
        EXPECT_TRUE(draft_input.combo_position_ids.device().is_cpu());
        EXPECT_EQ(draft_input.combo_position_ids.data_ptr(), positions_alias.data_ptr());
        EXPECT_EQ(draft_input.combo_position_ids.is_pinned(), position_options.pinned_memory());
        EXPECT_EQ(draft_input.last_hidden_states.data_ptr(), hidden_states.data_ptr());
        if (device_input) {
            EXPECT_EQ(toVec<int32_t>(tokens_alias), (std::vector<int32_t>{11, 21, 22}));
            EXPECT_NE(draft_input.combo_tokens.data_ptr(), tokens_alias.data_ptr());
            EXPECT_NE(draft_input.input_lengths.data_ptr(), lengths_alias.data_ptr());
            const auto holds_pinned_source = [&](const std::vector<int32_t>& values) {
                for (const auto& source : holder.tensors) {
                    if (source.device().is_cpu() && source.is_pinned() && toVec<int32_t>(source) == values) {
                        return true;
                    }
                }
                return false;
            };
            EXPECT_TRUE(holds_pinned_source({1, 2}));
            EXPECT_TRUE(holds_pinned_source({12, 22, 23}));
        } else {
            EXPECT_EQ(toVec<int32_t>(tokens_alias), (std::vector<int32_t>{12, 22, 23}));
        }
    }
}

TEST_F(MtpComputeTest, PrefillAlignsMultimodalFieldsWithShiftedTokens) {
    for (bool device_input : {false, true}) {
        for (bool starts_at_boundary : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "device_input=" << device_input
                                              << " starts_at_boundary=" << starts_at_boundary);
            const auto device = device_input ? torch::kCUDA : torch::kCPU;
            const int rows    = starts_at_boundary ? 2 : 1;
            const std::vector<int32_t> masks = starts_at_boundary ? std::vector<int32_t>{0, 0, 1, 0, 0} :
                                                                  std::vector<int32_t>{1, 0, 1, 1, 0};
            const std::vector<int32_t> types = starts_at_boundary ? std::vector<int32_t>{1, 1, 0, 1, 1} :
                                                                  std::vector<int32_t>{0, 1, 0, 0, 1};
            const std::vector<int32_t> locs = starts_at_boundary ? std::vector<int32_t>{0, 3} :
                                                                 std::vector<int32_t>{1, 4};
            GptModelInputs target_input;
            target_input.combo_tokens          = torch::tensor({10, 11, 12, 20, 21}, torch::kInt32).to(device);
            target_input.input_lengths         = torch::tensor({3, 2}, torch::kInt32).to(device);
            target_input.text_tokens_mask      = torch::tensor(masks, torch::kInt32).to(device);
            target_input.combo_tokens_type_ids = torch::tensor(types, torch::kInt32).to(device);
            target_input.mm_features_locs      = torch::tensor(locs, torch::kInt32).to(device);
            target_input.multimodal_features = std::vector<torch::Tensor>{
                torch::arange(rows * 4, torch::kFloat32).reshape({rows, 4}).to(device),
                (torch::arange(rows * 4, torch::kFloat32) + 100).reshape({rows, 4}).to(device)};
            target_input.mm_extra_input = std::vector<torch::Tensor>{
                torch::arange(rows * 8, torch::kFloat32).to(device),
                (torch::arange(rows * 8, torch::kFloat32) + 100).to(device)};
            auto draft_input = target_input;
            auto sampled     = torch::tensor({{30}, {40}}, torch::kInt32).to(device);
            TensorHolder holder;

            mtp::prepareDraftPrefillAfterTargetPrefill(draft_input, {}, sampled, {}, 1, holder);

            EXPECT_EQ(toVec<int32_t>(draft_input.combo_tokens), (std::vector<int32_t>{11, 12, 30, 21, 40}));
            EXPECT_EQ(toVec<int32_t>(draft_input.text_tokens_mask), (std::vector<int32_t>{0, 1, 1, 0, 1}));
            EXPECT_EQ(toVec<int32_t>(draft_input.combo_tokens_type_ids), (std::vector<int32_t>{1, 0, 0, 1, 0}));
            EXPECT_EQ(toVec<int32_t>(draft_input.mm_features_locs), (std::vector<int32_t>{0, 3}));
            ASSERT_EQ(draft_input.multimodal_features->size(), 2u);
            ASSERT_EQ(draft_input.mm_extra_input->size(), 2u);
            const auto expected_feature = starts_at_boundary ? torch::tensor({4.f, 5.f, 6.f, 7.f}) :
                                                              torch::tensor({0.f, 1.f, 2.f, 3.f});
            const auto expected_extra = starts_at_boundary ? torch::tensor({4.f, 5.f, 6.f, 7.f, 12.f, 13.f, 14.f, 15.f}) :
                                                            torch::tensor({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
            for (size_t i = 0; i < 2; ++i) {
                const auto offset = static_cast<float>(i * 100);
                EXPECT_TRUE(torch::equal(draft_input.multimodal_features->at(i).cpu(),
                                         (expected_feature + offset).reshape({1, 4})));
                EXPECT_TRUE(torch::equal(draft_input.mm_extra_input->at(i).cpu(), expected_extra + offset));
                EXPECT_TRUE(torch::equal(target_input.multimodal_features->at(i).cpu(),
                                         (torch::arange(rows * 4, torch::kFloat32) + offset).reshape({rows, 4})));
                EXPECT_TRUE(torch::equal(target_input.mm_extra_input->at(i).cpu(),
                                         torch::arange(rows * 8, torch::kFloat32) + offset));
            }
            EXPECT_EQ(toVec<int32_t>(target_input.text_tokens_mask), masks);
            EXPECT_EQ(toVec<int32_t>(target_input.combo_tokens_type_ids), types);
            EXPECT_EQ(toVec<int32_t>(target_input.mm_features_locs), locs);
        }
    }
}

TEST_F(MtpComputeTest, PrefillDropsFeaturesRemovedAtRequestStart) {
    for (bool device_input : {false, true}) {
        SCOPED_TRACE(device_input);
        const auto device = device_input ? torch::kCUDA : torch::kCPU;
        GptModelInputs target_input;
        target_input.combo_tokens          = torch::tensor({10, 20, 21}, torch::kInt32).to(device);
        target_input.input_lengths         = torch::tensor({1, 2}, torch::kInt32).to(device);
        target_input.text_tokens_mask      = torch::tensor({0, 0, 1}, torch::kInt32).to(device);
        target_input.combo_tokens_type_ids = torch::tensor({1, 1, 0}, torch::kInt32).to(device);
        target_input.mm_features_locs      = torch::tensor({0, 1}, torch::kInt32).to(device);
        target_input.multimodal_features = std::vector<torch::Tensor>{
            torch::ones({1, 4}, torch::kFloat32).to(device), torch::ones({1, 4}, torch::kFloat32).to(device)};
        target_input.mm_extra_input = std::vector<torch::Tensor>{
            torch::ones({8}, torch::kFloat32).to(device), torch::ones({8}, torch::kFloat32).to(device)};
        auto draft_input = target_input;
        auto sampled     = torch::tensor({{30}, {40}}, torch::kInt32).to(device);
        TensorHolder holder;

        mtp::prepareDraftPrefillAfterTargetPrefill(draft_input, {}, sampled, {}, 1, holder);

        EXPECT_EQ(toVec<int32_t>(draft_input.combo_tokens), (std::vector<int32_t>{30, 21, 40}));
        EXPECT_EQ(toVec<int32_t>(draft_input.text_tokens_mask), (std::vector<int32_t>{1, 1, 1}));
        EXPECT_EQ(toVec<int32_t>(draft_input.combo_tokens_type_ids), (std::vector<int32_t>{0, 0, 0}));
        EXPECT_EQ(draft_input.mm_features_locs.numel(), 0);
        EXPECT_TRUE(draft_input.multimodal_features->empty());
        EXPECT_TRUE(draft_input.mm_extra_input->empty());
        EXPECT_EQ(target_input.multimodal_features->size(), 2u);
        EXPECT_EQ(target_input.mm_extra_input->size(), 2u);
        EXPECT_EQ(toVec<int32_t>(target_input.mm_features_locs), (std::vector<int32_t>{0, 1}));
    }
}

TEST_F(MtpComputeTest, SharedDecodeInputCompactsDifferentAcceptLengths) {
    for (bool device_input : {false, true}) {
        const auto device = device_input ? torch::kCUDA : torch::kCPU;
        GptModelInputs verify_input;
        verify_input.is_target_verify = true;
        verify_input.combo_position_ids = torch::tensor({10, 11, 12, 20, 21, 22}, torch::kInt32).to(device);
        verify_input.prefix_lengths = torch::tensor({10, 20}, torch::kInt32).to(device);
        auto hidden_states = torch::arange(12, torch::kFloat32).reshape({6, 2}).cuda();
        auto accepted_lengths = torch::tensor({1, 3}, torch::kInt32).to(device);
        auto accepted_token_ids = torch::tensor({{4, 0, 0}, {5, 6, 7}}, torch::kInt32).to(device);
        auto draft_input = verify_input;
        torch::Tensor hidden_states_d_t;
        TensorHolder holder;

        mtp::prepareDraftPrefillAfterVerify(draft_input,
                                           hidden_states,
                                           accepted_token_ids,
                                           accepted_lengths,
                                           2,
                                           2,
                                           mtp::DraftInputLayout::COMPACT,
                                           1,
                                           hidden_states_d_t,
                                           holder);

        EXPECT_EQ(toVec<int32_t>(draft_input.combo_tokens), (std::vector<int32_t>{4, 5, 6, 7}));
        EXPECT_EQ(toVec<int32_t>(draft_input.input_lengths), (std::vector<int32_t>{1, 3}));
        EXPECT_EQ(toVec<int32_t>(draft_input.lm_output_indexes), (std::vector<int32_t>{0, 3}));
        EXPECT_EQ(toVec<int32_t>(draft_input.combo_position_ids), (std::vector<int32_t>{10, 20, 21, 22}));
        EXPECT_EQ(toVec<int32_t>(draft_input.prefix_lengths), (std::vector<int32_t>{10, 20}));
        EXPECT_TRUE(draft_input.combo_tokens.is_pinned());
        EXPECT_TRUE(draft_input.combo_position_ids.is_pinned());
        EXPECT_FALSE(draft_input.is_target_verify);
        EXPECT_TRUE(torch::equal(draft_input.last_hidden_states, hidden_states.index_select(
            0, torch::tensor({0, 3, 4, 5}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA)))));
        EXPECT_TRUE(verify_input.is_target_verify);
        EXPECT_EQ(toVec<int32_t>(verify_input.combo_position_ids), (std::vector<int32_t>{10, 11, 12, 20, 21, 22}));
        EXPECT_EQ(toVec<int32_t>(accepted_token_ids), (std::vector<int32_t>{4, 0, 0, 5, 6, 7}));
    }
}

TEST_F(MtpComputeTest, SharedDecodeInputRejectsAcceptLengthBatchMismatch) {
    for (int length_count : {1, 3}) {
        SCOPED_TRACE(length_count);
        GptModelInputs draft_input;
        torch::Tensor  hidden_states_d_t;
        TensorHolder   holder;
        auto hidden_states    = torch::zeros({6, 2});
        auto accepted_tokens  = torch::zeros({2, 3}, torch::kInt32);
        auto accepted_lengths = torch::ones({length_count}, torch::kInt32);

        try {
            mtp::prepareDraftPrefillAfterVerify(draft_input,
                                               hidden_states,
                                               accepted_tokens,
                                               accepted_lengths,
                                               2,
                                               2,
                                               mtp::DraftInputLayout::COMPACT,
                                               1,
                                               hidden_states_d_t,
                                               holder);
            FAIL() << "Expected accept_len batch mismatch";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("accept_len batch mismatch"), std::string::npos);
        }
    }
}

TEST_F(MtpComputeTest, SharedDecodeInputRejectsInvalidAcceptLengths) {
    for (const auto& lengths : std::vector<std::vector<int32_t>>{{4}, {4, 1}, {1, 4}, {0, 1}}) {
        SCOPED_TRACE(::testing::PrintToString(lengths));
        const auto batch_size = static_cast<int64_t>(lengths.size());
        GptModelInputs draft_input;
        torch::Tensor  hidden_states_d_t;
        TensorHolder   holder;
        auto hidden_states    = torch::zeros({batch_size * 3, 2});
        auto accepted_tokens  = torch::arange(batch_size * 3, torch::kInt32).reshape({batch_size, 3});
        auto accepted_lengths = torch::tensor(lengths, torch::kInt32);

        try {
            mtp::prepareDraftPrefillAfterVerify(draft_input,
                                               hidden_states,
                                               accepted_tokens,
                                               accepted_lengths,
                                               batch_size,
                                               2,
                                               mtp::DraftInputLayout::COMPACT,
                                               1,
                                               hidden_states_d_t,
                                               holder);
            FAIL() << "Expected invalid accept_len";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("invalid accept_len"), std::string::npos);
        }
    }
}

TEST_F(MtpComputeTest, SharedAcceptedHostViewsPreserveFailedRowPlaceholders) {
    speculative::SpeculativeSamplerOutput output;
    output.accept_tokens = torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt32).cuda();
    output.accept_len    = torch::tensor({3, 2}, torch::kInt32).cuda();
    output.accept_tokens_cpu = output.accept_tokens.to(torch::kCPU, true);
    output.accept_len_cpu    = output.accept_len.to(torch::kCPU, true);
    output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());

    auto device = mtp::getDeviceAcceptedTokens(output);
    EXPECT_EQ(device.token_ids.unsafeGetTensorImpl(), output.accept_tokens.unsafeGetTensorImpl());
    EXPECT_EQ(device.lengths.unsafeGetTensorImpl(), output.accept_len.unsafeGetTensorImpl());
    auto host = mtp::getHostAcceptedTokens(output);
    EXPECT_EQ(host.token_ids.unsafeGetTensorImpl(), output.accept_tokens_cpu.unsafeGetTensorImpl());
    EXPECT_EQ(host.lengths.unsafeGetTensorImpl(), output.accept_len_cpu.unsafeGetTensorImpl());
    EXPECT_EQ(toVec<int32_t>(host.lengths), (std::vector<int32_t>{3, 2}));

    /** Failed-row placeholders mutate the host mirror; successful rows retain their accepted prefix. */
    host.lengths[1] = 1;
    host.token_ids[1].zero_();
    auto modified_host = mtp::getHostAcceptedTokens(output, false);
    EXPECT_EQ(toVec<int32_t>(modified_host.lengths), (std::vector<int32_t>{3, 1}));
    EXPECT_EQ(toVec<int32_t>(modified_host.token_ids), (std::vector<int32_t>{10, 11, 12, 0, 0, 0}));
    EXPECT_EQ(toVec<int32_t>(device.lengths), (std::vector<int32_t>{3, 2}));
    EXPECT_EQ(toVec<int32_t>(device.token_ids), (std::vector<int32_t>{10, 11, 12, 20, 21, 22}));
}

TEST_F(MtpComputeTest, SharedAcceptedHostViewsCopyOnlyMissingMirrors) {
    for (bool has_token_mirror : {false, true}) {
        for (bool has_length_mirror : {false, true}) {
            SCOPED_TRACE(has_token_mirror);
            SCOPED_TRACE(has_length_mirror);
            speculative::SpeculativeSamplerOutput output;
            output.accept_tokens = torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt32).cuda();
            output.accept_len    = torch::tensor({3, 2}, torch::kInt32).cuda();
            if (has_token_mirror) {
                output.accept_tokens_cpu = torch::tensor({{10, 11, 12}, {0, 0, 0}}, torch::kInt32);
            }
            if (has_length_mirror) {
                output.accept_len_cpu = torch::tensor({3, 1}, torch::kInt32);
            }

            const auto host = mtp::getHostAcceptedTokens(output, false);
            EXPECT_TRUE(host.token_ids.device().is_cpu());
            EXPECT_TRUE(host.lengths.device().is_cpu());
            EXPECT_EQ(toVec<int32_t>(host.token_ids),
                      has_token_mirror ? (std::vector<int32_t>{10, 11, 12, 0, 0, 0}) :
                                         (std::vector<int32_t>{10, 11, 12, 20, 21, 22}));
            EXPECT_EQ(toVec<int32_t>(host.lengths),
                      has_length_mirror ? (std::vector<int32_t>{3, 1}) : (std::vector<int32_t>{3, 2}));
            if (has_token_mirror) {
                EXPECT_EQ(host.token_ids.unsafeGetTensorImpl(), output.accept_tokens_cpu.unsafeGetTensorImpl());
            }
            if (has_length_mirror) {
                EXPECT_EQ(host.lengths.unsafeGetTensorImpl(), output.accept_len_cpu.unsafeGetTensorImpl());
            }
        }
    }
}

TEST_F(MtpComputeTest, SharedDraftContinuationUsesValidLengthsForBothLayouts) {
    for (bool device_state : {false, true}) {
        for (bool device_metadata : {false, true}) {
            SCOPED_TRACE(device_state);
            SCOPED_TRACE(device_metadata);
            const auto device = device_metadata ? torch::kCUDA : torch::kCPU;
            const auto layout = mtp::selectDraftInputLayout(device_state);
            GptModelInputs verify;
            verify.combo_tokens      = torch::tensor({1, 2, 3, 4, 5, 6}, torch::kInt32).to(device);
            verify.input_lengths     = torch::tensor({3, 3}, torch::kInt32).to(device);
            verify.prefix_lengths    = torch::tensor({7, 11}, torch::kInt32).to(device);
            verify.sequence_lengths  = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
            verify.lm_output_indexes = torch::arange(6, torch::TensorOptions().dtype(torch::kInt32).device(device));
            verify.combo_position_ids =
                torch::tensor({7, 17, 27, 8, 18, 28, 9, 19, 29, 11, 21, 31, 12, 22, 32, 13, 23, 33}, torch::kInt32)
                    .to(device);
            verify.sequence_lengths_plus_1 = torch::tensor({8, 12}, torch::kInt32).to(device);
            verify.request_id = torch::tensor({101, 202}, torch::kInt64);
            verify.request_pd_separation = torch::ones({2}, torch::kBool);
            verify.cache_keys = torch::ones({2, 3}, torch::kInt64);
            auto target_hidden   = torch::arange(24, torch::kFloat32).reshape({6, 4}).cuda();
            auto accepted_tokens = torch::tensor({{31, 0, 0}, {41, 42, 0}}, torch::kInt32).cuda();
            auto accepted_lengths = torch::tensor({1, 2}, torch::kInt32).cuda();
            auto input = verify;
            torch::Tensor hidden_states_d_t;
            TensorHolder holder;
            mtp::prepareDraftPrefillAfterVerify(
                input, target_hidden, accepted_tokens, accepted_lengths, 2, 2, layout, 3, hidden_states_d_t, holder);
            ASSERT_EQ(hidden_states_d_t.unsafeGetTensorImpl(), input.last_hidden_states.unsafeGetTensorImpl());
            const auto prefill_hidden_before_decode = input.last_hidden_states.cpu();
            EXPECT_EQ(input.combo_tokens.numel(), device_state ? 6 : 3);
            EXPECT_EQ(toVec<int32_t>(input.lm_output_indexes),
                      device_state ? (std::vector<int32_t>{0, 4}) : (std::vector<int32_t>{0, 2}));
            auto draft_hidden = input.last_hidden_states.clone();
            auto expected_hidden = target_hidden.index_select(
                0, torch::tensor({0, 4}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA)));

            mtp::prepareDraftDecodeAfterPrefill(
                input, draft_hidden, torch::tensor({{50}, {60}}, torch::kInt32).cuda(), 3);
            draft_hidden.fill_(-1);
            EXPECT_TRUE(torch::equal(input.last_hidden_states, expected_hidden));
            EXPECT_TRUE(torch::equal(hidden_states_d_t.cpu(), prefill_hidden_before_decode));
            EXPECT_EQ(toVec<int32_t>(input.combo_tokens), (std::vector<int32_t>{50, 60}));
            EXPECT_EQ(toVec<int32_t>(input.input_lengths), (std::vector<int32_t>{1, 1}));
            EXPECT_EQ(toVec<int32_t>(input.sequence_lengths), (std::vector<int32_t>{8, 13}));
            EXPECT_EQ(toVec<int32_t>(input.lm_output_indexes), (std::vector<int32_t>{0, 1}));
            EXPECT_EQ(toVec<int32_t>(input.combo_position_ids), (std::vector<int32_t>{8, 18, 28, 13, 23, 33}));
            EXPECT_TRUE(input.combo_position_ids.is_pinned());
            EXPECT_EQ(input.prefix_lengths.numel(), 0);
            EXPECT_FALSE(input.sequence_lengths_plus_1.defined());
            EXPECT_FALSE(input.request_id.defined());
            EXPECT_FALSE(input.request_pd_separation.defined());
            EXPECT_FALSE(input.cache_keys.defined());
            EXPECT_EQ(toVec<int32_t>(verify.combo_tokens), (std::vector<int32_t>{1, 2, 3, 4, 5, 6}));
            EXPECT_EQ(toVec<int32_t>(verify.input_lengths), (std::vector<int32_t>{3, 3}));
            EXPECT_EQ(toVec<int32_t>(verify.prefix_lengths), (std::vector<int32_t>{7, 11}));
            EXPECT_TRUE(verify.request_id.defined());
            EXPECT_TRUE(verify.sequence_lengths_plus_1.defined());
        }
    }
}

TEST_F(MtpComputeTest, SharedAdvanceReusesHiddenStates) {
    for (bool device_input : {false, true}) {
        const auto device = device_input ? torch::kCUDA : torch::kCPU;
        GptModelInputs input;
        input.combo_tokens = torch::tensor({1, 2}, torch::kInt32).to(device);
        input.sequence_lengths = torch::tensor({8, 10}, torch::kInt32).to(device);
        input.combo_position_ids = torch::tensor({7, 9}, torch::kInt32).to(device);
        auto original_lengths = input.sequence_lengths;
        auto original_positions = input.combo_position_ids;
        auto hidden_buffer = torch::ones({2, 2}, torch::TensorOptions().device(torch::kCUDA));
        auto draft_tokens = torch::tensor({3, 4}, torch::kInt32).reshape({2, 1}).cuda();
        TensorHolder holder;

        mtp::advanceDraftInput(input, hidden_buffer, draft_tokens, 1, holder);

        EXPECT_EQ(toVec<int32_t>(input.combo_tokens), (std::vector<int32_t>{3, 4}));
        EXPECT_EQ(toVec<int32_t>(input.sequence_lengths), (std::vector<int32_t>{9, 11}));
        EXPECT_EQ(toVec<int32_t>(input.combo_position_ids), (std::vector<int32_t>{8, 10}));
        EXPECT_TRUE(input.sequence_lengths.is_cuda());
        EXPECT_TRUE(input.combo_position_ids.is_pinned());
        EXPECT_EQ(input.last_hidden_states.data_ptr(), hidden_buffer.data_ptr());
        EXPECT_TRUE(torch::equal(input.last_hidden_states, torch::ones_like(hidden_buffer)));
        EXPECT_EQ(toVec<int32_t>(original_lengths), (std::vector<int32_t>{8, 10}));
        EXPECT_EQ(toVec<int32_t>(original_positions), (std::vector<int32_t>{7, 9}));
    }
}

TEST_F(MtpComputeTest, SharedRejectionUsesExplicitSamplingParams) {
    speculative::SpeculativeSampler sampler(torch::Tensor(), 2);
    speculative::SpeculativeSamplingParams params;
    params.do_sample = torch::tensor({false, false}, torch::kBool);
    params.force_accept = torch::tensor({false, false}, torch::kBool);
    params.generators.resize(2);
    SamplerOutput draft;
    draft.token_ids = torch::tensor({{1, 2}, {1, 2}}, torch::kInt32).cuda();
    draft.all_probs = torch::zeros({2, 2, 4}, torch::TensorOptions().device(torch::kCUDA));
    draft.all_probs.scatter_(-1, draft.token_ids.to(torch::kLong).unsqueeze(-1), 1.0);
    SamplerOutput target;
    target.token_ids = torch::tensor({{1}, {2}, {3}, {3}, {2}, {0}}, torch::kInt32).cuda();
    target.all_probs = torch::zeros({2, 3, 4}, torch::TensorOptions().device(torch::kCUDA));
    target.all_probs.scatter_(-1, target.token_ids.to(torch::kLong).reshape({2, 3, 1}), 1.0);

    SpecLogitsVerifyRunner::LaunchResult verify_result;
    verify_result.processor_errors.resize(2);
    verify_result.processor_errors[1] = ErrorInfo(ErrorCode::INVALID_PARAMS, "speculative verification failed");
    speculative::SpeculativeSamplerOutput accepted;
    mtp::runRejectionSampling(sampler, params, draft, target, verify_result, accepted);
    ASSERT_TRUE(accepted.accept_tokens_cpu.defined());
    ASSERT_TRUE(accepted.accept_len_cpu.defined());
    accepted.transfer_done_event->synchronize();
    EXPECT_EQ(toVec<int32_t>(accepted.accept_len_cpu), (std::vector<int32_t>{3, 1}));
    EXPECT_EQ(toVec<int32_t>(accepted.accept_tokens_cpu[0]), (std::vector<int32_t>{1, 2, 3}));
    EXPECT_EQ(toVec<int32_t>(accepted.accept_len), (std::vector<int32_t>{3, 1}));
    EXPECT_EQ(toVec<int32_t>(accepted.accept_tokens[0]), (std::vector<int32_t>{1, 2, 3}));
    EXPECT_EQ(accepted.accept_tokens[1][0].item<int32_t>(), 3);
    ASSERT_EQ(accepted.processor_errors.size(), 2);
    EXPECT_FALSE(accepted.processor_errors[0].has_value());
    ASSERT_TRUE(accepted.processor_errors[1].has_value());
    EXPECT_EQ(accepted.processor_errors[1]->code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(accepted.processor_errors[1]->ToString(), "speculative verification failed");

    params.force_accept[1] = true;
    mtp::runRejectionSampling(sampler, params, draft, target, {}, accepted);
    accepted.transfer_done_event->synchronize();
    EXPECT_EQ(toVec<int32_t>(accepted.accept_len_cpu), (std::vector<int32_t>{3, 3}));
    EXPECT_EQ(toVec<int32_t>(accepted.accept_tokens_cpu[1]), (std::vector<int32_t>{1, 2, 0}));
    EXPECT_EQ(toVec<int32_t>(accepted.accept_len), (std::vector<int32_t>{3, 3}));
    EXPECT_EQ(toVec<int32_t>(accepted.accept_tokens[1]), (std::vector<int32_t>{1, 2, 0}));
    EXPECT_TRUE(accepted.processor_errors.empty());
}

TEST_F(MtpComputeTest, SharedRejectionAppliesCapBeforeCpuTransfer) {
    speculative::SpeculativeSampler sampler(torch::Tensor(), 2);
    speculative::SpeculativeSamplingParams params;
    params.do_sample = torch::tensor({false, false, false}, torch::kBool);
    params.force_accept = torch::tensor({true, false, false}, torch::kBool);
    params.generators.resize(3);
    SamplerOutput draft;
    draft.token_ids = torch::tensor({{1, 2}, {4, 0}, {5, 6}}, torch::kInt32).cuda();
    draft.token_ids_are_point_mass = true;
    SamplerOutput target;
    target.token_ids =
        torch::tensor({{-1, 8}, {-1, 9}, {-1, 10}, {-1, 11}, {-1, 12}, {-1, 13}, {-1, 5}, {-1, 6}, {-1, 7}},
                      torch::kInt32).cuda();
    target.all_probs = torch::zeros({3, 3, 17}, torch::TensorOptions().device(torch::kCUDA));
    target.all_probs.scatter_(-1, target.token_ids.select(1, 1).to(torch::kLong).reshape({3, 3, 1}), 1.0);

    for (bool device_cap : {false, true}) {
        SpecLogitsVerifyRunner::LaunchResult verify_result;
        verify_result.spec_cap_cpu = torch::tensor({1, 1, 2}, torch::kInt32);
        if (device_cap) {
            verify_result.spec_cap_gpu = verify_result.spec_cap_cpu.cuda();
            verify_result.spec_cap_cpu = torch::zeros_like(verify_result.spec_cap_cpu);
        }
        verify_result.processor_errors.resize(3);
        verify_result.processor_errors[1] = ErrorInfo(ErrorCode::INVALID_PARAMS, "speculative cap failed");
        speculative::SpeculativeSamplerOutput accepted;
        mtp::runRejectionSampling(sampler, params, draft, target, verify_result, accepted);

        ASSERT_TRUE(accepted.accept_tokens_cpu.defined());
        ASSERT_TRUE(accepted.accept_len_cpu.defined());
        EXPECT_TRUE(accepted.accept_tokens_cpu.device().is_cpu());
        EXPECT_TRUE(accepted.accept_len_cpu.device().is_cpu());
        EXPECT_TRUE(accepted.accept_tokens_cpu.is_contiguous());
        EXPECT_TRUE(accepted.accept_len_cpu.is_contiguous());
        accepted.transfer_done_event->synchronize();
        EXPECT_EQ(toVec<int32_t>(accepted.accept_len_cpu), (std::vector<int32_t>{2, 1, 3}));
        EXPECT_EQ(toVec<int32_t>(accepted.accept_tokens_cpu[0]), (std::vector<int32_t>{1, 9, 10}));
        EXPECT_EQ(accepted.accept_tokens_cpu[1][0].item<int32_t>(), 11);
        EXPECT_EQ(toVec<int32_t>(accepted.accept_tokens_cpu[2]), (std::vector<int32_t>{5, 6, 7}));
        EXPECT_TRUE(accepted.accept_tokens.is_cuda());
        EXPECT_TRUE(accepted.accept_len.is_cuda());
        EXPECT_EQ(toVec<int32_t>(accepted.accept_len), (std::vector<int32_t>{2, 1, 3}));
        EXPECT_EQ(toVec<int32_t>(accepted.accept_tokens[0]), (std::vector<int32_t>{1, 9, 10}));
        EXPECT_EQ(accepted.accept_tokens[1][0].item<int32_t>(), 11);
        EXPECT_EQ(toVec<int32_t>(accepted.accept_tokens[2]), (std::vector<int32_t>{5, 6, 7}));
        ASSERT_EQ(accepted.processor_errors.size(), 3);
        EXPECT_FALSE(accepted.processor_errors[0].has_value());
        ASSERT_TRUE(accepted.processor_errors[1].has_value());
        EXPECT_EQ(accepted.processor_errors[1]->code(), ErrorCode::INVALID_PARAMS);
        EXPECT_EQ(accepted.processor_errors[1]->ToString(), "speculative cap failed");
        EXPECT_FALSE(accepted.processor_errors[2].has_value());
    }
}

}  // namespace rtp_llm
