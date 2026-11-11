#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <unistd.h>
#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/TestLayoutSpec.h"
#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "alog/Configurator.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/normal_engine/HiddenStateCapturePolicy.h"

#define private public
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#include <ATen/cuda/CUDAContext.h>
#endif

namespace rtp_llm {

void registerExecCtxOps(pybind11::module& m);

using namespace std;
namespace spec = speculative;

TEST(MtpExecutorPolicyTest, DSparkPrefillCPRequiresPrefillRole) {
    PrefillCPConfig prefill_cp_config;
    prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_TRUE(MtpExecutor::dsparkPrefillCPRoleIsValid(prefill_cp_config, RoleType::PREFILL));

    PrefillCPConfig decode_cp_config;
    decode_cp_config.method = CPRotateMethod::PREFILL_CP;
    EXPECT_FALSE(decode_cp_config.is_enabled());
    EXPECT_TRUE(MtpExecutor::dsparkPrefillCPRoleIsValid(decode_cp_config, RoleType::DECODE));

    // A split-enabled decode/colocated engine would conflate the two roles:
    // its proposal block would be fed through the prefill CP splitter.
    decode_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_FALSE(MtpExecutor::dsparkPrefillCPRoleIsValid(decode_cp_config, RoleType::DECODE));
    EXPECT_FALSE(MtpExecutor::dsparkPrefillCPRoleIsValid(decode_cp_config, RoleType::PDFUSION));
}

TEST(MtpExecutorPolicyTest, DSparkPrefillRoleDisablesDraftGraphCapture) {
    // A DSpARK PREFILL worker only seeds the draft feature KV and never runs
    // the fixed-width decode proposal/commit; capturing those graphs there
    // wastes graph-pool memory and can OOM CP-RR startup.
    EXPECT_FALSE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/true, RoleType::PREFILL));
    EXPECT_TRUE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/true, RoleType::DECODE));
    EXPECT_TRUE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/true, RoleType::PDFUSION));
    EXPECT_TRUE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/false, RoleType::PREFILL));
    EXPECT_TRUE(MtpExecutor::dsparkDraftGraphAllowed(/*is_dspark=*/false, RoleType::DECODE));
}

TEST(MtpExecutorPolicyTest, MtpInputHiddenSizeDerivesCheckedHyperConnectionWidth) {
    ModelConfig config;
    config.hidden_size = 8;
    config.hc_mult     = 4;
    EXPECT_EQ(config.getMtpInputHiddenSize(), 32);

    config.hidden_size = 0;
    EXPECT_THROW(config.getMtpInputHiddenSize(), std::runtime_error);
    config.hidden_size = 8;
    config.hc_mult     = 0;
    EXPECT_THROW(config.getMtpInputHiddenSize(), std::runtime_error);
    config.hc_mult = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(config.getMtpInputHiddenSize(), std::runtime_error);
}

TEST(MtpExecutorPolicyTest, CpRestoreSnapshotOwnsMutableHostInput) {
    auto         input_lengths = torch::tensor({3683}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    TensorHolder holder;
    auto         snapshot = MtpExecutor::snapshotMutableHostInputToCuda(input_lengths, holder);

    ASSERT_TRUE(snapshot.is_cuda());
    ASSERT_EQ(holder.tensors.size(), 1);
    ASSERT_TRUE(holder.tensors.front().is_pinned());
    ASSERT_NE(holder.tensors.front().data_ptr(), input_lengths.data_ptr());
    input_lengths.fill_(922);
    EXPECT_EQ(holder.tensors.front().item<int32_t>(), 3683);
    EXPECT_EQ(snapshot.cpu().item<int32_t>(), 3683);

    auto device_input    = input_lengths.to(torch::kCUDA);
    auto device_snapshot = MtpExecutor::snapshotMutableHostInputToCuda(device_input, holder);
    EXPECT_EQ(device_snapshot.data_ptr(), device_input.data_ptr());
    EXPECT_EQ(holder.tensors.size(), 1);
}

TEST(MtpExecutorPolicyTest, HybridCacheWaitsBeforeReadingNextRoundHostState) {
    EXPECT_FALSE(MtpExecutor::mustWaitBeforeStreamPreparation(
        /*stream_async=*/false, /*drop_broad_sync=*/false, /*linear_attention=*/true));
    EXPECT_TRUE(MtpExecutor::mustWaitBeforeStreamPreparation(
        /*stream_async=*/true, /*drop_broad_sync=*/false, /*linear_attention=*/false));
    EXPECT_FALSE(MtpExecutor::mustWaitBeforeStreamPreparation(
        /*stream_async=*/true, /*drop_broad_sync=*/true, /*linear_attention=*/false));
    EXPECT_TRUE(MtpExecutor::mustWaitBeforeStreamPreparation(
        /*stream_async=*/true, /*drop_broad_sync=*/true, /*linear_attention=*/true));
    EXPECT_FALSE(MtpExecutor::mustWaitBeforeStreamPreparation(/*stream_async=*/true,
                                                              /*drop_broad_sync=*/true,
                                                              /*linear_attention=*/true,
                                                              /*cache_snapshot_ready=*/true));
}

TEST(MtpExecutorPolicyTest, MtpSequenceUpperBoundChainsUntilBookkeepingCatchesUp) {
    constexpr int width      = 8;
    const int     first_next = MtpExecutor::selectMtpPreviousSeqLenUpperBound(
                               /*pending=*/false, /*previous_next_bound=*/-1, /*host_seq_len=*/96)
                           + width;
    const int second_next = MtpExecutor::selectMtpPreviousSeqLenUpperBound(
                                /*pending=*/true, /*previous_next_bound=*/first_next, /*host_seq_len=*/96)
                            + width;
    const int third_next = MtpExecutor::selectMtpPreviousSeqLenUpperBound(
                               /*pending=*/true, /*previous_next_bound=*/second_next, /*host_seq_len=*/96)
                           + width;
    EXPECT_EQ(104, first_next);
    EXPECT_EQ(112, second_next);
    EXPECT_EQ(120, third_next);
    EXPECT_EQ(96,
              MtpExecutor::selectMtpPreviousSeqLenUpperBound(
                  /*pending=*/false, /*previous_next_bound=*/third_next, /*host_seq_len=*/96));
    EXPECT_EQ(96,
              MtpExecutor::selectMtpPreviousSeqLenUpperBound(
                  /*pending=*/true, /*previous_next_bound=*/-1, /*host_seq_len=*/96));
}

TEST(MtpExecutorPolicyTest, DSparkPositionStateAdvancesByAcceptedLength) {
    auto verify_positions = torch::tensor(
        {10, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33, 100, 200, 300, 101, 201, 301, 102, 202, 302, 103, 203, 303},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto accept_len = torch::tensor({2, 4}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto next_positions =
        MtpExecutor::advanceDSparkPositionIds(verify_positions, accept_len, /*batch_size=*/2, /*verify_width=*/4);

    EXPECT_TRUE(
        torch::equal(next_positions.reshape({-1}).cpu(), torch::tensor({12, 22, 32, 104, 204, 304}, torch::kInt32)));
    EXPECT_TRUE(next_positions.is_cuda());
}

struct MtpExecutorTestConfig {
    size_t  max_seq_len            = 2048;
    size_t  vocab_size             = 4;
    size_t  num_layers             = 1;
    size_t  gen_num_per_cycle      = 4;
    size_t  vocab_size_override    = 0;  // 0 means use vocab_size
    int64_t mm_position_ids_style  = 0;
    int     position_id_len_factor = 1;
    int     tp_size                = 1;

    SpeculativeType sp_type                   = SP_TYPE_MTP;
    int64_t         dspark_mask_token_id      = -1;
    RoleType        role_type                 = RoleType::PDFUSION;
    bool            enable_batch_record       = false;
    double          record_decode_sample_rate = 1.0;
};

template<typename T>
struct TestDataHolder {
    queue<T> test_data;

    T get() {
        if (test_data.empty()) {
            throw std::runtime_error("[test] Test data is empty");
        }

        T res = test_data.front();
        test_data.pop();
        return res;
    }

    void push(const T& res) {
        test_data.push(res);
    }

    void push(const vector<T>& res) {
        for (const auto& r : res) {
            test_data.push(r);
        }
    }
};

template<typename T>
vector<T> createRandomVector(size_t size, int max_val) {
    std::random_device                rd;
    std::mt19937                      gen(rd());
    std::uniform_real_distribution<T> dis(0.0, max_val);
    vector<T>                         vec(size);
    for (size_t i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

// for int type, use uniform_int_distribution
vector<int> createRandomVector(size_t size, int max_val) {
    std::random_device                 rd;
    std::mt19937                       gen(rd());
    std::uniform_int_distribution<int> dis(0, max_val);
    vector<int>                        vec(size);
    for (size_t i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

void checkTensorEqual(const torch::Tensor& t1, const torch::Tensor& t2) {
    bool t1_empty = !t1.defined() || t1.numel() == 0;
    bool t2_empty = !t2.defined() || t2.numel() == 0;
    if (t1_empty && t2_empty)
        return;
    if (t1_empty || t2_empty) {
        string t1_info = t1_empty ? "t1 is empty" : "t1 size: " + to_string(t1.numel());
        string t2_info = t2_empty ? "t2 is empty" : "t2 size: " + to_string(t2.numel());
        throw std::runtime_error("[test] Tensor mismatch: " + t1_info + " " + t2_info);
    }
    auto a = t1.cpu().contiguous();
    auto b = t2.cpu().contiguous();
    EXPECT_TRUE(torch::equal(a, b)) << "Tensors are not equal:\n" << a << "\nvs\n" << b;
}

template<typename T>
vector<T> toVec(const torch::Tensor& t) {
    auto c = t.cpu().contiguous();
    return vector<T>(c.data_ptr<T>(), c.data_ptr<T>() + c.numel());
}

template<typename T>
vector<T> catVectors(const vector<vector<T>>& vectors) {
    vector<T> result;
    for (const auto& vec : vectors) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    return result;
}

class FakeModel: public ModelBase {
public:
    FakeModel() = default;
    std::function<void(GptModelInputs&)> forward_hook;

    FakeModel(const GptModelInitParams& params) {
        weights_  = params.weights;
        model_id_ = params.model_id;
    }

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        if (!skip_input_check_) {
            checkInputs(inputs);
        }
        if (!expected_target_verify_.empty()) {
            EXPECT_EQ(inputs.is_target_verify, expected_target_verify_.front());
            expected_target_verify_.pop();
        }
        if (forward_trace_ != nullptr) {
            forward_trace_->push_back(forward_trace_label_);
        }
        if (post_forward_combo_tokens_.defined()) {
            auto& mutable_inputs        = const_cast<GptModelInputs&>(inputs);
            mutable_inputs.combo_tokens = post_forward_combo_tokens_;
            mutable_inputs.input_lengths.copy_(post_forward_input_lengths_);
        }
        if (forward_hook) {
            forward_hook(const_cast<GptModelInputs&>(inputs));
        }
        ++forward_count_;
        recordEvent("forward");
        if (forward_observer_) {
            forward_observer_();
        }
        return output_holder.get();
    }

    size_t forwardCount() const {
        return forward_count_;
    }

    std::string waitCacheStorePublication() override {
        recordEvent("wait_publication");
        if (publication_observer_) {
            publication_observer_();
        }
        if (publication_exception_) {
            std::rethrow_exception(publication_exception_);
        }
        return publication_error_;
    }

    std::function<std::optional<bool>(bool)> host_status_reducer;
    std::optional<bool>                      reduceSpeculativeTargetPrepareStatus(bool local_ok) override {
        return host_status_reducer ? host_status_reducer(local_ok) : std::nullopt;
    }

    bool hasSpeculativeTargetCommitHooks() const override {
        return speculative_target_commit_enabled_;
    }

    std::string prepareSpeculativeTargetCommit(const torch::Tensor& accept_len) override {
        if (!speculative_target_commit_enabled_) {
            return {};
        }
        recordEvent("prepare_target_commit");
        prepared_accept_len_ = accept_len.cpu().contiguous();
        if (speculative_target_prepare_throws_) {
            throw std::runtime_error("injected prepare exception");
        }
        return speculative_target_prepare_error_;
    }

    std::string finishSpeculativeTargetCommit(bool commit) override {
        if (!speculative_target_commit_enabled_) {
            return {};
        }
        recordEvent(commit ? "commit_target" : "abort_target");
        if (commit) {
            ++speculative_target_commit_count_;
            speculative_target_state_committed_ = true;
            speculative_target_undo_retained_   = true;
            if (speculative_target_commit_throws_) {
                throw std::runtime_error("injected commit exception");
            }
            return {};
        } else {
            ++speculative_target_abort_count_;
            speculative_target_state_committed_ = false;
            speculative_target_undo_retained_   = false;
            return {};
        }
    }

    std::string finalizeSpeculativeTargetCommit() override {
        if (!speculative_target_commit_enabled_) {
            return {};
        }
        recordEvent("finalize_target");
        ++speculative_target_finalize_count_;
        speculative_target_undo_retained_ = false;
        return {};
    }

    void enableSpeculativeTargetCommit(std::string prepare_error = {}) {
        speculative_target_commit_enabled_ = true;
        speculative_target_prepare_error_  = std::move(prepare_error);
    }

    void setSpeculativeTargetPrepareThrows(bool enabled) {
        speculative_target_prepare_throws_ = enabled;
    }

    void setSpeculativeTargetCommitThrows(bool enabled) {
        speculative_target_commit_throws_ = enabled;
    }

    bool speculativeTargetStateCommitted() const {
        return speculative_target_state_committed_;
    }

    bool speculativeTargetUndoRetained() const {
        return speculative_target_undo_retained_;
    }

    const torch::Tensor& preparedAcceptLen() const {
        return prepared_accept_len_;
    }

    size_t speculativeTargetCommitCount() const {
        return speculative_target_commit_count_;
    }

    size_t speculativeTargetAbortCount() const {
        return speculative_target_abort_count_;
    }

    size_t speculativeTargetFinalizeCount() const {
        return speculative_target_finalize_count_;
    }

    void setPublicationResult(std::string error) {
        publication_error_ = std::move(error);
    }

    void setPublicationException(std::exception_ptr exception) {
        publication_exception_ = std::move(exception);
    }

    void setPublicationObserver(std::function<void()> observer) {
        publication_observer_ = std::move(observer);
    }

    void setEventLog(std::shared_ptr<std::vector<std::string>> events, std::string name) {
        event_log_  = std::move(events);
        event_name_ = std::move(name);
    }

    std::optional<std::string> takeDeferredHiddenStateCaptureError() override {
        if (deferred_hidden_state_capture_error_.has_value() && forward_trace_ != nullptr) {
            forward_trace_->push_back(forward_trace_label_ + ".capture_error_taken");
        }
        return std::exchange(deferred_hidden_state_capture_error_, std::nullopt);
    }

    void setDeferredHiddenStateCaptureError(std::string error_message) {
        deferred_hidden_state_capture_error_ = std::move(error_message);
    }

    void prepareAttentionInputs(const GptModelInputs& inputs) override {
        if (input_observer_) {
            input_observer_(inputs);
        }
        ++prepare_attention_count_;
        recordEvent("prepare_attention");
        if (prepare_input_holder.test_data.empty()) {
            return;
        }
        GptModelInputs expected_inputs = prepare_input_holder.get();
        checkTensorField("prepared input_lengths", inputs.input_lengths, expected_inputs.input_lengths);
        checkTensorField("prepared sequence_lengths", inputs.sequence_lengths, expected_inputs.sequence_lengths);
        checkTensorField("prepared prefix_lengths", inputs.prefix_lengths, expected_inputs.prefix_lengths);
        checkTensorField("prepared sequence_lengths_plus_1",
                         inputs.sequence_lengths_plus_1,
                         expected_inputs.sequence_lengths_plus_1);
        checkTensorField("prepared lm_output_indexes", inputs.lm_output_indexes, expected_inputs.lm_output_indexes);
    }

    void checkTensorField(const char* name, const torch::Tensor& actual, const torch::Tensor& expected) {
        RTP_LLM_LOG_INFO("check %s", name);
        checkTensorEqual(actual, expected);
    }

    void checkInputs(const GptModelInputs& inputs) {
        if (input_observer_) {
            input_observer_(inputs);
        }
        GptModelInputs expected_inputs = input_holder.get();
        if (expected_is_target_verify_.has_value()) {
            EXPECT_EQ(inputs.is_target_verify, expected_is_target_verify_.value());
        }
        checkTensorField("combo_tokens", inputs.combo_tokens, expected_inputs.combo_tokens);
        checkTensorField("input_lengths", inputs.input_lengths, expected_inputs.input_lengths);
        checkTensorField("sequence_lengths", inputs.sequence_lengths, expected_inputs.sequence_lengths);
        checkTensorField("prefix_lengths", inputs.prefix_lengths, expected_inputs.prefix_lengths);
        checkTensorField("lm_output_indexes", inputs.lm_output_indexes, expected_inputs.lm_output_indexes);
        checkTensorField("last_hidden_states", inputs.last_hidden_states, expected_inputs.last_hidden_states);
        checkTensorField("combo_position_ids", inputs.combo_position_ids, expected_inputs.combo_position_ids);
        EXPECT_EQ(inputs.skip_lm_head, expected_inputs.skip_lm_head);
        EXPECT_EQ(inputs.capture_hidden_states, expected_inputs.capture_hidden_states);
    }

    void setOutputs(const vector<GptModelOutputs>& outputs) {
        output_holder.push(outputs);
    }

    void setInputs(const vector<GptModelInputs>& inputs) {
        input_holder.push(inputs);
    }

    void setInputObserver(std::function<void(const GptModelInputs&)> observer) {
        input_observer_ = std::move(observer);
    }

    void setPrepareInputs(const vector<GptModelInputs>& inputs) {
        prepare_input_holder.push(inputs);
    }

    void expectTargetVerify(bool expected) {
        expected_is_target_verify_ = expected;
    }

    bool hasPendingPrepareInputs() const {
        return !prepare_input_holder.test_data.empty();
    }

    size_t prepareAttentionCount() const {
        return prepare_attention_count_;
    }

    // Test stand-in for the shared MTP hidden buffer view (CP-local or DSpARK aux rows).
    void setMtpTargetHiddenStates(torch::Tensor rows) {
        mtp_target_hidden_rows_ = std::move(rows);
    }

    torch::Tensor getMtpTargetHiddenStates(int64_t num_tokens) override {
        mtp_hidden_requests.push_back(num_tokens);
        if (!mtp_target_hidden_rows_.defined()) {
            return torch::Tensor();
        }
        if (num_tokens < 0) {
            return mtp_target_hidden_rows_;
        }
        RTP_LLM_CHECK_WITH_INFO(num_tokens <= mtp_target_hidden_rows_.size(0),
                                "test MTP target hidden request exceeds local rows");
        return mtp_target_hidden_rows_.slice(0, 0, num_tokens);
    }

    bool hasMtpTargetHiddenBuffer() const override {
        return mtp_target_hidden_rows_.defined();
    }

    void setMtpLastHiddenStates(torch::Tensor rows) {
        mtp_last_hidden_rows_ = std::move(rows);
    }

    torch::Tensor getMtpLastHiddenStates(int64_t num_tokens) override {
        if (!mtp_last_hidden_rows_.defined()) {
            return torch::Tensor();
        }
        if (num_tokens < 0) {
            return mtp_last_hidden_rows_;
        }
        RTP_LLM_CHECK_WITH_INFO(num_tokens <= mtp_last_hidden_rows_.size(0),
                                "test MTP last-hidden request exceeds per-request rows");
        return mtp_last_hidden_rows_.slice(0, 0, num_tokens);
    }

    void setPostForwardInputMutation(torch::Tensor combo_tokens, torch::Tensor input_lengths) {
        post_forward_combo_tokens_  = std::move(combo_tokens);
        post_forward_input_lengths_ = std::move(input_lengths);
    }

    void setSkipInputCheck(bool skip_input_check) {
        skip_input_check_ = skip_input_check;
    }

    void setExpectedTargetVerify(const vector<bool>& expected) {
        for (bool value : expected) {
            expected_target_verify_.push(value);
        }
    }

    bool hasPendingTargetVerify() const {
        return !expected_target_verify_.empty();
    }

    void setForwardTrace(vector<string>* trace, string label) {
        forward_trace_       = trace;
        forward_trace_label_ = std::move(label);
    }

    void setForwardObserver(std::function<void()> observer) {
        forward_observer_ = std::move(observer);
    }

    std::vector<int64_t> mtp_hidden_requests;

private:
    void recordEvent(const char* event) {
        if (event_log_) {
            event_log_->push_back(event_name_ + "." + event);
        }
    }

    TestDataHolder<GptModelInputs>             input_holder;
    TestDataHolder<GptModelInputs>             prepare_input_holder;
    TestDataHolder<GptModelOutputs>            output_holder;
    torch::Tensor                              mtp_target_hidden_rows_;
    torch::Tensor                              mtp_last_hidden_rows_;
    torch::Tensor                              post_forward_combo_tokens_;
    torch::Tensor                              post_forward_input_lengths_;
    queue<bool>                                expected_target_verify_;
    vector<string>*                            forward_trace_ = nullptr;
    string                                     forward_trace_label_;
    bool                                       skip_input_check_ = false;
    size_t                                     forward_count_    = 0;
    std::optional<bool>                        expected_is_target_verify_;
    std::optional<std::string>                 deferred_hidden_state_capture_error_;
    std::string                                publication_error_;
    std::exception_ptr                         publication_exception_;
    std::function<void()>                      publication_observer_;
    std::function<void()>                      forward_observer_;
    std::function<void(const GptModelInputs&)> input_observer_;
    std::shared_ptr<std::vector<std::string>>  event_log_;
    std::string                                event_name_;
    size_t                                     prepare_attention_count_            = 0;
    bool                                       speculative_target_commit_enabled_  = false;
    bool                                       speculative_target_prepare_throws_  = false;
    bool                                       speculative_target_commit_throws_   = false;
    bool                                       speculative_target_state_committed_ = false;
    bool                                       speculative_target_undo_retained_   = false;
    std::string                                speculative_target_prepare_error_;
    torch::Tensor                              prepared_accept_len_;
    size_t                                     speculative_target_commit_count_   = 0;
    size_t                                     speculative_target_abort_count_    = 0;
    size_t                                     speculative_target_finalize_count_ = 0;
};

class FakeFastTopKSampler: public spec::FastTopKSampler {
public:
    FakeFastTopKSampler(): spec::FastTopKSampler(torch::Tensor()) {}

    spec::FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1) override {
        ++forward_count_;
        checkInputs(logits);
        return output_holder.get();
    }

    size_t forwardCount() const {
        return forward_count_;
    }

    void checkInputs(const torch::Tensor& logits) {
        auto expected_logits = logits_holder.get();
        RTP_LLM_LOG_INFO("check fast_topk_sampler logits");
        checkTensorEqual(logits, expected_logits);
    }

    void setOutputs(const vector<spec::FastTopKSamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

    void setInputs(const vector<torch::Tensor>& inputs) {
        logits_holder.push(inputs);
    }

private:
    TestDataHolder<torch::Tensor>               logits_holder;
    TestDataHolder<spec::FastTopKSamplerOutput> output_holder;
    size_t                                      forward_count_ = 0;
};

class FakeSpeculativeSampler: public spec::SpeculativeSampler {
public:
    FakeSpeculativeSampler(size_t propose_step): spec::SpeculativeSampler(torch::Tensor(), propose_step) {}

    spec::SpeculativeSamplerOutput forward(const spec::SpeculativeSamplingParams& params,
                                           SamplerOutput&                         draft_sampler_output,
                                           SamplerOutput&                         target_sampler_output) override {
        ++forward_count_;
        last_params = params;
        return output_holder.get();
    }

    spec::SpeculativeSamplingParams last_params;

    size_t forwardCount() const {
        return forward_count_;
    }

    void checkInputs(const std::list<GenerateStreamPtr>& streams,
                     SamplerOutput&                      draft_sampler_output,
                     SamplerOutput&                      target_sampler_output) {
        auto [expected_draft_sampler_input, expected_target_sampler_input] = input_holder.get();
        RTP_LLM_LOG_INFO("check draft_sampler_output.token_ids");
        checkTensorEqual(draft_sampler_output.token_ids, expected_draft_sampler_input.token_ids);
        RTP_LLM_LOG_INFO("check draft_sampler_output.all_probs");
        checkTensorEqual(draft_sampler_output.all_probs, expected_draft_sampler_input.all_probs);
        RTP_LLM_LOG_INFO("check target_sampler_output.all_probs");
        checkTensorEqual(target_sampler_output.all_probs, expected_target_sampler_input.all_probs);
    }

    void setOutputs(const vector<spec::SpeculativeSamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

    void setInputs(const pair<SamplerOutput, SamplerOutput>& inputs) {
        input_holder.push(inputs);
    }

private:
    TestDataHolder<pair<SamplerOutput, SamplerOutput>> input_holder;
    TestDataHolder<spec::SpeculativeSamplerOutput>     output_holder;
    size_t                                             forward_count_ = 0;
};

class FakeSampler: public Sampler {
public:
    FakeSampler(const SamplerInitParams& params): Sampler(params) {}

    SamplerOutput forward(const SamplerInputs& inputs) override {
        ++forward_count_;
        if (inputs.logits_processor_states_ptr) {
            inputs.logits_processor_states_ptr->batchProcess(inputs);
        }
        checkInputs(inputs);
        return output_holder.get();
    }

    size_t forwardCount() const {
        return forward_count_;
    }

    void checkInputs(const SamplerInputs& inputs) {
        auto expected_inputs = input_holder.get();
        RTP_LLM_LOG_INFO("check sampler logits");
        checkTensorEqual(inputs.logits, expected_inputs.logits);
    }

    void setInputs(const vector<SamplerInputs>& inputs) {
        input_holder.push(inputs);
    }

    void setOutputs(const vector<SamplerOutput>& outputs) {
        output_holder.push(outputs);
    }

private:
    TestDataHolder<SamplerInputs> input_holder;
    TestDataHolder<SamplerOutput> output_holder;
    size_t                        forward_count_ = 0;
};

class RejectDraftTokenSpecProcessor: public BaseLogitsProcessor {
public:
    explicit RejectDraftTokenSpecProcessor(int32_t rejected_token, int64_t accepted_token_len):
        rejected_token_(rejected_token), accepted_token_len_(accepted_token_len) {}

    std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        inputs.logits.narrow(0, start_idx, finish_idx - start_idx).fill_(BaseLogitsProcessor::neg_inf);
        return std::nullopt;
    }
    void                     updateMultiSeqStatus(const std::vector<int>&) override {}
    std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t num_new_tokens) override {
        accepted_token_len_ += num_new_tokens;
        return std::nullopt;
    }

    bool isStateful() const override {
        return true;
    }

    int64_t acceptedTokenLen() const override {
        return accepted_token_len_;
    }

    MtpProcessorCapability mtpCapability() const override {
        return {MtpProcessorMode::SPEC_VERIFY, {}};
    }

    ErrorResult<int> prepareSpeculative(const SpecLogitsProcessorRequest& request) override {
        {
            std::lock_guard<std::mutex> lock(observation_mutex_);
            invocation_thread_id_ = std::this_thread::get_id();
            observed_draft_tokens_.clear();
            if (request.draft_tokens != nullptr && request.propose_step > 0) {
                observed_draft_tokens_.assign(request.draft_tokens, request.draft_tokens + request.propose_step);
            }
        }
        if (request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
            return ErrorResult<int>(int(request.propose_step));
        }
        std::fill_n(request.bitmask_cpu_out,
                    static_cast<size_t>(request.propose_step + 1) * request.bitmask_size_int32,
                    SpecLogitsProcessorRequest::kBitmaskAllowAll);
        if (request.bitmask_size_int32 > 0 && rejected_token_ >= 0
            && static_cast<size_t>(rejected_token_) < request.vocab_size) {
            request.bitmask_cpu_out[rejected_token_ / 32] &= ~(1u << (rejected_token_ % 32));
        }
        if (request.draft_tokens != nullptr && request.draft_tokens[0] == rejected_token_) {
            return ErrorResult<int>(0);
        }
        return ErrorResult<int>(int(request.propose_step));
    }

    std::thread::id invocationThreadId() const {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        return invocation_thread_id_;
    }

    std::vector<int32_t> observedDraftTokens() const {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        return observed_draft_tokens_;
    }

private:
    int32_t rejected_token_;
    int64_t accepted_token_len_;

    mutable std::mutex   observation_mutex_;
    std::thread::id      invocation_thread_id_;
    std::vector<int32_t> observed_draft_tokens_;
};

class TestableMtpExecutor: public MtpExecutor {
public:
    using MtpExecutor::MtpExecutor;
    using MtpExecutor::asyncPrepareAllowed;
    using MtpExecutor::prepareAndCommitSpeculativeTargetState;
    using MtpExecutor::reduceSpeculativeTargetPrepareStatus;
    using MtpExecutor::rendezvousSpeculativeTargetPhaseStatus;
    using MtpExecutor::rendezvousSpeculativeTargetSamplingStatus;
    using MtpExecutor::runTargetVerifyForward;
    using MtpExecutor::validateSpeculativeTargetExecutionPolicy;
};

TEST(MtpExecutorPolicyTest, TargetHooksDisableSpeculativeAsyncPrepare) {
    EXPECT_TRUE(TestableMtpExecutor::asyncPrepareAllowed(/*requested=*/true, /*target_commit_hooks=*/false));
    EXPECT_FALSE(TestableMtpExecutor::asyncPrepareAllowed(/*requested=*/true, /*target_commit_hooks=*/true));
    EXPECT_FALSE(TestableMtpExecutor::asyncPrepareAllowed(/*requested=*/false, /*target_commit_hooks=*/false));
    EXPECT_FALSE(TestableMtpExecutor::asyncPrepareAllowed(/*requested=*/false, /*target_commit_hooks=*/true));
}

TEST(MtpExecutorPolicyTest, TargetHooksRejectEplbBeforeForward) {
    EXPECT_TRUE(TestableMtpExecutor::validateSpeculativeTargetExecutionPolicy(
                    /*target_commit_hooks=*/false, /*eplb_enabled=*/true)
                    .ok());
    EXPECT_TRUE(TestableMtpExecutor::validateSpeculativeTargetExecutionPolicy(
                    /*target_commit_hooks=*/true, /*eplb_enabled=*/false)
                    .ok());
    const auto unsupported = TestableMtpExecutor::validateSpeculativeTargetExecutionPolicy(
        /*target_commit_hooks=*/true, /*eplb_enabled=*/true);
    EXPECT_EQ(unsupported.code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_NE(std::string(unsupported.message()).find("do not yet support EPLB"), std::string::npos);
}

struct MtpExecutorComponents {
    std::unique_ptr<TestableMtpExecutor>    executor;
    std::unique_ptr<FakeModel>              fake_target_model;
    std::unique_ptr<FakeModel>              fake_draft_model;
    std::unique_ptr<FakeModel>              fake_draft_prefill_model;
    std::unique_ptr<FakeFastTopKSampler>    fake_fast_topk_sampler;
    std::unique_ptr<FakeSpeculativeSampler> fake_speculative_sampler;
    std::unique_ptr<FakeSampler>            fake_sampler;
    ModelConfig                             model_config;
    RuntimeConfig                           runtime_config;
    ResourceContext                         resource_context;
};

class MtpExecutorTest: public DeviceTestBase {
public:
    void runMultiBatchDecode(bool use_mtp_hidden_buffer);

    GenerateStreamPtr createContextStream(const ModelConfig&     model_config,
                                          const RuntimeConfig&   runtime_config,
                                          const ResourceContext& resource_context,
                                          const vector<int>&     input_ids,
                                          int                    max_new_tokens = -1) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        query->generate_config = make_shared<GenerateConfig>();
        if (max_new_tokens >= 0) {
            query->generate_config->max_new_tokens = max_new_tokens;
        }
        GenerateStreamPtr stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        return stream;
    }

    GenerateStreamPtr createDecodeStream(const ModelConfig&          model_config,
                                         const RuntimeConfig&        runtime_config,
                                         const ResourceContext&      resource_context,
                                         const vector<int>&          input_ids,
                                         const StreamSpecUpdateInfo& spec_update_info) {
        GenerateStreamPtr stream = createContextStream(model_config, runtime_config, resource_context, input_ids);

        auto sp_buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_buffer->tokens = torch::tensor({-1, -1}, torch::kInt32).reshape({1, 2});

        stream->setSPOutputBuffer(sp_buffer);
        stream->specUpdate(spec_update_info);
        return stream;
    }

    void checkOutput(const GenerateStreamPtr& stream,
                     const vector<int>&       expect_token_ids,
                     const vector<int>&       expect_propose_tokens,
                     const vector<float>&     expect_all_probs,
                     const vector<float>&     expect_last_hidden_states) {
        auto token_ids = stream->getCompleteTokenIds()->completeTokenIdsVec(0);
        EXPECT_EQ(expect_token_ids, token_ids);

        auto sp_output_buffer = stream->getSPOutputBuffer();
        auto tokens           = sp_output_buffer->tokens;
        auto tokens_h         = tokens.cpu().clone();
        EXPECT_EQ(expect_propose_tokens, toVec<int>(tokens_h));

        auto all_probs   = sp_output_buffer->all_probs;
        auto all_probs_h = all_probs.is_cuda() ? all_probs.cpu() : all_probs;
        EXPECT_EQ(expect_all_probs, toVec<float>(all_probs_h));

        if (expect_last_hidden_states.size() > 0) {
            auto last_hidden_states   = sp_output_buffer->hidden_states;
            auto last_hidden_states_h = last_hidden_states.is_cuda() ? last_hidden_states.cpu() : last_hidden_states;
            EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states_h));
        } else {
            EXPECT_TRUE(!sp_output_buffer->hidden_states.defined());
        }
    }

    MtpExecutorComponents createMtpExecutorComponents(const MtpExecutorTestConfig& test_config,
                                                      const CacheConfig*           cache_config_override = nullptr) {
        CustomConfig               config;
        ModelConfig                model_config;
        RuntimeConfig              runtime_config;
        KVCacheConfig              kv_cache_config;
        ResourceContext            resource_context;
        SpeculativeExecutionConfig sp_config;

        model_config.max_seq_len                          = test_config.max_seq_len;
        model_config.vocab_size                           = test_config.vocab_size;
        model_config.num_layers                           = test_config.num_layers;
        model_config.attn_config.rope_config.index_factor = test_config.position_id_len_factor;
        sp_config.type                                    = test_config.sp_type;
        sp_config.gen_num_per_cycle                       = test_config.gen_num_per_cycle;
        sp_config.sp_dspark_mask_token_id                 = test_config.dspark_mask_token_id;

        resource_context.cache_manager =
            std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                                            /*block_num=*/10,
                                                                            /*tokens_per_block=*/2,
                                                                            rtp_llm::TYPE_INT8,
                                                                            /*local_head_num_kv=*/128,
                                                                            /*size_per_head=*/256));
        resource_context.role_type = test_config.role_type;

        EngineInitParams params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
        // createEngineInitParams resets MM defaults; apply the requested style afterwards.
        model_config.mm_model_config.mm_position_ids_style              = test_config.mm_position_ids_style;
        params.model_config_.mm_model_config.mm_position_ids_style      = test_config.mm_position_ids_style;
        params.sp_config                                                = sp_config;
        params.pd_sep_config.role_type                                  = test_config.role_type;
        params.parallelism_config.role_type                             = test_config.role_type;
        params.parallelism_config.tp_size                               = test_config.tp_size;
        params.parallelism_config.world_size                            = test_config.tp_size;
        params.profiling_debug_logging_config.enable_batch_record       = test_config.enable_batch_record;
        params.profiling_debug_logging_config.record_decode_sample_rate = test_config.record_decode_sample_rate;
        if (test_config.vocab_size_override > 0) {
            params.model_config_.vocab_size = test_config.vocab_size_override;
        }

        ModelConfig score_cache_model_config   = params.model_config_;
        ModelConfig propose_cache_model_config = params.model_config_;
        score_cache_model_config.num_layers    = 1;
        propose_cache_model_config.num_layers  = 1;
        setDefaultMhaKVCacheSpecDescs(score_cache_model_config);
        setDefaultMhaKVCacheSpecDescs(propose_cache_model_config);

        KVCacheConfig test_kv_cache_config;
        test_kv_cache_config.test_block_num            = 10;
        test_kv_cache_config.seq_size_per_block        = 2;
        test_kv_cache_config.kernel_seq_size_per_block = 2;

        auto configure_cache_model = [](ModelConfig& cache_model_config) {
            cache_model_config.data_type                           = rtp_llm::TYPE_INT8;
            cache_model_config.attn_config.kv_head_num             = 128;
            cache_model_config.attn_config.size_per_head           = 256;
            cache_model_config.attn_config.tokens_per_block        = 2;
            cache_model_config.attn_config.kernel_tokens_per_block = 2;
            cache_model_config.attn_config.kv_cache_dtype          = KvCacheDataType::BASE;
        };
        configure_cache_model(score_cache_model_config);
        configure_cache_model(propose_cache_model_config);

        auto cache_sp_config              = sp_config;
        cache_sp_config.gen_num_per_cycle = 1;
        auto           cache_config       = CacheConfigCreator::createConfig(score_cache_model_config,
                                                             params.parallelism_config,
                                                             test_kv_cache_config,
                                                             cache_sp_config,
                                                             &propose_cache_model_config,
                                                             /*is_mtp=*/true,
                                                             /*is_eagle=*/false);
        const uint32_t local_block_num    = CacheConfigCreator::computeLocalBlockNum(cache_config,
                                                                                  score_cache_model_config,
                                                                                  params.runtime_config,
                                                                                  test_kv_cache_config,
                                                                                  params.parallelism_config,
                                                                                  /*warm_up_result=*/std::nullopt,
                                                                                  cache_sp_config);
        cache_config.finalizeBlockNums(local_block_num, params.runtime_config);

        if (cache_config_override != nullptr) {
            cache_config = *cache_config_override;
        }

        // Create propose model engine init params
        auto mtp_model_params   = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        auto mtp_params         = std::make_unique<EngineInitParams>(params);
        mtp_params->py_sp_model = py::none();
        if (test_config.sp_type == SP_TYPE_DSPARK) {
            auto       markov_options    = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            const auto markov_vocab_size = static_cast<int64_t>(params.model_config_.vocab_size);
            mtp_params->gpt_weights.dspark_markov_w1 = torch::zeros({markov_vocab_size, 1}, markov_options);
            mtp_params->gpt_weights.dspark_markov_w2 = torch::zeros({markov_vocab_size, 1}, markov_options);
        }

        mtp_model_params->push_back(std::move(mtp_params));

        auto propose_params = std::make_unique<ProposeModelEngineInitParams>(
            test_config.sp_type, sp_config.gen_num_per_cycle, std::move(mtp_model_params));

        // Create cache managers
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init();

        // Create MtpExecutor
        auto executor = std::make_unique<TestableMtpExecutor>(params, propose_params, cache_manager);

        // Create fake models
        GptModelInitParams target_model_params(
            {params.gpt_weights,
             Executor::genModelDescription(
                 params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
             std::nullopt,
             params.model_id,
             params.parallelism_config});

        GptModelInitParams draft_model_params(
            {params.gpt_weights,
             Executor::genModelDescription(
                 params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
             std::nullopt,
             params.model_id,
             params.parallelism_config});

        auto fake_target_model        = std::make_unique<FakeModel>(target_model_params);
        auto fake_draft_model         = std::make_unique<FakeModel>(draft_model_params);
        auto fake_draft_prefill_model = std::make_unique<FakeModel>(draft_model_params);
        auto fake_fast_topk_sampler   = std::make_unique<FakeFastTopKSampler>();
        auto fake_speculative_sampler = std::make_unique<FakeSpeculativeSampler>(sp_config.gen_num_per_cycle);
        auto fake_sampler             = std::make_unique<FakeSampler>(SamplerInitParams{});

        MtpExecutorComponents components;
        components.executor                 = std::move(executor);
        components.fake_target_model        = std::move(fake_target_model);
        components.fake_draft_model         = std::move(fake_draft_model);
        components.fake_draft_prefill_model = std::move(fake_draft_prefill_model);
        components.fake_fast_topk_sampler   = std::move(fake_fast_topk_sampler);
        components.fake_speculative_sampler = std::move(fake_speculative_sampler);
        components.fake_sampler             = std::move(fake_sampler);
        components.model_config             = model_config;
        components.runtime_config           = runtime_config;
        components.resource_context         = resource_context;

        return components;
    }

    void setupFakeModels(MtpExecutor*                            executor,
                         std::unique_ptr<FakeModel>              fake_target_model,
                         std::unique_ptr<FakeModel>              fake_draft_model,
                         std::unique_ptr<FakeFastTopKSampler>    fake_fast_topk_sampler,
                         std::unique_ptr<FakeSpeculativeSampler> fake_speculative_sampler,
                         std::unique_ptr<FakeSampler>            fake_sampler,
                         std::unique_ptr<FakeModel>              fake_draft_prefill_model = nullptr) {
        executor->setTargetModel(std::move(fake_target_model));
        executor->setDraftModel(std::move(fake_draft_model));
        if (fake_draft_prefill_model) {
            executor->setDraftPrefillModel(std::move(fake_draft_prefill_model));
        }
        executor->setFastTopKSampler(std::move(fake_fast_topk_sampler));
        executor->setSpeculativeSampler(std::move(fake_speculative_sampler));
        executor->setSampler(std::move(fake_sampler));
    }

    void runTargetOnlyPrefillScenario(RoleType                   role_type,
                                      bool                       simulate_multi_dp,
                                      bool                       expect_draft_forward,
                                      bool                       simulate_cp_mutation   = false,
                                      std::optional<std::string> deferred_capture_error = std::nullopt,
                                      SpeculativeType            sp_type                = SP_TYPE_MTP,
                                      bool                       existing_stream_error  = false) {
        MtpExecutorTestConfig test_config;
        test_config.role_type = role_type;
        test_config.sp_type   = sp_type;
        if (sp_type == SP_TYPE_DSPARK) {
            test_config.dspark_mask_token_id = 0;
        }
        auto components = createMtpExecutorComponents(test_config);
        if (simulate_multi_dp) {
            ASSERT_EQ(components.executor->parallelism_config_.tp_size, 1);
            components.executor->parallelism_config_.dp_size    = 2;
            components.executor->parallelism_config_.world_size = 2;
        }
        auto target_only_stream = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3}, 0);
        EXPECT_TRUE(target_only_stream->generateConfig()->isPrefillOnly());
        EXPECT_TRUE(target_only_stream->generateConfig()->reuse_cache);
        EXPECT_FALSE(target_only_stream->reuseCache());

        auto target_input                  = GptModelInputs{};
        target_input.combo_tokens          = torch::tensor({0, 1, 2, 3}, torch::kInt32);
        target_input.input_lengths         = torch::tensor({4}, torch::kInt32);
        target_input.prefix_lengths        = torch::tensor({0}, torch::kInt32);
        target_input.lm_output_indexes     = torch::tensor({3}, torch::kInt32);
        target_input.skip_lm_head          = true;
        target_input.capture_hidden_states = true;

        auto target_output = GptModelOutputs{};
        target_output.all_hidden_states =
            torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f}).reshape({4, 2});
        vector<string> forward_trace;
        components.fake_target_model->setInputs({target_input});
        components.fake_target_model->setOutputs({target_output});
        components.fake_target_model->setForwardTrace(&forward_trace, "target");
        if (existing_stream_error) {
            components.fake_target_model->setForwardObserver([target_only_stream, &forward_trace]() {
                target_only_stream->reportError(ErrorCode::INVALID_PARAMS, "existing request error");
                forward_trace.push_back("target.request_error_reported");
            });
        }
        auto* draft_forward_model =
            sp_type == SP_TYPE_DSPARK ? components.fake_draft_prefill_model.get() : components.fake_draft_model.get();
        draft_forward_model->setForwardTrace(&forward_trace, "draft");
        if (deferred_capture_error.has_value()) {
            components.fake_target_model->setDeferredHiddenStateCaptureError(*deferred_capture_error);
        }

        if (expect_draft_forward) {
            auto draft_input                  = GptModelInputs{};
            draft_input.combo_tokens          = sp_type == SP_TYPE_DSPARK ? torch::tensor({0, 1, 2, 3}, torch::kInt32) :
                                                                            torch::tensor({1, 2, 3, 0}, torch::kInt32);
            draft_input.input_lengths         = torch::tensor({4}, torch::kInt32);
            draft_input.prefix_lengths        = torch::tensor({0}, torch::kInt32);
            draft_input.lm_output_indexes     = torch::tensor({3}, torch::kInt32);
            draft_input.last_hidden_states    = target_output.all_hidden_states;
            draft_input.skip_lm_head          = true;
            draft_input.capture_hidden_states = false;
            draft_forward_model->setInputs({draft_input});
            draft_forward_model->setOutputs({GptModelOutputs{}});
        }

        auto* target_model        = components.fake_target_model.get();
        auto* draft_model         = draft_forward_model;
        auto* draft_sampler       = components.fake_fast_topk_sampler.get();
        auto* speculative_sampler = components.fake_speculative_sampler.get();
        auto* target_sampler      = components.fake_sampler.get();
        setupFakeModels(components.executor.get(),
                        std::move(components.fake_target_model),
                        std::move(components.fake_draft_model),
                        std::move(components.fake_fast_topk_sampler),
                        std::move(components.fake_speculative_sampler),
                        std::move(components.fake_sampler),
                        std::move(components.fake_draft_prefill_model));

        auto status = components.executor->process(ScheduleOutput{{target_only_stream}});
        ASSERT_TRUE(status.ok());
        if (deferred_capture_error.has_value()) {
            EXPECT_TRUE(target_only_stream->hasError());
            auto output = target_only_stream->nextOutput();
            ASSERT_FALSE(output.ok());
            if (existing_stream_error) {
                EXPECT_EQ(output.status().code(), ErrorCode::INVALID_PARAMS);
                EXPECT_EQ(target_only_stream->stopReason(), "existing request error");
                EXPECT_NE(output.status().ToString().find("existing request error"), std::string::npos);
            } else {
                EXPECT_NE(output.status().ToString().find(*deferred_capture_error), std::string::npos);
            }
        } else {
            EXPECT_FALSE(target_only_stream->hasError());
            EXPECT_TRUE(target_only_stream->hasEvent(StreamEvents::GenerateDone));
            EXPECT_EQ(target_only_stream->getCompleteTokenIds()->completeTokenIdsVec(0), (vector<int>{0, 1, 2, 3}));
            EXPECT_EQ(target_only_stream->outputTokenLen(), 0);
            EXPECT_EQ(target_only_stream->getSPOutputBuffer(), nullptr);

            auto output = target_only_stream->nextOutput();
            ASSERT_TRUE(output.ok());
            ASSERT_EQ(output.value().generate_outputs.size(), 1);
            EXPECT_TRUE(output.value().generate_outputs[0].finished);
            EXPECT_EQ(output.value().generate_outputs[0].output_ids.sizes(), (torch::IntArrayRef{1, 0}));
            EXPECT_FALSE(target_only_stream->hasOutput());
        }

        vector<string> expected_forward_trace{"target"};
        if (existing_stream_error) {
            expected_forward_trace.push_back("target.request_error_reported");
        }
        if (expect_draft_forward) {
            expected_forward_trace.push_back("draft");
        }
        if (deferred_capture_error.has_value()) {
            expected_forward_trace.push_back("target.capture_error_taken");
        }
        EXPECT_EQ(forward_trace, expected_forward_trace);
        EXPECT_EQ(target_model->forwardCount(), 1);
        EXPECT_EQ(draft_model->forwardCount(), expect_draft_forward ? 1 : 0);
        EXPECT_EQ(draft_sampler->forwardCount(), 0);
        EXPECT_EQ(speculative_sampler->forwardCount(), 0);
        EXPECT_EQ(target_sampler->forwardCount(), 0);
    }

    void runSnapshotFaultScenario(const char* directory);
    void runSingleBatchPrefillScenario(bool                         simulate_cp_mutation,
                                       bool                         with_cp_hidden_buffers = true,
                                       const MtpExecutorTestConfig* config_override        = nullptr,
                                       MtpExecutorComponents*       out                    = nullptr);
    void runSingleBatchDecodeScenario(bool                         with_prefill_only,
                                      const MtpExecutorTestConfig* config_override   = nullptr,
                                      int64_t*                     out_kv_len_before = nullptr);

    GptModelOutputs createRandomGptModelOutputs(size_t token_num, size_t vocab_size, size_t hidden_size) {
        auto output              = GptModelOutputs{};
        output.logits            = torch::rand({(int64_t)token_num, (int64_t)vocab_size}, torch::kFloat32);
        output.all_hidden_states = torch::rand({(int64_t)token_num, (int64_t)hidden_size}, torch::kFloat32);
        return output;
    }
};

class MtpCacheStrideTest: public MtpExecutorTest, public ::testing::WithParamInterface<int> {};

TEST_P(MtpCacheStrideTest, ForwardAndPrepareUseSingleGroupStrideOnly) {
    const bool multiple_groups  = GetParam() != 0;
    const bool draft_has_linear = GetParam() == 2;
    auto       make_config      = [](bool with_linear, uint32_t head_size, bool single_layer = false) {
        auto full   = test::makeMhaSpec("full", 4, TYPE_FP16, 1, head_size);
        auto config = test::makeSingleLayerCacheConfig(full, CacheGroupType::FULL, 8);
        if (with_linear) {
            config.layer_num = single_layer ? 1 : 2;
            config.fromGroupedSpecs({test::makeLinearSpec("linear", 4, TYPE_FP16, 1, 1), full},
                                    {{0}, {single_layer ? 0 : 1}},
                                    {CacheGroupType::LINEAR, CacheGroupType::FULL},
                                    {"linear", "full"});
        }
        auto groups = config.topology().groups();
        for (auto& group : groups) {
            if (group.tag == "full") {
                test::setGroupLayout(group, head_size * 16, head_size * 4);  // bytes/physical block/layer
            }
        }
        config.setTopology(std::move(groups), config.topology().layers());
        return config;
    };
    auto target = make_config(multiple_groups, 3);
    auto draft  = make_config(draft_has_linear, 1, true);
    target.mtp_sub_configs.push_back(target.mergeMTPModule(draft, 0, target.layer_num));
    target.finalizeBlockNums(8, RuntimeConfig{});
    if (multiple_groups && !draft_has_linear) {
        EXPECT_TRUE(target.mtp_sub_configs[0]->layerIdsForGroup("linear").empty());
        EXPECT_EQ(target.mtp_sub_configs[0]->groupNums(), 2);  // groups, including placeholder
    }

    auto         components         = createMtpExecutorComponents(MtpExecutorTestConfig{}, &target);
    auto*        executor           = components.executor.get();
    const size_t target_stride      = multiple_groups ? 0 : 48;
    const size_t target_scale       = multiple_groups ? 0 : 12;
    const size_t draft_stride       = multiple_groups ? 0 : 16;
    const size_t draft_scale        = multiple_groups ? 0 : 4;
    bool         saw_target_prepare = false;
    bool         saw_draft_prepare  = false;
    components.fake_target_model->setInputObserver([&](const GptModelInputs& inputs) {
        saw_target_prepare = true;
        EXPECT_EQ(inputs.kv_block_stride_bytes, target_stride);
        EXPECT_EQ(inputs.kv_scale_stride_bytes, target_scale);
    });
    components.fake_draft_prefill_model->setInputObserver([&](const GptModelInputs& inputs) {
        saw_draft_prepare = true;
        EXPECT_EQ(inputs.kv_block_stride_bytes, draft_stride);
        EXPECT_EQ(inputs.kv_scale_stride_bytes, draft_scale);
    });
    executor->setTargetModel(std::move(components.fake_target_model));
    executor->setDraftPrefillModel(std::move(components.fake_draft_prefill_model));

    const auto     cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    GptModelInputs input;
    input.combo_tokens          = torch::zeros({1}, cuda_i32);
    input.input_lengths         = torch::ones({1}, cuda_i32);
    input.prefix_lengths        = torch::ones({1}, cuda_i32);
    input.sequence_lengths      = torch::full({1}, 2, cuda_i32);
    input.kv_block_stride_bytes = 999;
    input.kv_scale_stride_bytes = 777;
    if (executor->useAsyncPrepare()) {
        executor->launchTargetVerifyPrepareAsync(input, 1);
        executor->launchDraftPrefillPrepareAsync(input);
        executor->target_verify_prepare_runner_.sync(cuda_graph::graphGetCurrentStream());
        executor->draft_prefill_prepare_runner_.sync(cuda_graph::graphGetCurrentStream());
        EXPECT_TRUE(saw_target_prepare);
        EXPECT_TRUE(saw_draft_prepare);
        EXPECT_EQ(input.kv_block_stride_bytes, 999u);
        EXPECT_EQ(input.kv_scale_stride_bytes, 777u);
    }

    executor->broadcastPostRejectionInputs(input);
    EXPECT_EQ(input.kv_block_stride_bytes, draft_stride);
    EXPECT_EQ(input.kv_scale_stride_bytes, draft_scale);

    bool saw_proposal = false;
    components.fake_draft_model->setInputObserver([&](const GptModelInputs& inputs) {
        saw_proposal = true;
        EXPECT_EQ(inputs.kv_block_stride_bytes, draft_stride);
        EXPECT_EQ(inputs.kv_scale_stride_bytes, draft_scale);
    });
    components.fake_draft_model->setInputs({input});
    components.fake_draft_model->setOutputs({GptModelOutputs{}});
    executor->setDraftModel(std::move(components.fake_draft_model));
    input.kv_block_stride_bytes = 999;
    input.kv_scale_stride_bytes = 777;
    // A non-root proposal forwards inputs without entering root-only sampling.
    executor->tp_rank_ = 1;
    StreamGroups          streams(std::list<GenerateStreamPtr>{});
    mtp::DSparkRoundState round;
    SamplerOutput         output;
    int64_t               forward_us = 0;
    executor->runDSparkProposal(input, streams, round, output, forward_us);
    EXPECT_TRUE(saw_proposal);
}

INSTANTIATE_TEST_SUITE_P(CacheTopology, MtpCacheStrideTest, ::testing::Values(0, 1, 2));

TEST_F(MtpExecutorTest, MixedCacheSnapshotsPrepareTokensLengthsAndPositionsWithoutWaiting) {
    const char* previous = std::getenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE");
    struct EnvGuard {
        std::optional<std::string> previous;
        ~EnvGuard() {
            if (previous)
                setenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE");
        }
    } env_guard{previous ? std::optional<std::string>(previous) : std::nullopt};
    setenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "1", 1);
    for (int width : {1, 3}) {
        SCOPED_TRACE(width);
        MtpExecutorTestConfig config;
        config.gen_num_per_cycle = width;
        auto components          = createMtpExecutorComponents(config);
        auto initial             = StreamSpecUpdateInfo{torch::tensor({{3}}, torch::kInt32),
                                            1,
                                            torch::tensor({2}, torch::kInt32),
                                            torch::ones({1, 2}),
                                            torch::ones({1, 4})};
        auto steady              = createDecodeStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 1, 2}, initial);
        auto fresh = createDecodeStream(
            components.model_config, components.runtime_config, components.resource_context, {1, 2}, initial);
        const auto&                         cache = components.executor->cache_manager_->cacheConfig();
        GenerateStream::MtpAsyncDeviceState state;
        state.next_kv_cache_block_id_gpu =
            torch::ones({cache.groupNums(), 1, 2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        state.next_kv_cache_kernel_block_id_gpu = state.next_kv_cache_block_id_gpu.clone();
        state.accept_tokens_gpu                 = torch::tensor({{1, 3}}, torch::kInt32).cuda();
        state.accept_len_gpu                    = torch::tensor({2}, torch::kInt32).cuda();
        state.next_seq_len_gpu                  = torch::tensor({6}, torch::kInt32).cuda();
        state.propose_tokens_gpu                = torch::tensor({{1}}, torch::kInt32).cuda();
        state.last_hidden_states_gpu            = torch::ones({1, 2}).cuda();
        steady->setMtpAsyncDeviceState(std::move(state));
        steady->incPendingAsyncBookkeeping();
        struct PendingGuard {
            GenerateStreamPtr stream;
            ~PendingGuard() {
                stream->decPendingAsyncBookkeepingAndMaybeRelease();
            }
        } pending{steady};
        auto model_config                    = components.model_config;
        model_config.has_positional_encoding = true;
        SpeculativeExecutionConfig speculative;
        speculative.gen_num_per_cycle = width;
        MtpBatchStreamProcessor processor(
            model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache, speculative, false);
        TensorHolder holder;
        // There is no worker to finish this counter: an added fallback wait
        // would deadlock instead of consuming the immutable snapshot.
        components.executor->waitPreviousBookkeepingBeforeStreamPreparation({steady, fresh});
        ASSERT_TRUE(steady->hasPendingAsyncBookkeeping());
        ASSERT_TRUE(steady->hasMtpCacheSnapshot());
        auto input = processor.gatherDecodeModelInput(StreamGroups({steady, fresh}), holder);
        ASSERT_TRUE(input.ok());
        if (width == 1) {
            processor.prepareOneStepSpecDecodeModelInput(StreamGroups({steady, fresh}), *input, holder);
            EXPECT_EQ((std::vector<int>{3, 1, 3, 2}), toVec<int>(input->combo_tokens));
            EXPECT_EQ((std::vector<int>{5, 6, 2, 3}), toVec<int>(input->combo_position_ids));
        } else {
            processor.prepareDecodeDraftModelInput(StreamGroups({steady, fresh}), *input, holder);
            EXPECT_EQ((std::vector<int>{1, 2}), toVec<int>(input->combo_tokens));
            EXPECT_EQ((std::vector<int>{5, 2}), toVec<int>(input->combo_position_ids));
        }
        EXPECT_EQ((std::vector<int>{5, 2}), toVec<int>(input->prefix_lengths));
        EXPECT_TRUE(steady->hasPendingAsyncBookkeeping());
        fresh->incPendingAsyncBookkeeping();
        EXPECT_THROW(processor.expandTargetVerifyPositionIds(StreamGroups({fresh}), *input), std::runtime_error);
        fresh->decPendingAsyncBookkeepingAndMaybeRelease();
    }
}

TEST_F(MtpExecutorTest, testMtpConstructorRejectsMissingProposal) {
    EngineInitParams params;
    params.model_id                    = 0;
    params.sp_config.gen_num_per_cycle = 3;
    std::unique_ptr<ProposeModelEngineInitParams> proposal;

    for (const auto role : {RoleType::PREFILL, RoleType::DECODE, RoleType::PDFUSION}) {
        params.parallelism_config.role_type = role;
        params.pd_sep_config.role_type      = role;
        for (const auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
            params.sp_config.type = type;
            for (const bool warm_up : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "role=" << static_cast<int>(role)
                                                  << " type=" << static_cast<int>(type) << " warm_up=" << warm_up);
                try {
                    MtpExecutor executor(params, proposal, nullptr, 1, warm_up);
                    FAIL() << "MtpExecutor without a proposer must fail construction";
                } catch (const std::exception& error) {
                    EXPECT_NE(std::string(error.what()).find("MtpExecutor requires local proposal parameters"),
                              std::string::npos)
                        << error.what();
                }
            }
        }
    }
}

// The recorder singleton is process-wide: constructing an MtpExecutor without
// enable_batch_record first would leave it Disabled for the whole binary, so
// this runs as a death test to own a fresh process (RecordingEngineTest pattern).
TEST_F(MtpExecutorTest, CtorPublishesSpMetadata) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            char directory[] = "/tmp/rtp-mtp-record-ctor-XXXXXX";
            ASSERT_NE(mkdtemp(directory), nullptr);
            setenv("LOG_PATH", directory, 1);
            MtpExecutorTestConfig test_config;
            test_config.enable_batch_record = true;
            test_config.gen_num_per_cycle   = 2;
            auto components                 = createMtpExecutorComponents(test_config);
            ExecutionRecorder::instance().close();
            bool        manifest_found = false;
            std::string manifest;
            for (const auto& entry : std::filesystem::directory_iterator(std::string(directory) + "/batch_record")) {
                if (entry.path().filename().string().rfind("owner-", 0) == 0) {
                    std::ifstream in(entry.path() / "manifest.json");
                    std::string   content((std::istreambuf_iterator<char>(in)), {});
                    if (content.find("\"scope\":\"mtp_executor\"") != std::string::npos) {
                        manifest_found = true;
                        manifest       = content;
                    }
                }
            }
            ASSERT_TRUE(manifest_found) << "no mtp_executor manifest under " << directory;
            for (const auto* field : {"cache_block_tokens", "cache_groups", "cache_config"}) {
                EXPECT_NE(manifest.find(std::string("\"") + field + "\":"), std::string::npos) << manifest;
            }
            EXPECT_NE(manifest.find("\"sp_type\":\"mtp\""), std::string::npos) << manifest;
            EXPECT_NE(manifest.find("\"propose_step\":2"), std::string::npos) << manifest;
            EXPECT_NE(manifest.find("\"is_dspark\":false"), std::string::npos) << manifest;
            EXPECT_NE(manifest.find("\"decode_filter_policy\":\"sp_verify_step_sampling\""), std::string::npos)
                << manifest;
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

// Snapshot-anomaly containment: a gather path that skips the CPU length
// snapshot must fail the recording only (warning + dropped record), never the
// serving step — mirroring NormalExecutor's try/catch contract. Runs as a
// death test for process-wide singleton isolation.
class SnapshotFaultProcessor: public MtpBatchStreamProcessor {
public:
    using MtpBatchStreamProcessor::MtpBatchStreamProcessor;
    absl::StatusOr<GptModelInputs>
    gatherModelInput(const StreamGroups& stream_groups, TensorHolder& host_holder, bool record_lengths) const override {
        (void)record_lengths;  // Drop the flag: simulate a snapshot-less gather.
        return NormalBatchStreamProcessor::gatherModelInput(stream_groups, host_holder, false);
    }
};

void MtpExecutorTest::runSnapshotFaultScenario(const char* directory) {
    setenv("LOG_PATH", directory, 1);
    MtpExecutorTestConfig test_config;
    test_config.enable_batch_record = true;
    auto components                 = createMtpExecutorComponents(test_config);
    // Minimal single-request prefill data, mirroring the batch scenario.
    GenerateStreamPtr stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});
    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3}, torch::kInt32);
    target_output.logits           = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({1, 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f}).reshape({4, 2});
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});
    auto draft_input               = GptModelInputs{};
    auto draft_output              = GptModelOutputs{};
    draft_input.combo_tokens       = torch::tensor({1, 2, 3, 1}, torch::kInt32);
    draft_input.input_lengths      = torch::tensor({4}, torch::kInt32);
    draft_input.prefix_lengths     = torch::tensor({0}, torch::kInt32);
    draft_input.lm_output_indexes  = torch::tensor({3}, torch::kInt32);
    draft_input.last_hidden_states = target_output.all_hidden_states;
    draft_output.logits            = torch::tensor({0.5f, 0.6f, 0.7f, 0.8f}).reshape({1, 4});
    draft_output.all_hidden_states =
        torch::tensor({0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f}).reshape({4, 2});
    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});
    auto sampler_input  = SamplerInputs{target_output.logits};
    auto sampler_output = SamplerOutput{torch::tensor({1}, torch::kInt32).reshape({1, 1})};
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});
    const auto topk_probs = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4});
    const auto topk_ids   = torch::tensor({2}, torch::kInt32).reshape({1, 1});
    const auto fast_topk  = spec::FastTopKSamplerOutput{topk_probs, topk_ids};
    components.fake_fast_topk_sampler->setInputs({draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({fast_topk});
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));
    components.executor->setBatchProcessor(
        std::make_unique<SnapshotFaultProcessor>(components.model_config,
                                                 PDSepConfig{},
                                                 ProfilingDebugLoggingConfig{},
                                                 components.resource_context.cache_manager->cacheConfig(),
                                                 SpeculativeExecutionConfig{},
                                                 false));
    bool step_ok = false;
    try {
        const auto status = components.executor->process(ScheduleOutput{{stream}});
        step_ok           = status.ok() && !stream->hasError();
    } catch (...) {}
    ExecutionRecorder::instance().close();
    if (!step_ok) {
        fprintf(stderr, "snapshot fault must not fail the serving step\n");
        _exit(1);
    }
    _exit(0);
}

TEST_F(MtpExecutorTest, SnapshotFaultDoesNotFailServingStep) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    char directory[]                        = "/tmp/rtp-mtp-record-fault-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    ASSERT_EXIT(runSnapshotFaultScenario(directory), ::testing::ExitedWithCode(0), "");
}

TEST_F(MtpExecutorTest, testMtpConstructorRejectsZeroProposalWidth) {
    MtpExecutorTestConfig test_config;
    test_config.sp_type           = SP_TYPE_MTP;
    test_config.gen_num_per_cycle = 0;

    try {
        (void)createMtpExecutorComponents(test_config);
        FAIL() << "expected ordinary MTP construction to reject zero proposal width";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("proposal width/gen_num_per_cycle must be positive"), std::string::npos);
    }
}

void MtpExecutorTest::runSingleBatchPrefillScenario(bool                         simulate_cp_mutation,
                                                    bool                         with_cp_hidden_buffers,
                                                    const MtpExecutorTestConfig* config_override,
                                                    MtpExecutorComponents*       out) {
    MtpExecutorTestConfig test_config;
    if (config_override) {
        test_config = *config_override;
    }
    test_config.gen_num_per_cycle = 4;
    auto components               = createMtpExecutorComponents(test_config);
    if (simulate_cp_mutation) {
        components.executor->parallelism_config_.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    }
    int profile_starts   = 0;
    int profile_finishes = 0;
    components.executor->setModelForwardCallbacks([&]() { ++profile_starts; }, [&]() { ++profile_finishes; });
    ASSERT_TRUE(components.executor->process({}).ok());
    EXPECT_EQ(profile_starts, 0);
    EXPECT_EQ(profile_finishes, 0);

    size_t batch_size = 1;

    // Create context stream
    GenerateStreamPtr stream1 = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});
    if (simulate_cp_mutation) {
        stream1->generateConfig()->max_new_tokens = 8;
        components.fake_target_model->setPostForwardInputMutation(torch::tensor({0, 1}, torch::kInt32),
                                                                  torch::tensor({2}, torch::kInt32));
    }

    // set fake model outputs
    auto target_input  = GptModelInputs{};
    auto target_output = GptModelOutputs{};

    // set fake target model inputs
    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3}, torch::kInt32);
    target_output.logits           = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({(int64_t)batch_size, 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f}).reshape({4, 2});
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set fake draft model outputs
    auto draft_input               = GptModelInputs{};
    auto draft_output              = GptModelOutputs{};
    draft_input.combo_tokens       = torch::tensor({1, 2, 3, 1}, torch::kInt32);
    draft_input.input_lengths      = torch::tensor({4}, torch::kInt32);
    draft_input.prefix_lengths     = torch::tensor({0}, torch::kInt32);
    draft_input.lm_output_indexes  = torch::tensor({3}, torch::kInt32);
    draft_input.last_hidden_states = target_output.all_hidden_states;
    draft_output.logits            = torch::tensor({0.5f, 0.6f, 0.7f, 0.8f}).reshape({(int64_t)batch_size, 4});
    draft_output.all_hidden_states =
        torch::tensor({0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f}).reshape({4, 2});
    if (simulate_cp_mutation && with_cp_hidden_buffers) {
        // Global token geometry is restored, but hidden rows stay CP-local.
        draft_input.last_hidden_states = target_output.all_hidden_states.narrow(0, 0, 2);
        components.fake_target_model->setMtpTargetHiddenStates(draft_input.last_hidden_states);
        components.fake_draft_model->setMtpLastHiddenStates(draft_output.all_hidden_states.narrow(0, 3, 1).clone());
        draft_output.all_hidden_states = draft_output.all_hidden_states.narrow(0, 0, 2);
        EXPECT_TRUE(components.fake_target_model->hasMtpTargetHiddenBuffer());
        EXPECT_EQ(draft_input.last_hidden_states.size(0), 2);
        EXPECT_EQ(draft_output.all_hidden_states.size(0), 2);
    }
    if (!simulate_cp_mutation) {
        // Draft prefill uses the output's own row count. The public MTP hidden
        // tensor must replace the ordinary hidden states without requiring an
        // explicit row count from the caller.
        draft_output.mtp_target_hidden_states =
            torch::tensor({0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f, 0.28f}).reshape({4, 2});
    }

    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    // set fake sampler outputs
    auto sampler_input  = SamplerInputs{target_output.logits};
    auto sampler_output = SamplerOutput{torch::tensor({1}, torch::kInt32).reshape({(int64_t)batch_size, 1})};
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // set fake fast topk sampler outputs
    auto fast_topk_sampler_output =
        spec::FastTopKSamplerOutput{torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({(int64_t)batch_size, 4}),
                                    torch::tensor({2}, torch::kInt32).reshape({(int64_t)batch_size, 1})};
    components.fake_fast_topk_sampler->setInputs({draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({fast_topk_sampler_output});

    auto* target_model        = components.fake_target_model.get();
    auto* draft_model         = components.fake_draft_model.get();
    auto* draft_sampler       = components.fake_fast_topk_sampler.get();
    auto* speculative_sampler = components.fake_speculative_sampler.get();
    auto* target_sampler      = components.fake_sampler.get();

    // Replace models with fake models
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    if (simulate_cp_mutation && !with_cp_hidden_buffers) {
        EXPECT_FALSE(target_model->hasMtpTargetHiddenBuffer());
        try {
            (void)components.executor->process(ScheduleOutput{{stream1}});
            FAIL() << "expected CP prefill to reject a missing draft last-hidden buffer";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("CP MTP draft last-hidden buffer must contain per-request rows"),
                      std::string::npos);
        }
        if (out) {
            *out = std::move(components);
        }
        return;
    }

    // Verify executor was created successfully
    auto status = components.executor->process(ScheduleOutput{{stream1}});
    ASSERT_TRUE(status.ok());
    if (simulate_cp_mutation) {
        EXPECT_FALSE(stream1->hasError());
    }
    // PDFUSION runs prefill followed by an idle decode phase here: count one
    // real work step, with balanced callbacks, rather than both phases.
    EXPECT_EQ(profile_starts, 1);
    EXPECT_EQ(profile_finishes, 1);

    // check stream result
    checkOutput(stream1,
                {0, 1, 2, 3, 1},
                {1, 2},
                {0.0, 0.0, 1.0, 0.0},
                simulate_cp_mutation ? std::vector<float>{0.17, 0.18} : std::vector<float>{0.27, 0.28});
    if (simulate_cp_mutation) {
        EXPECT_EQ(target_model->forwardCount(), 1);
        EXPECT_EQ(draft_model->forwardCount(), 1);
        EXPECT_EQ(draft_sampler->forwardCount(), 1);
        EXPECT_EQ(speculative_sampler->forwardCount(), 0);
        EXPECT_EQ(target_sampler->forwardCount(), 1);
    }
    if (out) {
        *out = std::move(components);
    }
}

TEST_F(MtpExecutorTest, testSingleBatchPrefill) {
    runSingleBatchPrefillScenario(/*simulate_cp_mutation=*/false);
}

// Phase-1 speculative recording: prefill batches only, request timing via the
// shared FIFO enqueue path (request_id itself is scheduler-side and asserted
// by RecordingEngineTest). Death test for singleton isolation.
TEST_F(MtpExecutorTest, PrefillStepRecordsBatch) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            char directory[] = "/tmp/rtp-mtp-record-prefill-XXXXXX";
            ASSERT_NE(mkdtemp(directory), nullptr);
            setenv("LOG_PATH", directory, 1);
            const auto batch_config =
                std::string("alog.max_msg_len=0\nalog.logger.batch_schedule=INFO, batchScheduleAppender\n")
                + "inherit.batch_schedule=false\nalog.appender.batchScheduleAppender=FileAppender\n"
                  "alog.appender.batchScheduleAppender.fileName="
                + std::string(directory)
                + "/batch_schedule.log\n"
                  "alog.appender.batchScheduleAppender.layout=PatternLayout\n"
                  "alog.appender.batchScheduleAppender.layout.LogPattern=%%m\n"
                  "alog.appender.batchScheduleAppender.async_flush=true\n";
            alog::Configurator::configureLoggerFromString(batch_config.c_str());
            MtpExecutorTestConfig test_config;
            test_config.enable_batch_record = true;
            MtpExecutorComponents components;
            runSingleBatchPrefillScenario(false, true, &test_config, &components);
            // Empty alignment round must not record anything new.
            ASSERT_TRUE(components.executor->process({}).ok());
            // close() drains the recorder queue before flushing alog; flushing
            // alog alone cannot wait for records still queued in the recorder.
            ExecutionRecorder::instance().close();
            std::ifstream in(std::string(directory) + "/batch_schedule.log");
            ASSERT_TRUE(in.good());
            const std::string  batches((std::istreambuf_iterator<char>(in)), {});
            std::istringstream lines(batches);
            size_t             batch_count = 0;
            for (std::string line; std::getline(lines, line);) {
                if (!line.empty()) {
                    ++batch_count;
                }
            }
            EXPECT_EQ(batch_count, 1u) << "expected only the prefill batch; empty process() must not record: "
                                       << batches;
            EXPECT_NE(batches.find("\"phase\":\"prefill\""), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"input_len\":4"), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"q_len\":4"), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"kv_len\":0"), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"ttft_us\":"), std::string::npos) << batches;
            EXPECT_EQ(batches.find("\"ttft_us\":null"), std::string::npos) << batches;
            {
                const auto key_pos = batches.find("\"step_duration_us\":");
                ASSERT_NE(key_pos, std::string::npos) << batches;
                ASSERT_GT(std::stoull(batches.substr(key_pos + strlen("\"step_duration_us\":"), 12)), 0u) << batches;
            }
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

// Phase-2 speculative recording: a decode step records one sp_verify row per
// request slot with the verify width as q_len, the pre-verify cached length as
// kv_len, and the accepted token count backfilled at dispatch (same timing
// pattern as ttft_us). Draft-forward composition stays invisible to replay.
TEST_F(MtpExecutorTest, DecodeStepRecordsVerifyRows) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            char directory[] = "/tmp/rtp-mtp-record-decode-XXXXXX";
            ASSERT_NE(mkdtemp(directory), nullptr);
            setenv("LOG_PATH", directory, 1);
            const auto batch_config =
                std::string("alog.max_msg_len=0\nalog.logger.batch_schedule=INFO, batchScheduleAppender\n")
                + "inherit.batch_schedule=false\nalog.appender.batchScheduleAppender=FileAppender\n"
                  "alog.appender.batchScheduleAppender.fileName="
                + std::string(directory)
                + "/batch_schedule.log\n"
                  "alog.appender.batchScheduleAppender.layout=PatternLayout\n"
                  "alog.appender.batchScheduleAppender.layout.LogPattern=%%m\n"
                  "alog.appender.batchScheduleAppender.async_flush=true\n";
            alog::Configurator::configureLoggerFromString(batch_config.c_str());
            MtpExecutorTestConfig test_config;
            test_config.enable_batch_record = true;
            int64_t kv_len_before           = -1;
            runSingleBatchDecodeScenario(false, &test_config, &kv_len_before);
            ASSERT_GE(kv_len_before, 0);
            rtp_llm::Logger::getBatchScheduleLogger()->flush();
            ExecutionRecorder::instance().close();
            std::ifstream     in(std::string(directory) + "/batch_schedule.log");
            const std::string batches((std::istreambuf_iterator<char>(in)), {});
            EXPECT_NE(batches.find("\"phase\":\"sp_verify\""), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"q_len\":5"), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"kv_len\":" + std::to_string(kv_len_before)), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"accept_len\":3"), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"request_id\":\"r"), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"ttft_us\":"), std::string::npos) << batches;
            EXPECT_EQ(batches.find("\"accept_len\":null"), std::string::npos) << batches;
            EXPECT_EQ(batches.find("\"kv_len\":null"), std::string::npos) << batches;
            {
                const auto key_pos = batches.find("\"step_duration_us\":");
                ASSERT_NE(key_pos, std::string::npos) << batches;
                ASSERT_GT(std::stoull(batches.substr(key_pos + strlen("\"step_duration_us\":"), 12)), 0u) << batches;
            }
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

// The Bresenham decode sampling gate applies to speculative verify steps with
// the same semantics as the normal engine: rate < 1 thins steps without
// spending credit on non-recordable (empty/warmup) rounds.
TEST_F(MtpExecutorTest, DecodeSampleRateThinsSpVerifyRecords) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            char directory[] = "/tmp/rtp-mtp-record-sample-XXXXXX";
            ASSERT_NE(mkdtemp(directory), nullptr);
            setenv("LOG_PATH", directory, 1);
            const auto batch_config =
                std::string("alog.max_msg_len=0\nalog.logger.batch_schedule=INFO, batchScheduleAppender\n")
                + "inherit.batch_schedule=false\nalog.appender.batchScheduleAppender=FileAppender\n"
                  "alog.appender.batchScheduleAppender.fileName="
                + std::string(directory)
                + "/batch_schedule.log\n"
                  "alog.appender.batchScheduleAppender.layout=PatternLayout\n"
                  "alog.appender.batchScheduleAppender.layout.LogPattern=%%m\n"
                  "alog.appender.batchScheduleAppender.async_flush=true\n";
            alog::Configurator::configureLoggerFromString(batch_config.c_str());
            MtpExecutorTestConfig test_config;
            test_config.enable_batch_record       = true;
            test_config.record_decode_sample_rate = 0.5;
            // First decode step: credit 0.5 < 1.0 -> the step is skipped whole.
            runSingleBatchDecodeScenario(false, &test_config, nullptr);
            rtp_llm::Logger::getBatchScheduleLogger()->flush();
            ExecutionRecorder::instance().close();
            std::ifstream     in(std::string(directory) + "/batch_schedule.log");
            const std::string batches((std::istreambuf_iterator<char>(in)), {});
            EXPECT_EQ(batches.find("\"phase\":\"sp_verify\""), std::string::npos) << batches;
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(MtpExecutorTest, testSingleBatchPrefillRestoresCpInputBeforeDraftShift) {
    runSingleBatchPrefillScenario(/*simulate_cp_mutation=*/true);
}

TEST_F(MtpExecutorTest, testCpPrefillDoesNotFallBackWithoutHiddenBuffers) {
    runSingleBatchPrefillScenario(/*simulate_cp_mutation=*/true, /*with_cp_hidden_buffers=*/false);
}

TEST_F(MtpExecutorTest, testTargetOnlyRunsTargetPrefillWithoutDraftOrSampling) {
    runTargetOnlyPrefillScenario(RoleType::PREFILL, false, false);
}

TEST_F(MtpExecutorTest, testTargetOnlyRunsDraftPrefillForMultiDpAlignmentWithoutSampling) {
    runTargetOnlyPrefillScenario(RoleType::PREFILL, true, true);
}

TEST_F(MtpExecutorTest, testDeferredCaptureErrorDoesNotOverwriteExistingStreamError) {
    runTargetOnlyPrefillScenario(RoleType::PREFILL, false, false, false, "injected capture failure", SP_TYPE_MTP, true);
}

TEST_F(MtpExecutorTest, testDeferredCaptureErrorCompletesMultiDpDraftAlignmentBeforeFailing) {
    runTargetOnlyPrefillScenario(RoleType::PREFILL, true, true, false, "injected capture failure");
}

TEST_F(MtpExecutorTest, testDSparkTargetOnlySingleDpRunsCommitBeforeDispatch) {
    runTargetOnlyPrefillScenario(RoleType::PREFILL, false, true, false, std::nullopt, SP_TYPE_DSPARK);
}

TEST_F(MtpExecutorTest, testDSparkDeferredCaptureErrorSingleDpRunsCommitBeforeFailing) {
    runTargetOnlyPrefillScenario(
        RoleType::PREFILL, false, true, false, "injected dspark capture failure", SP_TYPE_DSPARK);
}

TEST_F(MtpExecutorTest, testDSparkTargetOnlyMultiDpAlignmentRunsCommitDraftAfterTargetWithoutSampling) {
    runTargetOnlyPrefillScenario(RoleType::PREFILL, true, true, false, std::nullopt, SP_TYPE_DSPARK);
}

TEST_F(MtpExecutorTest, testDSparkDeferredCaptureErrorCompletesMultiDpCommitAlignmentBeforeFailing) {
    runTargetOnlyPrefillScenario(
        RoleType::PREFILL, true, true, false, "injected dspark capture failure", SP_TYPE_DSPARK);
}

TEST_F(MtpExecutorTest, testTargetOnlyEarlyReturnRequiresSingleDpReplica) {
    EXPECT_TRUE(MtpExecutor::canEarlyReturnTargetOnlyPrefill(1));
    EXPECT_FALSE(MtpExecutor::canEarlyReturnTargetOnlyPrefill(2));
}

TEST_F(MtpExecutorTest, testEmptyDecodeIsSkipped) {
    EXPECT_TRUE(MtpExecutor::shouldSkipEmptyDecode(true));
    EXPECT_FALSE(MtpExecutor::shouldSkipEmptyDecode(false));
}

TEST_F(MtpExecutorTest, testHiddenStateCapturePolicy) {
    const std::vector<int64_t> configured_layer_ids = {0, 2};

    EXPECT_EQ(selectHiddenStateCaptureLayerIds(
                  HiddenStateCaptureModelRole::TARGET, RoleType::PREFILL, false, configured_layer_ids),
              configured_layer_ids);
    EXPECT_EQ(selectHiddenStateCaptureLayerIds(
                  HiddenStateCaptureModelRole::TARGET, RoleType::PDFUSION, false, configured_layer_ids),
              configured_layer_ids);
    EXPECT_TRUE(selectHiddenStateCaptureLayerIds(
                    HiddenStateCaptureModelRole::TARGET, RoleType::DECODE, false, configured_layer_ids)
                    .empty());
    EXPECT_TRUE(selectHiddenStateCaptureLayerIds(
                    HiddenStateCaptureModelRole::TARGET, RoleType::PREFILL, true, configured_layer_ids)
                    .empty());
    EXPECT_TRUE(selectHiddenStateCaptureLayerIds(
                    HiddenStateCaptureModelRole::DRAFT, RoleType::PREFILL, false, configured_layer_ids)
                    .empty());
}

TEST_F(MtpExecutorTest, testTargetOnlyPrefillCollectsContextTokenAndTpsMetrics) {
    MtpExecutorTestConfig test_config;
    test_config.role_type = RoleType::PREFILL;
    auto components       = createMtpExecutorComponents(test_config);
    auto stream           = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3}, 0);

    StreamGroups        stream_groups({stream});
    MtpMetricsCollector metrics_collector;
    components.executor->collectPrefillMetrics(
        stream_groups, metrics_collector, /*schedule_time_us=*/1, /*model_forward_us=*/123);

    const auto& executor_collector = metrics_collector.executor_collector;
    EXPECT_EQ(executor_collector.context_batch_size, 1);
    EXPECT_EQ(executor_collector.execute_token_size, 4);
    EXPECT_EQ(executor_collector.max_seq_len, 4);
    EXPECT_EQ(executor_collector.context_batch_size_when_has_context, 1);
    EXPECT_EQ(executor_collector.execute_token_size_when_has_context, 4);
    EXPECT_EQ(executor_collector.max_seq_len_when_has_context, 4);
    EXPECT_EQ(executor_collector.model_forward_us, 123);
    EXPECT_TRUE(metrics_collector.tps_collector.hasContextTPS());
    EXPECT_TRUE(metrics_collector.tps_collector.hasContextTPSWithCache());
    EXPECT_DOUBLE_EQ(metrics_collector.tps_collector.totalTPS(), 4.0);
}

TEST_F(MtpExecutorTest, testDpFakeDecodePreservesDecodeWorkContract) {
    constexpr int         propose_step = 4;
    MtpExecutorTestConfig test_config;
    test_config.role_type         = RoleType::DECODE;
    test_config.gen_num_per_cycle = propose_step;
    auto components               = createMtpExecutorComponents(test_config);
    auto fake_decode_stream       = MtpExecutor::createMinFakeDecodeStream(propose_step,
                                                                     components.model_config,
                                                                     components.runtime_config,
                                                                     components.resource_context,
                                                                     components.model_config.vocab_size);

    // NormalEngine uses this stream when a DP decode rank is idle. It must remain decode work so the rank enters
    // the same model collective sequence; skipping it requires a separate cross-rank collective-safety proof.
    ASSERT_NE(fake_decode_stream, nullptr);
    EXPECT_TRUE(fake_decode_stream->isFakeStream());
    EXPECT_FALSE(fake_decode_stream->isContextStream());
    EXPECT_FALSE(fake_decode_stream->generateConfig()->isPrefillOnly());
    ASSERT_NE(fake_decode_stream->getSPOutputBuffer(), nullptr);
    EXPECT_EQ(fake_decode_stream->getSPOutputBuffer()->propose_step, propose_step);

    auto* target_model = components.fake_target_model.get();
    target_model->enableSpeculativeTargetCommit("fake block tables must not be prepared");
    components.executor->setTargetModel(std::move(components.fake_target_model));
    size_t reductions                                                = 0;
    components.executor->speculative_target_status_reducer_for_test_ = [&](bool ok) {
        ++reductions;
        return ok;
    };
    auto accept_len = torch::tensor({propose_step + 1}, torch::kInt32).to(torch::kCUDA);
    auto status = components.executor->prepareAndCommitSpeculativeTargetState(accept_len, 1, /*is_fake_stream=*/true);
    ASSERT_TRUE(status.ok());
    EXPECT_FALSE(target_model->preparedAcceptLen().defined());
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 1u);
    EXPECT_EQ(target_model->speculativeTargetFinalizeCount(), 0u);
    EXPECT_EQ(reductions, 1u);
}

TEST_F(MtpExecutorTest, testDecodeRoleRejectsPrefillOnlyWithoutModelWork) {
    MtpExecutorTestConfig test_config;
    test_config.role_type = RoleType::DECODE;
    auto components       = createMtpExecutorComponents(test_config);

    auto stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3}, 0);
    EXPECT_TRUE(stream->generateConfig()->isPrefillOnly());
    EXPECT_TRUE(stream->generateConfig()->reuse_cache);
    EXPECT_FALSE(stream->reuseCache());

    auto* target_model   = components.fake_target_model.get();
    auto* draft_model    = components.fake_draft_model.get();
    auto* target_sampler = components.fake_sampler.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    auto status = components.executor->process(ScheduleOutput{{stream}});
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(stream->stopReason(), kDecodeRolePrefillOnlyError);

    auto output = stream->nextOutput();
    ASSERT_FALSE(output.ok());
    EXPECT_EQ(output.status().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(target_model->forwardCount(), 0);
    EXPECT_EQ(draft_model->forwardCount(), 0);
    EXPECT_EQ(target_sampler->forwardCount(), 0);
}

TEST_F(MtpExecutorTest, testPrepareStreamsRejectsMixedPrefillExecutionBatchPerRequest) {
    MtpExecutorTestConfig test_config;
    test_config.role_type = RoleType::PDFUSION;
    auto components       = createMtpExecutorComponents(test_config);

    for (bool generation_first : {false, true}) {
        SCOPED_TRACE(generation_first ? "generation-first" : "prefill-first");
        auto prefill_only = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 1}, 0);
        auto generation = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {2, 3}, 1);
        auto streams = generation_first ? std::list<GenerateStreamPtr>{generation, prefill_only} :
                                          std::list<GenerateStreamPtr>{prefill_only, generation};

        std::list<GenerateStreamPtr> prefill_streams;
        std::list<GenerateStreamPtr> decode_streams;
        components.executor->prepareStreams(streams, prefill_streams, decode_streams);

        EXPECT_TRUE(prefill_streams.empty());
        EXPECT_TRUE(decode_streams.empty());
        for (const auto& stream : {prefill_only, generation}) {
            EXPECT_TRUE(stream->hasError());
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(stream->stopReason(), kMixedExecutionModeBatchError);
        }
    }
}

TEST_F(MtpExecutorTest, testCPChunkedPrefillRestoresGlobalPositionIds) {
    // Use the real CP splitter with fake model forwards and TP collectives.
    for (const bool explicit_position_ids : {false, true}) {
        for (const bool middle_chunk : {false, true}) {
            SCOPED_TRACE(testing::Message()
                         << "explicit_position_ids=" << explicit_position_ids << ", middle_chunk=" << middle_chunk);
            MtpExecutorTestConfig test_config;
            test_config.gen_num_per_cycle                         = 2;
            test_config.mm_position_ids_style                     = explicit_position_ids ? MROPE : DEFAULT;
            test_config.position_id_len_factor                    = explicit_position_ids ? 3 : 1;
            auto  components                                      = createMtpExecutorComponents(test_config);
            auto& executor                                        = *components.executor;
            executor.parallelism_config_.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
            executor.warm_up_ = true;  // Stop after draft forward, before draft sampling.

            constexpr int prefix_len = 4;
            constexpr int chunk_len  = 8;
            const int     prompt_len = prefix_len + chunk_len + (middle_chunk ? 2 : 0);
            vector<int>   prompt{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1};
            prompt.resize(prompt_len);
            auto stream = createContextStream(
                components.model_config, components.runtime_config, components.resource_context, prompt);
            stream->setReuseLength(prefix_len);
            stream->setChunkSize(chunk_len);
            stream->enableWarmupChunkWindow();
            ASSERT_EQ(stream->isMiddleChunk(), middle_chunk);

            torch::Tensor prompt_positions;
            if (explicit_position_ids) {
                prompt_positions =
                    (torch::arange(prompt_len, torch::kInt32).unsqueeze(1) * 10 + torch::arange(3, torch::kInt32))
                        .flatten();
                stream->setContextPositionIds(prompt_positions);
            }

            GptModelInputs target_input;
            target_input.combo_tokens      = torch::tensor({0, 1, 2, 3, 0, 1, 2, 3}, torch::kInt32);
            target_input.input_lengths     = torch::tensor({chunk_len}, torch::kInt32);
            target_input.prefix_lengths    = torch::tensor({prefix_len}, torch::kInt32);
            target_input.lm_output_indexes = torch::tensor({chunk_len - 1}, torch::kInt32);
            if (explicit_position_ids) {
                target_input.combo_position_ids = prompt_positions.narrow(0, prefix_len * 3, chunk_len * 3);
            }
            GptModelOutputs target_output;
            target_output.logits            = torch::zeros({1, 4});
            target_output.all_hidden_states = torch::zeros({chunk_len / 2, 2});
            components.fake_target_model->setInputs({target_input});
            components.fake_target_model->setOutputs({target_output});
            components.fake_target_model->forward_hook = [explicit_position_ids](GptModelInputs& input) {
                ParallelismConfig cp_config;
                cp_config.tp_size = 2;
                cp_config.tp_rank = 0;
                torch_ext::PyContextParallelParams cp_params;
                ZigZagProcessor(cp_config).handleInputs(input, cp_params);
                const int factor = explicit_position_ids ? 3 : 1;
                EXPECT_EQ(input.combo_tokens.numel(), chunk_len / 2);
                EXPECT_EQ(input.combo_position_ids.numel(), chunk_len / 2 * factor);
                // Preserve the real rank-local shape, but give the old buggy
                // memmove enough backing storage to fail assertions safely.
                auto storage         = torch::full({chunk_len * factor}, -999, input.combo_position_ids.options());
                auto local_positions = storage.narrow(0, 0, input.combo_position_ids.numel());
                local_positions.copy_(input.combo_position_ids);
                input.combo_position_ids = local_positions;
            };

            GptModelInputs draft_input     = target_input;
            draft_input.combo_tokens       = torch::tensor({1, 2, 3, 0, 1, 2, 3, middle_chunk ? 0 : 2}, torch::kInt32);
            draft_input.last_hidden_states = target_output.all_hidden_states;
            if (explicit_position_ids) {
                auto tail = middle_chunk ? torch::tensor({120, 121, 122}, torch::kInt32) :
                                           torch::tensor({112, 112, 112}, torch::kInt32);
                draft_input.combo_position_ids =
                    torch::cat({prompt_positions.narrow(0, (prefix_len + 1) * 3, (chunk_len - 1) * 3), tail});
            }
            components.fake_draft_model->setInputs({draft_input});
            components.fake_draft_model->setOutputs({GptModelOutputs{}});
            components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
            components.fake_sampler->setOutputs({SamplerOutput{torch::tensor({{2}}, torch::kInt32)}});
            executor.setTargetModel(std::move(components.fake_target_model));
            executor.setDraftModel(std::move(components.fake_draft_model));
            executor.setSampler(std::move(components.fake_sampler));
            MtpMetricsCollector metrics;
            EXPECT_TRUE(executor.prefillStep({stream}, metrics, 0).ok());
        }
    }
}

TEST_F(MtpExecutorTest, testDSparkPrefillCommitDoesNotUseTargetVerifyContract) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle    = 3;
    test_config.vocab_size_override  = test_config.vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    test_config.role_type            = RoleType::PREFILL;
    auto components                  = createMtpExecutorComponents(test_config);

    GenerateStreamPtr stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({1, 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f}).reshape({4, 2});
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    GptModelInputs commit_input     = target_input;
    commit_input.last_hidden_states = target_output.all_hidden_states;
    components.fake_draft_prefill_model->setInputs({commit_input});
    components.fake_draft_prefill_model->setOutputs({GptModelOutputs{}});
    components.fake_draft_prefill_model->expectTargetVerify(false);

    auto publication_events = std::make_shared<std::vector<std::string>>();
    components.fake_target_model->setEventLog(publication_events, "target");
    components.fake_draft_prefill_model->setEventLog(publication_events, "draft");
    auto assert_not_dispatched = [stream]() {
        EXPECT_EQ((std::vector<int>{0, 1, 2, 3}), stream->getCompleteTokenIds()->completeTokenIdsVec(0));
    };
    components.fake_target_model->setPublicationObserver(assert_not_dispatched);
    components.fake_draft_prefill_model->setPublicationObserver(assert_not_dispatched);

    auto sampler_input  = SamplerInputs{target_output.logits};
    auto sampler_output = SamplerOutput{torch::tensor({1}, torch::kInt32).reshape({1, 1})};
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler),
                    std::move(components.fake_draft_prefill_model));

    auto status = components.executor->process(ScheduleOutput{{stream}});
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ((std::vector<std::string>{
                  "target.forward", "draft.forward", "target.wait_publication", "draft.wait_publication"}),
              *publication_events);
    EXPECT_EQ((std::vector<int>{0, 1, 2, 3, 1}), stream->getCompleteTokenIds()->completeTokenIdsVec(0));
    EXPECT_TRUE(stream->getProposeToken().empty());
}

namespace {

enum class PublicationOutcome {
    Success,
    Error,
    Exception
};

struct PublicationCase {
    const char*        name;
    int                tp_size;
    PublicationOutcome target;
    PublicationOutcome draft;
    bool               peer_ok;
};

/** Exercise the real TP publication reduction while scripting the collective's peer result. */
class PublicationCollectives {
public:
    explicit PublicationCollectives(std::function<bool(bool)> reduce) {
        ops_ = py::module_::import("types").attr("ModuleType")("mtp_publication_test").cast<py::module_>();
        registerExecCtxOps(ops_);
        ops_.attr("register_comm_ops")(
            py::cpp_function([](const std::vector<torch::Tensor>&, int64_t root, int mode) {
                EXPECT_EQ(root, 0);
                EXPECT_EQ(mode, static_cast<int>(ParallelMode::TP));
            }),
            py::cpp_function([reduce](torch::Tensor status, int op, int mode, py::object dest) {
                EXPECT_EQ(op, static_cast<int>(ReduceOp::Min));
                EXPECT_EQ(mode, static_cast<int>(ParallelMode::TP));
                EXPECT_TRUE(dest.is_none());
                return torch::full_like(status, reduce(status.item<int32_t>() != 0));
            }),
            py::cpp_function([](py::args) { ADD_FAILURE() << "unexpected allgather"; }),
            py::cpp_function([](py::args) { ADD_FAILURE() << "unexpected isend"; }),
            py::cpp_function([](py::args) { ADD_FAILURE() << "unexpected irecv"; }));
    }

    ~PublicationCollectives() {
        ops_.attr("clear_comm_ops")();
    }

private:
    py::module_ ops_;
};

}  // namespace

class MtpExecutorPublicationTest: public MtpExecutorTest, public ::testing::WithParamInterface<PublicationCase> {
public:
    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<py::scoped_interpreter>();
        /** Resolve Python OpenSSL modules before the executable's linked BoringSSL symbols. */
        py::exec(R"(
import os
import sys
flags = sys.getdlopenflags()
try:
    sys.setdlopenflags(flags | os.RTLD_DEEPBIND)
    import hashlib
    import ssl
finally:
    sys.setdlopenflags(flags)
)");
        py::module_::import("torch");
    }

    static void TearDownTestSuite() {
        interpreter_.reset();
    }

private:
    inline static std::unique_ptr<py::scoped_interpreter> interpreter_;
};

TEST_P(MtpExecutorPublicationTest, DispatchesOnlyAfterAllRanksPublishBothModels) {
    const auto& publication = GetParam();
    const bool  local_ok =
        publication.target == PublicationOutcome::Success && publication.draft == PublicationOutcome::Success;
    const bool            success = local_ok && publication.peer_ok;
    MtpExecutorTestConfig test_config;
    test_config.tp_size              = publication.tp_size;
    test_config.gen_num_per_cycle    = 3;
    test_config.vocab_size_override  = test_config.vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    test_config.role_type            = RoleType::PREFILL;
    auto components                  = createMtpExecutorComponents(test_config);

    GenerateStreamPtr stream = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({1, 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f}).reshape({4, 2});
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    int  target_waits          = 0;
    int  draft_waits           = 0;
    int  reductions            = 0;
    auto assert_not_dispatched = [&] {
        EXPECT_FALSE(stream->hasError());
        EXPECT_TRUE(stream->isContextStream());
        EXPECT_EQ((std::vector<int>{0, 1, 2, 3}), stream->getCompleteTokenIds()->completeTokenIdsVec(0));
    };
    auto configure_publication = [&](FakeModel& model, PublicationOutcome outcome, int& waits) {
        model.setPublicationObserver([&, counter = &waits] {
            ++*counter;
            assert_not_dispatched();
        });
        if (outcome == PublicationOutcome::Error) {
            model.setPublicationResult("store publication failed");
        } else if (outcome == PublicationOutcome::Exception) {
            model.setPublicationException(std::make_exception_ptr(std::runtime_error("store wait threw")));
        }
    };
    configure_publication(*components.fake_target_model, publication.target, target_waits);
    configure_publication(*components.fake_draft_prefill_model, publication.draft, draft_waits);
    PublicationCollectives collectives([&](bool reported_local_ok) {
        ++reductions;
        EXPECT_EQ(target_waits, 1);
        EXPECT_EQ(draft_waits, 1);
        EXPECT_EQ(reported_local_ok, local_ok);
        assert_not_dispatched();
        return reported_local_ok && publication.peer_ok;
    });

    GptModelInputs commit_input     = target_input;
    commit_input.last_hidden_states = target_output.all_hidden_states;
    components.fake_draft_prefill_model->setInputs({commit_input});
    components.fake_draft_prefill_model->setOutputs({GptModelOutputs{}});
    components.fake_draft_prefill_model->expectTargetVerify(false);

    auto sampler_input  = SamplerInputs{target_output.logits};
    auto sampler_output = SamplerOutput{torch::tensor({1}, torch::kInt32).reshape({1, 1})};
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    auto* target_model  = components.fake_target_model.get();
    auto* draft_prefill = components.fake_draft_prefill_model.get();
    auto* draft_model   = components.fake_draft_model.get();
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler),
                    std::move(components.fake_draft_prefill_model));

    auto status = components.executor->process(ScheduleOutput{{stream}});
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(target_model->forwardCount(), 1u);
    EXPECT_EQ(draft_prefill->forwardCount(), 1u);
    EXPECT_EQ(draft_model->forwardCount(), 0u);
    EXPECT_EQ(target_waits, 1);
    EXPECT_EQ(draft_waits, 1);
    EXPECT_EQ(reductions, publication.tp_size > 1 ? 1 : 0);
    EXPECT_EQ(stream->hasError(), !success);
    /** specUpdate clears context before checking errors, so unchanged tokens alone cannot exclude dispatch. */
    EXPECT_EQ(stream->isContextStream(), !success);
    if (success) {
        EXPECT_EQ((std::vector<int>{0, 1, 2, 3, 1}), stream->getCompleteTokenIds()->completeTokenIdsVec(0));
    } else {
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CACHE_STORE_STORE_FAILED);
        EXPECT_EQ((std::vector<int>{0, 1, 2, 3}), stream->getCompleteTokenIds()->completeTokenIdsVec(0));
    }
    EXPECT_TRUE(stream->getProposeToken().empty());
}

INSTANTIATE_TEST_SUITE_P(
    DSpark,
    MtpExecutorPublicationTest,
    ::testing::Values(
        PublicationCase{"SingleRankSuccess", 1, PublicationOutcome::Success, PublicationOutcome::Success, true},
        PublicationCase{"SingleRankBothFail", 1, PublicationOutcome::Exception, PublicationOutcome::Error, true},
        PublicationCase{"AllRanksSuccess", 2, PublicationOutcome::Success, PublicationOutcome::Success, true},
        PublicationCase{"PeerOnlyFailure", 2, PublicationOutcome::Success, PublicationOutcome::Success, false},
        PublicationCase{"TargetError", 2, PublicationOutcome::Error, PublicationOutcome::Success, true},
        PublicationCase{"DraftError", 2, PublicationOutcome::Success, PublicationOutcome::Error, true},
        PublicationCase{"TargetException", 2, PublicationOutcome::Exception, PublicationOutcome::Success, true},
        PublicationCase{"DraftException", 2, PublicationOutcome::Success, PublicationOutcome::Exception, true},
        PublicationCase{"BothFail", 2, PublicationOutcome::Exception, PublicationOutcome::Error, true}),
    [](const ::testing::TestParamInfo<PublicationCase>& info) { return info.param.name; });

TEST_F(MtpExecutorTest, testMultiBatchPrefill) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 4;
    auto components               = createMtpExecutorComponents(test_config);

    size_t batch_size = 2;

    // Create context stream
    GenerateStreamPtr stream1 = createContextStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});
    GenerateStreamPtr stream2 =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {2, 3});

    // set fake model outputs
    auto target_input  = GptModelInputs{};
    auto target_output = GptModelOutputs{};

    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4, 2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0, 0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3, 5}, torch::kInt32);
    target_output.logits =
        torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 1.1f, 1.2f, 1.3f, 1.4f}).reshape({(int64_t)batch_size, 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 1.01f, 1.02f, 1.03f, 1.04f})
            .reshape({6, 2});

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set fake draft model inputs
    auto draft_input  = GptModelInputs{};
    auto draft_output = GptModelOutputs{};

    draft_input.combo_tokens       = torch::tensor({1, 2, 3, 1, 3, 0}, torch::kInt32);
    draft_input.input_lengths      = torch::tensor({4, 2}, torch::kInt32);
    draft_input.prefix_lengths     = torch::tensor({0, 0}, torch::kInt32);
    draft_input.lm_output_indexes  = torch::tensor({3, 5}, torch::kInt32);
    draft_input.last_hidden_states = target_output.all_hidden_states;
    draft_output.logits =
        torch::tensor({0.5f, 0.6f, 0.7f, 0.8f, 1.5f, 1.6f, 1.7f, 1.8f}).reshape({(int64_t)batch_size, 4});
    draft_output.all_hidden_states =
        torch::tensor({0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 1.11f, 1.12f, 1.13f, 1.14f})
            .reshape({6, 2});

    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    // set fake sampler outputs
    auto sampler_input  = SamplerInputs{target_output.logits};
    auto sampler_output = SamplerOutput{torch::tensor({1, 0}, torch::kInt32).reshape({(int64_t)batch_size, 1})};

    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // set fake fast topk sampler inputs
    auto fast_topk_sampler_output = spec::FastTopKSamplerOutput{
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({(int64_t)batch_size, 4}),
        torch::tensor({2, 1}, torch::kInt32).reshape({(int64_t)batch_size, 1})};

    components.fake_fast_topk_sampler->setInputs({draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({fast_topk_sampler_output});

    // Replace models with fake models
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    // Verify executor was created successfully
    auto status = components.executor->process(ScheduleOutput{{stream1, stream2}});
    ASSERT_TRUE(status.ok());

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 1}, {1, 2}, {0.0, 0.0, 1.0, 0.0}, {0.17, 0.18});
    checkOutput(stream2, {2, 3, 0}, {0, 1}, {0.0, 0.0, 1.0, 0.0}, {1.13, 1.14});
}

void MtpExecutorTest::runSingleBatchDecodeScenario(bool                         with_prefill_only,
                                                   const MtpExecutorTestConfig* config_override,
                                                   int64_t*                     out_kv_len_before) {
    // test single batch decode accept partial
    // input [0, 1, 2] + [3]
    // darft [3] + [2, 1, 3]
    // verify [3, 2, 0, 0, 0]
    // accept [3, 2, 0]
    // next draft [1]
    size_t propose_step = 4;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = 4;
    if (config_override) {
        test_config.enable_batch_record       = config_override->enable_batch_record;
        test_config.record_decode_sample_rate = config_override->record_decode_sample_rate;
    }
    auto components = createMtpExecutorComponents(test_config);

    size_t batch_size = 1;

    auto stream1_new_tokens        = torch::tensor({{2}}, torch::kInt32);
    auto stream1_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto stream1_draft_token_probs = torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}});

    StreamSpecUpdateInfo spec_update_info1{
        stream1_new_tokens, 1, torch::tensor({3}, torch::kInt32), stream1_hidden_states, stream1_draft_token_probs};

    GenerateStreamPtr stream1 = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1}, spec_update_info1);
    GenerateStreamPtr prefill_only_stream;
    if (with_prefill_only) {
        prefill_only_stream = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3}, 0);
        EXPECT_TRUE(prefill_only_stream->generateConfig()->isPrefillOnly());
        EXPECT_TRUE(prefill_only_stream->generateConfig()->reuse_cache);
        EXPECT_FALSE(prefill_only_stream->reuseCache());
    }

    // set fake prefill-only model inputs
    auto prefill_only_input = GptModelInputs{};
    if (with_prefill_only) {
        prefill_only_input.combo_tokens          = torch::tensor({0, 1, 2, 3}, torch::kInt32);
        prefill_only_input.input_lengths         = torch::tensor({4}, torch::kInt32);
        prefill_only_input.prefix_lengths        = torch::tensor({0}, torch::kInt32);
        prefill_only_input.lm_output_indexes     = torch::tensor({3}, torch::kInt32);
        prefill_only_input.skip_lm_head          = true;
        prefill_only_input.capture_hidden_states = true;
    }

    /** A disabled sampling request stays greedy even when its configured top_k is not one. */
    stream1->generateConfig()->do_sample = false;
    stream1->generateConfig()->top_k     = 4;

    // set 3 step draft model outputs
    auto draft_input_1  = GptModelInputs{};
    auto draft_input_2  = GptModelInputs{};
    auto draft_input_3  = GptModelInputs{};
    auto draft_output_1 = createRandomGptModelOutputs(1, 4, 2);
    auto draft_output_2 = createRandomGptModelOutputs(1, 4, 2);
    auto draft_output_3 = createRandomGptModelOutputs(1, 4, 2);

    draft_input_1.combo_tokens       = torch::tensor({3}, torch::kInt32);
    draft_input_1.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_1.sequence_lengths   = torch::tensor({2}, torch::kInt32);
    draft_input_1.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_1.last_hidden_states = stream1_hidden_states;

    draft_input_2.combo_tokens       = torch::tensor({2}, torch::kInt32);
    draft_input_2.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_2.sequence_lengths   = torch::tensor({3}, torch::kInt32);
    draft_input_2.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_2.last_hidden_states = draft_output_1.all_hidden_states;

    draft_input_3.combo_tokens       = torch::tensor({1}, torch::kInt32);
    draft_input_3.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_3.sequence_lengths   = torch::tensor({4}, torch::kInt32);
    draft_input_3.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_3.last_hidden_states = draft_output_2.all_hidden_states;

    auto next_draft_input    = GptModelInputs{};
    auto next_draft_output   = GptModelOutputs{};
    next_draft_output.logits = torch::tensor({1.9f, 1.10f, 1.11f, 1.12f}).reshape({(int64_t)batch_size, 4});
    next_draft_output.all_hidden_states =
        torch::tensor({0.1f, 0.1f, 0.2f, 0.22f, 0.3f, 0.33f, 0.0f, 0.0f, 0.0f, 0.0f}).reshape({5, 2});

    next_draft_input.combo_tokens      = torch::tensor({3, 2, 0}, torch::kInt32);
    next_draft_input.input_lengths     = torch::tensor({3}, torch::kInt32);
    next_draft_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes = torch::tensor({2}, torch::kInt32);

    // set fake model outputs
    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({2, 3, 2, 1, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({5}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2, 3, 4}, torch::kInt32);

    auto target_prepare_input                    = GptModelInputs{};
    target_prepare_input.input_lengths           = torch::tensor({5}, torch::kInt32);
    target_prepare_input.prefix_lengths          = torch::tensor({2}, torch::kInt32);
    target_prepare_input.sequence_lengths_plus_1 = torch::tensor({3}, torch::kInt32);
    target_prepare_input.lm_output_indexes       = torch::tensor({0}, torch::kInt32);
    components.fake_target_model->setPrepareInputs({target_prepare_input});

    target_output.logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 1.1f, 1.2f, 1.3f, 1.4f, 2.1f, 2.2f,
                                          2.3f, 2.4f, 3.1f, 3.2f, 3.3f, 3.4f, 4.1f, 4.2f, 4.3f, 4.4f})
                               .reshape({(int64_t)(batch_size * (propose_step + 1)), 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f})
            .reshape({(int64_t)(propose_step + 1), 2});
    target_output.mtp_target_hidden_states = target_output.all_hidden_states;
    // Replay returns the current features above even if the model's Python
    // attribute still refers to rows from graph capture.
    components.fake_target_model->setMtpTargetHiddenStates(torch::full_like(target_output.all_hidden_states, -1));

    next_draft_input.last_hidden_states = target_output.all_hidden_states.narrow(0, 0, 3);

    components.fake_draft_model->setInputs({draft_input_1, draft_input_2, draft_input_3});
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3});
    components.fake_draft_model->expectTargetVerify(false);
    components.fake_draft_prefill_model->setInputs({next_draft_input});
    components.fake_draft_prefill_model->setOutputs({next_draft_output});
    components.fake_draft_prefill_model->expectTargetVerify(false);
    if (components.executor->useAsyncPrepare()) {
        // Rejection accepts three of five rows. Preparing the dense verify
        // layout would leave the draft forward using metadata for five rows.
        components.fake_draft_prefill_model->setPrepareInputs({next_draft_input});
    }

    if (with_prefill_only) {
        components.fake_target_model->setInputs({prefill_only_input, target_input});
        components.fake_target_model->setOutputs({GptModelOutputs{}, target_output});
    } else {
        components.fake_target_model->setInputs({target_input});
        components.fake_target_model->setOutputs({target_output});
    }

    // set fake sampler outputs
    auto target_sample_all_probs_data = createRandomVector<float>(batch_size * (propose_step + 1) * vocab_size, 1);
    auto sampler_input                = SamplerInputs{target_output.logits};
    auto sampler_output =
        SamplerOutput{torch::tensor({3, 2, 0, 0, 0}, torch::kInt32).reshape({(int64_t)batch_size, 5})};
    sampler_output.all_probs = torch::tensor(target_sample_all_probs_data)
                                   .reshape({(int64_t)batch_size, (int64_t)(propose_step + 1), (int64_t)vocab_size});
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // draft sampler output [2, 1, 3, 0]
    auto draft_sampler_output_1    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_2    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_3    = spec::FastTopKSamplerOutput{};
    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{};

    draft_sampler_output_1.token_ids    = torch::tensor({2}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_1.all_probs    = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({(int64_t)batch_size, 4});
    draft_sampler_output_2.token_ids    = torch::tensor({1}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_2.all_probs    = torch::tensor({0.0f, 0.0f, 0.0f, 1.0f}).reshape({(int64_t)batch_size, 4});
    draft_sampler_output_3.token_ids    = torch::tensor({3}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_3.all_probs    = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}).reshape({(int64_t)batch_size, 4});
    next_draft_sampler_output.token_ids = torch::tensor({1}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    next_draft_sampler_output.all_probs = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f}).reshape({(int64_t)batch_size, 4});

    components.fake_fast_topk_sampler->setInputs(
        {draft_output_1.logits, draft_output_2.logits, draft_output_3.logits, next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs(
        {draft_sampler_output_1, draft_sampler_output_2, draft_sampler_output_3, next_draft_sampler_output});

    // set fake speculative sampler outputs
    auto accept_tokens                           = torch::tensor({{3, 2, 0, 0, 0}}, torch::kInt32);
    auto speculative_sampler_output              = spec::SpeculativeSamplerOutput();
    speculative_sampler_output.accept_tokens_cpu = accept_tokens;
    speculative_sampler_output.accept_tokens     = accept_tokens.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({3}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    auto draft_spec_sample_input                 = SamplerOutput{};
    auto target_spec_sample_input                = SamplerOutput{};

    vector<vector<float>> draft_all_probs_list;
    draft_all_probs_list.push_back(toVec<float>(stream1_draft_token_probs));
    draft_all_probs_list.push_back(toVec<float>(draft_output_1.logits));
    draft_all_probs_list.push_back(toVec<float>(draft_output_2.logits));
    draft_all_probs_list.push_back(toVec<float>(draft_output_3.logits));
    draft_spec_sample_input.token_ids  = torch::tensor({3, 2, 1, 3}, torch::kInt32).reshape({1, 4});
    draft_spec_sample_input.all_probs  = torch::tensor(catVectors(draft_all_probs_list)).reshape({4, 4});
    target_spec_sample_input.all_probs = draft_spec_sample_input.all_probs;

    components.fake_speculative_sampler->setInputs({draft_spec_sample_input, target_spec_sample_input});
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    // The active draft model runs the decode proposal steps (the first draft
    // token is carried over from the previous cycle); the next-round draft
    // prefill executes on the dedicated prefill slot.
    auto* active_draft_model       = components.fake_draft_model.get();
    auto* draft_prefill_fake_model = components.fake_draft_prefill_model.get();
    auto* fake_target_model        = components.fake_target_model.get();
    auto* fake_speculative_sampler = components.fake_speculative_sampler.get();

    // Replace models with fake models
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler),
                    std::move(components.fake_draft_prefill_model));

    // Verify executor was created successfully
    // Mirror the production enqueue path so recorded rows carry request
    // identity, then capture the pre-step cached length (kv before verify) on
    // the same thread that builds the record rows: the decode step reads the
    // same value after waitPreviousBookkeeping, before any update commits new
    // tokens.
    if (config_override && config_override->enable_batch_record) {
        stream1->recordSchedulerEnqueueTime(autil::TimeUtility::currentTimeInMicroSeconds());
    }
    if (out_kv_len_before && !with_prefill_only) {
        *out_kv_len_before = stream1->seqLength() - 1;
    }
    auto status = with_prefill_only ? components.executor->process(ScheduleOutput{{prefill_only_stream, stream1}}) :
                                      components.executor->process(ScheduleOutput{{stream1}});
    ASSERT_TRUE(status.ok());
    if (with_prefill_only) {
        EXPECT_TRUE(prefill_only_stream->hasEvent(StreamEvents::GenerateDone));
        auto prefill_output = prefill_only_stream->nextOutput();
        ASSERT_TRUE(prefill_output.ok());
        ASSERT_EQ(prefill_output.value().generate_outputs.size(), 1);
        EXPECT_TRUE(prefill_output.value().generate_outputs[0].finished);
        EXPECT_EQ(prefill_output.value().generate_outputs[0].output_ids.sizes(), (torch::IntArrayRef{1, 0}));
    }
    const auto& do_sample = fake_speculative_sampler->last_params.do_sample;
    ASSERT_TRUE(do_sample.defined());
    EXPECT_TRUE(do_sample.device().is_cpu());
    ASSERT_EQ(do_sample.scalar_type(), torch::kBool);
    ASSERT_EQ(do_sample.sizes().vec(), (std::vector<int64_t>{1}));
    EXPECT_FALSE(do_sample[0].item<bool>());
    EXPECT_EQ(active_draft_model->forwardCount(), propose_step - 1);
    EXPECT_EQ(draft_prefill_fake_model->forwardCount(), 1u);
    if (with_prefill_only) {
        EXPECT_EQ(fake_target_model->forwardCount(), 2u);
    }
    if (components.executor->useAsyncPrepare()) {
        EXPECT_FALSE(fake_target_model->hasPendingPrepareInputs());
        EXPECT_FALSE(draft_prefill_fake_model->hasPendingPrepareInputs());
    }

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 2, 0}, {0, 1}, {0.0, 1.0, 0.0, 0.0}, {0.3, 0.33});
}

TEST_F(MtpExecutorTest, testSingleBatchDecode) {
    runSingleBatchDecodeScenario(/*with_prefill_only=*/false);
}

TEST_F(MtpExecutorTest, testPdfusionPrefillOnlyDoesNotSkipSingleBatchDecode) {
    runSingleBatchDecodeScenario(/*with_prefill_only=*/true);
}

TEST_F(MtpExecutorTest, testDecodeSpecLogitsCapReplacesInvalidDraftWithTargetToken) {
    size_t propose_step = 2;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = vocab_size;
    auto components                 = createMtpExecutorComponents(test_config);

    auto                 stream_new_tokens        = torch::tensor({{2}}, torch::kInt32);
    auto                 stream_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto                 stream_draft_token_probs = torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}});
    StreamSpecUpdateInfo spec_update_info{
        stream_new_tokens, 1, torch::tensor({3}, torch::kInt32), stream_hidden_states, stream_draft_token_probs};

    GenerateStreamPtr stream = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1}, spec_update_info);
    stream->sampling_state_.logits_processors.push_back(
        std::make_shared<RejectDraftTokenSpecProcessor>(3, stream->outputTokenLen()));

    auto draft_input_1               = GptModelInputs{};
    auto draft_output_1              = GptModelOutputs{};
    draft_input_1.combo_tokens       = torch::tensor({3}, torch::kInt32);
    draft_input_1.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input_1.sequence_lengths   = torch::tensor({2}, torch::kInt32);
    draft_input_1.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input_1.last_hidden_states = stream_hidden_states;
    draft_output_1.logits            = torch::tensor({0.4f, 0.3f, 0.2f, 0.1f}).reshape({1, 4});
    draft_output_1.all_hidden_states = torch::tensor({0.11f, 0.12f}).reshape({1, 2});

    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({2, 3, 0}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({3}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2}, torch::kInt32);
    target_output.logits = torch::tensor({0.1f, 0.9f, 0.2f, 0.3f, 0.2f, 0.1f, 0.8f, 0.4f, 0.7f, 0.2f, 0.1f, 0.0f})
                               .reshape({3, 4})
                               .to(torch::kCUDA);
    target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f}).reshape({3, 2});

    auto next_draft_input               = GptModelInputs{};
    auto next_draft_output              = GptModelOutputs{};
    next_draft_input.combo_tokens       = torch::tensor({1}, torch::kInt32);
    next_draft_input.input_lengths      = torch::tensor({1}, torch::kInt32);
    next_draft_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    next_draft_input.last_hidden_states = target_output.all_hidden_states.narrow(0, 0, 1);
    next_draft_output.logits            = torch::tensor({0.2f, 0.1f, 0.8f, 0.0f}).reshape({1, 4});
    next_draft_output.all_hidden_states = torch::tensor({0.21f, 0.22f}).reshape({1, 2});

    components.fake_draft_model->setInputs({draft_input_1, next_draft_input});
    // Transactional targets disable the speculative early async prepare. The
    // draft must instead prepare from the final post-rejection ragged input,
    // after the accept-len cap changed the preliminary 3 to 1.
    components.fake_draft_model->setPrepareInputs({next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, next_draft_output});
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});
    auto  transaction_events = std::make_shared<std::vector<std::string>>();
    auto* target_model       = components.fake_target_model.get();
    auto* draft_model        = components.fake_draft_model.get();
    target_model->enableSpeculativeTargetCommit();
    target_model->setEventLog(transaction_events, "target");
    components.fake_draft_model->setEventLog(transaction_events, "draft");

    auto draft_sampler_output_1 = spec::FastTopKSamplerOutput{torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}).reshape({1, 4}),
                                                              torch::tensor({0}, torch::kInt32).reshape({1, 1})};
    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4}), torch::tensor({2}, torch::kInt32).reshape({1, 1})};
    components.fake_fast_topk_sampler->setInputs({draft_output_1.logits, next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({draft_sampler_output_1, next_draft_sampler_output});

    auto sampler_input         = SamplerInputs{target_output.logits.clone()};
    sampler_input.logits[0][3] = BaseLogitsProcessor::neg_inf;
    auto target_sampler_output = SamplerOutput{torch::tensor({1, 2, 2}, torch::kInt32).reshape({3, 1})};
    target_sampler_output.all_probs =
        torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({3, 4});
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({target_sampler_output});

    auto forced_accept_tokens                    = torch::tensor({{3, 0, 0}}, torch::kInt32);
    auto speculative_sampler_output              = spec::SpeculativeSamplerOutput();
    speculative_sampler_output.accept_tokens_cpu = forced_accept_tokens;
    speculative_sampler_output.accept_tokens     = forced_accept_tokens.to(torch::kCUDA);
    // The rejection sampler initially accepts all rows; the logits processor
    // cap lowers the final accepted length to one. The target transaction must
    // observe the capped value, not this preliminary value.
    speculative_sampler_output.accept_len_cpu = torch::tensor({3}, torch::kInt32);
    speculative_sampler_output.accept_len     = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    auto status = components.executor->process(ScheduleOutput{{stream}});
    ASSERT_TRUE(status.ok());

    EXPECT_EQ(toVec<int32_t>(target_model->preparedAcceptLen()), (std::vector<int32_t>{1}));
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 1u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetFinalizeCount(), 1u);
    EXPECT_FALSE(target_model->speculativeTargetUndoRetained());
    EXPECT_EQ(target_model->prepareAttentionCount(), 1u);
    EXPECT_EQ(draft_model->prepareAttentionCount(), 1u);
    auto target_prepare = std::find(transaction_events->begin(), transaction_events->end(), "target.prepare_attention");
    auto target_forward = std::find(transaction_events->begin(), transaction_events->end(), "target.forward");
    auto prepare_commit =
        std::find(transaction_events->begin(), transaction_events->end(), "target.prepare_target_commit");
    auto finish_commit   = std::find(transaction_events->begin(), transaction_events->end(), "target.commit_target");
    auto finalize_commit = std::find(transaction_events->begin(), transaction_events->end(), "target.finalize_target");
    ASSERT_NE(target_prepare, transaction_events->end());
    ASSERT_NE(target_forward, transaction_events->end());
    ASSERT_NE(prepare_commit, transaction_events->end());
    ASSERT_NE(finish_commit, transaction_events->end());
    ASSERT_NE(finalize_commit, transaction_events->end());
    EXPECT_LT(target_prepare, target_forward);
    EXPECT_LT(target_forward, prepare_commit);
    EXPECT_LT(prepare_commit, finish_commit);
    EXPECT_LT(finish_commit, finalize_commit);
    auto draft_prepare = std::find(finalize_commit + 1, transaction_events->end(), "draft.prepare_attention");
    auto draft_forward = std::find(finalize_commit + 1, transaction_events->end(), "draft.forward");
    ASSERT_NE(draft_prepare, transaction_events->end());
    ASSERT_NE(draft_forward, transaction_events->end());
    EXPECT_LT(draft_prepare, draft_forward);

    checkOutput(stream, {0, 1, 2, 1}, {1, 2}, {0.0, 0.0, 1.0, 0.0}, {0.21, 0.22});
}

TEST_F(MtpExecutorTest, testDSparkPrefillUsesCommitOnlyExecutorPolicy) {
    MtpExecutorTestConfig test_config;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.gen_num_per_cycle    = 3;
    test_config.vocab_size_override  = test_config.vocab_size;
    test_config.dspark_mask_token_id = 0;
    test_config.role_type            = RoleType::PREFILL;

    auto components = createMtpExecutorComponents(test_config);

    EXPECT_TRUE(components.executor->dspark_prefill_commit_only_);
    EXPECT_FALSE(components.executor->draft_model_);
    EXPECT_FALSE(components.executor->dspark_markov_w1_.defined());
    EXPECT_FALSE(components.executor->dspark_markov_w2_.defined());
}

TEST_F(MtpExecutorTest, testDSparkDecodeKeepsProposalAndMarkovContract) {
    MtpExecutorTestConfig test_config;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.gen_num_per_cycle    = 3;
    test_config.vocab_size_override  = test_config.vocab_size;
    test_config.dspark_mask_token_id = 0;
    test_config.role_type            = RoleType::DECODE;

    auto components = createMtpExecutorComponents(test_config);

    EXPECT_FALSE(components.executor->dspark_prefill_commit_only_);
    EXPECT_TRUE(components.executor->dspark_markov_w1_.defined());
    EXPECT_TRUE(components.executor->dspark_markov_w2_.defined());
}

TEST_F(MtpExecutorTest, testDSparkGammaThreeSpecLogitsVerifyRunsOnAsyncWorker) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;

    auto components = createMtpExecutorComponents(test_config);

    GenerateStreamPtr stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    stream->generateConfig()->do_sample   = false;
    stream->generateConfig()->temperature = 0.0f;
    auto sp_buffer                        = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_buffer->propose_step               = gamma;
    sp_buffer->tokens                     = torch::empty({1, 1}, torch::kInt32);
    stream->setSPOutputBuffer(sp_buffer);

    /** PD handoff contains no draft state; decode builds proposals from the committed target anchor. */
    StreamSpecUpdateInfo spec_update_info{
        torch::tensor({{2}}, torch::kInt32), 1, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    stream->specUpdate(spec_update_info);
    EXPECT_TRUE(stream->getProposeToken().empty());
    EXPECT_FALSE(stream->getProposeTokensGpu().defined());

    auto processor = std::make_shared<RejectDraftTokenSpecProcessor>(3, stream->outputTokenLen());
    stream->sampling_state_.logits_processors.push_back(processor);
    const auto main_thread_id = std::this_thread::get_id();

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({2, 2, 1, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({gamma + 1}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::arange(0, gamma + 1, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits =
        torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.3f, 0.2f, 0.2f, 0.6f, 0.1f, 0.1f, 0.7f, 0.1f, 0.1f, 0.1f})
            .reshape({gamma + 1, vocab_size})
            .to(torch::kCUDA);
    auto target_aux_features =
        torch::arange(0, 2 * (gamma + 1), torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
            .reshape({gamma + 1, 2});
    target_output.all_hidden_states = target_aux_features;
    components.fake_target_model->setMtpTargetHiddenStates(target_aux_features);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    auto sampler_input         = SamplerInputs{target_output.logits.clone()};
    sampler_input.logits[0][3] = BaseLogitsProcessor::neg_inf;
    SamplerOutput target_sampler_output{torch::tensor({1, 2, 1, 0}, torch::kInt32).reshape({gamma + 1, 1})};
    target_sampler_output.all_probs = torch::eye(vocab_size, torch::kFloat32).to(torch::kCUDA);

    speculative::SpeculativeSamplerOutput speculative_sampler_output;
    speculative_sampler_output.accept_tokens_cpu = torch::tensor({{1, 0, 0, 0}}, torch::kInt32);
    speculative_sampler_output.accept_tokens     = speculative_sampler_output.accept_tokens_cpu.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({1}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    // Round-head propose call: fixed-width block anchored on the stream's
    // current last token at its own position (committed_end = seq_len - 1),
    // no feature input.
    GptModelInputs draft_input;
    draft_input.combo_tokens      = torch::tensor({2, 0, 0}, torch::kInt32);
    draft_input.input_lengths     = torch::tensor({gamma}, torch::kInt32);
    draft_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    draft_input.lm_output_indexes = torch::arange(0, gamma, torch::kInt32);

    /** COMMIT writes dense verify features; the next prefix excludes rejected rows. */
    GptModelInputs commit_input;
    commit_input.combo_tokens       = target_input.combo_tokens;
    commit_input.input_lengths      = torch::tensor({gamma + 1}, torch::kInt32);
    commit_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    commit_input.lm_output_indexes  = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    commit_input.last_hidden_states = target_aux_features;

    GptModelOutputs commit_output;
    commit_output.hidden_states = torch::zeros({gamma + 1, 2}, torch::kFloat32).to(torch::kCUDA);

    GptModelOutputs draft_output;
    draft_output.logits = torch::tensor({0.1f, 0.2f, 0.9f, 0.3f, 0.1f, 0.8f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, 0.9f})
                              .reshape({gamma, vocab_size})
                              .to(torch::kCUDA);
    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});
    components.fake_draft_model->expectTargetVerify(true);
    components.fake_draft_prefill_model->setInputs({commit_input});
    components.fake_draft_prefill_model->setOutputs({commit_output});
    components.fake_draft_prefill_model->expectTargetVerify(true);

    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({target_sampler_output});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler),
                    std::move(components.fake_draft_prefill_model));

    auto status = components.executor->process(ScheduleOutput{{stream}});
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NE(std::thread::id(), processor->invocationThreadId());
    EXPECT_NE(main_thread_id, processor->invocationThreadId());
    EXPECT_EQ((std::vector<int32_t>{2, 1, 3}), processor->observedDraftTokens());
}

TEST_F(MtpExecutorTest, testDSparkDraftUsesFlashInferSamplingAndReturnsExactQ) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    stream->generateConfig()->do_sample   = true;
    stream->generateConfig()->top_k       = 0;
    stream->generateConfig()->top_p       = 1.0f;
    stream->generateConfig()->temperature = 0.5f;
    StreamGroups stream_groups({stream});

    auto base_logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.3f, 0.2f, 0.7f, 0.1f, 0.0f, -0.1f})
                           .reshape({1, gamma, vocab_size})
                           .to(torch::kCUDA);
    auto anchors       = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto padded_logits = torch::cat(
        {base_logits.reshape({gamma, vocab_size}),
         torch::full({gamma, 3}, 1000.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))},
        1);

    auto output = components.executor->sampleDSparkDraft(stream_groups, padded_logits, anchors);

    ASSERT_TRUE(output.token_ids.is_cuda());
    ASSERT_EQ((std::vector<int64_t>{1, gamma}), output.token_ids.sizes().vec());
    ASSERT_TRUE(output.all_probs.is_cuda());
    ASSERT_EQ((std::vector<int64_t>{1, gamma, vocab_size}), output.all_probs.sizes().vec());
    EXPECT_FALSE(output.token_ids_are_point_mass);
    EXPECT_TRUE(torch::allclose(output.all_probs, torch::softmax(base_logits / 0.5f, -1), 1e-5, 1e-6));
    auto sampled_q = output.all_probs.gather(2, output.token_ids.to(torch::kLong).unsqueeze(-1));
    EXPECT_TRUE(sampled_q.gt(0).all().item<bool>());
}

TEST_F(MtpExecutorTest, testDSparkDraftZeroTemperatureUsesClampedFullSoftmaxQ) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    stream->generateConfig()->do_sample   = true;
    stream->generateConfig()->top_k       = 0;
    stream->generateConfig()->top_p       = 0.8f;
    stream->generateConfig()->temperature = 0.0f;
    StreamGroups stream_groups({stream});

    auto base_logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.4f, 0.3f})
                           .reshape({1, gamma, vocab_size})
                           .to(torch::kCUDA);
    auto anchors = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto output =
        components.executor->sampleDSparkDraft(stream_groups, base_logits.reshape({gamma, vocab_size}), anchors);

    ASSERT_FALSE(output.token_ids_are_point_mass);
    ASSERT_EQ((std::vector<int64_t>{1, gamma, vocab_size}), output.all_probs.sizes().vec());
    auto expected_q = torch::softmax(base_logits / 1.0e-6f, -1);
    EXPECT_TRUE(torch::allclose(output.all_probs, expected_q));
    EXPECT_TRUE(torch::isfinite(output.all_probs).all().item<bool>());
    EXPECT_TRUE(torch::allclose(output.all_probs.sum(-1).cpu(), torch::ones({1, gamma})));
    EXPECT_TRUE(torch::equal(output.token_ids.cpu(), expected_q.argmax(-1).to(torch::kInt32).cpu()));
}

TEST_F(MtpExecutorTest, testDSparkDraftGreedyRequestsCollapseQToArgmaxOneHot) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto base_logits = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.4f, 0.3f})
                           .reshape({1, gamma, vocab_size})
                           .to(torch::kCUDA);
    auto anchors    = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto expected_q = torch::softmax(base_logits / 1.0e-6f, -1);

    // do_sample=false is greedy even with a hot request temperature: the target
    // takes its argmax and acceptance requires an exact token match.
    {
        auto stream = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 1});
        stream->generateConfig()->do_sample   = false;
        stream->generateConfig()->temperature = 1.0f;
        StreamGroups stream_groups({stream});

        auto output =
            components.executor->sampleDSparkDraft(stream_groups, base_logits.reshape({gamma, vocab_size}), anchors);
        EXPECT_TRUE(torch::allclose(output.all_probs, expected_q));
        EXPECT_TRUE(torch::equal(output.token_ids.cpu(), expected_q.argmax(-1).to(torch::kInt32).cpu()));
    }

    // top_k=1 is greedy as well regardless of do_sample/temperature.
    {
        auto stream = createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 1});
        stream->generateConfig()->do_sample   = true;
        stream->generateConfig()->top_k       = 1;
        stream->generateConfig()->temperature = 0.7f;
        StreamGroups stream_groups({stream});

        auto output =
            components.executor->sampleDSparkDraft(stream_groups, base_logits.reshape({gamma, vocab_size}), anchors);
        EXPECT_TRUE(torch::allclose(output.all_probs, expected_q));
        EXPECT_TRUE(torch::equal(output.token_ids.cpu(), expected_q.argmax(-1).to(torch::kInt32).cpu()));
    }
}

TEST_F(MtpExecutorTest, testDSparkDraftUsesDenseSequentialMarkovDistribution) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 4;
    constexpr float   top_p      = 0.7f;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    stream->generateConfig()->do_sample   = true;
    stream->generateConfig()->top_k       = 0;
    stream->generateConfig()->top_p       = top_p;
    stream->generateConfig()->temperature = 1.0f;
    StreamGroups stream_groups({stream});

    auto cuda_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto markov_w1 = torch::tensor({0.0f, 0.4f, 0.8f, 1.2f}, torch::kFloat32).reshape({vocab_size, 1}).to(torch::kCUDA);
    auto markov_w2 =
        torch::tensor({-0.5f, -0.1f, 0.3f, 0.7f}, torch::kFloat32).reshape({vocab_size, 1}).to(torch::kCUDA);
    components.executor->dspark_markov_w1_ = markov_w1;
    components.executor->dspark_markov_w2_ = markov_w2;

    auto base_logits = torch::tensor({-10.0f, -10.0f, -10.0f, 10.0f, 0.1f, 0.4f, 0.3f, 0.2f, 0.2f, 0.1f, 0.4f, 0.3f})
                           .reshape({1, gamma, vocab_size})
                           .to(cuda_options);
    auto anchors = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto output =
        components.executor->sampleDSparkDraft(stream_groups, base_logits.reshape({gamma, vocab_size}), anchors);

    ASSERT_EQ((std::vector<int64_t>{1, gamma, vocab_size}), output.all_probs.sizes().vec());
    ASSERT_EQ(output.token_ids[0][0].item<int32_t>(), 3);
    auto previous_tokens = anchors.to(torch::kLong);
    for (int64_t step = 0; step < gamma; ++step) {
        auto markov_bias = torch::mm(markov_w1.index_select(0, previous_tokens), markov_w2.transpose(0, 1));
        auto raw_probs   = torch::softmax(base_logits.select(1, step) + markov_bias, -1);
        if (step == 1) {
            auto anchor_bias =
                torch::mm(markov_w1.index_select(0, anchors.to(torch::kLong)), markov_w2.transpose(0, 1));
            EXPECT_FALSE(torch::allclose(raw_probs, torch::softmax(base_logits.select(1, step) + anchor_bias, -1)));
        }
        auto actual_q = output.all_probs.select(1, step);
        EXPECT_TRUE(torch::allclose(actual_q, raw_probs, 1e-5, 1e-6));
        EXPECT_EQ(actual_q.gt(0).sum().item<int64_t>(), vocab_size);
        auto sampled_token = output.token_ids.select(1, step).to(torch::kLong);
        EXPECT_TRUE(actual_q.gather(1, sampled_token.unsqueeze(1)).gt(0).all().item<bool>());
        previous_tokens = std::move(sampled_token);
    }
}

TEST_F(MtpExecutorTest, testDSparkReducedVocabFeedsMappedTargetTokenIntoMarkovChain) {
    constexpr int32_t gamma             = 2;
    constexpr int32_t draft_vocab_size  = 3;
    constexpr int32_t target_vocab_size = 6;

    const auto cuda_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto       d2t        = torch::tensor({1, 3, 5}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    spec::SpeculativeSampler sampler(d2t, gamma);

    auto base_logits = torch::tensor({-100.0f, -100.0f, 100.0f, 0.0f, 0.0f, 0.0f}, torch::kFloat32)
                           .reshape({gamma, draft_vocab_size})
                           .to(torch::kCUDA);
    auto anchors     = torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto temperature = torch::ones({1}, cuda_float);
    auto markov_w1   = torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, torch::kFloat32)
                         .reshape({target_vocab_size, 1})
                         .to(torch::kCUDA);
    auto markov_w2 =
        torch::tensor({-100.0f, 0.0f, 100.0f}, torch::kFloat32).reshape({draft_vocab_size, 1}).to(torch::kCUDA);

    auto output = sampler.sampleDSparkDraft(base_logits, anchors, temperature, markov_w1, markov_w2, draft_vocab_size);

    EXPECT_EQ((std::vector<int32_t>{5, 5}), toVec<int32_t>(output.token_ids.cpu()));
    auto expected_second_q = torch::softmax(torch::tensor({-100.0f, 0.0f, 100.0f}), -1);
    EXPECT_TRUE(torch::allclose(output.all_probs[0][1].cpu(), expected_second_q));
}

TEST_F(MtpExecutorTest, testDecodeOneStepSpecLogitsCapReplacesInvalidDraftWithTargetToken) {
    size_t propose_step = 1;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = vocab_size;
    auto components                 = createMtpExecutorComponents(test_config);

    auto                 stream_new_tokens        = torch::tensor({{2}}, torch::kInt32);
    auto                 stream_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto                 stream_draft_token_probs = torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}});
    StreamSpecUpdateInfo spec_update_info{
        stream_new_tokens, 1, torch::tensor({3}, torch::kInt32), stream_hidden_states, stream_draft_token_probs};

    GenerateStreamPtr stream = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1}, spec_update_info);
    stream->sampling_state_.logits_processors.push_back(
        std::make_shared<RejectDraftTokenSpecProcessor>(3, stream->outputTokenLen()));

    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);
    target_output.logits =
        torch::tensor({0.1f, 0.9f, 0.2f, 0.3f, 0.7f, 0.2f, 0.1f, 0.0f}).reshape({2, 4}).to(torch::kCUDA);
    target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f}).reshape({2, 2});

    auto next_draft_input               = GptModelInputs{};
    auto next_draft_output              = GptModelOutputs{};
    next_draft_input.combo_tokens       = torch::tensor({1}, torch::kInt32);
    next_draft_input.input_lengths      = torch::tensor({1}, torch::kInt32);
    next_draft_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    next_draft_input.last_hidden_states = target_output.all_hidden_states.narrow(0, 0, 1);
    next_draft_output.logits            = torch::tensor({0.2f, 0.1f, 0.8f, 0.0f}).reshape({1, 4});
    next_draft_output.all_hidden_states = torch::tensor({0.21f, 0.22f}).reshape({1, 2});

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});
    components.fake_draft_model->setInputs({next_draft_input});
    components.fake_draft_model->setOutputs({next_draft_output});

    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4}), torch::tensor({2}, torch::kInt32).reshape({1, 1})};
    components.fake_fast_topk_sampler->setInputs({next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({next_draft_sampler_output});

    auto sampler_input              = SamplerInputs{target_output.logits.clone()};
    sampler_input.logits[0][3]      = BaseLogitsProcessor::neg_inf;
    auto target_sampler_output      = SamplerOutput{torch::tensor({1, 2}, torch::kInt32).reshape({2, 1})};
    target_sampler_output.all_probs = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({2, 4});
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({target_sampler_output});

    auto forced_accept_tokens                    = torch::tensor({{3, 2}}, torch::kInt32);
    auto speculative_sampler_output              = spec::SpeculativeSamplerOutput();
    speculative_sampler_output.accept_tokens_cpu = forced_accept_tokens;
    speculative_sampler_output.accept_tokens     = forced_accept_tokens.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({2}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    auto status = components.executor->process(ScheduleOutput{{stream}});
    ASSERT_TRUE(status.ok());

    checkOutput(stream, {0, 1, 2, 1}, {1, 2}, {0.0, 0.0, 1.0, 0.0}, {});
}

TEST_F(MtpExecutorTest, testDraftPrefillReadsHiddenFromActualCommitModel) {
    auto           components   = createMtpExecutorComponents(MtpExecutorTestConfig{});
    auto*          commit_model = components.fake_target_model.get();
    auto*          draft_model  = components.fake_draft_model.get();
    GptModelInputs input;
    input.combo_tokens      = torch::tensor({11, 12}, torch::kInt32);
    input.input_lengths     = torch::tensor({2}, torch::kInt32);
    input.prefix_lengths    = torch::tensor({3}, torch::kInt32);
    input.sequence_lengths  = torch::empty({0}, torch::kInt32);
    input.lm_output_indexes = torch::tensor({1}, torch::kInt32);
    GptModelOutputs output;
    output.all_hidden_states = torch::zeros({2, 2});
    auto commit_hidden       = torch::arange(16, torch::kFloat32).reshape({4, 4});
    commit_model->setInputs({input});
    commit_model->setOutputs({output});
    commit_model->setMtpTargetHiddenStates(commit_hidden);
    draft_model->setMtpTargetHiddenStates(torch::full({4, 4}, -1.0f));
    components.executor->sp_prefill_draft_model_ = std::move(components.fake_target_model);
    components.executor->draft_model_            = std::move(components.fake_draft_model);

    auto resolved_output = components.executor->runDraftPrefillForward(input);
    EXPECT_TRUE(torch::equal(resolved_output.all_hidden_states, commit_hidden.narrow(0, 0, 2)));
    EXPECT_EQ(resolved_output.all_hidden_states.data_ptr(), commit_hidden.data_ptr());
    EXPECT_EQ(commit_model->mtp_hidden_requests, (std::vector<int64_t>{2}));
    EXPECT_EQ(commit_model->forwardCount(), 1);
    EXPECT_TRUE(draft_model->mtp_hidden_requests.empty());
    EXPECT_EQ(draft_model->forwardCount(), 0);
}

TEST_F(MtpExecutorTest, testMultiBatchDecode) {
    runMultiBatchDecode(false);
}

TEST_F(MtpExecutorTest, testMultiBatchDecodeCompactsMtpHiddenBuffer) {
    runMultiBatchDecode(true);
}

void MtpExecutorTest::runMultiBatchDecode(bool use_mtp_hidden_buffer) {
    // test multi batch decode not accept & accept all
    // input s1:[0, 1, 2, 3] + [2] s2:[3, 2, 1] + [3]
    // darft s1:[2]+[1,2,3] s2:[3]+[0,2,2]
    // verify [3, 2, 0, 0, 0], [3, 0, 2, 2, 1]
    // accept [3], [3, 0, 2, 2, 1]
    // next draft [1], [2]
    size_t propose_step = 4;
    size_t vocab_size   = 4;
    size_t batch_size   = 2;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size          = vocab_size;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = vocab_size;
    auto components                 = createMtpExecutorComponents(test_config);

    // Create context stream
    auto stream1_new_tokens        = torch::tensor({{3}}, torch::kInt32);
    auto stream1_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto stream1_draft_token_probs = torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}});

    auto stream2_new_tokens        = torch::tensor({{1}}, torch::kInt32);
    auto stream2_hidden_states     = torch::tensor({{2.1f, 2.12f}});
    auto stream2_draft_token_probs = torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}});

    StreamSpecUpdateInfo spec_update_info1{
        stream1_new_tokens, 1, torch::tensor({2}, torch::kInt32), stream1_hidden_states, stream1_draft_token_probs};
    StreamSpecUpdateInfo spec_update_info2{
        stream2_new_tokens, 1, torch::tensor({3}, torch::kInt32), stream2_hidden_states, stream2_draft_token_probs};

    GenerateStreamPtr stream1 = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1, 2}, spec_update_info1);

    GenerateStreamPtr stream2 = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {3, 2}, spec_update_info2);

    // set fake model outputs
    // set 3 step draft model outputs
    // darft s1:[2]+[1,2,3] s2:[3]+[0,2,2]
    auto draft_input_1  = GptModelInputs{};
    auto draft_input_2  = GptModelInputs{};
    auto draft_input_3  = GptModelInputs{};
    auto draft_output_1 = createRandomGptModelOutputs(2, 4, 2);
    auto draft_output_2 = createRandomGptModelOutputs(2, 4, 2);
    auto draft_output_3 = createRandomGptModelOutputs(2, 4, 2);

    draft_input_1.combo_tokens       = torch::tensor({2, 3}, torch::kInt32);
    draft_input_1.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    draft_input_1.sequence_lengths   = torch::tensor({3, 2}, torch::kInt32);
    draft_input_1.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input_1.last_hidden_states = torch::tensor({0.03f, 0.04f, 2.1f, 2.12f}).reshape({2, 2});

    draft_input_2.combo_tokens       = torch::tensor({1, 0}, torch::kInt32);
    draft_input_2.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    draft_input_2.sequence_lengths   = torch::tensor({4, 3}, torch::kInt32);
    draft_input_2.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input_2.last_hidden_states = draft_output_1.all_hidden_states;

    draft_input_3.combo_tokens       = torch::tensor({2, 2}, torch::kInt32);
    draft_input_3.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    draft_input_3.sequence_lengths   = torch::tensor({5, 4}, torch::kInt32);
    draft_input_3.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    draft_input_3.last_hidden_states = draft_output_2.all_hidden_states;

    // accept [3], [3, 0, 2, 2, 1]
    auto next_draft_input  = GptModelInputs{};
    auto next_draft_output = GptModelOutputs{};
    next_draft_output.logits =
        torch::tensor({1.9f, 1.10f, 1.11f, 1.12f, 2.9f, 2.10f, 2.11f, 2.12f}).reshape({(int64_t)batch_size, 4});
    next_draft_output.all_hidden_states =
        torch::tensor({0.1f, 0.11f, 1.1f, 1.11f, 1.2f, 1.22f, 1.3f, 1.33f, 1.4f, 1.44f, 1.5f, 1.55f}).reshape({6, 2});

    next_draft_input.combo_tokens      = torch::tensor({3, 3, 0, 2, 2, 1}, torch::kInt32);
    next_draft_input.input_lengths     = torch::tensor({1, 5}, torch::kInt32);
    next_draft_input.prefix_lengths    = torch::tensor({3, 2}, torch::kInt32);
    next_draft_input.lm_output_indexes = torch::tensor({0, 5}, torch::kInt32);

    // set target model
    // verify [3, 2, 0, 0, 0], [3, 0, 2, 2, 1]
    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({3, 2, 1, 2, 3, 1, 3, 0, 2, 2}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({5, 5}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({3, 2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, torch::kInt32);

    target_output.logits =
        torch::tensor({0.1f,  0.2f,  0.3f,  0.4f,  1.1f,  1.2f,  1.3f,  1.4f,  2.1f,  2.2f,  2.3f,  2.4f,  3.1f,  3.2f,
                       3.3f,  3.4f,  4.1f,  4.2f,  4.3f,  4.4f,  -0.1f, -0.2f, -0.3f, -0.4f, -1.1f, -1.2f, -1.3f, -1.4f,
                       -2.1f, -2.2f, -2.3f, -2.4f, -3.1f, -3.2f, -3.3f, -3.4f, -4.1f, -4.2f, -4.3f, -4.4f})
            .reshape({(int64_t)(batch_size * (propose_step + 1)), 4});
    target_output.all_hidden_states =
        torch::tensor({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f,
                       0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f})
            .reshape({(int64_t)(batch_size * (propose_step + 1)), 2});

    auto target_hidden = target_output.all_hidden_states;
    if (use_mtp_hidden_buffer) {
        /** Extra capacity and distinct values expose both fallback misuse and a post-COMPACT prefix overwrite. */
        auto mtp_hidden_buffer = torch::arange(24, torch::kFloat32).reshape({12, 2}) + 100;
        components.fake_target_model->setMtpTargetHiddenStates(mtp_hidden_buffer);
        target_hidden = mtp_hidden_buffer.narrow(0, 0, 10);
    }
    next_draft_input.last_hidden_states = torch::cat({target_hidden.narrow(0, 0, 1), target_hidden.narrow(0, 5, 5)});

    components.fake_draft_model->setInputs({draft_input_1, draft_input_2, draft_input_3, next_draft_input});
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3, next_draft_output});

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    // set draft sampler outputs
    // darft s1:[2]+[1,2,3] s2:[3]+[0,2,2]
    // next draft [1], [2]
    auto draft_sampler_output_1    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_2    = spec::FastTopKSamplerOutput{};
    auto draft_sampler_output_3    = spec::FastTopKSamplerOutput{};
    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{};

    draft_sampler_output_1.token_ids = torch::tensor({1, 0}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_1.all_probs =
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}).reshape({(int64_t)batch_size, 4});
    draft_sampler_output_2.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_2.all_probs =
        torch::tensor({0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}).reshape({(int64_t)batch_size, 4});
    draft_sampler_output_3.token_ids = torch::tensor({3, 2}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_3.all_probs =
        torch::tensor({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}).reshape({(int64_t)batch_size, 4});
    next_draft_sampler_output.token_ids = torch::tensor({1, 2}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    next_draft_sampler_output.all_probs =
        torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}).reshape({(int64_t)batch_size, 4});

    components.fake_fast_topk_sampler->setInputs(
        {draft_output_1.logits, draft_output_2.logits, draft_output_3.logits, next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs(
        {draft_sampler_output_1, draft_sampler_output_2, draft_sampler_output_3, next_draft_sampler_output});

    // set fake sampler outputs
    auto target_sample_all_probs_data = createRandomVector<float>(batch_size * (propose_step + 1) * vocab_size, 1);
    auto sampler_input                = SamplerInputs{target_output.logits};
    auto sampler_output =
        SamplerOutput{torch::tensor({3, 2, 0, 0, 0, 3, 0, 2, 2, 1}, torch::kInt32).reshape({(int64_t)batch_size, 5})};
    sampler_output.all_probs = torch::tensor(target_sample_all_probs_data)
                                   .reshape({(int64_t)batch_size, (int64_t)(propose_step + 1), (int64_t)vocab_size});
    components.fake_sampler->setInputs({sampler_input});
    components.fake_sampler->setOutputs({sampler_output});

    // set fake speculative sampler outputs
    auto accept_tokens                           = torch::tensor({{3, 0, 0, 0, 0}, {3, 0, 2, 2, 1}}, torch::kInt32);
    auto speculative_sampler_output              = spec::SpeculativeSamplerOutput();
    speculative_sampler_output.accept_tokens_cpu = accept_tokens;
    speculative_sampler_output.accept_tokens     = accept_tokens.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({1, 5}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    auto draft_spec_sample_input                 = SamplerOutput{};
    auto target_spec_sample_input                = SamplerOutput{};

    vector<vector<float>> draft_all_probs_list;
    draft_all_probs_list.push_back({0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    draft_all_probs_list.push_back(toVec<float>(draft_output_1.logits));
    draft_all_probs_list.push_back(toVec<float>(draft_output_2.logits));
    draft_all_probs_list.push_back(toVec<float>(draft_output_3.logits));
    draft_spec_sample_input.token_ids = torch::tensor({2, 1, 2, 3, 3, 0, 2, 2}, torch::kInt32).reshape({2, 4});
    draft_spec_sample_input.all_probs = torch::tensor(catVectors(draft_all_probs_list)).reshape({4, 8});
    target_spec_sample_input.all_probs =
        torch::tensor(target_sample_all_probs_data)
            .reshape({(int64_t)batch_size, (int64_t)(propose_step + 1), (int64_t)vocab_size});

    components.fake_speculative_sampler->setInputs({draft_spec_sample_input, target_spec_sample_input});
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    auto* target_model = components.fake_target_model.get();

    // Replace models with fake models
    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    // Verify executor was created successfully
    auto status = components.executor->process(ScheduleOutput{{stream1, stream2}});
    ASSERT_TRUE(status.ok());

    // check stream result
    checkOutput(stream1, {0, 1, 2, 3, 3}, {3, 1}, {0, 1, 0, 0}, {0.1, 0.11});
    checkOutput(stream2, {3, 2, 1, 3, 0, 2, 2, 1}, {1, 2}, {0.0, 1.0, 0.0, 0.0}, {1.5, 1.55});
    EXPECT_EQ(target_model->mtp_hidden_requests, (std::vector<int64_t>{10}));
}

TEST_F(MtpExecutorTest, testDraftModelDecodeExpandsTargetVerifyPositionIds) {
    size_t propose_step = 4;
    size_t batch_size   = 2;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle      = propose_step;
    test_config.vocab_size_override    = vocab_size;
    test_config.mm_position_ids_style  = MROPE;
    test_config.position_id_len_factor = 3;
    auto components                    = createMtpExecutorComponents(test_config);

    auto stream_model_config                                  = components.model_config;
    stream_model_config.mm_model_config.mm_position_ids_style = MROPE;
    stream_model_config.attn_config.rope_config.index_factor  = 3;
    auto stream1                                              = createContextStream(
        stream_model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3, 0, 1});
    auto stream2 = createContextStream(
        stream_model_config, components.runtime_config, components.resource_context, {1, 2, 3, 0, 1, 2, 3, 0});
    stream1->setContextPositionIds(torch::tensor({0, 0, 0}, torch::kInt32));
    stream2->setContextPositionIds(torch::tensor({0, 0, 0}, torch::kInt32));

    auto sp_output_buffer1    = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_output_buffer1->tokens = torch::tensor({10, 11}, torch::kInt32).reshape({1, 2});
    stream1->setSPOutputBuffer(sp_output_buffer1);

    auto sp_output_buffer2    = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_output_buffer2->tokens = torch::tensor({20, 21}, torch::kInt32).reshape({1, 2});
    stream2->setSPOutputBuffer(sp_output_buffer2);

    StreamGroups stream_groups({stream1, stream2});

    GptModelInputs model_input;
    model_input.combo_tokens       = torch::tensor({11, 21}, torch::kInt32);
    model_input.input_lengths      = torch::tensor({5, 6}, torch::kInt32);
    model_input.sequence_lengths   = torch::tensor({5, 7}, torch::kInt32);
    model_input.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
    model_input.last_hidden_states = torch::tensor({0.1f, 0.2f, 1.1f, 1.2f}, torch::kFloat32).reshape({2, 2});
    model_input.combo_position_ids = torch::tensor({5, 5, 5, 7, 7, 7}, torch::kInt32);

    auto makeDraftInput = [](std::vector<int> combo_tokens,
                             std::vector<int> sequence_lengths,
                             torch::Tensor    last_hidden_states,
                             std::vector<int> combo_position_ids = {}) {
        GptModelInputs input;
        input.combo_tokens       = torch::tensor(combo_tokens, torch::kInt32);
        input.input_lengths      = torch::tensor({5, 6}, torch::kInt32);
        input.sequence_lengths   = torch::tensor(sequence_lengths, torch::kInt32);
        input.lm_output_indexes  = torch::tensor({0, 1}, torch::kInt32);
        input.last_hidden_states = std::move(last_hidden_states);
        if (!combo_position_ids.empty()) {
            input.combo_position_ids = torch::tensor(combo_position_ids, torch::kInt32);
        }
        return input;
    };

    auto draft_output_1 = createRandomGptModelOutputs(batch_size, vocab_size, 2);
    auto draft_output_2 = createRandomGptModelOutputs(batch_size, vocab_size, 2);
    auto draft_output_3 = createRandomGptModelOutputs(batch_size, vocab_size, 2);

    components.fake_draft_model->setInputs({
        makeDraftInput({11, 21}, {5, 7}, model_input.last_hidden_states, {5, 5, 5, 7, 7, 7}),
        makeDraftInput({12, 22}, {6, 8}, draft_output_1.all_hidden_states, {6, 6, 6, 8, 8, 8}),
        makeDraftInput({13, 23}, {7, 9}, draft_output_2.all_hidden_states, {7, 7, 7, 9, 9, 9}),
    });
    components.fake_draft_model->setOutputs({draft_output_1, draft_output_2, draft_output_3});

    spec::FastTopKSamplerOutput draft_sampler_output_1;
    draft_sampler_output_1.token_ids = torch::tensor({12, 22}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_1.all_probs = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, torch::kFloat32)
                                           .reshape({(int64_t)batch_size, (int64_t)vocab_size});

    spec::FastTopKSamplerOutput draft_sampler_output_2;
    draft_sampler_output_2.token_ids = torch::tensor({13, 23}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_2.all_probs = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, torch::kFloat32)
                                           .reshape({(int64_t)batch_size, (int64_t)vocab_size});

    spec::FastTopKSamplerOutput draft_sampler_output_3;
    draft_sampler_output_3.token_ids = torch::tensor({14, 24}, torch::kInt32).reshape({(int64_t)batch_size, 1});
    draft_sampler_output_3.all_probs = torch::tensor({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f}, torch::kFloat32)
                                           .reshape({(int64_t)batch_size, (int64_t)vocab_size});

    components.fake_fast_topk_sampler->setInputs({draft_output_1.logits, draft_output_2.logits, draft_output_3.logits});
    components.fake_fast_topk_sampler->setOutputs(
        {draft_sampler_output_1, draft_sampler_output_2, draft_sampler_output_3});

    components.executor->setDraftModel(std::move(components.fake_draft_model));
    components.executor->setFastTopKSampler(std::move(components.fake_fast_topk_sampler));

    std::vector<torch::Tensor> draft_probs_list;
    torch::Tensor              draft_token_ids_t;
    int64_t                    model_forward_us = 0;
    components.executor->draftModelDecode(
        model_input, stream_groups, draft_probs_list, draft_token_ids_t, model_forward_us);

    EXPECT_EQ((std::vector<int>{10, 11, 12, 13, 14, 20, 21, 22, 23, 24}), toVec<int>(model_input.combo_tokens));
    EXPECT_EQ((std::vector<int>{5, 5}), toVec<int>(model_input.input_lengths));
    // Without an explicit prefix, target verification resumes at the same
    // cache position as the first shifted draft write.
    EXPECT_EQ((std::vector<int>{5, 7}), toVec<int>(model_input.prefix_lengths));
    EXPECT_EQ(0, model_input.sequence_lengths.numel());
    EXPECT_EQ((std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), toVec<int>(model_input.lm_output_indexes));

    EXPECT_EQ((std::vector<int>{5, 5, 5, 6, 6, 6, 7, 7, 7, 8,  8,  8,  9,  9,  9,
                                7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11}),
              toVec<int>(model_input.combo_position_ids));

    // Multimodal decode retains a B-row mask; target verification consumes
    // B * (gamma + 1) generated text rows and must not reuse that mask.
    model_input.text_tokens_mask = torch::ones({(int64_t)batch_size}, torch::kInt32);
    components.fake_target_model->setInputs({model_input});
    components.fake_target_model->setOutputs(
        {createRandomGptModelOutputs(batch_size * (propose_step + 1), vocab_size, 2)});
    components.fake_target_model->forward_hook = [](GptModelInputs& inputs) {
        EXPECT_TRUE(inputs.is_target_verify);
        EXPECT_FALSE(inputs.text_tokens_mask.defined());
    };
    components.executor->setTargetModel(std::move(components.fake_target_model));
    components.executor->runTargetVerifyForward(model_input, stream_groups);
}

TEST_F(MtpExecutorTest, testDSparkDraftMasksInvalidTemperatureAndUsesProbabilitySampling) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 6;

    MtpExecutorTestConfig test_config;
    test_config.vocab_size           = vocab_size;
    test_config.gen_num_per_cycle    = gamma;
    test_config.vocab_size_override  = vocab_size;
    test_config.sp_type              = SP_TYPE_DSPARK;
    test_config.dspark_mask_token_id = 0;
    auto components                  = createMtpExecutorComponents(test_config);

    auto stochastic_stream =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    auto config         = stochastic_stream->generateConfig();
    config->do_sample   = true;
    config->top_k       = 0;
    config->temperature = std::numeric_limits<float>::quiet_NaN();
    StreamGroups stream_groups({stochastic_stream});

    auto base_logits = torch::tensor({6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}, torch::kFloat32).repeat({gamma, 1});
    base_logits      = base_logits.to(torch::kCUDA);
    auto anchors     = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto output = components.executor->sampleDSparkDraft(stream_groups, base_logits, anchors);

    EXPECT_FLOAT_EQ(config->temperature, 1.0f);
    ASSERT_FALSE(output.token_ids_are_point_mass);
    ASSERT_EQ((std::vector<int64_t>{1, gamma, vocab_size}), output.all_probs.sizes().vec());
    EXPECT_TRUE(output.token_ids.ge(0).all().item<bool>());
    EXPECT_TRUE(output.token_ids.lt(vocab_size).all().item<bool>());
    auto expected_q = torch::softmax(base_logits.reshape({1, gamma, vocab_size}), -1);
    EXPECT_TRUE(torch::allclose(output.all_probs, expected_q));

    config->temperature  = -1.0f;
    auto negative_output = components.executor->sampleDSparkDraft(stream_groups, base_logits, anchors);
    EXPECT_FLOAT_EQ(config->temperature, 1.0f);
    EXPECT_TRUE(torch::allclose(negative_output.all_probs, expected_q));
}

TEST_F(MtpExecutorTest, testDSparkFakeDecodeStartsWithoutProposalState) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 16;

    ModelConfig     model_config;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    model_config.max_seq_len = 64;
    model_config.vocab_size  = vocab_size;
    model_config.hidden_size = 8;
    model_config.data_type   = TYPE_FP16;

    auto stream =
        MtpExecutor::createMinFakeDecodeStream(gamma, model_config, runtime_config, resource_context, vocab_size, true);
    auto sp_buffer = stream->getSPOutputBuffer();
    ASSERT_NE(sp_buffer, nullptr);
    EXPECT_EQ((std::vector<int64_t>{1, 1}), sp_buffer->tokens.sizes().vec());
    EXPECT_FALSE(sp_buffer->all_probs.defined());
    EXPECT_FALSE(sp_buffer->hidden_states.defined());
    EXPECT_FALSE(stream->getProposeTokensGpu().defined());

    StreamSpecUpdateInfo update_info{
        torch::tensor({7}, torch::kInt32).reshape({1, 1}), 1, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    update_info.speculative_propose_step = 3;
    update_info.accepted_draft_tokens    = 2;
    stream->specUpdate(update_info);

    EXPECT_EQ((std::vector<int32_t>{7}), toVec<int32_t>(sp_buffer->tokens));
    EXPECT_TRUE(stream->getProposeToken().empty());
    EXPECT_EQ(stream->spIterCount(), 1);
    EXPECT_EQ((std::vector<int32_t>{1, 1, 0}), stream->speculativeAcceptedTokensPerPos());
}

TEST_F(MtpExecutorTest, testFakeDecodeUsesHyperConnectionInputHiddenSize) {
    constexpr int32_t gamma      = 3;
    constexpr int32_t vocab_size = 16;

    ModelConfig     model_config;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    model_config.max_seq_len = 64;
    model_config.vocab_size  = vocab_size;
    model_config.hidden_size = 8;
    model_config.hc_mult     = 4;
    model_config.data_type   = TYPE_FP16;

    auto stream = MtpExecutor::createMinFakeDecodeStream(
        gamma, model_config, runtime_config, resource_context, vocab_size, false);
    auto sp_buffer = stream->getSPOutputBuffer();
    ASSERT_NE(sp_buffer, nullptr);
    ASSERT_TRUE(sp_buffer->hidden_states.defined());
    EXPECT_EQ((std::vector<int64_t>{1, 32}), sp_buffer->hidden_states.sizes().vec());
}

TEST_F(MtpExecutorTest, MixedPreviousTargetTokensUseDeviceStateWithoutBookkeepingWait) {
    auto components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    auto steady =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {0, 1});
    auto fresh =
        createContextStream(components.model_config, components.runtime_config, components.resource_context, {1, 2});
    auto stale    = std::make_shared<SpeculativeExecutorStreamOutput>();
    stale->tokens = torch::tensor({-999}, torch::kInt32);
    steady->setSPOutputBuffer(stale);
    auto initial    = std::make_shared<SpeculativeExecutorStreamOutput>();
    initial->tokens = torch::tensor({20}, torch::kInt32);
    fresh->setSPOutputBuffer(initial);
    GenerateStream::MtpAsyncDeviceState state;
    state.accept_tokens_gpu = torch::tensor({10, 11, 12}, torch::kInt32).cuda();
    state.accept_len_gpu    = torch::tensor({2}, torch::kInt32).cuda();
    steady->setMtpAsyncDeviceState(state);
    steady->incPendingAsyncBookkeeping();
    bool          from_device = false;
    torch::Tensor result;
    try {
        result = components.executor->gatherPreviousTargetTokens(StreamGroups({steady, fresh}), 2, true, from_device);
    } catch (...) {
        steady->decPendingAsyncBookkeepingAndMaybeRelease();
        throw;
    }
    // No worker will finish until the call above returns, so an inserted join
    // would deadlock. The poisoned host row must never reach the output.
    EXPECT_TRUE(steady->hasPendingAsyncBookkeeping());
    steady->decPendingAsyncBookkeepingAndMaybeRelease();
    EXPECT_TRUE(from_device);
    EXPECT_EQ((std::vector<int>{11, 20}), toVec<int>(result));
    initial->tokens.fill_(21);
    result = components.executor->gatherPreviousTargetTokens(StreamGroups({fresh}), 1, true, from_device);
    EXPECT_FALSE(from_device);
    EXPECT_EQ((std::vector<int>{21}), toVec<int>(result));
}

TEST_F(MtpExecutorTest, HostTargetStatusDoesNotAllocateDeviceStatusAndPropagatesFailure) {
    auto  components = createMtpExecutorComponents(MtpExecutorTestConfig{});
    auto* model      = components.fake_target_model.get();
    model->enableSpeculativeTargetCommit();
    components.executor->setTargetModel(std::move(components.fake_target_model));
    components.executor->parallelism_config_.tp_size = 2;
    std::vector<bool> seen;
    model->host_status_reducer = [&seen](bool ok) -> std::optional<bool> {
        seen.push_back(ok);
        return false;
    };
    EXPECT_FALSE(components.executor->reduceSpeculativeTargetPrepareStatus(true));
    EXPECT_FALSE(components.executor->reduceSpeculativeTargetPrepareStatus(false));
    EXPECT_EQ(seen, (std::vector<bool>{true, false}));
    model->host_status_reducer = [](bool ok) -> std::optional<bool> { return ok; };
    EXPECT_TRUE(components.executor->reduceSpeculativeTargetPrepareStatus(true));
    model->host_status_reducer = [](bool) -> std::optional<bool> { throw std::runtime_error("poisoned host status"); };
    EXPECT_THROW(components.executor->reduceSpeculativeTargetPrepareStatus(true), std::runtime_error);
    model->host_status_reducer = {};
    EXPECT_THROW(components.executor->reduceSpeculativeTargetPrepareStatus(true), std::runtime_error);
    // TP1 remains local and never enters the transport, even when configured.
    components.executor->parallelism_config_.tp_size = 1;
    EXPECT_TRUE(components.executor->reduceSpeculativeTargetPrepareStatus(true));
    EXPECT_FALSE(components.executor->reduceSpeculativeTargetPrepareStatus(false));
}

TEST_F(MtpExecutorTest, testSpeculativeTargetCommitDefaultIsNoop) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    components.executor->setTargetModel(std::move(components.fake_target_model));
    auto accept_len = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto status = components.executor->prepareAndCommitSpeculativeTargetState(accept_len, 1);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(target_model->preparedAcceptLen().defined());
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 0u);
}

TEST_F(MtpExecutorTest, testLocalSamplingFailureRendezvousAbortsWithoutCommit) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    target_model->enableSpeculativeTargetCommit();
    components.executor->setTargetModel(std::move(components.fake_target_model));
    size_t reduction_count                                           = 0;
    components.executor->speculative_target_status_reducer_for_test_ = [&reduction_count](bool local_ok) {
        ++reduction_count;
        return local_ok;
    };

    auto status = components.executor->rendezvousSpeculativeTargetSamplingStatus(
        absl::InternalError("injected local sampling failure"));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(std::string(status.message()), "speculative target sampling/cap failed on at least one TP rank");
    EXPECT_EQ(reduction_count, 2u);  // sampling readiness, then rollback consensus
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 1u);
    EXPECT_EQ(target_model->speculativeTargetFinalizeCount(), 0u);
}

TEST_F(MtpExecutorTest, testRemoteSpecLogitsLaunchFailureRendezvousAbortsBeforeTargetForward) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    target_model->enableSpeculativeTargetCommit();
    components.executor->setTargetModel(std::move(components.fake_target_model));
    size_t reduction_count                                           = 0;
    components.executor->speculative_target_status_reducer_for_test_ = [&reduction_count](bool local_ok) {
        ++reduction_count;
        if (reduction_count == 1) {
            EXPECT_TRUE(local_ok);
            return false;  // Simulate rank 0 failing event/check/async launch.
        }
        return local_ok;
    };

    auto status = components.executor->rendezvousSpeculativeTargetPhaseStatus(absl::OkStatus(), "spec-logits launch");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(std::string(status.message()), "speculative target spec-logits launch failed on at least one TP rank");
    EXPECT_EQ(reduction_count, 2u);  // launch readiness, then rollback consensus
    EXPECT_EQ(target_model->forwardCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 1u);
    EXPECT_EQ(target_model->speculativeTargetFinalizeCount(), 0u);
}

TEST_F(MtpExecutorTest, testRemoteSamplingFailureRendezvousAbortsWithoutCommit) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    target_model->enableSpeculativeTargetCommit();
    components.executor->setTargetModel(std::move(components.fake_target_model));
    size_t reduction_count                                           = 0;
    components.executor->speculative_target_status_reducer_for_test_ = [&reduction_count](bool local_ok) {
        ++reduction_count;
        if (reduction_count == 1) {
            EXPECT_TRUE(local_ok);
            return false;  // Simulate another TP rank failing sampling.
        }
        return local_ok;
    };

    auto status = components.executor->rendezvousSpeculativeTargetSamplingStatus(absl::OkStatus());
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(std::string(status.message()), "speculative target sampling/cap failed on at least one TP rank");
    EXPECT_EQ(reduction_count, 2u);  // sampling readiness, then rollback consensus
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 1u);
    EXPECT_EQ(target_model->speculativeTargetFinalizeCount(), 0u);
}

TEST_F(MtpExecutorTest, testSamplingFailureWithoutHooksPreservesLocalStatus) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    components.executor->setTargetModel(std::move(components.fake_target_model));
    size_t reduction_count                                           = 0;
    components.executor->speculative_target_status_reducer_for_test_ = [&reduction_count](bool local_ok) {
        ++reduction_count;
        return local_ok;
    };
    const auto local_status = absl::InvalidArgumentError("legacy no-hook sampling failure");

    auto status = components.executor->rendezvousSpeculativeTargetSamplingStatus(local_status);
    EXPECT_EQ(status, local_status);
    EXPECT_EQ(reduction_count, 0u);
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 0u);
}

TEST_F(MtpExecutorTest, testSpeculativeTargetPrepareFailureAbortsBeforeCommit) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    target_model->enableSpeculativeTargetCommit("injected prepare failure");
    components.executor->setTargetModel(std::move(components.fake_target_model));
    auto accept_len = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto status = components.executor->prepareAndCommitSpeculativeTargetState(accept_len, 1);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.ToString().find("injected prepare failure"), std::string::npos);
    EXPECT_EQ(toVec<int32_t>(target_model->preparedAcceptLen()), (std::vector<int32_t>{2}));
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 1u);
}

TEST_F(MtpExecutorTest, testSpeculativeTargetPrepareExceptionBecomesStatusAndAborts) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    target_model->enableSpeculativeTargetCommit();
    target_model->setSpeculativeTargetPrepareThrows(true);
    components.executor->setTargetModel(std::move(components.fake_target_model));
    auto accept_len = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto status = components.executor->prepareAndCommitSpeculativeTargetState(accept_len, 1);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.ToString().find("injected prepare exception"), std::string::npos);
    EXPECT_EQ(toVec<int32_t>(target_model->preparedAcceptLen()), (std::vector<int32_t>{2}));
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 1u);
}

TEST_F(MtpExecutorTest, testSpeculativeTargetInvalidAcceptLenRendezvousBeforeReturn) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    target_model->enableSpeculativeTargetCommit();
    components.executor->setTargetModel(std::move(components.fake_target_model));
    size_t reduction_count                                           = 0;
    components.executor->speculative_target_status_reducer_for_test_ = [&reduction_count](bool local_ok) {
        ++reduction_count;
        return local_ok;
    };
    auto invalid_accept_len = torch::tensor({2}, torch::kInt32);

    auto status = components.executor->prepareAndCommitSpeculativeTargetState(invalid_accept_len, 1);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.ToString().find("accept_len validation failed"), std::string::npos);
    EXPECT_EQ(reduction_count, 2u);  // readiness failure, then rollback consensus
    EXPECT_FALSE(target_model->preparedAcceptLen().defined());
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 0u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 1u);
}

TEST_F(MtpExecutorTest, testSpeculativeTargetCommitExceptionRollsBack) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    target_model->enableSpeculativeTargetCommit();
    target_model->setSpeculativeTargetCommitThrows(true);
    components.executor->setTargetModel(std::move(components.fake_target_model));
    auto accept_len = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto status = components.executor->prepareAndCommitSpeculativeTargetState(accept_len, 1);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.ToString().find("injected commit exception"), std::string::npos);
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 1u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 1u);
    EXPECT_EQ(target_model->speculativeTargetFinalizeCount(), 0u);
    EXPECT_FALSE(target_model->speculativeTargetStateCommitted());
    EXPECT_FALSE(target_model->speculativeTargetUndoRetained());
}

TEST_F(MtpExecutorTest, testSpeculativeTargetPeerCommitFailureRollsBackLocalCommit) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 2;
    auto components               = createMtpExecutorComponents(test_config);

    auto* target_model = components.fake_target_model.get();
    target_model->enableSpeculativeTargetCommit();
    components.executor->setTargetModel(std::move(components.fake_target_model));
    size_t reduction_count                                           = 0;
    components.executor->speculative_target_status_reducer_for_test_ = [&reduction_count](bool local_ok) {
        ++reduction_count;
        if (reduction_count == 3) {
            return false;  // Simulate another TP rank failing tentative commit.
        }
        return local_ok;
    };
    auto accept_len = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto status = components.executor->prepareAndCommitSpeculativeTargetState(accept_len, 1);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.ToString().find("commit failed on at least one TP rank"), std::string::npos);
    EXPECT_EQ(reduction_count, 4u);  // readiness, prepare, commit, rollback
    EXPECT_EQ(target_model->speculativeTargetCommitCount(), 1u);
    EXPECT_EQ(target_model->speculativeTargetAbortCount(), 1u);
    EXPECT_EQ(target_model->speculativeTargetFinalizeCount(), 0u);
    EXPECT_FALSE(target_model->speculativeTargetStateCommitted());
    EXPECT_FALSE(target_model->speculativeTargetUndoRetained());
}

TEST_F(MtpExecutorTest, DispatchStateUsesAcceptedLengths) {
    /** Rejected rows must not advance state; derive sequence lengths and hidden indexes from accepted counts. */
    const int64_t batch_size = 8;
    auto          cuda_i32   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto          cuda_i64   = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    auto accept_len   = torch::tensor({3, 1, 5, 2, 4, 1, 3, 2}, cuda_i32);
    auto prev_seq_len = torch::tensor({100, 200, 50, 300, 150, 400, 75, 250}, cuda_i32);
    auto next_seq_len = torch::empty({batch_size}, cuda_i32);
    auto hidden_idx   = torch::empty({batch_size}, cuda_i64);

#if USING_CUDA
    invokeMtpDispatchStatePrepare(
        accept_len, prev_seq_len, next_seq_len, hidden_idx, batch_size, at::cuda::getCurrentCUDAStream().stream());
    cudaDeviceSynchronize();
#endif

    auto expected_next = (prev_seq_len + accept_len).cpu();
    auto actual_next   = next_seq_len.cpu();
    EXPECT_TRUE(torch::equal(actual_next, expected_next)) << "next_seq_len mismatch:\n"
                                                          << actual_next << "\nvs expected:\n"
                                                          << expected_next;

    auto expected_idx = (accept_len.to(torch::kInt64) - 1).cpu();
    auto actual_idx   = hidden_idx.cpu();
    EXPECT_TRUE(torch::equal(actual_idx, expected_idx)) << "hidden_idx mismatch:\n"
                                                        << actual_idx << "\nvs expected:\n"
                                                        << expected_idx;
}

/** An incompatible processor fails its request; healthy requests must continue decoding. */
class IncompatibleMtpProcessor: public BaseLogitsProcessor {
public:
    std::optional<ErrorInfo> process(const SamplerInputs&, size_t, size_t) override {
        return std::nullopt;
    }
    void                     updateMultiSeqStatus(const std::vector<int>&) override {}
    std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t) override {
        return std::nullopt;
    }

    bool isStateful() const override {
        return true;
    }

    MtpProcessorCapability mtpCapability() const override {
        return {MtpProcessorMode::UNSUPPORTED, "test processor has no spec-verify support"};
    }
};

TEST_F(MtpExecutorTest, testErroredSpecLogitsStreamDoesNotAbortExecutor) {
    size_t propose_step = 1;
    size_t vocab_size   = 4;

    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = propose_step;
    test_config.vocab_size_override = vocab_size;
    auto components                 = createMtpExecutorComponents(test_config);

    auto                 stream_new_tokens        = torch::tensor({{2}}, torch::kInt32);
    auto                 stream_hidden_states     = torch::tensor({{0.03f, 0.04f}});
    auto                 stream_draft_token_probs = torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}});
    StreamSpecUpdateInfo spec_update_info{
        stream_new_tokens, 1, torch::tensor({3}, torch::kInt32), stream_hidden_states, stream_draft_token_probs};

    GenerateStreamPtr stream = createDecodeStream(
        components.model_config, components.runtime_config, components.resource_context, {0, 1}, spec_update_info);
    stream->sampling_state_.logits_processors.push_back(std::make_shared<IncompatibleMtpProcessor>());
    stream->reportError(ErrorCode::INVALID_PARAMS, "grammar accept_token error: parser rejected token");

    auto target_input              = GptModelInputs{};
    auto target_output             = GptModelOutputs{};
    target_input.combo_tokens      = torch::tensor({2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);
    target_output.logits =
        torch::tensor({0.1f, 0.9f, 0.2f, 0.3f, 0.7f, 0.2f, 0.1f, 0.0f}).reshape({2, 4}).to(torch::kCUDA);
    target_output.all_hidden_states = torch::tensor({0.01f, 0.02f, 0.03f, 0.04f}).reshape({2, 2});

    auto next_draft_input               = GptModelInputs{};
    auto next_draft_output              = GptModelOutputs{};
    next_draft_input.combo_tokens       = torch::tensor({3, 2}, torch::kInt32);
    next_draft_input.input_lengths      = torch::tensor({2}, torch::kInt32);
    next_draft_input.prefix_lengths     = torch::tensor({2}, torch::kInt32);
    next_draft_input.lm_output_indexes  = torch::tensor({1}, torch::kInt32);
    next_draft_input.last_hidden_states = target_output.all_hidden_states;
    next_draft_output.logits            = torch::tensor({0.2f, 0.1f, 0.8f, 0.0f}).reshape({1, 4});
    next_draft_output.all_hidden_states = torch::tensor({0.21f, 0.22f, 0.23f, 0.24f}).reshape({2, 2});

    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});
    components.fake_draft_model->setInputs({next_draft_input});
    components.fake_draft_model->setOutputs({next_draft_output});

    auto next_draft_sampler_output = spec::FastTopKSamplerOutput{
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4}), torch::tensor({2}, torch::kInt32).reshape({1, 1})};
    components.fake_fast_topk_sampler->setInputs({next_draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({next_draft_sampler_output});

    auto target_sampler_output      = SamplerOutput{torch::tensor({1, 2}, torch::kInt32).reshape({2, 1})};
    target_sampler_output.all_probs = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({2, 4});
    components.fake_sampler->setInputs({SamplerInputs{target_output.logits.clone()}});
    components.fake_sampler->setOutputs({target_sampler_output});

    auto forced_accept_tokens                    = torch::tensor({{3, 2}}, torch::kInt32);
    auto speculative_sampler_output              = spec::SpeculativeSamplerOutput();
    speculative_sampler_output.accept_tokens_cpu = forced_accept_tokens;
    speculative_sampler_output.accept_tokens     = forced_accept_tokens.to(torch::kCUDA);
    speculative_sampler_output.accept_len_cpu    = torch::tensor({2}, torch::kInt32);
    speculative_sampler_output.accept_len        = speculative_sampler_output.accept_len_cpu.to(torch::kCUDA);
    components.fake_speculative_sampler->setOutputs({speculative_sampler_output});

    setupFakeModels(components.executor.get(),
                    std::move(components.fake_target_model),
                    std::move(components.fake_draft_model),
                    std::move(components.fake_fast_topk_sampler),
                    std::move(components.fake_speculative_sampler),
                    std::move(components.fake_sampler));

    auto status = components.executor->process(ScheduleOutput{{stream}});
    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(stream->hasError());
}

}  // namespace rtp_llm
