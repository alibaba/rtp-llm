/**
 * Single-stage PPExecutor tests, organized by ownership:
 *   Common: construction, warmup, communication waits and profiler boundaries.
 *   First: stream preparation, inflight slots, result consumption and batch metrics.
 *   Middle: local target execution and activation handoff.
 *   Last: sampling state, request errors, target verification and draft execution.
 * Add PD, MTP/EAGLE, DSpARK and TP/CP cases beside the stage behavior they exercise.
 *
 * Each execution case owns one PP stage. Build request-derived plans through gatherExecutionPlan,
 * supply plans/results through the test transport, and observe model calls, state and outgoing results.
 * Prefer process() for orchestration; call helpers directly for focused draft or metric boundaries.
 * TP replay verifies local root/peer ownership and broadcast contents, not real collectives.
 * History equivalence cases rebuild reference inputs from prompts and committed outputs, then compare
 * real sampling probabilities, exact request RNG states and processor state across rounds.
 *
 * Processor transformations, serialization formats and scheduler policy belong in their own tests.
 * Real multi-stage communication and overlap belong in smoke tests.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <list>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include <pybind11/embed.h>
#include <torch/extension.h>
#include "torch/all.h"
#include "autil/TimeUtility.h"
#include "kmonitor/client/common/MinMaxCalculator.h"
#include "kmonitor/client/core/MetricsCollector.h"

#define private public
#define protected public
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfo.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#undef protected
#undef private

namespace rtp_llm {

void registerExecCtxOps(pybind11::module& m);

namespace {

template<typename T>
std::vector<T> tensorToVector(const torch::Tensor& tensor) {
    const auto cpu = tensor.cpu().contiguous();
    return std::vector<T>(cpu.data_ptr<T>(), cpu.data_ptr<T>() + cpu.numel());
}

torch::Tensor intTensor(std::vector<int32_t> values) {
    return torch::tensor(std::move(values), torch::kInt32);
}

/** Snapshot inputs before later forwards and TP broadcasts reuse their storage. */
GptModelInputs snapshotInput(const GptModelInputs& input) {
    auto copy = input;
    for (auto* tensor : {&copy.combo_tokens,
                         &copy.engram_token_windows,
                         &copy.mm_features_spans,
                         &copy.input_lengths,
                         &copy.sequence_lengths,
                         &copy.prefix_lengths,
                         &copy.lm_output_indexes,
                         &copy.combo_position_ids,
                         &copy.last_hidden_states,
                         &copy.request_id,
                         &copy.request_pd_separation,
                         &copy.cache_keys,
                         &copy.kv_cache_block_id,
                         &copy.kv_cache_kernel_block_id}) {
        if (tensor->defined()) {
            auto snapshot = torch::empty_like(*tensor, tensor->options().pinned_memory(tensor->is_pinned()));
            snapshot.copy_(*tensor);
            *tensor = std::move(snapshot);
        }
    }
    return copy;
}

class RecordingDraftModel: public ModelBase {
public:
    GptModelOutputs forward(const GptModelInputs& input) override {
        if (on_forward) {
            on_forward();
        }
        /** Draft inputs may alias the previous output; consume them before reusing hidden_buffer. */
        inputs.push_back(snapshotInput(input));
        const auto step    = static_cast<int64_t>(inputs.size() - 1);
        const auto rows    = input.combo_tokens.numel();
        const auto options = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
        if (!hidden_buffer.defined() || hidden_buffer.size(0) < rows) {
            hidden_buffer = torch::empty({rows, 4}, options);
        }
        hidden_buffer.narrow(0, 0, rows).copy_(
            torch::arange(rows * 4, options).reshape({rows, 4}) + step * 100 + hidden_offset);
        GptModelOutputs output;
        output.all_hidden_states = hidden_buffer.narrow(0, 0, rows).narrow(1, 0, 2);
        if (input.skip_lm_head) {
            return output;
        }
        const int64_t logit_rows  = next_tokens.empty() ? input.input_lengths.numel() : next_tokens.size();
        output.logits            = torch::zeros({logit_rows, 64}, options);
        for (int64_t row = 0; row < logit_rows; ++row) {
            output.logits[row][next_tokens.empty() ? 10 + step * 2 + row : next_tokens[row]] = 100;
        }
        if (scripted_logits.defined()) {
            output.logits = scripted_logits.clone();
        }
        return output;
    }

    torch::Tensor getMtpTargetHiddenStates(int64_t rows) override {
        hidden_requests.push_back(rows);
        return expose_mtp_hidden ? hidden_buffer.narrow(0, 0, rows) : torch::Tensor();
    }

    torch::Tensor getMtpLastHiddenStates(int64_t rows) override {
        last_hidden_requests.push_back(rows);
        return hidden_buffer.index_select(0, inputs.back().lm_output_indexes.to(hidden_buffer.device(), torch::kLong));
    }

    std::string waitCacheStorePublication() override {
        return wait_publication ? wait_publication() : std::string();
    }

    std::function<std::string()> wait_publication;
    std::vector<GptModelInputs> inputs;
    std::vector<int64_t>        hidden_requests;
    std::vector<int64_t>        last_hidden_requests;
    /** Script model outputs while retaining the executor's real sampling and verification. */
    std::vector<int32_t>   next_tokens;
    std::function<void()> on_forward;
    torch::Tensor         hidden_buffer;
    torch::Tensor         scripted_logits;
    bool                  expose_mtp_hidden = true;
    int64_t               hidden_offset     = 0;
};

class RecordingCPTargetModel: public RecordingDraftModel {
public:
    RecordingCPTargetModel(bool cp_enabled, int rank): cp_enabled_(cp_enabled), rank_(rank) {}

    GptModelOutputs forward(const GptModelInputs& input) override {
        auto output = RecordingDraftModel::forward(input);
        hidden_buffer.add_(rank_ * 1000);
        if (cp_enabled_) {
            const auto           global_lengths = tensorToVector<int32_t>(input.input_lengths);
            std::vector<int32_t> local_lengths;
            int64_t              local_rows = 0;
            for (auto length : global_lengths) {
                /** CP2 pads to four tokens before assigning two chunks per rank. */
                const auto local_length = ((length + 3) / 4) * 2;
                local_lengths.push_back(local_length);
                local_rows += local_length;
            }
            /** Match CP's rebinding of model input fields to local CUDA tensors. */
            target_input_lengths      = intTensor(local_lengths);
            auto& local_input         = const_cast<GptModelInputs&>(input);
            local_input.input_lengths = target_input_lengths.to(torch::kCUDA);
            local_input.combo_tokens  = torch::zeros({local_rows}, input.combo_tokens.options().device(torch::kCUDA));
            /** A one-token fake prefill is padded to two local CP rows. */
            if (hidden_buffer.size(0) < local_rows) {
                hidden_buffer = torch::arange(local_rows * 4, hidden_buffer.options()).reshape({local_rows, 4})
                                    + rank_ * 1000;
            } else {
                hidden_buffer = hidden_buffer.narrow(0, 0, local_rows);
            }
        }
        output.all_hidden_states = hidden_buffer.narrow(1, 0, 2);
        return output;
    }

    torch::Tensor getMtpTargetHiddenStates(int64_t rows) override {
        hidden_requests.push_back(rows);
        return rows < 0 ? hidden_buffer : hidden_buffer.narrow(0, 0, rows);
    }

    torch::Tensor target_input_lengths;

private:
    bool cp_enabled_;
    int  rank_;
};

/** The two fixed-role wrappers share a call log so ordering assertions cover COMMIT then PROPOSE. */
struct RecordingDSparkCalls {
    torch::Tensor               scripted_logits;
    std::function<void()>        on_forward;
    std::function<std::string()> wait_publication;
    std::vector<GptModelInputs>  inputs;
};

class RecordingDSparkModel: public ModelBase {
public:
    RecordingDSparkModel(DSparkModelRole role, int64_t propose_step, std::shared_ptr<RecordingDSparkCalls> calls):
        role_(role), propose_step_(propose_step), calls_(std::move(calls)) {}

    GptModelOutputs forward(const GptModelInputs& input) override {
        if (calls_->on_forward) {
            calls_->on_forward();
        }
        calls_->inputs.push_back(snapshotInput(input));

        GptModelOutputs output;
        EXPECT_EQ(input.last_hidden_states.defined(), role_ == DSparkModelRole::COMMIT);
        if (role_ == DSparkModelRole::PROPOSE && !input.skip_lm_head) {
            const auto batch_size = input.input_lengths.numel();
            output.logits = torch::full({batch_size * propose_step_, 64},
                                        -100.0f,
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            for (int64_t row = 0; row < batch_size * propose_step_; ++row) {
                output.logits[row][10 + row] = 100.0f;
            }
            if (calls_->scripted_logits.defined()) {
                output.logits = calls_->scripted_logits.clone();
            }
        }
        return output;
    }

    torch::Tensor getMtpTargetHiddenStates(int64_t) override {
        ADD_FAILURE() << "DSpARK draft output must not enter the MTP hidden handoff";
        return torch::Tensor();
    }

    torch::Tensor getMtpLastHiddenStates(int64_t) override {
        ADD_FAILURE() << "DSpARK COMMIT does not return MTP decode hidden";
        return torch::Tensor();
    }

    std::string waitCacheStorePublication() override {
        EXPECT_EQ(role_, DSparkModelRole::COMMIT);
        return calls_->wait_publication ? calls_->wait_publication() : std::string();
    }

private:
    const DSparkModelRole                 role_;
    const int64_t                        propose_step_;
    std::shared_ptr<RecordingDSparkCalls> calls_;
};

class RecordingLogitsProcessor: public BaseLogitsProcessor {
public:
    std::optional<ErrorInfo> process(const SamplerInputs&, size_t, size_t) override {
        return process_error;
    }

    ErrorResult<int> prepareSpeculative(const SpecLogitsProcessorRequest& request) override {
        if (verify_error.has_value()) {
            return verify_error.value();
        }
        return static_cast<int>(request.propose_step);
    }

    void updateMultiSeqStatus(const std::vector<int>&) override {}

    MtpProcessorCapability mtpCapability() const override {
        return {mtp_supported ? MtpProcessorMode::SPEC_VERIFY : MtpProcessorMode::UNSUPPORTED,
                "test processor capability"};
    }

    std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t) override {
        if (on_update) {
            on_update();
        }
        committed_tokens.push_back(new_tokens.clone());
        return update_error;
    }

    std::vector<torch::Tensor> committed_tokens;
    std::function<void()>      on_update;
    bool                      mtp_supported = true;
    std::optional<ErrorInfo>   process_error;
    std::optional<ErrorInfo>   verify_error;
    std::optional<ErrorInfo>   update_error;
};

/** Observe a single stage's activation boundary; no neighboring executor is created. */
class RecordingStageModel: public RecordingDraftModel {
public:
    explicit RecordingStageModel(const ParallelismConfig& config): is_last_stage_(config.pp_rank == config.pp_size - 1) {}

    GptModelOutputs forward(const GptModelInputs& input) override {
        has_input.push_back(!input.pp_intermediates.empty());
        if (!input.pp_intermediates.empty()) {
            received_hidden.push_back(input.pp_intermediates.at("hidden_states").clone());
        }
        auto output = RecordingDraftModel::forward(input);
        if (!is_last_stage_) {
            auto hidden                             = output.all_hidden_states;
            output                                  = GptModelOutputs{};
            output.pp_intermediates["hidden_states"] = std::move(hidden);
        }
        has_output.push_back(!output.pp_intermediates.empty());
        return output;
    }

    PPIntermediateTensors makePPWarmUpInputTensors(const GptModelInputs& input, bool) override {
        ++warmup_inputs;
        const auto hidden = torch::full({input.combo_tokens.numel(), 2},
                                        7.0,
                                        torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
        return {{{"hidden_states", hidden}}};
    }

    void releaseBuffers() override {
        ++releases;
    }

    std::vector<bool>          has_input;
    std::vector<bool>          has_output;
    std::vector<torch::Tensor> received_hidden;
    int                       warmup_inputs = 0;
    int                       releases     = 0;

private:
    const bool is_last_stage_;
};

struct WorkState {
    int  waits     = 0;
    int  destroyed = 0;
    bool completed = false;
    bool fail      = false;
};

/** Controllable completion and failures keep lifetime tests independent of backend timing. */
class RecordingWork: public P2PWork {
public:
    explicit RecordingWork(std::shared_ptr<WorkState> state): state_(std::move(state)) {}

    ~RecordingWork() override {
        ++state_->destroyed;
    }

    void wait() override {
        ++state_->waits;
        if (state_->fail) {
            throw std::runtime_error("peer disconnected");
        }
        state_->completed = true;
    }

    bool isCompleted() override {
        return state_->completed;
    }

private:
    std::shared_ptr<WorkState> state_;
};

class InMemoryP2PWork final: public P2PWork {
public:
    explicit InMemoryP2PWork(torch::Tensor tensor,
                             std::function<void(const torch::Tensor&)> on_wait = {}):
        tensor_(std::move(tensor)), on_wait_(std::move(on_wait)) {
        if (tensor_.is_cuda()) {
            ready_.emplace(cuda_graph::makeGraphEvent());
            ready_->record(cuda_graph::graphGetCurrentStream());
        }
    }

    /** Match CUDA Work::wait: order the caller's stream after the submitted copy. */
    void wait() override {
        if (on_wait_) {
            on_wait_(tensor_);
        }
        if (ready_) {
            ready_->block(cuda_graph::graphGetCurrentStream());
        }
        waited_ = true;
    }

    bool isCompleted() override {
        /** CUDA copies expose event completion; CPU work mirrors Gloo's wait-driven completion state. */
        return ready_ ? ready_->query() : waited_;
    }

    const torch::Tensor& tensor() const {
        return tensor_;
    }

private:
    torch::Tensor                            tensor_;
    std::function<void(const torch::Tensor&)> on_wait_;
    std::optional<torch::Event>               ready_;
    bool                                     waited_ = false;
};

class InMemoryPPTransport: public PPTransport {
public:
    void enqueueObject(const torch::Tensor& payload) {
        received_tensors.push_back(torch::tensor({payload.numel()}, torch::kInt64));
        received_tensors.push_back(payload);
    }

    std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) override {
        if (on_send) {
            on_send(tensor);
        }
        sent_tensors.push_back(tensor.clone());
        return std::make_unique<PPCommTicket>(std::make_unique<InMemoryP2PWork>(tensor));
    }

    std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor) override {
        if (on_receive) {
            on_receive(tensor);
        }
        const auto& source = received_tensors.at(receive_index++);
        tensor.copy_(source, /*non_blocking=*/source.is_pinned());
        return std::make_unique<PPCommTicket>(std::make_unique<InMemoryP2PWork>(tensor, on_wait));
    }

    std::vector<torch::Tensor> received_tensors;
    std::vector<torch::Tensor> sent_tensors;
    size_t                    receive_index = 0;
    std::function<void(const torch::Tensor&)> on_receive;
    std::function<void(const torch::Tensor&)> on_send;
    std::function<void(const torch::Tensor&)> on_wait;
};

/** Replay root broadcasts through the real TP input packing and non-root unpacking paths. */
class ReplayTpBroadcast {
public:
    ReplayTpBroadcast() {
        ops_ = py::module_::import("types").attr("ModuleType")("pp_draft_sync_test").cast<py::module_>();
        registerExecCtxOps(ops_);
        const auto unused = py::cpp_function([](py::args) { ADD_FAILURE() << "unexpected collective"; });
        ops_.attr("register_comm_ops")(
            py::cpp_function([this](const std::vector<torch::Tensor>& tensors, int64_t root, int mode) {
                EXPECT_EQ(root, 0);
                EXPECT_EQ(mode, static_cast<int>(ParallelMode::TP));
                if (!replay_) {
                    std::vector<torch::Tensor> buffers;
                    for (const auto& tensor : tensors) {
                        buffers.push_back(tensor.clone());
                    }
                    broadcasts_.push_back(std::move(buffers));
                } else {
                    const auto& buffers = broadcasts_.at(cursor_++);
                    ASSERT_EQ(tensors.size(), buffers.size());
                    for (size_t i = 0; i < tensors.size(); ++i) {
                        ASSERT_EQ(tensors[i].sizes(), buffers[i].sizes());
                        ASSERT_EQ(tensors[i].scalar_type(), buffers[i].scalar_type());
                        ASSERT_EQ(tensors[i].device(), buffers[i].device());
                        tensors[i].copy_(buffers[i]);
                    }
                }
            }),
            py::cpp_function([this](torch::Tensor status, int op, int mode, py::object dest) {
                EXPECT_EQ(op, static_cast<int>(ReduceOp::Min));
                EXPECT_EQ(mode, static_cast<int>(ParallelMode::TP));
                EXPECT_TRUE(dest.is_none());
                EXPECT_TRUE(static_cast<bool>(reduce_status));
                const bool local_ok = status.item<int32_t>() != 0;
                return torch::full_like(status, reduce_status ? reduce_status(local_ok) : local_ok);
            }),
            unused);
    }

    ~ReplayTpBroadcast() {
        ops_.attr("clear_comm_ops")();
    }

    void replay() {
        replay_ = true;
        cursor_ = 0;
    }

    size_t position() const {
        return replay_ ? cursor_ : broadcasts_.size();
    }

    size_t size() const {
        return broadcasts_.size();
    }

    std::function<bool(bool)> reduce_status;

private:
    py::module_                            ops_;
    std::vector<std::vector<torch::Tensor>> broadcasts_;
    bool                                  replay_ = false;
    size_t                                cursor_ = 0;
};

}  // namespace

class PPExecutorTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        previous_backend_ = LogitsProcessorFactory::grammarBackend();
        previous_factory_ = PPExecutor::test_model_factory;
        LogitsProcessorFactory::grammarBackend().reset();
        PPExecutor::test_model_factory = nullptr;
    }

    void TearDown() override {
        LogitsProcessorFactory::grammarBackend() = std::move(previous_backend_);
        PPExecutor::test_model_factory = std::move(previous_factory_);
        DeviceTestBase::TearDown();
    }

    std::shared_ptr<XGrammarBackend> previous_backend_;
    PPExecutor::ModelFactory         previous_factory_;

    static ModelConfig makeModelConfig() {
        ModelConfig model_config{};
        model_config.special_tokens.eos_token_id = 63;
        model_config.max_seq_len               = 32;
        model_config.vocab_size                = 64;
        model_config.input_vocab_size          = 64;
        model_config.num_layers                = 1;
        model_config.hidden_size               = 4;
        model_config.attn_config.head_num      = 2;
        model_config.attn_config.kv_head_num   = 2;
        model_config.attn_config.size_per_head = 2;
        return model_config;
    }

    /** Sampling results require one output and error slot per request, including failed requests. */
    static PPExecutionResult makeInitializedResult(const PPSamplingPlan& plan) {
        PPExecutionResult result;
        const auto        stream_count = plan.request_ids.size(0);
        result.request_ids             = plan.request_ids.to(torch::kCPU).contiguous();
        result.request_errors.assign(stream_count, ErrorInfo::OkStatus());
        result.request_outputs.resize(stream_count);
        return result;
    }

    static EngineInitParams makeMtpParams(size_t num_draft_tokens) {
        EngineInitParams params;
        params.model_id                                 = 0;
        params.model_config_                            = makeModelConfig();
        params.model_config_.num_layers                 = 2;
        params.parallelism_config.pp_size               = 2;
        params.parallelism_config.world_size            = 2;
        params.parallelism_config.pp_stage_layer_counts = {1, 1};
        params.sp_config.type                           = SP_TYPE_MTP;
        params.sp_config.gen_num_per_cycle              = num_draft_tokens;
        params.py_model                                 = py::none();
        params.py_sp_model                              = py::none();
        return params;
    }

    static std::unique_ptr<ProposeModelEngineInitParams> makeMtpProposeParams(const EngineInitParams& params,
                                                                             size_t model_count = 1) {
        auto mtp_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        for (size_t i = 0; i < model_count; ++i) {
            auto draft                = std::make_unique<EngineInitParams>();
            draft->model_id           = i + 1;
            draft->model_config_      = makeModelConfig();
            draft->parallelism_config = params.parallelism_config;
            draft->sp_config          = params.sp_config;
            draft->py_model           = py::none();
            draft->py_sp_model        = py::none();
            if (params.sp_config.type == SP_TYPE_DSPARK) {
                const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
                draft->gpt_weights.dspark_markov_w1 = torch::zeros({64, 1}, options);
                draft->gpt_weights.dspark_markov_w2 = torch::zeros({64, 1}, options);
            }
            mtp_params->emplace_back(std::move(draft));
        }
        return std::make_unique<ProposeModelEngineInitParams>(
            params.sp_config.type, params.sp_config.gen_num_per_cycle, std::move(mtp_params));
    }

    static GenerateStreamPtr makeStream(const ResourceContext& resource_context,
                                        const ModelConfig&     model_config,
                                        int64_t                request_id,
                                        std::vector<int32_t>   input_ids,
                                        const GenerateConfig&  config) {
        auto input                                   = std::make_shared<GenerateInput>();
        input->request_id                            = request_id;
        input->input_ids                             = intTensor(std::move(input_ids));
        input->generate_config                       = std::make_shared<GenerateConfig>(config);

        RuntimeConfig runtime_config;
        auto          stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }

    static GenerateStreamPtr makeStream(const ResourceContext& resource_context,
                                        const ModelConfig&     model_config,
                                        int64_t                request_id,
                                        std::vector<int32_t>   input_ids,
                                        int32_t                num_return_sequences,
                                        int32_t                max_new_tokens = 20) {
        GenerateConfig config;
        config.num_return_sequences = num_return_sequences;
        config.max_new_tokens = max_new_tokens;
        config.do_sample = false;
        return makeStream(resource_context, model_config, request_id, std::move(input_ids), config);
    }

    static EngineInitParams makeStageParams(int pp_rank, SpeculativeType type = SP_TYPE_NONE, int64_t width = 3) {
        auto params = makeMtpParams(width);
        params.model_config_.num_layers                 = 3;
        params.parallelism_config.pp_size               = 3;
        params.parallelism_config.world_size            = 3;
        params.parallelism_config.pp_stage_layer_counts = {1, 1, 1};
        params.parallelism_config.pp_rank               = pp_rank;
        params.parallelism_config.world_rank            = pp_rank;
        params.sp_config.type                           = type;
        params.sp_config.sp_dspark_mask_token_id         = 63;
        return params;
    }

    static std::shared_ptr<RecordingDSparkCalls> installDSparkModels(PPExecutor& executor, int64_t width) {
        auto calls = std::make_shared<RecordingDSparkCalls>();
        executor.draft_commit_model_ =
            std::make_unique<RecordingDSparkModel>(DSparkModelRole::COMMIT, width, calls);
        if (executor.role_type_ != RoleType::PREFILL) {
            executor.draft_model_ =
                std::make_unique<RecordingDSparkModel>(DSparkModelRole::PROPOSE, width, calls);
        }
        return calls;
    }

    static InMemoryPPTransport* attachTransport(PPExecutor& executor) {
        auto  transport     = std::make_unique<InMemoryPPTransport>();
        auto* wire          = transport.get();
        executor.transport_ = std::move(transport);
        return wire;
    }

    static PPIntermediateTensors activations(int64_t rows) {
        const auto hidden =
            torch::arange(rows * 2, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA)).reshape({rows, 2});
        return {{{"hidden_states", hidden}}};
    }

    static void enqueuePlan(InMemoryPPTransport& wire, const PPExecutionPlan& plan, bool non_root = false) {
        wire.enqueueObject(pp_serialization::serializePlan(plan, non_root));
        if (!plan.model_input.skip_run) {
            const auto tensors = activations(plan.model_input.combo_tokens.numel());
            wire.enqueueObject(pp_serialization::serializeTensorsMetadata(tensors));
            for (const auto& [name, tensor] : tensors.tensors) {
                /** Preloaded host data is ready before the executor submits receives on its independent stream. */
                wire.received_tensors.push_back(tensor.cpu().pin_memory());
            }
        }
    }

    static PPExecutionPlan makeShutdownPlan() {
        PPExecutionPlan plan;
        plan.model_input.shutdown = true;
        plan.model_input.skip_run = true;
        return plan;
    }

    static PPExecutionResult lastResult(const InMemoryPPTransport& wire) {
        return pp_serialization::deserializeExecutionResult(wire.sent_tensors.back());
    }

    using RequestHistories = std::unordered_map<RequestId, std::vector<std::vector<int32_t>>>;
    using RequestPrompts   = std::unordered_map<RequestId, std::vector<int32_t>>;

    /** Full-history transport preserves sampling state, but has no dependency on the sender's delta cursor. */
    static PPExecutionPlan fullHistoryPlan(const PPExecutionPlan& delta,
                                           const RequestPrompts&   prompts,
                                           const RequestHistories& histories,
                                           PPExecutor&             reference) {
        auto                 plan = delta;
        std::vector<int32_t> tokens;
        std::vector<int32_t> lengths;
        std::vector<int32_t> input_lengths;
        std::vector<int64_t> offsets{0};
        for (const auto id : tensorToVector<RequestId>(plan.sampling_plan.request_ids)) {
            const auto& sequences = histories.at(id);
            for (const auto& history : sequences) {
                tokens.insert(tokens.end(), history.begin(), history.end());
                lengths.push_back(history.size());
                input_lengths.push_back(prompts.at(id).size());
                offsets.push_back(tokens.size());
            }
            const auto state = reference.sampling_states_.find(id);
            if (state != reference.sampling_states_.end()) {
                state->second.token_history.assign(sequences.size(), {});
            }
        }
        plan.sampling_plan.token_ids        = intTensor(std::move(tokens));
        plan.sampling_plan.token_offsets    = torch::tensor(offsets, torch::kInt64);
        plan.sampling_plan.input_lengths    = intTensor(std::move(input_lengths));
        plan.sampling_plan.sequence_lengths = intTensor(std::move(lengths));
        return plan;
    }

    /** These fixtures exclude EOS/stop tokens and budget clipping, so accepted result prefixes are exactly
     * the stream's committed tokens. Proposals and unused result slots never enter the oracle. */
    static void appendCommittedHistory(const PPExecutionResult& result, RequestHistories& histories) {
        int64_t row = 0;
        for (const auto id : tensorToVector<RequestId>(result.request_ids)) {
            for (auto& history : histories.at(id)) {
                const auto length = result.new_token_lengths[row].item<int32_t>();
                const auto tokens = tensorToVector<int32_t>(result.new_token_ids[row].narrow(0, 0, length));
                history.insert(history.end(), tokens.begin(), tokens.end());
                ++row;
            }
        }
    }

    /** Build stage inputs through the processor so model geometry and history deltas stay consistent. */
    static PPExecutionPlan makePlan(const EngineInitParams&                 params,
                                    bool                                    decode,
                                    bool                                    prefill_only,
                                    std::unordered_map<RequestId, int32_t>& history_lengths) {
        const auto first = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1, prefill_only ? 0 : 20);
        const auto second =
            makeStream(ResourceContext{}, params.model_config_, 202, {3, 4, 5}, 1, prefill_only ? 0 : 20);
        if (decode) {
            first->update({intTensor({7}).reshape({1, 1}), 1}, false);
            second->update({intTensor({8}).reshape({1, 1}), 1}, false);
        }
        PPBatchStreamProcessor processor(params.model_config_,
                                         params.pd_sep_config,
                                         ProfilingDebugLoggingConfig{},
                                         CacheConfig{},
                                         params.sp_config,
                                         false);
        int                    row = 0;
        for (const auto& stream : {first, second}) {
            stream->generateConfig()->pd_separation = params.pd_sep_config.role_type == RoleType::PREFILL;
            if (decode && params.sp_config.type != SP_TYPE_NONE) {
                const auto           width = params.sp_config.gen_num_per_cycle;
                std::vector<int32_t> tokens{7 + row};
                for (int step = 0; step < width; ++step) {
                    tokens.push_back(11 + 10 * row + step);
                }
                auto proposals          = std::make_shared<SpeculativeExecutorStreamOutput>();
                proposals->propose_step = width;
                proposals->tokens       = intTensor(std::move(tokens)).reshape({1, width + 1});
                stream->setSPOutputBuffer(proposals);
            }
            ++row;
        }
        TensorHolder holder;
        auto plan = processor.gatherExecutionPlan(StreamGroups({first, second}), {}, false, history_lengths, holder);
        EXPECT_TRUE(plan.ok()) << plan.status().ToString();
        return std::move(plan.value());
    }

    static PPExecutionPlan makePlan(const EngineInitParams& params, bool decode = false, bool prefill_only = false) {
        std::unordered_map<RequestId, int32_t> history_lengths;
        return makePlan(params, decode, prefill_only, history_lengths);
    }

    static void cacheDSparkProposals(PPExecutor& executor, const PPExecutionPlan& plan) {
        if (!executor.is_dspark_) {
            return;
        }
        const auto batch = plan.sampling_plan.request_ids.numel();
        const auto width = executor.propose_step_;
        const auto tokens = plan.model_input.combo_tokens.reshape({batch, width + 1})
                                .narrow(1, 1, width).to(torch::kCUDA, torch::kLong);
        const auto q = torch::zeros({batch, width, static_cast<int64_t>(executor.draft_vocab_size_)},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                           .scatter_(-1, tokens.unsqueeze(-1), 1.0f);
        const auto* ids = plan.sampling_plan.request_ids.data_ptr<int64_t>();
        for (int64_t row = 0; row < batch; ++row) {
            executor.dspark_q_.emplace(ids[row], q[row]);
        }
    }

    static void enableMetrics(PPExecutor& executor) {
        /** Attach after construction so background TPS threads cannot drain the collectors under test. */
        executor.metrics_reporter_ = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags());
    }

    static GenerateStreamPtr makeFakeStream(const EngineInitParams& params,
                                            const ResourceContext& resources,
                                            bool                   decode = false) {
        return decode ? PPExecutor::createMinFakeDecodeStream(
                            params.model_config_, params.runtime_config, resources, params.sp_config) :
                        PPExecutor::createMinFakePrefillStream(params.model_config_, params.runtime_config,
                            resources, params.sp_config, params.pd_sep_config.role_type);
    }

    static PPExecutionPlan makeFakePlan(const EngineInitParams& params, bool decode) {
        ResourceContext resources;
        resources.cache_manager =
            std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(3, 16, 4, DataType::TYPE_FP16));
        EXPECT_TRUE(resources.cache_manager->init());
        auto                                   stream = makeFakeStream(params, resources, decode);
        PPBatchStreamProcessor                 processor(params.model_config_,
                                         params.pd_sep_config,
                                         ProfilingDebugLoggingConfig{},
                                         resources.cache_manager->cacheConfig(),
                                         params.sp_config,
                                         false);
        const StreamGroups                     groups({stream});
        TensorHolder                           holder;
        std::unordered_map<RequestId, int32_t> history_lengths;
        auto plan = processor.gatherExecutionPlan(groups, {}, false, history_lengths, holder);
        EXPECT_TRUE(plan.ok()) << plan.status().ToString();
        EXPECT_TRUE(history_lengths.empty());
        return std::move(plan.value());
    }

    static std::shared_ptr<RecordingLogitsProcessor> recordState(PPExecutor& executor, int64_t id, int sequences = 1) {
        auto processor = std::make_shared<RecordingLogitsProcessor>();
        executor.sampling_states_[id].token_history.resize(sequences);
        auto& state             = executor.sampling_states_[id].sampling;
        state.logits_processors = {processor};
        state.cum_log_probs = torch::zeros({sequences}, torch::kFloat32);
        return processor;
    }

    /** A single-request result carrying one new token. */
    static PPExecutionResult makeSingleTokenResult(int64_t request_id, int32_t token) {
        PPExecutionResult result;
        result.request_ids       = torch::tensor({request_id}, torch::kInt64);
        result.new_token_ids     = intTensor({token}).reshape({1, 1});
        result.new_token_lengths = intTensor({1});
        result.request_errors.resize(1);
        result.request_outputs.resize(1);
        return result;
    }

    /** Stage params for a two-rank TP group; world_rank follows the stage * 2 + tp_rank layout. */
    static EngineInitParams
    makeTpStageParams(int pp_rank, int tp_rank, SpeculativeType type = SP_TYPE_NONE, int64_t width = 3) {
        auto params = makeStageParams(pp_rank, type, width);
        params.parallelism_config.tp_size    = 2;
        params.parallelism_config.world_size = 6;
        params.parallelism_config.tp_rank    = tp_rank;
        params.parallelism_config.world_rank = pp_rank * 2 + tp_rank;
        return params;
    }

    /** Install the draft model for a speculative type and return its recorded inputs; SP_TYPE_NONE installs nothing.
     * Optional callbacks attach to the installed model so a test can observe forward and publication. */
    static std::vector<GptModelInputs>*
    installDraftModel(PPExecutor& executor, SpeculativeType type, int64_t width, std::function<void()> on_forward = {},
                      std::function<std::string()> wait_publication = {}) {
        if (type == SP_TYPE_DSPARK) {
            auto calls              = installDSparkModels(executor, width);
            calls->on_forward       = std::move(on_forward);
            calls->wait_publication = std::move(wait_publication);
            return &calls->inputs;
        }
        if (type == SP_TYPE_NONE) {
            return nullptr;
        }
        auto draft              = std::make_unique<RecordingDraftModel>();
        draft->on_forward       = std::move(on_forward);
        draft->wait_publication = std::move(wait_publication);
        auto* inputs            = &draft->inputs;
        executor.draft_model_   = std::move(draft);
        return inputs;
    }

    /** Construct a recording model, hand it to the executor, and return it for assertions. */
    template <typename Model, typename... Args>
    static Model* installModel(PPExecutor& executor, Args&&... args) {
        auto model     = std::make_unique<Model>(std::forward<Args>(args)...);
        auto* recorded = model.get();
        executor.setModel(std::move(model));
        return recorded;
    }

    /** Every slot has released its asynchronous sends. */
    static void expectSlotsDrained(const PPExecutor& executor) {
        for (const auto& batch : executor.slots_) {
            EXPECT_TRUE(batch.plan_sends.empty());
            EXPECT_TRUE(batch.activation_sends.empty());
            EXPECT_TRUE(batch.execution_result_sends.empty());
        }
    }
};

class PPExecutorTpTest: public PPExecutorTest {
public:
    /** Keep one Python runtime for the TP callback cases; clear callbacks before teardown. */
    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<py::scoped_interpreter>();
        /** Python's OpenSSL modules must resolve their own symbols before the linked BoringSSL symbols. */
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

/** Common construction/configuration contracts. */
TEST_F(PPExecutorTest, FirstLocalMoeLayerUsesStageRange) {
    const int64_t expected[] = {-1, 2, 5};
    for (int stage : {0, 1, 2}) {
        SCOPED_TRACE(stage);
        auto params                                     = makeStageParams(stage);
        params.model_config_.num_layers                 = 8;
        params.model_config_.moe_layer_index            = {2, 4, 5, 7};
        params.parallelism_config.pp_stage_layer_counts = {2, 3, 3};
        PPExecutor executor(params, nullptr, true);
        EXPECT_EQ(executor.firstLocalMoeLayer(params.model_config_), expected[stage]);
        params.model_config_.moe_layer_index.clear();
        EXPECT_EQ(executor.firstLocalMoeLayer(params.model_config_), -1);
    }
}

TEST_F(PPExecutorTest, CommonDenseStageDoesNotStartExpertBalancer) {
    auto params                                     = makeStageParams(0);
    params.model_config_.num_layers                 = 8;
    params.model_config_.moe_style                  = 1;
    params.model_config_.expert_num                 = 2;
    params.model_config_.moe_layer_index            = {2, 4, 5, 7};
    params.parallelism_config.pp_stage_layer_counts = {2, 3, 3};
    params.eplb_config.eplb_mode                    = EplbMode::ALL;
    PPExecutor executor(params, nullptr, true);
    EXPECT_EQ(executor.expert_balancer_, nullptr);
}

TEST_F(PPExecutorTpTest, CommonPythonTargetIgnoresGenerationPrefillGraphBuckets) {
    /** The embedded test reaches initialize without loading the production extension's type registry. */
    py::class_<torch_ext::PyModelInitResources>(py::module_::import("__main__"), "PyModelInitResources");
    py::dict scope;
    py::exec(R"(
class TargetInitializationProbe:
    @property
    def initialize(self):
        raise RuntimeError("PP target reached eager Python initialization")
)", scope);
    for (int stage : {0, 2}) {
        SCOPED_TRACE(stage);
        auto params = makeStageParams(stage);
        params.parallelism_config.role_type = RoleType::PDFUSION;
        params.pd_sep_config.role_type = RoleType::PDFUSION;
        params.hw_kernel_config.enable_cuda_graph = true;
        params.hw_kernel_config.generation_prefill_capture_token_buckets = {8, 16};
        params.py_model = scope["TargetInitializationProbe"]();
        /** Exercise the real wrapper and stop before Python binding casts or graph capture. A missing
         * allow_cuda_graph=false fails the single-device graph check before reaching this probe. */
        try {
            PPExecutor executor(params, nullptr, true);
            FAIL() << "target initialization probe must stop construction";
        } catch (const py::error_already_set& error) {
            EXPECT_NE(std::string(error.what()).find("PP target reached eager Python initialization"),
                      std::string::npos);
        }
    }
}

TEST_F(PPExecutorTest, CommonRejectsTorchSpecCaptureBeforeModelConstruction) {
    bool model_constructed = false;
    PPExecutor::test_model_factory = [&model_constructed](const GptModelInitParams&) {
        model_constructed = true;
        return std::make_unique<RecordingDraftModel>();
    };
    for (int stage : {0, 1, 2}) {
        for (bool warm_up : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "stage=" << stage << ", warm_up=" << warm_up);
            auto params = makeStageParams(stage);
            params.model_config_.hidden_state_capture_layer_ids = {0};
            try {
                PPExecutor executor(params, nullptr, warm_up);
                FAIL() << "PP accepted TorchSpec hidden-state capture";
            } catch (const std::runtime_error& error) {
                EXPECT_NE(std::string(error.what()).find(
                              "pipeline parallelism does not support TorchSpec hidden-state capture"),
                          std::string::npos)
                    << error.what();
            }
            EXPECT_FALSE(model_constructed);
        }
    }
}

TEST_F(PPExecutorTest, CommonRejectsBatchDecodeSchedulerWithOrWithoutWarmup) {
    for (int stage : {0, 1, 2}) {
        for (bool warm_up : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "stage=" << stage << ", warm_up=" << warm_up);
            auto params = makeStageParams(stage);
            params.runtime_config.warm_up = warm_up;
            params.runtime_config.use_batch_decode_scheduler = true;
            try {
                PPExecutor executor(params, nullptr, warm_up);
                FAIL() << "PP accepted use_batch_decode_scheduler";
            } catch (const std::runtime_error& error) {
                EXPECT_NE(std::string(error.what()).find("does not support use_batch_decode_scheduler"),
                          std::string::npos)
                    << error.what();
            }
        }
    }
}

TEST_F(PPExecutorTest, CommonBuildsOnlyTheFirstConfiguredDraftModule) {
    auto params                            = makeMtpParams(2);
    params.parallelism_config.pp_rank      = 1;
    params.parallelism_config.world_rank   = 1;
    params.parallelism_config.world_size   = 2;
    auto                propose_params     = makeMtpProposeParams(params, 2);
    std::vector<size_t> constructed_models = {};
    PPExecutor::test_model_factory = [&constructed_models](const GptModelInitParams& init_params) {
        constructed_models.push_back(init_params.model_id);
        return std::make_unique<RecordingDraftModel>();
    };

    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose_params.get());

    EXPECT_EQ(constructed_models, (std::vector<size_t>{0, 1}));
}

TEST_F(PPExecutorTest, CommonActiveCpAllowsOrdinaryDecodeWithReplicatedKv) {
    for (const auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (const auto role : {RoleType::PREFILL, RoleType::DECODE, RoleType::PDFUSION}) {
            for (const bool kv_sharded : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "type=" << type << ", role=" << role
                                                  << ", kv_sharded=" << kv_sharded);
                auto params = makeMtpParams(3);
                params.sp_config.type = type;
                params.sp_config.sp_dspark_mask_token_id = 63;
                params.pd_sep_config.role_type = role;
                params.parallelism_config.role_type = role;
                params.parallelism_config.tp_size = 2;
                params.parallelism_config.world_size = 4;
                params.parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
                params.parallelism_config.prefill_cp_config.kv_cache_sharded = kv_sharded;
                if (role == RoleType::PREFILL || (type == SP_TYPE_NONE && !kv_sharded)) {
                    EXPECT_NO_THROW({ PPExecutor executor(params, nullptr, true); });
                } else {
                    try {
                        PPExecutor executor(params, nullptr, true);
                        FAIL() << "active CP accepted speculative or KV-sharded decode";
                    } catch (const std::runtime_error& error) {
                        EXPECT_NE(std::string(error.what()).find(
                                      "pipeline parallelism with active CP requires the PREFILL role"),
                                  std::string::npos);
                    }
                }
                if (role == RoleType::DECODE) {
                    params.parallelism_config.prefill_cp_config.method = CPRotateMethod::PREFILL_CP;
                    EXPECT_NO_THROW({ PPExecutor executor(params, nullptr, true); });
                }
            }
        }
    }
}

TEST_F(PPExecutorTest, CommonRejectsNonPositiveSpeculativeWidth) {
    auto params = makeMtpParams(3);
    for (const int64_t width : {-1, 0}) {
        SCOPED_TRACE(width);
        params.sp_config.gen_num_per_cycle = width;
        try {
            PPExecutor executor(params, nullptr, true);
            FAIL() << "expected non-positive speculative width to be rejected";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("PP speculative decoding requires a positive gen_num_per_cycle"),
                      std::string::npos)
                << e.what();
        }
    }
}

/** Common: execution mechanics shared by stages, without constructing a pipeline. */
TEST_F(PPExecutorTest, CommonProfileStartsBeforeActivationReceiveAndSkipsEmptyExecution) {
    auto params = makeStageParams(2);
    PPExecutor executor(params, nullptr, false);
    auto* wire = attachTransport(executor);
    enqueuePlan(*wire, makePlan(params));
    PPExecutionPlan empty;
    empty.model_input.skip_run = true;
    enqueuePlan(*wire, empty);
    bool profiling = false;
    int starts = 0;
    int finishes = 0;
    executor.profile_step_start_ = [&] {
        EXPECT_FALSE(profiling);
        EXPECT_EQ(wire->receive_index, 2u);
        profiling = true;
        ++starts;
    };
    executor.profile_step_finish_ = [&] {
        EXPECT_TRUE(profiling);
        EXPECT_EQ(wire->sent_tensors.size(), 2u);
        profiling = false;
        ++finishes;
    };
    auto model = std::make_unique<RecordingStageModel>(params.parallelism_config);
    model->on_forward = [&] { EXPECT_TRUE(profiling); };
    executor.setModel(std::move(model));
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_FALSE(profiling);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_EQ(starts, 1);
    EXPECT_EQ(finishes, 1);
    EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
}

TEST_F(PPExecutorTest, CommonWarmupUsesLocalStageInputsWithoutPipelineTraffic) {
    for (int stage : {0, 1, 2}) {
        SCOPED_TRACE(stage);
        auto params = makeStageParams(stage);
        PPExecutor executor(params, nullptr, true);
        auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
        auto* wire = attachTransport(executor);
        int starts = 0;
        int finishes = 0;
        executor.profile_step_start_ = [&] { ++starts; };
        executor.profile_step_finish_ = [&] { ++finishes; };
        auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1);
        stream->setIsFakeStream(true);
        ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());
        ASSERT_EQ(recorded->inputs.size(), 1u);
        EXPECT_EQ(recorded->has_input, std::vector<bool>{stage != 0});
        EXPECT_EQ(recorded->has_output, std::vector<bool>{stage != 2});
        EXPECT_EQ(recorded->warmup_inputs, stage == 0 ? 0 : 1);
        EXPECT_EQ(recorded->releases, 2);
        if (stage != 0) {
            EXPECT_TRUE(torch::equal(recorded->received_hidden[0],
                                    torch::full_like(recorded->received_hidden[0], 7)));
        }
        EXPECT_TRUE(wire->sent_tensors.empty());
        EXPECT_EQ(wire->receive_index, 0u);
        EXPECT_TRUE(executor.sampling_states_.empty());
        EXPECT_EQ(starts, 0);
        EXPECT_EQ(finishes, 0);
    }
}

TEST_F(PPExecutorTest, CommonSlotReuseWaitsForPreviousSendsBeforeForward) {
    for (int stage : {0, 1, 2}) {
        SCOPED_TRACE(stage);
        auto params = makeStageParams(stage);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto previous_plan = std::make_shared<WorkState>();
        auto previous_payload = std::make_shared<WorkState>();
        auto other_slot_payload = std::make_shared<WorkState>();
        other_slot_payload->fail = true;
        auto& slot = executor.slots_[executor.current_slot_];
        auto& other_slot = executor.slots_[(executor.current_slot_ + 1) % executor.slots_.size()];
        auto add_work = [](PPTickets& tickets, const std::shared_ptr<WorkState>& state) {
            tickets.push_back(std::make_unique<PPCommTicket>(std::make_unique<RecordingWork>(state)));
        };
        if (stage == 2) {
            add_work(slot.execution_result_sends, previous_payload);
            add_work(other_slot.execution_result_sends, other_slot_payload);
        } else {
            add_work(slot.plan_sends, previous_plan);
            add_work(slot.activation_sends, previous_payload);
            add_work(other_slot.activation_sends, other_slot_payload);
        }
        /** Reusing one slot must leave another batch's unfinished sends owned and unwaited. */
        auto* model = installModel<RecordingStageModel>(executor, params.parallelism_config);
        model->on_forward = [&] {
            EXPECT_EQ(previous_plan->waits, stage == 2 ? 0 : 1);
            EXPECT_EQ(previous_payload->waits, 1);
            EXPECT_EQ(other_slot_payload->waits, 0);
            EXPECT_EQ(other_slot_payload->destroyed, 0);
        };
        ScheduleOutput schedule;
        if (stage == 0) {
            schedule.streams = {makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1)};
        } else {
            enqueuePlan(*wire, makePlan(params));
        }
        ASSERT_TRUE(executor.process(schedule).ok());
        EXPECT_EQ(previous_payload->waits, 1);
        EXPECT_EQ(other_slot_payload->waits, 0);
        EXPECT_EQ(other_slot_payload->destroyed, 0);
        /** New sends belong to the current execution, not to the completed work. */
        executor.waitAll(slot.plan_sends);
        executor.waitAll(slot.activation_sends);
        executor.waitAll(slot.execution_result_sends);
        EXPECT_EQ(previous_payload->waits, 1);
    }
}

TEST_F(PPExecutorTest, CommonActivationSendsReapedAcrossSlotsIncludingSkippedBatches) {
    for (const bool skip_run : {false, true}) {
        SCOPED_TRACE(skip_run);
        auto params = makeStageParams(1);
        PPExecutor executor(params, nullptr, false);
        auto* wire  = attachTransport(executor);
        auto* model = installModel<RecordingStageModel>(executor, params.parallelism_config);
        auto plan   = makePlan(params);
        plan.model_input.skip_run = skip_run;
        enqueuePlan(*wire, plan);

        std::vector<std::shared_ptr<WorkState>> metadata;
        std::vector<std::shared_ptr<WorkState>> pending;
        std::vector<std::shared_ptr<WorkState>> completed;
        for (size_t slot = 1; slot < executor.slots_.size(); ++slot) {
            auto& batch = executor.slots_[slot];
            ASSERT_FALSE(batch.result_pending);
            metadata.push_back(std::make_shared<WorkState>());
            pending.push_back(std::make_shared<WorkState>());
            completed.push_back(std::make_shared<WorkState>());
            completed.back()->completed = true;
            /** Pending metadata and payloads must not retain a later completed snapshot. */
            for (const auto& state : {metadata.back(), pending.back(), completed.back()}) {
                batch.activation_sends.push_back(
                    std::make_unique<PPCommTicket>(std::make_unique<RecordingWork>(state)));
            }
        }

        const auto expect_reclaimed = [&] {
            for (size_t i = 0; i < completed.size(); ++i) {
                EXPECT_EQ(completed[i]->waits, 1);
                EXPECT_EQ(completed[i]->destroyed, 1);
                EXPECT_EQ(metadata[i]->waits, 0);
                EXPECT_EQ(metadata[i]->destroyed, 0);
                EXPECT_EQ(pending[i]->waits, 0);
                EXPECT_EQ(pending[i]->destroyed, 0);
                EXPECT_EQ(executor.slots_[i + 1].activation_sends.size(), 2u);
            }
        };
        model->on_forward = expect_reclaimed;
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_EQ(model->inputs.size(), skip_run ? 0u : 1u);
        expect_reclaimed();

        for (auto& state : pending) {
            state->completed = true;
        }
        executor.reapCompletedActivationSends();
        for (size_t i = 0; i < pending.size(); ++i) {
            EXPECT_EQ(pending[i]->waits, 1);
            EXPECT_EQ(pending[i]->destroyed, 1);
            EXPECT_EQ(metadata[i]->waits, 0);
            EXPECT_EQ(metadata[i]->destroyed, 0);
            auto& tickets = executor.slots_[i + 1].activation_sends;
            ASSERT_EQ(tickets.size(), 1u);
            executor.waitAll(tickets);
            EXPECT_TRUE(tickets.empty());
            EXPECT_EQ(metadata[i]->waits, 1);
            EXPECT_EQ(metadata[i]->destroyed, 1);
        }
    }
}

TEST_F(PPExecutorTest, CommonActivationReapingPropagatesFailureWithoutDroppingRemainingSends) {
    auto params = makeStageParams(1);
    PPExecutor executor(params, nullptr, false);
    auto pending   = std::make_shared<WorkState>();
    auto completed = std::make_shared<WorkState>();
    auto failed    = std::make_shared<WorkState>();
    auto later     = std::make_shared<WorkState>();
    completed->completed = true;
    failed->completed    = true;
    failed->fail         = true;
    later->completed     = true;
    auto& tickets = executor.slots_[1].activation_sends;
    for (const auto& state : {pending, completed, failed, later}) {
        tickets.push_back(std::make_unique<PPCommTicket>(std::make_unique<RecordingWork>(state)));
    }

    EXPECT_THROW(executor.reapCompletedActivationSends(), std::runtime_error);
    EXPECT_EQ(completed->waits, 1);
    EXPECT_EQ(completed->destroyed, 1);
    EXPECT_EQ(failed->waits, 1);
    EXPECT_EQ(failed->destroyed, 0);
    EXPECT_EQ(pending->waits, 0);
    EXPECT_EQ(pending->destroyed, 0);
    EXPECT_EQ(later->waits, 0);
    EXPECT_EQ(later->destroyed, 0);
    EXPECT_EQ(tickets.size(), 3u);
}

TEST_F(PPExecutorTest, CommonActivationSendKeepsSnapshotUntilReclaimedAndMetadataUntilWait) {
    for (bool noncontiguous : {false, true}) {
        SCOPED_TRACE(noncontiguous);
        auto params = makeStageParams(1);
        PPExecutor executor(params, nullptr, false);
        attachTransport(executor);
        auto backing = torch::arange(16, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA)).reshape({4, 4});
        auto source = noncontiguous ? backing.narrow(1, 0, 2) : backing;
        ASSERT_EQ(source.is_contiguous(), !noncontiguous);
        const auto expected = source.clone();
        auto& tickets = executor.slots_[0].activation_sends;
        executor.asyncSendTensors({{{"hidden_states", source}}}, tickets);
        ASSERT_EQ(tickets.size(), 3u);
        auto* work = static_cast<InMemoryP2PWork*>(tickets.back()->work_.get());
        ASSERT_NE(work, nullptr);
        EXPECT_TRUE(work->tensor().is_contiguous());
        EXPECT_NE(work->tensor().data_ptr(), source.data_ptr());

        source.fill_(-1);
        tickets.back()->wait();
        ASSERT_EQ(tickets.back()->work_.get(), work);
        EXPECT_TRUE(torch::equal(work->tensor(), expected));
        executor.reapCompletedActivationSends();
        ASSERT_EQ(tickets.size(), 2u);
        executor.waitAll(tickets);
        EXPECT_TRUE(tickets.empty());
    }
}

TEST_F(PPExecutorTest, CommonActivationReceivesReuseTwoBuffersAcrossForwardsAndSkippedBatches) {
    auto params = makeStageParams(1);
    PPExecutor executor(params, nullptr, false);
    auto* wire  = attachTransport(executor);
    auto* model = installModel<RecordingStageModel>(executor, params.parallelism_config);
    PPExecutionPlan empty;
    empty.model_input.skip_run = true;
    const std::vector<PPExecutionPlan> plans{
        makePlan(params), makePlan(params), empty, makePlan(params, true), makePlan(params, true), makePlan(params)};
    std::vector<torch::Tensor> expected;
    for (const auto& plan : plans) {
        enqueuePlan(*wire, plan);
        if (!plan.model_input.skip_run) {
            auto& payload = wire->received_tensors.back();
            payload.add_(static_cast<int64_t>(expected.size()) * 100);
            expected.push_back(payload.clone());
        }
    }

    const auto compute_stream = cuda_graph::graphGetCurrentStream();
    EXPECT_NE(cuda_graph::toGraphStream(executor.recv_stream_), compute_stream);
    std::vector<torch::Tensor> received;
    size_t                    waits = 0;
    wire->on_receive = [&](const torch::Tensor& tensor) {
        if (tensor.is_cuda()) {
            EXPECT_EQ(cuda_graph::graphGetCurrentStream(), cuda_graph::toGraphStream(executor.recv_stream_));
            received.push_back(tensor);
        }
    };
    wire->on_wait = [&](const torch::Tensor& tensor) {
        if (tensor.is_cuda()) {
            EXPECT_EQ(cuda_graph::graphGetCurrentStream(), compute_stream);
            ++waits;
        }
    };

    size_t executions = 0;
    for (const auto& plan : plans) {
        SCOPED_TRACE(testing::Message() << "executions=" << executions << ", skip_run=" << plan.model_input.skip_run);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_EQ(cuda_graph::graphGetCurrentStream(), compute_stream);
        if (!plan.model_input.skip_run) {
            ++executions;
        }
        ASSERT_EQ(received.size(), executions);
        ASSERT_EQ(model->received_hidden.size(), executions);
        EXPECT_EQ(waits, executions);
        EXPECT_EQ(received.back().sizes().vec(), expected[executions - 1].sizes().vec());
        EXPECT_TRUE(torch::equal(model->received_hidden.back().cpu(), expected[executions - 1]));
        if (executions >= 2) {
            EXPECT_NE(received[0].data_ptr(), received[1].data_ptr());
            EXPECT_TRUE(torch::equal(received[executions - 2].cpu(), expected[executions - 2]));
        }
        if (executions >= 3) {
            EXPECT_EQ(received.back().data_ptr(), received[(executions - 1) % 2].data_ptr());
            EXPECT_EQ(received.back().storage().nbytes(), received[(executions - 1) % 2].storage().nbytes());
        }
    }
    EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
}

TEST_F(PPExecutorTest, FinishWaitsForEarlierSendsAndPropagatesFailures) {
    for (int stage : {1, 2}) {
        for (bool fail : {false, true}) {
            SCOPED_TRACE(testing::Message() << "stage=" << stage << ", fail=" << fail);
            auto params = makeStageParams(stage);
            PPExecutor executor(params, nullptr, false);
            auto* wire = attachTransport(executor);
            auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
            enqueuePlan(*wire, makeShutdownPlan());

            /** Earlier batches still own sends in slots other than the one used by the terminal plan. */
            std::vector<std::shared_ptr<WorkState>> pending;
            for (size_t slot = 1; slot < executor.slots_.size(); ++slot) {
                auto& batch = executor.slots_[slot];
                for (auto* tickets : {&batch.plan_sends, &batch.activation_sends, &batch.execution_result_sends}) {
                    if ((stage == 2) != (tickets == &batch.execution_result_sends)) {
                        continue;
                    }
                    auto state = std::make_shared<WorkState>();
                    tickets->push_back(std::make_unique<PPCommTicket>(std::make_unique<RecordingWork>(state)));
                    pending.push_back(state);
                }
            }
            pending.back()->fail = fail;
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
            EXPECT_TRUE(executor.stopped_);
            for (const auto& state : pending) {
                EXPECT_EQ(state->waits, 0);
            }
            const auto sends = wire->sent_tensors.size();
            const auto receives = wire->receive_index;
            if (fail) {
                EXPECT_THROW((void)executor.finish(), std::runtime_error);
            } else {
                ASSERT_TRUE(executor.finish().ok());
                expectSlotsDrained(executor);
            }
            for (const auto& state : pending) {
                EXPECT_EQ(state->waits, 1);
            }
            EXPECT_TRUE(recorded->inputs.empty());
            EXPECT_EQ(wire->sent_tensors.size(), sends);
            EXPECT_EQ(wire->receive_index, receives);
        }
    }
}

TEST_F(PPExecutorTest, CommonModelExecutionExceptionDoesNotSendSuccessfulOutput) {
    for (int stage : {0, 1, 2}) {
        SCOPED_TRACE(stage);
        auto params = makeStageParams(stage);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
        target->on_forward = [] { throw std::runtime_error("target forward failed"); };
        executor.setModel(std::move(target));
        ScheduleOutput schedule;
        if (stage == 0) {
            schedule.streams = {makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1)};
        } else {
            enqueuePlan(*wire, makePlan(params));
        }
        EXPECT_THROW((void)executor.process(schedule), std::runtime_error);
        /** Non-last stages have sent only their plan; no activation/result follows a failed forward. */
        EXPECT_EQ(wire->sent_tensors.size(), stage == 2 ? 0u : 2u);
    }
}

/** First: own scheduled streams and consume the result belonging to each inflight batch. */
TEST_F(PPExecutorTest, FirstSendsHistoryDeltasAndRetiresCursorBeforeRequestIdReuse) {
    auto            params = makeStageParams(0);
    ResourceContext resources;
    resources.cache_manager =
        std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(3, 16, 4, DataType::TYPE_FP16));
    ASSERT_TRUE(resources.cache_manager->init());
    PPExecutor executor(params, resources.cache_manager, false);
    auto*      wire = attachTransport(executor);
    installModel<RecordingStageModel>(executor, params.parallelism_config);
    auto stream = makeStream(resources, params.model_config_, 101, {1, 2}, 1);
    stream->fakeInitKVBlock(4);

    wire->on_send = [&](const torch::Tensor&) {
        if (wire->sent_tensors.size() < 2) {
            EXPECT_EQ(executor.token_history_lengths_.at(101), 2);
        }
    };
    wire->enqueueObject(pp_serialization::serializeExecutionResult(makeSingleTokenResult(101, 7)));
    stream->setPPInflight();
    ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());
    EXPECT_EQ(executor.token_history_lengths_.at(101), 2);
    const auto first = pp_serialization::deserializePlan(wire->sent_tensors.at(1));
    EXPECT_EQ(tensorToVector<int32_t>(first.sampling_plan.token_ids), (std::vector<int32_t>{1, 2}));
    ASSERT_TRUE(executor.drainPendingResults().ok());
    EXPECT_EQ(stream->seqLength(), 3);
    EXPECT_EQ(executor.token_history_lengths_.at(101), 2);

    const auto sends = wire->sent_tensors.size();
    wire->enqueueObject(pp_serialization::serializeExecutionResult(makeSingleTokenResult(101, 8)));
    stream->setPPInflight();
    ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());
    const auto delta = pp_serialization::deserializePlan(wire->sent_tensors.at(sends + 1));
    EXPECT_EQ(tensorToVector<int32_t>(delta.sampling_plan.token_ids), (std::vector<int32_t>{7}));
    EXPECT_EQ(tensorToVector<int64_t>(delta.sampling_plan.token_offsets), (std::vector<int64_t>{0, 1}));
    EXPECT_EQ(executor.token_history_lengths_.at(101), 3);
    ASSERT_TRUE(executor.drainPendingResults().ok());

    ScheduleOutput cleanup;
    cleanup.finished_request_ids = {101};
    ASSERT_TRUE(executor.process(cleanup).ok());
    EXPECT_TRUE(executor.token_history_lengths_.empty());
    const auto cleanup_plan = pp_serialization::deserializePlan(wire->sent_tensors.back());
    EXPECT_TRUE(cleanup_plan.model_input.skip_run);
    EXPECT_EQ(cleanup_plan.finished_request_ids, cleanup.finished_request_ids);
    auto       reused = makeStream(resources, params.model_config_, 101, {9, 10, 11}, 1);
    const auto fresh  = executor.buildPlan(StreamGroups({reused}), {}, false);
    ASSERT_TRUE(fresh.ok());
    EXPECT_EQ(tensorToVector<int32_t>(fresh->sampling_plan.token_ids), (std::vector<int32_t>{9, 10, 11}));
    EXPECT_EQ(executor.token_history_lengths_.at(101), 3);
}

TEST_F(PPExecutorTest, FirstExecutesPrefillAndDecodeWithoutUpstreamActivations) {
    for (bool decode : {false, true}) {
        SCOPED_TRACE(decode);
        auto params = makeStageParams(0);
        ResourceContext resources;
        resources.cache_manager = std::make_shared<KVCacheManager>(
            test::makeSimpleMhaCacheConfig(3, 16, 4, DataType::TYPE_FP16));
        ASSERT_TRUE(resources.cache_manager->init());
        PPExecutor executor(params, resources.cache_manager, false);
        auto* wire = attachTransport(executor);
        auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
        auto stream = makeStream(resources, params.model_config_, 101, {1, 2}, 1);
        stream->fakeInitKVBlock(4);
        if (decode) {
            stream->update({intTensor({7}).reshape({1, 1}), 1}, false);
        }
        stream->setPPInflight();
        ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());
        ASSERT_EQ(recorded->inputs.size(), 1u);
        EXPECT_EQ(recorded->has_input, std::vector<bool>{false});
        EXPECT_EQ(recorded->has_output, std::vector<bool>{true});
        EXPECT_EQ(recorded->inputs[0].sequence_lengths.numel(), decode ? 1 : 0);
        EXPECT_FALSE(recorded->inputs[0].is_target_verify);
        ASSERT_EQ(wire->sent_tensors.size(), 5u);
        const auto plan = pp_serialization::deserializePlan(wire->sent_tensors.at(1));
        EXPECT_EQ(tensorToVector<int32_t>(plan.sampling_plan.token_ids),
                  decode ? (std::vector<int32_t>{1, 2, 7}) : (std::vector<int32_t>{1, 2}));
        EXPECT_EQ(executor.token_history_lengths_.at(101), stream->seqLength());
        EXPECT_TRUE(torch::equal(wire->sent_tensors.back(), recorded->hidden_buffer.narrow(1, 0, 2)));
        EXPECT_EQ(wire->receive_index, 0u);
        EXPECT_TRUE(stream->isPPInflight());
        EXPECT_TRUE(executor.sampling_states_.empty());
    }
}

TEST_F(PPExecutorTest, FirstEmptyPlansDrainEachBatchOnceAcrossSlotWraparound) {
    for (int pp_size : {2, 3, 4}) {
        for (int batches : {1, 7}) {
            SCOPED_TRACE(testing::Message() << "pp_size=" << pp_size << ", batches=" << batches);
            auto params = makeStageParams(0);
            params.model_config_.num_layers = pp_size;
            params.parallelism_config.pp_size = pp_size;
            params.parallelism_config.world_size = pp_size;
            params.parallelism_config.pp_stage_layer_counts.assign(pp_size, 1);
            PPExecutor executor(params, nullptr, false);
            enableMetrics(executor);
            auto* wire = attachTransport(executor);
            auto* recorded = installModel<RecordingDraftModel>(executor);
            std::vector<GenerateStreamPtr> streams;
            for (int index = 0; index < batches; ++index) {
                auto stream = makeStream(ResourceContext{}, params.model_config_, 101 + index, {1, 2}, 1);
                streams.push_back(stream);
                const auto result = makeSingleTokenResult(101 + index, 20 + index);
                wire->enqueueObject(pp_serialization::serializeExecutionResult(result));
            }
            const auto start = autil::TimeUtility::currentTimeInMicroSeconds();
            /** After the real batches, ordinary empty plans alone must drain results, including a single request. */
            for (int round = 0; round < batches + pp_size; ++round) {
                SCOPED_TRACE(round);
                ScheduleOutput schedule;
                if (round < batches) {
                    streams[round]->setPPInflight();
                    schedule.streams = {streams[round]};
                }
                const auto slot = executor.current_slot_;
                const auto sends = wire->sent_tensors.size();
                ASSERT_TRUE(executor.process(schedule, start + round).ok());
                EXPECT_EQ(executor.current_slot_, (slot + 1) % executor.slots_.size());
                const auto& admitted = executor.slots_[slot];
                EXPECT_EQ(admitted.schedule_time_us, start + round);
                if (round >= batches) {
                    ASSERT_EQ(wire->sent_tensors.size(), sends + 2);
                    const auto empty_plan = pp_serialization::deserializePlan(wire->sent_tensors[sends + 1]);
                    EXPECT_TRUE(empty_plan.model_input.skip_run);
                    EXPECT_FALSE(empty_plan.model_input.is_fake_stream);
                    EXPECT_FALSE(empty_plan.model_input.shutdown);
                    EXPECT_TRUE(empty_plan.finished_request_ids.empty());
                    EXPECT_FALSE(admitted.result_pending);
                }
                EXPECT_EQ(recorded->inputs.size(), std::min(round + 1, batches));
                const int returned = std::clamp(round - pp_size + 1, 0, batches);
                const int pending  = std::min(round + 1, batches) - returned;
                EXPECT_EQ(executor.tps_reporter_.active_count_, pending);
                EXPECT_EQ(executor.wall_tps_reporter_.active_count_, pending);
                EXPECT_EQ(wire->receive_index, 2u * returned);
                for (int index = 0; index < std::min(round + 1, batches); ++index) {
                    EXPECT_EQ(streams[index]->isPPInflight(), index >= returned);
                    EXPECT_EQ(streams[index]->seqLength(), index < returned ? 3 : 2);
                    if (index < returned) {
                        EXPECT_EQ(streams[index]->completeTokenIdsVec(0).back(), 20 + index);
                    }
                }
            }
            EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
            for (const auto& batch : executor.slots_) {
                EXPECT_FALSE(batch.result_pending);
            }
        }
    }
}

TEST_F(PPExecutorTest, FirstCancelledAndTimedOutBatchesStillConsumeTheirResult) {
    for (auto code : {ErrorCode::CANCELLED, ErrorCode::GENERATE_TIMEOUT}) {
        SCOPED_TRACE(static_cast<int>(code));
        auto params = makeStageParams(0);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        executor.setModel(std::make_unique<RecordingDraftModel>());
        auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1);
        stream->setPPInflight();
        ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());
        stream->reportError(code, "ended while in flight");
        wire->enqueueObject(pp_serialization::serializeExecutionResult(makeSingleTokenResult(101, 20)));
        for (int round = 0; round < params.parallelism_config.pp_size; ++round) {
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        }
        EXPECT_EQ(wire->receive_index, 2u);
        EXPECT_FALSE(stream->isPPInflight());
        EXPECT_EQ(stream->statusInfo().code(), code);
        EXPECT_EQ(stream->seqLength(), 2);
    }
}

TEST_F(PPExecutorTest, FirstShutdownDrainsOnlyPreviouslySubmittedBatches) {
    for (int pp_size : {2, 3, 4}) {
        /** Dense, mixed and fake-only histories exercise results separated by skip_run slots. */
        for (int pattern : {0, 1, 2}) {
            for (int rounds = 0; rounds <= 2 * (pp_size + 1); ++rounds) {
                SCOPED_TRACE(testing::Message() << "pp=" << pp_size << ", pattern=" << pattern
                                               << ", rounds=" << rounds);
                auto params = makeStageParams(0);
                params.model_config_.num_layers = pp_size;
                params.parallelism_config.pp_size = pp_size;
                params.parallelism_config.world_size = pp_size;
                params.parallelism_config.pp_stage_layer_counts.assign(pp_size, 1);
                ResourceContext resources;
                resources.cache_manager = std::make_shared<KVCacheManager>(
                    test::makeSimpleMhaCacheConfig(pp_size, 16, 4, DataType::TYPE_FP16));
                ASSERT_TRUE(resources.cache_manager->init());
                PPExecutor executor(params, resources.cache_manager, false);
                auto* wire = attachTransport(executor);
                auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
                std::vector<GenerateStreamPtr> streams;
                std::vector<int32_t> expected_tokens;
                for (int round = 0; round < rounds; ++round) {
                    ScheduleOutput schedule;
                    const bool empty = pattern == 1 && round % 4 == 1;
                    const bool fake = pattern == 2 || (pattern == 1 && round % 4 == 2);
                    if (!empty) {
                        PPExecutionResult result;
                        if (fake) {
                            schedule.streams = {makeFakeStream(params, resources)};
                        } else {
                            auto stream = makeStream(resources, params.model_config_, 101 + round, {1, 2}, 1);
                            stream->fakeInitKVBlock(4);
                            stream->setPPInflight();
                            schedule.streams = {stream};
                            streams.push_back(stream);
                            expected_tokens.push_back(20 + round);
                            result = makeSingleTokenResult(101 + round, 20 + round);
                        }
                        wire->enqueueObject(pp_serialization::serializeExecutionResult(result));
                    }
                    ASSERT_TRUE(executor.process(schedule).ok());
                    EXPECT_FALSE(executor.stopped_);
                }

                /** stop() can cancel streams before their results arrive; matching receives are still required. */
                for (size_t i = 0; i < streams.size(); ++i) {
                    if (i % 2 == 0 && streams[i]->isPPInflight()) {
                        streams[i]->reportError(ErrorCode::CANCELLED, "cancelled before shutdown drain");
                    }
                }
                const auto sends = wire->sent_tensors.size();
                const auto forwards = recorded->inputs.size();
                const auto slot = executor.current_slot_;
                wire->on_receive = [&](const torch::Tensor&) {
                    /** Downstream ranks must receive the terminal plan before the root drains their results. */
                    ASSERT_EQ(wire->sent_tensors.size(), sends + 2);
                    const auto terminal = pp_serialization::deserializePlan(wire->sent_tensors[sends + 1]);
                    EXPECT_TRUE(terminal.model_input.shutdown);
                    EXPECT_TRUE(terminal.model_input.skip_run);
                };
                ASSERT_TRUE(executor.finish().ok());
                EXPECT_TRUE(executor.token_history_lengths_.empty());
                EXPECT_TRUE(executor.stopped_);
                /** No terminal result is queued: extra or duplicate receives fail the transport. */
                EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
                ASSERT_EQ(wire->sent_tensors.size(), sends + 2);
                const auto terminal = pp_serialization::deserializePlan(wire->sent_tensors[sends + 1]);
                EXPECT_TRUE(terminal.model_input.shutdown);
                EXPECT_TRUE(terminal.model_input.skip_run);
                EXPECT_FALSE(terminal.model_input.is_fake_stream);
                EXPECT_EQ(recorded->inputs.size(), forwards);
                EXPECT_EQ(executor.current_slot_, (slot + 1) % executor.slots_.size());
                const auto stopped_slot = executor.current_slot_;
                ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                EXPECT_EQ(executor.current_slot_, stopped_slot);
                EXPECT_EQ(wire->sent_tensors.size(), sends + 2);
                EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
                for (size_t i = 0; i < streams.size(); ++i) {
                    EXPECT_FALSE(streams[i]->isPPInflight());
                    if (streams[i]->statusInfo().code() == ErrorCode::CANCELLED) {
                        EXPECT_EQ(streams[i]->seqLength(), 2);
                    } else {
                        EXPECT_EQ(streams[i]->seqLength(), 3);
                        EXPECT_EQ(streams[i]->completeTokenIdsVec(0).back(), expected_tokens[i]);
                    }
                }
                expectSlotsDrained(executor);
            }
        }
    }
}

TEST_F(PPExecutorTest, FirstExplicitDrainPreservesOrderAcrossWrapsAndServingOverlap) {
    for (int pp_size : {2, 3, 4}) {
        SCOPED_TRACE(pp_size);
        auto params = makeStageParams(0);
        params.model_config_.num_layers = pp_size;
        params.parallelism_config.pp_size = pp_size;
        params.parallelism_config.world_size = pp_size;
        params.parallelism_config.pp_stage_layer_counts.assign(pp_size, 1);
        ResourceContext resources;
        resources.cache_manager = std::make_shared<KVCacheManager>(
            test::makeSimpleMhaCacheConfig(pp_size, 16, 4, DataType::TYPE_FP16));
        ASSERT_TRUE(resources.cache_manager->init());
        PPExecutor executor(params, resources.cache_manager, false);
        auto* wire = attachTransport(executor);
        executor.setModel(std::make_unique<RecordingStageModel>(params.parallelism_config));
        std::vector<GenerateStreamPtr> streams;
        std::vector<int32_t> expected_tokens;

        auto submit = [&](int round, int kind, bool synchronous) {
            ScheduleOutput schedule;
            if (kind != 0) {
                PPExecutionResult result;
                if (kind == 1) {
                    schedule.streams = {makeFakeStream(params, resources)};
                } else {
                    auto stream = makeStream(resources, params.model_config_, 101 + round, {1, 2}, 1);
                    stream->fakeInitKVBlock(4);
                    stream->setPPInflight();
                    schedule.streams = {stream};
                    streams.push_back(stream);
                    expected_tokens.push_back(20 + round);
                    result = makeSingleTokenResult(101 + round, 20 + round);
                }
                wire->enqueueObject(pp_serialization::serializeExecutionResult(result));
            } else {
                schedule.finished_request_ids = {100 + round};
            }
            const auto sends = wire->sent_tensors.size();
            const auto slot = executor.current_slot_;
            ASSERT_TRUE(executor.process(schedule).ok());
            if (synchronous) {
                ASSERT_TRUE(executor.drainPendingResults().ok());
            }
            /** Waiting never submits pump plans or advances the ring beyond the single submitted plan. */
            EXPECT_EQ(wire->sent_tensors.size() - sends, kind == 0 ? 2u : 5u);
            EXPECT_EQ(executor.current_slot_, (slot + 1) % executor.slots_.size());
            if (synchronous) {
                EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
                for (const auto& batch : executor.slots_) {
                    EXPECT_FALSE(batch.result_pending);
                }
                /** Waiting for results does not retire the new slot's asynchronous sends. */
                EXPECT_FALSE(executor.slots_[slot].plan_sends.empty());
                for (const auto& stream : streams) {
                    EXPECT_FALSE(stream->isPPInflight());
                }
            }
        };

        const int rounds = 4 * (pp_size + 1);
        for (int round = 0; round < rounds; ++round) {
            submit(round, round % 3, round % (pp_size + 1) == pp_size - 1);
        }
        submit(rounds, 2, true);
        /** After explicitly draining pending results, normal serving must still retain its pipeline overlap. */
        const auto receives = wire->receive_index;
        for (int round = 0; round < pp_size; ++round) {
            submit(rounds + 1 + round, 2, false);
            EXPECT_TRUE(streams.back()->isPPInflight());
            EXPECT_EQ(wire->receive_index, receives);
        }
        ASSERT_TRUE(executor.finish().ok());
        EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
        for (const auto& batch : executor.slots_) {
            EXPECT_FALSE(batch.result_pending);
        }
        for (size_t i = 0; i < streams.size(); ++i) {
            EXPECT_FALSE(streams[i]->isPPInflight());
            EXPECT_EQ(streams[i]->completeTokenIdsVec(0), (std::vector<int>{1, 2, expected_tokens[i]}));
        }
    }
}

TEST_F(PPExecutorTest, FirstConsumedInvalidResultIsNotReceivedAgain) {
    auto params = makeStageParams(0);
    PPExecutor executor(params, nullptr, false);
    enableMetrics(executor);
    auto* wire = attachTransport(executor);
    auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1);
    auto& batch = executor.slots_[0];
    batch.result_pending        = true;
    batch.stream_groups         = StreamGroups({stream});
    batch.tps_active_guard      = executor.tps_reporter_.makeActiveGuard();
    batch.wall_tps_active_guard = executor.wall_tps_reporter_.makeActiveGuard();
    ASSERT_EQ(executor.tps_reporter_.active_count_, 1);
    ASSERT_EQ(executor.wall_tps_reporter_.active_count_, 1);
    stream.reset();
    /** A deserialized but invalid result has already consumed its wire message. */
    wire->enqueueObject(pp_serialization::serializeExecutionResult(PPExecutionResult{}));
    EXPECT_THROW((void)executor.drainPendingResults(), std::exception);
    EXPECT_FALSE(batch.result_pending);
    EXPECT_EQ(executor.tps_reporter_.active_count_, 0);
    EXPECT_EQ(executor.wall_tps_reporter_.active_count_, 0);
    EXPECT_EQ(wire->receive_index, 2u);
    EXPECT_TRUE(executor.drainPendingResults().ok());
    EXPECT_EQ(wire->receive_index, 2u);
}

TEST_F(PPExecutorTest, DownstreamFinishProcessesOutstandingPlansUntilShutdown) {
    for (int stage : {1, 2}) {
        SCOPED_TRACE(stage);
        auto params = makeStageParams(stage);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
        enqueuePlan(*wire, makePlan(params));
        PPExecutionPlan cleanup;
        cleanup.model_input.skip_run = true;
        cleanup.finished_request_ids = {101, 202};
        enqueuePlan(*wire, cleanup);
        enqueuePlan(*wire, makeShutdownPlan());

        /** Local stop can arrive before the upstream terminal plan. finish must still forward outstanding work. */
        ASSERT_TRUE(executor.finish().ok());
        EXPECT_TRUE(executor.stopped_);
        EXPECT_EQ(recorded->inputs.size(), 1u);
        EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
        EXPECT_TRUE(executor.sampling_states_.empty());
        if (stage == 1) {
            ASSERT_EQ(wire->sent_tensors.size(), 9u);
            EXPECT_FALSE(pp_serialization::deserializePlan(wire->sent_tensors[1]).model_input.shutdown);
            const auto forwarded_cleanup = pp_serialization::deserializePlan(wire->sent_tensors[6]);
            EXPECT_EQ(forwarded_cleanup.finished_request_ids, cleanup.finished_request_ids);
            EXPECT_FALSE(forwarded_cleanup.model_input.shutdown);
            EXPECT_TRUE(pp_serialization::deserializePlan(wire->sent_tensors[8]).model_input.shutdown);
        } else {
            ASSERT_EQ(wire->sent_tensors.size(), 2u);
            EXPECT_EQ(tensorToVector<int64_t>(lastResult(*wire).request_ids), (std::vector<int64_t>{101, 202}));
        }
        expectSlotsDrained(executor);
        const auto slot = executor.current_slot_;
        const auto sends = wire->sent_tensors.size();
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_EQ(executor.current_slot_, slot);
        EXPECT_EQ(recorded->inputs.size(), 1u);
        EXPECT_EQ(wire->sent_tensors.size(), sends);
        EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
    }
}

TEST_F(PPExecutorTest, FirstEmptyExecutionStillDispatchesThePreviousBatchOutsideProfile) {
    for (bool skip_run : {false, true}) {
        SCOPED_TRACE(skip_run);
        auto params           = makeMtpParams(3);
        params.sp_config.type = SP_TYPE_NONE;
        PPExecutor executor(params, nullptr, false);
        auto current  = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 0);
        auto previous = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4}, 0);
        previous->setPPInflight();
        executor.slots_[1].result_pending = true;
        executor.slots_[1].stream_groups = StreamGroups({previous});

        const auto payload = pp_serialization::serializeExecutionResult(makeSingleTokenResult(202, 17));
        auto transport     = std::make_unique<InMemoryPPTransport>();
        auto* wire         = transport.get();
        wire->enqueueObject(payload);
        executor.transport_ = std::move(transport);

        bool profiling               = false;
        bool finished                = false;
        executor.profile_step_start_ = [&]() {
            EXPECT_FALSE(skip_run);
            EXPECT_EQ(wire->receive_index, 0u);
            EXPECT_TRUE(previous->isPPInflight());
            profiling = true;
        };
        executor.profile_step_finish_ = [&]() {
            EXPECT_FALSE(skip_run);
            EXPECT_TRUE(profiling);
            EXPECT_EQ(wire->receive_index, 0u);
            EXPECT_TRUE(previous->isPPInflight());
            profiling = false;
            finished  = true;
        };
        auto target        = std::make_unique<RecordingDraftModel>();
        target->on_forward = [&]() { EXPECT_TRUE(profiling); };
        executor.setModel(std::move(target));

        ScheduleOutput scheduled;
        if (!skip_run) {
            scheduled.streams = {current};
        }
        ASSERT_TRUE(executor.process(scheduled).ok());
        EXPECT_EQ(finished, !skip_run);
        EXPECT_FALSE(profiling);
        EXPECT_EQ(wire->receive_index, 2u);
        EXPECT_FALSE(previous->isPPInflight());
        EXPECT_EQ(previous->completeTokenIdsVec(0), (std::vector<int>{3, 4, 17}));
    }
}

/** First: PD handoff, request validation and TP ownership. */
TEST_F(PPExecutorTest, FirstPdDecodeUsesPreparedCandidatesWithoutOverwritingThem) {
    for (auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        SCOPED_TRACE(type);
        auto params = makeStageParams(0, type);
        params.pd_sep_config.role_type = RoleType::DECODE;
        ResourceContext resources;
        resources.role_type = RoleType::DECODE;
        resources.cache_manager = std::make_shared<KVCacheManager>(
            test::makeSimpleMhaCacheConfig(3, 16, 4, DataType::TYPE_FP16));
        ASSERT_TRUE(resources.cache_manager->init());
        PPExecutor executor(params, resources.cache_manager, false);
        attachTransport(executor);
        auto* recorded = installModel<RecordingDraftModel>(executor);
        auto stream = makeStream(resources, params.model_config_, 101, {1, 2}, 1);
        stream->fakeInitKVBlock(4);
        stream->update({intTensor({7}).reshape({1, 1}), 1}, false);
        stream->initSpeculativeHandoffPositions();
        auto buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
        buffer->propose_step = 3;
        buffer->tokens = intTensor({7, type == SP_TYPE_DSPARK ? 0 : 11, 0, 0}).reshape({1, 4});
        stream->setSPOutputBuffer(buffer);
        stream->setProposeToken(tensorToVector<int32_t>(buffer->tokens));
        const auto before = buffer->tokens.clone();
        stream->setPPInflight();
        ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());
        ASSERT_EQ(recorded->inputs.size(), 1u);
        EXPECT_TRUE(recorded->inputs[0].is_target_verify);
        EXPECT_TRUE(torch::equal(recorded->inputs[0].combo_tokens.cpu(), before.flatten()));
        EXPECT_EQ(stream->getSPOutputBuffer(), buffer);
        EXPECT_TRUE(torch::equal(buffer->tokens, before));
        EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 7}));
    }
}

TEST_F(PPExecutorTest, FirstAdmissionRejectsBeamSearchBeforeForward) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP}) {
        SCOPED_TRACE(type);
        auto params = makeStageParams(0, type);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto* recorded = installModel<RecordingDraftModel>(executor);
        std::vector<GenerateStreamPtr> rejected;
        for (bool variable : {false, true}) {
            GenerateConfig config;
            config.max_new_tokens = 20;
            if (variable) {
                config.variable_num_beams = {1, 2};
            } else {
                config.num_beams = 2;
            }
            auto stream = makeStream(ResourceContext{}, params.model_config_, variable ? 202 : 101, {1, 2}, config);
            stream->setPPInflight();
            rejected.push_back(stream);
        }
        auto healthy = makeStream(ResourceContext{}, params.model_config_, 303, {3, 4}, 1);
        healthy->setPPInflight();

        ASSERT_TRUE(executor.process(ScheduleOutput{{rejected[0], rejected[1], healthy}}).ok());

        for (const auto& stream : rejected) {
            EXPECT_TRUE(stream->hasError());
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_NE(stream->statusInfo().ToString().find("beam search"), std::string::npos);
            EXPECT_FALSE(stream->isPPInflight());
            EXPECT_FALSE(stream->getSPOutputBuffer());
        }
        EXPECT_FALSE(healthy->hasError());
        EXPECT_TRUE(healthy->isPPInflight());
        ASSERT_EQ(recorded->inputs.size(), 1u);
        const auto plan = pp_serialization::deserializePlan(wire->sent_tensors.at(1));
        EXPECT_EQ(tensorToVector<int64_t>(plan.sampling_plan.request_ids), std::vector<int64_t>{303});
    }
}

TEST_F(PPExecutorTest, FirstAdmissionRejectsLossWithoutDiscardingHealthyRequests) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        SCOPED_TRACE(type);
        auto params = makeStageParams(0, type);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto* recorded = installModel<RecordingDraftModel>(executor);
        std::list<GenerateStreamPtr> streams;
        for (int loss_mode : {1, 2}) {
            GenerateConfig config;
            config.max_new_tokens = 20;
            config.calculate_loss = loss_mode;
            auto stream = makeStream(ResourceContext{}, params.model_config_, 100 + loss_mode, {1, 2}, config);
            stream->setPPInflight();
            streams.push_back(stream);
        }
        auto healthy = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4}, 1);
        healthy->setPPInflight();
        streams.push_back(healthy);

        ASSERT_TRUE(executor.process(ScheduleOutput{streams}).ok());

        for (const auto& stream : streams) {
            if (stream == healthy) {
                continue;
            }
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_NE(stream->statusInfo().ToString().find("calculate_loss"), std::string::npos);
            EXPECT_FALSE(stream->isPPInflight());
            EXPECT_FALSE(stream->getSPOutputBuffer());
        }
        EXPECT_FALSE(healthy->hasError());
        EXPECT_TRUE(healthy->isPPInflight());
        ASSERT_EQ(recorded->inputs.size(), 1u);
        const auto plan = pp_serialization::deserializePlan(wire->sent_tensors.at(1));
        EXPECT_EQ(tensorToVector<int64_t>(plan.sampling_plan.request_ids), std::vector<int64_t>{202});
        EXPECT_FALSE(plan.output_config.calculate_loss);
    }
}

TEST_F(PPExecutorTest, FirstAdmissionAllowsOptionalOutputsOnlyWithoutSpeculation) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (bool prompt_logits : {false, true}) {
            SCOPED_TRACE(testing::Message() << "type=" << type << ", prompt_logits=" << prompt_logits);
            auto params = makeStageParams(0, type);
            PPExecutor executor(params, nullptr, false);
            auto* wire = attachTransport(executor);
            auto* recorded = installModel<RecordingDraftModel>(executor);
            GenerateConfig config;
            config.max_new_tokens = 20;
            config.return_prompt_logits = prompt_logits;
            config.return_softmax_probs = !prompt_logits;
            auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, config);
            stream->setPPInflight();

            ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());

            const auto plan = pp_serialization::deserializePlan(wire->sent_tensors.at(1));
            if (type == SP_TYPE_NONE) {
                EXPECT_FALSE(stream->hasError());
                EXPECT_TRUE(stream->isPPInflight());
                EXPECT_FALSE(plan.model_input.skip_run);
                EXPECT_EQ(recorded->inputs.size(), 1u);
                EXPECT_EQ(plan.output_config.return_softmax_probs, !prompt_logits);
                ASSERT_EQ(plan.output_config.requests.size(), 1u);
                EXPECT_EQ(plan.output_config.requests[0].prompt_logits.enabled, prompt_logits);
            } else {
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
                EXPECT_NE(stream->statusInfo().ToString().find(
                              prompt_logits ? "return_prompt_logits" : "return_softmax_probs"),
                          std::string::npos);
                EXPECT_FALSE(stream->isPPInflight());
                EXPECT_FALSE(stream->getSPOutputBuffer());
                EXPECT_TRUE(plan.model_input.skip_run);
                EXPECT_TRUE(recorded->inputs.empty());
                EXPECT_FALSE(executor.slots_[0].result_pending);
                ASSERT_TRUE(executor.drainPendingResults().ok());
                EXPECT_EQ(wire->receive_index, 0u);
            }
        }
    }
}

TEST_F(PPExecutorTest, FirstAdmissionRejectsTiledSamplingWithSpeculation) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        SCOPED_TRACE(type);
        auto params = makeStageParams(0, type);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto* recorded = installModel<RecordingDraftModel>(executor);
        auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 2);
        stream->setPPInflight();

        ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());

        const auto plan = pp_serialization::deserializePlan(wire->sent_tensors.at(1));
        if (type != SP_TYPE_NONE) {
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_NE(stream->statusInfo().ToString().find(
                          type == SP_TYPE_DSPARK ? "tiled sampling" : "incompatible with logits processor"),
                      std::string::npos);
            EXPECT_FALSE(stream->isPPInflight());
            EXPECT_FALSE(stream->getSPOutputBuffer());
            EXPECT_TRUE(plan.model_input.skip_run);
            EXPECT_TRUE(recorded->inputs.empty());
        } else {
            EXPECT_FALSE(stream->hasError());
            EXPECT_TRUE(stream->isPPInflight());
            EXPECT_EQ(recorded->inputs.size(), 1u);
            EXPECT_FALSE(plan.model_input.skip_run);
            EXPECT_EQ(plan.sampling_plan.num_return_sequences, std::vector<int32_t>{2});
        }
    }
}

TEST_F(PPExecutorTest, FirstDecodeRequiresValidProposalStateBeforeSendingAPlan) {
    for (auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (bool missing : {false, true}) {
            SCOPED_TRACE(testing::Message() << "type=" << type << ", missing=" << missing);
            auto params = makeStageParams(0, type);
            PPExecutor executor(params, nullptr, false);
            auto* wire = attachTransport(executor);
            auto* recorded = installModel<RecordingDraftModel>(executor);
            auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1);
            stream->setIsContextStream(false);
            if (!missing) {
                auto buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
                /** K=3 requires an anchor plus three draft tokens; this buffer is one token short. */
                buffer->tokens = intTensor({7, 11, 12});
                stream->setSPOutputBuffer(buffer);
            }
            try {
                (void)executor.process(ScheduleOutput{{stream}});
                FAIL() << "decode accepted invalid proposal state";
            } catch (const std::runtime_error& error) {
                EXPECT_NE(std::string(error.what()).find(
                              missing ? "requires proposal tokens" : "token buffer must contain"),
                          std::string::npos)
                    << error.what();
            }
            EXPECT_FALSE(stream->hasError());
            EXPECT_TRUE(recorded->inputs.empty());
            EXPECT_TRUE(wire->sent_tensors.empty());
        }
    }
}

TEST_F(PPExecutorTest, FirstSpAdmissionFiltersUnsupportedRequestsBeforeForward) {
    for (bool all_failed : {false, true}) {
        SCOPED_TRACE(all_failed);
        auto params = makeStageParams(0, SP_TYPE_MTP);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto* recorded = installModel<RecordingDraftModel>(executor);
        auto failed = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1);
        auto healthy = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4}, 1);
        auto unsupported = std::make_shared<RecordingLogitsProcessor>();
        unsupported->mtp_supported = false;
        failed->sampling_state_.logits_processors = {unsupported};
        if (all_failed) {
            healthy->sampling_state_.logits_processors = {unsupported};
        }
        for (auto stream : {failed, healthy}) {
            stream->setPPInflight();
        }
        const ScheduleOutput schedule{{failed, healthy}};
        ASSERT_TRUE(executor.process(schedule).ok());
        if (all_failed) {
            ASSERT_TRUE(executor.drainPendingResults().ok());
        }
        EXPECT_TRUE(failed->hasError());
        EXPECT_FALSE(failed->isPPInflight());
        EXPECT_EQ(healthy->hasError(), all_failed);
        EXPECT_EQ(healthy->isPPInflight(), !all_failed);
        EXPECT_EQ(recorded->inputs.size(), all_failed ? 0u : 1u);
        const auto sent_plan = pp_serialization::deserializePlan(wire->sent_tensors.at(1));
        EXPECT_EQ(sent_plan.model_input.skip_run, all_failed);
        EXPECT_EQ(executor.slots_[0].result_pending, !all_failed);
        EXPECT_EQ(wire->receive_index, 0u);
        if (!all_failed) {
            EXPECT_EQ(tensorToVector<int64_t>(sent_plan.sampling_plan.request_ids), std::vector<int64_t>{202});
            EXPECT_NE(healthy->getSPOutputBuffer(), nullptr);
        }
    }
}

TEST_F(PPExecutorTest, FirstPrefillOnlySkipsSamplingCompatibilityAndDrainsItsResult) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        SCOPED_TRACE(type);
        auto params = makeStageParams(0, type);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto* model = installModel<RecordingDraftModel>(executor);
        auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1, 0);
        auto unsupported = std::make_shared<RecordingLogitsProcessor>();
        unsupported->mtp_supported = false;
        stream->sampling_state_.logits_processors = {unsupported};
        stream->setPPInflight();
        PPExecutionResult result;
        result.request_ids = torch::tensor({101}, torch::kInt64);
        result.new_token_ids = torch::zeros({1, 1}, torch::kInt32);
        result.new_token_lengths = intTensor({1});
        result.request_errors.resize(1);
        result.request_outputs.resize(1);
        wire->enqueueObject(pp_serialization::serializeExecutionResult(result));

        ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());
        ASSERT_EQ(model->inputs.size(), 1u);
        EXPECT_TRUE(model->inputs[0].skip_lm_head);
        EXPECT_FALSE(model->inputs[0].capture_hidden_states);
        const auto plan = pp_serialization::deserializePlan(wire->sent_tensors.at(1));
        EXPECT_TRUE(plan.model_input.skip_lm_head);
        EXPECT_FALSE(plan.sampling_plan.token_ids.defined());
        EXPECT_FALSE(plan.sampling_plan.token_offsets.defined());
        EXPECT_TRUE(executor.token_history_lengths_.empty());
        EXPECT_FALSE(stream->getSPOutputBuffer());
        EXPECT_TRUE(stream->getProposeToken().empty());
        ASSERT_TRUE(executor.drainPendingResults().ok());
        EXPECT_FALSE(stream->hasError());
        EXPECT_FALSE(stream->isPPInflight());
        EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        EXPECT_TRUE(unsupported->committed_tokens.empty());
        EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2}));
        EXPECT_FALSE(executor.slots_[0].result_pending);
        ASSERT_TRUE(executor.finish().ok());
        expectSlotsDrained(executor);
    }
}

TEST_F(PPExecutorTest, NonLastStagesRetainSpeculativeModeWithoutOwningDraftModel) {
    for (const auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (int stage : {0, 1}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << " stage=" << stage);
            auto params = makeStageParams(stage, type);
            std::vector<size_t> constructed_models;
            PPExecutor::test_model_factory = [&constructed_models](const GptModelInitParams& init_params) {
                constructed_models.push_back(init_params.model_id);
                return std::make_unique<RecordingDraftModel>();
            };

            PPExecutor executor(params, nullptr, false);

            EXPECT_EQ(constructed_models, (std::vector<size_t>{0}));
            EXPECT_TRUE(executor.sp_enabled_);
            EXPECT_TRUE(executor.batch_stream_processor_->sp_enabled_);
            EXPECT_EQ(executor.draft_model_, nullptr);
            EXPECT_EQ(executor.draft_commit_model_, nullptr);
        }
    }
}

TEST_F(PPExecutorTest, FirstBuildsFakePlansWithoutLocalProposeParameters) {
    for (const auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        for (const auto role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
            SCOPED_TRACE("type=" + std::to_string(type) + ", role=" + std::to_string(role));
            auto params                             = makeMtpParams(3);
            params.sp_config.type                    = type;
            params.sp_config.sp_dspark_mask_token_id = 63;
            params.pd_sep_config.role_type           = role;
            /** The first stage has no local draft model or proposer parameters. */
            auto cache_manager = std::make_shared<KVCacheManager>(
                test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
            ASSERT_TRUE(cache_manager->init());
            PPExecutor executor(params, cache_manager, false);
            ResourceContext resources;
            resources.cache_manager = cache_manager;
            resources.role_type     = role;
            const bool decode   = role == RoleType::DECODE;
            const auto stream   = decode ?
                                      PPExecutor::createMinFakeDecodeStream(params.model_config_,
                                                                            params.runtime_config,
                                                                            resources,
                                                                            params.sp_config) :
                                      PPExecutor::createMinFakePrefillStream(params.model_config_,
                                                                             params.runtime_config,
                                                                             resources,
                                                                             params.sp_config,
                                                                             role);
            ASSERT_TRUE(stream->isFakeStream());
            EXPECT_EQ(stream->isContextStream(), !decode);
            const auto history = stream->completeTokenIdsVec(0);
            std::list<GenerateStreamPtr> streams{stream};
            executor.prepareStreams(streams);
            auto plan_status = executor.buildPlan(StreamGroups(streams), {}, false);
            ASSERT_TRUE(plan_status.ok()) << plan_status.status().ToString();
            const auto& plan = plan_status.value();
            EXPECT_TRUE(plan.model_input.is_fake_stream);
            EXPECT_FALSE(plan.sampling_plan.token_ids.defined());
            EXPECT_FALSE(plan.sampling_plan.token_offsets.defined());
            EXPECT_TRUE(executor.token_history_lengths_.empty());
            EXPECT_FALSE(plan.model_input.skip_run);
            EXPECT_EQ(plan.is_decode, decode);
            EXPECT_EQ(plan.model_input.is_target_verify, decode && type != SP_TYPE_NONE);
            const int64_t token_count = decode && type != SP_TYPE_NONE ? 4 : 1;
            EXPECT_EQ(plan.model_input.combo_tokens.numel(), token_count);
            EXPECT_EQ(stream->completeTokenIdsVec(0), history);
        }
    }
}

TEST_F(PPExecutorTpTest, ShutdownPlanStopsEveryTpRankBeforeOrDuringFinishWithoutExecution) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP}) {
        for (bool terminal_before_finish : {false, true}) {
            for (int stage : {0, 1, 2}) {
                ReplayTpBroadcast broadcasts;
                for (int rank : {0, 1}) {
                    SCOPED_TRACE(testing::Message() << "type=" << type << ", stage=" << stage << ", tp=" << rank
                                                   << ", terminal_before_finish=" << terminal_before_finish);
                    if (rank != 0) {
                        broadcasts.replay();
                    }
                    auto params = makeTpStageParams(stage, rank, type);
                    auto propose = makeMtpProposeParams(params);
                    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr,
                                        stage == 2 && type == SP_TYPE_MTP ? propose.get() : nullptr);
                    auto* wire = attachTransport(executor);
                    auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
                    RecordingDraftModel* recorded_draft = nullptr;
                    if (stage == 2 && type == SP_TYPE_MTP) {
                        auto draft = std::make_unique<RecordingDraftModel>();
                        recorded_draft = draft.get();
                        executor.draft_model_ = std::move(draft);
                    }
                    if (stage == 0 && rank == 0) {
                        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                        EXPECT_FALSE(executor.stopped_);
                    } else {
                        if (stage != 0) {
                            PPExecutionPlan empty;
                            empty.model_input.skip_run = true;
                            enqueuePlan(*wire, empty, rank != 0);
                            auto terminal = makeShutdownPlan();
                            /** Forwarded metadata keeps the first-stage peer's value until local TP sync. */
                            terminal.model_input.shutdown = rank == 0;
                            enqueuePlan(*wire, terminal, rank != 0);
                        }
                        if (terminal_before_finish) {
                            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                            EXPECT_FALSE(executor.stopped_);
                            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                            EXPECT_TRUE(executor.stopped_);
                            const auto slot = executor.current_slot_;
                            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                            EXPECT_EQ(executor.current_slot_, slot);
                        }
                    }
                    ASSERT_TRUE(executor.finish().ok());
                    EXPECT_TRUE(executor.stopped_);
                    EXPECT_TRUE(recorded->inputs.empty());
                    EXPECT_EQ(executor.current_slot_, 2u);
                    EXPECT_EQ(wire->receive_index, stage == 0 ? 0u : 4u);
                    ASSERT_EQ(wire->sent_tensors.size(), stage == 2 ? 0u : 4u);
                    /** TP sync propagates shutdown locally; forwarding preserves each rank's original header value. */
                    if (stage != 2) {
                        EXPECT_FALSE(pp_serialization::deserializePlan(wire->sent_tensors[1]).model_input.shutdown);
                        const auto terminal = pp_serialization::deserializePlan(wire->sent_tensors[3]);
                        EXPECT_EQ(terminal.model_input.shutdown, rank == 0);
                        if (rank == 0) {
                            EXPECT_TRUE(terminal.model_input.skip_run);
                            EXPECT_FALSE(terminal.model_input.is_fake_stream);
                        }
                    }
                    if (recorded_draft) {
                        EXPECT_TRUE(recorded_draft->inputs.empty());
                    }
                    EXPECT_TRUE(executor.sampling_states_.empty());
                    EXPECT_TRUE(executor.dspark_q_.empty());
                    for (const auto& batch : executor.slots_) {
                        EXPECT_FALSE(batch.result_pending);
                        EXPECT_TRUE(batch.plan_sends.empty());
                        EXPECT_TRUE(batch.activation_sends.empty());
                        EXPECT_TRUE(batch.execution_result_sends.empty());
                    }
                    if (rank != 0) {
                        EXPECT_EQ(broadcasts.position(), broadcasts.size());
                    }
                }
            }
        }
    }
}

TEST_F(PPExecutorTpTest, FirstTpPeerExecutesSyncedInputWithoutOwningRequests) {
    ReplayTpBroadcast broadcasts;
    torch::Tensor root_tokens;
    for (int rank : {0, 1}) {
        SCOPED_TRACE(rank);
        if (rank != 0) {
            broadcasts.replay();
        }
        auto params = makeTpStageParams(0, rank);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
        ScheduleOutput schedule;
        if (rank == 0) {
            schedule.streams = {makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1)};
        }
        ASSERT_TRUE(executor.process(schedule).ok());
        ASSERT_EQ(recorded->inputs.size(), 1u);
        EXPECT_EQ(wire->receive_index, 0u);
        EXPECT_EQ(wire->sent_tensors.size(), 5u);
        EXPECT_EQ(executor.slots_[0].stream_groups.size(), rank == 0 ? 1u : 0u);
        EXPECT_TRUE(executor.sampling_states_.empty());
        if (rank == 0) {
            root_tokens = recorded->inputs[0].combo_tokens.cpu().clone();
        } else {
            EXPECT_TRUE(torch::equal(recorded->inputs[0].combo_tokens.cpu(), root_tokens));
            EXPECT_EQ(broadcasts.position(), broadcasts.size());
        }
    }
}

TEST_F(PPExecutorTpTest, MiddleTpPeerKeepsUpstreamCacheTagsThroughPlanAndTpSync) {
    ReplayTpBroadcast broadcasts;
    for (int rank : {0, 1}) {
        SCOPED_TRACE(rank);
        if (rank != 0) {
            broadcasts.replay();
        }
        auto params = makeTpStageParams(1, rank);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto plan = makePlan(params);
        plan.model_input.kv_cache_group_tags = {"swa", "full"};
        plan.model_input.kv_cache_block_id = intTensor({11, 12, 21, 22}).reshape({2, 2, 1});
        plan.model_input.kv_cache_kernel_block_id = plan.model_input.kv_cache_block_id.clone();
        enqueuePlan(*wire, plan, rank != 0);
        auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);

        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        ASSERT_EQ(recorded->inputs.size(), 1u);
        EXPECT_EQ(recorded->inputs[0].kv_cache_group_tags, plan.model_input.kv_cache_group_tags);
        EXPECT_TRUE(torch::equal(recorded->inputs[0].kv_cache_block_id.cpu(), plan.model_input.kv_cache_block_id));
        const auto forwarded = pp_serialization::deserializePlan(wire->sent_tensors[1]);
        EXPECT_EQ(forwarded.model_input.kv_cache_group_tags, plan.model_input.kv_cache_group_tags);
        if (rank != 0) {
            EXPECT_FALSE(forwarded.model_input.combo_tokens.defined());
            EXPECT_EQ(broadcasts.position(), broadcasts.size());
        }
    }
}

/** A PP stage owns only part of the model's cache groups, so the [N,3] update mapping is
 * decoded by tag and rows whose tag is absent from the local topology are skipped. The raw
 * [N,3] tensor cannot go to the 2-column block-copy overload, which checks size(1) == 2. */
TEST_F(PPExecutorTest, MiddleStageDecodesCacheUpdateMappingByGroup) {
    auto            params = makeStageParams(/*pp_rank=*/1);
    ResourceContext resources;
    resources.cache_manager = std::make_shared<KVCacheManager>(
        test::makeSimpleMhaCacheConfig(/*layer_num=*/3, /*block_num=*/16, /*tokens_per_block=*/4, DataType::TYPE_FP16));
    ASSERT_TRUE(resources.cache_manager->init());
    PPExecutor executor(params, resources.cache_manager, false);
    auto*      wire = attachTransport(executor);

    auto plan = makePlan(params);
    // Payload carries "default" (owned by this stage) and "foreign" (owned by another stage).
    plan.model_input.kv_cache_group_tags      = {"default", "foreign"};
    plan.model_input.kv_cache_block_id        = intTensor({1, 2, 3, 4}).reshape({2, 2, 1});
    plan.model_input.kv_cache_kernel_block_id = plan.model_input.kv_cache_block_id.clone();
    // Row 0 -> tag row 0 ("default"): copy block 3 to 4. Row 1 -> tag row 1 ("foreign"): skipped.
    plan.model_input.kv_cache_update_mapping = intTensor({0, 3, 4, 1, 5, 6}).reshape({2, 3});

    enqueuePlan(*wire, plan);
    executor.setModel(std::make_unique<RecordingStageModel>(params.parallelism_config));

    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
}

/** Inflight batches retain metric ownership until their results are consumed. */
TEST_F(PPExecutorTest, FirstPendingBatchKeepsTpsActiveAcrossEmptyRoundsAndExplicitDrain) {
    for (int pp_size : {2, 3, 4}) {
        SCOPED_TRACE(pp_size);
        auto params                          = makeStageParams(0);
        params.model_config_.num_layers      = pp_size;
        params.parallelism_config.pp_size    = pp_size;
        params.parallelism_config.world_size = pp_size;
        params.parallelism_config.pp_stage_layer_counts.assign(pp_size, 1);
        PPExecutor executor(params, nullptr, false);
        enableMetrics(executor);
        auto* wire = attachTransport(executor);
        installModel<RecordingDraftModel>(executor);

        auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1);
        stream->setPPInflight();
        wire->enqueueObject(pp_serialization::serializeExecutionResult(makeSingleTokenResult(101, 20)));
        ScheduleOutput schedule;
        schedule.streams = {stream};
        ASSERT_TRUE(executor.process(schedule).ok());

        /** The reporters have no completed sample yet. Zero active count here would emit an idle zero. */
        auto expect_pending_active = [&]() {
            EXPECT_TRUE(stream->isPPInflight());
            EXPECT_FALSE(executor.tps_reporter_.collector_.hasMetrics());
            EXPECT_FALSE(executor.wall_tps_reporter_.collector_.hasMetrics());
            EXPECT_GT(executor.tps_reporter_.active_count_, 0);
            EXPECT_GT(executor.wall_tps_reporter_.active_count_, 0);
        };
        expect_pending_active();
        for (int round = 1; round < pp_size; ++round) {
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
            expect_pending_active();
        }

        ASSERT_TRUE(executor.drainPendingResults().ok());
        EXPECT_FALSE(stream->isPPInflight());
        EXPECT_EQ(stream->completeTokenIdsVec(0).back(), 20);
        EXPECT_EQ(executor.tps_reporter_.active_count_, 0);
        EXPECT_EQ(executor.wall_tps_reporter_.active_count_, 0);
        EXPECT_TRUE(executor.tps_reporter_.collector_.hasMetrics());
        EXPECT_TRUE(executor.wall_tps_reporter_.collector_.hasMetrics());

        /** Repeated drain and slot reuse must neither reactivate a completed batch nor underflow the count. */
        ASSERT_TRUE(executor.drainPendingResults().ok());
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_EQ(executor.tps_reporter_.active_count_, 0);
        EXPECT_EQ(executor.wall_tps_reporter_.active_count_, 0);
    }
}

TEST_F(PPExecutorTest, FirstFakeResultIsConsumedWithoutDispatchOrMetrics) {
    auto params = makeMtpParams(3);
    PPExecutor executor(params, nullptr, false);
    enableMetrics(executor);
    ResourceContext resources;
    auto stream = PPExecutor::createMinFakeDecodeStream(
        params.model_config_, params.runtime_config, resources, params.sp_config);
    const auto history = stream->completeTokenIdsVec(0);
    PPExecutor::InflightBatch batch;
    batch.result_pending   = true;
    batch.stream_groups    = StreamGroups({stream});
    batch.schedule_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - 100000;

    /** An empty fake result must be consumed without dispatching or stealing the next result. */
    PPExecutionResult fake_result;
    PPExecutionResult next_result;
    next_result.request_ids = torch::tensor({101}, torch::kInt64);
    next_result.new_token_ids = intTensor({7}).reshape({1, 1});
    next_result.new_token_lengths = intTensor({1});
    auto transport = std::make_unique<InMemoryPPTransport>();
    auto* recorded_transport = transport.get();
    for (const auto& result : {fake_result, next_result}) {
        transport->enqueueObject(pp_serialization::serializeExecutionResult(result));
    }
    executor.transport_ = std::move(transport);

    ASSERT_TRUE(executor.processExecutionResult(batch).ok());
    EXPECT_FALSE(batch.result_pending);
    EXPECT_EQ(recorded_transport->receive_index, 2);
    EXPECT_EQ(stream->completeTokenIdsVec(0), history);
    EXPECT_FALSE(executor.tps_reporter_.collector_.hasMetrics());
    EXPECT_FALSE(executor.wall_tps_reporter_.collector_.hasMetrics());
    const auto received = pp_serialization::deserializeExecutionResult(executor.receiveObject());
    EXPECT_EQ(tensorToVector<int64_t>(received.request_ids), (std::vector<int64_t>{101}));
    EXPECT_EQ(tensorToVector<int32_t>(received.new_token_ids), (std::vector<int32_t>{7}));
    EXPECT_EQ(recorded_transport->receive_index, 4);
}

TEST_F(PPExecutorTest, FirstPrefillCountsInputWorkEvenWhenSamplingFails) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        SCOPED_TRACE(type);
        auto params                             = makeMtpParams(3);
        params.sp_config.type                   = type;
        params.sp_config.sp_dspark_mask_token_id = 63;
        PPExecutor executor(params, nullptr, false);
        enableMetrics(executor);
        auto accepted = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2, 3, 4}, 0);
        auto failed   = makeStream(ResourceContext{}, params.model_config_, 202, {5, 6}, 0);
        accepted->setReuseLength(1);
        accepted->generate_input_->priority = 1;
        failed->generate_input_->priority   = 2;

        PPExecutionResult result;
        result.request_ids = torch::tensor({101, 202}, torch::kInt64);
        result.request_errors.resize(2);
        result.request_errors[1] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "prefill sampling failed");

        StreamGroups::TokenCountsByPriority     counts;
        RtpLLMSpeculativeEngineMetricsCollector sp_collector;
        executor.collectTokenCounts(StreamGroups({accepted, failed}), result, counts, sp_collector);
        ASSERT_EQ(counts.size(), 2u);
        EXPECT_EQ(counts.at(1).context, 3);
        EXPECT_EQ(counts.at(1).context_with_cache, 4);
        EXPECT_EQ(counts.at(1).generate, 0);
        EXPECT_EQ(counts.at(1).total, 3);
        EXPECT_EQ(counts.at(2).context, 2);
        EXPECT_EQ(counts.at(2).context_with_cache, 2);
        EXPECT_EQ(counts.at(2).generate, 0);
        EXPECT_EQ(counts.at(2).total, 2);
        EXPECT_EQ(sp_collector.total_stream_num, 0);
        EXPECT_EQ(sp_collector.total_propose_token_num, 0);
        EXPECT_EQ(sp_collector.total_accepted_token_num, 0);
    }
}

TEST_F(PPExecutorTest, FirstReturnedAcceptanceSurvivesBudgetClippingAndCancellation) {
    auto params = makeMtpParams(3);
    auto cache_manager =
        std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
    ASSERT_TRUE(cache_manager->init());
    PPExecutor executor(params, cache_manager, false);
    enableMetrics(executor);
    ResourceContext resources;
    resources.cache_manager = cache_manager;
    auto accepted  = makeStream(resources, params.model_config_, 101, {1, 2}, 0);
    auto capped    = makeStream(resources, params.model_config_, 202, {3, 4}, 0);
    auto failed    = makeStream(resources, params.model_config_, 303, {5, 6}, 0);
    auto cancelled = makeStream(resources, params.model_config_, 404, {7, 8}, 0);
    const std::list<GenerateStreamPtr> streams{accepted, capped, failed, cancelled};
    for (const auto& stream : streams) {
        stream->generateConfig()->max_new_tokens = 8;
        stream->generate_input_->priority       = 1;
        auto buffer                             = std::make_shared<SpeculativeExecutorStreamOutput>();
        buffer->propose_step                     = 3;
        buffer->tokens                           = torch::zeros({1, 4}, torch::kInt32);
        stream->setSPOutputBuffer(buffer);
        stream->fakeInitKVBlock(4);
        stream->setIsContextStream(false);
        stream->setPPInflight();
    }
    capped->generateConfig()->max_new_tokens = 1;
    capped->generate_input_->priority        = 2;
    cancelled->generate_input_->priority     = 2;
    cancelled->reportError(ErrorCode::CANCELLED, "cancelled while in flight");

    PPExecutor::InflightBatch batch;
    batch.stream_groups    = StreamGroups(streams);
    batch.schedule_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - 100000;
    PPExecutionResult result;
    result.request_ids       = torch::tensor({101, 202, 303, 404}, torch::kInt64);
    result.new_token_ids     = intTensor({10, 11, 12, 13}).repeat({4, 1});
    /** The second row returns all accepted tokens; only the first-stage write is capped. */
    result.new_token_lengths = intTensor({2, 4, 1, 3});
    result.new_token_ids[2].zero_();
    result.propose_token_ids = intTensor({20, 21, 22}).repeat({4, 1});
    result.request_outputs.resize(4);
    result.request_errors.resize(4);
    result.request_errors[2] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "failed placeholder");
    auto transport = std::make_unique<InMemoryPPTransport>();
    transport->enqueueObject(pp_serialization::serializeExecutionResult(result));
    executor.transport_ = std::move(transport);

    StreamGroups::TokenCountsByPriority     counts;
    RtpLLMSpeculativeEngineMetricsCollector sp_collector;
    executor.collectTokenCounts(batch.stream_groups, result, counts, sp_collector);
    ASSERT_TRUE(executor.processExecutionResult(batch).ok());
    EXPECT_EQ(accepted->seqLength(), 4);
    EXPECT_EQ(capped->seqLength(), 3);
    EXPECT_TRUE(capped->hasEvent(StreamEvents::GenerateDone));
    EXPECT_EQ(failed->seqLength(), 2);
    EXPECT_EQ(cancelled->seqLength(), 2);
    ASSERT_EQ(counts.size(), 2u);
    EXPECT_EQ(counts.at(1).generate, 2);
    EXPECT_EQ(counts.at(1).total, 2);
    /** A successful returned row is counted even if its request was cancelled on the first stage. */
    EXPECT_EQ(counts.at(2).generate, 7);
    EXPECT_EQ(counts.at(2).total, 7);
    EXPECT_EQ(counts.at(1).context + counts.at(2).context, 0);
    EXPECT_EQ(sp_collector.spec_steps, 3);
    EXPECT_EQ(sp_collector.total_stream_num, 3);
    EXPECT_EQ(sp_collector.total_propose_token_num, 9);
    EXPECT_EQ(sp_collector.total_accepted_token_num, 9);
    for (const auto* collector : {&executor.tps_reporter_.collector_, &executor.wall_tps_reporter_.collector_}) {
        EXPECT_EQ(collector->generateTPS(), 9);
        EXPECT_EQ(collector->totalTPS(), 9);
        const auto priorities = collector->priorityCollectorsForReport();
        ASSERT_EQ(priorities.size(), 2u);
        EXPECT_EQ(priorities.at(1).generateTPS(), 2);
        EXPECT_EQ(priorities.at(2).generateTPS(), 7);
    }
    for (const auto& stream : streams) {
        EXPECT_FALSE(stream->isPPInflight());
    }
}

TEST_F(PPExecutorTest, GrammarMetricsAreReportedOnlyByFirstStageTpRoots) {
    std::vector<std::string> vocab;
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    const auto tokenizer_json =
        xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW, 128, std::vector<int32_t>{0}).SerializeJSON();
    struct TestCase {
        int  world_rank;
        bool warmup;
        bool reports;
    };
    // PP=3, TP=2, DP=2: both first-stage DP roots, a TP peer, middle/tail roots and warmup.
    for (const auto& test_case : {TestCase{0, false, true},
                                  TestCase{2, false, true},
                                  TestCase{1, false, false},
                                  TestCase{4, false, false},
                                  TestCase{8, false, false},
                                  TestCase{0, true, false}}) {
        SCOPED_TRACE(testing::Message() << "world_rank=" << test_case.world_rank << ", warmup=" << test_case.warmup);
        auto params                               = makeStageParams(test_case.world_rank / 4);
        params.parallelism_config.tp_size         = 2;
        params.parallelism_config.dp_size         = 2;
        params.parallelism_config.world_size      = 12;
        params.parallelism_config.world_rank      = test_case.world_rank;
        params.parallelism_config.dp_rank         = (test_case.world_rank % 4) / 2;
        params.parallelism_config.tp_rank         = test_case.world_rank % 2;
        params.grammar_config.tokenizer_info_json = tokenizer_json;
        params.grammar_config.num_workers         = 1;
        params.grammar_config.compile_timeout_ms  = 30000;
        auto monitor                              = std::make_shared<kmonitor::KMonitor>("pp_grammar_metrics_test");
        monitor->SetServiceName("");
        const kmonitor::MetricsTags tags("dp_rank", std::to_string(params.parallelism_config.dp_rank));
        params.metrics_reporter = std::make_shared<kmonitor::MetricsReporter>(monitor, "", tags);
        PPExecutor executor(params, nullptr, test_case.warmup);
        auto       backend = LogitsProcessorFactory::grammarBackend();
        ASSERT_NE(backend, nullptr);

        std::set<std::string> suffixes = {"overload_qps",
                                          "compile_inflight",
                                          "verdict_cache_bytes",
                                          "total_cache_budget_bytes",
                                          "compiler_cache_budget_bytes",
                                          "verdict_cache_budget_bytes"};
        if (test_case.world_rank == 0 && !test_case.warmup) {
            // Exercise the full event chain once; other configurations only check reporter routing.
            for (int attempt = 0; attempt < 2; ++attempt) {
                ASSERT_TRUE(backend->createMatcherFromKey({"regex", "[ab]+"}).ok());
                ASSERT_FALSE(backend->createMatcherFromKey({"json", "{"}).ok());
            }
            EXPECT_EQ(backend->stats().compile_total, 2);
            EXPECT_EQ(backend->stats().cache_hit, 1);
            EXPECT_EQ(backend->stats().invalid_hit, 1);
            suffixes.insert({"compile_qps", "compile_invalid_qps", "cache_hit_qps", "compile_latency_us"});
        }
        backend->reportOverload();

        std::set<std::string> expected;
        if (test_case.reports) {
            for (const auto& suffix : suffixes) {
                expected.insert("rtp_llm_grammar_" + suffix);
            }
        }
        kmonitor::MetricsCollector snapshot;
        monitor->GetMetrics(&snapshot, {kmonitor::NORMAL}, 10000);
        std::set<std::string> seen;
        for (const auto* record : snapshot.GetRecords().getRecords()) {
            for (const auto* value : record->Values()) {
                if (value->Name().find("grammar_") == std::string::npos) {
                    continue;
                }
                EXPECT_TRUE(seen.insert(value->Name()).second);
                EXPECT_EQ(record->Tags()->FindTag("pp_rank"), "");
                EXPECT_EQ(record->Tags()->FindTag("dp_rank"), tags.FindTag("dp_rank"));
            }
        }
        EXPECT_EQ(seen, expected);
    }
}

TEST_F(PPExecutorTest, FirstResultMetricsAreRestrictedToItsTpRootInEachDpReplica) {
    for (int world_rank = 0; world_rank < 12; ++world_rank) {
        SCOPED_TRACE(world_rank);
        auto params                                    = makeMtpParams(3);
        params.model_config_.num_layers                 = 3;
        params.parallelism_config.pp_stage_layer_counts = {1, 1, 1};
        params.parallelism_config.pp_size              = 3;
        params.parallelism_config.tp_size              = 2;
        params.parallelism_config.dp_size              = 2;
        params.parallelism_config.world_size           = 12;
        params.parallelism_config.world_rank           = world_rank;
        params.parallelism_config.pp_rank              = world_rank / 4;
        params.parallelism_config.dp_rank              = (world_rank % 4) / 2;
        params.parallelism_config.tp_rank              = world_rank % 2;
        PPExecutor executor(params, nullptr, false);
        enableMetrics(executor);

        auto first  = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 0);
        auto second = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4, 5}, 0);
        for (const auto& stream : {first, second}) {
            stream->setIsContextStream(false);
            stream->generate_input_->priority = 7;
        }
        PPExecutor::InflightBatch batch;
        batch.stream_groups = StreamGroups({first, second});
        const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
        executor.slots_[executor.current_slot_].schedule_time_us = now;
        batch.schedule_time_us = now - 100000;

        /** Verify uses prefill-shaped model inputs, but represents two decode requests. */
        GptModelInputs model_input;
        model_input.is_target_verify = true;
        model_input.input_lengths    = intTensor({4, 4});
        model_input.sequence_lengths = intTensor({});
        model_input.combo_tokens     = intTensor({2, 10, 11, 12, 5, 20, 21, 22});
        executor.collectExecutorMetrics(model_input, intTensor({2, 3}), batch.executor_collector);
        PPExecutionResult result;
        result.request_ids       = torch::tensor({101, 202}, torch::kInt64);
        result.request_errors.resize(2);
        result.new_token_lengths = intTensor({2, 4});
        StreamGroups::TokenCountsByPriority     counts;
        RtpLLMSpeculativeEngineMetricsCollector sp_collector;
        executor.collectTokenCounts(batch.stream_groups, result, counts, sp_collector);

        const auto before = autil::TimeUtility::currentTimeInMicroSeconds();
        executor.reportResultMetrics(batch, counts, sp_collector);
        const auto after = autil::TimeUtility::currentTimeInMicroSeconds();
        const bool reports = params.parallelism_config.pp_rank == 0 && params.parallelism_config.tp_rank == 0;
        if (reports) {
            EXPECT_EQ(batch.executor_collector.generate_batch_size, 2);
            EXPECT_EQ(batch.executor_collector.context_batch_size, 0);
            EXPECT_EQ(batch.executor_collector.execute_token_size, 8);
            EXPECT_EQ(batch.executor_collector.max_seq_len, 3);
            EXPECT_GE(sp_collector.step_latency_us, before - batch.schedule_time_us);
            EXPECT_LE(sp_collector.step_latency_us, after - batch.schedule_time_us);
            EXPECT_EQ(sp_collector.spec_steps, 3);
            EXPECT_EQ(sp_collector.total_stream_num, 2);
            EXPECT_EQ(sp_collector.total_propose_token_num, 6);
            EXPECT_EQ(sp_collector.total_accepted_token_num, 6);
        } else {
            EXPECT_EQ(batch.executor_collector.generate_batch_size, 0);
            EXPECT_EQ(batch.executor_collector.execute_token_size, 0);
            EXPECT_EQ(sp_collector.step_latency_us, 0);
        }
        for (const auto* collector : {&executor.tps_reporter_.collector_, &executor.wall_tps_reporter_.collector_}) {
            EXPECT_EQ(collector->hasMetrics(), reports);
            EXPECT_EQ(collector->generateTPS(), reports ? 6 : 0);
            const auto priorities = collector->priorityCollectorsForReport();
            if (reports) {
                ASSERT_EQ(priorities.size(), 1u);
                EXPECT_EQ(priorities.at(7).generateTPS(), 6);
            } else {
                EXPECT_TRUE(priorities.empty());
            }
        }
        EXPECT_EQ(sp_collector.propose_step_latency_us, 0);
        EXPECT_EQ(sp_collector.score_step_latency_us, 0);
        EXPECT_EQ(sp_collector.speculative_sampler_latency_us, 0);
    }
}

TEST_F(PPExecutorTest, FirstWarmupAndDisabledReporterDoNotReportResults) {
    for (bool warm_up : {false, true}) {
        SCOPED_TRACE(warm_up);
        auto params = makeMtpParams(3);
        PPExecutor executor(params, nullptr, warm_up);
        if (warm_up) {
            enableMetrics(executor);
        }
        auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 0);
        stream->setIsContextStream(false);
        PPExecutor::InflightBatch batch;
        batch.stream_groups    = StreamGroups({stream});
        batch.schedule_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - 100000;
        PPExecutionResult result;
        result.request_ids = torch::tensor({101}, torch::kInt64);
        result.request_errors.resize(1);
        result.new_token_lengths = intTensor({4});
        StreamGroups::TokenCountsByPriority     counts;
        RtpLLMSpeculativeEngineMetricsCollector sp_collector;
        executor.collectTokenCounts(batch.stream_groups, result, counts, sp_collector);
        executor.reportResultMetrics(batch, counts, sp_collector);

        EXPECT_EQ(sp_collector.step_latency_us, 0);
        EXPECT_FALSE(executor.tps_reporter_.collector_.hasMetrics());
        EXPECT_FALSE(executor.wall_tps_reporter_.collector_.hasMetrics());
        if (!warm_up) {
            EXPECT_TRUE(counts.empty());
            EXPECT_EQ(sp_collector.total_stream_num, 0);
        }
    }
}

TEST_F(PPExecutorTest, FirstInflightMetricsStayWithTheirBatchThroughExecutionAndDrain) {
    auto params = makeMtpParams(3);
    auto cache = std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16));
    ASSERT_TRUE(cache->init());
    PPExecutor executor(params, cache, false);
    auto monitor = std::make_shared<kmonitor::KMonitor>("pp_sampling_metrics_test");
    monitor->SetServiceName("");
    kmonitor::MetricsTags base_tags("dp_rank", "0");
    executor.metrics_reporter_ = std::make_shared<kmonitor::MetricsReporter>(monitor, "", base_tags);
    int64_t snapshot_time_ms = 0;
    auto expect_sampling_metric = [&](int64_t duration) {
        kmonitor::MetricsCollector snapshot;
        monitor->GetMetrics(&snapshot, {kmonitor::NORMAL}, snapshot_time_ms += 10000);
        size_t samples = 0;
        for (const auto* record : snapshot.GetRecords().getRecords()) {
            for (const auto* value : record->Values()) {
                if (value->Name() != "rtp_llm_sample_input_us") {
                    continue;
                }
                ++samples;
                EXPECT_EQ(record->Tags()->FindTag("dp_rank"), "0");
                // First stage reports its own executor metrics tagged pp_rank=0.
                EXPECT_EQ(record->Tags()->FindTag("pp_rank"), "0");
                kmonitor::MinMaxCalculator expected;
                expected.Add(duration);
                EXPECT_EQ(value->Value(), expected.ToString());
            }
        }
        EXPECT_EQ(samples, duration > 0 ? 1u : 0u);
    };
    executor.setModel(std::make_unique<RecordingDraftModel>());
    auto transport = std::make_unique<InMemoryPPTransport>();
    ResourceContext resources;
    resources.cache_manager = cache;
    auto decode  = makeStream(resources, params.model_config_, 101, {1, 2}, 0);
    auto prefill = makeStream(resources, params.model_config_, 202, {3, 4, 5}, 0);
    for (const auto& stream : {decode, prefill}) {
        stream->generateConfig()->max_new_tokens = 8;
        stream->fakeInitKVBlock(4);
        stream->setPPInflight();
    }
    decode->setIsContextStream(false);
    auto sp_buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_buffer->propose_step = 3;
    sp_buffer->tokens       = torch::zeros({1, 4}, torch::kInt32);
    decode->setSPOutputBuffer(sp_buffer);

    PPExecutionResult decode_result;
    decode_result.request_ids       = torch::tensor({101}, torch::kInt64);
    decode_result.new_token_ids     = intTensor({10, 11, 12, 13}).reshape({1, 4});
    decode_result.new_token_lengths = intTensor({4});
    decode_result.propose_token_ids = intTensor({20, 21, 22}).reshape({1, 3});
    decode_result.request_errors.resize(1);
    decode_result.request_outputs.resize(1);
    auto prefill_result              = decode_result;
    prefill_result.request_ids       = torch::tensor({202}, torch::kInt64);
    prefill_result.new_token_ids     = intTensor({14}).reshape({1, 1});
    prefill_result.new_token_lengths = intTensor({1});
    prefill_result.sample_input_us   = 456;
    for (const auto& result : {decode_result, prefill_result}) {
        transport->enqueueObject(pp_serialization::serializeExecutionResult(result));
    }
    executor.transport_ = std::move(transport);

    const auto schedule_time = autil::TimeUtility::currentTimeInMicroSeconds();
    auto& retired            = executor.slots_[1];
    retired.result_pending   = true;
    retired.stream_groups   = StreamGroups({decode});
    retired.schedule_time_us = schedule_time - 100000;
    retired.executor_collector.generate_batch_size = 1;
    retired.executor_collector.execute_token_size  = 4;
    retired.executor_collector.model_forward_us    = 123;

    ScheduleOutput schedule_output;
    schedule_output.streams = {prefill};
    ASSERT_TRUE(executor.process(schedule_output, schedule_time).ok());
    EXPECT_EQ(decode->seqLength(), 6);
    EXPECT_EQ(prefill->seqLength(), 3);
    EXPECT_EQ(retired.executor_collector.generate_batch_size, 1);
    EXPECT_EQ(retired.executor_collector.model_forward_us, 123);
    EXPECT_EQ(retired.executor_collector.sample_input_us, 0);
    expect_sampling_metric(0);
    const auto& admitted = executor.slots_[0];
    EXPECT_EQ(admitted.schedule_time_us, schedule_time);
    EXPECT_EQ(admitted.executor_collector.context_batch_size, 1);
    EXPECT_EQ(admitted.executor_collector.generate_batch_size, 0);
    EXPECT_EQ(admitted.executor_collector.sample_input_us, 0);
    EXPECT_EQ(admitted.executor_collector.execute_token_size, 3);
    EXPECT_EQ(admitted.executor_collector.context_batch_size_when_has_context, 1);
    EXPECT_EQ(admitted.executor_collector.execute_token_size_when_has_context, 3);
    EXPECT_EQ(admitted.executor_collector.max_seq_len, 3);
    EXPECT_EQ(executor.tps_reporter_.collector_.generateTPS(), 4);
    EXPECT_FALSE(executor.tps_reporter_.collector_.hasContextTPS());

    /** Empty rounds still retire pending results; reusing a slot clears its previous metrics. */
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_EQ(retired.executor_collector.generate_batch_size, 0);
    EXPECT_EQ(retired.executor_collector.model_forward_us, 0);
    EXPECT_EQ(retired.executor_collector.dispatch_output_us, 0);
    EXPECT_EQ(retired.executor_collector.sample_input_us, 0);
    expect_sampling_metric(0);
    EXPECT_EQ(executor.tps_reporter_.collector_.generateTPS(), 4);
    EXPECT_EQ(executor.tps_reporter_.collector_.totalTPS(), 4);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_EQ(prefill->seqLength(), 4);
    EXPECT_EQ(admitted.executor_collector.sample_input_us, 456);
    expect_sampling_metric(456);
    EXPECT_EQ(admitted.executor_collector.context_batch_size, 1);
    EXPECT_EQ(admitted.executor_collector.execute_token_size, 3);
    EXPECT_EQ(executor.tps_reporter_.collector_.generateTPS(), 4);
    EXPECT_EQ(executor.tps_reporter_.collector_.totalTPS(), 7);
    EXPECT_TRUE(executor.tps_reporter_.collector_.hasContextTPS());
    EXPECT_EQ(executor.wall_tps_reporter_.collector_.generateTPS(), 4);
    EXPECT_EQ(executor.wall_tps_reporter_.collector_.totalTPS(), 7);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_EQ(admitted.executor_collector.context_batch_size, 0);
    EXPECT_EQ(admitted.executor_collector.execute_token_size, 0);
    EXPECT_EQ(admitted.executor_collector.context_batch_size_when_has_context, 0);
    EXPECT_EQ(admitted.executor_collector.sample_input_us, 0);
    expect_sampling_metric(0);
    EXPECT_EQ(executor.tps_reporter_.collector_.totalTPS(), 7);
    EXPECT_EQ(executor.wall_tps_reporter_.collector_.totalTPS(), 7);
}

TEST_F(PPExecutorTest, FirstDecodeMetricsCountSequencesAndFilterErrorsByRequest) {
    auto params = makeMtpParams(3);
    params.sp_config.type = SP_TYPE_NONE;
    PPExecutor executor(params, nullptr, false);
    enableMetrics(executor);
    auto accepted = makeStream(ResourceContext{}, params.model_config_, 202, {5, 6}, 2);
    auto failed   = makeStream(ResourceContext{}, params.model_config_, 303, {7, 8}, 3);
    accepted->generate_input_->priority = 1;
    failed->generate_input_->priority   = 2;
    accepted->setIsContextStream(false);
    failed->setIsContextStream(false);

    /** A decode batch has one error entry per request, even when each request has several sequences. */
    const StreamGroups groups({failed, accepted});
    PPExecutionResult result;
    result.request_ids = torch::tensor({303, 202}, torch::kInt64);
    result.request_errors.resize(2);
    result.request_errors[0] = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "decode sampling failed");

    StreamGroups::TokenCountsByPriority     counts;
    RtpLLMSpeculativeEngineMetricsCollector sp_collector;
    executor.collectTokenCounts(groups, result, counts, sp_collector);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.at(1).context, 0);
    EXPECT_EQ(counts.at(1).context_with_cache, 0);
    EXPECT_EQ(counts.at(1).generate, 2);
    EXPECT_EQ(counts.at(1).total, 2);
    EXPECT_EQ(counts.count(2), 0u);
    EXPECT_EQ(sp_collector.total_stream_num, 0);
    EXPECT_EQ(sp_collector.total_propose_token_num, 0);
    EXPECT_EQ(sp_collector.total_accepted_token_num, 0);
}

/** Middle: consume upstream activations, execute locally and publish model outputs. */
TEST_F(PPExecutorTest, MiddleForwardsOriginalPayloadWhileCopyingMultimodalInputsToCuda) {
    auto params = makeStageParams(1);
    PPExecutor executor(params, nullptr, false);
    auto* wire = attachTransport(executor);
    auto plan = makePlan(params);
    const auto options = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
    plan.model_input.multimodal_features = std::vector<torch::Tensor>{torch::arange(8, options).reshape({2, 4})};
    plan.model_input.mm_extra_input = std::vector<torch::Tensor>{torch::arange(6, options)};
    plan.model_input.engram_token_windows = torch::arange(20, torch::kInt32).reshape({5, 4});
    plan.model_input.mm_features_spans    = torch::tensor({0, 0, 2}, torch::kInt64).reshape({1, 3});
    enqueuePlan(*wire, plan);
    const auto cpu_plan = pp_serialization::deserializePlan(wire->received_tensors[1]);
    ASSERT_TRUE(cpu_plan.model_input.multimodal_features.has_value());
    ASSERT_TRUE(cpu_plan.model_input.mm_extra_input.has_value());
    EXPECT_TRUE(cpu_plan.model_input.multimodal_features->at(0).is_cpu());
    EXPECT_TRUE(cpu_plan.model_input.mm_extra_input->at(0).is_cpu());
    EXPECT_TRUE(torch::equal(cpu_plan.model_input.engram_token_windows, plan.model_input.engram_token_windows));
    EXPECT_TRUE(torch::equal(cpu_plan.model_input.mm_features_spans, plan.model_input.mm_features_spans));

    const void* received_payload = nullptr;
    int forwarded_payloads = 0;
    wire->on_receive = [&](const torch::Tensor& tensor) {
        if (wire->receive_index == 1) {
            received_payload = tensor.data_ptr();
        }
    };
    wire->on_send = [&](const torch::Tensor& tensor) {
        if (wire->sent_tensors.size() == 1) {
            EXPECT_TRUE(tensor.is_cpu());
            EXPECT_NE(received_payload, nullptr);
            EXPECT_EQ(tensor.data_ptr(), received_payload);
            ++forwarded_payloads;
        }
    };
    auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);

    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_EQ(forwarded_payloads, 1);
    EXPECT_TRUE(torch::equal(wire->sent_tensors[1], wire->received_tensors[1]));
    ASSERT_EQ(recorded->inputs.size(), 1u);
    const auto& input = recorded->inputs[0];
    ASSERT_TRUE(input.multimodal_features.has_value());
    ASSERT_TRUE(input.mm_extra_input.has_value());
    EXPECT_TRUE(input.multimodal_features->at(0).is_cuda());
    EXPECT_TRUE(input.mm_extra_input->at(0).is_cuda());
    EXPECT_TRUE(torch::equal(input.multimodal_features->at(0).cpu(), cpu_plan.model_input.multimodal_features->at(0)));
    EXPECT_TRUE(torch::equal(input.mm_extra_input->at(0).cpu(), cpu_plan.model_input.mm_extra_input->at(0)));
    EXPECT_TRUE(torch::equal(input.engram_token_windows.cpu(), plan.model_input.engram_token_windows));
    EXPECT_TRUE(torch::equal(input.mm_features_spans.cpu(), plan.model_input.mm_features_spans));
}

TEST_F(PPExecutorTest, MiddlePassesActivationsThroughTargetForEachExecutionPhase) {
    for (int phase : {0, 1, 2}) {
        SCOPED_TRACE(phase);
        auto params = makeStageParams(1, phase == 2 ? SP_TYPE_MTP : SP_TYPE_NONE);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        const auto plan = makePlan(params, phase != 0);
        enqueuePlan(*wire, plan);
        auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        ASSERT_EQ(recorded->inputs.size(), 1u);
        EXPECT_EQ(recorded->has_input, std::vector<bool>{true});
        EXPECT_EQ(recorded->has_output, std::vector<bool>{true});
        EXPECT_EQ(recorded->inputs[0].is_target_verify, phase == 2);
        EXPECT_TRUE(torch::equal(recorded->received_hidden[0],
                                activations(plan.model_input.combo_tokens.numel()).tensors.at("hidden_states")));
        ASSERT_EQ(wire->sent_tensors.size(), 5u);
        EXPECT_TRUE(torch::equal(wire->sent_tensors.back(), recorded->hidden_buffer.narrow(1, 0, 2)));
        EXPECT_TRUE(executor.sampling_states_.empty());
        EXPECT_EQ(executor.draft_model_, nullptr);
        EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
    }
}

TEST_F(PPExecutorTest, MiddleEmptyPlanSkipsTargetExecutionWithoutStopping) {
    auto       params = makeStageParams(1);
    PPExecutor executor(params, nullptr, false);
    auto*      wire               = attachTransport(executor);
    auto*      recorded           = installModel<RecordingStageModel>(executor, params.parallelism_config);
    int        profile_calls      = 0;
    executor.profile_step_start_  = [&] { ++profile_calls; };
    executor.profile_step_finish_ = [&] { ++profile_calls; };
    PPExecutionPlan plan;
    plan.model_input.skip_run = true;
    enqueuePlan(*wire, plan);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_EQ(wire->sent_tensors.size(), 2u);
    EXPECT_FALSE(executor.stopped_);
    EXPECT_TRUE(recorded->inputs.empty());
    EXPECT_EQ(profile_calls, 0);
    EXPECT_EQ(wire->receive_index, 2u);
    enqueuePlan(*wire, makePlan(params));
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_EQ(recorded->inputs.size(), 1u);
    EXPECT_EQ(profile_calls, 2);
    EXPECT_FALSE(executor.stopped_);
    EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
}

/** Last-stage sampling retains request state and returns results for every completed batch. */
TEST_F(PPExecutorTest, LastSamplesAcrossRoundsWithoutReinitializingRequestState) {
    auto params = makeStageParams(2);
    PPExecutor executor(params, nullptr, false);
    auto* wire = attachTransport(executor);
    auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
    auto* recorded = target.get();
    recorded->next_tokens = {7, 8};
    executor.setModel(std::move(target));
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto                                   prefill = makePlan(params, false, false, history_lengths);
    prefill.sampling_plan.random_seeds = {123, 456};
    prefill.sampling_plan.do_sample.fill_(true);
    prefill.sampling_plan.top_k.fill_(2);
    prefill.output_config.return_all_hidden_states = true;
    for (auto& request : prefill.output_config.requests) {
        request.return_all_hidden_states = true;
    }
    prefill.model_input.need_all_hidden_states = true;
    enqueuePlan(*wire, prefill);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    ASSERT_EQ(wire->sent_tensors.size(), 2u);
    auto first = lastResult(*wire);
    EXPECT_GT(first.sample_input_us, 0);
    EXPECT_EQ(tensorToVector<int32_t>(first.new_token_ids), (std::vector<int32_t>{7, 8}));
    EXPECT_TRUE(torch::equal(first.request_outputs[0].all_hidden_states,
                             recorded->hidden_buffer.narrow(0, 0, 2).narrow(1, 0, 2).cpu()));
    EXPECT_TRUE(torch::equal(first.request_outputs[1].all_hidden_states,
                             recorded->hidden_buffer.narrow(0, 2, 3).narrow(1, 0, 2).cpu()));
    EXPECT_TRUE(recorded->hidden_requests.empty());
    EXPECT_EQ(tensorToVector<int32_t>(first.new_token_lengths), (std::vector<int32_t>{1, 1}));
    EXPECT_FALSE(first.propose_token_ids.defined());
    const auto generator                                         = executor.sampling_states_.at(101).sampling.generator;
    const auto rng_before = generator.get_state();
    auto processor = std::make_shared<RecordingLogitsProcessor>();
    executor.sampling_states_.at(101).sampling.logits_processors = {processor};

    auto decode = makePlan(params, true, false, history_lengths);
    EXPECT_EQ(tensorToVector<int32_t>(decode.sampling_plan.token_ids), (std::vector<int32_t>{7, 8}));
    EXPECT_EQ(tensorToVector<int64_t>(decode.sampling_plan.token_offsets), (std::vector<int64_t>{0, 1, 2}));
    EXPECT_EQ(history_lengths.at(101), 3);
    EXPECT_EQ(history_lengths.at(202), 4);
    decode.sampling_plan.random_seeds = {999, 999};
    decode.sampling_plan.do_sample.fill_(true);
    decode.sampling_plan.top_k.fill_(2);
    recorded->next_tokens = {11, 21};
    enqueuePlan(*wire, decode);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    ASSERT_EQ(wire->sent_tensors.size(), 4u);
    const auto result = lastResult(*wire);
    EXPECT_GT(result.sample_input_us, 0);
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{11, 21}));
    EXPECT_EQ(executor.sampling_states_.at(101).sampling.generator, generator);
    EXPECT_EQ(executor.sampling_states_.at(101).token_history, (std::vector<std::vector<int32_t>>{{1, 2, 7}}));
    EXPECT_EQ(executor.sampling_states_.at(202).token_history, (std::vector<std::vector<int32_t>>{{3, 4, 5, 8}}));
    EXPECT_EQ(generator.current_seed(), 123);
    EXPECT_FALSE(torch::equal(generator.get_state(), rng_before));
    ASSERT_EQ(processor->committed_tokens.size(), 1u);
    EXPECT_EQ(tensorToVector<int32_t>(processor->committed_tokens[0]), std::vector<int32_t>{11});
    EXPECT_EQ(recorded->has_input, (std::vector<bool>{true, true}));
    EXPECT_EQ(recorded->has_output, (std::vector<bool>{false, false}));
}

TEST_F(PPExecutorTest, LastHistoryDeltasMatchFullHistorySamplingAcrossReorderedAndPausedRequests) {
    auto params = makeStageParams(2);
    std::vector<std::string> vocabulary(64);
    for (size_t token = 0; token < vocabulary.size(); ++token) {
        vocabulary[token] = "unused" + std::to_string(token);
    }
    vocabulary[0] = "a";
    vocabulary[1] = "b";
    vocabulary[2] = "c";
    vocabulary[3] = "d";
    vocabulary[63] = "<eos>";
    params.grammar_config.num_workers = 1;
    params.grammar_config.tokenizer_info_json = xgrammar_impl::serializeTokenizerInfo(
        vocabulary,
        R"({"vocab_size":64,"stop_token_ids":[63],"vocab_type":"RAW","add_prefix_space":false})");
    PPExecutor delta(params, nullptr, false);
    PPExecutor full(params, nullptr, false);
    ASSERT_NE(LogitsProcessorFactory::grammarBackend(), nullptr);

    GenerateConfig multi_config;
    multi_config.max_new_tokens       = 16;
    multi_config.num_return_sequences = 2;
    multi_config.do_sample            = true;
    multi_config.top_k                = 4;
    multi_config.temperature          = .8f;
    multi_config.random_seed          = 123;
    multi_config.repetition_penalty   = 1.7f;
    multi_config.presence_penalty     = .4f;
    multi_config.frequency_penalty    = .6f;
    multi_config.no_repeat_ngram_size = 2;
    multi_config.return_all_probs     = ReturnAllProbsMode::DEFAULT;
    auto grammar_config_request = multi_config;
    grammar_config_request.num_return_sequences = 1;
    grammar_config_request.random_seed          = 456;
    grammar_config_request.no_repeat_ngram_size = 0;
    grammar_config_request.regex                = "(a[bc]){8}";
    const RequestPrompts prompts{{101, {0, 1, 0, 1}}, {202, {3, 2}}};
    RequestHistories histories{{101, {prompts.at(101), prompts.at(101)}}, {202, {prompts.at(202)}}};
    auto multi = makeStream(ResourceContext{}, params.model_config_, 101, prompts.at(101), multi_config);
    auto grammar = makeStream(ResourceContext{}, params.model_config_, 202, prompts.at(202), grammar_config_request);
    PPBatchStreamProcessor producer(params.model_config_, params.pd_sep_config, ProfilingDebugLoggingConfig{},
                                     CacheConfig{}, params.sp_config, false);
    auto* delta_wire = attachTransport(delta);
    auto* full_wire  = attachTransport(full);
    auto* delta_model = installModel<RecordingDraftModel>(delta);
    auto* full_model  = installModel<RecordingDraftModel>(full);
    std::unordered_map<RequestId, int32_t> cursors;
    const std::vector<std::list<GenerateStreamPtr>> batches{
        {multi, grammar}, {grammar, multi}, {multi}, {multi, grammar}, {grammar, multi}};
    for (size_t round = 0; round < batches.size(); ++round) {
        SCOPED_TRACE(round);
        const StreamGroups groups(batches[round]);
        TensorHolder holder;
        auto gathered = producer.gatherExecutionPlan(groups, {}, false, cursors, holder);
        ASSERT_TRUE(gathered.ok()) << gathered.status().ToString();
        const auto& plan = gathered.value();
        const auto ids = tensorToVector<RequestId>(plan.sampling_plan.request_ids);

        /** Recreate the reference matcher from the full generated text, independently of incremental updates. */
        if (full.sampling_states_.count(202)) {
            auto input = std::make_shared<GenerateInput>();
            input->input_ids       = intTensor(prompts.at(202));
            input->generate_config = std::make_shared<GenerateConfig>(grammar_config_request);
            auto processors = LogitsProcessorFactory::createLogitsProcessors(input, 1, 1, 63);
            ASSERT_TRUE(processors.ok()) << processors.status().ToString();
            const auto& history = histories.at(202).front();
            const auto output_length = history.size() - prompts.at(202).size();
            const auto outputs = intTensor(std::vector<int32_t>(history.begin() + prompts.at(202).size(), history.end()))
                                     .reshape({1, static_cast<int64_t>(output_length)});
            for (const auto& processor : processors.value()) {
                ASSERT_FALSE(processor->updateStatus(outputs, output_length).has_value());
            }
            full.sampling_states_.at(202).sampling.logits_processors = std::move(processors.value());
        }
        auto reference = fullHistoryPlan(plan, prompts, histories, full);
        auto logits = torch::full({plan.sampling_plan.sequence_lengths.numel(), 64},
                                   -std::numeric_limits<float>::infinity(),
                                   torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
        logits.narrow(1, 0, 4).copy_(torch::tensor({1.5f, 1.0f, .5f, .25f}, logits.options()));
        delta_model->scripted_logits = logits;
        full_model->scripted_logits  = logits;
        enqueuePlan(*delta_wire, plan);
        enqueuePlan(*full_wire, reference);
        ASSERT_TRUE(delta.process(ScheduleOutput{}).ok());
        ASSERT_TRUE(full.process(ScheduleOutput{}).ok());
        const auto actual   = lastResult(*delta_wire);
        const auto expected = lastResult(*full_wire);
        EXPECT_TRUE(torch::equal(actual.new_token_ids, expected.new_token_ids));
        EXPECT_TRUE(torch::equal(actual.new_token_lengths, expected.new_token_lengths));
        for (size_t request = 0; request < ids.size(); ++request) {
            const auto id = ids[request];
            ASSERT_TRUE(actual.request_errors[request].ok()) << actual.request_errors[request].ToString();
            ASSERT_TRUE(expected.request_errors[request].ok()) << expected.request_errors[request].ToString();
            const auto& actual_state = delta.sampling_states_.at(id);
            const auto& expected_state = full.sampling_states_.at(id);
            EXPECT_EQ(actual_state.token_history, histories.at(id));
            EXPECT_EQ(expected_state.token_history, histories.at(id));
            EXPECT_TRUE(torch::equal(actual_state.sampling.generator.get_state(),
                                     expected_state.sampling.generator.get_state()));
            ASSERT_TRUE(actual.request_outputs[request].all_probs.defined());
            EXPECT_TRUE(torch::allclose(actual.request_outputs[request].all_probs,
                                         expected.request_outputs[request].all_probs));
            if (id == 101) {
                /** Repeated prompt tokens make this a history-sensitive distribution, even on the first round. */
                const auto raw = torch::softmax(logits[0] / multi_config.temperature, -1).cpu();
                EXPECT_FALSE(torch::allclose(actual.request_outputs[request].all_probs[0], raw));
            } else {
                const auto output_length = histories.at(id).front().size() - prompts.at(id).size();
                ASSERT_EQ(actual_state.sampling.logits_processors.size(), 1u);
                EXPECT_EQ(actual_state.sampling.logits_processors[0]->committedOutputLen(),
                          std::optional<int64_t>(output_length + 1));
                EXPECT_EQ(actual_state.sampling.logits_processors[0]->committedOutputLen(),
                          expected_state.sampling.logits_processors[0]->committedOutputLen());
                const auto probabilities = actual.request_outputs[request].all_probs[0];
                EXPECT_EQ(probabilities[3].item<float>(), 0);
                if (output_length % 2 == 0) {
                    EXPECT_FLOAT_EQ(probabilities[0].item<float>(), 1);
                } else {
                    EXPECT_EQ(probabilities[0].item<float>(), 0);
                    EXPECT_GT(probabilities[1].item<float>() + probabilities[2].item<float>(), .99f);
                }
            }
        }
        appendCommittedHistory(actual, histories);
        ASSERT_TRUE(producer.dispatchExecutionResult(groups, actual).ok());
        for (const auto& stream : groups.allStreams()) {
            const auto& sequences = histories.at(stream->streamId());
            for (size_t sequence = 0; sequence < sequences.size(); ++sequence) {
                EXPECT_EQ(stream->completeTokenIdsVec(sequence), sequences[sequence]);
            }
        }
    }
}

TEST_F(PPExecutorTest, LastRetirementAllowsRequestIdReuseWithFreshSamplingState) {
    for (bool fake_cleanup : {false, true}) {
        SCOPED_TRACE(fake_cleanup);
        auto params = makeStageParams(2);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
        auto first = makePlan(params);
        first.sampling_plan.random_seeds = {123, 456};
        enqueuePlan(*wire, first);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        const auto old_generator = executor.sampling_states_.at(101).sampling.generator;
        auto old_processor = std::make_shared<RecordingLogitsProcessor>();
        executor.sampling_states_.at(101).sampling.logits_processors = {old_processor};

        /** Both control-only cleanup and a scheduled fake batch must retire state before request IDs are reused. */
        PPExecutionPlan cleanup = fake_cleanup ? makeFakePlan(params, false) : PPExecutionPlan{};
        cleanup.model_input.skip_run = !fake_cleanup;
        cleanup.finished_request_ids = {101, 202};
        enqueuePlan(*wire, cleanup);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_TRUE(executor.sampling_states_.empty());
        EXPECT_FALSE(executor.stopped_);
        EXPECT_EQ(wire->sent_tensors.size(), fake_cleanup ? 4u : 2u);

        auto second = makePlan(params);
        second.sampling_plan.random_seeds = {789, 999};
        enqueuePlan(*wire, second);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_EQ(executor.sampling_states_.at(101).sampling.generator.current_seed(), 789u);
        EXPECT_EQ(executor.sampling_states_.at(202).sampling.generator.current_seed(), 999u);
        EXPECT_NE(executor.sampling_states_.at(101).sampling.generator, old_generator);
        EXPECT_EQ(executor.sampling_states_.at(101).token_history, (std::vector<std::vector<int32_t>>{{1, 2}}));
        EXPECT_TRUE(old_processor->committed_tokens.empty());
        EXPECT_EQ(recorded->inputs.size(), fake_cleanup ? 3u : 2u);
        EXPECT_EQ(wire->sent_tensors.size(), fake_cleanup ? 6u : 4u);
        EXPECT_EQ(tensorToVector<int64_t>(lastResult(*wire).request_ids), (std::vector<int64_t>{101, 202}));
        EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
    }
}

TEST_F(PPExecutorTest, LastUpdatesStatesAtRequestSequenceOffsetsAndPreservesEarlierErrors) {
    auto               params = makeStageParams(2);
    PPExecutor         executor(params, nullptr, false);
    auto               failed        = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 2);
    auto               healthy       = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4}, 2);
    auto               update_failed = makeStream(ResourceContext{}, params.model_config_, 303, {5, 6}, 1);
    auto               unrequested   = makeStream(ResourceContext{}, params.model_config_, 404, {7, 8}, 1);
    const StreamGroups groups({failed, healthy, update_failed, unrequested});
    TensorHolder                           holder;
    std::unordered_map<RequestId, int32_t> history_lengths;
    auto                                   execution_plan =
        executor.batch_stream_processor_->gatherExecutionPlan(groups, {}, false, history_lengths, holder);
    ASSERT_TRUE(execution_plan.ok());
    const auto& plan                               = execution_plan->sampling_plan;
    auto first = recordState(executor, 101, 2);
    auto second = recordState(executor, 202, 2);
    auto third = recordState(executor, 303);
    auto fourth                             = recordState(executor, 404);
    third->update_error = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "state update failed");
    auto result = makeInitializedResult(plan);
    result.new_token_ids                    = intTensor({0, 0, 20, 21, 30, 40}).reshape({6, 1});
    result.new_token_lengths                = torch::ones({6}, torch::kInt32);
    result.request_outputs[0].cum_log_probs = torch::full({2}, -3.0, torch::kFloat32);
    result.request_outputs[1].cum_log_probs = torch::full({2}, -3.0, torch::kFloat32);
    result.request_outputs[2].cum_log_probs = torch::full({1}, -3.0, torch::kFloat32);
    result.request_errors[0] = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "sampling failed");
    executor.advanceSamplingStates(plan, result);
    EXPECT_TRUE(first->committed_tokens.empty());
    ASSERT_EQ(second->committed_tokens.size(), 1u);
    EXPECT_EQ(tensorToVector<int32_t>(second->committed_tokens[0]), (std::vector<int32_t>{20, 21}));
    ASSERT_EQ(third->committed_tokens.size(), 1u);
    EXPECT_EQ(tensorToVector<int32_t>(third->committed_tokens[0]), std::vector<int32_t>{30});
    EXPECT_EQ(result.request_errors[0].code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_EQ(result.request_errors[2].ToString(), third->update_error->ToString());
    EXPECT_EQ(tensorToVector<float>(executor.sampling_states_.at(101).sampling.cum_log_probs),
              (std::vector<float>{0, 0}));
    EXPECT_EQ(tensorToVector<float>(executor.sampling_states_.at(202).sampling.cum_log_probs),
              (std::vector<float>{-3, -3}));
    EXPECT_EQ(executor.sampling_states_.at(303).sampling.cum_log_probs.item<float>(), 0);
    ASSERT_EQ(fourth->committed_tokens.size(), 1u);
    EXPECT_EQ(tensorToVector<int32_t>(fourth->committed_tokens[0]), std::vector<int32_t>{40});
    EXPECT_TRUE(result.request_errors[3].ok());
    EXPECT_FALSE(result.request_outputs[3].cum_log_probs.defined());
    EXPECT_EQ(executor.sampling_states_.at(404).sampling.cum_log_probs.item<float>(), 0);
}

TEST_F(PPExecutorTest, LastCleansFinishedRequestsBeforeEmptyOrActiveExecution) {
    for (bool empty : {false, true}) {
        SCOPED_TRACE(empty);
        auto params = makeStageParams(2);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        recordState(executor, 101);
        recordState(executor, 202);
        recordState(executor, 303);
        auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
        auto* recorded = target.get();
        target->on_forward = [&] { EXPECT_EQ(executor.sampling_states_.count(303), 0u); };
        executor.setModel(std::move(target));
        auto plan = empty ? PPExecutionPlan{} : makePlan(params, true);
        plan.model_input.skip_run = empty;
        plan.finished_request_ids = {303};
        enqueuePlan(*wire, plan);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_EQ(executor.sampling_states_.count(303), 0u);
        EXPECT_EQ(executor.sampling_states_.count(101), 1u);
        EXPECT_EQ(executor.sampling_states_.count(202), 1u);
        EXPECT_EQ(recorded->inputs.size(), empty ? 0u : 1u);
        EXPECT_EQ(wire->sent_tensors.size(), empty ? 0u : 2u);
    }
}

TEST_F(PPExecutorTest, LastPrefetchesGrammarBeforeTargetForward) {
    std::vector<std::string> vocab;
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    auto params                               = makeStageParams(2);
    params.grammar_config.tokenizer_info_json =
        xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW, 128, std::vector<int32_t>{0}).SerializeJSON();
    params.grammar_config.num_workers        = 1;
    params.grammar_config.compile_timeout_ms = 30000;

    PPExecutor executor(params, nullptr, false);
    auto*      wire    = attachTransport(executor);
    auto       backend = LogitsProcessorFactory::grammarBackend();
    ASSERT_NE(backend, nullptr);

    auto compile_entered        = std::make_shared<std::promise<void>>();
    auto compile_entered_future = compile_entered->get_future();
    auto release_compile        = std::make_shared<std::promise<void>>();
    auto release_compile_future = release_compile->get_future().share();
    backend->setCompileFnForTest([compile_entered, release_compile_future](const GrammarKeyCpp&) {
        compile_entered->set_value();
        release_compile_future.wait_for(std::chrono::seconds(5));
        GrammarCompileResult result;
        result.status = absl::InvalidArgumentError("invalid grammar from overlap test");
        return result;
    });

    auto plan = makePlan(params);
    plan.sampling_plan.logits_processor_configs[0].grammar_type  = "regex";
    plan.sampling_plan.logits_processor_configs[0].grammar_value = "[ab]+";
    auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
    target->next_tokens = {7, 8};
    target->on_forward  = [&compile_entered_future, release_compile] {
        EXPECT_EQ(compile_entered_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        release_compile->set_value();
    };
    executor.setModel(std::move(target));

    enqueuePlan(*wire, plan);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    const auto result = lastResult(*wire);
    ASSERT_EQ(result.request_errors.size(), 2u);
    EXPECT_EQ(result.request_errors[0].code(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_EQ(backend->stats().compile_total, 1);
}

TEST_F(PPExecutorTest, LastRequestErrorsPreserveHealthyExecutionAndStillSendResults) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP}) {
        for (const std::string phase : {"initialize", "sample", "update"}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << ", phase=" << phase);
            auto params = makeStageParams(2, type);
            auto propose = makeMtpProposeParams(params);
            PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr,
                                type == SP_TYPE_NONE ? nullptr : propose.get());
            auto* wire = attachTransport(executor);
            auto plan = makePlan(params);
            auto target = std::make_unique<RecordingDraftModel>();
            target->next_tokens = {7, 8};
            executor.setModel(std::move(target));
            RecordingDraftModel* recorded_draft = nullptr;
            if (type != SP_TYPE_NONE) {
                auto draft = std::make_unique<RecordingDraftModel>();
                recorded_draft = draft.get();
                executor.draft_model_ = std::move(draft);
            }
            auto healthy = recordState(executor, 202);
            std::shared_ptr<RecordingLogitsProcessor> failed;
            if (phase == "initialize") {
                /** A legal grammar configuration cannot initialize on a tail without its backend. */
                plan.sampling_plan.logits_processor_configs[0].grammar_type = "regex";
                plan.sampling_plan.logits_processor_configs[0].grammar_value = "[ab]+";
                ASSERT_EQ(LogitsProcessorFactory::grammarBackend(), nullptr);
            } else {
                failed = recordState(executor, 101);
                if (phase == "sample") {
                    failed->process_error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "processor failed");
                } else {
                    failed->update_error = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "state update failed");
                    failed->on_update = [&] {
                        if (recorded_draft) {
                            EXPECT_EQ(recorded_draft->inputs.size(), 3u);
                        }
                    };
                }
            }
            enqueuePlan(*wire, plan);
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
            ASSERT_EQ(wire->sent_tensors.size(), 2u);
            const auto result = lastResult(*wire);
            ASSERT_EQ(result.request_errors.size(), 2u);
            EXPECT_EQ(result.request_errors[0].code(), phase == "initialize" ? ErrorCode::INVALID_PARAMS :
                                                      phase == "sample" ? ErrorCode::EXECUTION_EXCEPTION :
                                                                          ErrorCode::UNKNOWN_ERROR);
            EXPECT_TRUE(result.request_errors[1].ok());
            ASSERT_EQ(healthy->committed_tokens.size(), 1u);
            EXPECT_EQ(tensorToVector<int32_t>(healthy->committed_tokens[0]), std::vector<int32_t>{8});
            if (failed) {
                EXPECT_EQ(failed->committed_tokens.size(), phase == "update" ? 1u : 0u);
            }
            if (recorded_draft) {
                ASSERT_EQ(recorded_draft->inputs.size(), 3u);
                EXPECT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{2, 3}));
                EXPECT_EQ(tensorToVector<int32_t>(recorded_draft->inputs[0].combo_tokens),
                          (std::vector<int32_t>{2, phase == "update" ? 7 : 0, 4, 5, 8}));
            }
        }
    }
}

TEST_F(PPExecutorTest, LastOrdinaryAllFailedRequestsStillSendTheirErrors) {
    auto params = makeStageParams(2);
    PPExecutor executor(params, nullptr, false);
    auto* wire = attachTransport(executor);
    enqueuePlan(*wire, makePlan(params));
    executor.setModel(std::make_unique<RecordingDraftModel>());
    const ErrorInfo error(ErrorCode::EXECUTION_EXCEPTION, "processor failed");
    auto first = recordState(executor, 101);
    auto second = recordState(executor, 202);
    first->process_error = error;
    second->process_error = error;
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    ASSERT_EQ(wire->sent_tensors.size(), 2u);
    const auto result = lastResult(*wire);
    ASSERT_EQ(result.request_errors.size(), 2u);
    EXPECT_GT(result.sample_input_us, 0);
    for (const auto& returned : result.request_errors) {
        EXPECT_EQ(returned.ToString(), error.ToString());
    }
    EXPECT_TRUE(first->committed_tokens.empty());
    EXPECT_TRUE(second->committed_tokens.empty());
}

TEST_F(PPExecutorTest, LastShutdownPlanCleansFinishedStatesWithoutExecutionOrResult) {
    auto params = makeStageParams(2);
    PPExecutor executor(params, nullptr, false);
    auto* wire = attachTransport(executor);
    auto* recorded = installModel<RecordingStageModel>(executor, params.parallelism_config);
    recordState(executor, 101);
    auto plan = makeShutdownPlan();
    plan.finished_request_ids = {101};
    enqueuePlan(*wire, plan);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_TRUE(recorded->inputs.empty());
    EXPECT_TRUE(executor.stopped_);
    EXPECT_TRUE(executor.sampling_states_.empty());
    EXPECT_TRUE(wire->sent_tensors.empty());
}

/** Last-stage draft execution: PD prefill, verification, failed rows and hidden-state handoff. */
TEST_F(PPExecutorTest, LastPrefillOnlyRunsNormalDraftWorkOnRealKvWithoutTargetSampling) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (auto role : {RoleType::PREFILL, RoleType::PDFUSION}) {
            for (int dp : {1, 2}) {
                for (int width : {1, 3}) {
                    SCOPED_TRACE(::testing::Message() << "type=" << type << ", role=" << role
                                                      << ", dp=" << dp << ", K=" << width);
                    auto params = makeStageParams(2, type, width);
                    params.pd_sep_config.role_type = role;
                    params.parallelism_config.dp_size = dp;
                    params.parallelism_config.world_size *= dp;
                    params.parallelism_config.world_rank *= dp;
                    auto propose = type == SP_TYPE_NONE ? nullptr : makeMtpProposeParams(params);
                    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO,
                                        nullptr, nullptr, propose.get());
                    auto* wire = attachTransport(executor);
                    auto plan = makePlan(params, false, true);
                    plan.model_input.kv_cache_group_tags      = {"full"};
                    plan.model_input.kv_cache_block_id        = intTensor({5, 6, 7, 8}).reshape({1, 2, 2});
                    plan.model_input.kv_cache_kernel_block_id = plan.model_input.kv_cache_block_id.clone();
                    enqueuePlan(*wire, plan);
                    auto* target = installModel<RecordingDraftModel>(executor);
                    int publication_waits = 0;
                    target->wait_publication = [&] { ++publication_waits; return std::string(); };
                    auto* drafts = installDraftModel(executor, type, width, {},
                        [&] { ++publication_waits; return std::string(); });
                    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                    ASSERT_EQ(target->inputs.size(), 1u);
                    EXPECT_TRUE(target->inputs[0].skip_lm_head);
                    EXPECT_TRUE(executor.sampling_states_.empty());
                    EXPECT_TRUE(executor.dspark_q_.empty());
                    EXPECT_EQ(publication_waits, 0);
                    ASSERT_EQ(wire->sent_tensors.size(), 2u);
                    const auto result = lastResult(*wire);
                    EXPECT_EQ(result.new_token_ids.sizes().vec(), (std::vector<int64_t>{2, 1}));
                    EXPECT_EQ(result.new_token_ids.count_nonzero().item<int64_t>(), 0);
                    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{1, 1}));
                    if (type == SP_TYPE_NONE) {
                        EXPECT_FALSE(result.propose_all_probs.defined());
                        continue;
                    }
                    const size_t count = dp == 1 ? 0 : role == RoleType::PREFILL ? 1 :
                        type == SP_TYPE_DSPARK ? 2 : width;
                    ASSERT_EQ(drafts->size(), count);
                    if (count == 0) {
                        EXPECT_TRUE(target->hidden_requests.empty());
                        EXPECT_FALSE(result.propose_all_probs.defined());
                        continue;
                    }
                    EXPECT_EQ(result.propose_all_probs.defined(), type != SP_TYPE_DSPARK && role == RoleType::PREFILL);
                    const auto& first = drafts->front();
                    EXPECT_TRUE(torch::equal(first.kv_cache_block_id.cpu(), plan.model_input.kv_cache_block_id));
                    EXPECT_TRUE(torch::equal(first.last_hidden_states, target->hidden_buffer));
                    EXPECT_EQ(tensorToVector<int32_t>(first.combo_tokens), type == SP_TYPE_DSPARK ?
                        (std::vector<int32_t>{1, 2, 3, 4, 5}) : (std::vector<int32_t>{2, 0, 4, 5, 0}));
                    for (size_t step = 0; step < count; ++step) {
                        const auto& input = drafts->at(step);
                        EXPECT_FALSE(input.skip_lm_head);
                        EXPECT_FALSE(input.pd_separation);
                        EXPECT_FALSE(input.is_fake_stream);
                        EXPECT_EQ(input.input_lengths.numel(), 2);
                        EXPECT_EQ(input.kv_cache_group_tags, plan.model_input.kv_cache_group_tags);
                        EXPECT_TRUE(torch::equal(input.kv_cache_block_id.cpu(), plan.model_input.kv_cache_block_id));
                        EXPECT_TRUE(torch::equal(input.kv_cache_kernel_block_id.cpu(),
                                                plan.model_input.kv_cache_kernel_block_id));
                        if (step == 0) {
                            continue;
                        }
                        EXPECT_FALSE(input.request_id.defined());
                        if (type == SP_TYPE_DSPARK) {
                            EXPECT_EQ(tensorToVector<int32_t>(input.prefix_lengths), (std::vector<int32_t>{2, 3}));
                            EXPECT_EQ(tensorToVector<int32_t>(input.input_lengths),
                                      (std::vector<int32_t>{width, width}));
                        } else {
                            EXPECT_EQ(tensorToVector<int32_t>(input.sequence_lengths),
                                      (std::vector<int32_t>{static_cast<int32_t>(step + 1),
                                                            static_cast<int32_t>(step + 2)}));
                            EXPECT_EQ(input.last_hidden_states.sizes().vec(), (std::vector<int64_t>{2, 4}));
                            const auto first_token = static_cast<int32_t>(10 + 2 * (step - 1));
                            EXPECT_EQ(tensorToVector<int32_t>(input.combo_tokens),
                                      (std::vector<int32_t>{first_token, first_token + 1}));
                        }
                    }
                }
            }
        }
    }
}

TEST_F(PPExecutorTest, LastPrefillProducesOnlyTheDraftWorkRequiredByItsRole) {
    for (auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (auto role : {RoleType::PREFILL, RoleType::PDFUSION}) {
            for (int width : {1, 3}) {
                SCOPED_TRACE(::testing::Message() << "type=" << type << ", role=" << role << ", K=" << width);
                auto params = makeStageParams(2, type, width);
                params.pd_sep_config.role_type = role;
                auto propose = makeMtpProposeParams(params);
                PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
                auto* wire = attachTransport(executor);
                const auto plan = makePlan(params);
                enqueuePlan(*wire, plan);
                auto target = std::make_unique<RecordingCPTargetModel>(false, 1);
                auto* recorded_target = target.get();
                target->next_tokens = {7, 8};
                executor.setModel(std::move(target));
                bool profiling = false;
                int target_waits = 0;
                int draft_waits = 0;
                recorded_target->wait_publication = [&] { ++target_waits; return std::string(); };
                auto* draft_inputs = installDraftModel(
                    executor, type, width, [&] { EXPECT_TRUE(profiling); },
                    [&] { ++draft_waits; return std::string(); });
                const int offset = type == SP_TYPE_EAGLE ? 17 : 0;
                if (offset) {
                    auto mapping = (torch::arange(64, torch::TensorOptions(torch::kLong).device(torch::kCUDA)) + offset)
                                       .remainder(64);
                    executor.fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>(mapping);
                }
                executor.profile_step_start_ = [&] { profiling = true; };
                executor.profile_step_finish_ = [&] {
                    EXPECT_TRUE(profiling);
                    EXPECT_FALSE(draft_inputs->empty());
                    EXPECT_EQ(wire->sent_tensors.size(), 2u);
                    profiling = false;
                };
                recorded_target->on_forward = [&] { EXPECT_TRUE(profiling); };
                ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                EXPECT_FALSE(profiling);
                ASSERT_EQ(wire->sent_tensors.size(), 2u);
                const auto result = lastResult(*wire);
                EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{7, 8}));
                EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{1, 1}));
                const bool commit_only = type == SP_TYPE_DSPARK && role == RoleType::PREFILL;
                EXPECT_EQ(target_waits, commit_only ? 1 : 0);
                EXPECT_EQ(draft_waits, commit_only ? 1 : 0);
                const int count = role == RoleType::PREFILL ? 1 : width;
                ASSERT_EQ(draft_inputs->size(), type == SP_TYPE_DSPARK ? (commit_only ? 1u : 2u) : count);
                const auto& first = draft_inputs->front();
                EXPECT_TRUE(torch::equal(first.last_hidden_states, recorded_target->hidden_buffer));
                EXPECT_TRUE(torch::equal(first.request_id, plan.model_input.request_id));
                if (type == SP_TYPE_DSPARK) {
                    EXPECT_TRUE(first.last_hidden_states.defined());
                    EXPECT_TRUE(torch::equal(first.combo_tokens, plan.model_input.combo_tokens));
                    if (!commit_only) {
                        const auto& next = draft_inputs->back();
                        EXPECT_FALSE(next.last_hidden_states.defined());
                        EXPECT_EQ(tensorToVector<int32_t>(next.prefix_lengths), (std::vector<int32_t>{2, 3}));
                        EXPECT_FALSE(next.request_id.defined());
                        EXPECT_FALSE(next.last_hidden_states.defined());
                    }
                } else {
                    EXPECT_EQ(tensorToVector<int32_t>(first.combo_tokens), (std::vector<int32_t>{2, 7, 4, 5, 8}));
                }
                if (type == SP_TYPE_DSPARK || role != RoleType::PREFILL) {
                    EXPECT_FALSE(result.propose_hidden_states.defined());
                }
                if (commit_only) {
                    EXPECT_FALSE(result.propose_token_ids.defined());
                    EXPECT_FALSE(result.propose_all_probs.defined());
                } else {
                    ASSERT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{2, count}));
                    EXPECT_TRUE(result.propose_token_ids.device().is_cpu());
                    if (type == SP_TYPE_DSPARK) {
                        EXPECT_FALSE(result.propose_all_probs.defined());
                        ASSERT_EQ(executor.dspark_q_.size(), 2u);
                        for (const auto id : {101, 202}) {
                            const auto& q = executor.dspark_q_.at(id);
                            EXPECT_EQ(q.sizes().vec(), (std::vector<int64_t>{count, params.model_config_.vocab_size}));
                            EXPECT_TRUE(q.is_cuda());
                            EXPECT_EQ(q.scalar_type(), torch::kFloat32);
                        }
                    } else if (role == RoleType::PREFILL) {
                        ASSERT_EQ(result.propose_all_probs.sizes().vec(), (std::vector<int64_t>{2, 1, 64}));
                        auto expected_q = torch::zeros({2, 1, 64}, torch::kFloat32);
                        expected_q[0][0][10] = 1;
                        expected_q[1][0][11] = 1;
                        EXPECT_TRUE(torch::equal(result.propose_all_probs, expected_q));
                        ASSERT_EQ(result.propose_hidden_states.sizes().vec(), (std::vector<int64_t>{2, 4}));
                        auto expected_hidden = torch::arange(20, torch::kFloat32).reshape({5, 4})
                                                   .index_select(0, torch::tensor({1, 4}, torch::kLong));
                        EXPECT_TRUE(torch::equal(result.propose_hidden_states, expected_hidden));
                    } else {
                        EXPECT_FALSE(result.propose_all_probs.defined());
                    }
                    for (int row = 0; row < 2; ++row) {
                        for (int step = 0; step < count; ++step) {
                            EXPECT_EQ(result.propose_token_ids[row][step].item<int32_t>(),
                                      type == SP_TYPE_DSPARK ? 10 + row * width + step : 10 + row + 2 * step + offset);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(PPExecutorTest, LastDSparkPublicationWaitsBothWritersBeforeReturningResult) {
    for (const std::string failure : {"", "target error", "draft error", "target throw", "draft throw"}) {
        SCOPED_TRACE(failure);
        auto params = makeStageParams(2, SP_TYPE_DSPARK);
        params.pd_sep_config.role_type = RoleType::PREFILL;
        auto propose = makeMtpProposeParams(params);
        PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
        auto* wire = attachTransport(executor);
        enqueuePlan(*wire, makePlan(params));
        std::vector<std::string> events;
        auto first_state = recordState(executor, 101);
        auto second_state = recordState(executor, 202);
        if (!failure.empty()) {
            second_state->process_error = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "earlier sampling error");
        }
        auto wait = [&](const std::string& name) -> std::string {
            events.push_back(name + ".wait");
            EXPECT_TRUE(wire->sent_tensors.empty());
            EXPECT_TRUE(first_state->committed_tokens.empty());
            EXPECT_TRUE(second_state->committed_tokens.empty());
            if (failure == name + " throw") {
                throw std::runtime_error(failure);
            }
            return failure == name + " error" ? failure : std::string();
        };
        auto target = std::make_unique<RecordingDraftModel>();
        target->on_forward = [&] { events.push_back("target.forward"); };
        target->wait_publication = [&] { return wait("target"); };
        executor.setModel(std::move(target));
        auto draft = installDSparkModels(executor, 3);
        draft->on_forward = [&] { events.push_back("commit.forward"); };
        draft->wait_publication = [&] { return wait("draft"); };
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_EQ(events, (std::vector<std::string>{"target.forward", "commit.forward", "target.wait", "draft.wait"}));
        ASSERT_EQ(draft->inputs.size(), 1u);
        ASSERT_EQ(wire->sent_tensors.size(), 2u);
        const auto result = lastResult(*wire);
        EXPECT_FALSE(result.propose_token_ids.defined());
        EXPECT_FALSE(result.propose_all_probs.defined());
        ASSERT_EQ(result.request_errors.size(), 2u);
        if (failure.empty()) {
            EXPECT_TRUE(result.request_errors[0].ok());
            EXPECT_TRUE(result.request_errors[1].ok());
            EXPECT_EQ(first_state->committed_tokens.size(), 1u);
            EXPECT_EQ(second_state->committed_tokens.size(), 1u);
        } else {
            EXPECT_EQ(result.request_errors[0].code(), ErrorCode::CACHE_STORE_STORE_FAILED);
            EXPECT_EQ(result.request_errors[1].code(), ErrorCode::UNKNOWN_ERROR);
            EXPECT_TRUE(first_state->committed_tokens.empty());
            EXPECT_TRUE(second_state->committed_tokens.empty());
            /** Consume the serialized result through the first-stage dispatcher contract. */
            auto first = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 0);
            auto second = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4, 5}, 0);
            first->setPPInflight();
            second->setPPInflight();
            ASSERT_TRUE(executor.batch_stream_processor_->dispatchSpeculativeExecutionResult(
                            StreamGroups({first, second}), result).ok());
            EXPECT_TRUE(first->hasError());
            EXPECT_TRUE(second->hasError());
            EXPECT_EQ(first->seqLength(), 2);
            EXPECT_EQ(second->seqLength(), 3);
            EXPECT_FALSE(first->isPPInflight());
            EXPECT_FALSE(second->isPPInflight());
        }
    }
}

TEST_F(PPExecutorTest, DSparkPublicationSkipsUpstreamNonPdAndWarmupWork) {
    for (const auto& config : std::vector<std::pair<int, bool>>{{1, false}, {2, false}, {2, true}}) {
        const auto stage = config.first;
        const auto warmup = config.second;
        SCOPED_TRACE(::testing::Message() << "stage=" << stage << ", warmup=" << warmup);
        auto params = makeStageParams(stage, SP_TYPE_DSPARK);
        params.pd_sep_config.role_type = RoleType::PREFILL;
        auto propose = makeMtpProposeParams(params);
        PPExecutor executor(params, nullptr, warmup, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
        auto* wire = attachTransport(executor);
        int waits = 0;
        auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
        target->wait_publication = [&] { ++waits; return std::string(); };
        executor.setModel(std::move(target));
        if (warmup) {
            auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1);
            stream->setIsFakeStream(true);
            ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());
            EXPECT_TRUE(wire->sent_tensors.empty());
        } else {
            auto plan = makePlan(params);
            if (stage == 2) {
                /** A P-role engine can run a batch with no PD cache publication. */
                plan.model_input.pd_separation = false;
                plan.model_input.request_pd_separation.zero_();
                auto draft = installDSparkModels(executor, 3);
                draft->wait_publication = [&] { ++waits; return std::string(); };
            }
            enqueuePlan(*wire, plan);
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
            EXPECT_FALSE(wire->sent_tensors.empty());
        }
        EXPECT_EQ(waits, 0);
    }
}

TEST_F(PPExecutorTpTest, LastDSparkPublicationReducesPeerFailureAfterBothLocalWaits) {
    ReplayTpBroadcast broadcasts;
    int reductions = 0;
    for (int rank : {0, 1}) {
        SCOPED_TRACE(rank);
        if (rank != 0) {
            broadcasts.replay();
        }
        auto params = makeTpStageParams(2, rank, SP_TYPE_DSPARK);
        params.pd_sep_config.role_type = RoleType::PREFILL;
        auto propose = makeMtpProposeParams(params);
        PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
        auto* wire = attachTransport(executor);
        enqueuePlan(*wire, makePlan(params), rank != 0);
        int waits = 0;
        auto target = std::make_unique<RecordingDraftModel>();
        target->wait_publication = [&]() -> std::string {
            ++waits;
            if (rank == 1) {
                throw std::runtime_error("peer target publication failed");
            }
            return {};
        };
        executor.setModel(std::move(target));
        auto draft = installDSparkModels(executor, 3);
        draft->wait_publication = [&] { ++waits; return std::string(); };
        broadcasts.reduce_status = [&](bool local_ok) {
            ++reductions;
            EXPECT_EQ(waits, 2);
            EXPECT_EQ(local_ok, rank == 0);
            EXPECT_TRUE(wire->sent_tensors.empty());
            /** Replay the MIN result when this stage's TP peer failed publication. */
            return false;
        };
        std::shared_ptr<RecordingLogitsProcessor> state;
        if (rank == 0) {
            state = recordState(executor, 101);
        }
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_EQ(waits, 2);
        EXPECT_EQ(reductions, rank + 1);
        if (rank == 0) {
            EXPECT_TRUE(state->committed_tokens.empty());
            const auto result = lastResult(*wire);
            ASSERT_EQ(result.request_errors.size(), 2u);
            for (const auto& error : result.request_errors) {
                EXPECT_EQ(error.code(), ErrorCode::CACHE_STORE_STORE_FAILED);
            }
        } else {
            EXPECT_TRUE(executor.sampling_states_.empty());
            EXPECT_TRUE(wire->sent_tensors.empty());
            EXPECT_EQ(broadcasts.position(), broadcasts.size());
        }
    }
}

TEST_F(PPExecutorTpTest, LastDSparkExceptionDrainsLocalWritersWithoutStatusReduction) {
    ReplayTpBroadcast broadcasts;
    auto params = makeTpStageParams(2, 0, SP_TYPE_DSPARK);
    params.pd_sep_config.role_type = RoleType::PREFILL;
    auto propose = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto* wire = attachTransport(executor);
    std::vector<std::string> waits;
    auto target = std::make_unique<RecordingDraftModel>();
    target->wait_publication = [&]() -> std::string {
        waits.push_back("target");
        throw std::runtime_error("target drain failed");
    };
    executor.setModel(std::move(target));
    auto draft = installDSparkModels(executor, 3);
    draft->on_forward = [] { throw std::runtime_error("commit forward failed"); };
    draft->wait_publication = [&] { waits.push_back("draft"); return std::string(); };
    broadcasts.reduce_status = [](bool local_ok) {
        ADD_FAILURE() << "unwinding must not enter a TP collective";
        return local_ok;
    };
    enqueuePlan(*wire, makePlan(params));
    EXPECT_THROW((void)executor.process(ScheduleOutput{}), std::runtime_error);
    EXPECT_EQ(waits, (std::vector<std::string>{"target", "draft"}));
    EXPECT_TRUE(wire->sent_tensors.empty());
}

TEST_F(PPExecutorTest, PrefillHandoffUsesExplicitDraftOutputWithoutMtpBuffer) {
    auto params = makeStageParams(2, SP_TYPE_MTP, 1);
    params.pd_sep_config.role_type = RoleType::PREFILL;
    auto propose = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto draft = std::make_unique<RecordingDraftModel>();
    draft->expose_mtp_hidden = false;
    executor.draft_model_ = std::move(draft);
    auto plan = makePlan(params);
    PPExecutionResult result;
    result.new_token_ids = intTensor({7, 8}).reshape({2, 1});
    result.new_token_lengths = torch::ones({2}, torch::kInt32);
    const auto target_hidden = torch::full({5, 4}, -100.0f, torch::TensorOptions().device(torch::kCUDA));
    executor.runDraftStep(plan.model_input, plan.draft_next_position_ids, target_hidden,
                          false, plan.sampling_plan, result);
    auto expected = torch::arange(20, torch::kFloat32).reshape({5, 4}).narrow(1, 0, 2)
                        .index_select(0, torch::tensor({1, 4}, torch::kLong));
    ASSERT_EQ(result.propose_hidden_states.sizes().vec(), (std::vector<int64_t>{2, 2}));
    EXPECT_TRUE(torch::equal(result.propose_hidden_states, expected));
}

TEST_F(PPExecutorTest, LastDSparkHistoryDeltasMatchFullHistoryAcrossAcceptedPrefixesAndReordering) {
    constexpr int64_t width = 2;
    auto params = makeStageParams(2, SP_TYPE_DSPARK, width);
    params.model_config_.attn_config.tokens_per_block = 4;
    auto propose = makeMtpProposeParams(params);
    PPExecutor delta(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    PPExecutor full(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto* delta_wire  = attachTransport(delta);
    auto* full_wire   = attachTransport(full);
    auto* delta_model = installModel<RecordingDraftModel>(delta);
    auto* full_model  = installModel<RecordingDraftModel>(full);
    auto delta_draft  = installDSparkModels(delta, width);
    auto full_draft   = installDSparkModels(full, width);
    ResourceContext resources;
    resources.cache_manager =
        std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(3, 32, 4, DataType::TYPE_FP16));
    ASSERT_TRUE(resources.cache_manager->init());
    GenerateConfig config;
    config.max_new_tokens     = 20;
    config.do_sample          = true;
    config.top_k              = 0;
    config.temperature        = .8f;
    config.repetition_penalty = 1.7f;
    config.presence_penalty   = .4f;
    config.frequency_penalty  = .6f;
    config.random_seed        = 123;
    const RequestPrompts prompts{{101, {0, 1, 0, 1}}, {202, {2, 3, 2}}};
    RequestHistories histories{{101, {prompts.at(101)}}, {202, {prompts.at(202)}}};
    auto first = makeStream(resources, params.model_config_, 101, prompts.at(101), config);
    config.random_seed = 456;
    auto second = makeStream(resources, params.model_config_, 202, prompts.at(202), config);
    for (const auto& stream : {first, second}) {
        stream->fakeInitKVBlock(12);
        auto buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
        buffer->propose_step = width;
        buffer->tokens      = torch::zeros({1, width + 1}, torch::kInt32);
        stream->setSPOutputBuffer(buffer);
    }
    PPBatchStreamProcessor producer(params.model_config_, params.pd_sep_config, ProfilingDebugLoggingConfig{},
                                     resources.cache_manager->cacheConfig(), params.sp_config, false);
    std::unordered_map<RequestId, int32_t> cursors;
    std::unordered_map<RequestId, std::shared_ptr<RecordingLogitsProcessor>> delta_processors;
    std::unordered_map<RequestId, std::shared_ptr<RecordingLogitsProcessor>> full_processors;
    const std::vector<std::list<GenerateStreamPtr>> batches{
        {first, second}, {first, second}, {second, first}, {first, second}};
    const auto options = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
    for (size_t round = 0; round < batches.size(); ++round) {
        SCOPED_TRACE(round);
        const StreamGroups groups(batches[round]);
        TensorHolder holder;
        auto gathered = producer.gatherExecutionPlan(groups, {}, false, cursors, holder);
        ASSERT_TRUE(gathered.ok()) << gathered.status().ToString();
        const auto& plan = gathered.value();
        const auto ids = tensorToVector<RequestId>(plan.sampling_plan.request_ids);
        auto reference = fullHistoryPlan(plan, prompts, histories, full);
        if (round > 0) {
            EXPECT_LT(plan.sampling_plan.token_ids.numel(), reference.sampling_plan.token_ids.numel());
        }
        const int64_t score_width = round == 0 ? 1 : width + 1;
        auto logits = torch::full({2 * score_width, 64}, -std::numeric_limits<float>::infinity(), options);
        auto next_q_logits = torch::full({2 * width, 64}, -std::numeric_limits<float>::infinity(), options);
        std::vector<int32_t> accepted_lengths;
        const auto candidates = round == 0 ? torch::Tensor() : plan.model_input.combo_tokens.reshape({2, width + 1});
        for (size_t request = 0; request < ids.size(); ++request) {
            const auto id = ids[request];
            /** Point-mass Q fixes acceptance boundaries; correction/bonus rows still exercise seeded sampling
             * and history penalties. */
            const int reject_at = (round == 1 && id == 101) || (round == 3 && id == 202) ? 1 :
                                  round == 2 && id == 202 ? 0 : -1;
            accepted_lengths.push_back(round == 0 ? 1 : reject_at < 0 ? width + 1 : reject_at + 1);
            for (int64_t step = 0; step < score_width; ++step) {
                auto row = logits[request * score_width + step];
                if (round > 0 && step < width && (reject_at < 0 || step < reject_at)) {
                    row[candidates[request][step + 1].item<int32_t>()] = 0;
                } else {
                    row[histories.at(id).front().back()] = 1.5f;
                    row[32 + round]                     = .5f;
                }
            }
            for (int64_t step = 0; step < width; ++step) {
                const auto token = 10 + (id == 202 ? 4 : 0) + 2 * round + step;
                next_q_logits[request * width + step][token] = 0;
            }
        }
        delta_model->scripted_logits = logits;
        full_model->scripted_logits  = logits;
        delta_draft->scripted_logits = next_q_logits;
        full_draft->scripted_logits  = next_q_logits;
        enqueuePlan(*delta_wire, plan);
        enqueuePlan(*full_wire, reference);
        ASSERT_TRUE(delta.process(ScheduleOutput{}).ok());
        ASSERT_TRUE(full.process(ScheduleOutput{}).ok());
        const auto actual   = lastResult(*delta_wire);
        const auto expected = lastResult(*full_wire);
        EXPECT_EQ(tensorToVector<int32_t>(actual.new_token_lengths), accepted_lengths);
        EXPECT_TRUE(torch::equal(actual.new_token_lengths, expected.new_token_lengths));
        EXPECT_TRUE(torch::equal(actual.propose_token_ids, expected.propose_token_ids));
        for (size_t request = 0; request < ids.size(); ++request) {
            const auto id = ids[request];
            ASSERT_TRUE(actual.request_errors[request].ok()) << actual.request_errors[request].ToString();
            ASSERT_TRUE(expected.request_errors[request].ok()) << expected.request_errors[request].ToString();
            const auto length = actual.new_token_lengths[request].item<int32_t>();
            EXPECT_TRUE(torch::equal(actual.new_token_ids[request].narrow(0, 0, length),
                                     expected.new_token_ids[request].narrow(0, 0, length)));
            EXPECT_EQ(delta.sampling_states_.at(id).token_history, histories.at(id));
            EXPECT_EQ(full.sampling_states_.at(id).token_history, histories.at(id));
            EXPECT_TRUE(torch::equal(delta.sampling_states_.at(id).sampling.generator.get_state(),
                                     full.sampling_states_.at(id).sampling.generator.get_state()));
            EXPECT_TRUE(torch::equal(delta.dspark_q_.at(id), full.dspark_q_.at(id)));
            if (round == 0) {
                auto delta_processor = std::make_shared<RecordingLogitsProcessor>();
                auto full_processor  = std::make_shared<RecordingLogitsProcessor>();
                const auto token = actual.new_token_ids.narrow(0, request, 1);
                ASSERT_FALSE(delta_processor->updateStatus(token, 1).has_value());
                ASSERT_FALSE(full_processor->updateStatus(token, 1).has_value());
                delta.sampling_states_.at(id).sampling.logits_processors = {delta_processor};
                full.sampling_states_.at(id).sampling.logits_processors  = {full_processor};
                delta_processors.emplace(id, std::move(delta_processor));
                full_processors.emplace(id, std::move(full_processor));
            }
        }
        appendCommittedHistory(actual, histories);
        for (const auto id : ids) {
            const auto& history = histories.at(id).front();
            const std::vector<int32_t> outputs(history.begin() + prompts.at(id).size(), history.end());
            EXPECT_EQ(tensorToVector<int32_t>(torch::cat(delta_processors.at(id)->committed_tokens, 1)), outputs);
            EXPECT_EQ(tensorToVector<int32_t>(torch::cat(full_processors.at(id)->committed_tokens, 1)), outputs);
        }
        ASSERT_TRUE(producer.dispatchExecutionResult(groups, actual).ok());
        for (const auto& stream : groups.allStreams()) {
            EXPECT_EQ(stream->completeTokenIdsVec(0), histories.at(stream->streamId()).front());
        }
    }
}

TEST_F(PPExecutorTest, LastDSparkPrefillQIsConsumedByNextVerification) {
    constexpr int width = 2;
    auto params = makeStageParams(2, SP_TYPE_DSPARK, width);
    params.pd_sep_config.role_type = RoleType::PDFUSION;
    auto propose = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto* wire = attachTransport(executor);
    auto* target = installModel<RecordingDraftModel>(executor);
    target->next_tokens = {7, 8};
    auto draft = installDSparkModels(executor, width);
    std::unordered_map<RequestId, int32_t> history_lengths;
    enqueuePlan(*wire, makePlan(params, false, false, history_lengths));
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    const auto prefill = lastResult(*wire);
    ASSERT_EQ(executor.dspark_q_.size(), 2u);
    EXPECT_FALSE(prefill.propose_all_probs.defined());

    auto verify                     = makePlan(params, true, false, history_lengths);
    verify.model_input.combo_tokens = torch::cat({prefill.new_token_ids, prefill.propose_token_ids}, 1).flatten();
    const auto expected = torch::cat({prefill.propose_token_ids, intTensor({30, 31}).reshape({2, 1})}, 1);
    target->next_tokens = tensorToVector<int32_t>(expected);
    draft->on_forward = [&] { EXPECT_TRUE(executor.dspark_q_.empty()); };
    enqueuePlan(*wire, verify);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    const auto result = lastResult(*wire);
    EXPECT_TRUE(result.request_errors[0].ok());
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_TRUE(torch::equal(result.new_token_ids, expected));
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{width + 1, width + 1}));
    EXPECT_FALSE(result.propose_all_probs.defined());
    EXPECT_EQ(executor.dspark_q_.size(), 2u);
    EXPECT_TRUE(executor.dspark_q_.at(101).is_cuda());
}

TEST_F(PPExecutorTest, LastDSparkConsumesOldQBeforeDraftAndDiscardsQAfterUpdateFailure) {
    for (bool fake_cleanup : {false, true}) {
        SCOPED_TRACE(fake_cleanup);
        auto params = makeStageParams(2, SP_TYPE_DSPARK);
        params.pd_sep_config.role_type = RoleType::DECODE;
        auto propose = makeMtpProposeParams(params);
        PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
        auto* wire = attachTransport(executor);
        auto plan = makePlan(params, true);
        cacheDSparkProposals(executor, plan);
        auto failed = recordState(executor, 101);
        recordState(executor, 202);
        failed->update_error = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "state update failed after draft");
        const auto other_q = executor.dspark_q_.at(202);
        executor.dspark_q_.emplace(303, other_q);
        auto* target = installModel<RecordingDraftModel>(executor);
        target->next_tokens = {11, 12, 13, 14, 21, 22, 23, 24};
        auto draft = installDSparkModels(executor, 3);
        draft->on_forward = [&] {
            EXPECT_EQ(executor.dspark_q_.count(101), 0u);
            EXPECT_EQ(executor.dspark_q_.count(202), 0u);
            EXPECT_TRUE(torch::equal(executor.dspark_q_.at(303), other_q));
        };
        failed->on_update = [&] {
            EXPECT_EQ(draft->inputs.size(), 2u);
            EXPECT_TRUE(executor.dspark_q_.at(101).is_cuda());
            EXPECT_TRUE(executor.dspark_q_.at(202).is_cuda());
        };
        enqueuePlan(*wire, plan);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        const auto result = lastResult(*wire);
        EXPECT_EQ(result.request_errors[0].ToString(), failed->update_error->ToString());
        EXPECT_TRUE(result.request_errors[1].ok());
        EXPECT_FALSE(result.propose_all_probs.defined());
        EXPECT_EQ(executor.dspark_q_.count(101), 0u);
        EXPECT_TRUE(executor.dspark_q_.at(202).is_cuda());
        EXPECT_TRUE(torch::equal(executor.dspark_q_.at(303), other_q));

        /** Completion/cancellation can retire the new proposal without another verification. */
        draft->on_forward = {};
        PPExecutionPlan cleanup = fake_cleanup ? makeFakePlan(params, false) : PPExecutionPlan{};
        cleanup.model_input.skip_run = !fake_cleanup;
        cleanup.finished_request_ids = {101, 202, 303};
        enqueuePlan(*wire, cleanup);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        EXPECT_TRUE(executor.dspark_q_.empty());
        EXPECT_TRUE(executor.sampling_states_.empty());

        /** Reused IDs start with zero-padded handoffs; the old Q and failed processor must be gone. */
        plan.model_input.combo_tokens.reshape({2, executor.propose_step_ + 1}).slice(1, 1).zero_();
        enqueuePlan(*wire, plan);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        const auto reused = lastResult(*wire);
        EXPECT_TRUE(reused.request_errors[0].ok());
        EXPECT_TRUE(reused.request_errors[1].ok());
        EXPECT_EQ(executor.dspark_q_.size(), 2u);
    }
}

TEST_F(PPExecutorTest, LastDSparkMissingQFailsRequestsWithoutRecreatingTheirCache) {
    for (bool all_missing : {false, true}) {
        SCOPED_TRACE(all_missing);
        auto params = makeStageParams(2, SP_TYPE_DSPARK);
        params.pd_sep_config.role_type = RoleType::DECODE;
        auto propose = makeMtpProposeParams(params);
        PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
        auto* wire = attachTransport(executor);
        auto plan = makePlan(params, true);
        recordState(executor, 101);
        recordState(executor, 202);
        if (!all_missing) {
            cacheDSparkProposals(executor, plan);
            executor.dspark_q_.erase(202);
        }
        auto* target = installModel<RecordingDraftModel>(executor);
        target->next_tokens = {11, 12, 13, 14, 21, 22, 23, 24};
        auto draft = installDSparkModels(executor, 3);
        enqueuePlan(*wire, plan);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        const auto result = lastResult(*wire);
        EXPECT_EQ(result.request_errors[0].hasError(), all_missing);
        EXPECT_TRUE(result.request_errors[1].hasError());
        EXPECT_NE(result.request_errors[1].ToString().find("missing PP DSpARK proposal probabilities"),
                  std::string::npos);
        EXPECT_EQ(result.new_token_ids[1].count_nonzero().item<int64_t>(), 0);
        EXPECT_EQ(executor.dspark_q_.count(101), all_missing ? 0u : 1u);
        EXPECT_EQ(executor.dspark_q_.count(202), 0u);
        EXPECT_EQ(draft->inputs.size(), 2u);
        EXPECT_FALSE(result.propose_all_probs.defined());
    }
}

TEST_F(PPExecutorTest, LastDSparkHandoffVerifiesZeroCandidatesAndCachesNextQ) {
    constexpr int width = 3;
    auto params = makeStageParams(2, SP_TYPE_DSPARK, width);
    params.pd_sep_config.role_type = RoleType::DECODE;
    auto propose = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto* wire = attachTransport(executor);

    auto plan = makePlan(params, true);
    plan.model_input.combo_tokens      = intTensor({7, 0, 0, 0, 8, 0, 0, 0});
    plan.sampling_plan.do_sample.fill_(true);
    plan.sampling_plan.spec_do_sample.fill_(true);
    plan.sampling_plan.top_k.zero_();
    plan.sampling_plan.top_p.fill_(1);
    plan.model_input.input_lengths     = intTensor({4, 4});
    plan.model_input.prefix_lengths    = intTensor({2, 3});
    plan.model_input.sequence_lengths  = intTensor({});
    plan.model_input.lm_output_indexes = torch::arange(8, torch::kInt32);
    enqueuePlan(*wire, plan);

    auto target = std::make_unique<RecordingDraftModel>();
    auto* recorded_target = target.get();
    auto target_logits = torch::full({8, params.model_config_.vocab_size},
                                     -std::numeric_limits<float>::infinity(),
                                     torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    const std::vector<int32_t> target_tokens{40, 41, 42, 43, 50, 51, 52, 53};
    for (int64_t row = 0; row < target_logits.size(0); ++row) {
        target_logits[row][target_tokens[row]] = 100;
    }
    target->scripted_logits = target_logits;
    executor.setModel(std::move(target));
    auto draft = installDSparkModels(executor, width);
    auto* recorded_draft = draft.get();

    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    const auto result = lastResult(*wire);
    ASSERT_EQ(recorded_target->inputs.size(), 1u);
    EXPECT_TRUE(recorded_target->inputs[0].is_target_verify);
    ASSERT_EQ(result.new_token_ids.sizes().vec(), (std::vector<int64_t>{2, width + 1}));
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids.select(1, 0)), (std::vector<int32_t>{40, 50}));
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{1, 1}));
    ASSERT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{2, width}));
    EXPECT_FALSE(result.propose_all_probs.defined());
    ASSERT_EQ(executor.dspark_q_.size(), 2u);
    EXPECT_EQ(executor.dspark_q_.at(101).sizes().vec(),
              (std::vector<int64_t>{width, params.model_config_.vocab_size}));
    EXPECT_TRUE(executor.dspark_q_.at(101).is_cuda());
    ASSERT_EQ(recorded_draft->inputs.size(), 2u);
    EXPECT_TRUE(recorded_draft->inputs.front().is_target_verify);
    EXPECT_TRUE(recorded_draft->inputs.front().last_hidden_states.defined());
    EXPECT_TRUE(torch::equal(recorded_draft->inputs.front().combo_tokens, plan.model_input.combo_tokens));
    EXPECT_FALSE(recorded_draft->inputs.back().last_hidden_states.defined());
    EXPECT_EQ(tensorToVector<int32_t>(recorded_draft->inputs.back().prefix_lengths),
              (std::vector<int32_t>{3, 4}));
}

TEST_F(PPExecutorTest, LastDSparkMixedQVerifiesBeforeCommitAndFreshProposal) {
    constexpr int width            = 3;
    auto          params           = makeStageParams(2, SP_TYPE_DSPARK, width);
    params.pd_sep_config.role_type = RoleType::DECODE;
    auto       propose             = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto*      wire                       = attachTransport(executor);
    auto       plan                       = makePlan(params, true);
    plan.model_input.combo_tokens         = intTensor({7, 11, 12, 13, 8, 0, 0, 0});
    plan.model_input.engram_token_windows = intTensor({7, 6, 5, 4, 11, 7, 6, 5, 12, 11, 7, 6, 13, 12, 11, 7,
                                                       8, 7, 6, 5, 0,  8, 7, 6, 0,  0,  8, 7, 0,  0,  0,  8})
                                                .reshape({8, 4});
    plan.sampling_plan.do_sample.fill_(true);
    plan.sampling_plan.spec_do_sample.fill_(true);
    plan.sampling_plan.top_k.zero_();
    plan.sampling_plan.top_p.fill_(1);
    plan.sampling_plan.temperature.fill_(1);
    plan.sampling_plan.random_seeds = {123, 456};
    recordState(executor, 101);
    const auto old_q = torch::zeros({width, params.model_config_.vocab_size},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    executor.dspark_q_.emplace(101, old_q);
    auto  target          = std::make_unique<RecordingDraftModel>();
    auto* recorded_target = target.get();
    auto  target_logits   = torch::full({2 * (width + 1), params.model_config_.vocab_size + 8},
                                     -std::numeric_limits<float>::infinity(),
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    /** Padding would dominate every real token if verify used the physical lm-head width. */
    target_logits.narrow(1, params.model_config_.vocab_size, 8).fill_(100);
    /** Q = P makes request 101 accept every candidate; request 202 rejects its zero-padded handoff. */
    for (int step = 0; step < width; ++step) {
        target_logits[step][11 + step] = 0;
        target_logits[step][40]        = 0;
        old_q[step][11 + step]         = .5;
        old_q[step][40]                = .5;
    }
    target_logits[width][43] = 0;
    for (int step = 0; step <= width; ++step) {
        target_logits[width + 1 + step][50 + step] = 0;
    }
    target->scripted_logits = target_logits;
    executor.setModel(std::move(target));
    auto       draft          = installDSparkModels(executor, width);
    auto*      recorded_draft = draft.get();
    const auto original_q     = old_q.clone();
    enqueuePlan(*wire, plan);

    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    const auto result = lastResult(*wire);
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{4, 1}));
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids[0]), (std::vector<int32_t>{11, 12, 13, 43}));
    EXPECT_EQ(result.new_token_ids[1][0].item<int32_t>(), 50);
    EXPECT_TRUE(torch::equal(old_q, original_q));
    ASSERT_EQ(recorded_target->inputs.size(), 1u);
    EXPECT_TRUE(recorded_target->inputs[0].is_target_verify);
    EXPECT_TRUE(torch::equal(recorded_target->inputs[0].engram_token_windows, plan.model_input.engram_token_windows));
    ASSERT_EQ(recorded_draft->inputs.size(), 2u);
    EXPECT_TRUE(recorded_draft->inputs[0].last_hidden_states.defined());
    EXPECT_TRUE(torch::equal(recorded_draft->inputs[0].combo_tokens, plan.model_input.combo_tokens));
    EXPECT_FALSE(recorded_draft->inputs[1].last_hidden_states.defined());
    EXPECT_FALSE(recorded_draft->inputs[0].engram_token_windows.defined());
    EXPECT_FALSE(recorded_draft->inputs[1].engram_token_windows.defined());
    EXPECT_EQ(tensorToVector<int32_t>(recorded_draft->inputs[1].prefix_lengths), (std::vector<int32_t>{6, 4}));
    EXPECT_FALSE(result.propose_all_probs.defined());
    const auto next_q = torch::stack({executor.dspark_q_.at(101), executor.dspark_q_.at(202)}).cpu();
    ASSERT_EQ(next_q.sizes().vec(), (std::vector<int64_t>{2, width, 64}));
    EXPECT_TRUE(torch::equal(next_q.sum(-1), torch::ones({2, width})));
    EXPECT_TRUE(torch::equal(next_q.argmax(-1).to(torch::kInt32), result.propose_token_ids));
    EXPECT_EQ(next_q[1].select(1, 0).sum().item<float>(), 0);
}

TEST_F(PPExecutorTest, LastDSparkProposalUsesRequestSamplingMode) {
    constexpr int width            = 2;
    auto          params           = makeStageParams(2, SP_TYPE_DSPARK, width);
    params.pd_sep_config.role_type = RoleType::DECODE;
    auto       propose             = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto       draft  = installDSparkModels(executor, width);
    auto       logits = torch::full({2 * width, 64},
                              -std::numeric_limits<float>::infinity(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    logits.select(1, 10).fill_(1);
    logits.select(1, 11).fill_(0);
    draft->scripted_logits = logits;
    const auto hidden      = torch::zeros({2 * (width + 1), 4}, logits.options());
    for (const auto& config :
         std::vector<std::pair<bool, bool>>{{false, false}, {false, true}, {true, false}, {true, true}}) {
        const bool do_sample = config.first;
        const bool non_top1  = config.second;
        SCOPED_TRACE(::testing::Message() << "do_sample=" << do_sample << ", non_top1=" << non_top1);
        GenerateConfig cfg;
        cfg.do_sample   = do_sample;
        cfg.top_k       = non_top1 ? 0 : 1;
        cfg.temperature = .5f;
        auto first      = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, cfg);
        auto second     = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4, 5}, cfg);
        for (const auto& stream : {first, second}) {
            stream->update({intTensor({7}).reshape({1, 1}), 1}, false);
            auto proposals          = std::make_shared<SpeculativeExecutorStreamOutput>();
            proposals->propose_step = width;
            proposals->tokens       = intTensor({7, 11, 12}).reshape({1, width + 1});
            stream->setSPOutputBuffer(proposals);
        }
        TensorHolder                           holder;
        std::unordered_map<RequestId, int32_t> history_lengths;
        auto                                   gathered = executor.batch_stream_processor_->gatherExecutionPlan(
            StreamGroups({first, second}), {}, false, history_lengths, holder);
        ASSERT_TRUE(gathered.ok()) << gathered.status().ToString();
        auto&       plan         = gathered.value();
        auto        result       = makeInitializedResult(plan.sampling_plan);
        result.new_token_ids     = intTensor({30, 31}).reshape({2, 1});
        result.new_token_lengths = intTensor({1, 1});
        executor.runDraftStep(plan.model_input, {}, hidden, true, plan.sampling_plan, result);
        const auto  q                    = torch::stack({executor.dspark_q_.at(101), executor.dspark_q_.at(202)});
        const float expected_temperature = do_sample && non_top1 ? .5f : 1.e-6f;
        const auto  expected_q = torch::softmax(logits / expected_temperature, -1).reshape({2, width, 64}).cpu();
        EXPECT_TRUE(q.is_cuda());
        EXPECT_FALSE(result.propose_all_probs.defined());
        EXPECT_TRUE(torch::allclose(q.cpu(), expected_q));
        if (!do_sample || !non_top1) {
            EXPECT_TRUE(torch::equal(result.propose_token_ids, torch::full({2, width}, 10, torch::kInt32)));
        }
    }
}

TEST_F(PPExecutorTest, LastVerifyKeepsAcceptanceForDraftWhileCappingStateUpdates) {
    for (auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        for (int width : {1, 3}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << ", K=" << width);
            auto params = makeStageParams(2, type, width);
            params.pd_sep_config.role_type = RoleType::DECODE;
            auto propose = makeMtpProposeParams(params);
            PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
            auto* wire = attachTransport(executor);
            auto plan = makePlan(params, true);
            cacheDSparkProposals(executor, plan);
            /** Verify the full proposal even when the remaining output budget permits fewer tokens. */
            const int remaining = width == 1 ? 1 : 2;
            plan.sampling_plan.max_tokens[1] = plan.sampling_plan.sequence_lengths[1].item<int32_t>() + remaining;
            enqueuePlan(*wire, plan);
            auto target = std::make_unique<RecordingDraftModel>();
            auto* recorded_target = target.get();
            target->next_tokens = width == 1 ? std::vector<int32_t>{40, 12, 21, 22} :
                                              std::vector<int32_t>{40, 12, 13, 14, 21, 22, 23, 24};
            executor.setModel(std::move(target));
            auto* draft_inputs = installDraftModel(executor, type, width);
            auto first = recordState(executor, 101);
            auto second = recordState(executor, 202);
            const size_t forwards = type == SP_TYPE_DSPARK ? 2 : width;
            first->on_update = [&] { EXPECT_EQ(draft_inputs->size(), forwards); };
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
            const auto result = lastResult(*wire);
            EXPECT_EQ(result.sample_input_us, 0);
            EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{1, width + 1}));
            const auto accepted = torch::arange(21, 22 + width, torch::kInt32);
            EXPECT_TRUE(torch::equal(result.new_token_ids[1], accepted));
            ASSERT_EQ(first->committed_tokens.size(), 1u);
            ASSERT_EQ(second->committed_tokens.size(), 1u);
            EXPECT_EQ(tensorToVector<int32_t>(first->committed_tokens[0]), std::vector<int32_t>{40});
            EXPECT_TRUE(torch::equal(second->committed_tokens[0].flatten(), accepted.narrow(0, 0, remaining)));
            EXPECT_TRUE(result.request_errors[0].ok());
            EXPECT_TRUE(result.request_errors[1].ok());
            ASSERT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{2, width}));
            if (type == SP_TYPE_DSPARK) {
                EXPECT_FALSE(result.propose_all_probs.defined());
                ASSERT_EQ(executor.dspark_q_.size(), 2u);
                EXPECT_TRUE(executor.dspark_q_.at(101).is_cuda());
                EXPECT_EQ(executor.dspark_q_.at(101).sizes().vec(),
                          (std::vector<int64_t>{width, params.model_config_.vocab_size}));
            } else {
                EXPECT_FALSE(result.propose_all_probs.defined());
            }
            ASSERT_EQ(draft_inputs->size(), forwards);
            const auto& commit = draft_inputs->front();
            if (type == SP_TYPE_MTP) {
                EXPECT_TRUE(torch::equal(commit.combo_tokens.cpu(), torch::cat({intTensor({40}), accepted})));
                EXPECT_EQ(tensorToVector<int32_t>(commit.input_lengths), (std::vector<int32_t>{1, width + 1}));
                const auto rows = torch::cat({intTensor({0}), torch::arange(width + 1, 2 * (width + 1), torch::kInt32)})
                                      .to(torch::kCUDA, torch::kLong);
                EXPECT_TRUE(torch::equal(commit.last_hidden_states,
                                         recorded_target->hidden_buffer.index_select(0, rows)));
                for (int step = 1; step < width; ++step) {
                    EXPECT_EQ(tensorToVector<int32_t>(draft_inputs->at(step).sequence_lengths),
                              (std::vector<int32_t>{2 + step, width + 3 + step}));
                }
            } else {
                EXPECT_TRUE(commit.last_hidden_states.defined());
                EXPECT_TRUE(torch::equal(commit.combo_tokens, plan.model_input.combo_tokens));
                const auto& next = draft_inputs->back();
                EXPECT_FALSE(next.last_hidden_states.defined());
                EXPECT_EQ(tensorToVector<int32_t>(next.prefix_lengths), (std::vector<int32_t>{3, width + 4}));
                auto expected = torch::full({2, width}, 63, torch::kInt32);
                expected[0][0] = 40;
                expected[1][0] = 21 + width;
                EXPECT_TRUE(torch::equal(next.combo_tokens.cpu(), expected.flatten()));
            }
            if (type == SP_TYPE_DSPARK) {
                /** The head retires the length-limited row without another decode; an empty plan must release q. */
                PPExecutionPlan cleanup;
                cleanup.model_input.skip_run = true;
                cleanup.finished_request_ids = {202};
                enqueuePlan(*wire, cleanup);
                ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                EXPECT_EQ(executor.dspark_q_.count(202), 0u);
                EXPECT_EQ(executor.dspark_q_.count(101), 1u);
                cleanup.finished_request_ids = {101};
                enqueuePlan(*wire, cleanup);
                ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                EXPECT_TRUE(executor.dspark_q_.empty());
                EXPECT_EQ(draft_inputs->size(), forwards);
            }
        }
    }
}

TEST_F(PPExecutorTest, LastVerifyFailuresKeepDraftRowsForPartialAndWholeBatchErrors) {
    for (auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        for (bool initialization_failed : {false, true}) {
            for (bool all_failed : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "type=" << type << ", init_error=" << initialization_failed
                                                  << ", all_failed=" << all_failed);
                auto params = makeStageParams(2, type);
                params.pd_sep_config.role_type = RoleType::DECODE;
                auto propose = makeMtpProposeParams(params);
                PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
                auto* wire = attachTransport(executor);
                auto plan = makePlan(params, true);
                cacheDSparkProposals(executor, plan);
                auto target = std::make_unique<RecordingDraftModel>();
                target->next_tokens = {11, 40, 13, 14, 21, 22, 23, 24};
                executor.setModel(std::move(target));
                auto* inputs = installDraftModel(executor, type, 3);
                const auto code =
                    initialization_failed ? ErrorCode::INVALID_PARAMS : ErrorCode::GRAMMAR_VERIFY_EXCEPTION;
                const ErrorInfo error(code, "verify failed");
                std::shared_ptr<RecordingLogitsProcessor> healthy;
                std::shared_ptr<RecordingLogitsProcessor> failed;
                if (initialization_failed) {
                    for (int row = all_failed ? 0 : 1; row < 2; ++row) {
                        plan.sampling_plan.logits_processor_configs[row].grammar_type = "regex";
                        plan.sampling_plan.logits_processor_configs[row].grammar_value = "[ab]+";
                        if (type == SP_TYPE_DSPARK) {
                            const auto request_id = plan.sampling_plan.request_ids[row].item<int64_t>();
                            executor.dspark_q_.erase(request_id);
                            plan.model_input.combo_tokens.reshape({2, 4})[row].slice(0, 1).zero_();
                        }
                    }
                    if (!all_failed) {
                        healthy = recordState(executor, 101);
                    }
                } else {
                    healthy = recordState(executor, 101);
                    failed = recordState(executor, 202);
                    failed->verify_error = error;
                    if (all_failed) {
                        healthy->verify_error = error;
                    }
                }
                enqueuePlan(*wire, plan);
                ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                ASSERT_EQ(wire->sent_tensors.size(), 2u);
                const auto result = lastResult(*wire);
                EXPECT_EQ(result.request_errors[0].hasError(), all_failed);
                EXPECT_EQ(result.request_errors[1].code(), error.code());
                EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths),
                          (std::vector<int32_t>{all_failed ? 1 : 2, 1}));
                EXPECT_TRUE(torch::equal(result.new_token_ids[1], torch::zeros_like(result.new_token_ids[1])));
                if (failed) {
                    EXPECT_TRUE(failed->committed_tokens.empty());
                }
                if (healthy) {
                    EXPECT_EQ(healthy->committed_tokens.size(), all_failed ? 0u : 1u);
                }
                EXPECT_EQ(inputs->size(), type == SP_TYPE_DSPARK ? 2u : 3u);
                EXPECT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{2, 3}));
                if (type == SP_TYPE_DSPARK) {
                    EXPECT_EQ(executor.dspark_q_.count(101), all_failed ? 0u : 1u);
                    EXPECT_EQ(executor.dspark_q_.count(202), 0u);
                }
            }
        }
    }
}

TEST_F(PPExecutorTest, LastOrdinaryFakeReturnsAResultButShutdownSkipsTargetAndDraft) {
    for (auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        for (bool decode : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << ", decode=" << decode);
            auto params = makeStageParams(2, type);
            auto propose = makeMtpProposeParams(params);
            PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr,
                                type == SP_TYPE_NONE ? nullptr : propose.get());
            auto* wire = attachTransport(executor);
            auto plan = makeFakePlan(params, decode);
            enqueuePlan(*wire, plan);
            auto* recorded = installModel<RecordingDraftModel>(executor);
            auto* draft_inputs = installDraftModel(executor, type, 3);
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
            ASSERT_EQ(wire->sent_tensors.size(), 2u);
            EXPECT_TRUE(executor.sampling_states_.empty());
            EXPECT_TRUE(executor.dspark_q_.empty());
            const auto result = lastResult(*wire);
            EXPECT_EQ(result.sample_input_us, 0);
            const int token_count = draft_inputs && decode ? 4 : 1;
            EXPECT_EQ(result.new_token_ids.sizes().vec(), (std::vector<int64_t>{1, token_count}));
            EXPECT_EQ(result.new_token_lengths.item<int32_t>(), token_count);
            if (draft_inputs) {
                EXPECT_EQ(draft_inputs->size(), type == SP_TYPE_DSPARK ? 2u : 3u);
                EXPECT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{1, 3}));
            } else {
                EXPECT_FALSE(result.propose_token_ids.defined());
            }

            enqueuePlan(*wire, makeShutdownPlan());
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
            EXPECT_TRUE(executor.stopped_);
            EXPECT_EQ(recorded->inputs.size(), 1u);
            EXPECT_EQ(wire->sent_tensors.size(), 2u);
            EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
            EXPECT_TRUE(executor.sampling_states_.empty());
            EXPECT_TRUE(executor.dspark_q_.empty());
            if (draft_inputs) {
                EXPECT_EQ(draft_inputs->size(), type == SP_TYPE_DSPARK ? 2u : 3u);
            }
            ASSERT_TRUE(executor.finish().ok());
            EXPECT_EQ(wire->sent_tensors.size(), 2u);
        }
    }
}

TEST_F(PPExecutorTest, LastVerifyIgnoresSamplerFailuresAfterTheRejectedPrefix) {
    auto params = makeStageParams(2, SP_TYPE_MTP);
    auto propose = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto* wire = attachTransport(executor);
    auto plan = makePlan(params, true);
    plan.sampling_plan.do_sample.fill_(true);
    plan.sampling_plan.top_k.fill_(2);
    plan.sampling_plan.spec_do_sample.fill_(false);
    enqueuePlan(*wire, plan);
    auto target = std::make_unique<RecordingDraftModel>();
    auto logits = torch::full({8, 64}, -std::numeric_limits<float>::infinity(),
                              torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    const std::vector<int> tokens{11, 12, 13, 14, 40, 22, 23, 24};
    for (int row = 0; row < 8; ++row) {
        logits[row][tokens[row]] = 100;
    }
    /** Row 4 already rejects the second request's proposal; row 7 is never committed. */
    logits[7].fill_(-std::numeric_limits<float>::infinity());
    target->scripted_logits = logits;
    executor.setModel(std::move(target));
    executor.draft_model_ = std::make_unique<RecordingDraftModel>();
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    const auto result = lastResult(*wire);
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{4, 1}));
    EXPECT_TRUE(result.request_errors[0].ok());
    EXPECT_TRUE(result.request_errors[1].ok());
    EXPECT_EQ(result.new_token_ids[1][0].item<int32_t>(), 40);
    EXPECT_TRUE(result.propose_token_ids.defined());
}

TEST_F(PPExecutorTest, LastDraftForwardUsesDraftCacheStrides) {
    for (bool multiple_groups : {false, true}) {
        SCOPED_TRACE(multiple_groups);
        auto       params       = makeStageParams(2, SP_TYPE_MTP, 1);
        auto       cache_config = test::makeSimpleMhaCacheConfig(1, 16, 4, DataType::TYPE_FP16);
        const auto draft_config = cache_config;
        if (multiple_groups) {
            cache_config.layer_num = 2;
            cache_config.fromGroupedSpecs(
                {test::makeMhaSpec("default", 4, TYPE_FP16, 1, 1), test::makeMhaSpec("extra", 4, TYPE_FP16, 1, 1)},
                {{0}, {1}},
                {CacheGroupType::FULL, CacheGroupType::FULL});
        }
        cache_config.mtp_sub_configs.push_back(cache_config.mergeMTPModule(draft_config, 0, cache_config.layer_num));
        cache_config.finalizeBlockNums(16, RuntimeConfig{});
        auto cache = std::make_shared<KVCacheManager>(cache_config);
        ASSERT_TRUE(cache->init());
        PPExecutor executor(params, cache, false);
        auto       draft                       = std::make_unique<RecordingDraftModel>();
        auto*      recorded                    = draft.get();
        executor.draft_model_                  = std::move(draft);
        executor.fast_topk_sampler_            = std::make_unique<speculative::FastTopKSampler>();
        auto plan                              = makePlan(params);
        plan.model_input.kv_block_stride_bytes = 17;
        plan.model_input.kv_scale_stride_bytes = 19;
        const auto        hidden = torch::zeros({5, 4}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
        PPExecutionResult execution_result;
        execution_result.new_token_ids     = intTensor({7, 8}).reshape({2, 1});
        execution_result.new_token_lengths = intTensor({1, 1});
        executor.runDraftStep(plan.model_input, {}, hidden, false, plan.sampling_plan, execution_result);
        ASSERT_EQ(recorded->inputs.size(), 1u);
        const auto& expected = cache->getMTPModuleCacheConfig(0);
        ASSERT_EQ(expected.groupNums(), multiple_groups ? 2 : 1);
        if (multiple_groups) {
            /** Multiple cache groups require group-specific strides, including groups without local draft layers. */
            EXPECT_TRUE(expected.layerIdsForGroup("extra").empty());
            EXPECT_EQ(recorded->inputs[0].kv_block_stride_bytes, 0u);
            EXPECT_EQ(recorded->inputs[0].kv_scale_stride_bytes, 0u);
        } else {
            EXPECT_EQ(recorded->inputs[0].kv_block_stride_bytes, expected.group("default").kvBlockStrideBytes());
            EXPECT_EQ(recorded->inputs[0].kv_scale_stride_bytes, expected.group("default").kvScaleStrideBytes());
        }
        EXPECT_EQ(plan.model_input.kv_block_stride_bytes, 17);
        EXPECT_EQ(plan.model_input.kv_scale_stride_bytes, 19);
    }
}

TEST_F(PPExecutorTest, LastDraftChainUsesTargetAndPreviousDraftHidden) {
    for (bool expose_mtp_hidden : {false, true}) {
        SCOPED_TRACE(expose_mtp_hidden);
        auto       params = makeStageParams(2, SP_TYPE_MTP);
        PPExecutor executor(params, nullptr, false);
        auto       model                = std::make_unique<RecordingDraftModel>();
        auto*      recorded             = model.get();
        recorded->expose_mtp_hidden     = expose_mtp_hidden;
        executor.draft_model_           = std::move(model);
        executor.fast_topk_sampler_     = std::make_unique<speculative::FastTopKSampler>();
        const int64_t hidden_width      = expose_mtp_hidden ? 4 : 2;
        const auto target_hidden =
            torch::arange(20, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA))
                .reshape({5, 4})
                .narrow(1, 0, hidden_width);

        GptModelInputs target_input;
        target_input.combo_tokens          = intTensor({0, 1, 0, 3, 4});
        target_input.input_lengths         = intTensor({2, 3});
        target_input.prefix_lengths        = intTensor({3, 5});
        target_input.sequence_lengths      = intTensor({});
        target_input.lm_output_indexes     = intTensor({1, 4});
        target_input.request_id            = torch::tensor({101, 202}, torch::kInt64);
        target_input.request_pd_separation = torch::ones({2}, torch::kBool);
        target_input.cache_keys            = torch::ones({2, 3}, torch::kInt64);
        target_input.combo_position_ids    = intTensor({3, 4, 5, 6, 7});
        executor.position_id_len_factor_   = 1;

        PPSamplingPlan   sampling_plan;
        PPExecutionResult execution_result;
        execution_result.new_token_ids     = intTensor({2, 5}).reshape({2, 1});
        execution_result.new_token_lengths = intTensor({1, 1});
        executor.runDraftStep(target_input,
                              intTensor({5, 8}),
                              target_hidden,
                              false,
                              sampling_plan,
                              execution_result);
        const auto& proposed = execution_result.propose_token_ids;
        ASSERT_EQ(recorded->inputs.size(), 3u);
        EXPECT_EQ(recorded->hidden_requests, (std::vector<int64_t>{5, 2, 2}));
        EXPECT_TRUE(proposed.device().is_cpu());
        EXPECT_EQ(proposed.scalar_type(), torch::kInt32);
        EXPECT_TRUE(torch::equal(proposed, intTensor({10, 12, 14, 11, 13, 15}).reshape({2, 3})));
        EXPECT_TRUE(torch::equal(recorded->inputs.front().last_hidden_states, target_hidden));
        EXPECT_EQ(tensorToVector<int32_t>(recorded->inputs.front().combo_tokens),
                  (std::vector<int32_t>{1, 2, 3, 4, 5}));
        EXPECT_EQ(tensorToVector<int32_t>(recorded->inputs.front().combo_position_ids),
                  (std::vector<int32_t>{4, 5, 6, 7, 8}));
        for (int64_t step = 1; step < 3; ++step) {
            const auto& input = recorded->inputs[step];
            EXPECT_TRUE(torch::equal(input.combo_tokens.cpu(), proposed.select(1, step - 1)));
            EXPECT_TRUE(torch::equal(input.input_lengths.cpu(), intTensor({1, 1})));
            EXPECT_EQ(tensorToVector<int32_t>(input.combo_position_ids),
                      (std::vector<int32_t>{5 + static_cast<int32_t>(step), 8 + static_cast<int32_t>(step)}));
            EXPECT_FALSE(input.request_id.defined());
            EXPECT_FALSE(input.request_pd_separation.defined());
            EXPECT_FALSE(input.cache_keys.defined());
            auto expected_hidden = step == 1 ? torch::tensor({{4.f, 5.f, 6.f, 7.f}, {16.f, 17.f, 18.f, 19.f}}) :
                                               torch::arange(8, torch::kFloat32).reshape({2, 4}) + 100;
            EXPECT_TRUE(torch::equal(input.last_hidden_states.cpu(), expected_hidden.narrow(1, 0, hidden_width)));
        }
        EXPECT_TRUE(torch::equal(target_hidden.cpu(),
                                 torch::arange(20, torch::kFloat32).reshape({5, 4}).narrow(1, 0, hidden_width)));
    }
}

TEST_F(PPExecutorTpTest, LastDraftDecodeSynchronizesOnlyItsInitialInput) {
    for (const auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE}) {
        for (const int64_t width : {1, 2, 3}) {
            for (const bool expose_mtp_hidden : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "type=" << type << ", width=" << width
                                                  << ", mtp_hidden=" << expose_mtp_hidden);
                ReplayTpBroadcast broadcasts;
                std::vector<size_t> root_forward_positions;
                for (const int rank : {0, 1}) {
                    SCOPED_TRACE(rank);
                    if (rank != 0) {
                        broadcasts.replay();
                    }
                    auto params  = makeTpStageParams(2, rank, type, width);
                    params.model_config_.attn_config.rope_config.index_factor = 2;
                    auto propose = makeMtpProposeParams(params);
                    propose->mtp_model_params_->front()->gpt_weights.d2t_map =
                        63 - torch::arange(64, torch::TensorOptions(torch::kInt64).device(torch::kCUDA));
                    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
                    ASSERT_NE(executor.fast_topk_sampler_, nullptr);

                    auto  model                = std::make_unique<RecordingDraftModel>();
                    auto* recorded             = model.get();
                    recorded->expose_mtp_hidden = expose_mtp_hidden;
                    /** Rank-specific hidden values catch accidental rebroadcasts. */
                    recorded->hidden_offset = rank * 1000;
                    std::vector<size_t> forward_positions;
                    recorded->on_forward = [&] { forward_positions.push_back(broadcasts.position()); };
                    executor.draft_model_ = std::move(model);

                    GptModelInputs input;
                    input.combo_tokens             = intTensor({0, 1, 0, 3, 4});
                    input.input_lengths            = intTensor({2, 3});
                    input.prefix_lengths           = intTensor({3, 5});
                    input.sequence_lengths         = intTensor({});
                    input.lm_output_indexes        = intTensor({1, 4});
                    input.combo_position_ids       = intTensor({3, 103, 4, 104, 5, 105, 6, 106, 7, 107});
                    input.kv_cache_block_id         = intTensor({1, 2, 3, 4, 5, 6}).reshape({2, 3});
                    input.kv_cache_kernel_block_id = input.kv_cache_block_id.clone();
                    const int64_t hidden_width = expose_mtp_hidden ? 4 : 2;
                    auto target_hidden = torch::arange(20, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA))
                                             .reshape({5, 4})
                                             .narrow(1, 0, hidden_width);
                    PPExecutionResult result;
                    if (rank == 0) {
                        result.new_token_ids     = intTensor({2, 5}).reshape({2, 1});
                        result.new_token_lengths = intTensor({1, 1});
                    }
                    executor.runDraftStep(
                        input, intTensor({5, 105, 8, 108}), target_hidden, false, {}, result);

                    ASSERT_EQ(recorded->inputs.size(), static_cast<size_t>(width));
                    ASSERT_EQ(forward_positions.size(), static_cast<size_t>(width));
                    EXPECT_GT(forward_positions.front(), 0u);
                    if (width > 1) {
                        EXPECT_GT(forward_positions[1], forward_positions[0]);
                    }
                    /** Only the prefill-to-decode transition broadcasts; later forwards reuse local state. */
                    for (int64_t step = 2; step < width; ++step) {
                        EXPECT_EQ(forward_positions[step], forward_positions[1]);
                    }
                    EXPECT_EQ(broadcasts.position(), forward_positions.back());
                    for (int64_t step = 1; step < width; ++step) {
                        const auto& decode = recorded->inputs[step];
                        const auto  offset = static_cast<int32_t>(step);
                        EXPECT_TRUE(torch::equal(decode.combo_tokens.cpu(), intTensor({55 - 2 * offset, 54 - 2 * offset})));
                        EXPECT_TRUE(torch::equal(decode.input_lengths.cpu(), intTensor({1, 1})));
                        EXPECT_TRUE(torch::equal(decode.sequence_lengths.cpu(),
                                                 intTensor({4 + offset, 7 + offset})));
                        EXPECT_TRUE(torch::equal(decode.combo_position_ids.cpu(),
                                                 intTensor({5 + offset, 105 + offset, 8 + offset, 108 + offset})));
                        EXPECT_TRUE(torch::equal(decode.kv_cache_block_id.cpu(), input.kv_cache_block_id));
                        EXPECT_TRUE(
                            torch::equal(decode.kv_cache_kernel_block_id.cpu(), input.kv_cache_kernel_block_id));
                        auto expected_hidden = step == 1 ?
                            torch::tensor({{4.f, 5.f, 6.f, 7.f}, {16.f, 17.f, 18.f, 19.f}}) :
                            torch::arange(8, torch::kFloat32).reshape({2, 4}) + (step - 1) * 100 + rank * 1000;
                        EXPECT_TRUE(
                            torch::equal(decode.last_hidden_states.cpu(), expected_hidden.narrow(1, 0, hidden_width)));
                    }
                    if (rank == 0) {
                        root_forward_positions = forward_positions;
                        std::vector<int32_t> expected_tokens;
                        for (int row = 0; row < 2; ++row) {
                            for (int64_t step = 0; step < width; ++step) {
                                expected_tokens.push_back(static_cast<int32_t>(53 - 2 * step - row));
                            }
                        }
                        EXPECT_TRUE(torch::equal(result.propose_token_ids,
                                                 intTensor(std::move(expected_tokens)).reshape({2, width})));
                    } else {
                        EXPECT_EQ(forward_positions, root_forward_positions);
                        EXPECT_EQ(broadcasts.position(), broadcasts.size());
                        EXPECT_FALSE(result.propose_token_ids.defined());
                        EXPECT_FALSE(result.propose_all_probs.defined());
                    }
                    EXPECT_TRUE(executor.sampling_states_.empty());
                    EXPECT_TRUE(executor.dspark_q_.empty());
                }
            }
        }
    }
}

/** Last-stage TP/CP: replay one stage's root broadcasts and retain rank-local CP features. */
TEST_F(PPExecutorTpTest, LastTpVerifySynchronizesAcceptedRowsAndKeepsPeersOutOfTargetSampling) {
    for (auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        for (bool all_failed : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << ", all_failed=" << all_failed);
            ReplayTpBroadcast broadcasts;
            std::vector<GptModelInputs> root_inputs;
            torch::Tensor root_hidden;
            std::vector<size_t> root_forward_positions;
            for (int rank : {0, 1}) {
                SCOPED_TRACE(rank);
                if (rank != 0) {
                    broadcasts.replay();
                }
                auto params = makeTpStageParams(2, rank, type);
                if (type == SP_TYPE_MTP) {
                    params.model_config_.attn_config.rope_config.index_factor = 2;
                }
                auto propose = makeMtpProposeParams(params);
                if (type == SP_TYPE_DSPARK) {
                    propose->mtp_model_params_->front()->gpt_weights.dspark_markov_w2 =
                        torch::zeros({128, 1}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
                }
                PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
                ASSERT_NE(executor.speculative_sampler_, nullptr);
                ASSERT_NE(executor.spec_logits_verify_runner_, nullptr);
                if (type == SP_TYPE_DSPARK) {
                    const auto& weights = propose->mtp_model_params_->front()->gpt_weights;
                    EXPECT_EQ(executor.dspark_markov_w1_.data_ptr(), weights.dspark_markov_w1.data_ptr());
                    EXPECT_EQ(executor.dspark_markov_w2_.data_ptr(), weights.dspark_markov_w2.data_ptr());
                    EXPECT_EQ(executor.dspark_markov_w2_.size(0), 64);
                }
                auto plan = makePlan(params, true);
                if (type == SP_TYPE_MTP) {
                    const auto positions = intTensor({2, 3, 4, 5, 3, 4, 5, 6});
                    plan.model_input.combo_position_ids = torch::stack({positions, positions + 100}, -1).flatten();
                }
                auto* wire = attachTransport(executor);
                enqueuePlan(*wire, plan, rank != 0);
                std::vector<size_t> forward_positions;
                auto target = std::make_unique<RecordingCPTargetModel>(false, rank);
                auto* recorded_target = target.get();
                target->next_tokens = {11, 40, 13, 14, 21, 22, 23, 24};
                target->on_forward = [&] { forward_positions.push_back(broadcasts.position()); };
                executor.setModel(std::move(target));
                auto* draft_inputs = installDraftModel(
                    executor, type, 3, [&] { forward_positions.push_back(broadcasts.position()); });
                if (rank == 0) {
                    cacheDSparkProposals(executor, plan);
                    auto healthy         = recordState(executor, 101);
                    auto failed = recordState(executor, 202);
                    failed->verify_error = ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION, "verify failed");
                    if (all_failed) {
                        healthy->verify_error = failed->verify_error;
                    }
                }
                ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                ASSERT_EQ(recorded_target->inputs.size(), 1u);
                EXPECT_TRUE(recorded_target->inputs[0].is_target_verify);
                EXPECT_FALSE(recorded_target->inputs[0].pp_intermediates.empty());
                ASSERT_EQ(draft_inputs->size(), type == SP_TYPE_MTP ? 3u : 2u);
                ASSERT_EQ(forward_positions.size(), 1 + draft_inputs->size());
                if (type == SP_TYPE_MTP) {
                    /** Draft decode stays local after accepted-row synchronization. */
                    EXPECT_EQ(forward_positions[3], forward_positions[2]);
                }
                for (const auto& input : *draft_inputs) {
                    EXPECT_TRUE(input.pp_intermediates.empty());
                }
                /** Non-root has no sampling plan; its COMMIT phase comes from the TP-synchronized input. */
                EXPECT_EQ(draft_inputs->front().is_target_verify, type == SP_TYPE_DSPARK);
                if (rank == 0) {
                    root_hidden = recorded_target->hidden_buffer.clone();
                    root_inputs = *draft_inputs;
                    root_forward_positions = forward_positions;
                    ASSERT_EQ(wire->sent_tensors.size(), 2u);
                    EXPECT_EQ(tensorToVector<int32_t>(lastResult(*wire).new_token_lengths),
                              (std::vector<int32_t>{all_failed ? 1 : 2, 1}));
                } else {
                    EXPECT_TRUE(wire->sent_tensors.empty());
                    EXPECT_TRUE(executor.sampling_states_.empty());
                    EXPECT_TRUE(executor.dspark_q_.empty());
                    EXPECT_EQ(broadcasts.position(), broadcasts.size());
                    EXPECT_EQ(forward_positions, root_forward_positions);
                    for (size_t step = 0; step < draft_inputs->size(); ++step) {
                        EXPECT_TRUE(torch::equal(draft_inputs->at(step).combo_tokens.cpu(),
                                                 root_inputs[step].combo_tokens.cpu()));
                        EXPECT_TRUE(torch::equal(draft_inputs->at(step).input_lengths.cpu(),
                                                 root_inputs[step].input_lengths.cpu()));
                        EXPECT_TRUE(torch::equal(draft_inputs->at(step).prefix_lengths.cpu(),
                                                 root_inputs[step].prefix_lengths.cpu()));
                        EXPECT_TRUE(torch::equal(draft_inputs->at(step).sequence_lengths.cpu(),
                                                 root_inputs[step].sequence_lengths.cpu()));
                        if (type == SP_TYPE_MTP) {
                            EXPECT_TRUE(torch::equal(draft_inputs->at(step).combo_position_ids.cpu(),
                                                     root_inputs[step].combo_position_ids.cpu()));
                        }
                    }
                }
                if (type == SP_TYPE_MTP) {
                    const auto rows =
                        (all_failed ? intTensor({0, 4}) : intTensor({0, 1, 4})).to(torch::kCUDA, torch::kLong);
                    EXPECT_TRUE(torch::equal(draft_inputs->front().last_hidden_states,
                                             root_hidden.index_select(0, rows)));
                    const auto expected_positions =
                        all_failed ? intTensor({2, 102, 3, 103}) : intTensor({2, 102, 3, 103, 3, 103});
                    EXPECT_TRUE(torch::equal(draft_inputs->front().combo_position_ids.cpu(), expected_positions));
                } else {
                    EXPECT_TRUE(torch::equal(draft_inputs->front().last_hidden_states, recorded_target->hidden_buffer));
                }
            }
        }
    }
}

TEST_F(PPExecutorTpTest, PrefillOnlyTpPeersFollowTheSameDraftScheduleAndKeepCpFeaturesLocal) {
    for (auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        for (bool cp : {false, true}) {
            for (int dp : {1, 2}) {
                SCOPED_TRACE(::testing::Message() << "type=" << type << ", cp=" << cp << ", dp=" << dp);
                ReplayTpBroadcast broadcasts;
                broadcasts.reduce_status = [](bool local_ok) {
                    ADD_FAILURE() << "prefill-only must not wait for PD publication";
                    return local_ok;
                };
                for (int rank : {0, 1}) {
                    SCOPED_TRACE(rank);
                    if (rank != 0) {
                        broadcasts.replay();
                    }
                    auto params = makeTpStageParams(2, rank, type);
                    params.parallelism_config.dp_size = dp;
                    params.parallelism_config.world_size *= dp;
                    params.parallelism_config.world_rank = 4 * dp + rank;
                    params.pd_sep_config.role_type = cp ? RoleType::PREFILL : RoleType::PDFUSION;
                    if (cp) {
                        params.parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
                    }
                    auto propose = makeMtpProposeParams(params);
                    if (type == SP_TYPE_DSPARK && params.pd_sep_config.role_type == RoleType::PREFILL) {
                        /** Commit-only checkpoints omit proposal weights on every TP rank. */
                        propose->mtp_model_params_->front()->gpt_weights.dspark_markov_w1 = torch::Tensor();
                        propose->mtp_model_params_->front()->gpt_weights.dspark_markov_w2 = torch::Tensor();
                    }
                    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO,
                                        nullptr, nullptr, propose.get());
                    auto* wire = attachTransport(executor);
                    const auto plan = makePlan(params, false, true);
                    enqueuePlan(*wire, plan, rank != 0);
                    auto* target = installModel<RecordingCPTargetModel>(executor, cp, rank);
                    auto* drafts = installDraftModel(executor, type, 3);
                    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                    ASSERT_EQ(target->inputs.size(), 1u);
                    EXPECT_TRUE(target->inputs[0].skip_lm_head);
                    const size_t count = dp == 1 ? 0 : cp ? 1 : type == SP_TYPE_DSPARK ? 2 : 3;
                    ASSERT_EQ(drafts->size(), count);
                    for (const auto& input : *drafts) {
                        EXPECT_FALSE(input.skip_lm_head);
                        EXPECT_FALSE(input.pd_separation);
                    }
                    if (count != 0) {
                        const auto& first = drafts->front();
                        EXPECT_EQ(first.combo_tokens.numel(), 5);
                        EXPECT_TRUE(torch::equal(first.input_lengths.cpu(), plan.model_input.input_lengths));
                        const auto expected_hidden = cp || type == SP_TYPE_DSPARK ? target->hidden_buffer :
                            torch::arange(20, target->hidden_buffer.options()).reshape({5, 4});
                        EXPECT_TRUE(torch::equal(first.last_hidden_states, expected_hidden));
                        EXPECT_TRUE(first.pp_intermediates.empty());
                    }
                    EXPECT_TRUE(executor.sampling_states_.empty());
                    EXPECT_TRUE(executor.dspark_q_.empty());
                    if (rank == 0) {
                        EXPECT_EQ(lastResult(*wire).new_token_ids.sizes().vec(), (std::vector<int64_t>{2, 1}));
                    } else {
                        EXPECT_TRUE(wire->sent_tensors.empty());
                        EXPECT_EQ(broadcasts.position(), broadcasts.size());
                    }
                }
            }
        }
    }
}

TEST_F(PPExecutorTpTest, LastCpPrefillPreservesGlobalInputsAndRetainsRankLocalHidden) {
    for (auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        for (bool fake : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << ", fake=" << fake);
            ReplayTpBroadcast broadcasts;
            for (int rank : {0, 1}) {
                SCOPED_TRACE(rank);
                if (rank != 0) {
                    broadcasts.replay();
                }
                auto params = makeTpStageParams(2, rank, type);
                params.pd_sep_config.role_type = RoleType::PREFILL;
                params.parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
                auto propose = makeMtpProposeParams(params);
                PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
                const auto plan = fake ? makeFakePlan(params, false) : makePlan(params);
                auto* wire = attachTransport(executor);
                enqueuePlan(*wire, plan, rank != 0);
                auto* recorded_target = installModel<RecordingCPTargetModel>(executor, true, rank);
                std::vector<GptModelInputs>* draft_inputs;
                RecordingDraftModel* mtp_model = nullptr;
                if (type == SP_TYPE_DSPARK) {
                    auto draft = installDSparkModels(executor, 3);
                    draft_inputs = &draft->inputs;
                } else {
                    auto draft = std::make_unique<RecordingDraftModel>();
                    mtp_model = draft.get();
                    draft_inputs = &draft->inputs;
                    executor.draft_model_ = std::move(draft);
                }
                int reductions = 0;
                broadcasts.reduce_status = [&](bool local_ok) {
                    ++reductions;
                    EXPECT_TRUE(local_ok);
                    return local_ok;
                };
                ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                EXPECT_EQ(reductions, type == SP_TYPE_DSPARK && plan.model_input.pd_separation ? 1 : 0);
                ASSERT_EQ(draft_inputs->size(), 1u);
                const auto& input = draft_inputs->front();
                ASSERT_EQ(recorded_target->inputs.size(), 1u);
                EXPECT_FALSE(recorded_target->inputs[0].pp_intermediates.empty());
                EXPECT_TRUE(input.pp_intermediates.empty());
                EXPECT_TRUE(torch::equal(input.input_lengths.cpu(), plan.model_input.input_lengths));
                EXPECT_EQ(input.combo_tokens.numel(), plan.model_input.combo_tokens.numel());
                if (type == SP_TYPE_DSPARK) {
                    EXPECT_TRUE(torch::equal(input.combo_tokens.cpu(), plan.model_input.combo_tokens));
                }
                EXPECT_TRUE(torch::equal(input.last_hidden_states, recorded_target->hidden_buffer));
                EXPECT_EQ(recorded_target->hidden_requests, std::vector<int64_t>{-1});
                EXPECT_EQ(tensorToVector<int32_t>(recorded_target->target_input_lengths),
                          fake ? std::vector<int32_t>{2} : (std::vector<int32_t>{2, 2}));
                EXPECT_EQ(input.last_hidden_states.size(0), fake ? 2 : 4);
                if (mtp_model) {
                    EXPECT_TRUE(mtp_model->hidden_requests.empty());
                    EXPECT_EQ(mtp_model->last_hidden_requests, std::vector<int64_t>{fake ? 1 : 2});
                } else {
                    EXPECT_TRUE(input.last_hidden_states.defined());
                }
                if (rank == 0) {
                    ASSERT_EQ(wire->sent_tensors.size(), 2u);
                    const auto result = lastResult(*wire);
                    if (type == SP_TYPE_MTP) {
                        EXPECT_EQ(result.propose_token_ids.sizes().vec(), (std::vector<int64_t>{fake ? 1 : 2, 1}));
                        ASSERT_TRUE(result.propose_hidden_states.defined());
                        auto expected_hidden = mtp_model->hidden_buffer.index_select(
                            0, input.lm_output_indexes.to(mtp_model->hidden_buffer.device(), torch::kLong));
                        EXPECT_EQ(result.propose_hidden_states.size(1), 4);
                        EXPECT_TRUE(torch::equal(result.propose_hidden_states, expected_hidden.cpu()));
                    } else {
                        EXPECT_FALSE(result.propose_token_ids.defined());
                    }
                } else {
                    EXPECT_TRUE(wire->sent_tensors.empty());
                    EXPECT_EQ(broadcasts.position(), broadcasts.size());
                }
                EXPECT_EQ(executor.sampling_states_.empty(), fake || rank != 0);
            }
        }
    }
}

}  /** namespace rtp_llm */
