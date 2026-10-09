/**
 * Single-stage PPExecutor tests, ordered by ownership:
 *   Common: construction, warmup, communication waits and profiler boundaries.
 *   First: stream preparation, inflight slots, result consumption and batch metrics.
 *   Middle: local target execution and activation handoff.
 *   Last: sampling state, request errors, target verification and draft execution.
 * Add PD, MTP/EAGLE, DSpARK and TP/CP cases beside the stage behavior they change.
 *
 * Each execution case owns one PP stage. Supply legal plans/results through the
 * test transport and observe that stage's model calls, state and outgoing result.
 * Prefer process() for orchestration; use helpers directly for focused draft or
 * metric boundaries. TP replay checks local root/peer behavior, not real collectives.
 * Processor transformations, serialization formats and scheduler admission belong
 * to their own tests. Real multi-stage communication and overlap belong to smoke tests.
 */

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include <pybind11/embed.h>
#include <torch/extension.h>
#include "torch/all.h"
#include "autil/TimeUtility.h"

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
                         &copy.input_lengths,
                         &copy.sequence_lengths,
                         &copy.prefix_lengths,
                         &copy.lm_output_indexes,
                         &copy.combo_position_ids,
                         &copy.last_hidden_states,
                         &copy.request_id,
                         &copy.request_pd_separation,
                         &copy.cache_keys}) {
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
        inputs.push_back(snapshotInput(input));
        const auto step    = static_cast<int64_t>(inputs.size() - 1);
        const auto rows    = input.combo_tokens.numel();
        const auto options = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
        if (!hidden_buffer.defined() || hidden_buffer.size(0) < rows) {
            hidden_buffer = torch::empty({rows, 4}, options);
        }
        hidden_buffer.narrow(0, 0, rows).copy_(torch::arange(rows * 4, options).reshape({rows, 4}) + step * 100);
        if (input.last_hidden_states.defined()) {
            EXPECT_TRUE(torch::equal(input.last_hidden_states, inputs.back().last_hidden_states));
        }
        GptModelOutputs output;
        output.all_hidden_states = hidden_buffer.narrow(0, 0, rows).narrow(1, 0, 2);
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
        if (role_ == DSparkModelRole::PROPOSE) {
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
    int  waits = 0;
    bool fail  = false;
};

/** Control wait outcomes without sleeping or depending on a distributed backend. */
class RecordingWork: public P2PWork {
public:
    explicit RecordingWork(std::shared_ptr<WorkState> state): state_(std::move(state)) {}

    void wait() override {
        ++state_->waits;
        if (state_->fail) {
            throw std::runtime_error("peer disconnected");
        }
    }

private:
    std::shared_ptr<WorkState> state_;
};

class InMemoryP2PWork final: public P2PWork {
public:
    explicit InMemoryP2PWork(torch::Tensor tensor): tensor_(std::move(tensor)) {}

    /** The test transport copies payloads eagerly; wait ordering is tested with RecordingWork. */
    void wait() override {}

private:
    torch::Tensor tensor_;
};

class InMemoryPPTransport: public PPTransport {
public:
    void enqueueObject(const torch::Tensor& payload) {
        received_tensors.push_back(torch::tensor({payload.numel()}, torch::kInt64));
        received_tensors.push_back(payload);
    }

    std::unique_ptr<PPCommTicket> asyncSend(const torch::Tensor& tensor) override {
        sent_tensors.push_back(tensor.clone());
        return std::make_unique<PPCommTicket>(std::make_unique<InMemoryP2PWork>(tensor));
    }

    std::unique_ptr<PPCommTicket> asyncReceive(torch::Tensor& tensor) override {
        if (on_receive) {
            on_receive();
        }
        tensor.copy_(received_tensors.at(receive_index++));
        return std::make_unique<PPCommTicket>(std::make_unique<InMemoryP2PWork>(tensor));
    }

    std::vector<torch::Tensor> received_tensors;
    std::vector<torch::Tensor> sent_tensors;
    size_t                    receive_index = 0;
    std::function<void()>     on_receive;
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

    /** Individual stage tests initialize the fields normally owned by sampleTokens(). */
    static PPExecutionResult makeInitializedResult(const PPSamplingPlan& plan) {
        PPExecutionResult result;
        const auto        stream_count = plan.request_ids.size(0);
        result.request_ids             = plan.request_ids.to(torch::kCPU).contiguous();
        result.request_errors.assign(stream_count, ErrorInfo::OkStatus());
        result.prompt_logits.resize(stream_count);
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
                                        int32_t                num_return_sequences) {
        auto input                                   = std::make_shared<GenerateInput>();
        input->request_id                            = request_id;
        input->input_ids                             = intTensor(std::move(input_ids));
        input->generate_config                       = std::make_shared<GenerateConfig>();
        input->generate_config->num_return_sequences = num_return_sequences;
        input->generate_config->max_new_tokens       = 20;
        input->generate_config->do_sample            = false;

        RuntimeConfig runtime_config;
        auto          stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        stream->setPipelineParallel(true);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
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
                wire.received_tensors.push_back(tensor);
            }
        }
    }

    static PPExecutionPlan makeShutdownPlan() {
        PPExecutionPlan plan;
        plan.shutdown = true;
        plan.model_input.skip_run = true;
        return plan;
    }

    static PPExecutionResult lastResult(const InMemoryPPTransport& wire) {
        return pp_serialization::deserializeExecutionResult(wire.sent_tensors.back());
    }

    /** Legal stage input for two requests; processor field packing is tested separately. */
    static PPExecutionPlan makePlan(const EngineInitParams& params, bool decode = false) {
        const auto first = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1);
        const auto second = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4, 5}, 1);
        if (decode) {
            first->updateFromPP({intTensor({7}).reshape({1, 1}), 1});
            second->updateFromPP({intTensor({8}).reshape({1, 1}), 1});
        }
        PPBatchStreamProcessor processor(params.model_config_, params.pd_sep_config,
                                        ProfilingDebugLoggingConfig{}, CacheConfig{}, false, params.sp_config.type);
        const StreamGroups groups({first, second});
        PPExecutionPlan plan;
        plan.is_decode = decode;
        plan.sampling_plan = processor.gatherSamplingPlan(groups);
        plan.output_config = processor.gatherOutputConfig(groups);
        auto& input = plan.model_input;
        input.skip_run = false;
        input.request_id = torch::tensor({101, 202}, torch::kInt64);
        input.request_pd_separation =
            torch::full({2}, params.pd_sep_config.role_type == RoleType::PREFILL, torch::kBool);
        input.pd_separation = params.pd_sep_config.role_type == RoleType::PREFILL;
        input.is_target_verify = decode && params.sp_config.type != SP_TYPE_NONE;
        if (input.is_target_verify) {
            const auto width = params.sp_config.gen_num_per_cycle;
            std::vector<int32_t> tokens;
            for (int row = 0; row < 2; ++row) {
                tokens.push_back(7 + row);
                for (int step = 0; step < width; ++step) {
                    tokens.push_back(11 + 10 * row + step);
                }
            }
            input.combo_tokens = intTensor(tokens);
            input.input_lengths = torch::full({2}, width + 1, torch::kInt32);
            input.prefix_lengths = intTensor({2, 3});
            input.sequence_lengths = intTensor({});
            input.lm_output_indexes = torch::arange(2 * (width + 1), torch::kInt32);
            if (params.sp_config.type == SP_TYPE_DSPARK) {
                /** Model a prior C++ proposal round with an exact q for each draft token. */
                const auto draft_tokens = input.combo_tokens.reshape({2, width + 1}).narrow(1, 1, width).to(torch::kLong);
                plan.sampling_plan.draft_all_probs =
                    torch::zeros({2, width, params.model_config_.vocab_size}, torch::kFloat32)
                        .scatter_(-1, draft_tokens.unsqueeze(-1), 1.0f)
                        .contiguous();
            }
        } else if (decode) {
            input.combo_tokens = intTensor({7, 8});
            input.input_lengths = intTensor({2, 3});
            input.prefix_lengths = intTensor({});
            input.sequence_lengths = intTensor({2, 3});
            input.lm_output_indexes = intTensor({0, 1});
        } else {
            input.combo_tokens = intTensor({1, 2, 3, 4, 5});
            input.input_lengths = intTensor({2, 3});
            input.prefix_lengths = intTensor({0, 0});
            input.sequence_lengths = intTensor({});
            input.lm_output_indexes = intTensor({1, 4});
        }
        return plan;
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
        resources.cache_manager = std::make_shared<KVCacheManager>(
            test::makeSimpleMhaCacheConfig(3, 16, 4, DataType::TYPE_FP16));
        EXPECT_TRUE(resources.cache_manager->init());
        auto stream = makeFakeStream(params, resources, decode);
        PPBatchStreamProcessor processor(params.model_config_, params.pd_sep_config, ProfilingDebugLoggingConfig{},
                                        resources.cache_manager->cacheConfig(), false, params.sp_config.type);
        const StreamGroups groups({stream});
        TensorHolder holder;
        auto input = decode && params.sp_config.type != SP_TYPE_NONE ?
                         processor.gatherTargetVerifyModelInput(groups, params.sp_config.gen_num_per_cycle, holder) :
                         processor.gatherModelInput(groups, holder);
        EXPECT_TRUE(input.ok()) << input.status().ToString();
        PPExecutionPlan plan;
        plan.model_input = std::move(input.value());
        plan.model_input.skip_run = false;
        plan.is_decode = decode;
        plan.sampling_plan = processor.gatherSamplingPlan(groups);
        plan.output_config = processor.gatherOutputConfig(groups);
        plan.draft_next_position_ids = processor.gatherDraftNextPositionIds(groups, plan.model_input);
        return plan;
    }

    static std::shared_ptr<RecordingLogitsProcessor> recordState(PPExecutor& executor, int64_t id, int sequences = 1) {
        auto processor = std::make_shared<RecordingLogitsProcessor>();
        auto& state = executor.sampling_states_[id];
        state.logits_processors = {processor};
        state.cum_log_probs = torch::zeros({sequences}, torch::kFloat32);
        return processor;
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

TEST_F(PPExecutorTest, CommonActiveCpRequiresPrefillButImportedCpAllowsDecode) {
    for (const auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        for (const auto role : {RoleType::PREFILL, RoleType::DECODE, RoleType::PDFUSION}) {
            SCOPED_TRACE(::testing::Message() << "type=" << type << ", role=" << role);
            auto params = makeMtpParams(3);
            params.sp_config.type = type;
            params.sp_config.sp_dspark_mask_token_id = 63;
            params.pd_sep_config.role_type = role;
            params.parallelism_config.tp_size = 2;
            params.parallelism_config.world_size = 4;
            params.parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
            if (role == RoleType::PREFILL) {
                EXPECT_NO_THROW({ PPExecutor executor(params, nullptr, true); });
            } else {
                try {
                    PPExecutor executor(params, nullptr, true);
                    FAIL() << "active CP accepted a role that runs verify/decode";
                } catch (const std::runtime_error& error) {
                    EXPECT_NE(std::string(error.what()).find("PP context parallel execution requires the PREFILL role"),
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
        auto model = std::make_unique<RecordingStageModel>(params.parallelism_config);
        auto* recorded = model.get();
        executor.setModel(std::move(model));
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
        auto& slot = executor.slots_[executor.current_slot_];
        auto add_work = [](PPTickets& tickets, const std::shared_ptr<WorkState>& state) {
            tickets.push_back(std::make_unique<PPCommTicket>(std::make_unique<RecordingWork>(state)));
        };
        if (stage == 2) {
            add_work(slot.execution_result_sends, previous_payload);
        } else {
            add_work(slot.plan_sends, previous_plan);
            add_work(slot.activation_sends, previous_payload);
        }
        auto model = std::make_unique<RecordingStageModel>(params.parallelism_config);
        model->on_forward = [&] {
            EXPECT_EQ(previous_plan->waits, stage == 2 ? 0 : 1);
            EXPECT_EQ(previous_payload->waits, 1);
        };
        executor.setModel(std::move(model));
        ScheduleOutput schedule;
        if (stage == 0) {
            schedule.streams = {makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1)};
        } else {
            enqueuePlan(*wire, makePlan(params));
        }
        ASSERT_TRUE(executor.process(schedule).ok());
        EXPECT_EQ(previous_payload->waits, 1);
        /** New sends belong to the current execution, not to the completed work. */
        executor.waitAll(slot.plan_sends);
        executor.waitAll(slot.activation_sends);
        executor.waitAll(slot.execution_result_sends);
        EXPECT_EQ(previous_payload->waits, 1);
    }
}

TEST_F(PPExecutorTest, CommonCommunicationWaitCompletesOnceAndPropagatesFailures) {
    auto running = std::make_shared<WorkState>();
    PPCommTicket running_ticket(std::make_unique<RecordingWork>(running));
    running_ticket.wait();
    running_ticket.wait();
    EXPECT_EQ(running->waits, 1);

    auto failed = std::make_shared<WorkState>();
    failed->fail = true;
    PPCommTicket failed_ticket(std::make_unique<RecordingWork>(failed));
    EXPECT_THROW(failed_ticket.wait(), std::runtime_error);
}

TEST_F(PPExecutorTest, FinishWaitsForEarlierSendsAndPropagatesFailures) {
    for (int stage : {1, 2}) {
        for (bool fail : {false, true}) {
            SCOPED_TRACE(testing::Message() << "stage=" << stage << ", fail=" << fail);
            auto params = makeStageParams(stage);
            PPExecutor executor(params, nullptr, false);
            auto* wire = attachTransport(executor);
            auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
            auto* recorded = target.get();
            executor.setModel(std::move(target));
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
                for (const auto& batch : executor.slots_) {
                    EXPECT_TRUE(batch.plan_sends.empty());
                    EXPECT_TRUE(batch.activation_sends.empty());
                    EXPECT_TRUE(batch.execution_result_sends.empty());
                }
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
        auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
        auto* recorded = target.get();
        executor.setModel(std::move(target));
        auto stream = makeStream(resources, params.model_config_, 101, {1, 2}, 1);
        stream->fakeInitKVBlock(4);
        if (decode) {
            stream->updateFromPP({intTensor({7}).reshape({1, 1}), 1});
        }
        stream->setPPInflight();
        ASSERT_TRUE(executor.process(ScheduleOutput{{stream}}).ok());
        ASSERT_EQ(recorded->inputs.size(), 1u);
        EXPECT_EQ(recorded->has_input, std::vector<bool>{false});
        EXPECT_EQ(recorded->has_output, std::vector<bool>{true});
        EXPECT_EQ(recorded->inputs[0].sequence_lengths.numel(), decode ? 1 : 0);
        EXPECT_FALSE(recorded->inputs[0].is_target_verify);
        ASSERT_EQ(wire->sent_tensors.size(), 5u);
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
            auto* wire = attachTransport(executor);
            auto target = std::make_unique<RecordingDraftModel>();
            auto* recorded = target.get();
            executor.setModel(std::move(target));
            std::vector<GenerateStreamPtr> streams;
            for (int index = 0; index < batches; ++index) {
                auto stream = makeStream(ResourceContext{}, params.model_config_, 101 + index, {1, 2}, 1);
                streams.push_back(stream);
                PPExecutionResult result;
                result.request_ids = torch::tensor({101 + index}, torch::kInt64);
                result.new_token_ids = intTensor({20 + index}).reshape({1, 1});
                result.new_token_lengths = intTensor({1});
                result.request_errors.resize(1);
                result.prompt_logits.resize(1);
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
                    EXPECT_FALSE(empty_plan.shutdown);
                    EXPECT_TRUE(empty_plan.finished_request_ids.empty());
                    EXPECT_FALSE(admitted.result_pending);
                }
                EXPECT_EQ(recorded->inputs.size(), std::min(round + 1, batches));
                const int returned = std::clamp(round - pp_size + 1, 0, batches);
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
        PPExecutionResult result;
        result.request_ids = torch::tensor({101}, torch::kInt64);
        result.new_token_ids = intTensor({20}).reshape({1, 1});
        result.new_token_lengths = intTensor({1});
        result.request_errors.resize(1);
        result.prompt_logits.resize(1);
        wire->enqueueObject(pp_serialization::serializeExecutionResult(result));
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
                auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
                auto* recorded = target.get();
                executor.setModel(std::move(target));
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
                            result.request_ids = torch::tensor({101 + round}, torch::kInt64);
                            result.new_token_ids = intTensor({20 + round}).reshape({1, 1});
                            result.new_token_lengths = intTensor({1});
                            result.request_errors.resize(1);
                            result.prompt_logits.resize(1);
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
                wire->on_receive = [&] {
                    /** Downstream ranks must receive the terminal plan before the root drains their results. */
                    ASSERT_EQ(wire->sent_tensors.size(), sends + 2);
                    const auto terminal = pp_serialization::deserializePlan(wire->sent_tensors[sends + 1]);
                    EXPECT_TRUE(terminal.shutdown);
                    EXPECT_TRUE(terminal.model_input.skip_run);
                };
                ASSERT_TRUE(executor.finish().ok());
                EXPECT_TRUE(executor.stopped_);
                /** No terminal result is queued: extra or duplicate receives fail the transport. */
                EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
                ASSERT_EQ(wire->sent_tensors.size(), sends + 2);
                const auto terminal = pp_serialization::deserializePlan(wire->sent_tensors[sends + 1]);
                EXPECT_TRUE(terminal.shutdown);
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
                for (const auto& batch : executor.slots_) {
                    EXPECT_TRUE(batch.plan_sends.empty());
                    EXPECT_TRUE(batch.activation_sends.empty());
                    EXPECT_TRUE(batch.execution_result_sends.empty());
                }
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
                    result.request_ids = torch::tensor({101 + round}, torch::kInt64);
                    result.new_token_ids = intTensor({20 + round}).reshape({1, 1});
                    result.new_token_lengths = intTensor({1});
                    result.request_errors.resize(1);
                    result.prompt_logits.resize(1);
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
    auto* wire = attachTransport(executor);
    auto stream = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 1);
    auto& batch = executor.slots_[0];
    batch.result_pending = true;
    batch.stream_groups = StreamGroups({stream});
    stream.reset();
    /** A deserialized but invalid result has already consumed its wire message. */
    wire->enqueueObject(pp_serialization::serializeExecutionResult(PPExecutionResult{}));
    EXPECT_THROW((void)executor.drainPendingResults(), std::exception);
    EXPECT_FALSE(batch.result_pending);
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
        auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
        auto* recorded = target.get();
        executor.setModel(std::move(target));
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
            EXPECT_FALSE(pp_serialization::deserializePlan(wire->sent_tensors[1]).shutdown);
            const auto forwarded_cleanup = pp_serialization::deserializePlan(wire->sent_tensors[6]);
            EXPECT_EQ(forwarded_cleanup.finished_request_ids, cleanup.finished_request_ids);
            EXPECT_FALSE(forwarded_cleanup.shutdown);
            EXPECT_TRUE(pp_serialization::deserializePlan(wire->sent_tensors[8]).shutdown);
        } else {
            ASSERT_EQ(wire->sent_tensors.size(), 2u);
            EXPECT_EQ(tensorToVector<int64_t>(lastResult(*wire).request_ids), (std::vector<int64_t>{101, 202}));
        }
        for (const auto& batch : executor.slots_) {
            EXPECT_TRUE(batch.plan_sends.empty());
            EXPECT_TRUE(batch.activation_sends.empty());
            EXPECT_TRUE(batch.execution_result_sends.empty());
        }
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

        PPExecutionResult result;
        result.request_ids   = torch::tensor({202}, torch::kInt64);
        result.new_token_ids = intTensor({17}).reshape({1, 1});
        result.new_token_lengths = intTensor({1});
        result.request_errors.resize(1);
        result.prompt_logits.resize(1);
        const auto payload = pp_serialization::serializeExecutionResult(result);
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

/** First-stage additions: PD handoff, speculative admission and TP ownership. */
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
        auto target = std::make_unique<RecordingDraftModel>();
        auto* recorded = target.get();
        executor.setModel(std::move(target));
        auto stream = makeStream(resources, params.model_config_, 101, {1, 2}, 1);
        stream->fakeInitKVBlock(4);
        stream->updateFromPP({intTensor({7}).reshape({1, 1}), 1});
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

TEST_F(PPExecutorTest, FirstSpAdmissionFiltersUnsupportedRequestsBeforeForward) {
    for (bool all_failed : {false, true}) {
        SCOPED_TRACE(all_failed);
        auto params = makeStageParams(0, SP_TYPE_MTP);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto target = std::make_unique<RecordingDraftModel>();
        auto* recorded = target.get();
        executor.setModel(std::move(target));
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
            auto plan_status = executor.buildPlan(StreamGroups(streams), {});
            ASSERT_TRUE(plan_status.ok()) << plan_status.status().ToString();
            const auto& plan = plan_status.value();
            EXPECT_TRUE(plan.model_input.is_fake_stream);
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
                    auto params = makeStageParams(stage, type);
                    params.parallelism_config.tp_size = 2;
                    params.parallelism_config.world_size = 6;
                    params.parallelism_config.tp_rank = rank;
                    params.parallelism_config.world_rank = stage * 2 + rank;
                    auto propose = makeMtpProposeParams(params);
                    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr,
                                        stage == 2 && type == SP_TYPE_MTP ? propose.get() : nullptr);
                    auto* wire = attachTransport(executor);
                    auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
                    auto* recorded = target.get();
                    executor.setModel(std::move(target));
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
                            enqueuePlan(*wire, makeShutdownPlan(), rank != 0);
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
                    /** Both full and metadata-only plans must distinguish ordinary emptiness from shutdown. */
                    if (stage != 2) {
                        EXPECT_FALSE(pp_serialization::deserializePlan(wire->sent_tensors[1]).shutdown);
                        const auto terminal = pp_serialization::deserializePlan(wire->sent_tensors[3]);
                        EXPECT_TRUE(terminal.shutdown);
                        if (rank == 0) {
                            EXPECT_TRUE(terminal.model_input.skip_run);
                            EXPECT_FALSE(terminal.model_input.is_fake_stream);
                        }
                    }
                    if (recorded_draft) {
                        EXPECT_TRUE(recorded_draft->inputs.empty());
                    }
                    EXPECT_TRUE(executor.sampling_states_.empty());
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
        auto params = makeStageParams(0);
        params.parallelism_config.tp_size = 2;
        params.parallelism_config.world_size = 6;
        params.parallelism_config.tp_rank = rank;
        params.parallelism_config.world_rank = rank;
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
        auto* recorded = target.get();
        executor.setModel(std::move(target));
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
        auto params = makeStageParams(1);
        params.parallelism_config.tp_size = 2;
        params.parallelism_config.tp_rank = rank;
        params.parallelism_config.world_size = 6;
        params.parallelism_config.world_rank = 2 + rank;
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto plan = makePlan(params);
        plan.model_input.kv_cache_group_tags = {"swa", "full"};
        plan.model_input.kv_cache_block_id = intTensor({11, 12, 21, 22}).reshape({2, 2, 1});
        plan.model_input.kv_cache_kernel_block_id = plan.model_input.kv_cache_block_id.clone();
        enqueuePlan(*wire, plan, rank != 0);
        auto model = std::make_unique<RecordingStageModel>(params.parallelism_config);
        auto* recorded = model.get();
        executor.setModel(std::move(model));

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
    auto params = makeStageParams(/*pp_rank=*/1);
    ResourceContext resources;
    resources.cache_manager = std::make_shared<KVCacheManager>(
        test::makeSimpleMhaCacheConfig(/*layer_num=*/3, /*block_num=*/16, /*tokens_per_block=*/4, DataType::TYPE_FP16));
    ASSERT_TRUE(resources.cache_manager->init());
    PPExecutor executor(params, resources.cache_manager, false);
    auto* wire = attachTransport(executor);

    auto plan = makePlan(params);
    // Payload carries "default" (owned by this stage) and "foreign" (owned by another stage).
    plan.model_input.kv_cache_group_tags      = {"default", "foreign"};
    plan.model_input.kv_cache_block_id        = intTensor({1, 2, 3, 4}).reshape({2, 2, 1});
    plan.model_input.kv_cache_kernel_block_id = plan.model_input.kv_cache_block_id.clone();
    // Row 0 -> tag row 0 ("default"): copy block 3 to 4. Row 1 -> tag row 1 ("foreign"): skipped.
    plan.model_input.kv_cache_update_mapping  = intTensor({0, 3, 4, 1, 5, 6}).reshape({2, 3});

    enqueuePlan(*wire, plan);
    executor.setModel(std::make_unique<RecordingStageModel>(params.parallelism_config));

    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
}

/** First-stage result metrics: preserve batch ownership and the existing counting semantics. */
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
    result.prompt_logits.resize(4);
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
    enableMetrics(executor);
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
    decode_result.prompt_logits.resize(1);
    auto prefill_result              = decode_result;
    prefill_result.request_ids       = torch::tensor({202}, torch::kInt64);
    prefill_result.new_token_ids     = intTensor({14}).reshape({1, 1});
    prefill_result.new_token_lengths = intTensor({1});
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
    const auto& admitted = executor.slots_[0];
    EXPECT_EQ(admitted.schedule_time_us, schedule_time);
    EXPECT_EQ(admitted.executor_collector.context_batch_size, 1);
    EXPECT_EQ(admitted.executor_collector.generate_batch_size, 0);
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
    EXPECT_EQ(executor.tps_reporter_.collector_.generateTPS(), 4);
    EXPECT_EQ(executor.tps_reporter_.collector_.totalTPS(), 4);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    EXPECT_EQ(prefill->seqLength(), 4);
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
TEST_F(PPExecutorTest, MiddlePassesActivationsThroughTargetForEachExecutionPhase) {
    for (int phase : {0, 1, 2}) {
        SCOPED_TRACE(phase);
        auto params = makeStageParams(1, phase == 2 ? SP_TYPE_MTP : SP_TYPE_NONE);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        const auto plan = makePlan(params, phase != 0);
        enqueuePlan(*wire, plan);
        auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
        auto* recorded = target.get();
        executor.setModel(std::move(target));
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
    auto params = makeStageParams(1);
    PPExecutor executor(params, nullptr, false);
    auto* wire = attachTransport(executor);
    auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
    auto* recorded = target.get();
    executor.setModel(std::move(target));
    int profile_calls = 0;
    executor.profile_step_start_ = [&] { ++profile_calls; };
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

/** Last: sampling and request state first, then PD/speculative/TP/CP additions. */
TEST_F(PPExecutorTest, LastSamplesAcrossRoundsWithoutReinitializingRequestState) {
    auto params = makeStageParams(2);
    PPExecutor executor(params, nullptr, false);
    auto* wire = attachTransport(executor);
    auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
    auto* recorded = target.get();
    recorded->next_tokens = {7, 8};
    executor.setModel(std::move(target));
    auto prefill = makePlan(params);
    prefill.sampling_plan.random_seeds = {123, 456};
    prefill.sampling_plan.do_sample.fill_(true);
    prefill.sampling_plan.top_k.fill_(2);
    prefill.output_config.return_all_hidden_states = true;
    prefill.model_input.need_all_hidden_states = true;
    enqueuePlan(*wire, prefill);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    ASSERT_EQ(wire->sent_tensors.size(), 2u);
    auto first = lastResult(*wire);
    EXPECT_EQ(tensorToVector<int32_t>(first.new_token_ids), (std::vector<int32_t>{7, 8}));
    EXPECT_TRUE(torch::equal(first.all_hidden_states, recorded->hidden_buffer.narrow(1, 0, 2).cpu()));
    EXPECT_TRUE(recorded->hidden_requests.empty());
    EXPECT_EQ(tensorToVector<int32_t>(first.new_token_lengths), (std::vector<int32_t>{1, 1}));
    EXPECT_FALSE(first.propose_token_ids.defined());
    const auto generator = executor.sampling_states_.at(101).generator;
    const auto rng_before = generator.get_state();
    auto processor = std::make_shared<RecordingLogitsProcessor>();
    executor.sampling_states_.at(101).logits_processors = {processor};

    auto decode = makePlan(params, true);
    decode.sampling_plan.random_seeds = {999, 999};
    decode.sampling_plan.do_sample.fill_(true);
    decode.sampling_plan.top_k.fill_(2);
    recorded->next_tokens = {11, 21};
    enqueuePlan(*wire, decode);
    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    ASSERT_EQ(wire->sent_tensors.size(), 4u);
    const auto result = lastResult(*wire);
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids), (std::vector<int32_t>{11, 21}));
    EXPECT_EQ(executor.sampling_states_.at(101).generator, generator);
    EXPECT_EQ(generator.current_seed(), 123);
    EXPECT_FALSE(torch::equal(generator.get_state(), rng_before));
    ASSERT_EQ(processor->committed_tokens.size(), 1u);
    EXPECT_EQ(tensorToVector<int32_t>(processor->committed_tokens[0]), std::vector<int32_t>{11});
    EXPECT_EQ(recorded->has_input, (std::vector<bool>{true, true}));
    EXPECT_EQ(recorded->has_output, (std::vector<bool>{false, false}));
}

TEST_F(PPExecutorTest, LastRetirementAllowsRequestIdReuseWithFreshSamplingState) {
    for (bool fake_cleanup : {false, true}) {
        SCOPED_TRACE(fake_cleanup);
        auto params = makeStageParams(2);
        PPExecutor executor(params, nullptr, false);
        auto* wire = attachTransport(executor);
        auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
        auto* recorded = target.get();
        executor.setModel(std::move(target));
        auto first = makePlan(params);
        first.sampling_plan.random_seeds = {123, 456};
        enqueuePlan(*wire, first);
        ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
        const auto old_generator = executor.sampling_states_.at(101).generator;
        auto old_processor = std::make_shared<RecordingLogitsProcessor>();
        executor.sampling_states_.at(101).logits_processors = {old_processor};

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
        EXPECT_EQ(executor.sampling_states_.at(101).generator.current_seed(), 789u);
        EXPECT_EQ(executor.sampling_states_.at(202).generator.current_seed(), 999u);
        EXPECT_NE(executor.sampling_states_.at(101).generator, old_generator);
        EXPECT_TRUE(old_processor->committed_tokens.empty());
        EXPECT_EQ(recorded->inputs.size(), fake_cleanup ? 3u : 2u);
        EXPECT_EQ(wire->sent_tensors.size(), fake_cleanup ? 6u : 4u);
        EXPECT_EQ(tensorToVector<int64_t>(lastResult(*wire).request_ids), (std::vector<int64_t>{101, 202}));
        EXPECT_EQ(wire->receive_index, wire->received_tensors.size());
    }
}

TEST_F(PPExecutorTest, LastUpdatesStatesAtRequestSequenceOffsetsAndPreservesEarlierErrors) {
    auto params = makeStageParams(2);
    PPExecutor executor(params, nullptr, false);
    auto failed = makeStream(ResourceContext{}, params.model_config_, 101, {1, 2}, 2);
    auto healthy = makeStream(ResourceContext{}, params.model_config_, 202, {3, 4}, 2);
    auto update_failed = makeStream(ResourceContext{}, params.model_config_, 303, {5, 6}, 1);
    const auto plan =
        executor.batch_stream_processor_->gatherSamplingPlan(StreamGroups({failed, healthy, update_failed}));
    auto first = recordState(executor, 101, 2);
    auto second = recordState(executor, 202, 2);
    auto third = recordState(executor, 303);
    third->update_error = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "state update failed");
    auto result = makeInitializedResult(plan);
    result.new_token_ids     = intTensor({0, 0, 20, 21, 30}).reshape({5, 1});
    result.new_token_lengths = torch::ones({5}, torch::kInt32);
    result.cum_log_probs     = torch::full({5}, -3.0, torch::kFloat32);
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
    EXPECT_EQ(tensorToVector<float>(executor.sampling_states_.at(101).cum_log_probs), (std::vector<float>{0, 0}));
    EXPECT_EQ(tensorToVector<float>(executor.sampling_states_.at(202).cum_log_probs), (std::vector<float>{-3, -3}));
    EXPECT_EQ(executor.sampling_states_.at(303).cum_log_probs.item<float>(), 0);
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
    auto target = std::make_unique<RecordingStageModel>(params.parallelism_config);
    auto* recorded = target.get();
    executor.setModel(std::move(target));
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

/** DSpARK proposal geometry preserves both checkpoint-defined anchor modes. */
TEST_F(PPExecutorTest, DSparkProposalInputSelectsExactlyGammaLogitRows) {
    constexpr int32_t width = 3;
    for (bool sample_from_anchor : {true, false}) {
        SCOPED_TRACE(sample_from_anchor);
        auto params                                      = makeStageParams(2, SP_TYPE_DSPARK, width);
        params.pd_sep_config.role_type                   = RoleType::PDFUSION;
        params.sp_config.sp_dspark_sample_from_anchor    = sample_from_anchor;
        auto propose                                     = makeMtpProposeParams(params);
        PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());

        GptModelInputs target_input;
        target_input.combo_tokens      = intTensor({1, 2, 3, 4, 5});
        target_input.input_lengths     = intTensor({2, 3});
        target_input.prefix_lengths    = intTensor({0, 0});
        target_input.sequence_lengths  = intTensor({});
        target_input.lm_output_indexes = intTensor({1, 4});
        target_input.combo_position_ids = intTensor({10, 11, 20, 21, 22});

        auto draft_input = executor.prepareDSparkProposeInput(
            target_input, false, intTensor({7, 8}).reshape({2, 1}), intTensor({1, 1}));
        const int32_t query_width = width + static_cast<int32_t>(!sample_from_anchor);
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.input_lengths),
                  (std::vector<int32_t>{query_width, query_width}));
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.combo_tokens),
                  sample_from_anchor ? (std::vector<int32_t>{7, 63, 63, 8, 63, 63}) :
                                       (std::vector<int32_t>{7, 63, 63, 63, 8, 63, 63, 63}));
        EXPECT_EQ(tensorToVector<int32_t>(draft_input.lm_output_indexes),
                  sample_from_anchor ? (std::vector<int32_t>{0, 1, 2, 3, 4, 5}) :
                                       (std::vector<int32_t>{1, 2, 3, 5, 6, 7}));
        EXPECT_FALSE(draft_input.combo_position_ids.defined());
    }
}

/** Last-stage draft execution: PD prefill, verification, failed rows and hidden-state handoff. */
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
                std::vector<GptModelInputs>* draft_inputs;
                if (type == SP_TYPE_DSPARK) {
                    auto draft = installDSparkModels(executor, width);
                    draft->on_forward = [&] { EXPECT_TRUE(profiling); };
                    draft->wait_publication = [&] { ++draft_waits; return std::string(); };
                    draft_inputs = &draft->inputs;
                } else {
                    auto draft = std::make_unique<RecordingDraftModel>();
                    draft->on_forward = [&] { EXPECT_TRUE(profiling); };
                    draft->wait_publication = [&] { ++draft_waits; return std::string(); };
                    draft_inputs = &draft->inputs;
                    executor.draft_model_ = std::move(draft);
                }
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
                        ASSERT_EQ(result.propose_all_probs.sizes().vec(),
                                  (std::vector<int64_t>{2, count, params.model_config_.vocab_size}));
                        EXPECT_TRUE(result.propose_all_probs.device().is_cpu());
                        EXPECT_EQ(result.propose_all_probs.scalar_type(), torch::kFloat32);
                        EXPECT_TRUE(result.propose_all_probs.is_contiguous());
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
        auto params = makeStageParams(2, SP_TYPE_DSPARK);
        params.pd_sep_config.role_type = RoleType::PREFILL;
        params.parallelism_config.tp_size = 2;
        params.parallelism_config.world_size = 6;
        params.parallelism_config.tp_rank = rank;
        params.parallelism_config.world_rank = 4 + rank;
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
    auto params = makeStageParams(2, SP_TYPE_DSPARK);
    params.pd_sep_config.role_type = RoleType::PREFILL;
    params.parallelism_config.tp_size = 2;
    params.parallelism_config.world_size = 6;
    params.parallelism_config.world_rank = 4;
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
    const auto plan = makePlan(params);
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

TEST_F(PPExecutorTest, LastDSparkHandoffVerifiesPointMassThenProposesWithDenseProbabilities) {
    constexpr int width = 3;
    auto params = makeStageParams(2, SP_TYPE_DSPARK, width);
    params.pd_sep_config.role_type = RoleType::DECODE;
    auto propose = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto* wire = attachTransport(executor);

    auto plan = makePlan(params, true);
    plan.model_input.combo_tokens      = intTensor({7, 0, 0, 0, 8, 0, 0, 0});
    plan.model_input.input_lengths     = intTensor({4, 4});
    plan.model_input.prefix_lengths    = intTensor({2, 3});
    plan.model_input.sequence_lengths  = intTensor({});
    plan.model_input.lm_output_indexes = torch::arange(8, torch::kInt32);
    plan.sampling_plan.draft_all_probs = torch::Tensor();
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
    ASSERT_EQ(result.propose_all_probs.sizes().vec(),
              (std::vector<int64_t>{2, width, params.model_config_.vocab_size}));
    ASSERT_EQ(recorded_draft->inputs.size(), 2u);
    EXPECT_TRUE(recorded_draft->inputs.front().is_target_verify);
    EXPECT_TRUE(recorded_draft->inputs.front().last_hidden_states.defined());
    EXPECT_TRUE(torch::equal(recorded_draft->inputs.front().combo_tokens, plan.model_input.combo_tokens));
    EXPECT_FALSE(recorded_draft->inputs.back().last_hidden_states.defined());
    EXPECT_EQ(tensorToVector<int32_t>(recorded_draft->inputs.back().prefix_lengths),
              (std::vector<int32_t>{3, 4}));
}

TEST_F(PPExecutorTest, LastDSparkMixedQVerifiesBeforeCommitAndFreshProposal) {
    constexpr int width = 3;
    auto params = makeStageParams(2, SP_TYPE_DSPARK, width);
    params.pd_sep_config.role_type = RoleType::DECODE;
    auto propose = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto* wire = attachTransport(executor);
    auto plan = makePlan(params, true);
    plan.model_input.combo_tokens = intTensor({7, 11, 12, 13, 8, 0, 0, 0});
    plan.sampling_plan.do_sample.fill_(true);
    plan.sampling_plan.spec_do_sample.fill_(true);
    plan.sampling_plan.top_k.zero_();
    plan.sampling_plan.top_p.fill_(1);
    plan.sampling_plan.temperature.fill_(1);
    plan.sampling_plan.random_seeds = {123, 456};
    plan.sampling_plan.draft_all_probs.zero_();
    plan.sampling_plan.draft_point_mass_rows = torch::tensor({1}, torch::kInt64);
    auto target = std::make_unique<RecordingDraftModel>();
    auto* recorded_target = target.get();
    auto target_logits = torch::full({2 * (width + 1), params.model_config_.vocab_size + 8},
                                     -std::numeric_limits<float>::infinity(),
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    /** Padding would dominate every real token if verify used the physical lm-head width. */
    target_logits.narrow(1, params.model_config_.vocab_size, 8).fill_(100);
    /** A's true q equals p, so all its candidates are accepted; B's fixed 0 is rejected. */
    for (int step = 0; step < width; ++step) {
        target_logits[step][11 + step] = 0;
        target_logits[step][40] = 0;
        plan.sampling_plan.draft_all_probs[0][step][11 + step] = .5;
        plan.sampling_plan.draft_all_probs[0][step][40] = .5;
    }
    target_logits[width][43] = 0;
    for (int step = 0; step <= width; ++step) {
        target_logits[width + 1 + step][50 + step] = 0;
    }
    target->scripted_logits = target_logits;
    executor.setModel(std::move(target));
    auto draft = installDSparkModels(executor, width);
    auto* recorded_draft = draft.get();
    const auto original_q = plan.sampling_plan.draft_all_probs.clone();
    enqueuePlan(*wire, plan);

    ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
    const auto result = lastResult(*wire);
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_lengths), (std::vector<int32_t>{4, 1}));
    EXPECT_EQ(tensorToVector<int32_t>(result.new_token_ids[0]), (std::vector<int32_t>{11, 12, 13, 43}));
    EXPECT_EQ(result.new_token_ids[1][0].item<int32_t>(), 50);
    EXPECT_TRUE(torch::equal(plan.sampling_plan.draft_all_probs, original_q));
    ASSERT_EQ(recorded_target->inputs.size(), 1u);
    EXPECT_TRUE(recorded_target->inputs[0].is_target_verify);
    ASSERT_EQ(recorded_draft->inputs.size(), 2u);
    EXPECT_TRUE(recorded_draft->inputs[0].last_hidden_states.defined());
    EXPECT_TRUE(torch::equal(recorded_draft->inputs[0].combo_tokens, plan.model_input.combo_tokens));
    EXPECT_FALSE(recorded_draft->inputs[1].last_hidden_states.defined());
    EXPECT_EQ(tensorToVector<int32_t>(recorded_draft->inputs[1].prefix_lengths), (std::vector<int32_t>{6, 4}));
    ASSERT_EQ(result.propose_all_probs.sizes().vec(), (std::vector<int64_t>{2, width, 64}));
    EXPECT_TRUE(torch::equal(result.propose_all_probs.sum(-1), torch::ones({2, width})));
    EXPECT_TRUE(torch::equal(result.propose_all_probs.argmax(-1).to(torch::kInt32), result.propose_token_ids));
    EXPECT_EQ(result.propose_all_probs[1].select(1, 0).sum().item<float>(), 0);
}

TEST_F(PPExecutorTest, DSparkDraftTemperatureMatchesMainStochasticCondition) {
    constexpr int width = 2;
    auto params = makeStageParams(2, SP_TYPE_DSPARK, width);
    params.pd_sep_config.role_type = RoleType::DECODE;
    auto propose = makeMtpProposeParams(params);
    PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
    auto draft = installDSparkModels(executor, width);
    auto logits = torch::full({2 * width, 64}, -std::numeric_limits<float>::infinity(),
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    logits.select(1, 10).fill_(1);
    logits.select(1, 11).fill_(0);
    draft->scripted_logits = logits;
    const auto hidden = torch::zeros({2 * (width + 1), 4}, logits.options());
    for (const auto& config : std::vector<std::pair<bool, bool>>{{false, false}, {false, true}, {true, false}, {true, true}}) {
        const bool do_sample = config.first;
        const bool non_top1 = config.second;
        SCOPED_TRACE(::testing::Message() << "do_sample=" << do_sample << ", non_top1=" << non_top1);
        auto plan = makePlan(params, true);
        plan.sampling_plan.do_sample.fill_(do_sample);
        plan.sampling_plan.spec_do_sample.fill_(do_sample && non_top1);
        plan.sampling_plan.top_k.fill_(do_sample && non_top1 ? 0 : 1);
        plan.sampling_plan.temperature.fill_(do_sample ? .5f : 1.f);
        auto result = makeInitializedResult(plan.sampling_plan);
        result.new_token_ids = intTensor({30, 31}).reshape({2, 1});
        result.new_token_lengths = intTensor({1, 1});
        executor.runDraftStep(plan.model_input, {}, hidden, true, plan.sampling_plan, result);
        const float expected_temperature = do_sample && non_top1 ? .5f : 1.e-6f;
        const auto expected_q = torch::softmax(logits / expected_temperature, -1).reshape({2, width, 64}).cpu();
        EXPECT_TRUE(torch::allclose(result.propose_all_probs, expected_q));
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
            /** Verify the full proposal even when the remaining output budget permits fewer tokens. */
            const int remaining = width == 1 ? 1 : 2;
            plan.sampling_plan.max_tokens[1] = plan.sampling_plan.sequence_lengths[1].item<int32_t>() + remaining;
            enqueuePlan(*wire, plan);
            auto target = std::make_unique<RecordingDraftModel>();
            auto* recorded_target = target.get();
            target->next_tokens = width == 1 ? std::vector<int32_t>{40, 12, 21, 22} :
                                              std::vector<int32_t>{40, 12, 13, 14, 21, 22, 23, 24};
            executor.setModel(std::move(target));
            std::vector<GptModelInputs>* draft_inputs;
            if (type == SP_TYPE_DSPARK) {
                auto draft = installDSparkModels(executor, width);
                draft_inputs = &draft->inputs;
            } else {
                auto draft = std::make_unique<RecordingDraftModel>();
                draft_inputs = &draft->inputs;
                executor.draft_model_ = std::move(draft);
            }
            auto first = recordState(executor, 101);
            auto second = recordState(executor, 202);
            const size_t forwards = type == SP_TYPE_DSPARK ? 2 : width;
            first->on_update = [&] { EXPECT_EQ(draft_inputs->size(), forwards); };
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
            const auto result = lastResult(*wire);
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
                ASSERT_EQ(result.propose_all_probs.sizes().vec(),
                          (std::vector<int64_t>{2, width, params.model_config_.vocab_size}));
                EXPECT_TRUE(result.propose_all_probs.device().is_cpu());
                EXPECT_EQ(result.propose_all_probs.scalar_type(), torch::kFloat32);
                EXPECT_TRUE(result.propose_all_probs.is_contiguous());
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
                auto target = std::make_unique<RecordingDraftModel>();
                target->next_tokens = {11, 40, 13, 14, 21, 22, 23, 24};
                executor.setModel(std::move(target));
                std::vector<GptModelInputs>* inputs;
                if (type == SP_TYPE_DSPARK) {
                    auto draft = installDSparkModels(executor, 3);
                    inputs = &draft->inputs;
                } else {
                    auto draft = std::make_unique<RecordingDraftModel>();
                    inputs = &draft->inputs;
                    executor.draft_model_ = std::move(draft);
                }
                const auto code =
                    initialization_failed ? ErrorCode::INVALID_PARAMS : ErrorCode::GRAMMAR_VERIFY_EXCEPTION;
                const ErrorInfo error(code, "verify failed");
                std::shared_ptr<RecordingLogitsProcessor> healthy;
                std::shared_ptr<RecordingLogitsProcessor> failed;
                if (initialization_failed) {
                    /** PD decode may reach this tail before it has created the request's sampling state. */
                    for (int row = all_failed ? 0 : 1; row < 2; ++row) {
                        plan.sampling_plan.logits_processor_configs[row].grammar_type = "regex";
                        plan.sampling_plan.logits_processor_configs[row].grammar_value = "[ab]+";
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
            auto target = std::make_unique<RecordingDraftModel>();
            auto* recorded = target.get();
            executor.setModel(std::move(target));
            std::vector<GptModelInputs>* draft_inputs = nullptr;
            if (type == SP_TYPE_DSPARK) {
                auto draft = installDSparkModels(executor, 3);
                draft_inputs = &draft->inputs;
            } else if (type == SP_TYPE_MTP) {
                auto draft = std::make_unique<RecordingDraftModel>();
                draft_inputs = &draft->inputs;
                executor.draft_model_ = std::move(draft);
            }
            ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
            ASSERT_EQ(wire->sent_tensors.size(), 2u);
            EXPECT_TRUE(executor.sampling_states_.empty());
            const auto result = lastResult(*wire);
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

TEST_F(PPExecutorTest, LastDraftForwardUsesDraftCacheStridesWithoutMutatingTargetInput) {
    for (bool multiple_groups : {false, true}) {
        SCOPED_TRACE(multiple_groups);
        auto params = makeStageParams(2, SP_TYPE_MTP, 1);
        auto cache_config = test::makeSimpleMhaCacheConfig(1, 16, 4, DataType::TYPE_FP16);
        const auto draft_config = cache_config;
        if (multiple_groups) {
            cache_config.layer_num = 2;
            cache_config.fromGroupedSpecs({test::makeMhaSpec("default", 4, TYPE_FP16, 1, 1),
                                           test::makeMhaSpec("extra", 4, TYPE_FP16, 1, 1)},
                                          {{0}, {1}},
                                          {CacheGroupType::FULL, CacheGroupType::FULL});
        }
        cache_config.mtp_sub_configs.push_back(cache_config.mergeMTPModule(draft_config, 0, cache_config.layer_num));
        cache_config.finalizeBlockNums(16, RuntimeConfig{});
        auto cache = std::make_shared<KVCacheManager>(cache_config);
        ASSERT_TRUE(cache->init());
        PPExecutor executor(params, cache, false);
        auto draft = std::make_unique<RecordingDraftModel>();
        auto* recorded = draft.get();
        executor.draft_model_ = std::move(draft);
        executor.fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>();
        auto plan = makePlan(params);
        plan.model_input.kv_block_stride_bytes = 17;
        plan.model_input.kv_scale_stride_bytes = 19;
        const auto hidden = torch::zeros({5, 4}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
        PPExecutionResult execution_result;
        execution_result.new_token_ids     = intTensor({7, 8}).reshape({2, 1});
        execution_result.new_token_lengths = intTensor({1, 1});
        executor.runDraftStep(plan.model_input, {}, hidden, false, plan.sampling_plan, execution_result);
        ASSERT_EQ(recorded->inputs.size(), 1u);
        const auto& expected = cache->getMTPModuleCacheConfig(0);
        ASSERT_EQ(expected.groupNums(), multiple_groups ? 2 : 1);
        if (multiple_groups) {
            /** A draft placeholder group still makes the legacy single-group stride hints ambiguous. */
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

TEST_F(PPExecutorTest, LastDraftChainUsesEachForwardHiddenAndPreservesTargetInput) {
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

        PPSamplingPlan   sampling_plan;
        PPExecutionResult execution_result;
        execution_result.new_token_ids     = intTensor({2, 5}).reshape({2, 1});
        execution_result.new_token_lengths = intTensor({1, 1});
        executor.runDraftStep(target_input,
                              torch::Tensor(),
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
        for (int64_t step = 1; step < 3; ++step) {
            const auto& input = recorded->inputs[step];
            EXPECT_TRUE(torch::equal(input.combo_tokens.cpu(), proposed.select(1, step - 1)));
            EXPECT_FALSE(input.request_id.defined());
            EXPECT_FALSE(input.request_pd_separation.defined());
            EXPECT_FALSE(input.cache_keys.defined());
            auto expected_hidden = step == 1 ? torch::tensor({{4.f, 5.f, 6.f, 7.f}, {16.f, 17.f, 18.f, 19.f}}) :
                                               torch::arange(8, torch::kFloat32).reshape({2, 4}) + 100;
            EXPECT_TRUE(torch::equal(input.last_hidden_states.cpu(), expected_hidden.narrow(1, 0, hidden_width)));
        }
        EXPECT_EQ(tensorToVector<int32_t>(target_input.combo_tokens), (std::vector<int32_t>{0, 1, 0, 3, 4}));
        EXPECT_EQ(tensorToVector<int32_t>(target_input.input_lengths), (std::vector<int32_t>{2, 3}));
        EXPECT_EQ(tensorToVector<int32_t>(target_input.prefix_lengths), (std::vector<int32_t>{3, 5}));
        EXPECT_TRUE(torch::equal(target_hidden.cpu(),
                                 torch::arange(20, torch::kFloat32).reshape({5, 4}).narrow(1, 0, hidden_width)));
    }
}

/** Last-stage TP/CP: replay one stage's root broadcasts and retain rank-local CP features. */
TEST_F(PPExecutorTpTest, LastTpVerifySynchronizesAcceptedRowsAndKeepsPeersOutOfSampling) {
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
                auto params = makeStageParams(2, type);
                params.parallelism_config.tp_size = 2;
                params.parallelism_config.world_size = 6;
                params.parallelism_config.tp_rank = rank;
                params.parallelism_config.world_rank = 4 + rank;
                auto propose = makeMtpProposeParams(params);
                PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
                auto plan = makePlan(params, true);
                if (type == SP_TYPE_MTP) {
                    executor.position_id_len_factor_ = 2;
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
                std::vector<GptModelInputs>* draft_inputs;
                if (type == SP_TYPE_DSPARK) {
                    auto draft = installDSparkModels(executor, 3);
                    draft->on_forward = [&] { forward_positions.push_back(broadcasts.position()); };
                    draft_inputs = &draft->inputs;
                } else {
                    auto draft = std::make_unique<RecordingDraftModel>();
                    draft->on_forward = [&] { forward_positions.push_back(broadcasts.position()); };
                    draft_inputs = &draft->inputs;
                    executor.draft_model_ = std::move(draft);
                }
                if (rank == 0) {
                    auto failed = recordState(executor, 202);
                    failed->verify_error = ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION, "verify failed");
                    if (all_failed) {
                        recordState(executor, 101)->verify_error = failed->verify_error;
                    }
                }
                ASSERT_TRUE(executor.process(ScheduleOutput{}).ok());
                ASSERT_EQ(recorded_target->inputs.size(), 1u);
                EXPECT_TRUE(recorded_target->inputs[0].is_target_verify);
                EXPECT_FALSE(recorded_target->inputs[0].pp_intermediates.empty());
                ASSERT_EQ(draft_inputs->size(), type == SP_TYPE_MTP ? 3u : 2u);
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
                    EXPECT_EQ(broadcasts.position(), broadcasts.size());
                    EXPECT_EQ(forward_positions, root_forward_positions);
                    for (size_t step = 0; step < draft_inputs->size(); ++step) {
                        EXPECT_TRUE(torch::equal(draft_inputs->at(step).combo_tokens.cpu(),
                                                 root_inputs[step].combo_tokens.cpu()));
                        EXPECT_TRUE(torch::equal(draft_inputs->at(step).input_lengths.cpu(),
                                                 root_inputs[step].input_lengths.cpu()));
                        EXPECT_TRUE(torch::equal(draft_inputs->at(step).prefix_lengths.cpu(),
                                                 root_inputs[step].prefix_lengths.cpu()));
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
                auto params = makeStageParams(2, type);
                params.pd_sep_config.role_type = RoleType::PREFILL;
                params.parallelism_config.tp_size = 2;
                params.parallelism_config.world_size = 6;
                params.parallelism_config.tp_rank = rank;
                params.parallelism_config.world_rank = 4 + rank;
                params.parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
                auto propose = makeMtpProposeParams(params);
                PPExecutor executor(params, nullptr, false, MlaOpsType::AUTO, nullptr, nullptr, propose.get());
                const auto plan = fake ? makeFakePlan(params, false) : makePlan(params);
                auto* wire = attachTransport(executor);
                enqueuePlan(*wire, plan, rank != 0);
                auto target = std::make_unique<RecordingCPTargetModel>(true, rank);
                auto* recorded_target = target.get();
                executor.setModel(std::move(target));
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
