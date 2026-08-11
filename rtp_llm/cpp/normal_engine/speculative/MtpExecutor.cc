#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/SpeculativeConfigValidator.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpSamplerFailureValidator.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/TimeUtility.h"
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <thread>
#include <random>

namespace rtp_llm {

namespace {

enum class SamplerRowLayout { NORMAL, SCORE };

struct GeneratorStateSnapshot {
    at::Generator generator;
    torch::Tensor state;
};

struct StreamIterationSnapshot {
    GenerateStreamPtr stream;
    size_t            iter_count;
};

std::vector<GenerateStreamPtr> batchStreams(const StreamGroups& stream_groups) {
    const auto streams = stream_groups.allStreams();
    return {streams.begin(), streams.end()};
}

std::vector<StreamIterationSnapshot> captureStreamIterationStates(
    const std::vector<GenerateStreamPtr>& streams) {
    std::vector<StreamIterationSnapshot> snapshots;
    snapshots.reserve(streams.size());
    for (const auto& stream : streams) {
        snapshots.push_back({stream, stream->iterCount()});
    }
    return snapshots;
}

void restoreStreamIterationStates(const std::vector<StreamIterationSnapshot>& snapshots) {
    for (const auto& snapshot : snapshots) {
        snapshot.stream->restoreIterCount(snapshot.iter_count);
    }
}

std::vector<size_t> samplerRowCounts(const std::vector<GenerateStreamPtr>& streams, SamplerRowLayout layout) {
    std::vector<size_t> row_counts;
    row_counts.reserve(streams.size());
    for (const auto& stream : streams) {
        if (layout == SamplerRowLayout::SCORE) {
            row_counts.push_back(stream->scoreLen());
            continue;
        }
        const int row_count =
            stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();
        if (row_count <= 0) {
            throw std::invalid_argument("sampler stream row count must be positive");
        }
        row_counts.push_back(static_cast<size_t>(row_count));
    }
    return row_counts;
}

void reportAllStreamsError(const std::vector<GenerateStreamPtr>& streams,
                           ErrorCode                            error_code,
                           const std::string&                   message) {
    for (const auto& stream : streams) {
        if (!stream->hasError()) {
            stream->reportError(error_code, message);
        }
    }
}

bool reportSamplerFailures(const SamplerOutput&                  sampler_output,
                           const std::vector<GenerateStreamPtr>& streams,
                           SamplerRowLayout                      layout) {
    if (!sampler_output.success.defined()) {
        return false;
    }
    const auto failed_stream_indices = speculative::findFailedSamplerStreamIndices(
        sampler_output.success, samplerRowCounts(streams, layout));
    for (const size_t stream_index : failed_stream_indices) {
        streams[stream_index]->reportError(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
    }
    return !failed_stream_indices.empty();
}

bool reportSampledTokenInputVocabFailures(const torch::Tensor&                  token_ids,
                                          const std::vector<GenerateStreamPtr>& streams,
                                          SamplerRowLayout                      layout,
                                          size_t                                target_input_vocab_size,
                                          size_t                                propose_input_vocab_size,
                                          bool                                  allow_int64 = false) {
    if (!token_ids.defined()) {
        throw std::invalid_argument("sampler token ids are undefined");
    }
    const auto token_dtype = token_ids.scalar_type();
    if ((token_dtype != torch::kInt32 && !(allow_int64 && token_dtype == torch::kInt64)) || token_ids.dim() != 2
        || !token_ids.is_contiguous() || token_ids.size(1) <= 0) {
        throw std::invalid_argument(
            allow_int64 ? "sampler token ids must be a contiguous int32 or int64 tensor with shape [rows, width]" :
                          "sampler token ids must be a contiguous int32 tensor with shape [rows, width]");
    }
    if (!token_ids.is_cpu() && !token_ids.is_cuda()) {
        throw std::invalid_argument("sampler token ids must be on CPU or CUDA");
    }

    const auto row_counts   = samplerRowCounts(streams, layout);
    size_t     expected_rows = 0;
    for (const size_t row_count : row_counts) {
        if (row_count > std::numeric_limits<size_t>::max() - expected_rows) {
            throw std::invalid_argument("sampler token id row count overflow");
        }
        expected_rows += row_count;
    }
    if (static_cast<size_t>(token_ids.size(0)) != expected_rows) {
        throw std::invalid_argument("sampler token id row count does not match the executor batch");
    }

    const auto token_ids_cpu = token_ids.is_cuda() ? token_ids.cpu() : token_ids;
    const auto token_stride  = static_cast<size_t>(token_ids_cpu.size(1));

    bool   has_failure = false;
    size_t row_offset  = 0;
    for (size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        bool stream_failed = false;
        for (size_t row = 0; row < row_counts[stream_index]; ++row) {
            const size_t token_index = (row_offset + row) * token_stride + token_stride - 1;
            const int64_t token      = token_dtype == torch::kInt64 ?
                                           token_ids_cpu.data_ptr<int64_t>()[token_index] :
                                           token_ids_cpu.data_ptr<int32_t>()[token_index];
            stream_failed |= token < 0 || static_cast<uint64_t>(token) >= target_input_vocab_size
                             || static_cast<uint64_t>(token) >= propose_input_vocab_size;
        }
        if (stream_failed) {
            streams[stream_index]->reportError(ErrorCode::OUT_OF_VOCAB_RANGE,
                                               "sampled token is outside a model input vocabulary");
            has_failure = true;
        }
        row_offset += row_counts[stream_index];
    }
    return has_failure;
}

bool reportAcceptedTokenInputVocabFailures(
    const speculative::SpeculativeSamplerOutput& speculative_output,
    const std::vector<GenerateStreamPtr>&         streams,
    size_t                                        target_input_vocab_size,
    size_t                                        propose_input_vocab_size) {
    if (speculative_output.accept_tokens.size() != streams.size()
        || speculative_output.accept_len.size() != streams.size()) {
        throw std::invalid_argument("speculative sampler output size does not match the executor batch");
    }

    bool has_failure = false;
    for (size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        const int   accept_len = speculative_output.accept_len[stream_index];
        const auto& tokens     = speculative_output.accept_tokens[stream_index];
        const bool  valid_contract = accept_len > 0 && tokens.defined() && tokens.is_cpu()
                                     && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous()
                                     && tokens.dim() == 2 && tokens.size(0) == 1 && tokens.size(1) == accept_len;
        if (!valid_contract) {
            throw std::invalid_argument(
                "accepted token ids must be a contiguous CPU int32 tensor with shape [1, accept_len]");
        }

        bool        stream_failed = false;
        const auto* token_ptr     = tokens.data_ptr<int32_t>();
        for (int token_index = 0; token_index < accept_len; ++token_index) {
            const int32_t token = token_ptr[token_index];
            stream_failed |= token < 0 || static_cast<uint64_t>(token) >= target_input_vocab_size
                             || static_cast<uint64_t>(token) >= propose_input_vocab_size;
        }
        if (stream_failed) {
            streams[stream_index]->reportError(ErrorCode::OUT_OF_VOCAB_RANGE,
                                               "accepted token is outside a model input vocabulary");
            has_failure = true;
        }
    }
    return has_failure;
}

void syncSkipRunAcrossTensorParallelRanks(bool& skip_run, const ParallelismConfig& parallelism_config) {
    if (parallelism_config.tp_size <= 1) {
        return;
    }
    auto skip_flag = torch::zeros({1}, torch::kInt32).pin_memory();
    if (parallelism_config.tp_rank == 0) {
        skip_flag.data_ptr<int32_t>()[0] = skip_run ? 1 : 0;
    }
    execBroadcast({{skip_flag}, 0});
    execSyncCommunication(false);
    cudaSyncAndCheck();
    skip_run = skip_flag.data_ptr<int32_t>()[0] != 0;
}

void reportSamplingException(const std::vector<GenerateStreamPtr>& streams, const std::string& message) {
    RTP_LLM_LOG_ERROR("MTP sampling failed: %s", message.c_str());
    reportAllStreamsError(streams, ErrorCode::EXECUTION_EXCEPTION, message);
}

std::vector<GeneratorStateSnapshot> captureSeededGeneratorStates(const std::vector<GenerateStreamPtr>& streams) {
    std::vector<GeneratorStateSnapshot> snapshots;
    for (const auto& stream : streams) {
        if (stream->generateConfig()->random_seed.has_value()) {
            auto generator = stream->getGenerator();
            snapshots.push_back({generator, generator.get_state()});
        }
    }
    return snapshots;
}

void restoreSeededGeneratorStates(const std::vector<GeneratorStateSnapshot>& snapshots) noexcept {
    for (const auto& snapshot : snapshots) {
        try {
            auto generator = snapshot.generator;
            generator.set_state(snapshot.state);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("failed to restore sampler generator state after batch abort: %s", e.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("failed to restore sampler generator state after batch abort");
        }
    }
}

void clearDecodeTensorHolders(const std::list<GenerateStreamPtr>& streams) {
    for (const auto& stream : streams) {
        const auto& sp_output_buffer = stream->getSPOutputBuffer();
        if (sp_output_buffer) {
            sp_output_buffer->tensors_holder.clear();
        }
    }
}

std::optional<std::string> validatePrefillSpeculativeInput(const GenerateStreamPtr& stream,
                                                           size_t                   target_input_vocab_size,
                                                           size_t                   propose_input_vocab_size) {
    if (!stream) {
        return "speculative stream is missing";
    }
    if (!stream->isContextStream() || stream->currentBatchSize() <= 0) {
        return "speculative prefill requires a context stream with a positive batch size";
    }

    const auto text_token_mask = stream->textTokensMask();
    for (int batch_index = 0; batch_index < stream->currentBatchSize(); ++batch_index) {
        const auto input_tokens = stream->currentExecuteTokens(batch_index);
        if (input_tokens.empty()) {
            return "speculative prefill input must not be empty";
        }
        for (size_t token_index = 0; token_index < input_tokens.size(); ++token_index) {
            const bool is_text_token = token_index >= text_token_mask.size() || text_token_mask[token_index];
            if (!is_text_token) {
                return "speculative prefill does not support non-text tokens";
            }
            const int token = input_tokens[token_index];
            if (token < 0 || static_cast<uint64_t>(token) >= static_cast<uint64_t>(target_input_vocab_size)) {
                return "speculative prefill token is outside the target input vocabulary";
            }
            if (token_index > 0
                && static_cast<uint64_t>(token) >= static_cast<uint64_t>(propose_input_vocab_size)) {
                return "speculative prefill token is outside the proposal input vocabulary";
            }
        }
    }
    return std::nullopt;
}

bool reportInvalidPrefillSpeculativeInputs(const std::vector<GenerateStreamPtr>& streams,
                                           size_t                                target_input_vocab_size,
                                           size_t                                propose_input_vocab_size) {
    bool has_failure = false;
    for (const auto& stream : streams) {
        const auto error =
            validatePrefillSpeculativeInput(stream, target_input_vocab_size, propose_input_vocab_size);
        if (!error.has_value()) {
            continue;
        }
        if (!stream) {
            RTP_LLM_LOG_ERROR("invalid speculative prefill input: %s", error->c_str());
            has_failure = true;
            continue;
        }
        RTP_LLM_LOG_ERROR(
            "invalid speculative prefill input for stream %ld: %s", stream->streamId(), error->c_str());
        stream->reportError(ErrorCode::INVALID_PARAMS, "invalid speculative prefill input");
        has_failure = true;
    }
    return has_failure;
}

std::optional<std::string> validateDecodeSpeculativeInput(const GenerateStreamPtr& stream,
                                                          size_t                   propose_step,
                                                          size_t                   propose_vocab_size,
                                                          size_t                   target_input_vocab_size,
                                                          size_t                   propose_input_vocab_size,
                                                          size_t                   propose_hidden_size,
                                                          c10::ScalarType          propose_hidden_dtype) {
    if (!stream) {
        return "speculative stream is missing";
    }
    const auto& output = stream->getSPOutputBuffer();
    if (!output) {
        return "speculative output buffer is missing";
    }

    const auto& tokens = output->tokens;
    if (!tokens.defined() || !tokens.is_cpu() || tokens.scalar_type() != torch::kInt32 || !tokens.is_contiguous()
        || tokens.dim() != 2 || tokens.size(0) != 1 || tokens.size(1) != 2) {
        return "speculative tokens must be a contiguous CPU int32 tensor with shape [1, 2]";
    }

    if (stream->currentBatchSize() != 1 || stream->seqLength() <= 0) {
        return "speculative decode requires a non-empty single-sequence stream";
    }
    const auto* token_values = tokens.data_ptr<int32_t>();
    for (size_t token_index = 0; token_index < 2; ++token_index) {
        if (token_values[token_index] < 0
            || static_cast<uint64_t>(token_values[token_index]) >= static_cast<uint64_t>(propose_vocab_size)) {
            return "speculative token is outside the proposal vocabulary";
        }
        if (static_cast<uint64_t>(token_values[token_index]) >= static_cast<uint64_t>(target_input_vocab_size)
            || static_cast<uint64_t>(token_values[token_index])
                   >= static_cast<uint64_t>(propose_input_vocab_size)) {
            return "speculative token is outside a model input vocabulary";
        }
    }
    const auto latest_tokens = stream->getLatestTokens(1);
    if (latest_tokens.size() != 1 || token_values[0] != latest_tokens[0]) {
        return "speculative target token does not match the stream tail";
    }

    const auto& holders = output->tensors_holder;
    if (!holders.empty() && holders.size() != 2) {
        return "speculative tensor holder must contain either zero or two tensors";
    }

    const bool  is_remote_holder = !holders.empty();
    const auto& probs            = is_remote_holder ? holders[0] : output->all_probs;
    const auto& hidden           = is_remote_holder ? holders[1] : output->hidden_states;
    const auto  has_valid_device = [is_remote_holder](const torch::Tensor& tensor) {
        return is_remote_holder ? tensor.is_cpu() : tensor.is_cuda();
    };

    if (!probs.defined() || probs.scalar_type() != torch::kFloat32 || !probs.is_contiguous() || probs.dim() != 2
        || probs.size(0) != 1 || static_cast<size_t>(probs.size(1)) != propose_vocab_size) {
        return "speculative probabilities must be contiguous float32 with shape [1, proposal_vocab_size]";
    }
    if (!has_valid_device(probs)) {
        return is_remote_holder ? "speculative probability holder must be a CPU tensor" :
                                  "local speculative probabilities must be an accelerator tensor";
    }

    if (propose_step > 1) {
        if (!hidden.defined() || hidden.scalar_type() != propose_hidden_dtype || !hidden.is_contiguous()
            || hidden.dim() != 2 || hidden.size(0) != 1
            || static_cast<size_t>(hidden.size(1)) != propose_hidden_size) {
            return "speculative hidden states have an invalid dtype or shape";
        }
    } else if (is_remote_holder && !hidden.defined()) {
        return "speculative hidden-state holder is undefined";
    } else if (hidden.defined() && hidden.numel() > 0
               && (hidden.scalar_type() != propose_hidden_dtype || !hidden.is_contiguous() || hidden.dim() != 2
                   || hidden.size(0) != 1 || static_cast<size_t>(hidden.size(1)) != propose_hidden_size)) {
        return "optional speculative hidden states have an invalid dtype or shape";
    }
    if (hidden.defined() && !has_valid_device(hidden)) {
        return is_remote_holder ? "speculative hidden-state holder must be a CPU tensor" :
                                  "local speculative hidden states must be an accelerator tensor";
    }
    return std::nullopt;
}

bool reportInvalidDecodeSpeculativeInputs(const std::vector<GenerateStreamPtr>& streams,
                                          size_t                                propose_step,
                                          size_t                                propose_vocab_size,
                                          size_t                                target_input_vocab_size,
                                          size_t                                propose_input_vocab_size,
                                          size_t                                propose_hidden_size,
                                          c10::ScalarType                       propose_hidden_dtype) {
    bool has_failure = false;
    for (const auto& stream : streams) {
        const auto error = validateDecodeSpeculativeInput(
            stream,
            propose_step,
            propose_vocab_size,
            target_input_vocab_size,
            propose_input_vocab_size,
            propose_hidden_size,
            propose_hidden_dtype);
        if (!error.has_value()) {
            continue;
        }
        if (!stream) {
            RTP_LLM_LOG_ERROR("invalid speculative decode input: %s", error->c_str());
            has_failure = true;
            continue;
        }
        RTP_LLM_LOG_ERROR("invalid speculative decode input for stream %ld: %s",
                          stream->streamId(),
                          error->c_str());
        stream->reportError(ErrorCode::INVALID_PARAMS, "invalid speculative decode input");
        if (const auto& output = stream->getSPOutputBuffer()) {
            output->tensors_holder.clear();
        }
        has_failure = true;
    }
    return has_failure;
}

void materializeDecodeTensorHolders(const std::vector<GenerateStreamPtr>& streams) {
    struct PendingTensors {
        SpeculativeExecutorStreamOutputPtr output;
        torch::Tensor                      probs;
        torch::Tensor                      hidden;
    };

    std::vector<PendingTensors> pending;
    pending.reserve(streams.size());
    for (const auto& stream : streams) {
        const auto& output = stream->getSPOutputBuffer();
        if (output->tensors_holder.empty()) {
            continue;
        }
        pending.push_back({output,
                           output->tensors_holder[0].to(torch::kCUDA).clone(),
                           output->tensors_holder[1].to(torch::kCUDA).clone()});
    }

    for (auto& tensors : pending) {
        tensors.output->all_probs     = std::move(tensors.probs);
        tensors.output->hidden_states = std::move(tensors.hidden);
    }
}

void reportInputPreparationFailure(const std::vector<GenerateStreamPtr>& streams,
                                   ErrorCode                            error_code,
                                   const std::string&                   detail) {
    RTP_LLM_LOG_ERROR("MTP input preparation failed: %s", detail.c_str());
    reportAllStreamsError(streams, error_code, "speculative input preparation failed");
}

}  // namespace

bool MtpExecutor::isTpRank0() const {
    return tp_rank_ == 0;
}

void MtpExecutor::maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const {
    bool force = tp_rank_ == 0 && enable_detail_log_;
    if (force) {
        RTP_LLM_LOG_INFO("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    } else {
        RTP_LLM_LOG_DEBUG("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    }
}

void MtpExecutor::releaseModelBuffers() {
    if (model_) {
        model_->releaseBuffers();
    }
    if (draft_model_) {
        draft_model_->releaseBuffers();
    }
    if (sp_prefill_draft_model_ && sp_prefill_draft_model_.get() != draft_model_.get()) {
        sp_prefill_draft_model_->releaseBuffers();
    }
}

static std::shared_ptr<NormalGenerateStream> makeFakeStream(int                    max_new_tokens,
                                                            size_t                 reserved_blocks,
                                                            const ModelConfig&     model_config,
                                                            const RuntimeConfig&   runtime_config,
                                                            const ResourceContext& resource_context) {
    std::shared_ptr<GenerateInput> fake_input   = std::make_shared<GenerateInput>();
    fake_input->input_ids                       = torch::zeros({1}, torch::kInt32);
    fake_input->generate_config                 = std::make_shared<GenerateConfig>();
    fake_input->generate_config->max_new_tokens = max_new_tokens;
    fake_input->generate_config->top_k          = 1;
    fake_input->begin_time_us                   = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->fake_query                      = true;

    auto fake_stream = std::make_shared<NormalGenerateStream>(
        fake_input, model_config, runtime_config, resource_context, nullptr, max_new_tokens);
    fake_stream->setIsFakeStream(true);
    fake_stream->setMetricsReporter(nullptr);
    fake_stream->fakeInitKVBlock(reserved_blocks);

    return fake_stream;
}

static SpeculativeExecutorStreamOutputPtr makeFakeSPOutputBuffer(
    DataType data_type, size_t hidden_size, size_t vocab_size, size_t propose_step, int32_t initial_token) {
    RTP_LLM_CHECK_WITH_INFO(vocab_size > 0, "fake speculative output requires a non-empty vocabulary");
    RTP_LLM_CHECK_WITH_INFO(vocab_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                            "vocabulary size does not fit int64_t: %zu",
                            vocab_size);
    RTP_LLM_CHECK_WITH_INFO(initial_token >= 0 && static_cast<size_t>(initial_token) < vocab_size,
                            "fake speculative token %d is outside vocabulary size %zu",
                            initial_token,
                            vocab_size);
    auto sp_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();

    auto fake_hidden_states = torch::zeros(
        {1, (int64_t)hidden_size}, torch::TensorOptions().dtype(dataTypeToTorchType(data_type)).device(torch::kCUDA));
    auto fake_probs = torch::full({1, static_cast<int64_t>(vocab_size)},
                                  1.0 / static_cast<double>(vocab_size),
                                  torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    sp_buffer->propose_step  = propose_step;
    sp_buffer->all_probs     = fake_probs;
    sp_buffer->tokens        = torch::tensor({initial_token, initial_token}, torch::kInt32).reshape({1, 2});
    sp_buffer->hidden_states = fake_hidden_states;

    return sp_buffer;
}

GenerateStreamPtr MtpExecutor::createMinFakePrefillStream(int                    max_new_tokens,
                                                          const ModelConfig&     model_config,
                                                          const RuntimeConfig&   runtime_config,
                                                          const ResourceContext& resource_context) {
    return makeFakeStream(max_new_tokens, 1, model_config, runtime_config, resource_context);
}

GenerateStreamPtr MtpExecutor::createMinFakeDecodeStream(int                    max_new_tokens,
                                                         const ModelConfig&     target_model_config,
                                                         const ModelConfig&     proposal_model_config,
                                                         const RuntimeConfig&   runtime_config,
                                                         const ResourceContext& resource_context) {
    auto fake_stream = makeFakeStream(
        max_new_tokens, 1 + max_new_tokens, target_model_config, runtime_config, resource_context);

    auto sp_buffer = makeFakeSPOutputBuffer(proposal_model_config.data_type,
                                            proposal_model_config.hidden_size,
                                            proposal_model_config.vocab_size,
                                            max_new_tokens,
                                            0);

    auto new_tokens = torch::zeros({1, 1}, torch::kInt32);

    StreamUpdateInfo update_info{new_tokens,
                                 1,
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 false};

    fake_stream->update(update_info);
    fake_stream->setSPOutputBuffer(sp_buffer);
    return fake_stream;
}

MtpExecutor::MtpExecutor(const EngineInitParams&                        params,
                         std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                         const std::shared_ptr<KVCacheManager>&         cache_manager,
                         MlaOpsType                                     mla_ops_type,
                         int32_t                                        kv_cache_group_num,
                         const std::vector<int32_t>&                    kv_cache_layer_to_group,
                         bool                                           warm_up,
                         std::optional<RoleType>                        role_override):
    Executor(),
    cache_manager_(cache_manager),
    metrics_reporter_(params.metrics_reporter),
    speculative_sampler_(new speculative::SpeculativeSampler(propose_params->gen_num_per_circle)),
    fast_topk_sampler_(new speculative::FastTopKSampler()),
    warm_up_(warm_up),
    role_type_(role_override.value_or(params.pd_sep_config.role_type)) {
    data_type_               = propose_params->getEngineInitParams().model_config_.data_type;
    hidden_size_             = propose_params->getEngineInitParams().model_config_.hidden_size;
    propose_step_            = propose_params->gen_num_per_circle;
    vocab_size_              = params.model_config_.vocab_size;
    propose_vocab_size_      = propose_params->getEngineInitParams().model_config_.vocab_size;
    target_input_vocab_size_ = effectiveInputVocabSize(params.model_config_);
    propose_input_vocab_size_ =
        effectiveInputVocabSize(propose_params->getEngineInitParams().model_config_);
    sampled_token_input_vocab_guard_required_ = target_input_vocab_size_ < vocab_size_
                                                || propose_input_vocab_size_ < vocab_size_;

    enable_detail_log_  = params.profiling_debug_logging_config.enable_detail_log;
    tp_rank_            = params.parallelism_config.tp_rank;
    parallelism_config_ = params.parallelism_config;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d, tp_rank_ = %d", enable_detail_log_, tp_rank_);

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        // use first moe layer weight as moe weight type
        int         first_moe_layer = params.model_config_.moe_layer_index.front();
        const auto& moe_kernel      = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel;
        auto        moe_weight_type = torchDTypeToDataType(moe_kernel.dtype());
        bool        is_gated_activation = params.model_config_.isGatedActivation();
        auto        moe_inter_size      = is_gated_activation ? moe_kernel.size(1) / 2 : moe_kernel.size(1);

        expert_balancer_ =
            std::make_shared<ExpertBalancer>(params.model_config_.expert_num,
                                             params.eplb_config.phy_exp_num(params.model_config_.expert_num),
                                             params.model_config_.num_layers,
                                             moe_inter_size,
                                             params.model_config_.hidden_size,
                                             params.parallelism_config.ep_rank,
                                             params.parallelism_config.ep_size,
                                             params.py_eplb,
                                             moe_weight_type,
                                             params.model_config_.quant_algo,
                                             metrics_reporter_,
                                             params.eplb_config);
    }

    sampler_.reset(new Sampler(SamplerInitParams{}));

    // Optional per-layer cache buffers from KVCacheManager::allLayerCacheBase().
    std::optional<CacheLayerLayout> kv_cache_layer_layout = std::nullopt;
    if (cache_manager && cache_manager->cacheConfig().groupNums() > 1) {
        kv_cache_layer_layout = cache_manager->allLayerCacheBase();
    }

    auto target_cache_layer_layout = cache_manager->getMainModelCacheLayerLayout();
    auto draft_cache_layer_layout  = cache_manager->getMTPModuleCacheLayerLayout(0);

    GptModelInitParams model_init_params(
        {params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager ? std::make_optional(target_cache_layer_layout) : std::nullopt,
         params.model_id,
         params.parallelism_config,
         params.hw_kernel_config,
         params.profiling_debug_logging_config,
         params.runtime_config,
         params.concurrency_config,
         params.sp_config,
         params.device_resource_config,
         mla_ops_type,
         params.model_config_.max_seq_len,
         params.model_config_.hidden_size,
         params.model_config_.attn_config.tokens_per_block,
         params.model_config_.attn_config.kernel_tokens_per_block,
         kv_cache_group_num,
         kv_cache_layer_to_group,
         cache_manager});

    if (params.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }

    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(
            model_init_params, params.py_model, false, true, target_cache_layer_layout.layer_to_groups));
    }

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    auto executor_pd_sep_config = params.pd_sep_config;
    executor_pd_sep_config.role_type = role_type_;
    batch_stream_processor_.reset(new MtpBatchStreamProcessor(params.model_config_,
                                                              executor_pd_sep_config,
                                                              params.profiling_debug_logging_config,
                                                              cache_config,
                                                              params.sp_config,
                                                              warm_up_));

    LogitsProcessorFactory::init(params.model_config_.ckpt_path, params.sp_config.tree_decode_config);
    cudaProfilerBegin();

    for (auto& mtp_params : *propose_params->mtp_model_params_) {
        auto model_params =
            GptModelInitParams({mtp_params->gpt_weights,
                                Executor::genModelDescription(mtp_params->model_config_,
                                                              mtp_params->parallelism_config,
                                                              mtp_params->eplb_config,
                                                              mtp_params->moe_config),
                                cache_manager ? std::make_optional(draft_cache_layer_layout) : std::nullopt,
                                mtp_params->model_id,
                                mtp_params->parallelism_config,
                                params.hw_kernel_config,
                                params.profiling_debug_logging_config,
                                params.runtime_config,
                                params.concurrency_config,
                                params.sp_config,
                                params.device_resource_config,
                                mla_ops_type,
                                mtp_params->model_config_.max_seq_len,
                                mtp_params->model_config_.hidden_size,
                                mtp_params->model_config_.attn_config.tokens_per_block,
                                mtp_params->model_config_.attn_config.kernel_tokens_per_block,
                                kv_cache_group_num,
                                kv_cache_layer_to_group,
                                cache_manager});
        if (!params.py_sp_model.is_none()) {
            RTP_LLM_LOG_INFO("[speculative decoding] using py model");
            draft_model_.reset(new PyWrappedModel(
                model_params, params.py_sp_model, false, false, draft_cache_layer_layout.layer_to_groups));
            // Create separate model for speculative prefill with CUDA graph if enabled (from params)
            const bool enable_cuda_graph = params.hw_kernel_config.enable_cuda_graph;
            RTP_LLM_LOG_INFO(
                "[speculative decoding] enable_cuda_graph=%d (set ENABLE_CUDA_GRAPH=1 when starting server to enable sp_prefill_draft_model_)",
                static_cast<int>(enable_cuda_graph));
            if (enable_cuda_graph) {
                RTP_LLM_LOG_INFO(
                    "[speculative decoding] creating separate prefill draft model with CUDA graph support");
                sp_prefill_draft_model_.reset(new PyWrappedModel(
                    model_params, params.py_sp_model, true, false, draft_cache_layer_layout.layer_to_groups));
            }
        }
        break;  // NOTE: only support one mtp model now
    }

    target_kv_cache_layer_to_group =
        torch::empty({(int64_t)target_cache_layer_layout.layers_to_kv_buffer_ptrs.size()}, torch::kInt32);
    draft_kv_cache_layer_to_group =
        torch::empty({(int64_t)draft_cache_layer_layout.layers_to_kv_buffer_ptrs.size()}, torch::kInt32);

    memcpy(target_kv_cache_layer_to_group.data_ptr<int>(),
           target_cache_layer_layout.layer_to_groups.data(),
           target_cache_layer_layout.layer_to_groups.size() * sizeof(int));
    memcpy(draft_kv_cache_layer_to_group.data_ptr<int>(),
           draft_cache_layer_layout.layer_to_groups.data(),
           draft_cache_layer_layout.layer_to_groups.size() * sizeof(int));
}

/*
 * @brief mtp prefill step:
 *
 * +-----------------------------+
 * |     gather model input      |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |    target model forward     |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     target model sample     |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     update model input      |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     draft model forward     |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     draft model sample      |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |  dispatch output to streams |
 * +-----------------------------+
 *
 * @param streams
 * @return absl::Status
 */
absl::Status MtpExecutor::prefillStep(const std::list<GenerateStreamPtr>& streams,
                                      MtpMetricsCollector&                metrics_collector) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.prefill_step(prefill_stream_size=%zu)", streams.size());

    RtpLLMExecutorMetricsCollector& executor_collector = metrics_collector.executor_collector;
    RtpLLMTokenPSMetricsCollector&  tps_collector      = metrics_collector.tps_collector;

    StreamGroups    stream_groups(streams);
    const auto      iteration_snapshots = captureStreamIterationStates(batchStreams(stream_groups));
    GptModelInputs  model_input;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    GptModelOutputs draft_model_output;
    SamplerOutput   draft_sampler_output;
    std::vector<GeneratorStateSnapshot> generator_snapshots;

    // placeholder for some tensors
    torch::Tensor                      draft_probs;
    torch::Tensor                      draft_token_ids;
    speculative::FastTopKSamplerOutput fast_topk_sampler_output;

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(prepare_and_tp_sync_input)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        if (isTpRank0()) {
            model_input.skip_run = streams.empty() && !enable_ffn_disaggregate_;
            if (!model_input.skip_run) {
                const auto streams_in_batch = batchStreams(stream_groups);
                model_input.skip_run = reportInvalidPrefillSpeculativeInputs(
                    streams_in_batch, target_input_vocab_size_, propose_input_vocab_size_);
                if (!model_input.skip_run) {
                    try {
                        auto model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
                        if (!model_input_status.ok()) {
                            reportInputPreparationFailure(
                                streams_in_batch, ErrorCode::INVALID_PARAMS, model_input_status.status().ToString());
                            model_input.skip_run = true;
                        } else {
                            model_input = std::move(model_input_status.value());
                        }
                    } catch (const std::exception& e) {
                        reportInputPreparationFailure(streams_in_batch, ErrorCode::EXECUTION_EXCEPTION, e.what());
                        model_input.skip_run = true;
                    } catch (...) {
                        reportInputPreparationFailure(
                            streams_in_batch, ErrorCode::EXECUTION_EXCEPTION, "unknown prefill input exception");
                        model_input.skip_run = true;
                    }
                }
            }
        }
        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            restoreStreamIterationStates(iteration_snapshots);
            cudaSyncAndCheck();
            releaseModelBuffers();
            return warm_up_ ? absl::InternalError("prefill input preparation failed during warm-up") :
                              absl::OkStatus();
        }
        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;

    // release model input before forward
    releaseModelBuffers();

    // target model prefill
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(target_model_forward)");
        maybePrintModelInput(model_input, "prefill target model");
        model_input.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
        model_output                        = std::move(model_->forward(model_input));
    }

    // eplb
    if (expert_balancer_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(eplb_step_forward)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // target model sample
    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(target_model_sample)");
        if (model_input.is_fake_stream && !warm_up_) {
            model_input.last_hidden_states = model_output.all_hidden_states;
        } else {
            const auto                          streams_in_batch = batchStreams(stream_groups);
            try {
                auto sampler_input_status =
                    batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output);
                if (!sampler_input_status.ok()) {
                    reportSamplingException(streams_in_batch, sampler_input_status.status().ToString());
                    model_input.skip_run = true;
                } else {
                    generator_snapshots = captureSeededGeneratorStates(streams_in_batch);
                    sampler_output = std::move(sampler_->forward(sampler_input_status.value()));
                    model_input.skip_run =
                        reportSamplerFailures(sampler_output, streams_in_batch, SamplerRowLayout::NORMAL);
                    if (!model_input.skip_run && sampled_token_input_vocab_guard_required_) {
                        model_input.skip_run = reportSampledTokenInputVocabFailures(sampler_output.token_ids,
                                                                                   streams_in_batch,
                                                                                   SamplerRowLayout::NORMAL,
                                                                                   target_input_vocab_size_,
                                                                                   propose_input_vocab_size_);
                    }
                    if (!model_input.skip_run) {
                        batch_stream_processor_->updatePrefillPostDraftModelInput(
                            model_input, model_output, sampler_output);
                    }
                    if (model_input.skip_run) {
                        restoreSeededGeneratorStates(generator_snapshots);
                    }
                }
            } catch (const std::exception& e) {
                restoreSeededGeneratorStates(generator_snapshots);
                reportSamplingException(streams_in_batch, e.what());
                model_input.skip_run = true;
            } catch (...) {
                restoreSeededGeneratorStates(generator_snapshots);
                reportSamplingException(streams_in_batch, "unknown target sampling exception");
                model_input.skip_run = true;
            }
        }
    }

    // draft model prefill
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(draft_model_forward)");
        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            restoreStreamIterationStates(iteration_snapshots);
            cudaSyncAndCheck();
            releaseModelBuffers();
            return warm_up_ ? absl::InternalError("target sampling failed during warm-up") : absl::OkStatus();
        }
        maybePrintModelInput(model_input, "prefill post draft model");
        const auto& mtp_cache_cfg           = cache_manager_->getMTPModuleCacheConfig(0);
        model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        draft_model_output                  = std::move(draft_model_->forward(model_input));
    }

    if (!isTpRank0() || streams.empty() || (model_input.is_fake_stream && !warm_up_)) {
        cudaSyncAndCheck();
        releaseModelBuffers();
        return absl::OkStatus();
    }

    // draft model sample
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(draft_model_sample)");
        const auto streams_in_batch = batchStreams(stream_groups);
        try {
            fast_topk_sampler_output       = fast_topk_sampler_->forward(draft_model_output.logits);
            draft_sampler_output.all_probs = fast_topk_sampler_output.all_probs;
            draft_sampler_output.token_ids = fast_topk_sampler_output.token_ids;
            if (sampled_token_input_vocab_guard_required_) {
                model_input.skip_run = reportSampledTokenInputVocabFailures(draft_sampler_output.token_ids,
                                                                             streams_in_batch,
                                                                             SamplerRowLayout::NORMAL,
                                                                             target_input_vocab_size_,
                                                                             propose_input_vocab_size_,
                                                                             /*allow_int64=*/true);
            }
        } catch (const std::exception& e) {
            reportSamplingException(streams_in_batch, e.what());
            model_input.skip_run = true;
        } catch (...) {
            reportSamplingException(streams_in_batch, "unknown draft sampling exception");
            model_input.skip_run = true;
        }
        if (model_input.skip_run) {
            restoreSeededGeneratorStates(generator_snapshots);
            restoreStreamIterationStates(iteration_snapshots);
            cudaSyncAndCheck();
            releaseModelBuffers();
            return warm_up_ ? absl::InternalError("draft sampling failed during warm-up") : absl::OkStatus();
        }
    }

    if (warm_up_) {
        cudaSyncAndCheck();
        releaseModelBuffers();
        return absl::OkStatus();
    }

    // collect metrics
    if (metrics_reporter_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(collect_metrics)");
        executor_collector.context_batch_size = stream_groups.totalContextBatchSize();
        executor_collector.execute_token_size = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len        = stream_groups.maxSeqLen();

        executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
        executor_collector.execute_token_size_when_has_context = executor_collector.execute_token_size;
        executor_collector.max_seq_len_when_has_context        = executor_collector.max_seq_len;

        tps_collector.context_tps = stream_groups.modelExecuteTokenSize();
        tps_collector.total_tps   = tps_collector.context_tps;
    }

    // dispatch
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(dispatch_output)");
        auto result =
            batch_stream_processor_->dispatchPrefill(stream_groups,
                                                     {std::move(model_output), std::move(sampler_output)},
                                                     {std::move(draft_model_output), std::move(draft_sampler_output)});
        RTP_LLM_LOG_DEBUG("dispatch done");

        releaseModelBuffers();

        return result;
    }
}

/*
+-------------------------------+
|       gather model input      |
+-------------------------------+
        |
        v
+-------------------------------+
|     draft model forward       |<------------------+
+-------------------------------+                   |
        |                                           |
        v                              +------------------------+
+-------------------------------+      |    update model input  |
|     draft model sample        |      +------------------------+
+-------------------------------+                   |
        |                                           |
        |                                           |
        +---[if steps < propose_step-1] ------------+
        |
        |
        v
+-------------------------------+
|     update model input        |
+-------------------------------+
        |
        v
+-------------------------------+
|    target model forward       |
+-------------------------------+
        |
        v
+-------------------------------+
|     target model sample       |
+-------------------------------+
        |
        v
+-------------------------------+
|      rejection sample         |
+-------------------------------+
        |
        v
+-------------------------------+
|     update model input        |
+-------------------------------+
        |
        v
+-------------------------------+
|     draft model forward       |
+-------------------------------+
        |
        v
+-------------------------------+
|      draft model sample       |
+-------------------------------+
        |
        v
+-------------------------------+
|   dispatch output to streams  |
+-------------------------------+
*/

absl::Status MtpExecutor::decodeStep(const std::list<GenerateStreamPtr>& streams,
                                     MtpMetricsCollector&                metrics_collector) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(decode_stream_size=%zu)", streams.size());

    RtpLLMExecutorMetricsCollector&          executor_collector  = metrics_collector.executor_collector;
    RtpLLMTokenPSMetricsCollector&           tps_collector       = metrics_collector.tps_collector;
    RtpLLMSpeculativeEngineMetricsCollector& sp_engine_collector = metrics_collector.sp_engine_collector;

    StreamGroups    stream_groups(streams);
    const auto      iteration_snapshots = captureStreamIterationStates(batchStreams(stream_groups));
    GptModelInputs  model_input;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;

    GptModelOutputs                       draft_model_output;
    SamplerOutput                         draft_sampler_output;
    GptModelOutputs                       draft_prefill_model_output;
    SamplerOutput                         draft_prefill_sampler_output;
    speculative::SpeculativeSamplerOutput speculative_sampler_output;

    // placeholder for some tensors
    torch::Tensor                      draft_token_probs_d_t;
    torch::Tensor                      hidden_states_d_t;
    torch::Tensor                      draft_probs_t;
    torch::Tensor                      draft_token_ids_t;
    torch::Tensor                      spec_token_ids_t;
    std::vector<torch::Tensor>         draft_probs_list;
    speculative::FastTopKSamplerOutput fast_topk_sampler_output;
    std::vector<GeneratorStateSnapshot> generator_snapshots;

    size_t total_accept_len = 0;

    size_t batch_size = streams.size();
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_and_tp_sync_input)");
        int64_t start_time_us                = autil::TimeUtility::currentTimeInMicroSeconds();
        bool    preserve_valid_input_holders = false;
        if (isTpRank0()) {
            model_input.skip_run = streams.empty() && !enable_ffn_disaggregate_;
            if (!model_input.skip_run) {
                const auto streams_in_batch = batchStreams(stream_groups);
                preserve_valid_input_holders = reportInvalidDecodeSpeculativeInputs(
                    streams_in_batch,
                    propose_step_,
                    propose_vocab_size_,
                    target_input_vocab_size_,
                    propose_input_vocab_size_,
                    hidden_size_,
                    dataTypeToTorchType(data_type_));
                model_input.skip_run = preserve_valid_input_holders;
                if (!model_input.skip_run) {
                    try {
                        materializeDecodeTensorHolders(streams_in_batch);
                        auto model_input_status = batch_stream_processor_->gatherDecodeModelInput(stream_groups);
                        if (!model_input_status.ok()) {
                            reportInputPreparationFailure(
                                streams_in_batch, ErrorCode::INVALID_PARAMS, model_input_status.status().ToString());
                            model_input.skip_run = true;
                        } else {
                            model_input = std::move(model_input_status.value());
                            if (propose_step_ == 1) {
                                batch_stream_processor_->prepareOneStepSpecDecodeModelInput(stream_groups, model_input);
                            } else {
                                batch_stream_processor_->prepareDecodeDraftModelInput(stream_groups, model_input);
                            }
                        }
                    } catch (const std::exception& e) {
                        reportInputPreparationFailure(streams_in_batch, ErrorCode::EXECUTION_EXCEPTION, e.what());
                        model_input.skip_run = true;
                    } catch (...) {
                        reportInputPreparationFailure(
                            streams_in_batch, ErrorCode::EXECUTION_EXCEPTION, "unknown decode input exception");
                        model_input.skip_run = true;
                    }
                }
            }
        }
        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            restoreStreamIterationStates(iteration_snapshots);
            cudaSyncAndCheck();
            if (!preserve_valid_input_holders) {
                clearDecodeTensorHolders(streams);
            }
            releaseModelBuffers();
            return warm_up_ ? absl::InternalError("decode input preparation failed during warm-up") :
                              absl::OkStatus();
        }
        executor_collector.tp_sync_input_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;

    // TODO(yinzhi): consider beam search & lora

    // release hold buffers before draft model forward
    releaseModelBuffers();

    if (propose_step_ > 1) {
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode start");
        draftModelDecode(model_input, stream_groups, draft_probs_list, draft_token_ids_t);
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode end");
        if (model_input.skip_run) {
            restoreStreamIterationStates(iteration_snapshots);
            cudaSyncAndCheck();
            clearDecodeTensorHolders(streams);
            releaseModelBuffers();
            return warm_up_ ? absl::InternalError("draft sampling failed during warm-up") : absl::OkStatus();
        }
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_model_verify)");
        maybePrintModelInput(model_input, "decode target model");
        model_input.is_target_verify        = true;
        model_input.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
        RTP_LLM_LOG_DEBUG(
            "[MTP decode] target model verify forward start, input_lengths_size=%ld, prefix_lengths_size=%ld, seq_lengths_size=%ld",
            model_input.input_lengths.size(0),
            model_input.prefix_lengths.size(0),
            model_input.sequence_lengths.size(0));
        model_output = std::move(model_->forward(model_input));
        RTP_LLM_LOG_DEBUG("[MTP decode] target model verify forward end");
        model_input.is_target_verify = false;
    }

    // trick: update draft sampler output after spec decode to avoid kernel launch overhead
    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(update_draft_sampler_output)");
        if (!model_input.is_fake_stream || warm_up_) {
            if (propose_step_ == 1) {
                batch_stream_processor_->updateOneStepDraftSamplerOutput(
                    stream_groups, draft_sampler_output, draft_token_probs_d_t);
            } else {
                batch_stream_processor_->updateMultiStepDraftSamplerOutput(stream_groups,
                                                                           draft_sampler_output,
                                                                           draft_token_ids_t,
                                                                           spec_token_ids_t,
                                                                           draft_token_probs_d_t,
                                                                           draft_probs_list);
            }
        }
    }

    // eplb
    if (expert_balancer_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(eplb_step_forward)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(rejection_sampling)");
        const auto                          streams_in_batch = batchStreams(stream_groups);
        try {
            if (model_input.is_fake_stream && !warm_up_) {
                speculative_sampler_output.accept_len.assign(batch_size, 1);
                speculative_sampler_output.accept_tokens.reserve(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    speculative_sampler_output.accept_tokens.push_back(torch::zeros({1}, torch::kInt32));
                }
                cudaSyncAndCheck();
            } else {
                // target model sample
                auto sampler_input_status =
                    batch_stream_processor_->gatherSpecSamplerInput(stream_groups, model_input, model_output);
                if (!sampler_input_status.ok()) {
                    reportSamplingException(streams_in_batch, sampler_input_status.status().ToString());
                    model_input.skip_run = true;
                } else {
                    generator_snapshots = captureSeededGeneratorStates(streams_in_batch);
                    sampler_output = std::move(sampler_->forward(sampler_input_status.value()));
                    model_input.skip_run =
                        reportSamplerFailures(sampler_output, streams_in_batch, SamplerRowLayout::SCORE);
                    if (!model_input.skip_run) {
                        sampler_output.all_probs = sampler_output.all_probs.reshape(
                            {(int64_t)batch_size, (int64_t)(propose_step_ + 1), (int64_t)vocab_size_});

                        // rejection sampling
                        speculative_sampler_output =
                            speculative_sampler_->forward(streams, draft_sampler_output, sampler_output);
                        if (sampled_token_input_vocab_guard_required_) {
                            model_input.skip_run = reportAcceptedTokenInputVocabFailures(speculative_sampler_output,
                                                                                         streams_in_batch,
                                                                                         target_input_vocab_size_,
                                                                                         propose_input_vocab_size_);
                        }
                    }
                }
            }
            if (!model_input.skip_run) {
                // NOTE: here will have cuda device sync before update model input
                batch_stream_processor_->updateDecodePostDraftModelInput(
                    model_input, model_output, speculative_sampler_output, batch_size, hidden_states_d_t, total_accept_len);
            } else {
                restoreSeededGeneratorStates(generator_snapshots);
            }
        } catch (const std::exception& e) {
            restoreSeededGeneratorStates(generator_snapshots);
            reportSamplingException(streams_in_batch, e.what());
            model_input.skip_run = true;
        } catch (...) {
            restoreSeededGeneratorStates(generator_snapshots);
            reportSamplingException(streams_in_batch, "unknown speculative sampling exception");
            model_input.skip_run = true;
        }
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(tp_sync_post_rejection)");
        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            restoreStreamIterationStates(iteration_snapshots);
            cudaSyncAndCheck();
            clearDecodeTensorHolders(streams);
            releaseModelBuffers();
            return warm_up_ ? absl::InternalError("target sampling failed during warm-up") : absl::OkStatus();
        }
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_draft_prefill_input)");
        maybePrintModelInput(model_input, "decode post draft model");
        const auto& mtp_cache_cfg           = cache_manager_->getMTPModuleCacheConfig(0);
        model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(draft_model_forward)");
        auto* draft_prefill_model = sp_prefill_draft_model_ ? sp_prefill_draft_model_.get() : draft_model_.get();
        draft_prefill_model_output = std::move(draft_prefill_model->forward(model_input));
    }

    if (!isTpRank0() || streams.empty() || (model_input.is_fake_stream && !warm_up_)) {
        cudaSyncAndCheck();
        releaseModelBuffers();
        return absl::OkStatus();
    }

    // draft model sample
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(draft_model_sample)");
        const auto streams_in_batch = batchStreams(stream_groups);
        try {
            fast_topk_sampler_output               = fast_topk_sampler_->forward(draft_prefill_model_output.logits);
            draft_prefill_sampler_output.all_probs = fast_topk_sampler_output.all_probs;
            draft_prefill_sampler_output.token_ids = fast_topk_sampler_output.token_ids;
            if (sampled_token_input_vocab_guard_required_) {
                model_input.skip_run = reportSampledTokenInputVocabFailures(draft_prefill_sampler_output.token_ids,
                                                                             streams_in_batch,
                                                                             SamplerRowLayout::NORMAL,
                                                                             target_input_vocab_size_,
                                                                             propose_input_vocab_size_,
                                                                             /*allow_int64=*/true);
            }
        } catch (const std::exception& e) {
            reportSamplingException(streams_in_batch, e.what());
            model_input.skip_run = true;
        } catch (...) {
            reportSamplingException(streams_in_batch, "unknown draft sampling exception");
            model_input.skip_run = true;
        }
        if (model_input.skip_run) {
            restoreSeededGeneratorStates(generator_snapshots);
            restoreStreamIterationStates(iteration_snapshots);
            cudaSyncAndCheck();
            clearDecodeTensorHolders(streams);
            releaseModelBuffers();
            return warm_up_ ? absl::InternalError("draft sampling failed during warm-up") : absl::OkStatus();
        }
    }

    if (warm_up_) {
        cudaSyncAndCheck();
        releaseModelBuffers();
        return absl::OkStatus();
    }

    // collect metrics
    if (metrics_reporter_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(collect_metrics)");
        executor_collector.generate_batch_size = stream_groups.totalModelBatchSize();
        executor_collector.execute_token_size += total_accept_len;
        executor_collector.max_seq_len = stream_groups.maxSeqLen();

        executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
        executor_collector.execute_token_size_when_has_context = executor_collector.execute_token_size;
        executor_collector.max_seq_len_when_has_context        = executor_collector.max_seq_len;

        tps_collector.generate_tps = total_accept_len;
        tps_collector.total_tps += total_accept_len;

        sp_engine_collector.total_accepted_token_num = total_accept_len;
        sp_engine_collector.total_stream_num         = stream_groups.size();
        sp_engine_collector.total_propose_token_num  = stream_groups.size() * propose_step_;
    }

    // dispatch
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dispatch_output)");
        auto result = batch_stream_processor_->dispatchDecode(
            stream_groups,
            speculative_sampler_output,
            {std::move(draft_prefill_model_output), std::move(draft_prefill_sampler_output)});
        // clean holder tensors from grpc
        for (auto& stream : streams) {
            stream->getSPOutputBuffer()->tensors_holder.clear();
        }

        releaseModelBuffers();

        return result;
    }
}

void MtpExecutor::prepareStreams(const std::list<GenerateStreamPtr>& streams,
                                 std::list<GenerateStreamPtr>&       prefill_streams,
                                 std::list<GenerateStreamPtr>&       decode_streams) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.prepare_streams(stream_size=%zu)", streams.size());

    for (auto& stream : streams) {
        // split streams into prefill and decode
        if (stream->isContextStream()) {
            prefill_streams.push_back(stream);
        } else {
            stream->setScoreLen(propose_step_ + 1);
            if (stream->getSPOutputBuffer() == nullptr && stream->isPerfTest()) {
                const auto execute_tokens = stream->currentExecuteTokens();
                RTP_LLM_CHECK_WITH_INFO(!execute_tokens.empty(),
                                        "fake speculative stream requires at least one token");
                auto sp_output_buffer = makeFakeSPOutputBuffer(
                    data_type_, hidden_size_, vocab_size_, propose_step_, execute_tokens.back());
                stream->setSPOutputBuffer(sp_output_buffer);
            }
            decode_streams.push_back(stream);
        }

        // set base properties
        stream->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
        if (stream->getSPOutputBuffer() == nullptr) {
            auto sp_output_buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
            sp_output_buffer->tokens = torch::zeros({1, 2}, torch::kInt32);

            stream->setSPOutputBuffer(sp_output_buffer);
        }

        // set propose_step
        auto sp_output_buffer          = stream->getSPOutputBuffer();
        sp_output_buffer->propose_step = propose_step_;
    }
}

absl::Status MtpExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.process(stream_size=%zu,mtp_step=%zu)", streams.size(), propose_step_);

    MtpMetricsCollector metrics_collector;

    std::list<GenerateStreamPtr> prefill_streams;
    std::list<GenerateStreamPtr> decode_streams;

    // prepare streams
    prepareStreams(streams, prefill_streams, decode_streams);

    // step forward
    int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

    if (role_type_ == RoleType::PREFILL || role_type_ == RoleType::PDFUSION) {
        THROW_IF_STATUS_ERROR(prefillStep(prefill_streams, metrics_collector));
    }

    if (role_type_ == RoleType::DECODE || role_type_ == RoleType::PDFUSION) {
        THROW_IF_STATUS_ERROR(decodeStep(decode_streams, metrics_collector));
    }

    metrics_collector.sp_engine_collector.step_latency_us =
        autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;

    // report metrics
    if (isTpRank0() && metrics_reporter_ && metrics_collector.not_skip) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.process(report_metrics)");
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(
            nullptr, &metrics_collector.executor_collector);
        metrics_reporter_->report<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
            nullptr, &metrics_collector.tps_collector);
        metrics_reporter_->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(
            nullptr, &metrics_collector.sp_engine_collector);
    }

    return absl::OkStatus();
}

bool MtpExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

void MtpExecutor::draftModelDecode(GptModelInputs&             model_input,
                                   const StreamGroups&         stream_groups,
                                   std::vector<torch::Tensor>& draft_probs_list,
                                   torch::Tensor&              draft_token_ids_t) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.draft_model_decode(batch_size=%zu)", model_input.combo_tokens.size(0));

    // clear host buffers holder
    buffer_holder_.release();

    const auto& mtp_cache_cfg         = cache_manager_->getMTPModuleCacheConfig(0);
    model_input.kv_block_stride_bytes = mtp_cache_cfg.kv_block_stride_bytes;
    model_input.kv_scale_stride_bytes = mtp_cache_cfg.kv_scale_stride_bytes;

    GptModelOutputs            draft_decode_model_output;
    std::vector<torch::Tensor> draft_token_ids_list;
    torch::Tensor              spec_prefix_lengths;

    // update TP > 0 batch_size
    size_t batch_size   = model_input.combo_tokens.size(0);
    spec_prefix_lengths = model_input.sequence_lengths.cpu().clone().pin_memory();

    auto pre_propose_token_t_raw = model_input.combo_tokens.to(torch::kCUDA).clone();

    auto pre_target_token = torch::empty({(int64_t)batch_size}, torch::kInt32);
    int  batch_idx        = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        int* propose_tokens                         = stream->getSPOutputBuffer()->tokens.data_ptr<int>();
        pre_target_token.data_ptr<int>()[batch_idx] = propose_tokens[0];
        batch_idx++;
    }

    auto pre_target_token_t         = pre_target_token.to(torch::kCUDA);
    auto pre_target_token_t_reshape = pre_target_token_t.reshape({(int)batch_size, 1});
    draft_token_ids_list.push_back(pre_target_token_t_reshape);

    auto pre_propose_token_t_reshape = pre_propose_token_t_raw.reshape({(int)batch_size, 1});
    draft_token_ids_list.push_back(pre_propose_token_t_reshape);

    // n-1 steps draft model decode
    for (int i = 0; i < propose_step_ - 1; i++) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.draft_model_decode(loop_iter=%d)", i);
        RTP_LLM_LOG_DEBUG("[MTP draftDecode] loop step %d/%d start, batch_size %zu", i, propose_step_ - 1, batch_size);
        draft_decode_model_output = std::move(draft_model_->forward(model_input));
        RTP_LLM_LOG_DEBUG("[MTP draftDecode] loop step %d forward done", i);

        // sample
        auto fast_topk_sampler_output = fast_topk_sampler_->forward(draft_decode_model_output.logits, 1);
        auto draft_probs              = fast_topk_sampler_output.all_probs;
        auto draft_probs_reshape      = draft_probs.reshape({(int)batch_size, 1, -1});
        auto draft_token_ids          = fast_topk_sampler_output.token_ids;

        if (model_input.is_fake_stream) {
            draft_token_ids.zero_();
            draft_decode_model_output.all_hidden_states.zero_();
        }

        draft_token_ids = draft_token_ids.to(torch::kInt32).to(torch::kCUDA);
        if (sampled_token_input_vocab_guard_required_) {
            if (isTpRank0()) {
                try {
                    model_input.skip_run = reportSampledTokenInputVocabFailures(draft_token_ids,
                                                                                 batchStreams(stream_groups),
                                                                                 SamplerRowLayout::NORMAL,
                                                                                 target_input_vocab_size_,
                                                                                 propose_input_vocab_size_);
                } catch (const std::exception& e) {
                    reportSamplingException(batchStreams(stream_groups), e.what());
                    model_input.skip_run = true;
                } catch (...) {
                    reportSamplingException(batchStreams(stream_groups), "unknown draft sampling exception");
                    model_input.skip_run = true;
                }
            }
            syncSkipRunAcrossTensorParallelRanks(model_input.skip_run, parallelism_config_);
            if (model_input.skip_run) {
                return;
            }
        }
        draft_token_ids_list.push_back(draft_token_ids);
        draft_probs_list.push_back(draft_probs_reshape);

        // update model input
        if (i != propose_step_ - 2) {
            batch_stream_processor_->updateDecodeDraftModelInput(
                model_input, draft_decode_model_output, draft_token_ids);
        }
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_decode_input)");
        // prepare spec decode input
        draft_token_ids_t =
            torch::cat(draft_token_ids_list, 1).reshape({(int)batch_size, (int)(propose_step_ + 1)}).contiguous();

        auto lm_output_indexes =
            torch::empty({(int64_t)(batch_size * (propose_step_ + 1))},
                         torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
        auto input_lengths = torch::empty({(int64_t)batch_size},
                                          torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));

        for (int i = 0; i < batch_size; i++) {
            input_lengths.data_ptr<int>()[i] = propose_step_ + 1;
        }
        for (int i = 0; i < batch_size * (propose_step_ + 1); i++) {
            lm_output_indexes.data_ptr<int>()[i] = i;
        }

        model_input.input_lengths     = std::move(input_lengths);
        model_input.lm_output_indexes = std::move(lm_output_indexes);
        model_input.prefix_lengths    = spec_prefix_lengths;
        model_input.combo_tokens      = draft_token_ids_t.reshape({(int64_t)(batch_size * (propose_step_ + 1))});
        model_input.sequence_lengths =
            torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
        model_input.last_hidden_states = torch::Tensor();

        // Since other tp ranks don't have streams, its combo_tokens' first token is not correct.
        // Thus, we need to broadcast the combo_tokens to other tp ranks.
        if (parallelism_config_.tp_size > 1) {
            execBroadcast({{model_input.combo_tokens}, 0});
        }

        const auto& cache_cfg             = cache_manager_->cacheConfig();
        model_input.kv_block_stride_bytes = cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes = cache_cfg.kv_scale_stride_bytes;
    }
}

}  // namespace rtp_llm
