#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <typeinfo>
#include <ATen/Generator.h>
#if defined(USING_CUDA) || defined(USING_ROCM)
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/cpp/utils/LinearBlocksUtil.h"

using namespace std;

namespace rtp_llm {

namespace {

std::optional<std::string> validateOutputVocabRequest(GenerateConfig& config, size_t output_vocab_size) {
    if (config.repetition_penalty != 1.0f || config.presence_penalty != 0.0f || config.frequency_penalty != 0.0f) {
        return "output vocabulary pruning does not support active repetition, presence, or frequency penalties";
    }
    if (config.no_repeat_ngram_size.value_or(0) > 0) {
        return "output vocabulary pruning does not support no_repeat_ngram_size";
    }
    if (config.in_think_mode) {
        return "output vocabulary pruning does not support think mode";
    }
    if (config.return_logits || config.return_prompt_logits || config.return_all_probs != ReturnAllProbsMode::NONE
        || config.calculate_loss != 0 || !config.select_tokens_id.empty() || !config.select_tokens_str.empty()) {
        return "output vocabulary pruning does not support full-vocabulary logits, probabilities, labels, or loss";
    }
    if (config.hasNumBeams() && output_vocab_size <= 2 * static_cast<size_t>(config.maxNumBeams())) {
        return "output vocabulary size must be greater than twice the maximum beam width";
    }
    return std::nullopt;
}

bool useStreamAsyncReserveTokens() {
    static const bool enabled = autil::EnvUtil::getEnv("RTP_LLM_STREAM_ASYNC", false);
    return enabled;
}

}  // namespace

GenerateStream::GenerateStream(const shared_ptr<GenerateInput>& input,
                               const ModelConfig&               model_config,
                               const RuntimeConfig&             runtime_config,
                               const ResourceContext&           resource_context,
                               kmonitor::MetricsReporterPtr     metrics_reporter,
                               size_t                           extra_reserve_token_num,
                               bool                             perf_test):
    generate_input_(input),
    max_seq_len_(model_config.max_seq_len),
    vocab_size_(model_config.vocab_size),
    output_vocab_size_(model_config.output_vocab_ids.size()),
    stream_cache_resource_(std::make_shared<StreamCacheResource>(
        this, resource_context, input->need_release_resource, input->generate_config->adapter_name)),
    need_release_resource_(input->need_release_resource),
    gen_timeline_(input->generate_config->gen_timeline),
    metrics_reporter_(metrics_reporter),
    special_tokens_(model_config.special_tokens),
    mutex_(std::make_shared<std::mutex>()),
    consumer_cv_(std::make_shared<std::condition_variable>()),
    mm_position_ids_style_(PositionIdsStyle(model_config.mm_model_config.mm_position_ids_style)),
    dtype_(model_config.data_type),
    hidden_size_(model_config.hidden_size) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!updatePrefix(resource_context.system_prompt)) {
        return;
    }

    // batch size depends on perf_test_, initialize it first
    perf_test_ = perf_test || autil::EnvUtil::getEnv("PERF_TEST", false);
    if (perf_test_ && hasNumBeams()) {
        // TODO(zhangjianning.zjn): support perf test for beam search
        RTP_LLM_LOG_WARNING("beam search does not support PERF_TEST for now");
    }

    // Note it is invalid to use currentBatchSize here, because currentBatchSize depends on complete_token_ids_,
    // which has not been initialized yet
    const size_t init_batch_size = batchSize(0);

    begin_time_us_ = input->begin_time_us;
    if (generate_input_->generate_config->calculate_loss && inputLength() > 1) {
        loss_ = torch::zeros({(int64_t)inputLength() - 1}, torch::kFloat32);
    }
    if (generate_input_->generate_config->return_all_hidden_states) {
        setReturnLastHiddenStates(true);
    }
    complete_token_ids_ = std::make_shared<CompleteTokenIds>(
        init_batch_size, maxBatchSize(), max_seq_len_, model_config.attn_config.tokens_per_block);
    complete_token_ids_->init(input, extra_reserve_token_num);

    last_output_pos_ = seqLength();

    cum_log_probs_ = torch::zeros({(int64_t)init_batch_size}, torch::kFloat32);

    is_context_stream_  = std::make_shared<bool>();
    *is_context_stream_ = true;
    generate_status_    = std::make_shared<GenerateStateMachine>(stream_cache_resource_);
    sub_generate_status_.reserve(maxBatchSize());
    sub_generate_status_.clear();
    resizeSubGenerateStatus(init_batch_size);

    stream_cache_resource_->init(init_batch_size);

    setReturnAllProbs(generate_input_->generate_config->return_all_probs);

    int64_t processor_eos_token_id = special_tokens_.eos_token_id;
    if (output_vocab_size_ > 0) {
        const auto& output_vocab_ids = model_config.output_vocab_ids;
        const auto  eos_it =
            std::lower_bound(output_vocab_ids.begin(), output_vocab_ids.end(), special_tokens_.eos_token_id);
        if (eos_it == output_vocab_ids.end() || *eos_it != special_tokens_.eos_token_id) {
            reportError(ErrorCode::INVALID_PARAMS, "primary EOS token is absent from the configured output vocabulary");
            return;
        }
        processor_eos_token_id = std::distance(output_vocab_ids.begin(), eos_it);

        if (auto error = validateOutputVocabRequest(*generateConfig(), output_vocab_size_)) {
            reportError(ErrorCode::INVALID_PARAMS, *error);
            return;
        }
    }

    auto processors_result = LogitsProcessorFactory::createLogitsProcessors(
        generate_input_, init_batch_size, maxBatchSize(), processor_eos_token_id);
    if (processors_result.ok()) {
        auto processors = std::move(processors_result.value());
        if (output_vocab_size_ > 0) {
            for (const auto& processor : processors) {
                if (typeid(*processor) != typeid(MultiSeqLogitsProcessor)) {
                    reportError(ErrorCode::INVALID_PARAMS,
                                "output vocabulary pruning only supports the MultiSeq logits processor");
                    return;
                }
            }
        }
        logits_processor_list_ = std::move(processors);
    } else {
        const auto& err = processors_result.status();
        reportEventWithoutLock(StreamEvents::Error, err.code(), err.ToString());
    }

    if (calculateSoftmaxProbs()) {
        softmax_probs_ = torch::zeros({(int64_t)maxBatchSize(), (int64_t)maxTokenNum()}, torch::kFloat32);
    }

    if (generateConfig()->random_seed.has_value()) {
#if defined(USING_CUDA) || defined(USING_ROCM)
        generator_ = torch::make_generator<torch::CUDAGeneratorImpl>();
#else
        generator_ = torch::make_generator<torch::CPUGeneratorImpl>();
#endif
        generator_.set_current_seed(generateConfig()->random_seed.value());
    }
}

void GenerateStream::resetBeginTime(int64_t begin_time_us) {
    std::lock_guard<std::mutex> lock(*mutex_);
    begin_time_us_               = begin_time_us;
    wait_time_us_                = 0;
    scheduler_enqueue_time_us_   = 0;
    can_run_time_us_             = 0;
    loading_cache_start_time_us_ = 0;
    loading_cache_done_time_us_  = 0;
    first_running_time_us_       = 0;
    loading_cache_latency_us_    = 0;
    load_done_to_running_us_     = 0;
    if (running_started_) {
        running_started_time_us_ = begin_time_us;
    }
}

bool GenerateStream::hasCacheKeys() const {
    return stream_cache_resource_->hasCacheKeys();
}

const CacheKeysType& GenerateStream::cacheKeys(int32_t batch_id) const {
    return stream_cache_resource_->cacheKeys(batch_id);
}

absl::Status GenerateStream::initKVBlock() {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(*mutex_);
    if (generate_status_->status == StreamState::WAITING) {
        recordWaitLatency();
    }
    auto ret = stream_cache_resource_->initKVBlock();
    if (!ret.ok()) {
        RTP_LLM_LOG_WARNING("GenerateStream::initKVBlock: initKVBlock failed, stream_id: %lld", streamId());
    }
    return ret;
}

void GenerateStream::fakeInitKVBlock(size_t reserved_blocks) {
    std::lock_guard<std::mutex> lock(*mutex_);
    stream_cache_resource_->fakeInitKVBlock(reserved_blocks);
}

absl::Status GenerateStream::incrKVBlock() {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(*mutex_);
    return stream_cache_resource_->incrKVBlock();
}

void GenerateStream::releaseResource() {
    RTP_LLM_PROFILE_FUNCTION();
    // Return KV blocks only after all workers that captured this stream finish.
    // Earlier release could let a worker write into blocks owned by another stream.
    waitPendingAsyncBookkeeping();
    std::lock_guard<std::mutex> lock(*mutex_);
    if (!stream_cache_resource_->isResourceReleased()) {
        stream_cache_resource_->releaseResource();
    }
}

void GenerateStream::incPendingAsyncBookkeeping() {
    async_bookkeeping_->count.fetch_add(1, std::memory_order_acq_rel);
}

void GenerateStream::decPendingAsyncBookkeepingAndMaybeRelease() {
    int prev = async_bookkeeping_->count.fetch_sub(1, std::memory_order_acq_rel);
    RTP_LLM_CHECK(prev >= 1);
    if (prev == 1) {
        {
            std::lock_guard<std::mutex> lk(async_bookkeeping_->mu);
        }
        async_bookkeeping_->cv.notify_all();
        // The last worker performs any deferred release after its update lock
        // has unwound, so releaseResource() can safely re-enter mutex_.
        if (async_bookkeeping_->defer_release.exchange(false, std::memory_order_acq_rel)) {
            releaseResource();
        }
    }
}

bool GenerateStream::hasPendingAsyncBookkeeping() const {
    return async_bookkeeping_->count.load(std::memory_order_acquire) > 0;
}

void GenerateStream::waitPendingAsyncBookkeeping() {
    std::unique_lock<std::mutex> lk(async_bookkeeping_->mu);
    async_bookkeeping_->cv.wait(lk, [this] { return async_bookkeeping_->count.load(std::memory_order_acquire) == 0; });
}

void GenerateStream::markDeferredRelease() {
    async_bookkeeping_->defer_release.store(true, std::memory_order_release);
}

bool GenerateStream::isDeferredReleasePending() const {
    return async_bookkeeping_->defer_release.load(std::memory_order_acquire);
}
void GenerateStream::setNeedReleaseResource(bool need_release_resource) {
    need_release_resource_ = need_release_resource;
    stream_cache_resource_->setNeedReleaseResource(need_release_resource);
}
int GenerateStream::nextNeedBlockNums(int reserve_step) const {
    // TODO: maybe need fix when context and reuse
    return stream_cache_resource_->singleBatchNeedBlocks(seqLength(), reserve_step) * nextBatchSize();
}

int GenerateStream::estimateKVNeedBlocks(int remaining_tokens, int target_batch_size) const {
    const int reserve_step   = complete_token_ids_->getReserveStep();
    int       common_seq_len = std::min(complete_token_ids_->commonSeqLength(), seqLength());
    if (target_batch_size > 1) {
        common_seq_len = common_seq_len / seqSizePerBlock() * seqSizePerBlock();
    }
    return stream_cache_resource_->estimatePeakNeedBlocks(
        seqLength(), common_seq_len, remaining_tokens, reserve_step, target_batch_size);
}

int GenerateStream::estimateInitialNeedBlocks() const {
    return estimateKVNeedBlocks(/*remaining_tokens=*/0, currentBatchSize());
}

int GenerateStream::estimatePeakNeedBlocks(int remaining_tokens) const {
    return estimateKVNeedBlocks(remaining_tokens, maxBatchSize());
}

std::shared_ptr<GenerateInput> GenerateStream::generateInput() const {
    return generate_input_;
}
std::shared_ptr<GenerateConfig>& GenerateStream::generateConfig() const {
    return generate_input_->generate_config;
}
bool GenerateStream::isStreaming() const {
    if (generate_input_->generate_config->hasNumBeams()) {
        return false;
    }
    return generate_input_->generate_config->is_streaming;
}

int64_t GenerateStream::streamId() const {
    return generate_input_->request_id;
}

std::string GenerateStream::streamLogTag() const {
    const auto& request_info = generate_input_->request_info;
    std::string tag          = std::string("request_id=") + std::to_string(streamId()) + " trace_id=" + traceId();
    if (!request_info.request_id.empty()) {
        tag += " source_request_id=" + request_info.request_id;
    }
    if (!request_info.frontend_ip.empty()) {
        tag += " frontend_ip=" + request_info.frontend_ip;
    }
    if (!request_info.dash_ip.empty()) {
        tag += " dash_ip=" + request_info.dash_ip;
    }
    if (!request_info.source_role.empty()) {
        tag += " source_role=" + request_info.source_role;
    }
    return tag;
}

std::string GenerateStream::adapterName() const {
    return generate_input_->generate_config->adapter_name;
}

rtp_llm::SpecialTokens GenerateStream::specialTokens() const {
    return special_tokens_;
}

int GenerateStream::batchSize(int output_len) const {
    if (generate_input_->generate_config->hasNumBeams()) {
        return numBeams(output_len);
    } else {
        return std::max(numReturnSequences(), 1);
    }
}

int GenerateStream::currentBatchSize() const {
    return batchSize(outputTokenLen());
}

int GenerateStream::nextBatchSize() const {
    return batchSize(outputTokenLen() + 1);
}

int GenerateStream::maxBatchSize() const {
    if (generate_input_->generate_config->hasNumBeams()) {
        return maxNumBeams();
    } else {
        return std::max(numReturnSequences(), 1);
    };
}

int GenerateStream::numBeams(int output_length) const {
    if (output_length == 0) {
        // num_beams should be 1 for the first step
        return 1;
    }

    int var_steps = generate_input_->generate_config->variable_num_beams.size();
    if (var_steps == 0) {
        return generate_input_->generate_config->num_beams;
    } else {
        int idx = output_length < var_steps ? output_length - 1 : var_steps - 1;
        return generate_input_->generate_config->variable_num_beams[idx];
    }
}

int GenerateStream::currentNumBeams() const {
    return numBeams(outputTokenLen());
}

int GenerateStream::nextNumBeams() const {
    return numBeams(outputTokenLen() + 1);
}

int GenerateStream::maxNumBeams() const {
    return generate_input_->generate_config->maxNumBeams();
}

bool GenerateStream::hasNumBeams() const {
    return generate_input_->generate_config->hasNumBeams();
}

bool GenerateStream::usesBeamSearchTokenLayoutForCurrentStep() const {
    return currentNumBeams() > 1 || nextNumBeams() > 1;
}

bool GenerateStream::needTilingForSampling() const {
    return isContextStream() && currentBatchSize() != nextBatchSize() && !hasNumBeams();
}

int GenerateStream::numReturnSequences() const {
    return generate_input_->generate_config->num_return_sequences;
}

bool GenerateStream::calculateLoss() const {
    return loss_.defined() && loss_index_ < inputLength() - 1;
}

bool GenerateStream::calculateSoftmaxProbs() const {
    return generate_input_->generate_config->return_softmax_probs;
}

bool GenerateStream::returnLogits() const {
    return generate_input_->generate_config->return_logits;
}

bool GenerateStream::returnPromptLogits() const {
    return generate_input_->generate_config->return_prompt_logits;
}

bool GenerateStream::returnCumLogProbs() const {
    return generate_input_->generate_config->return_cum_log_probs;
}

bool GenerateStream::genTimeline() const {
    return seqLength() <= inputLength() + profileStep() - 1 ? gen_timeline_ : false;
}

void GenerateStream::setGenTimeline(bool gen_timeline) {
    gen_timeline_ = gen_timeline;
}

int GenerateStream::profileStep() const {
    return generate_input_->generate_config->profile_step;
}

bool GenerateStream::updatePrefix(const std::shared_ptr<SystemPrompt>& system_prompt) {
    if (system_prompt) {
        auto prefix_param = system_prompt->getPromptParams(*generate_input_->generate_config);
        if (!prefix_param.prompt_tokens.empty()) {
            auto total_input_len = inputLength() + prefix_param.prompt_tokens.size();
            if (total_input_len >= max_seq_len_) {
                reportEvent(StreamEvents::Error,
                            ErrorCode::LONG_PROMPT_ERROR,
                            "after update prefix, total input len " + std::to_string(total_input_len)
                                + " is greater than max seq len " + std::to_string(max_seq_len_));
                return false;
            }
            generate_input_->updatePrefix(prefix_param.prompt_tokens);
        }
    }
    return true;
}

size_t GenerateStream::maxSeqLen() const {
    return max_seq_len_;
}

int GenerateStream::inputLength() const {
    return generate_input_->inputLength();
}

int GenerateStream::seqLength() const {
    return complete_token_ids_->seqLength();
}

int GenerateStream::seqSizePerBlock() const {
    return stream_cache_resource_->seqSizePerBlock();
}

int GenerateStream::contextLength() const {
    int begin_pos = prefixLength();
    int end_pos   = seqLength();
    return end_pos - begin_pos;
}

int GenerateStream::prefixLength() const {
    return reuse_length_;
}

int GenerateStream::reuseLength() const {
    return reuse_length_;
}

int GenerateStream::initialReuseLength() const {
    return initial_reuse_length_;
}

void GenerateStream::setReuseLength(int reuse_length) {
    reuse_length_ = reuse_length;
}

void GenerateStream::setLocalReuseLength(int length) {
    local_reuse_length_ = length;
    setDeviceReuseLength(std::max(local_reuse_length_ - host_reuse_length_ - disk_reuse_length_, 0));
}

void GenerateStream::setRemoteReuseLength(int length) {
    remote_reuse_length_ = length;
}

int GenerateStream::localReuseLength() const {
    return local_reuse_length_;
}

void GenerateStream::setDeviceReuseLength(int length) {
    device_reuse_length_ = length;
}

int GenerateStream::deviceReuseLength() const {
    return device_reuse_length_;
}

int GenerateStream::remoteReuseLength() const {
    return remote_reuse_length_;
}

void GenerateStream::setHostReuseLength(int length) {
    host_reuse_length_ = length;
    setDeviceReuseLength(std::max(local_reuse_length_ - host_reuse_length_ - disk_reuse_length_, 0));
}

int GenerateStream::hostReuseLength() const {
    return host_reuse_length_;
}

void GenerateStream::setDiskReuseLength(int length) {
    disk_reuse_length_ = length;
    setDeviceReuseLength(std::max(local_reuse_length_ - host_reuse_length_ - disk_reuse_length_, 0));
}

int GenerateStream::diskReuseLength() const {
    return disk_reuse_length_;
}

void GenerateStream::setInitialReuseLength(int initial_reuse_length) {
    // NOTE: no upper-bound check against seqLength() here. Under CP sharding the
    // connector reuse lengths are accounted in canonical block tokens
    // (seq_size_per_block * cp_size), which can legitimately exceed the
    // rank-local sequence length.
    initial_reuse_length_ = initial_reuse_length;
}

void GenerateStream::setPrefillReuseLength(int64_t total, int64_t local, int64_t remote, int64_t memory) {
    prefill_total_reuse_len_  = total;
    prefill_local_reuse_len_  = local;
    prefill_remote_reuse_len_ = remote;
    prefill_memory_reuse_len_ = memory;
}

int64_t GenerateStream::prefillTotalReuseLen() const {
    return prefill_total_reuse_len_;
}

int64_t GenerateStream::prefillLocalReuseLen() const {
    return prefill_local_reuse_len_;
}

int64_t GenerateStream::prefillRemoteReuseLen() const {
    return prefill_remote_reuse_len_;
}

int64_t GenerateStream::prefillMemoryReuseLen() const {
    return prefill_memory_reuse_len_;
}

void GenerateStream::incLastOutputPos() {
    last_output_pos_++;
}

bool GenerateStream::isContextStream() const {
    return *is_context_stream_;
}

const torch::Tensor& GenerateStream::cumLogProbs() const {
    return cum_log_probs_;
}

torch::Tensor GenerateStream::completeTokenIds() {
    return complete_token_ids_->completeTokenIds();
}

std::vector<int> GenerateStream::completeTokenIdsVec(int batch_idx) {
    RTP_LLM_CHECK(batch_idx < currentBatchSize());
    return complete_token_ids_->completeTokenIdsVec(batch_idx);
}

int GenerateStream::currentExecuteTokenSize() {
    return currentExecuteTokens(0).size() * currentBatchSize();
}

std::vector<torch::Tensor> GenerateStream::multimodalFeatures() const {
    if (generate_input_->multimodal_features) {
        return generate_input_->multimodal_features.value();
    } else {
        return std::vector<torch::Tensor>();
    }
}

std::vector<torch::Tensor> GenerateStream::multimodalExtraInput() const {
    if (generate_input_->mm_extra_input) {
        return generate_input_->mm_extra_input.value();
    }
    return std::vector<torch::Tensor>();
}

bool GenerateStream::hasMultimodalExtraInput() const {
    if (generate_input_->mm_extra_input) {
        return generate_input_->mm_extra_input.value().size() > 0;
    }
    return false;
}

int GenerateStream::multimodalFeaturesLength() const {
    if (generate_input_->multimodal_features) {
        return generate_input_->multimodal_features.value().size() * currentBatchSize();
    } else {
        return 0;
    }
}

torch::Tensor GenerateStream::multimodalLocations() const {
    if (!generate_input_->mm_locs) {
        return torch::Tensor();
    }
    return generate_input_->mm_locs.value();
}

vector<int> GenerateStream::textTokensMask() const {
    if (!generate_input_->text_tokens_mask) {
        return {};
    }
    auto& mask  = generate_input_->text_tokens_mask.value();
    auto* data  = mask.data_ptr<int>();
    int   start = reuseLength() > 0 ? reuseLength() : 0;
    return vector<int>(data + start, data + mask.numel());
}

torch::Tensor GenerateStream::generateContextPositionIds() {
    context_position_ids_ = PositionIdsGenerator::generatePositionIds(generate_input_->inputLength(),
                                                                      mm_position_ids_style_,
                                                                      generate_input_->mm_locs,
                                                                      generate_input_->mm_position_ids);
    return context_position_ids_.value();
}

void GenerateStream::generateNextPositionId(int32_t* now_pos) {
    if (!context_position_ids_) {
        return;
    }
    PositionIdsGenerator::generateNextPositionId(
        now_pos, seqLength(), mm_position_ids_style_, context_position_ids_.value());
}

vector<int> GenerateStream::currentExecuteTokens(int batch_idx) const {
    // TODO(xinfei.sxf) 在query部分回退，重运行case下，这个不对
    if (isContextStream()) {
        return complete_token_ids_->contextTokens(batch_idx, prefixLength(), contextLength());
    } else {
        return complete_token_ids_->currentExecuteTokens(batch_idx);
    }
}

void GenerateStream::step() {
    // iter_count represents the times of the stream participates in running
    iter_count_++;
}

void GenerateStream::spStep() {
    sp_iter_count_++;
}

int64_t GenerateStream::getTimeoutMs() const {
    return generate_input_->generate_config->timeout_ms;
}

void GenerateStream::checkTimeoutWithoutLock() {
    // Consumer-visible events and timeout publication share mutex_. Once a
    // consumer-ready event has linearized, a later scheduler timeout must not
    // replace it with the higher-precedence timeout error.
    if (consumerReadyWithoutLock()) {
        return;
    }

    auto running_time_ms = (autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_) / 1000;
    auto timeout_ms      = getTimeoutMs();
    if (timeout_ms > 0 && timeout_ms < running_time_ms) {
        reportTimeoutWithoutLock(running_time_ms, timeout_ms);
    }
}

void GenerateStream::recordWaitLatency() {
    wait_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
}

void GenerateStream::recordSchedulerEnqueueTime(int64_t time_us) {
    if (scheduler_enqueue_time_us_ == 0) {
        scheduler_enqueue_time_us_ = time_us;
    }
}

void GenerateStream::recordCanRunTime() {
    if (can_run_time_us_ == 0) {
        can_run_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
    }
}

void GenerateStream::recordLoadingCacheStartTime() {
    if (loading_cache_start_time_us_ == 0) {
        loading_cache_start_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
    }
}

void GenerateStream::recordLoadingCacheDoneTime() {
    if (loading_cache_done_time_us_ == 0) {
        loading_cache_done_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
        if (loading_cache_start_time_us_ > 0) {
            loading_cache_latency_us_ = loading_cache_done_time_us_ - loading_cache_start_time_us_;
        }
    }
}

void GenerateStream::recordRunningTime() {
    if (first_running_time_us_ != 0) {
        return;
    }
    first_running_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
    if (loading_cache_done_time_us_ > 0) {
        load_done_to_running_us_ = first_running_time_us_ - loading_cache_done_time_us_;
    }
}

void GenerateStream::reportTimeoutWithoutLock(int64_t running_time_ms, int64_t timeout_ms) {
    reportEventWithoutLock(StreamEvents::Error,
                           ErrorCode::GENERATE_TIMEOUT,
                           "query has been running " + std::to_string(running_time_ms) + " ms, "
                               + "timeout_ms = " + std::to_string(timeout_ms) + ", it's timeout");
}

bool GenerateStream::reportUpdateErrorWithoutLock(const std::optional<ErrorInfo>& error_info) {
    if (!error_info.has_value() || !error_info->hasError()) {
        return false;
    }
    const auto& error = error_info.value();
    reportEventWithoutLock(StreamEvents::Error, error.code(), error.ToString());
    return true;
}

bool GenerateStream::hasEvent(StreamEvents::EventType event) const {
    std::lock_guard<std::mutex> lock(*mutex_);
    return hasEventWithoutLock(event);
}

bool GenerateStream::hasEventWithoutLock(StreamEvents::EventType event) const {
    return generate_status_->hasEvent(event);
}

void GenerateStream::clearCanRun() {
    std::lock_guard<std::mutex> lock(*mutex_);
    generate_status_->clearCanRun();
}

StreamState GenerateStream::getStatus() const {
    return generate_status_->getStatus();
}

bool GenerateStream::isFinished() const {
    return getStatus() == StreamState::FINISHED;
}

bool GenerateStream::isActive() const {
    std::lock_guard<std::mutex> lock(*mutex_);
    return !hasErrorWithoutLock() && getStatus() != StreamState::FINISHED;
}

void GenerateStream::setReserveStep(size_t reserve_step) {
    // Keep GenerateStream as the only entry point for setting reserve_step.
    reserve_step_ = reserve_step;
    generate_status_->setReserveStep(reserve_step);
    complete_token_ids_->setReserveStep(static_cast<int>(reserve_step));
}

StreamState GenerateStream::moveToNext() {
    StreamState state;
    bool        should_report_metric = false;
    {
        std::lock_guard<std::mutex> lock(*mutex_);
        checkTimeoutWithoutLock();
        const auto old_status = getStatus();
        state                 = generate_status_->moveToNext();
        const auto new_status = getStatus();

        if ((old_status == StreamState::WAITING && new_status != StreamState::WAITING)
            || (old_status != StreamState::RUNNING && new_status == StreamState::RUNNING && !running_started_)) {
            const auto transition_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            if (old_status == StreamState::WAITING && new_status != StreamState::WAITING) {
                wait_time_us_ = transition_time_us - begin_time_us_;
            }
            if (old_status != StreamState::RUNNING && new_status == StreamState::RUNNING && !running_started_) {
                running_started_         = true;
                running_started_time_us_ = transition_time_us;
            }
        }
        should_report_metric = old_status != StreamState::FINISHED && new_status == StreamState::FINISHED;

        if (new_status != old_status) {
            consumer_cv_->notify_all();
        }
    }
    // Report terminal stream metrics outside the stream mutex; the reporter
    // may take its own locks and must not nest under mutex_.
    if (should_report_metric) {
        reportMetricOnce();
    }
    return state;
}

bool GenerateStream::hasError() const {
    std::lock_guard<std::mutex> lock(*mutex_);
    return hasErrorWithoutLock();
}

bool GenerateStream::hasErrorWithoutLock() const {
    return generate_status_->error_info.hasError();
}

bool GenerateStream::isSubGenerateDoneWithoutLock(int batch_id) const {
    return getStatus() == StreamState::FINISHED || sub_generate_status_[batch_id] == StreamState::FINISHED;
}

ErrorInfo GenerateStream::statusInfo() {
    std::lock_guard<std::mutex> lock(*mutex_);
    return statusInfoWithoutLock();
}

ErrorInfo GenerateStream::statusInfoWithoutLock() const {
    return generate_status_->error_info;
}

bool GenerateStream::consumerFinishedWithoutLock() const {
    // Consumer completion includes pending GenerateDone, but lifecycle commit and
    // resource release remain scheduler-owned through moveToNext().
    return hasEventWithoutLock(StreamEvents::NeedRemoteGenerate) || generate_status_->checkFinished();
}

bool GenerateStream::consumerReadyWithoutLock() const {
    return hasErrorWithoutLock() || consumerFinishedWithoutLock();
}

std::string GenerateStream::stopReason() {
    std::lock_guard<std::mutex> lock(*mutex_);
    return statusInfoWithoutLock().ToString();
}

size_t GenerateStream::iterCount() const {
    return iter_count_;
}

size_t GenerateStream::spIterCount() const {
    return sp_iter_count_;
}

void GenerateStream::setSpIterCount(int sp_iter_count) {
    sp_iter_count_ = sp_iter_count;
}

void GenerateStream::setKVCache(const BatchKVCacheResource& kv_cache_resource) {
    stream_cache_resource_->setKVCache(kv_cache_resource);
}

const BatchKVCacheResource& GenerateStream::kvCache() const {
    return stream_cache_resource_->kvCache();
}

BatchKVCacheResource& GenerateStream::kvCacheMutable() {
    return stream_cache_resource_->kvCacheMutable();
}

BatchKVCacheResourcePtr GenerateStream::kvCachePtr() {
    // TODO: set deleter if use BatchKVCacheResource to manager life cycles of kv cache automatically
    return std::shared_ptr<BatchKVCacheResource>(&stream_cache_resource_->kvCacheMutable(),
                                                 [](BatchKVCacheResource*) {});
}

const ResourceContext& GenerateStream::resourceContext() const {
    return stream_cache_resource_->resourceContext();
}

size_t GenerateStream::curBlocksNum() const {
    return stream_cache_resource_->curBlocksNum();
}

size_t GenerateStream::maxTokenNum() const {
    int reserve_tokens = 0;
    if (sp_output_buffer_) {
        reserve_tokens = sp_output_buffer_->propose_step;
        if (useStreamAsyncReserveTokens()) {
            reserve_tokens = reserve_tokens * 2 + 1;
        }
    }

    return std::min(max_seq_len_ > reserve_tokens ? max_seq_len_ - reserve_tokens : 0,
                    generate_input_->generate_config->max_new_tokens + generate_input_->inputLength());
}

bool GenerateStream::needFinish() {
    return seqLength() >= maxTokenNum() || needFinishBySPTokens();
}

bool GenerateStream::needFinishBySPTokens() {
    if (hasNumBeams()) {
        // update sub_generate_status to RUNNING for beam search,
        // as the same batch_id may refers to different beams between steps
        fillSubGenerateStatus(StreamState::RUNNING);
    }

    if (seqLength() >= generate_input_->generate_config->min_new_tokens + inputLength()) {
        matchEosToken();
        matchStopWordsList();
    }

    // check if all batch finished
    return std::all_of(sub_generate_status_.begin(), sub_generate_status_.end(), [](StreamState state) {
        return state == StreamState::FINISHED;
    });
}

void GenerateStream::matchEosToken() {
    for (int i = 0; i < currentBatchSize(); ++i) {
        matchEosToken(i);
    }
}

void GenerateStream::matchEosToken(int batch_id) {
    if ((!generate_input_->generate_config->ignore_eos)
        && complete_token_ids_->matchEosToken(batch_id, special_tokens_.eos_token_id)) {
        sub_generate_status_[batch_id] = StreamState::FINISHED;
    }
}

std::vector<int> GenerateStream::getLatestTokens(size_t token_num) {
    return complete_token_ids_->getLatestTokens(token_num);
}

void GenerateStream::matchStopWordsList() {
    if (seqLength() == inputLength()) {
        return;
    }
    for (int i = 0; i < currentBatchSize(); ++i) {
        matchStopWordsList(i);
    }
}

void GenerateStream::matchStopWordsList(int batch_id) {
    RTP_LLM_PROFILE_FUNCTION();
    // note: stop_words_list in generate_config contains stop_words_list in special_tokens
    bool match = false;
    for (auto& stop_words : generate_input_->generate_config->stop_words_list) {
        if (generate_input_->generate_config->ignore_eos && stop_words.size() == 1
            && stop_words[0] == special_tokens_.eos_token_id) {
            continue;
        }
        if (complete_token_ids_->matchStopWordsList(batch_id, stop_words)) {
            match = true;
            break;
        }
    }
    if (match) {
        sub_generate_status_[batch_id] = StreamState::FINISHED;
    }
}

void GenerateStream::specUpdate(const StreamSpecUpdateInfo& update_info) {
    // Worker-thread MTP bookkeeping updates tokens/output and finish checks
    // before the next async dispatch. The speculative propose_step+1 window
    // already covers stop/EOS/max-token boundaries.
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(*mutex_);
    RTP_LLM_LOG_DEBUG("stream [%s] spec update", streamLogTag().c_str());
    *is_context_stream_ = false;
    if (reportUpdateErrorWithoutLock(update_info.error_info)) {
        return;
    }
    if ((hasErrorWithoutLock() || isFinished()) && !update_info.force_update_info) {
        return;
    }
    // Ignore stale worker updates after finish; committing them would duplicate
    // tokens and touch KV blocks only deferred until this worker exits.
    if (isFinished() && !update_info.force_update_info) {
        return;
    }

    const auto& new_tokens = update_info.new_tokens;

    if (isPerfTest()) {
        const_cast<torch::Tensor&>(new_tokens).zero_();
    }

    auto num_new_tokens = update_info.num_new_tokens;
    int  cur_cached_len = seqLength() - 1;

    int error_token_id = 0;
    if (!complete_token_ids_->update(new_tokens,
                                     begin_time_us_,
                                     num_new_tokens,
                                     generate_input_->inputLength(),
                                     maxTokenNum(),
                                     vocab_size_,
                                     hasNumBeams(),
                                     streamId(),
                                     error_token_id)) {
        reportEventWithoutLock(StreamEvents::Error,
                               ErrorCode::OUT_OF_VOCAB_RANGE,
                               "output token id:" + std::to_string(error_token_id)
                                   + " out of vocab size: " + std::to_string(vocab_size_));
        return;
    }

    int nxt_cached_len   = seqLength() - 1;
    int accept_token_num = nxt_cached_len - cur_cached_len;

    // The stream token history is authoritative. Advance every processor exactly
    // once before mutating speculative buffers, cache layout or published output.
    if (auto error = updateLogitProcessorStatus(new_tokens, accept_token_num); error.has_value()) {
        reportEventWithoutLock(StreamEvents::Error, error->code(), error->ToString());
        return;
    }

    if (update_info.speculative_propose_step > 0) {
        sp_iter_count_++;
        speculative_accepted_tokens_per_pos_.resize(update_info.speculative_propose_step, 0);
        const int accepted_draft_tokens =
            std::clamp(update_info.accepted_draft_tokens, 0, update_info.speculative_propose_step);
        for (int position = 0; position < accepted_draft_tokens; ++position) {
            speculative_accepted_tokens_per_pos_[position]++;
        }
    }

    // update speculative output buffer
    int  target_last_token = new_tokens.data_ptr<int>()[num_new_tokens - 1];
    int* spec_tokens       = sp_output_buffer_->tokens.data_ptr<int>();
    spec_tokens[0]         = target_last_token;
    if (update_info.draft_token >= 0) {
        RTP_LLM_CHECK_WITH_INFO(sp_output_buffer_->tokens.numel() >= 2,
                                "speculative token buffer must contain target and draft slots");
        spec_tokens[1] = update_info.draft_token;
        propose_token_ = {target_last_token, update_info.draft_token};
    } else {
        // The legacy CPU side channel has no proposal. This covers both
        // commit-only steps and GPU-only fast paths, so GPU proposal lifetime
        // is controlled independently by draft_token_gpu below.
        propose_token_.clear();
    }

    sp_output_buffer_->hidden_states = update_info.draft_hidden_states;
    sp_output_buffer_->all_probs     = update_info.draft_token_probs;
    // Apply the explicit tri-state proposal update. PD partial updates use
    // nullopt to preserve the handoff proposal; GPU/CPU proposal producers use
    // a defined/undefined Tensor respectively to replace or clear the mirror.
    if (update_info.draft_token_gpu.has_value()) {
        sp_output_buffer_->propose_tokens_gpu = *update_info.draft_token_gpu;
    }

    // for spec-decode linear attention, we need to adjust cache blocks
    if (accept_token_num > 1 && stream_cache_resource_) {
        int seq_size_per_block = seqSizePerBlock();

        // 1. swap cache blocks of accept tokens to corresponding blocks
        auto [cached_src_block_idx, cached_des_block_idx] =
            getCachedTokenBlockSwapIdx(cur_cached_len, nxt_cached_len, seq_size_per_block);
        stream_cache_resource_->swapLinearBlocks(0, cached_src_block_idx, cached_des_block_idx);

        // 2. swap final block of accept tokens to the next sequence block
        auto [src_block_idx, des_block_idx] =
            getFinalTokenBlockSwapIdx(cur_cached_len, nxt_cached_len, seq_size_per_block);
        stream_cache_resource_->swapLinearBlocks(0, src_block_idx, des_block_idx);

        RTP_LLM_LOG_DEBUG("[stream %s (%d -> %d)] swap cache blocks: %d -> %d, %d -> %d",
                          streamLogTag().c_str(),
                          cur_cached_len + 1,
                          nxt_cached_len + 1,
                          cached_src_block_idx,
                          cached_des_block_idx,
                          src_block_idx,
                          des_block_idx);
    } else {
        RTP_LLM_LOG_DEBUG("[stream %s (%d -> %d)] no swap cache blocks",
                          streamLogTag().c_str(),
                          cur_cached_len + 1,
                          nxt_cached_len + 1);
    }

    // update normal output buffer
    updateOutput({new_tokens,
                  num_new_tokens,
                  torch::Tensor(),
                  torch::Tensor(),
                  torch::Tensor(),
                  torch::Tensor(),
                  torch::Tensor(),
                  torch::Tensor(),
                  torch::Tensor(),
                  torch::Tensor(),
                  update_info.update_remote_generate,
                  update_info.force_update_info});
}

void GenerateStream::update(const StreamUpdateInfo& update_info) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(*mutex_);
    RTP_LLM_LOG_DEBUG("stream [%s] update", streamLogTag().c_str());
    *is_context_stream_ = false;
    if (reportUpdateErrorWithoutLock(update_info.error_info)) {
        return;
    }
    if ((hasErrorWithoutLock() || isFinished()) && !update_info.force_update_info) {
        return;
    }
    // Ignore stale worker updates after finish; committing them would duplicate
    // tokens and touch KV blocks only deferred until this worker exits.
    if (isFinished() && !update_info.force_update_info) {
        return;
    }

    const auto& new_tokens     = update_info.new_tokens;
    auto        num_new_tokens = update_info.num_new_tokens;

    int error_token_id = 0;
    if (!complete_token_ids_->update(new_tokens,
                                     begin_time_us_,
                                     num_new_tokens,
                                     generate_input_->inputLength(),
                                     maxTokenNum(),
                                     vocab_size_,
                                     usesBeamSearchTokenLayoutForCurrentStep(),
                                     streamId(),
                                     error_token_id)) {
        reportEventWithoutLock(StreamEvents::Error,
                               ErrorCode::OUT_OF_VOCAB_RANGE,
                               "output token id:" + std::to_string(error_token_id)
                                   + " out of vocab size: " + std::to_string(vocab_size_));
        return;
    }

    resizeSubGenerateStatus(update_info.new_tokens.size(0));

    // Update processor state before publishing output, including the batch that
    // finishes the stream, so normal and speculative decoding share one lifecycle.
    if (auto error = updateNormalLogitProcessorStatus(update_info); error.has_value()) {
        reportEventWithoutLock(StreamEvents::Error, error->code(), error->ToString());
        return;
    }

    // TODO(xinfei.sxf) fix this (update_queue)
    updateOutput(update_info);

    // checkFinished() 已将本轮 updateOutput 中上报的 GenerateDone/Error 事件应用到状态上，
    // 即使 moveToNext() 还未被调度器轮询，这里也能拿到与事件一致的"已完成"判断。
    bool is_done = generate_status_->checkFinished();

    if (!is_done || stream_cache_resource_->reuseCache()) {
        // kv cache blocks must be updated if REUSE_CACHE is on, even the stream is done
        auto update_res = updateKvCacheBlocks(update_info.src_batch_indices);
        if (!update_res) {
            reportEventWithoutLock(StreamEvents::Error, ErrorCode::MALLOC_FAILED, "update kv cache blocks failed");
            return;
        }
    }
}

// src_batch_indices: [batch_size] int, the element must less than the batch_size of last step.
bool GenerateStream::updateKvCacheBlocks(const torch::Tensor& src_batch_indices) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!src_batch_indices.defined() || src_batch_indices.numel() == 0) {
        // no need to update, clear update mapping
        stream_cache_resource_->clearKVBlockUpdateMapping();
        return true;
    }

    auto*            data = src_batch_indices.data_ptr<int32_t>();
    std::vector<int> block_src_batch(data, data + src_batch_indices.numel());
    RTP_LLM_CHECK(block_src_batch.size() == currentBatchSize());

    // NOTE: `1` is used here as updateKvCacheBlocks is called after updateOutput,
    // in which the seqLength has already increased
    bool is_seq_len_misaligned = seqLength() % seqSizePerBlock() != 1;

    return stream_cache_resource_->updateKVBlock(block_src_batch, is_seq_len_misaligned);
}

std::optional<ErrorInfo> GenerateStream::updateNormalLogitProcessorStatus(const StreamUpdateInfo& update_info) {
    updateLogitProcessorMultiSeqStatus(update_info.src_batch_indices);
    RTP_LLM_CHECK(update_info.new_tokens.size(0) == currentBatchSize());
    return updateLogitProcessorStatus(update_info.new_tokens, update_info.num_new_tokens);
}

std::optional<ErrorInfo> GenerateStream::updateLogitProcessorStatus(const torch::Tensor& new_tokens,
                                                                    int32_t              num_new_tokens) {
    RTP_LLM_PROFILE_FUNCTION();
    if (num_new_tokens <= 0) {
        return std::nullopt;
    }
    for (const auto& logit_processor_ptr : logits_processor_list_) {
        auto error = logit_processor_ptr->updateStatus(new_tokens, num_new_tokens);
        if (error.has_value()) {
            return error;
        }
    }
    return validateLogitsProcessorState();
}

void GenerateStream::updateLogitProcessorMultiSeqStatus(const torch::Tensor& src_batch_indices) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!src_batch_indices.defined() || !hasNumBeams()) {
        return;
    }

    auto*            data = src_batch_indices.data_ptr<int32_t>();
    std::vector<int> src_batch_indices_vec(data, data + src_batch_indices.numel());
    RTP_LLM_CHECK(src_batch_indices_vec.size() == currentBatchSize());

    for (const auto& logit_processor_ptr : logits_processor_list_) {
        logit_processor_ptr->updateMultiSeqStatus(src_batch_indices_vec);
    }
}

std::optional<ErrorInfo> GenerateStream::validateLogitsProcessorState() {
    if (hasErrorWithoutLock()) {
        return std::nullopt;
    }

    const auto  stream_output_len = static_cast<int64_t>(outputTokenLen());
    const auto& processors        = logits_processor_list_;
    for (size_t i = 0; i < processors.size(); ++i) {
        const auto& processor            = processors[i];
        const auto  processor_output_len = processor->committedOutputLen();
        if (processor_output_len.has_value() && processor_output_len.value() != stream_output_len) {
            return ErrorInfo(ErrorCode::UNKNOWN_ERROR,
                             "logits processor committed output length mismatch: processor_index=" + std::to_string(i)
                                 + ", processor=" + std::to_string(processor_output_len.value())
                                 + ", stream_output=" + std::to_string(stream_output_len));
        }
    }
    return std::nullopt;
}

void GenerateStream::setLoss(const torch::Tensor& loss) {
    RTP_LLM_PROFILE_FUNCTION();
    auto loss_cpu  = loss.is_cuda() ? loss.cpu() : loss;
    auto loss_size = loss_cpu.numel();
    RTP_LLM_CHECK(loss_index_ + loss_size < inputLength());
    memcpy(loss_.data_ptr<float>() + loss_index_, loss_cpu.data_ptr<float>(), loss_size * sizeof(float));
    loss_index_ += loss_size;
}

void GenerateStream::setSoftmaxProbs(const torch::Tensor& softmax_probs,
                                     int                  start_pos,
                                     const torch::Tensor& src_batch_indices) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(softmax_probs_.defined());
    auto probs_cpu = softmax_probs.to(torch::kCPU, torch::kFloat32).contiguous();
    RTP_LLM_CHECK(probs_cpu.dim() == 2);
    RTP_LLM_CHECK(probs_cpu.size(0) == currentBatchSize());
    auto num_probs = probs_cpu.size(1);
    RTP_LLM_CHECK(start_pos >= 0);
    RTP_LLM_CHECK(start_pos + num_probs <= softmax_probs_.size(1));
    RTP_LLM_CHECK(currentBatchSize() <= softmax_probs_.size(0));

    if (src_batch_indices.defined()) {
        auto indices_cpu = src_batch_indices.to(torch::kCPU, torch::kLong).contiguous();
        RTP_LLM_CHECK(indices_cpu.dim() == 1);
        RTP_LLM_CHECK(indices_cpu.numel() == currentBatchSize());
        RTP_LLM_CHECK(indices_cpu.min().item<int64_t>() >= 0);
        RTP_LLM_CHECK(indices_cpu.max().item<int64_t>() < softmax_probs_.size(0));

        if (start_pos > 0) {
            auto history = softmax_probs_.narrow(1, 0, start_pos).index_select(0, indices_cpu);
            softmax_probs_.narrow(0, 0, currentBatchSize()).narrow(1, 0, start_pos).copy_(history);
        }
    }

    softmax_probs_.narrow(0, 0, currentBatchSize()).narrow(1, start_pos, num_probs).copy_(probs_cpu);
}

torch::Tensor GenerateStream::getLoss() {
    return loss_;
}

torch::Tensor GenerateStream::getLastHiddenStates() const {
    return last_hidden_states_;
}

torch::Tensor GenerateStream::getSoftmaxProbs() {
    return softmax_probs_;
}

void GenerateStream::setMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter) {
    metrics_reporter_ = metrics_reporter;
}

void GenerateStream::reportMetricOnce() {
    if (metrics_reported_) {
        return;
    }
    metrics_reported_ = true;
    reportMetric();
}

void GenerateStream::reportMetric() {
    reportStreamMetrics();
    stream_cache_resource_->reportCacheReuseMetrics();
}

void GenerateStream::reportStreamMetrics() {
    RTP_LLM_PROFILE_FUNCTION();
    if (metrics_reporter_) {
        bool                         cancelled = statusInfo().code() == ErrorCode::CANCELLED;
        bool                         timeout   = statusInfo().code() == ErrorCode::GENERATE_TIMEOUT;
        RtpLLMStreamMetricsCollector collector;
        collector.qps               = true;
        collector.cancel_qps        = cancelled;
        collector.error_qps         = hasError() && !cancelled;
        collector.is_streaming_qps  = generate_input_->generate_config->is_streaming;
        collector.not_streaming_qps = !generate_input_->generate_config->is_streaming;
        if (getStatus() == StreamState::FINISHED || cancelled || timeout) {
            collector.reuse_length       = initial_reuse_length_;
            collector.input_token_length = inputLength();
            collector.effective_context_length =
                std::max<int64_t>(0, collector.input_token_length - initial_reuse_length_);
            collector.output_token_length    = outputTokenLen();
            collector.iterate_count          = iter_count_;
            collector.query_batch_size       = maxBatchSize();
            collector.total_latency_us       = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
            collector.first_token_latency_us = complete_token_ids_->firstTokenLatencyUs();
            RTP_LLM_LOG_DEBUG(
                "stream [%s] report first latency us = %ld", streamLogTag().c_str(), collector.first_token_latency_us);
            collector.wait_latency_us = wait_time_us_;
            if (scheduler_enqueue_time_us_ > 0 && can_run_time_us_ > scheduler_enqueue_time_us_) {
                collector.enqueue_to_canrun_us = can_run_time_us_ - scheduler_enqueue_time_us_;
            }
            if (can_run_time_us_ > 0 && first_running_time_us_ > can_run_time_us_) {
                collector.canrun_to_running_us = first_running_time_us_ - can_run_time_us_;
            }
            collector.loading_cache_latency_us = loading_cache_latency_us_;
            collector.load_done_to_running_us  = load_done_to_running_us_;
            collector.batch_with_prefill_times = batch_with_prefill_times_;
            collector.batch_with_prefill_len   = batch_with_prefill_len_;
            collector.malloc_failed_times      = stream_cache_resource_->mallocFailedTimes();
            if (timeout) {
                collector.timeout_latency_us = getTimeoutMs() * 1000;
            }
        }
        // pass tag will cause default tags deep copy
        static kmonitor::MetricsTags timeout_tag("timeout", "true");
        metrics_reporter_->report<RtpLLMStreamMetrics, RtpLLMStreamMetricsCollector>(timeout ? &timeout_tag : nullptr,
                                                                                     &collector);
    }
}

std::string GenerateStream::debugString() const {
    std::stringstream debug_string;
    debug_string << "GenerateStream {"
                 << "generate_input:" << generate_input_->debugString() << ", max_seq_len:" << max_seq_len_
                 << ", input_length:" << inputLength() << ", seq_length:" << seqLength()
                 << ", reuse_length:" << reuse_length_ << ", current_batch_size:" << currentBatchSize()
                 << ", next_batch_size:" << nextBatchSize() << ", need_release_resource: " << need_release_resource_
                 << ", sp_edit_search_index: " << sp_edit_search_index_ << ", mtp token indices" << mtp_token_index_
                 << ", contain_propose_token: " << contain_propose_token_ << ", propose_token: " << propose_token_;

    for (int i = 0; i < propose_token_.size(); i++) {
        debug_string << propose_token_[i] << " ";
    }
    if (last_hidden_states_.defined()) {
        debug_string << ", hidden_state_token_num: " << last_hidden_states_.size(0);
    }
    debug_string << ", complete_token_ids: [";
    for (size_t i = 0; i < complete_token_ids_->batchSize(); i++) {
        debug_string << complete_token_ids_->toString(i) << ",";
    }

    debug_string << ", cum_log_probs: [";
    if (cum_log_probs_.defined()) {
        auto cpu = cum_log_probs_.cpu().contiguous();
        for (int64_t i = 0; i < cpu.numel(); ++i) {
            if (i > 0)
                debug_string << ", ";
            debug_string << cpu.data_ptr<float>()[i];
        }
    }
    debug_string << "]";
    debug_string << ", stream_cache_resource: " << stream_cache_resource_->debugString();

    debug_string << "}";

    return debug_string.str();
}

int GenerateStream::reuseBlockSize() const {
    int reuse_length       = reuseLength();
    int seq_size_per_block = seqSizePerBlock();
    return reuse_length / seq_size_per_block;
}

void GenerateStream::setSeqLength(int seq_length) {
    complete_token_ids_->setSeqLength(seq_length);
}

void GenerateStream::setPerfTest(bool perf_test) {
    perf_test_ = perf_test;
}

void GenerateStream::setIsContextStream(bool is_context_stream) {
    *is_context_stream_ = is_context_stream;
}

StreamCacheResource& GenerateStream::streamCacheResource() {
    return *stream_cache_resource_;
}

void GenerateStream::CopyOnWrite(const GenerateStream& other_stream, bool copy_loss, bool share) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("GenerateStream::CopyOnWrite(copy_loss=%d, share=%d)", copy_loss, share);
    complete_token_ids_ = make_shared<CompleteTokenIds>(*other_stream.complete_token_ids_, share);
    grpc_normal_device_state_pending_ =
        std::make_shared<std::atomic<bool>>(other_stream.hasGrpcNormalDeviceStatePending());
    cum_log_probs_ = other_stream.cum_log_probs_.clone();
    if (other_stream.calculateLoss() && copy_loss) {
        loss_ = other_stream.loss_.clone();
    } else {
        loss_ = torch::Tensor();
    }
}

GenerateStream::TimeInfo GenerateStream::getTimeInfo() {
    std::lock_guard<std::mutex> lock(*mutex_);
    const auto                  first_token_time_us = complete_token_ids_->firstTokenTimeUs();

    TimeInfo time_info;
    time_info.begin_time_us           = begin_time_us_;
    time_info.wait_time_us            = wait_time_us_;
    time_info.running_started         = running_started_;
    time_info.running_started_time_us = running_started_time_us_;
    time_info.first_token_committed   = first_token_time_us > 0;
    time_info.first_token_time_us     = first_token_time_us;
    // Reuse the frozen firstTokenLatencyUs() (stamped at commit time) instead of
    // recomputing against begin_time_us_. Decode resets begin AFTER the first
    // token has already committed (DecodeRpcServer::update precedes
    // resetBeginTime; BatchDecodeScheduler resets running streams), so a
    // read-time subtraction would go negative and diverge from the frozen
    // metrics/aux_info definition. The frozen value is non-negative by
    // construction; clamping the subtraction to 0 would instead mask the reset.
    time_info.first_token_rt_us       = complete_token_ids_->firstTokenLatencyUs();
    time_info.generation_done         = generation_done_;
    time_info.generation_done_time_us = generation_done_time_us_;
    return time_info;
}

bool GenerateStream::queryPdSep() const {
    return generate_input_->generate_config->pd_separation;
}

void GenerateStream::incBatchWithPrefillTimes(int32_t times) {
    batch_with_prefill_times_ += times;
}

void GenerateStream::incBatchWithPrefillLen(int32_t len) {
    batch_with_prefill_len_ += len;
}

void GenerateStream::fillSubGenerateStatus(StreamState state) {
    for (size_t i = 0; i < sub_generate_status_.size(); ++i) {
        sub_generate_status_[i] = state;
    }
}

void GenerateStream::resizeSubGenerateStatus(size_t new_size) {
    if (sub_generate_status_.size() != new_size) {
        size_t old_size = sub_generate_status_.size();
        sub_generate_status_.resize(new_size);
        for (size_t i = old_size; i < new_size; ++i) {
            sub_generate_status_[i] = StreamState::RUNNING;
        }
    }
}

void GenerateStream::holdKVCacheForPDSep() {
    stream_cache_resource_->holdKVCacheForPDSep();
}

void GenerateStream::releaseKVCacheForPDSep() {
    stream_cache_resource_->releaseKVCacheForPDSep();
}

std::pair<std::string, uint32_t> GenerateStream::prefillAddr() const {
    for (const auto& role_addr : generate_input_->generate_config->role_addrs) {
        if (role_addr.role == RoleType::PREFILL) {
            return std::make_pair(role_addr.ip, role_addr.grpc_port);
        }
    }

    return std::make_pair("", 0);
}

}  // namespace rtp_llm
