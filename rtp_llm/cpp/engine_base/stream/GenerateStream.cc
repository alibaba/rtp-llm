#include <condition_variable>
#include <cstddef>
#include <memory>
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"

using namespace std;

namespace rtp_llm {

GenerateStream::GenerateStream(const shared_ptr<GenerateInput>& input,
                               const rtp_llm::GptInitParameter& params,
                               const ResourceContext&           resource_context,
                               kmonitor::MetricsReporterPtr     metrics_reporter,
                               size_t                           extra_reserve_token_num,
                               bool                             perf_test):
    generate_input_(input),
    max_seq_len_(params.max_seq_len_),
    vocab_size_(params.vocab_size_),
    stream_cache_resource_(std::make_shared<StreamCacheResource>(
        this, resource_context, input->need_release_resource, input->generate_config->adapter_name)),
    need_release_resource_(input->need_release_resource),
    enable_fast_gen_(params.enable_fast_gen_),
    gen_timeline_(input->generate_config->gen_timeline),
    metrics_reporter_(metrics_reporter),
    special_tokens_(params.special_tokens_),
    output_mutex_(std::make_shared<std::mutex>()),
    cv_(std::make_shared<std::condition_variable>()),
    mm_position_ids_style_(PositionIdsStyle(params.mm_position_ids_style_)),
    dtype_(params.data_type_),
    hidden_size_(params.hidden_size_) {
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
    device_        = rtp_llm::DeviceFactory::getDefaultDevice();
    if (generate_input_->generate_config->calculate_loss && inputLength() > 1) {
        loss_ = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {(size_t)inputLength() - 1}, rtp_llm::AllocationType::HOST}, {});
    }
    if (generate_input_->generate_config->return_softmax_probs) {
        softmax_probs_ = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {init_batch_size, (size_t)max_seq_len_}, rtp_llm::AllocationType::HOST}, {});
        memset(softmax_probs_->data(), 0, softmax_probs_->sizeBytes());
    }
    if (generate_input_->generate_config->return_all_hidden_states) {
        setReturnLastHiddenStates(true);
    }
    complete_token_ids_ = std::make_shared<CompleteTokenIds>(
        device_, init_batch_size, maxBatchSize(), max_seq_len_, params.seq_size_per_block_);
    complete_token_ids_->init(input, extra_reserve_token_num);

    last_output_pos_ = seqLength();
    max_chunk_len_   = seqLength();

    cum_log_probs_ =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_FP32, {init_batch_size}, rtp_llm::AllocationType::HOST}, {});
    memset(cum_log_probs_->data(), 0, cum_log_probs_->sizeBytes());

    is_context_stream_       = std::make_shared<bool>();
    *is_context_stream_      = true;
    generate_status_         = std::make_shared<GenerateStatus>();
    generate_status_->status = StreamState::WAITING;
    sub_generate_status_.clear();
    resizeSubGenerateStatus(init_batch_size);

    stream_cache_resource_->init(init_batch_size);

    setReturnAllProbs(generate_input_->generate_config->return_all_probs);

    think_logits_processor_ptr_ = ThinkModeLogitsProcessor::fromGenerateInput(device_, generate_input_, maxBatchSize());
    tree_logits_processor_ptr_  = TreeLogitsProcessor::fromGenerateInput(device_, generate_input_, init_batch_size);
    multi_seq_logits_processor_ptr_ =
        MultiSeqLogitsProcessor::fromGenerateInput(device_, generate_input_, special_tokens_.eos_token_id_);

    initializeLogitsProcessorList();
}

void GenerateStream::initializeLogitsProcessorList() {
    if (think_logits_processor_ptr_ != nullptr) {
        logits_processor_list_.push_back(std::static_pointer_cast<BaseLogitsProcessor>(think_logits_processor_ptr_));
    }
    if (tree_logits_processor_ptr_ != nullptr) {
        logits_processor_list_.push_back(std::static_pointer_cast<BaseLogitsProcessor>(tree_logits_processor_ptr_));
    }
    if (multi_seq_logits_processor_ptr_ != nullptr) {
        logits_processor_list_.push_back(
            std::static_pointer_cast<BaseLogitsProcessor>(multi_seq_logits_processor_ptr_));
    }
}

void GenerateStream::resetBeginTime(int64_t begin_time_us) {
    begin_time_us_ = begin_time_us;
}

bool GenerateStream::hasCacheKeys() const {
    return stream_cache_resource_->hasCacheKeys();
}

const std::vector<int64_t>& GenerateStream::cacheKeys(int32_t batch_id) const {
    return stream_cache_resource_->cacheKeys(batch_id);
}

absl::StatusOr<int> GenerateStream::acquireCapacity(int token_capacity) {
    if (token_capacity <= 0) {
        return absl::InternalError("token_capacity is <= 0");
    }
    if (isChunkStream()) {
        // TODO(xinfei.sxf) add min_chunk_len ?
        if (current_chunk_len_ == 0) {
            current_chunk_len_ = reuse_length_;
        }
        auto remaining_token = max_chunk_len_ - current_chunk_len_;
        last_chunk_len_      = current_chunk_len_;
        if (token_capacity > remaining_token) {
            current_chunk_len_ = max_chunk_len_;
            return remaining_token;
        } else {
            current_chunk_len_ += token_capacity;
            return token_capacity;
        }
    } else if (!isContextStream()) {
        return 1;
    }
    RTP_LLM_CHECK(false);
    return absl::InternalError("unexpected call");
}

void GenerateStream::cancel() {
    setStop(ErrorCode::CANCELLED, "cancel stream");
}

absl::StatusOr<int> GenerateStream::initKVBlock(int token_capacity, size_t reserve_step) {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    if (generate_status_->status == StreamState::WAITING) {
        wait_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    } else if (generate_status_->status == StreamState::PAUSED) {
        pause_time_us_ += autil::TimeUtility::currentTimeInMicroSeconds() - last_pause_us_;
    }
    return stream_cache_resource_->initKVBlock(token_capacity, reserve_step);
}

void GenerateStream::fakeInitKVBlock() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    stream_cache_resource_->fakeInitKVBlock();
}

absl::StatusOr<int> GenerateStream::incrKVBlock(int token_capacity, size_t reserve_step) {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return stream_cache_resource_->incrKVBlock(token_capacity, reserve_step);
}

int GenerateStream::tryReleaseKVBlock(int nums) {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    RTP_LLM_CHECK_WITH_INFO(nums >= 0, "release block nums is < 0");
    auto release_blocks = stream_cache_resource_->tryReleaseKVBlock(nums);
    incrFallbackBlock(release_blocks);
    return release_blocks;
}

void GenerateStream::releaseResource() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    stream_cache_resource_->releaseResource();
}
void GenerateStream::setNeedReleaseResource(bool need_release_resource) {
    need_release_resource_ = need_release_resource;
    stream_cache_resource_->setNeedReleaseResource(need_release_resource);
}
int GenerateStream::nextNeedBlockNums(size_t reserve_step) const {
    // TODO: maybe need fix when context and reuse
    return stream_cache_resource_->singleBatchNeedBlocks(seqLength() + reserve_step) * nextBatchSize();
}

void GenerateStream::incrFallbackBlock(int fallback_blocks) {
    fallback_blocks_ += fallback_blocks;
    fallback_times_ += 1;
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
    if (generate_input_->generate_config->inter_request_id != -1) {
        return generate_input_->generate_config->inter_request_id;
    }
    return generate_input_->request_id;
}
int GenerateStream::loraId() const {
    return generate_input_->lora_id;
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

bool GenerateStream::needTilingForSampling() const {
    return isContextStream() && currentBatchSize() != nextBatchSize() && !hasNumBeams();
}

int GenerateStream::numReturnSequences() const {
    return generate_input_->generate_config->num_return_sequences;
}

bool GenerateStream::calculateLoss() const {
    return loss_ && loss_index_ < inputLength() - 1;
}

bool GenerateStream::calculateSoftmaxProbs() const {
    return generate_input_->generate_config->return_softmax_probs;
}

bool GenerateStream::returnLogits() const {
    return generate_input_->generate_config->return_logits;
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
                setStop(ErrorCode::LONG_PROMPT_ERROR,
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

int GenerateStream::currentChunkLen() const {
    return current_chunk_len_;
}

void GenerateStream::resetChunkLen(int chunk_len, int max_chunk_len) {
    last_chunk_len_    = 0;
    current_chunk_len_ = chunk_len;
    max_chunk_len_     = max_chunk_len;
}

int GenerateStream::seqLength() const {
    return complete_token_ids_->seqLength();
}

int GenerateStream::adjustedCommonLen() const {
    return maxBatchSize() == 1 ? seqLength() : inputLength() / seqSizePerBlock() * seqSizePerBlock();
}

int GenerateStream::seqSizePerBlock() const {
    return stream_cache_resource_->seqSizePerBlock();
}

int GenerateStream::contextLength() const {
    int begin_pos = prefixLength();
    int end_pos   = isChunkStream() ? currentChunkLen() : seqLength();
    return end_pos - begin_pos;
}

int GenerateStream::inputPrefixLength() const {
    return generate_input_->prefix_length;
}

int GenerateStream::prefixLength() const {
    if (fallback_prefix_length_) {
        return fallback_prefix_length_;
    } else if (last_chunk_len_) {
        return last_chunk_len_;
    }
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
    if (generate_input_->mm_locs) {
        auto& locs = generate_input_->mm_locs.value();
        for (int i = locs->size() - 1; i >= 0; --i) {
            if (reuse_length_ > *locs->dataWithOffset<int32_t>(i)) {
                reuse_mm_length_ = i + 1;
                break;
            }
        }
    }
}

void GenerateStream::setLocalReuseLength(int length) {
    local_reuse_length_ = length;
}

void GenerateStream::setGpuReuseLength(int length) {
    gpu_reuse_length_ = length;
}

void GenerateStream::setMemoryReuseLength(int length) {
    memory_reuse_length_ = length;
}

void GenerateStream::setRemoteReuseLength(int length) {
    remote_reuse_length_ = length;
}

int GenerateStream::localReuseLength() const {
    return local_reuse_length_;
}

int GenerateStream::remoteReuseLength() const {
    return remote_reuse_length_;
}

int GenerateStream::gpuReuseLength() const {
    return gpu_reuse_length_;
}

int GenerateStream::memoryReuseLength() const {
    return memory_reuse_length_;
}

void GenerateStream::setInitialReuseLength(int initial_reuse_length) {
    initial_reuse_length_ = initial_reuse_length;
}

int GenerateStream::fallbackPrefixLength() const {
    return fallback_prefix_length_;
}

void GenerateStream::setFallbackPrefixLength(int fallback_prefix_length) {
    fallback_prefix_length_ = fallback_prefix_length;
}

void GenerateStream::incLastOutputPos() {
    last_output_pos_++;
}

bool GenerateStream::isContextStream() const {
    return *is_context_stream_;
}

bool GenerateStream::isChunkStream() const {
    return enable_fast_gen_ && current_chunk_len_ < max_chunk_len_;
}

const rtp_llm::BufferPtr& GenerateStream::cumLogProbs() const {
    return cum_log_probs_;
}

const rtp_llm::BufferPtr& GenerateStream::completeTokenIds() {
    return complete_token_ids_->completeTokenIds();
}

std::vector<int> GenerateStream::completeTokenIdsVec(int batch_idx) {
    RTP_LLM_CHECK(batch_idx < currentBatchSize());
    return complete_token_ids_->completeTokenIdsVec(batch_idx);
}

std::vector<int> GenerateStream::commonCompleteTokenIdsVec(int batch_idx) {
    RTP_LLM_CHECK(batch_idx < currentBatchSize());
    return complete_token_ids_->commonCompleteTokenIdsVec(batch_idx);
}

int GenerateStream::currentExecuteTokenSize() {
    return currentExecuteTokens(0).size() * currentBatchSize();
}

std::vector<torch::Tensor> GenerateStream::multimodalFeatures() const {
    if (generate_input_->multimodal_features) {
        auto& features = generate_input_->multimodal_features.value();
        return std::vector<torch::Tensor>(features.begin() + reuse_mm_length_, features.end());
    } else {
        return std::vector<torch::Tensor>();
    }
}

int GenerateStream::multimodalFeaturesLength() const {
    return multimodalFeatures().size() * currentBatchSize();
}

rtp_llm::BufferPtr GenerateStream::multimodalLocations() const {
    if (!generate_input_->mm_locs) {
        return nullptr;
    }
    auto& mm_locs = generate_input_->mm_locs.value();
    return mm_locs->slice(reuse_mm_length_, mm_locs->size() - reuse_mm_length_);
}

vector<vector<int>> GenerateStream::multimodalIntervals() const {
    if (!generate_input_->mm_locs && !generate_input_->multimodal_features) {
        return {};
    }
    vector<vector<int>> res;
    auto                locs     = generate_input_->mm_locs.value();
    auto                features = generate_input_->multimodal_features.value();
    for (int i = 0; i < locs->size(); ++i) {
        res.emplace_back(vector<int>({*locs->dataWithOffset<int>(i), int(features[i].sizes()[0])}));
    }
    return res;
}

vector<int> GenerateStream::textTokensMask() const {
    if (!generate_input_->text_tokens_mask) {
        return {};
    }
    auto token_masks = rtp_llm::buffer2vector<int>(*generate_input_->text_tokens_mask.value());
    if (reuseLength() > 0) {
        return vector<int>(token_masks.begin() + reuseLength(), token_masks.end());
    } else {
        return token_masks;
    }
}

rtp_llm::BufferPtr GenerateStream::generateContextPositionIds(rtp_llm::DeviceBase* device) {
    optional<vector<rtp_llm::BufferPtr>> position_ids_buffer = nullopt;
    if (generate_input_->mm_position_ids.has_value()) {
        position_ids_buffer = rtp_llm::torchTensorVec2BufferVec(generate_input_->mm_position_ids.value());
    }
    context_position_ids_ = PositionIdsGenerator::generatePositionIds(
        device, generate_input_->inputLength(), mm_position_ids_style_, generate_input_->mm_locs, position_ids_buffer);
    return context_position_ids_.value();
}

void GenerateStream::generateNextPositionId(int32_t* now_pos, rtp_llm::DeviceBase* device) {
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
    if (isContextStream()) {
        setFallbackPrefixLength(0);
    }
}

void GenerateStream::spStep() {
    sp_iter_count_++;
}

int64_t GenerateStream::getTimeoutMs() const {
    return generate_input_->generate_config->timeout_ms;
}

void GenerateStream::checkTimeout() {
    auto running_time_ms = (autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_) / 1000;
    auto timeout_ms      = getTimeoutMs();
    if (timeout_ms > 0 && timeout_ms < running_time_ms) {
        stopAndRelease(ErrorCode::GENERATE_TIMEOUT,
                       "query has been running " + std::to_string(running_time_ms) + " ms, "
                           + "timeout_ms = " + std::to_string(timeout_ms) + ", it's timeout");
    }
}

void GenerateStream::setStopWithoutLock(ErrorCode error_code, const std::string& error_msg) {
    auto cost_time_ms = (autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_) / 1000;
    RTP_LLM_LOG_WARNING("stop stream [%ld], error msg: [%s], current state [%s], "
                        "input len [%d], seq len [%d], timeout [%ld] ms, running [%ld] ms",
                        streamId(),
                        error_msg.c_str(),
                        StreamStateToString(generate_status_->status).c_str(),
                        inputLength(),
                        seqLength(),
                        getTimeoutMs(),
                        cost_time_ms);
    generate_status_->status     = StreamState::STOPPED;
    generate_status_->error_info = ErrorInfo(error_code, error_msg);
    cv_->notify_one();
}

void GenerateStream::setStop(ErrorCode error_code, const std::string& error_msg) {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    setStopWithoutLock(error_code, error_msg);
}

void GenerateStream::stopAndRelease(ErrorCode error_code, const std::string& error_msg) {
    setStop(error_code, error_msg);
    releaseResource();
}

ErrorInfo GenerateStream::statusInfo() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_->error_info;
}

bool GenerateStream::isDoneWithoutLock(int batch_id) const {
    auto status = sub_generate_status_[batch_id].status;
    return status == StreamState::FINISHED || status == StreamState::STOPPED;
}

void GenerateStream::setPaused() {
    // TODO(xinfei.sxf) fix mutex name
    std::lock_guard<std::mutex> lock(*output_mutex_);
    if (stoppedWithoutLock()) {
        return;
    }
    *is_context_stream_      = true;
    generate_status_->status = StreamState::PAUSED;
    last_pause_us_           = autil::TimeUtility::currentTimeInMicroSeconds();
}

bool GenerateStream::setRunning() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    if (stoppedWithoutLock()) {
        return false;
    }
    generate_status_->status = StreamState::RUNNING;
    return true;
}

void GenerateStream::setFinishedWithoutLock() {
    generate_status_->status = StreamState::FINISHED;
    fillSubGenerateStatus(StreamState::FINISHED);
    cv_->notify_one();
}

bool GenerateStream::stoppedWithoutLock() {
    return generate_status_->status == StreamState::STOPPED;
}

bool GenerateStream::stopped() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_->status == StreamState::STOPPED;
}

bool GenerateStream::waiting() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_->status == StreamState::WAITING;
}

bool GenerateStream::paused() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_->status == StreamState::PAUSED;
}

std::string GenerateStream::stopReason() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_->error_info.ToString();
}

bool GenerateStream::finishedWithoutLock() {
    return generate_status_->status == StreamState::FINISHED;
}

bool GenerateStream::running() {
    return generate_status_->status == StreamState::RUNNING;
}

void GenerateStream::cancelIfNotRunning() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    if (generate_status_->status == StreamState::WAITING || generate_status_->status == StreamState::REMOTE_RUNNING) {
        auto cost_time_ms = (autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_) / 1000;
        RTP_LLM_LOG_WARNING("stop stream: %ld %s, input len [%d], seq len [%d], timeout: [%ld] ms, running [%ld] ms",
                            streamId(),
                            "cancel stream in waiting or remote running",
                            inputLength(),
                            seqLength(),
                            getTimeoutMs(),
                            cost_time_ms);
        generate_status_->status     = StreamState::STOPPED;
        generate_status_->error_info = ErrorInfo(ErrorCode::CANCELLED, "cancel stream in waiting or remote running");
    }
}

bool GenerateStream::finished() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_->status == StreamState::FINISHED;
}

bool GenerateStream::isRemoteRunningWithoutLock() {
    return generate_status_->status == StreamState::REMOTE_RUNNING;
}

bool GenerateStream::needRemoteGenerate() const {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return need_remote_generate_;
}

bool GenerateStream::setRemoteGenerate() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    if (stoppedWithoutLock() || finishedWithoutLock()) {
        return false;
    }
    generate_status_->status = StreamState::REMOTE_RUNNING;
    cv_->notify_one();
    return true;
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

const ResourceContext& GenerateStream::resourceContext() const {
    return stream_cache_resource_->resourceContext();
}

size_t GenerateStream::maxBlockSize() const {
    return stream_cache_resource_->maxBlockSize();
}

size_t GenerateStream::maxTokenNum() const {
    return std::min(max_seq_len_, generate_input_->generate_config->max_new_tokens + generate_input_->inputLength());
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

    matchEosToken();
    matchStopWordsList();

    // check if all batch finished
    return std::all_of(sub_generate_status_.begin(), sub_generate_status_.end(), [](GenerateStatus& generate_status) {
        return generate_status.status == StreamState::FINISHED;
    });
}

void GenerateStream::matchEosToken() {
    for (int i = 0; i < currentBatchSize(); ++i) {
        matchEosToken(i);
    }
}

void GenerateStream::matchEosToken(int batch_id) {
    if ((!generate_input_->generate_config->ignore_eos)
        && complete_token_ids_->matchEosToken(batch_id, special_tokens_.eos_token_id_)) {
        sub_generate_status_[batch_id].status = StreamState::FINISHED;
    }
}

bool GenerateStream::waitForRemoteGenerate() {
    std::unique_lock<std::mutex> lock(*output_mutex_);
    // Wait until need_remote_generate_ is true or stream status -> done
    cv_->wait(lock, [this] {
        return generate_status_->status == StreamState::REMOTE_RUNNING
               || generate_status_->status == StreamState::STOPPED || generate_status_->status == StreamState::FINISHED;
    });
    // If stream status is abnormal, log the error info
    if (generate_status_->status == StreamState::STOPPED) {
        RTP_LLM_LOG_WARNING("waitForRemoteGenerate exits due to stream [%ld] stopped, error: %s",
                            streamId(),
                            generate_status_->error_info.ToString().c_str());
    }

    return generate_status_->status == StreamState::REMOTE_RUNNING;
}

std::vector<int> GenerateStream::getLatestTokens(size_t token_num) {
    return complete_token_ids_->getLatestTokens(token_num);
}

void GenerateStream::matchStopWordsList() {
    if (seqLength() < generate_input_->generate_config->min_new_tokens + inputLength()) {
        return;
    }
    if (seqLength() == inputLength()) {
        return;
    }
    for (int i = 0; i < currentBatchSize(); ++i) {
        matchStopWordsList(i);
    }
}

void GenerateStream::matchStopWordsList(int batch_id) {
    // note: stop_words_list in generate_config contains stop_words_list in special_tokens
    bool match = false;
    for (auto& stop_words : generate_input_->generate_config->stop_words_list) {
        if (generate_input_->generate_config->ignore_eos && stop_words.size() == 1
            && stop_words[0] == special_tokens_.eos_token_id_) {
            continue;
        }
        if (complete_token_ids_->matchStopWordsList(batch_id, stop_words)) {
            match = true;
            break;
        }
    }
    if (match) {
        sub_generate_status_[batch_id].status = StreamState::FINISHED;
    }
}

void GenerateStream::update(const StreamUpdateInfo& update_info) {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    RTP_LLM_LOG_DEBUG("stream [%ld] update", streamId());
    *is_context_stream_ = false;
    if (stoppedWithoutLock() && !update_info.force_update_info) {
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
                                     hasNumBeams(),
                                     streamId(),
                                     error_token_id)) {
        setStopWithoutLock(ErrorCode::OUT_OF_VOCAB_RANGE,
                           "output token id:" + std::to_string(error_token_id)
                               + " out of vocab size: " + std::to_string(vocab_size_));
        return;
    }

    resizeSubGenerateStatus(update_info.new_tokens->shape()[0]);

    // TODO(xinfei.sxf) fix this (update_queue)
    updateOutput(update_info);

    bool is_done = finishedWithoutLock() || stoppedWithoutLock();

    if (!is_done) {
        updateLogitProcessorStatus(update_info);
    }

    if (!is_done || reuseCache()) {
        // kv cache blocks must be updated if REUSE_CACHE is on, even the stream is done
        auto update_res = updateKvCacheBlocks(update_info.src_batch_indices);
        if (!update_res) {
            setStopWithoutLock(ErrorCode::MALLOC_FAILED, "update kv cache blocks failed");
            return;
        }
    }
}

// src_batch_indices: [batch_size] int, the element must less than the batch_size of last step.
bool GenerateStream::updateKvCacheBlocks(const rtp_llm::BufferPtr& src_batch_indices) {
    if (src_batch_indices == nullptr || src_batch_indices->size() == 0) {
        // no need to update, clear update mapping
        stream_cache_resource_->clearKVBlockUpdateMapping();
        return true;
    }

    auto block_src_batch = rtp_llm::buffer2vector<int>(*src_batch_indices);
    RTP_LLM_CHECK(block_src_batch.size() == currentBatchSize());

    // NOTE: `1` is used here as updateKvCacheBlocks is called after updateOutput,
    // in which the seqLength has already increased
    bool is_seq_len_misaligned = seqLength() % seqSizePerBlock() != 1;

    return stream_cache_resource_->updateKVBlock(block_src_batch, is_seq_len_misaligned);
}

void GenerateStream::updateLogitProcessorMultiSeqStatus(const rtp_llm::BufferPtr& src_batch_indices) {
    if (src_batch_indices == nullptr || !hasNumBeams()) {
        return;
    }

    auto src_batch_indices_vec = rtp_llm::buffer2vector<int>(*src_batch_indices);
    RTP_LLM_CHECK(src_batch_indices_vec.size() == currentBatchSize());

    for (auto logit_processor_ptr : getAllLogitsProcessorPtr()) {
        logit_processor_ptr->updateMultiSeqStatus(src_batch_indices_vec);
    }
}

void GenerateStream::updateLogitProcessorStatus(const StreamUpdateInfo& update_info) {
    updateLogitProcessorMultiSeqStatus(update_info.src_batch_indices);

    const auto& new_tokens = update_info.new_tokens;
    RTP_LLM_CHECK(new_tokens->shape()[0] == currentBatchSize());
    auto num_new_tokens = update_info.num_new_tokens;

    for (auto logit_processor_ptr : getAllLogitsProcessorPtr()) {
        logit_processor_ptr->updateStatus(new_tokens, num_new_tokens);
    }
}

void GenerateStream::setLoss(const rtp_llm::Buffer& loss) {
    RTP_LLM_CHECK(loss_index_ + loss.size() < inputLength());
    device_->copy({loss_->view(loss_index_, loss.size()), loss});
    loss_index_ += loss.size();
}

void GenerateStream::setSoftmaxProbs(const rtp_llm::Buffer& softmax_probs, int start_pos) {
    RTP_LLM_CHECK(softmax_probs.dim() == 2);
    RTP_LLM_CHECK(softmax_probs.shape()[0] == currentBatchSize());
    for (int i = 0; i < currentBatchSize(); ++i) {
        device_->copy({(*softmax_probs_)[i].view(start_pos, softmax_probs.shape()[1]), softmax_probs[i]});
    }
}

rtp_llm::BufferPtr GenerateStream::getLoss() {
    return loss_;
}

rtp_llm::BufferPtr GenerateStream::getLastHiddenStates() const {
    return last_hidden_states_;
}

rtp_llm::BufferPtr GenerateStream::getSoftmaxProbs() {
    return softmax_probs_;
}

void GenerateStream::setMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter) {
    metrics_reporter_ = metrics_reporter;
}
void GenerateStream::reportMetric() {
    if (metrics_reporter_) {
        bool                         cancelled = statusInfo().code() == ErrorCode::CANCELLED;
        bool                         timeout   = statusInfo().code() == ErrorCode::GENERATE_TIMEOUT;
        RtpLLMStreamMetricsCollector collector;
        collector.qps               = true;
        collector.cancel_qps        = cancelled;
        collector.error_qps         = stopped() && !cancelled;
        collector.is_streaming_qps  = generate_input_->generate_config->is_streaming;
        collector.not_streaming_qps = !generate_input_->generate_config->is_streaming;
        if (finished() || cancelled || timeout) {
            collector.reuse_length           = initial_reuse_length_;
            collector.input_token_length     = inputLength();
            collector.output_token_length    = outputTokenLen();
            collector.iterate_count          = iter_count_;
            collector.query_batch_size       = maxBatchSize();
            collector.total_latency_us       = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
            collector.first_token_latency_us = complete_token_ids_->firstTokenLatencyUs();
            RTP_LLM_LOG_DEBUG(
                "stream [%ld] report first latency us = %ld", streamId(), collector.first_token_latency_us);
            collector.wait_latency_us          = wait_time_us_;
            collector.pause_latency_us         = pause_time_us_;
            collector.fallback_tokens          = fallback_blocks_ * seqSizePerBlock();
            collector.fallback_times           = fallback_times_;
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
                 << ", reuse_length:" << reuse_length_ << ", current_chunk_len:" << current_chunk_len_
                 << ", last_chunk_len_:" << last_chunk_len_ << ", max_chunk_len_:" << max_chunk_len_
                 << ", current_batch_size:" << currentBatchSize() << ", next_batch_size:" << nextBatchSize()
                 << ", need_release_resource: " << need_release_resource_
                 << ", fallback_prefix_length: " << fallback_prefix_length_
                 << ", sp_edit_search_index: " << sp_edit_search_index_ << ", mtp token indices" << mtp_token_index_
                 << ", need_remote_generate: " << need_remote_generate_
                 << ", contain_propose_token: " << contain_propose_token_ << ", propose_token: " << propose_token_;

    for (int i = 0; i < propose_token_.size(); i++) {
        debug_string << propose_token_[i] << " ";
    }
    if (last_hidden_states_) {
        debug_string << ", hidden_state_token_num: " << last_hidden_states_->shape()[0];
    }
    debug_string << ", complete_token_ids: [";
    for (size_t i = 0; i < complete_token_ids_->batchSize(); i++) {
        debug_string << complete_token_ids_->toString(i) << ",";
    }

    debug_string << ", cum_log_probs: " << cum_log_probs_->debugStringWithData<float>();
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
    complete_token_ids_ = make_shared<CompleteTokenIds>(*other_stream.complete_token_ids_, share);
    cum_log_probs_      = device_->clone({*other_stream.cum_log_probs_, rtp_llm::AllocationType::HOST});
    if (other_stream.calculateLoss() && copy_loss) {
        loss_ = device_->clone({*other_stream.loss_, rtp_llm::AllocationType::HOST});
    } else {
        loss_ = nullptr;
    }
}

GenerateStream::TimeInfo GenerateStream::getTimeInfo() {
    return {begin_time_us_,
            wait_time_us_,
            complete_token_ids_->firstTokenTimeUs(),
            complete_token_ids_->firstTokenLatencyUs()};
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
        sub_generate_status_[i].status = state;
    }
}

void GenerateStream::resizeSubGenerateStatus(size_t new_size) {
    if (sub_generate_status_.size() != new_size) {
        size_t old_size = sub_generate_status_.size();
        sub_generate_status_.resize(new_size);
        for (size_t i = old_size; i < new_size; ++i) {
            sub_generate_status_[i].status = StreamState::RUNNING;
        }
    }
}

}  // namespace rtp_llm
