#include <cstddef>
#include <memory>
#include "autil/EnvUtil.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

using namespace std;

namespace rtp_llm {

GenerateStream::GenerateStream(const shared_ptr<GenerateInput>& input,
                               const ft::GptInitParameter&      params,
                               const ResourceContext&           resource_context,
                               kmonitor::MetricsReporterPtr     metrics_reporter)
    : generate_input_(input)
    , max_seq_len_(params.max_seq_len_)
    , vocab_size_(params.vocab_size_)
    , stream_cache_resource_(this, resource_context, input->need_release_resource)
    , need_release_resource_(input->need_release_resource)
    , enable_fast_gen_(params.enable_fast_gen_)
    , use_cache_store_(params.use_cache_store_)
    , metrics_reporter_(metrics_reporter)
    , special_tokens_(params.special_tokens_)
    , output_mutex_(std::make_shared<std::mutex>())
    , mm_position_ids_style_(PositionIdsStyle(params.mm_position_ids_style_))
{
    if (!updatePrefix(resource_context.system_prompt)) {
        return;
    }

    begin_time_us_      = input->begin_time_us;
    device_             = ft::DeviceFactory::getDefaultDevice();
    if (generate_input_->generate_config->calculate_loss && inputLength() > 1) {
        loss_ = device_->allocateBuffer(
                {ft::DataType::TYPE_FP32, {(size_t)inputLength() - 1}, ft::AllocationType::HOST}, {});
    }
    if (generate_input_->generate_config->return_softmax_probs) {
        softmax_probs_ = device_->allocateBuffer(
                {ft::DataType::TYPE_FP32, {(size_t)tileNum(), (size_t)max_seq_len_}, ft::AllocationType::HOST}, {});
        memset(softmax_probs_->data(), 0, softmax_probs_->sizeBytes());
    }
    complete_token_ids_ = std::make_shared<CompleteTokenIds>(device_, tileNum(), max_seq_len_, params.seq_size_per_block_);
    complete_token_ids_->init(input);

    last_output_pos_ = seqLength();
    max_chunk_len_ = seqLength();

    cum_log_probs_ =
        device_->allocateBuffer({ft::DataType::TYPE_FP32, {(size_t)tileNum()}, ft::AllocationType::HOST}, {});
    memset(cum_log_probs_->data(), 0, cum_log_probs_->sizeBytes());

    generate_status_.status = GenerateState::WAITING;
    sub_generate_status_.clear();
    sub_generate_status_.resize(tileNum());
    for (int i = 0; i < tileNum(); ++i) {
        sub_generate_status_[i].status = GenerateState::WAITING;
    }

    stream_cache_resource_.init(tileNum());

    perf_test_ = autil::EnvUtil::getEnv("PERF_TEST", false);
    // TODO(xinfei.sxf): need fix context block copy
    perf_test_ = true;

    setReturnAllProbs(generate_input_->generate_config->return_all_probs);
}

void GenerateStream::resetBeginTime(int64_t begin_time_us) {
    begin_time_us_ = begin_time_us;
}

bool GenerateStream::hasCacheKeys() const {
    return stream_cache_resource_.hasCacheKeys();
}

const std::vector<int64_t>& GenerateStream::cacheKeys(int32_t batch_id) const {
    return stream_cache_resource_.cacheKeys(batch_id);
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
        last_chunk_len_ = current_chunk_len_;
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
    FT_CHECK(false);
    return absl::InternalError("unexpected call");
}

void GenerateStream::cancel() {
    setStop(ErrorCode::CANCELLED, "cancel stream");
}

absl::StatusOr<int> GenerateStream::initKVBlock(int token_capacity, size_t reserve_step) {
    if (generate_status_.status == GenerateState::WAITING) {
        wait_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    } else if (generate_status_.status == GenerateState::PAUSED) {
        pause_time_us_ += autil::TimeUtility::currentTimeInMicroSeconds() - last_pause_us_;
    }
    return stream_cache_resource_.initKVBlock(token_capacity, reserve_step);
}

absl::StatusOr<int>GenerateStream::incrKVBlock(int token_capacity, size_t reserve_step) {
    return stream_cache_resource_.incrKVBlock(token_capacity, reserve_step);
}

int GenerateStream::tryReleaseKVBlock(int nums) {
    auto release_blocks = stream_cache_resource_.tryReleaseKVBlock(nums);
    incrFallbackBlock(release_blocks);
    return release_blocks;
}

void GenerateStream::releaseResource() {
    stream_cache_resource_.releaseResource();
}
void GenerateStream::setNeedReleaseResource(bool need_release_resource) {
    need_release_resource_ = need_release_resource;
    stream_cache_resource_.setNeedReleaseResource(need_release_resource);
}
int GenerateStream::nextNeedBlockNums(size_t reserve_step) const {
    // TODO: maybe need fix when context and reuse
    return stream_cache_resource_.singleBatchNeedBlocks(seqLength() + reserve_step) * batchSize();
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
    if (numBeams() > 1) {
        return false;
    }
    return generate_input_->generate_config->is_streaming;
}
int64_t GenerateStream::streamId() const {
    return generate_input_->request_id;
}
int GenerateStream::loraId() const {
    return generate_input_->lora_id;
}

std::string GenerateStream::adapterName() const {
    return generate_input_->generate_config->adapter_name;
}

ft::SpecialTokens GenerateStream::specialTokens() const {
    return special_tokens_;
}

int GenerateStream::tileNum() const {
    return std::max(numBeams(), numReturnSequences());
}

int GenerateStream::batchSize() const {
    return seqLength() == inputLength() && !perf_test_ ? 1 : tileNum();
}

int GenerateStream::numBeams() const {
    return generate_input_->generate_config->num_beams;
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

bool GenerateStream::updatePrefix(const std::shared_ptr<SystemPrompt>& system_prompt) {
    if (system_prompt) {
        auto prefix_param = system_prompt->getPromptParams(*generate_input_->generate_config);
        if (!prefix_param.prompt_tokens.empty()) {
            auto total_input_len = inputLength() + prefix_param.prompt_tokens.size();
            if (total_input_len >= max_seq_len_) {
                setStop(ErrorCode::LONG_PROMPT_ERROR, "after update prefix, total input len " + std::to_string(total_input_len)
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
    last_chunk_len_ = 0;
    current_chunk_len_ = chunk_len;
    max_chunk_len_ = max_chunk_len;
}

int GenerateStream::seqLength() const {
    return complete_token_ids_->seqLength();
}

int GenerateStream::adjustedCommonLen() const {
    return tileNum() == 1 ? seqLength() : inputLength() / seqSizePerBlock() * seqSizePerBlock();
}

int GenerateStream::seqSizePerBlock() const {
    return stream_cache_resource_.seqSizePerBlock();
}

int GenerateStream::contextLength() const {
    int begin_pos = prefixLength();
    int end_pos = isChunkStream() ? currentChunkLen() : seqLength();
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
    return is_context_stream_;
}

bool GenerateStream::isChunkStream() const {
    return enable_fast_gen_ && current_chunk_len_ < max_chunk_len_;
}

const ft::BufferPtr& GenerateStream::cumLogProbs() const {
    return cum_log_probs_;
}

const ft::BufferPtr& GenerateStream::completeTokenIds() {
    return complete_token_ids_->completeTokenIds();
}

std::vector<int> GenerateStream::completeTokenIdsVec(int batch_idx) {
    FT_CHECK(batch_idx < tileNum());
    return complete_token_ids_->completeTokenIdsVec(batch_idx);
}

std::vector<int> GenerateStream::commonCompleteTokenIdsVec(int batch_idx) {
    FT_CHECK(batch_idx < tileNum());
    return complete_token_ids_->commonCompleteTokenIdsVec(batch_idx);
}

int GenerateStream::currentExecuteTokenSize() {
    return currentExecuteTokens(0).size() * batchSize();
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
    return multimodalFeatures().size() * batchSize();
}

ft::BufferPtr GenerateStream::multimodalLocations() const {
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
    auto locs = generate_input_->mm_locs.value();
    auto features = generate_input_->multimodal_features.value();
    for (int i = 0;i < locs->size();++i) {
        res.emplace_back(vector<int>({*locs->dataWithOffset<int>(i), int(features[i].sizes()[0])}));
    }
    return res;
}

vector<int> GenerateStream::textTokensMask() const {
    if (!generate_input_->text_tokens_mask) {
        return {};
    }
    auto token_masks = fastertransformer::buffer2vector<int>(*generate_input_->text_tokens_mask.value());
    if (reuseLength() > 0) {
        return vector<int>(token_masks.begin() + reuseLength(), token_masks.end());
    } else {
        return token_masks;
    }
}

ft::BufferPtr GenerateStream::generateContextPositionIds(ft::DeviceBase* device) {
    optional<vector<ft::BufferPtr>> position_ids_buffer = nullopt;
    if (generate_input_->mm_position_ids.has_value()) {
        position_ids_buffer = ft::torchTensorVec2BufferVec(generate_input_->mm_position_ids.value());
    }
    context_position_ids_ = PositionIdsGenerator::generatePositionIds(device, generate_input_->inputLength(),
        mm_position_ids_style_, generate_input_->mm_locs, position_ids_buffer);
    return context_position_ids_.value();
}

void GenerateStream::generateNextPositionId(int32_t* now_pos) {
    if (!context_position_ids_) {
        return;
    }
    PositionIdsGenerator::generateNextPositionId(now_pos, seqLength(), mm_position_ids_style_, context_position_ids_.value());
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
    FT_LOG_WARNING("stop stream [%d], error msg: [%s], current state [%s], "
                    "input len [%d], seq len [%d], timeout [%ld] ms, running [%ld] ms",
                    streamId(), error_msg.c_str(), GenerateStateToString(generate_status_.status).c_str(),
                    inputLength(), seqLength(), getTimeoutMs(), cost_time_ms);
    generate_status_.status     = GenerateState::STOPPED;
    generate_status_.error_info = ErrorInfo(error_code, error_msg);
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
    return generate_status_.error_info;
}

void GenerateStream::setPaused() {
    // TODO(xinfei.sxf) fix mutex name
    std::lock_guard<std::mutex> lock(*output_mutex_);
    if (stoppedWithoutLock()) {
        return;
    }
    is_context_stream_      = true;
    generate_status_.status = GenerateState::PAUSED;
    last_pause_us_          = autil::TimeUtility::currentTimeInMicroSeconds();
}

bool GenerateStream::setRunning() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    if (stoppedWithoutLock()) {
        return false;
    }
    generate_status_.status = GenerateState::RUNNING;
    return true;
}

void GenerateStream::setFinishedWithoutLock() {
    generate_status_.status = GenerateState::FINISHED;
    for (int i = 0; i < tileNum(); ++i) {
        sub_generate_status_[i].status = GenerateState::FINISHED;
    }
}

bool GenerateStream::stoppedWithoutLock() {
    return generate_status_.status == GenerateState::STOPPED;
}

bool GenerateStream::stopped() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_.status == GenerateState::STOPPED;
}

bool GenerateStream::waiting() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_.status == GenerateState::WAITING;
}

bool GenerateStream::paused() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_.status == GenerateState::PAUSED;
}

std::string GenerateStream::stopReason() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_.error_info.ToString();
}

bool GenerateStream::finishedWithoutLock() {
    return generate_status_.status == GenerateState::FINISHED;
}

bool GenerateStream::running() {
    return generate_status_.status == GenerateState::RUNNING;
}

void GenerateStream::cancelIfNotRunning() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    if (generate_status_.status == GenerateState::WAITING
            || generate_status_.status == GenerateState::REMOTE_RUNNING) {
        auto cost_time_ms = (autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_) / 1000;
        FT_LOG_WARNING("stop stream: %d %s, input len [%d], seq len [%d], timeout: [%ld] ms, running [%ld] ms",
            streamId(), "cancel stream in waiting or remote running",
            inputLength(), seqLength(),
            getTimeoutMs(), cost_time_ms);
        generate_status_.status = GenerateState::STOPPED;
        generate_status_.error_info = ErrorInfo(ErrorCode::CANCELLED, "cancel stream in waiting or remote running");
    }
}

bool GenerateStream::finished() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_.status == GenerateState::FINISHED;
}

bool GenerateStream::needRemoteGenerate() const {
    return need_remote_generate_;
}

void GenerateStream::setRemoteGenerate() {
    generate_status_.status = GenerateState::REMOTE_RUNNING;
}

size_t GenerateStream::iterCount() const {
    return iter_count_;
}

void GenerateStream::setKVCache(const BatchKVCacheResource& kv_cache_resource) {
    stream_cache_resource_.setKVCache(kv_cache_resource);
}

const BatchKVCacheResource& GenerateStream::kvCache() const {
    return stream_cache_resource_.kvCache();
}

const ResourceContext& GenerateStream::resourceContext() const {
    return stream_cache_resource_.resourceContext();
}

size_t GenerateStream::maxBlockSize() const {
    return stream_cache_resource_.maxBlockSize();
}

size_t GenerateStream::maxTokenNum() const {
    return std::min(max_seq_len_,
        generate_input_->generate_config->max_new_tokens + generate_input_->inputLength());
}

bool GenerateStream::needFinish() {
    return seqLength() >= maxTokenNum() || needFinishBySPTokens();
}

bool GenerateStream::needFinishBySPTokens() {
    matchEosToken();
    matchStopWordsList();
    // num beams, finished by batch 0
    if (numBeams() != 1) {
        return sub_generate_status_[0].status == GenerateState::FINISHED;
    }
    // num sequence, finished by all batch
    return std::all_of(sub_generate_status_.begin(), sub_generate_status_.end(), [](GenerateStatus& generate_status) {
        return generate_status.status == GenerateState::FINISHED;
    });
}

void GenerateStream::matchEosToken() {
    for (int i = 0; i < tileNum(); ++i) {
        matchEosToken(i);
    }
}

void GenerateStream::matchEosToken(int batch_id) {
    if (complete_token_ids_->matchEosToken(batch_id, special_tokens_.eos_token_id_)) {
        sub_generate_status_[batch_id].status = GenerateState::FINISHED;
    }
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
    for (int i = 0; i < tileNum(); ++i) {
        matchStopWordsList(i);
    }
}

void GenerateStream::matchStopWordsList(int batch_id) {
    // note: stop_words_list in generate_config contains stop_words_list in special_tokens
    bool match = false;
    for (auto& stop_words : generate_input_->generate_config->stop_words_list) {
        if (complete_token_ids_->matchStopWordsList(batch_id, stop_words)) {
            match = true;
            break;
        }
    }
    if (match) {
        sub_generate_status_[batch_id].status = GenerateState::FINISHED;
    }
}

void GenerateStream::update(const StreamUpdateInfo& update_info) {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    FT_LOG_DEBUG("stream [%ld] update", streamId());
    is_context_stream_ = false;
    if (stoppedWithoutLock()) {
        return;
    }

    const auto& new_tokens = update_info.new_tokens;
    auto num_new_tokens = update_info.num_new_tokens;

    int error_token_id = 0;
    if (!complete_token_ids_->update(new_tokens, begin_time_us_, num_new_tokens, generate_input_->inputLength(), maxTokenNum(), vocab_size_, numBeams(), streamId(), error_token_id)) {
        setStopWithoutLock(ErrorCode::OUT_OF_VOCAB_RANGE,
                        "output token id:" + std::to_string(error_token_id) +
                        " out of vocab size: " + std::to_string(vocab_size_));
        return;
    }

    // TODO(xinfei.sxf) fix this (update_queue)
    updateOutput(update_info);
}


// void GenerateStream::update(const GptModelOutputs& model_outputs,
//                             SamplerOutput&   sampler_output)
// {
//     FT_LOG_DEBUG(__PRETTY_FUNCTION__);
// }


// beam_idx: [beam_width] int, the element must less than beam_width.
void GenerateStream::beamSearchKvCacheUpdate(ft::BufferPtr beam_idx) {
    auto beam_idx_vec = ft::buffer2vector<int>(*beam_idx);
    FT_CHECK(beam_idx_vec.size() == tileNum());

    stream_cache_resource_.beamSearchKvCacheUpdate(beam_idx_vec);
}



void GenerateStream::setLoss(const ft::Buffer& loss) {
    FT_CHECK(loss_index_ + loss.size() < inputLength());
    device_->copy({loss_->view(loss_index_, loss.size()), loss});
    loss_index_ += loss.size();
}

void GenerateStream::setSoftmaxProbs(const ft::Buffer& softmax_probs, int start_pos) {
    FT_CHECK(softmax_probs.dim() == 2);
    FT_CHECK(softmax_probs.shape()[0] == tileNum());
    for (int i = 0; i < tileNum(); ++i) {
        device_->copy({(*softmax_probs_)[i].view(start_pos, softmax_probs.shape()[1]), softmax_probs[i]});
    }
}

ft::BufferPtr GenerateStream::getLoss() {
    return loss_;
}

ft::BufferPtr GenerateStream::getSoftmaxProbs() {
    return softmax_probs_;
}

void GenerateStream::setMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter) {
    metrics_reporter_ = metrics_reporter;
}
void GenerateStream::reportMetric() {
    bool cancelled = statusInfo().code() == ErrorCode::CANCELLED;
    bool timeout = statusInfo().code() == ErrorCode::GENERATE_TIMEOUT;
    if (metrics_reporter_) {
        RtpLLMStreamMetricsCollector collector;
        collector.qps        = true;
        collector.cancel_qps = cancelled;
        collector.error_qps  = stopped() && !cancelled;
        if (finished() || cancelled || timeout) {
            collector.reuse_length           = reuse_length_;
            collector.input_token_length     = inputLength();
            collector.output_token_length    = outputTokenLen();
            collector.iterate_count          = iter_count_;
            collector.query_batch_size       = tileNum();
            collector.total_latency_us       = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
            collector.first_token_latency_us = complete_token_ids_->firstTokenLatencyUs();
            FT_LOG_DEBUG("stream [%ld] report first latency us = %ld", streamId(), collector.first_token_latency_us);
            collector.wait_latency_us        = wait_time_us_;
            collector.pause_latency_us       = pause_time_us_;
            collector.fallback_tokens        = fallback_blocks_ * seqSizePerBlock();
            collector.fallback_times         = fallback_times_;
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
                 << ", batch_size:" << batchSize()
                 << ", tile_num:" << tileNum()
                 << ", need_release_resource: " << need_release_resource_
                 << ", fallback_prefix_length: " << fallback_prefix_length_
                 << ", sp_edit_search_index: " << sp_edit_search_index_;

    debug_string << ", complete_token_ids: [";
    for (size_t i = 0; i < tileNum(); i++) {
        debug_string << complete_token_ids_->toString(i) << ",";
    }

    debug_string << ", cum_log_probs: " << cum_log_probs_->debugStringWithData<float>();
    debug_string << ", stream_cache_resource: "<< stream_cache_resource_.debugString();

    debug_string << "}";
    return debug_string.str();
}

int GenerateStream::reuseBlockSize() const {
    int reuse_length = reuseLength();
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
    is_context_stream_ = is_context_stream;
}

StreamCacheResource& GenerateStream::streamCacheResource() {
    return stream_cache_resource_;
}

void GenerateStream::CopyOnWrite(const GenerateStream& other_stream, bool copy_loss) {
    complete_token_ids_ = make_shared<CompleteTokenIds>(*other_stream.complete_token_ids_);
    cum_log_probs_ = device_->clone({*other_stream.cum_log_probs_, ft::AllocationType::HOST});
    if (other_stream.calculateLoss() && copy_loss) {
        loss_ = device_->clone({*other_stream.loss_, ft::AllocationType::HOST});
    } else {
        loss_ = nullptr;
    }
    stream_cache_resource_.setStream(this);
}

GenerateStream::TimeInfo GenerateStream::getTimeInfo() {
    return {begin_time_us_, complete_token_ids_->firstTokenTimeUs(), complete_token_ids_->firstTokenLatencyUs()};
}

bool GenerateStream::queryPdSep() const {
    return generate_input_->generate_config->pd_separation;
}

}  // namespace rtp_llm
