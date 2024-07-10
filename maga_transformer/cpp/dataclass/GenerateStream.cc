#include <atomic>
#include <memory>
#include "autil/EnvUtil.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

using namespace std;

namespace rtp_llm {

GenerateStream::GenerateStream(const shared_ptr<GenerateInput>& input,
                               const ft::GptInitParameter&      params,
                               const ResourceContext&           resource_context,
                               kmonitor::MetricsReporterPtr     metrics_reporter)
    : generate_input_(input)
    , max_seq_len_(params.max_seq_len_)
    , stream_cache_resource_(this, resource_context, input->need_release_resource)
    , need_release_resource_(input->need_release_resource)
    , enable_fast_gen_(params.enable_fast_gen_)
    , metrics_reporter_(metrics_reporter)
    , special_tokens_(params.special_tokens_)
{
    begin_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();

    updatePrefix(resource_context.system_prompt);
    seq_length_ = generate_input_->inputLength();
    last_output_pos_ = seq_length_;
    common_len_ = seq_length_;
    adjusted_common_len_ = tileNum() == 1 ? seq_length_ : seq_length_ / seqSizePerBlock() * seqSizePerBlock();
    max_chunk_len_ = seq_length_;

    
    begin_time_us_      = input->begin_time_ms;
    device_             = ft::DeviceFactory::getDefaultDevice();
    complete_token_ids_ = device_->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)tileNum(), (size_t)max_seq_len_}, ft::AllocationType::HOST}, {});
    memset(complete_token_ids_->data(), 0, complete_token_ids_->sizeBytes());
    for (int i = 0; i < tileNum(); ++i) {
        memcpy(complete_token_ids_->dataWithOffset<int32_t>(i * max_seq_len_),
               generate_input_->input_ids->data(),
               generate_input_->input_ids->sizeBytes());
    }

    generate_outputs_queue_.setCapacity(1000);
    cum_log_probs_ =
        device_->allocateBuffer({ft::DataType::TYPE_FP32, {(size_t)tileNum()}, ft::AllocationType::HOST}, {});
    memset(cum_log_probs_->data(), 0, cum_log_probs_->sizeBytes());

    generate_outputs_             = make_shared<GenerateOutputs>();
    generate_outputs_->request_id = generate_input_->request_id;

    generate_status_.status = GenerateState::WAITING;
    sub_generate_status_.clear();
    sub_generate_status_.resize(tileNum());
    for (int i = 0; i < tileNum(); ++i) {
        sub_generate_status_[i].status = GenerateState::WAITING;
    }

    stream_cache_resource_.init(tileNum());

    perf_test_ = autil::EnvUtil::getEnv("PERF_TEST", false);
    // TODO: need fix context block copy
    perf_test_ = true;
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
    cancelled_ = true;
    setStop("cancel stream");
}

absl::StatusOr<GenerateOutputs> GenerateStream::nextOutput() {
    // TODO(xinfei.sxf) 某些case下会出现1s的等待
    while ((!stopped()) && !finished() && generate_outputs_queue_.isEmpty()) {
        generate_outputs_queue_.waitNotEmpty();
    }
    if (stopped()) {
        std::lock_guard<std::mutex> lock(output_mutex_);
        return absl::Status(generate_status_.error_code, generate_status_.error_info);
    }
    if (generate_outputs_queue_.isEmpty()) {
        return absl::InternalError("no output any more");
    }
    return generate_outputs_queue_.getAndPopFront();
}

absl::StatusOr<int> GenerateStream::initKVBlock(int token_capacity) {
    if (generate_status_.status == GenerateState::WAITING) {
        wait_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    } else if (generate_status_.status == GenerateState::PAUSED) {
        pause_time_us_ += autil::TimeUtility::currentTimeInMicroSeconds() - last_pause_us_;
    }
    return stream_cache_resource_.initKVBlock(token_capacity);
}

absl::StatusOr<int>GenerateStream::incrKVBlock(int token_capacity) {
    return stream_cache_resource_.incrKVBlock(token_capacity);
}

int GenerateStream::tryReleaseKVBlock(int nums) {
    auto release_blocks = stream_cache_resource_.tryReleaseKVBlock(nums);
    incrFallbackBlock(release_blocks);
    return release_blocks;
}
void GenerateStream::releaseResource() {
    if (need_release_resource_) {
        stream_cache_resource_.releaseResource();
    }
}
int GenerateStream::nextNeedBlockNums() const {
    // TODO: maybe need fix when context and reuse
    return stream_cache_resource_.singleBatchNeedBlocks(seq_length_) * batchSize();
}

void GenerateStream::incrFallbackBlock(int fallback_blocks) {
    fallback_blocks_ += fallback_blocks;
    fallback_times_ += 1;
}

std::shared_ptr<GenerateInput> GenerateStream::generateInput() const {
    return generate_input_;
}
std::shared_ptr<GenerateConfig>& GenerateStream::generateConfig() {
    return generate_input_->generate_config;
}
bool GenerateStream::isStreaming() const {
    return generate_input_->generate_config->is_streaming;
}
int64_t GenerateStream::streamId() const {
    return generate_input_->request_id;
}
int GenerateStream::loraId() const {
    return generate_input_->lora_id;
}
ft::SpecialTokens GenerateStream::specialTokens() const {
    return special_tokens_;
}

int GenerateStream::tileNum() const {
    return std::max(numBeams(), numReturnSequences());
}

int GenerateStream::batchSize() const {
    return seq_length_ == inputLength() && !perf_test_ ? 1 : tileNum();
}

int GenerateStream::numBeams() const {
    return generate_input_->generate_config->num_beams;
}

int GenerateStream::numReturnSequences() const {
    return generate_input_->generate_config->num_return_sequences;
}

void GenerateStream::updatePrefix(const std::shared_ptr<SystemPrompt>& system_prompt) {
    if (system_prompt) {
        prompt_param_ = system_prompt->getPromptParams(*generate_input_->generate_config);
        if (!prompt_param_.prompt_token.empty()) {
            generate_input_->updatePrefix(prompt_param_.prompt_token);
        }
    }
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
    return seq_length_;
}

int GenerateStream::commonLen() const {
    return common_len_;
}

int GenerateStream::adjustedCommonLen() const {
    return adjusted_common_len_;
}

int GenerateStream::seqSizePerBlock() const {
    return stream_cache_resource_.seqSizePerBlock();
}

int GenerateStream::contextLength() const {
    int begin_pos = prefixLength();
    int end_pos = isChunkStream() ? currentChunkLen() : seq_length_;
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
    if (multimodalFeatures().has_value()) {
        // prompt with multimodal input cannot use reuse cache for now
        return 0;
    }
    return reuse_length_;
}

void GenerateStream::setReuseLength(int reuse_length) {
    reuse_length_ = reuse_length;
}

int GenerateStream::fallbackPrefixLength() const {
    return fallback_prefix_length_;
}

void GenerateStream::setFallbackPrefixLength(int fallback_prefix_length) {
    fallback_prefix_length_ = fallback_prefix_length;
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
    return complete_token_ids_;
}

std::vector<int> GenerateStream::completeTokenIdsVec(int batch_idx) {
    FT_CHECK(batch_idx < tileNum());
    return fastertransformer::buffer2vector<int>(complete_token_ids_->view(batch_idx, 1), seq_length_);
}

std::vector<int> GenerateStream::commonCompleteTokenIdsVec(int batch_idx) {
    FT_CHECK(batch_idx < tileNum());
    return fastertransformer::buffer2vector<int>(complete_token_ids_->view(batch_idx, 1), common_len_);
}

int GenerateStream::currentExecuteTokenSize() {
    return currentExecuteTokens(0).size() * batchSize();
}

vector<int> GenerateStream::contextTokens(int batch_idx) const {
    return fastertransformer::buffer2vector<int>(
            (*complete_token_ids_)[batch_idx].view(prefixLength(), contextLength()));
}

std::optional<std::vector<torch::Tensor>>& GenerateStream::multimodalFeatures() const {
    return generate_input_->multimodal_features;
}

int GenerateStream::multimodalFeaturesLength() const {
    auto& features = multimodalFeatures();
    if (features) {
        return features.value().size() * batchSize();
    }
    return 0;
}

std::optional<ft::BufferPtr>& GenerateStream::multimodalLocations() const {
    return generate_input_->mm_locs;
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

vector<int> GenerateStream::currentExecuteTokens(int batch_idx) const {
    // TODO(xinfei.sxf) 在query部分回退，重运行case下，这个不对
    if (isContextStream()) {
        return contextTokens(batch_idx);
    } else {
        return {*(*complete_token_ids_)[batch_idx].dataWithOffset<int>(seq_length_ - 1)};
    }
}

void GenerateStream::step() {
    // iter_count represents the times of the stream participates in running
    iter_count_++;
    if (isContextStream()) {
        setFallbackPrefixLength(0);
    }
}

void GenerateStream::checkTimeout() {
    auto running_time_ms = (autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_) / 1000;
    auto timeout_ms      = generate_input_->generate_config->timeout_ms;
    if (timeout_ms > 0 && timeout_ms < running_time_ms) {
        stopAndRelease("query has been running " + std::to_string(running_time_ms) + " ms, "
                       + "timeout_ms = " + std::to_string(timeout_ms) + ", it's timeout",
                       absl::StatusCode::kDeadlineExceeded);
    }
}

void GenerateStream::setStop(const std::string& err_msg, absl::StatusCode err_code) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    FT_LOG_WARNING("stop stream: %d %s", streamId(), err_msg.c_str());
    generate_status_.status     = GenerateState::STOPPED;
    generate_status_.error_code = err_code;
    generate_status_.error_info = err_msg;
}

void GenerateStream::stopAndRelease(const std::string& err_msg, absl::StatusCode err_code) {
    setStop(err_msg, err_code);
    releaseResource();
}

void GenerateStream::setPaused() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (stoppedWithoutLock()) {
        return;
    }
    is_context_stream_      = true;
    generate_status_.status = GenerateState::PAUSED;
    last_pause_us_          = autil::TimeUtility::currentTimeInMicroSeconds();
}

bool GenerateStream::setRunning() {
    std::lock_guard<std::mutex> lock(output_mutex_);
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
    std::lock_guard<std::mutex> lock(output_mutex_);
    return generate_status_.status == GenerateState::STOPPED;
}

bool GenerateStream::paused() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    return generate_status_.status == GenerateState::PAUSED;
}

std::string GenerateStream::stopReason() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    return generate_status_.error_info;
}

bool GenerateStream::finished() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    return generate_status_.status == GenerateState::FINISHED;
}

size_t GenerateStream::iterCount() const {
    return iter_count_;
}

void GenerateStream::setKVCache(const BatchKVCacheBlockAddr& kv_cache_block_addr) {
    stream_cache_resource_.setKVCache(kv_cache_block_addr);
}

const BatchKVCacheBlockAddr& GenerateStream::kvCache() const {
    return stream_cache_resource_.kvCache();
}

const ResourceContext& GenerateStream::resourceContext() const {
    return stream_cache_resource_.resourceContext();
}

size_t GenerateStream::maxBlockSize() const {
    return stream_cache_resource_.maxBlockSize();
}

bool GenerateStream::needFinish() {
    return seq_length_ >= std::min(max_seq_len_,
                                   generate_input_->generate_config->max_new_tokens + generate_input_->inputLength())
           || needFinishBySPTokens();
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
    int* token_ids_ = (int*)complete_token_ids_->view(batch_id, 1).data();
    if (special_tokens_.eos_token_id_ == token_ids_[seq_length_ - 1]) {
        sub_generate_status_[batch_id].status = GenerateState::FINISHED;
    }
}
void GenerateStream::matchStopWordsList() {
    if (seq_length_ < generate_input_->generate_config->min_new_tokens + inputLength()) {
        return;
    }
    if (seq_length_ == inputLength()) {
        return;
    }
    for (int i = 0; i < tileNum(); ++i) {
        matchStopWordsList(i);
    }
}

void GenerateStream::matchStopWordsList(int batch_id) {
    int* token_ids_ = (int*)complete_token_ids_->view(batch_id, 1).data();
    // note: stop_words_list in generate_config contains stop_words_list in special_tokens
    bool match = false;
    for (auto& stop_words : generate_input_->generate_config->stop_words_list) {
        bool   match_one   = true;
        size_t begin_index = seq_length_ - stop_words.size();
        for (auto& token : stop_words) {
            if (token != token_ids_[begin_index++]) {
                match_one = false;
                break;
            }
        }
        match = match_one;
        if (match) {
            break;
        }
    }
    if (match) {
        sub_generate_status_[batch_id].status = GenerateState::FINISHED;
    }
}

void GenerateStream::update(ft::BufferPtr&    new_tokens,
                            int               num_new_tokens,
                            const ft::Buffer& hidden_states,
                            const ft::Buffer& logits,
                            const ft::Buffer& cum_log_probs,
                            bool              not_update_output) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    is_context_stream_ = false;
    if (stoppedWithoutLock()) {
        return;
    }
    if (seq_length_ == generate_input_->inputLength()) {
        first_token_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    }
    // # NOTE: new tokens indicate num of newly genearted tokens
    // # typically 1 but can be > 1 under speculative decoding
    // # This differs from new_tokens.shape[-1] under beam search case,
    // # which needs to update all the generated tokens each update.
    FT_CHECK(new_tokens->dim() == 2);
    for (int i = 0; i < tileNum(); ++i) {
        *(*complete_token_ids_)[i].dataWithOffset<int>(seq_length_) = ((int*)new_tokens->data())[i];
    }
    setSeqLength(seq_length_ + num_new_tokens);
    bool finished = needFinish();
    if (finished) {
        setFinishedWithoutLock();
    }
    if (not_update_output) {
        return;
    }

    if (isStreaming() || finished) {
        updateOutput(hidden_states, logits, cum_log_probs);
    }
}

void GenerateStream::updateOutput(const ft::Buffer& hidden_states,
                                  const ft::Buffer& logits,
                                  const ft::Buffer& cum_log_probs) {
    device_->copy({*cum_log_probs_, cum_log_probs});

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t output_len = seq_length_ - last_output_pos_;
    generate_outputs_->generate_outputs.clear();
    for (int i = 0; i < tileNum(); i++) {
        GenerateOutput generate_output;
        generate_output.aux_info.iter_count = iter_count_;
        generate_output.aux_info.fallback_tokens = fallback_blocks_ * seqSizePerBlock();
        generate_output.aux_info.fallback_times = fallback_times_;

        generate_output.output_ids =
            device_->allocateBuffer({ft::DataType::TYPE_INT32, {1lu, output_len}, ft::AllocationType::HOST}, {});
        // TODO(xinfei.sxf) optimize this copy : only copy last token
        memcpy(generate_output.output_ids->data(),
               complete_token_ids_->view(i, 1).dataWithOffset<int32_t>(last_output_pos_),
               sizeof(int32_t) * output_len);
        if (generate_input_->generate_config->return_logits) {
            ft::BufferPtr host_logits;
            if (logits.shape()[0] == 1) {
                host_logits = device_->clone({logits, ft::AllocationType::HOST});
            } else {
                host_logits = device_->clone({logits.view(i, 1), ft::AllocationType::HOST});
            }
            if (!generate_input_->generate_config->select_tokens_id.empty()) {
                auto select_buf        = ft::vector2Buffer(generate_input_->generate_config->select_tokens_id);
                generate_output.logits = device_->select({*host_logits, *select_buf, 1});
            } else {
                // TODO(xinfei.sxf) not set logits in middle step for streaming
                generate_output.logits = host_logits;
            }
        }

        if (generate_input_->generate_config->return_hidden_states) {
            if (hidden_states.shape()[0] == 1) {
                generate_output.hidden_states = device_->clone({hidden_states, ft::AllocationType::HOST});
            } else {
                generate_output.hidden_states = device_->clone({hidden_states.view(i, 1), ft::AllocationType::HOST});
            }
        }
        if (generate_input_->generate_config->calculate_loss == 1) {
            auto x = device_->allocateBuffer({ft::DataType::TYPE_FP32, {1}, ft::AllocationType::HOST}, {});
            // TODO(xinfei.sxf) fix this loss
            // *((float*)x->data()) = 1.0f;
            generate_output.loss = std::move(x);
        }

        generate_output.finished              = sub_generate_status_[i].status == GenerateState::FINISHED;
        generate_output.aux_info.cost_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        generate_output.aux_info.input_len    = generate_input_->promptLength();
        generate_output.aux_info.prefix_len   = generate_input_->prefix_length;
        // TODO(xinfei.sxf) 提前结束的query，output len要设置正确
        generate_output.aux_info.output_len   = seq_length_ - generate_input_->inputLength();
        generate_output.aux_info.reuse_len    = reuse_length_;

        generate_output.aux_info.cum_log_probs =
            device_->allocateBuffer({ft::DataType::TYPE_FP32, {1lu}, ft::AllocationType::HOST}, {});
        memcpy(generate_output.aux_info.cum_log_probs.value()->data(),
               cum_log_probs_->dataWithOffset<float>(i),
               sizeof(float));

        generate_outputs_->generate_outputs.push_back(generate_output);
    }
    if (generate_outputs_queue_.getSize() >= generate_outputs_queue_.getCapacity()) {
        /* No matter if the queue is full for any reason, 
           the stream will be set to stop directly to prevent the push to queue from getting stuck. */
        setStop("queue is full");
        return;
    } else {
        generate_outputs_queue_.push(*generate_outputs_);
    }
    last_output_pos_ = seq_length_;
}

void GenerateStream::setMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter) {
    metrics_reporter_ = metrics_reporter;
}
void GenerateStream::reportMetric() {
    if (metrics_reporter_) {
        RtpLLMStreamMetricsCollector collector;
        collector.qps        = finished() || cancelled_;
        collector.cancel_qps = cancelled_;
        collector.error_qps  = stopped() && !cancelled_;
        if (finished() || cancelled_) {
            collector.reuse_length           = reuse_length_;
            collector.input_token_length     = inputLength();
            collector.output_token_length    = seq_length_ - generate_input_->inputLength();
            collector.iterate_count          = iter_count_;
            collector.query_batch_size       = tileNum();
            collector.total_latency_us       = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
            collector.first_token_latency_us = first_token_time_us_;
            collector.wait_latency_us        = wait_time_us_;
            collector.pause_latency_us       = pause_time_us_;
            collector.fallback_tokens        = fallback_blocks_ * seqSizePerBlock();
            collector.fallback_times         = fallback_times_;
        }
        metrics_reporter_->report<RtpLLMStreamMetrics, RtpLLMStreamMetricsCollector>(nullptr, &collector);
    }
}

std::string GenerateStream::debugString() const {
    std::stringstream debug_string;
    debug_string << "GenerateStream {"
                 << "generate_input:" << generate_input_->debugString() << ", max_seq_len:" << max_seq_len_
                 << ", input_length:" << inputLength() << ", seq_length:" << seq_length_
                 << ", reuse_length:" << reuse_length_ << ", current_chunk_len:" << current_chunk_len_
                 << ", last_chunk_len_:" << last_chunk_len_ << ", max_chunk_len_:" << max_chunk_len_
                 << ", batch_size:" << batchSize()
                 << ", tile_num:" << tileNum() << "}";
    return debug_string.str();
}

void GenerateStream::resetCommonLen() {
    if (tileNum() == 1) {
        common_len_ = seq_length_;
        adjusted_common_len_ = seq_length_;
    }
}

void GenerateStream::setSeqLength(int seq_length) {
    seq_length_ = seq_length;
    resetCommonLen();
}

void GenerateStream::setIsContextStream(bool is_context_stream) {
    is_context_stream_ = is_context_stream;
}

StreamCacheResource& GenerateStream::streamCacheResource() {
    return stream_cache_resource_;
}

}  // namespace rtp_llm
