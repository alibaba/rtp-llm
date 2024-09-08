#include <cstddef>
#include <memory>
#include "autil/EnvUtil.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/utils/assert_utils.h"

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
    , metrics_reporter_(metrics_reporter)
    , special_tokens_(params.special_tokens_)
    , output_mutex_(std::make_shared<std::mutex>())
    , generate_outputs_queue_(std::make_shared<autil::SynchronizedQueue<GenerateOutputs>>())

{
    updatePrefix(resource_context.system_prompt);
    seq_length_ = generate_input_->inputLength();
    start_check_seq_length_ = seq_length_;
    last_output_pos_ = seq_length_;
    common_len_ = seq_length_;
    max_chunk_len_ = seq_length_;

    begin_time_us_      = input->begin_time_ms;
    device_             = ft::DeviceFactory::getDefaultDevice();
    complete_token_ids_ = device_->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)tileNum(), (size_t)max_seq_len_}, ft::AllocationType::HOST}, {});
    if (generate_input_->generate_config->calculate_loss && inputLength() > 1) {
        loss_ = device_->allocateBuffer(
                {ft::DataType::TYPE_FP32, {(size_t)inputLength() - 1}, ft::AllocationType::HOST}, {});
    }
    memset(complete_token_ids_->data(), 0, complete_token_ids_->sizeBytes());
    for (int i = 0; i < tileNum(); ++i) {
        memcpy(complete_token_ids_->dataWithOffset<int32_t>(i * max_seq_len_),
               generate_input_->input_ids->data(),
               generate_input_->input_ids->sizeBytes());
    }

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
    if (need_release_resource_) {
        stream_cache_resource_.releaseResource();
    }
}
int GenerateStream::nextNeedBlockNums(size_t reserve_step) const {
    // TODO: maybe need fix when context and reuse
    return stream_cache_resource_.singleBatchNeedBlocks(seq_length_ + reserve_step) * batchSize();
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
    return seq_length_ == inputLength() && !perf_test_ ? 1 : tileNum();
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
    return tileNum() == 1 ? seq_length_ : inputLength() / seqSizePerBlock() * seqSizePerBlock();
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

bool GenerateStream::checkTokenId(int token_id) {
    return token_id >= 0 && token_id < vocab_size_;
}

void GenerateStream::setStop(const std::string& err_msg, absl::StatusCode err_code) {
    std::lock_guard<std::mutex> lock(*output_mutex_);
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

bool GenerateStream::paused() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_.status == GenerateState::PAUSED;
}

std::string GenerateStream::stopReason() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return generate_status_.error_info;
}

bool GenerateStream::finished() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
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
    for (size_t i = start_check_seq_length_; i <= seq_length_; ++i) {
        if (special_tokens_.eos_token_id_ == token_ids_[i - 1]) {
            sub_generate_status_[batch_id].status = GenerateState::FINISHED;
            seq_length_ = i;
        }
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
        for (size_t i = start_check_seq_length_; i <= seq_length_; ++i) {
            bool match_one   = true;
            size_t begin_index = i - stop_words.size();
            for (auto& token : stop_words) {
                if (token != token_ids_[begin_index++]) {
                    match_one = false;
                    break;
                }
            }

            if (match_one) {
                match = match_one;
                seq_length_ = i;
                break;
            }
        }

        if (match) {
            break;
        }
    }
    if (match) {
        sub_generate_status_[batch_id].status = GenerateState::FINISHED;
    }
}

void GenerateStream::update(const ft::BufferPtr&    new_tokens,
                            int               num_new_tokens,
                            const ft::BufferPtr& hidden_states,
                            const ft::BufferPtr& logits,
                            const ft::BufferPtr& cum_log_probs,
                            const ft::BufferPtr& all_probs) {
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
    for (size_t i = 0; i < tileNum(); ++i) {
        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = *(*new_tokens)[i].dataWithOffset<int>(j);
            if (!checkTokenId(current_token_id)) {
                setStop("output token id:" + to_string(current_token_id) + " out of vocab size: " + to_string(vocab_size_));
                return;
            }
        }
        
        device_->copy({(*complete_token_ids_)[i].view(seq_length_, num_new_tokens), (*new_tokens)[i].view(0, num_new_tokens)});
    }
    setSeqLength(seq_length_ + num_new_tokens);

    updateOutput(new_tokens, hidden_states, logits, cum_log_probs, all_probs);
}

void GenerateStream::setLoss(const ft::Buffer& loss) {
    FT_CHECK(loss_index_ + loss.size() < inputLength());
    device_->copy({loss_->view(loss_index_, loss.size()), loss});
    loss_index_ += loss.size();
}

ft::BufferPtr GenerateStream::getLoss() {
    return loss_;
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
                 << ", tile_num:" << tileNum()
                 << ", need_release_resource: " << need_release_resource_
                 << ", fallback_prefix_length: " << fallback_prefix_length_;

    debug_string << ", complete_token_ids: [";
    for (size_t i = 0; i < tileNum(); i++) {
        debug_string << (*complete_token_ids_)[i].view(0, seq_length_).debugStringWithData<int32_t>() << ",";
    }

    debug_string << ", cum_log_probs: " << cum_log_probs_->debugStringWithData<float>();
    debug_string << ", stream_cache_resource: "<< stream_cache_resource_.debugString();
         
    debug_string << "}";
    return debug_string.str();
}

void GenerateStream::resetCommonLen() {
    if (tileNum() == 1) {
        common_len_ = seq_length_;
    }
}

void GenerateStream::setSeqLength(int seq_length) {
    if (seq_length > seq_length_) {
        start_check_seq_length_ = seq_length_ + 1;
    } else {
        start_check_seq_length_ = seq_length;
    }
    seq_length_ = seq_length;
    resetCommonLen();
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

void GenerateStream::CopyOnWrite(const GenerateStream& other_stream) {
    complete_token_ids_ = device_->clone({*other_stream.complete_token_ids_, ft::AllocationType::HOST});
    cum_log_probs_ = device_->clone({*other_stream.cum_log_probs_, ft::AllocationType::HOST});
    if (other_stream.calculateLoss()) {
        loss_ = device_->clone({*other_stream.loss_, ft::AllocationType::HOST});
    }
    stream_cache_resource_.setStream(this);
}

}  // namespace rtp_llm
