#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include <atomic>
#include <memory>

using namespace std;

namespace rtp_llm {

GenerateStream::GenerateStream(const shared_ptr<GenerateInput>& input, const ft::GptInitParameter& params,
                               const ResourceContext& resource_context, kmonitor::MetricsReporterPtr metrics_reporter) :
    generate_input_(input),
    max_seq_len_(params.max_seq_len_),
    stream_cache_resource_(this, resource_context, input->need_release_resource),
    need_release_resource_(input->need_release_resource),
    metrics_reporter_(metrics_reporter),
    special_tokens_(params.special_tokens_)
{
    updatePrefix(resource_context.system_prompt);
    seq_length_ = generate_input_->inputLength();

    begin_time_us_      = autil::TimeUtility::currentTimeInMicroSeconds();

    device_             = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    complete_token_ids_ = device_->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)tileNum(), (size_t)max_seq_len_}, ft::AllocationType::HOST}, {});
    memset(complete_token_ids_->data(), 0, complete_token_ids_->sizeBytes());
    for (int i = 0; i < tileNum(); ++i) {
        memcpy(complete_token_ids_->dataWithOffset<int32_t>(i * max_seq_len_), generate_input_->input_ids->data(), generate_input_->input_ids->sizeBytes());
    }

    cum_log_probs_ = device_->allocateBuffer(
        {ft::DataType::TYPE_FP32, {(size_t)tileNum()}, ft::AllocationType::HOST}, {});
    memset(cum_log_probs_->data(), 0, cum_log_probs_->sizeBytes());

    generate_outputs_ = make_shared<GenerateOutputs>();
    generate_outputs_->request_id = generate_input_->request_id;

    sub_generate_status_.clear();
    sub_generate_status_.resize(tileNum());
    for (int i = 0; i < tileNum(); ++i) {
        sub_generate_status_[i].status = GenerateState::WAITING;
    }
}

void GenerateStream::cancel() {
    cancelled_ = true;
    setStop("cancel stream");
}

absl::StatusOr<GenerateOutputs> GenerateStream::nextOutput() {
    while (generate_outputs_queue_.isEmpty() && !stopped() && !finished()) {
        generate_outputs_queue_.waitNotEmpty();
    }
    if (stopped()) {
        return absl::InternalError(stopReason());
    }
    if (generate_outputs_queue_.isEmpty()) {
        return absl::InternalError("no output any more");
    }
    return generate_outputs_queue_.getAndPopFront();
}

bool GenerateStream::initKVBlock() {
    if (generate_status_.status == GenerateState::WAITING) {
        wait_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    } else if (generate_status_.status == GenerateState::PAUSED) {
        pause_time_us_ += autil::TimeUtility::currentTimeInMicroSeconds() - last_pause_us_;
    }
    return stream_cache_resource_.initKVBlock();
}
bool GenerateStream::incrKVBlock() {
    return stream_cache_resource_.incrKVBlock();
}
int GenerateStream::tryReleaseKVBlock(int nums) {
    return stream_cache_resource_.tryReleaseKVBlock(nums);
}
void GenerateStream::releaseResource() {
    if (need_release_resource_) {
        stream_cache_resource_.releaseResource();
    }
}
int GenerateStream::nextNeedBlockNums() const {
    return stream_cache_resource_.needKVCacheBlockNums();
}
int GenerateStream::needKVCacheBlockNums() const {
    return stream_cache_resource_.needKVCacheBlockNums();
}

std::shared_ptr<GenerateInput> GenerateStream::generateInput() const {
    return generate_input_;
}
std::shared_ptr<GenerateConfig>& GenerateStream::generateConfig() {
    return generate_input_->generate_config;
}
std::optional<ft::BufferPtr>& GenerateStream::imageEmbeddings() {
    return generate_input_->image_embeddings;
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
    return tileNum();
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
int GenerateStream::seqLength() const {
    return seq_length_;
}
int GenerateStream::contextLength() const {
    return seq_length_- reuse_length_;
}
int GenerateStream::prefixLength() const {
    return generate_input_->prefix_length;
}
int GenerateStream::reuseLength() const {
    return reuse_length_;
}
void GenerateStream::setReuseLength(int reuse_length) {
    reuse_length_ = reuse_length;
}

bool GenerateStream::isContextStream() const {
    return is_context_stream_;
}
const ft::BufferPtr& GenerateStream::cumLogProbs() const {
    return cum_log_probs_;
}

const ft::BufferPtr& GenerateStream::completeTokenIds() {
    return complete_token_ids_;
}
std::vector<int> GenerateStream::completeTokenIdsVec(int batch_id) {
    assert(batch_id < tileNum());
    return fastertransformer::buffer2vector<int>(complete_token_ids_->view(batch_id, 1), seq_length_);
}
int GenerateStream::currentExecuteTokenSize() {
    return currentExecuteTokens().size();
}
vector<int> GenerateStream::contextTokens() const {
    auto input_tokens = fastertransformer::buffer2vector<int>({ft::MemoryType::MEMORY_CPU, ft::DataType::TYPE_INT32, {(size_t)seq_length_}, complete_token_ids_->data()});
    if (reuseLength() > 0) {
        return vector<int>(input_tokens.begin() + reuseLength(), input_tokens.end());
    } else {
        return input_tokens;
    }
}
vector<int> GenerateStream::currentExecuteTokens() const {
    // TODO(xinfei.sxf) 在query部分回退，重运行case下，这个不对
    if (isContextStream()) {
        return contextTokens();
    } else {
        int         tile_num = tileNum();
        vector<int> current_tokens;
        current_tokens.reserve(tile_num);
        int* token_ids = (int*)complete_token_ids_->data();
        for (int i = 0; i < tile_num; ++i) {
            current_tokens.push_back(token_ids[i * max_seq_len_ + seqLength() - 1]);
        }
        return current_tokens;
    }
}

void GenerateStream::checkTimeout() {
    auto running_time_ms = (autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_) / 1000;
    auto timeout_ms = generate_input_->generate_config->timeout_ms;
    if (timeout_ms > 0 && timeout_ms < running_time_ms) {
        stopAndRelease("query has been running " + std::to_string(running_time_ms) + " ms, "
            + "timeout_ms = " + std::to_string(timeout_ms) + ", it's timeout");
    }
}
void GenerateStream::setStop(const std::string& err_msg) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    FT_LOG_WARNING("stop stream: %d %s", streamId(), err_msg.c_str());
    generate_status_.status = GenerateState::STOPPED;
    generate_status_.error_info = err_msg;
}
void GenerateStream::stopAndRelease(const std::string& err_msg) {
    setStop(err_msg);
    releaseResource();
}
bool GenerateStream::isDoneWithoutLock(int batch_id) const {
    assert(batch_id < tileNum());
    if (sub_generate_status_[batch_id].status == GenerateState::STOPPED || sub_generate_status_[batch_id].status == GenerateState::FINISHED) {
        return true;
    }
    return false;
}
void GenerateStream::setPaused() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (stoppedWithoutLock()) {
        return ;
    }
    is_context_stream_ = true;
    generate_status_.status = GenerateState::PAUSED;
    for (int i = 0; i < tileNum(); ++i) {
        if (!isDoneWithoutLock(i)) {
            sub_generate_status_[i].status = GenerateState::PAUSED;
        }
    }
    last_pause_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
}
bool GenerateStream::setRunning() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (stoppedWithoutLock()) {
        return false;
    }
    generate_status_.status = GenerateState::RUNNING;
    for (int i = 0; i < tileNum(); ++i) {
        if (!isDoneWithoutLock(i)) {
            sub_generate_status_[i].status = GenerateState::RUNNING;
        }   
    }
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
std::string GenerateStream::stopReason() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    return generate_status_.error_info;
}
bool GenerateStream::finished() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    return generate_status_.status == GenerateState::FINISHED;
}


void GenerateStream::setKVCache(const BatchKVCacheBlockAddr &kv_cache_block_addr) {
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
    return std::all_of(sub_generate_status_.begin(), sub_generate_status_.end(),
                                    [](GenerateStatus& generate_status) { return generate_status.status == GenerateState::FINISHED; });
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
    for (auto& stop_words: generate_input_->generate_config->stop_words_list) {
        bool match_one = true;
        size_t begin_index = seq_length_ - stop_words.size();
        for (auto& token: stop_words) {
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

void GenerateStream::update(ft::BufferPtr&     new_tokens,
                            int                num_new_tokens,
                            const ft::Buffer&  hidden_states,
                            const ft::Buffer&  logits,
                            const ft::Buffer&  cum_log_probs,
                            bool not_update_output) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    is_context_stream_ = false;
    if (stoppedWithoutLock()) {
        return;
    }
    if (iter_count_ == 0) {
        first_token_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    }
    iter_count_ += 1;
    // # NOTE: new tokens indicate num of newly genearted tokens
    // # typically 1 but can be > 1 under speculative decoding
    // # This differs from new_tokens.shape[-1] under beam search case,
    // # which needs to update all the generated tokens each update.
    assert(new_tokens->dim() == 2);
    auto update_length   = new_tokens->shape()[1];
    auto update_to_pos   = seq_length_ + num_new_tokens;
    auto update_from_pos = update_to_pos - update_length;
    // ft::bufferSliceCopy(complete_token_ids_, new_tokens, 1, update_from_pos, update_to_pos);
    int* token_ids_ = (int*)complete_token_ids_->data();
    for (int i = 0; i < batchSize(); ++i) {
        token_ids_[i * complete_token_ids_->shape()[1] + seq_length_] = ((int*)new_tokens->data())[i];
    }
    seq_length_ += num_new_tokens;
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
    size_t output_len = seq_length_ - inputLength();
    generate_outputs_->generate_outputs.clear();
    for (size_t i = 0; i < tileNum(); i++) {
        GenerateOutput generate_output;
        generate_output.aux_info.iter_count = iter_count_;
        generate_output.output_ids =
            device_->allocateBuffer({ft::DataType::TYPE_INT32, {1lu, output_len}, ft::AllocationType::HOST}, {});
        // TODO(xinfei.sxf) optimize this copy : only copy last token
        memcpy(generate_output.output_ids->data(), complete_token_ids_->view(i, 1).dataWithOffset<int32_t>(inputLength()), sizeof(int32_t) * output_len);
        if (generate_input_->generate_config->return_logits) {
            ft::BufferPtr host_logits;
            if (logits.shape()[0] == 1) {
                host_logits = device_->clone({logits, ft::AllocationType::HOST});
            } else {
                host_logits = device_->clone({logits.view(i, 1), ft::AllocationType::HOST});
            }
            if (!generate_input_->generate_config->select_tokens_id.empty()) {
                // TODO(xinfei.sxf) implement bufferIndexSelect at gpu
                ft::BufferPtr select_logits =
                    device_->allocateBuffer({host_logits->type(),
                                            {generate_input_->generate_config->select_tokens_id.size()},
                                            ft::AllocationType::HOST});
                ft::bufferIndexSelect<float>(
                    host_logits, select_logits, generate_input_->generate_config->select_tokens_id);
                generate_output.logits = select_logits;
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
            auto x = device_->allocateBuffer(
            {ft::DataType::TYPE_FP32, {1}, ft::AllocationType::HOST}, {});
            // TODO(xinfei.sxf) fix this loss
            // *((float*)x->data()) = 1.0f;
            generate_output.loss = std::move(x);
        }

        generate_output.finished              = sub_generate_status_[i].status == GenerateState::FINISHED;
        generate_output.aux_info.cost_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        generate_output.aux_info.input_len    = generate_input_->promptLength();
        generate_output.aux_info.prefix_len   = generate_input_->prefix_length;
        generate_output.aux_info.output_len   = seq_length_ - generate_input_->inputLength();
        generate_output.aux_info.reuse_len    = reuse_length_;

        generate_output.aux_info.cum_log_probs =
            device_->allocateBuffer({ft::DataType::TYPE_FP32, {1lu}, ft::AllocationType::HOST}, {});
        memcpy(generate_output.aux_info.cum_log_probs.value()->data(), cum_log_probs_->dataWithOffset<float>(i), sizeof(float));
        
        generate_outputs_->generate_outputs.push_back(generate_output);
    }
    generate_outputs_queue_.push(*generate_outputs_);
}

void GenerateStream::setMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter) {
    metrics_reporter_ = metrics_reporter;
}
void GenerateStream::reportMetric() {
    if (metrics_reporter_) {
        RtpLLMStreamMetricsCollector collector;
        collector.qps = finished();
        collector.cancel_qps = cancelled_;
        collector.error_qps = stopped() && !cancelled_;
        if (finished()) {
            collector.reuse_length = reuse_length_;
            collector.input_token_length = inputLength();
            collector.output_token_length = seq_length_ - generate_input_->inputLength();
            collector.iterate_count = iter_count_;
            collector.query_batch_size = tileNum();
            collector.total_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
            collector.first_token_latency_us = first_token_time_us_;
            collector.wait_latency_us = wait_time_us_;
            collector.pause_latency_us = pause_time_us_;
        }
        metrics_reporter_->report<RtpLLMStreamMetrics, RtpLLMStreamMetricsCollector>(nullptr, &collector);
    }
}

std::string GenerateStream::debugString() const {
    std::stringstream debug_string;
    debug_string << "GenerateStream {"
                    << "generate_input:" << generate_input_->debugString()
                    << ", max_seq_len:" << max_seq_len_
                    << ", input_length:" << inputLength()
                    << ", seq_length:" << seq_length_
                    << ", reuse_length:" << reuse_length_
                    << ", batch_size:" << batch_size_
                    << "}";
    return debug_string.str();
}

void GenerateStream::setSeqLength(int seq_length) {
    seq_length_ = seq_length;
}

void GenerateStream::setIsContextStream(bool is_context_stream) {
    is_context_stream_ = is_context_stream;
}

StreamCacheResource& GenerateStream::streamCacheResource() {
    return stream_cache_resource_;
}

}  // namespace rtp_llm
