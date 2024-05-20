#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include <atomic>
#include <memory>

using namespace std;

namespace rtp_llm {

GenerateStream::GenerateStream(const shared_ptr<GenerateInput>& input, const ResourceContext& resource_context, int max_seq_len):
    generate_input_(input), stream_cache_resource_(this, resource_context) {
    if (!input.get()) {
        return;
    }
    seq_length_ = generate_input_->inputLength();

    max_seq_len_        = max_seq_len;
    begin_time_us_      = autil::TimeUtility::currentTimeInMicroSeconds();
    device_             = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    complete_token_ids_ = device_->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)tileNum(), (size_t)max_seq_len}, ft::AllocationType::HOST}, {});
    // TODO(xinfei.sxf) copy batch
    memcpy(complete_token_ids_->data(), generate_input_->input_ids->data(), generate_input_->input_ids->sizeBytes());
    updatePrefix(resource_context.system_prompt);

    generate_output_ = make_shared<GenerateOutput>();
    sub_generate_status_.clear();
    sub_generate_status_.resize(tileNum());
    // TODO(xinfei.sxf) fix this
    for (int i = 0; i < tileNum(); ++i) {
        sub_generate_status_[i].status = GenerateState::RUNNING;
    }
}

absl::StatusOr<GenerateOutput> GenerateStream::nextOutput() {
    while (generate_outputs_.isEmpty() && !stopped() && !finished()) {
        generate_outputs_.waitNotEmpty();
    }
    if (stopped()) {
        return absl::InternalError(stopReason());
    }
    if (generate_outputs_.isEmpty()) {
        return absl::InternalError("no output any more");
    }
    return generate_outputs_.getAndPopFront();
}

bool GenerateStream::needFinishBySPTokens() const {
    // TODO: support batch
    return matchEosToken() || matchStopWordsList();
}

bool GenerateStream::matchEosToken() const {
    int* token_ids_ = (int*)complete_token_ids_->data();
    return special_tokens_.eos_token_id_ == token_ids_[seq_length_ - 1];
}

bool GenerateStream::matchStopWordsList() const {
    int* token_ids_ = (int*)complete_token_ids_->data();
    auto& stop_words_list = special_tokens_.stop_words_list_;
    for (auto& stop_words: stop_words_list) {
        bool match = true;
        size_t begin_index = seq_length_ - 1 - stop_words.size();
        for (auto& token: stop_words) {
            if (token != token_ids_[begin_index++]) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

void GenerateStream::cancel() {
    cancelled_ = true;
    setStop("cancel stream");
}

vector<int> GenerateStream::inputTokens() const {
    auto input_tokens = fastertransformer::buffer2vector<int>(generate_input_->input_ids);
    if (reuseLength() > 0) {
        return vector<int>(input_tokens.begin() + reuseLength(), input_tokens.end());
    } else {
        return input_tokens;
    }
}

int GenerateStream::tileNum() const {
    return std::max(numBeams(), numReturnSequences());
}

bool GenerateStream::isContextStream() const {
    return seqLength() == inputLength();
}

int GenerateStream::batchSize() const {
    int tile_num   = tileNum();
    int batch_size = 0;
    for (int i = 0; i < tile_num; ++i) {
        if (sub_generate_status_[i].status == GenerateState::RUNNING) {
            batch_size++;
        }
    }
    return batch_size;
}

size_t GenerateStream::maxSeqLen() const {
    return max_seq_len_;
}

std::shared_ptr<GenerateInput> GenerateStream::generateInput() const {
    return generate_input_;
}

void GenerateStream::updatePrefix(const std::shared_ptr<SystemPrompt>& system_prompt) {
    if (system_prompt) {
        prompt_param_ = system_prompt->getPromptParams(*generate_input_->generate_config);
        if (!prompt_param_.prompt_token.empty()) {
            generate_input_->updatePrefix(prompt_param_.prompt_token);
            seq_length_ = generate_input_->inputLength();
            memcpy(complete_token_ids_->data(), generate_input_->input_ids->data(), generate_input_->input_ids->sizeBytes());
        }
    }
}

vector<int> GenerateStream::currentExecuteTokens() const {
    // TODO(xinfei.sxf) 在query回退，重运行case下，这个不对
    if (isContextStream()) {
        return inputTokens();
    } else {
        int         tile_num = tileNum();
        vector<int> current_tokens;
        current_tokens.reserve(tile_num);
        int* token_ids = (int*)complete_token_ids_->data();
        for (int i = 0; i < tile_num; ++i) {
            assert(sub_generate_status_[i].status != GenerateState::WAITING);
            if (sub_generate_status_[i].status == GenerateState::RUNNING) {
                current_tokens.push_back(token_ids[i * max_seq_len_ + seqLength() - 1]);
            }
        }
        return current_tokens;
    }
}

void GenerateStream::update(ft::BufferPtr&           new_tokens,
                            int                      num_new_tokens,
                            bool                     finished,
                            optional<ft::BufferPtr>  hidden_states,
                            optional<ft::BufferPtr>  logits,
                            optional<ft::BufferPtr>  cum_log_probs,
                            bool not_update_output) {
    if (stoppedWithoutLock()) {
        return;
    }
    if (generate_output_->aux_info.iter_count == 0) {
        first_token_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    }
    generate_output_->aux_info.iter_count += 1;
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
    finished = finished || needFinish();
    if (finished) {
        setFinishedWithoutLock();
    }
    if (not_update_output) {
        return;
    }

    if (isStreaming() || finished) {
        updateOutput(finished, std::move(hidden_states), std::move(logits), std::move(cum_log_probs));
    }
}

void GenerateStream::updateOutput(bool finished,
                                  optional<ft::BufferPtr> hidden_states,
                                  optional<ft::BufferPtr> logits,
                                  optional<ft::BufferPtr> cum_log_probs) {
    size_t output_len = seq_length_ - inputLength();
    generate_output_->output_ids =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)batchSize(), output_len}, ft::AllocationType::HOST}, {});
    for (int i = 0; i < batchSize(); ++i) {
        memcpy(generate_output_->output_ids->view(i, 1).data(), complete_token_ids_->view(i, 1).dataWithOffset<int32_t>(inputLength()), sizeof(int32_t) * output_len);
    }
    if (generate_input_->generate_config->return_logits) {
        if (!generate_input_->generate_config->select_tokens_id.empty()) {
            ft::BufferPtr select_logits =
                device_->allocateBuffer({logits.value()->type(),
                                         {generate_input_->generate_config->select_tokens_id.size()},
                                         ft::AllocationType::HOST});
            ft::bufferIndexSelect<float>(
                logits.value(), select_logits, generate_input_->generate_config->select_tokens_id);
            generate_output_->logits = std::move(select_logits);
        } else {
            generate_output_->logits = std::move(logits.value());
        }
    }
    if (generate_input_->generate_config->return_hidden_states) {
        generate_output_->hidden_states = std::move(hidden_states.value());
    }
    if (generate_input_->generate_config->calculate_loss == 1) {
        auto x = device_->allocateBuffer(
        {ft::DataType::TYPE_FP32, {1}, ft::AllocationType::HOST}, {});
        // TODO(xinfei.sxf) fix this loss
        // *((float*)x->data()) = 1.0f;
        generate_output_->loss = std::move(x);
    }

    generate_output_->finished              = finished;
    generate_output_->aux_info.cost_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    generate_output_->aux_info.input_len    = generate_input_->promptLength();
    generate_output_->aux_info.prefix_len   = generate_input_->prefix_length;
    generate_output_->aux_info.output_len   = seq_length_ - generate_input_->inputLength();
    // TODO(xinfei.sxf) add return option for cum_log_probs
    if (cum_log_probs != std::nullopt) {
        generate_output_->aux_info.cum_log_probs = std::move(cum_log_probs.value());
    } else {
        generate_output_->aux_info.cum_log_probs = std::nullopt;
    }
    generate_output_->aux_info.reuse_len = reuse_length_;
    generate_outputs_.push(*generate_output_);
}

void GenerateStream::reportMetric() {
    if (metrics_reporter_) {
        RtpLLMStreamMetricsCollector collector;
        collector.qps = finished();
        collector.cancel_qps = cancelled_;
        collector.error_qps = stopped();
        if (finished()) {
            collector.reuse_length = reuse_length_;
            collector.input_token_length = inputLength();
            collector.output_token_length = seq_length_ - generate_input_->inputLength();
            collector.iterate_cout = generate_output_->aux_info.iter_count;
            collector.query_batch_size = tileNum();
            collector.total_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
            collector.first_token_latency_us = first_token_time_us_;
            collector.wait_latency_us = wait_time_us_;
        }
        metrics_reporter_->report<RtpLLMStreamMetrics, RtpLLMStreamMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
