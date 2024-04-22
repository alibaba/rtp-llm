#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include <atomic>
#include <memory>

using namespace std;

namespace rtp_llm {

GenerateStream::GenerateStream(const shared_ptr<GenerateInput>& input, int max_seq_len):
    generate_input_(input), stream_cache_resource_(this) {
    if (!input.get()) {
        return;
    }
    seq_length_ = generate_input_->inputLength();

    max_seq_len_        = max_seq_len;
    begin_time_         = TimeUtility::currentTimeInMilliSeconds();
    device_             = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    complete_token_ids_ = device_->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)numBeams(), (size_t)max_seq_len}, ft::AllocationType::HOST}, {});
    memcpy(complete_token_ids_->data(), generate_input_->input_ids->data(), generate_input_->input_ids->sizeBytes());
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

void GenerateStream::cancel() {
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

void GenerateStream::updatePrefix(const std::shared_ptr<PtuningBase>& ptuning) {
    if (ptuning) {
        prefix_info_ = ptuning->getPtuningInfo(*generate_input_->generate_config);
        if (!prefix_info_.prefix_prompt.empty()) {
            generate_input_->updatePrefix(prefix_info_.prefix_prompt);
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
                            optional<ft::BufferPtr> hidden_states,
                            optional<ft::BufferPtr> logits,
                            optional<ft::BufferPtr> cum_log_probs,
                            optional<ft::BufferPtr> loss,
                            bool not_update_output) {
    if (stoppedWithoutLock()) {
        return;
    }
    if (generate_output_->aux_info.iter_count == 0) {
        reportFirstTokenRt();
    }

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
    for (int i = 0; i < num_new_tokens; i++) {
        seq_length_ += 1;
        if (needFinish()) {
            finished = true;
            break;
        }
    }
    if (finished) {
        setFinishedWithoutLock();
    }
    if (not_update_output) {
        return;
    }
    updateOutput(finished, std::move(hidden_states), std::move(logits), std::move(cum_log_probs), std::move(loss));
}

void GenerateStream::updateOutput(bool finished,
                                  optional<ft::BufferPtr> hidden_states,
                                  optional<ft::BufferPtr> logits,
                                  optional<ft::BufferPtr> cum_log_probs,
                                  optional<ft::BufferPtr> loss) {
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
    if (generate_input_->generate_config->calculate_loss) {
        generate_output_->loss = std::move(loss.value());
    }

    generate_output_->finished              = finished;
    generate_output_->aux_info.cost_time_ms = TimeUtility::currentTimeInMilliSeconds() - begin_time_;
    generate_output_->aux_info.input_len    = generate_input_->promptLength();
    generate_output_->aux_info.prefix_len   = generate_input_->prefix_length;
    generate_output_->aux_info.output_len   = seq_length_ - generate_input_->inputLength();
    // TODO(xinfei.sxf) add return option for cum_log_probs
    if (cum_log_probs != std::nullopt) {
        generate_output_->aux_info.cum_log_probs = std::move(cum_log_probs.value());
    } else {
        generate_output_->aux_info.cum_log_probs = std::nullopt;
    }
    generate_output_->aux_info.iter_count += 1;
    generate_output_->aux_info.reuse_len = reuse_length_;
    generate_outputs_.push(*generate_output_);
}

}  // namespace rtp_llm
