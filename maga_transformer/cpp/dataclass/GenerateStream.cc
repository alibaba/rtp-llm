#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include <atomic>
#include <memory>

using namespace std;

namespace rtp_llm {

class StreamCounter {
public:
    StreamCounter() {}
    StreamCounter(const StreamCounter&)            = delete;
    StreamCounter& operator=(const StreamCounter&) = delete;

public:
    static StreamCounter& getInstance() {
        static StreamCounter streamCounter;
        return streamCounter;
    }

    int incrAndGet() {
        return ++counter_;
    }

private:
    std::atomic_int counter_ = 0;
};

GenerateStream::GenerateStream(const shared_ptr<GenerateInput>& input, int max_seq_len):
    generate_input_(input), done_(false), cancelled_(false), stream_cache_resource_(this) {
    if (!input.get()) {
        return;
    }
    seq_length_ = generate_input_->inputLength();

    max_seq_len_        = max_seq_len;
    begin_time_         = TimeUtility::currentTimeInMilliSeconds();
    auto device         = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    complete_token_ids_ = device->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)numBeams(), (size_t)max_seq_len}, ft::AllocationType::HOST}, {});
    memcpy(complete_token_ids_->data(), generate_input_->input_ids->data(), generate_input_->input_ids->sizeBytes());
    generate_output_ = make_shared<GenerateOutput>();
    sub_generate_status_.clear();
    sub_generate_status_.resize(tileNum());
    // TODO(xinfei.sxf) fix this
    for (int i = 0; i < tileNum(); ++i) {
        sub_generate_status_[i].status = GenerateState::RUNNING;
    }
    released_ = false;
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

// TODO(xinfei.sxf) remove this api?
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
                            optional<ft::BufferPtr>& hidden_states,
                            optional<ft::BufferPtr>& logits,
                            optional<ft::BufferPtr>& cum_log_probs,
                            optional<ft::BufferPtr>& loss) {
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
    auto update_length   = new_tokens->shape()[0];
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
    auto device = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    generate_output_->output_ids =
        device->allocateBuffer({new_tokens->type(), new_tokens->shape(), ft::AllocationType::HOST}, {});
    memcpy(generate_output_->output_ids->data(), new_tokens->data(), new_tokens->sizeBytes());
    if (generate_input_->generate_config->return_logits) {
        if (!generate_input_->generate_config->select_tokens_id.empty()) {
            ft::BufferPtr select_logits =
                device->allocateBuffer({logits.value()->type(),
                                        {generate_input_->generate_config->select_tokens_id.size()},
                                        ft::AllocationType::HOST},
                                       {});
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
