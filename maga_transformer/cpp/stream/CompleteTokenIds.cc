#include "maga_transformer/cpp/stream/CompleteTokenIds.h"

#include <sstream>

namespace rtp_llm {

CompleteTokenIds::CompleteTokenIds(ft::DeviceBase* device, int batch_size, int max_seq_len, int seq_size_per_block, bool in_think_mode, int max_thinking_tokens, int input_length, std::vector<int> end_think_token_ids) 
    : device_(device)
    , batch_size_(batch_size)
    , max_seq_len_(max_seq_len)
    , seq_size_per_block_(seq_size_per_block)
    , in_think_mode_(in_think_mode)
    , max_thinking_tokens_(max_thinking_tokens)
    , input_length_(input_length)
    , end_think_token_ids_(end_think_token_ids)
    , think_end_status_dfa_(batch_size, StringContainDFA<size_t, int>(end_think_token_ids)) {
}

CompleteTokenIds::CompleteTokenIds(const CompleteTokenIds& other)
    : device_(other.device_)
    , batch_size_(other.batch_size_)
    , max_seq_len_(other.max_seq_len_)
    , seq_size_per_block_(other.seq_size_per_block_)
    , in_think_mode_(other.in_think_mode_)
    , max_thinking_tokens_(other.max_thinking_tokens_)
    , input_length_(other.input_length_)
    , end_think_token_ids_(other.end_think_token_ids_)
    , seq_length_(other.seq_length_)
    , common_len_(other.common_len_)
    , start_check_seq_length_(other.start_check_seq_length_)
    , first_token_time_us_(other.first_token_time_us_)
    , first_token_latency_us_(other.first_token_latency_us_)
    , think_end_status_dfa_(other.think_end_status_dfa_) {
    complete_token_ids_ = device_->clone({*(other.complete_token_ids_), ft::AllocationType::HOST});
}

void CompleteTokenIds::init(const std::shared_ptr<GenerateInput>& generate_input) {
    FT_CHECK(device_ != nullptr && generate_input != nullptr);

    seq_length_ = generate_input->inputLength();
    common_len_ = seq_length_;
    start_check_seq_length_ = seq_length_;
    init_seq_size_ = seq_length_;
    in_think_mode_ = generate_input->generate_config->in_think_mode;
    max_thinking_tokens_ = generate_input->generate_config->max_thinking_tokens;
    input_length_ = generate_input->inputLength();
    end_think_token_ids_ = generate_input->generate_config->end_think_token_ids;
    for (auto& dfa: think_end_status_dfa_) {
        dfa.compile(end_think_token_ids_);
    }

    complete_token_ids_ = device_->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)batch_size_, (size_t)max_seq_len_}, ft::AllocationType::HOST}, {});

    memset(complete_token_ids_->data(), 0, complete_token_ids_->sizeBytes());
    for (int i = 0; i < batch_size_; ++i) {
        memcpy(complete_token_ids_->dataWithOffset<int32_t>(i * max_seq_len_),
               generate_input->input_ids->data(),
               generate_input->input_ids->sizeBytes());
    }

    FT_LOG_DEBUG("complete tokenids init done, %s", showStatus(0).c_str());
}

const ft::BufferPtr& CompleteTokenIds::completeTokenIds() {
    return complete_token_ids_;
}

std::vector<int> CompleteTokenIds::completeTokenIdsVec(int batch_idx) {
    FT_CHECK(batch_idx < batch_size_);
    return fastertransformer::buffer2vector<int>(complete_token_ids_->view(batch_idx, 1), seq_length_);
}

std::vector<int> CompleteTokenIds::commonCompleteTokenIdsVec(int batch_idx) {
    FT_CHECK(batch_idx < batch_size_);
    return fastertransformer::buffer2vector<int>(complete_token_ids_->view(batch_idx, 1), common_len_);
}

std::vector<int> CompleteTokenIds::currentExecuteTokens(int batch_idx) {
    FT_CHECK(batch_idx < batch_size_);
    return {*(*complete_token_ids_)[batch_idx].dataWithOffset<int>(seq_length_ - 1)};
}

std::vector<int> CompleteTokenIds::contextTokens(int batch_idx, int prefix_length, int context_length) {
    FT_CHECK(batch_idx < batch_size_);
    return fastertransformer::buffer2vector<int>(
            (*complete_token_ids_)[batch_idx].view(prefix_length, context_length));
}

std::vector<int> CompleteTokenIds::getLatestTokens(size_t token_num) {
    FT_CHECK(seq_length_ >= token_num);
    std::vector<int> latest_tokens(token_num);
    memcpy(latest_tokens.data(),
        complete_token_ids_->dataWithOffset<int32_t>(seq_length_ - token_num), sizeof(int32_t) * token_num);
    return latest_tokens;
}

bool CompleteTokenIds::matchEosToken(int batch_id, int token_id) {
    int* token_ids = (int*)complete_token_ids_->view(batch_id, 1).data();
    for (size_t i = start_check_seq_length_; i <= seq_length_; ++i) {
        if (token_id == token_ids[i - 1]) {
            seq_length_ = i;
            return true;
        }
    }
    return false;
}

bool CompleteTokenIds::matchStopWordsList(int batch_id, const std::vector<int> &stop_words) {
    int* token_ids = (int*)complete_token_ids_->view(batch_id, 1).data();
    for (size_t i = start_check_seq_length_; i <= seq_length_; ++i) {
        bool match_one   = true;
        size_t begin_index = i - stop_words.size();
        for (auto& token : stop_words) {
            if (token != token_ids[begin_index++]) {
                match_one = false;
                break;
            }
        }
        if (match_one) {
            seq_length_ = i;
            return true;
        }
    }
    return false;
}

bool CompleteTokenIds::update(const ft::BufferPtr& new_tokens, int64_t begin_time_us, int num_new_tokens, int input_length, int max_token_num, int vocab_size, int num_beams, int64_t stream_id, int& error_token_id) {
    if (seq_length_ == input_length) {
        first_token_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
        first_token_latency_us_ = first_token_time_us_ - begin_time_us;
    }

    if (seq_length_ + num_new_tokens > max_token_num) {
        num_new_tokens = max_token_num - seq_length_;
    }

    // # NOTE: new tokens indicate num of newly genearted tokens
    // # typically 1 but can be > 1 under speculative decoding
    // # This differs from new_tokens.shape[-1] under beam search case,
    // # which needs to update all the generated tokens each update.
    FT_CHECK(new_tokens->dim() == 2);
    
    for (size_t i = 0; i < batch_size_; ++i) {
        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = *(*new_tokens)[i].dataWithOffset<int>(j);
            if (!(current_token_id >= 0 && current_token_id < vocab_size)) { // check tokenid
                error_token_id = current_token_id; 
                return false;
            }
        }
        if (in_think_mode_) {
            dfaStatusForward(think_end_status_dfa_[i], (*new_tokens)[i], num_new_tokens, end_think_token_ids_,
                (seq_length_ + num_new_tokens >= max_thinking_tokens_ + input_length_));
        }
        if (num_beams > 1) {
            auto new_tokens_num = (*new_tokens)[i].shape()[0];
            device_->copy({(*complete_token_ids_)[i].view(0, new_tokens_num), (*new_tokens)[i]});
        } else {
            device_->copy({(*complete_token_ids_)[i].view(seq_length_, num_new_tokens), (*new_tokens)[i].view(0, num_new_tokens)});
        }
    }
    setSeqLength(seq_length_ + num_new_tokens);

    FT_LOG_DEBUG("update token, num_new_tokens: %d, after update is %s", num_new_tokens, showStatus(0).c_str());
    return true;
}

void CompleteTokenIds::setSeqLength(int seq_length) {
    if (seq_length > seq_length_) {
        start_check_seq_length_ = seq_length_ + 1;
    } else {
        start_check_seq_length_ = seq_length;
    }
    seq_length_ = seq_length;

    if (batch_size_ == 1) { // reset common len
        common_len_ = seq_length_;
    }
}

int CompleteTokenIds::seqLength() const {
    return seq_length_;
}

void CompleteTokenIds::copyTokensTo(int batch_id, void *dst, int offset, size_t token_num) {
    memcpy(dst, complete_token_ids_->view(batch_id, 1).dataWithOffset<int32_t>(offset),
               sizeof(int32_t) * token_num);
}

void CompleteTokenIds::appendTokens(int batch_id, size_t token_num, const ft::Buffer &src) {
    device_->copy({(*complete_token_ids_)[batch_id].view(seq_length_, token_num), src});
    setSeqLength(seq_length_ + token_num);
}

std::vector<size_t> CompleteTokenIds::thinkEndTokensStatus() {
    std::vector<size_t> status;
    for (auto& dfa: this->think_end_status_dfa_) {
        status.push_back(dfa.status());
    }
    return status;
}

int64_t CompleteTokenIds::firstTokenTimeUs() const {
    return first_token_time_us_;
}

int64_t CompleteTokenIds::firstTokenLatencyUs() const {
    return first_token_latency_us_;
}

std::string CompleteTokenIds::toString(int batch_id) const {
    return (*complete_token_ids_)[batch_id].view(0, seq_length_).debugStringWithData<int32_t>();
}

int32_t* CompleteTokenIds::data(int batch_id) {
    // eq to (*comple_token_ids).[batch_id].data<int32_t>();
    return complete_token_ids_->view(batch_id, 1).data<int32_t>();
}

std::string CompleteTokenIds::showStatus(int batch_id) {
    int start = seq_length_ > 10 ? seq_length_ - 10 : 0;
    int len = seq_length_ - start;
    auto debug_string = (*complete_token_ids_)[batch_id].view(start, len).debugStringWithData<int32_t>();

    std::ostringstream oss;
    oss << "complete tokenids seq length " << seq_length_ << ", tokenids size " << completeTokenIdsVec(batch_id).size() << ", last 10 tokenids is " << debug_string;
    return oss.str();
}

}
