#include "rtp_llm/cpp/stream/CompleteTokenIds.h"

#include <sstream>

namespace rtp_llm {

CompleteTokenIds::CompleteTokenIds(rtp_llm::DeviceBase* device, int batch_size, int max_seq_len, int seq_size_per_block)
    : device_(device)
    , batch_size_(batch_size)
    , max_seq_len_(max_seq_len)
    , seq_size_per_block_(seq_size_per_block) {
}

CompleteTokenIds::CompleteTokenIds(const CompleteTokenIds& other)
    : device_(other.device_)
    , batch_size_(other.batch_size_)
    , max_seq_len_(other.max_seq_len_)
    , seq_size_per_block_(other.seq_size_per_block_)
    , seq_length_(other.seq_length_)
    , common_len_(other.common_len_)
    , start_check_seq_length_(other.start_check_seq_length_)
    , first_token_time_us_(other.first_token_time_us_)
    , first_token_latency_us_(other.first_token_latency_us_) {
    complete_token_ids_ = device_->clone({*(other.complete_token_ids_), rtp_llm::AllocationType::HOST});
}

void CompleteTokenIds::init(const std::shared_ptr<GenerateInput>& generate_input) {
    RTP_LLM_CHECK(device_ != nullptr && generate_input != nullptr);

    seq_length_ = generate_input->inputLength();
    RTP_LLM_CHECK_WITH_INFO((seq_length_ <= max_seq_len_),
        "seq_length[%d] must be less than max_seq_len[%d]", seq_length_, max_seq_len_);

    common_len_ = seq_length_;
    start_check_seq_length_ = seq_length_;
    init_seq_size_ = seq_length_;

    complete_token_ids_ = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)batch_size_, (size_t)max_seq_len_}, rtp_llm::AllocationType::HOST}, {});

    memset(complete_token_ids_->data(), 0, complete_token_ids_->sizeBytes());
    for (int i = 0; i < batch_size_; ++i) {
        memcpy(complete_token_ids_->dataWithOffset<int32_t>(i * max_seq_len_),
               generate_input->input_ids->data(),
               generate_input->input_ids->sizeBytes());
    }

    RTP_LLM_LOG_DEBUG("complete tokenids init done, %s", showStatus(0).c_str());
}

const rtp_llm::BufferPtr& CompleteTokenIds::completeTokenIds() {
    return complete_token_ids_;
}

std::vector<int> CompleteTokenIds::completeTokenIdsVec(int batch_idx) {
    RTP_LLM_CHECK(batch_idx < batch_size_);
    return rtp_llm::buffer2vector<int>(complete_token_ids_->view(batch_idx, 1), seq_length_);
}

std::vector<int> CompleteTokenIds::commonCompleteTokenIdsVec(int batch_idx) {
    RTP_LLM_CHECK(batch_idx < batch_size_);
    return rtp_llm::buffer2vector<int>(complete_token_ids_->view(batch_idx, 1), common_len_);
}

std::vector<int> CompleteTokenIds::currentExecuteTokens(int batch_idx) {
    RTP_LLM_CHECK(batch_idx < batch_size_);
    return {*(data(batch_idx) + seq_length_ - 1)};
}

std::vector<int> CompleteTokenIds::contextTokens(int batch_idx, int prefix_length, int context_length) {
    RTP_LLM_CHECK(batch_idx < batch_size_);
    return rtp_llm::buffer2vector<int>(
            (*complete_token_ids_)[batch_idx].view(prefix_length, context_length));
}

std::vector<int> CompleteTokenIds::getLatestTokens(size_t token_num) {
    RTP_LLM_CHECK(seq_length_ >= token_num);
    std::vector<int> latest_tokens(token_num);
    memcpy(latest_tokens.data(),
        complete_token_ids_->dataWithOffset<int32_t>(seq_length_ - token_num), sizeof(int32_t) * token_num);
    return latest_tokens;
}

bool CompleteTokenIds::matchEosToken(int batch_id, int token_id) {
    int* token_ids = data(batch_id);
    for (size_t i = start_check_seq_length_; i <= seq_length_; ++i) {
        if (token_id == token_ids[i - 1]) {
            seq_length_ = i;
            return true;
        }
    }
    return false;
}

bool CompleteTokenIds::matchStopWordsList(int batch_id, const std::vector<int> &stop_words) {
    int* token_ids = data(batch_id);
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

bool CompleteTokenIds::update(const rtp_llm::BufferPtr& new_tokens, int64_t begin_time_us, int num_new_tokens, int input_length, int max_token_num, int vocab_size, int num_beams, int64_t stream_id, int& error_token_id) {
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
    RTP_LLM_CHECK(new_tokens->dim() == 2);

    auto new_tokens_ptr = new_tokens->data<int>(); // [batch_size, max_num_new_tokens]
    auto max_num_new_tokens = new_tokens->shape()[1];

    for (size_t i = 0; i < batch_size_; ++i) {
        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = (new_tokens_ptr + num_new_tokens * i)[j];
            if (!(current_token_id >= 0 && current_token_id < vocab_size)) { // check tokenid
                error_token_id = current_token_id;
                return false;
            }
        }
        if (num_beams > 1) {
            memcpy(data(i), new_tokens_ptr + i * max_num_new_tokens, sizeof(int) * max_num_new_tokens);
        } else {
            memcpy(data(i) + seq_length_, new_tokens_ptr + i * num_new_tokens, sizeof(int) * num_new_tokens);
        }
    }
    setSeqLength(seq_length_ + num_new_tokens);

    RTP_LLM_LOG_DEBUG("update token, num_new_tokens: %d, after update is %s", num_new_tokens, showStatus(0).c_str());
    return true;
}

void CompleteTokenIds::setSeqLength(int seq_length) {
    RTP_LLM_CHECK(seq_length <= max_seq_len_);
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
    memcpy(dst, data(batch_id) + offset, sizeof(int32_t) * token_num);
}

void CompleteTokenIds::appendTokens(int batch_id, size_t token_num, const rtp_llm::Buffer &src) {
    if (src.dim() == 2 && src.shape()[0] == 1) {
        device_->copy({(*complete_token_ids_)[batch_id].view(seq_length_, token_num),
                        src[0].view(0, token_num)});
    } else {
        device_->copy({(*complete_token_ids_)[batch_id].view(seq_length_, token_num),
                        src.view(0, token_num)});
    }

    setSeqLength(seq_length_ + token_num);
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
    // avoid construct Buffer to reduce overhead
    return complete_token_ids_->data<int32_t>() + batch_id * complete_token_ids_->shape()[1];
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
