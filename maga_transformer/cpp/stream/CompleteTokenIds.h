#pragma once

#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace rtp_llm {

class CompleteTokenIds {
public:
    CompleteTokenIds(ft::DeviceBase* device, int batch_size, int max_seq_len, int seq_size_per_block, bool in_think_mode, int max_thinking_tokens, int input_length, int end_think_token_id);
    CompleteTokenIds(const CompleteTokenIds& other);

public:
    void init(const std::shared_ptr<GenerateInput>& generate_input);

    std::vector<int> completeTokenIdsVec(int batch_id);
    std::vector<int> commonCompleteTokenIdsVec(int batch_id);
    std::vector<int> currentExecuteTokens(int batch_id);
    std::vector<int> contextTokens(int batch_id, int prefix_length, int context_length);
    std::vector<int> getLatestTokens(size_t token_num);

    bool matchEosToken(int batch_id, int token_id);
    bool matchStopWordsList(int batch_id, const std::vector<int>& stop_words);

    bool update(const ft::BufferPtr& new_tokens, int64_t begin_time_us, int num_new_tokens, int input_length, int max_token_num, int vocab_size, int num_beams, int64_t stream_id, int& error_token_id);
    void copyTokensTo(int batch_id, void *dst, int offset, size_t token_num);
    void appendTokens(int batch_id, size_t token_num, const ft::Buffer &src);
    const std::deque<bool>& isThinkEndTokenIdExist();

    int seqLength() const;
    void setSeqLength(int seq_length);

    const ft::BufferPtr& completeTokenIds();

    int64_t firstTokenTimeUs() const;
    int64_t firstTokenLatencyUs() const;
     
    std::string toString(int batch_id) const;

    int32_t* data(int batch_id);

    std::string showStatus(int batch_id);

private:
    ft::DeviceBase *device_;

    // eq to stream.tileNum()
    int batch_size_; 
    int max_seq_len_;
    int seq_size_per_block_;
    int init_seq_size_;
    bool in_think_mode_;
    int max_thinking_tokens_;
    int input_length_;
    int end_think_token_id_;

    int seq_length_; 
    int common_len_;
    int start_check_seq_length_;
    int64_t first_token_time_us_  = 0;
    int64_t first_token_latency_us_ = 0; 
    std::deque<bool> is_think_end_token_id_exist_;

    ft::BufferPtr complete_token_ids_;
};

}