#pragma once

#include <memory>
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

class CompleteTokenIds {
public:
    CompleteTokenIds(
        rtp_llm::DeviceBase* device, int batch_size, int max_batch_size, int max_seq_len, int seq_size_per_block);
    CompleteTokenIds(const CompleteTokenIds& other, bool share = false, int shift_token_num = 0);

public:
    void init(const std::shared_ptr<GenerateInput>& generate_input, size_t extra_reserve_token_num = 0);

    std::vector<int> completeTokenIdsVec(int batch_id);
    std::vector<int> commonCompleteTokenIdsVec(int batch_id);
    std::vector<int> currentExecuteTokens(int batch_id);
    std::vector<int> contextTokens(int batch_id, int prefix_length, int context_length);
    std::vector<int> getLatestTokens(size_t token_num);

    int maxBatchSize();
    int batchSize();

    bool matchEosToken(int batch_id, int token_id);
    bool matchStopWordsList(int batch_id, const std::vector<int>& stop_words);

    bool update(const rtp_llm::BufferPtr& new_tokens,
                int64_t                   begin_time_us,
                int                       num_new_tokens,
                int                       input_length,
                int                       max_token_num,
                int                       vocab_size,
                bool                      is_beam_search,
                int64_t                   stream_id,
                int&                      error_token_id);
    void copyTokensTo(int batch_id, void* dst, int offset, size_t token_num);
    void appendTokens(int batch_id, size_t token_num, const rtp_llm::Buffer& src);

    int  seqLength() const;
    void setSeqLength(int seq_length);

    const rtp_llm::BufferPtr& completeTokenIds();

    int64_t firstTokenTimeUs() const;
    int64_t firstTokenLatencyUs() const;

    std::string toString(int batch_id) const;

    int32_t* data(int batch_id);

    std::string showStatus(int batch_id);

private:
    rtp_llm::DeviceBase* device_;

    int batch_size_;
    int max_batch_size_;
    int max_seq_len_;
    int seq_size_per_block_;

    int     seq_length_;
    int     common_len_;
    int     start_check_seq_length_;
    int64_t first_token_time_us_    = 0;
    int64_t first_token_latency_us_ = 0;

    rtp_llm::BufferPtr complete_token_ids_;
};

using CompleteTokenIdsPtr = std::shared_ptr<CompleteTokenIds>;

}  // namespace rtp_llm