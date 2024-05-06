#include "maga_transformer/cpp/embedding_engine/EmbeddingScheduler.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/utils/logger.h"
#include <mutex>

using namespace std;
namespace rtp_llm {

EmbeddingScheduler::EmbeddingScheduler(const MagaInitParams& config) : config_(*config.gpt_init_parameter) {}

EmbeddingScheduler::~EmbeddingScheduler() {
    (void)stop();
    FT_LOG_INFO("destory EmbeddingScheduler");
}

absl::Status EmbeddingScheduler::stop() {
    FT_LOG_INFO("stop EmbeddingScheduler");
    lock_guard<mutex> lock(lock_);
    stop_ = true;
    cond_.notify_all();
    return absl::OkStatus();
}

absl::Status EmbeddingScheduler::enqueue(EmbeddingStreamPtr stream) {
    lock_guard<mutex> lock(lock_);    
    waiting_streams_.emplace_back(stream);    
    cond_.notify_all();
    return absl::OkStatus();
}

absl::StatusOr<list<EmbeddingStreamPtr>> EmbeddingScheduler::scheduleNew() {
    unique_lock<mutex> lock(lock_);
    cond_.wait(lock, [this]{return stop_ || !waiting_streams_.empty() || !running_streams_.empty();});
    std::list<EmbeddingStreamPtr> new_streams;
    int total_len = 0;
    auto it = waiting_streams_.begin();
    while (it != waiting_streams_.end()) {
        const auto& stream = *it;
        if (total_len + stream->inputLength() > config_.max_context_batch_size_ * config_.max_seq_len_) {
            break;
        }
        new_streams.push_back(stream);
        total_len += stream->inputLength();
        it = waiting_streams_.erase(it);
    }
    if (waiting_streams_.size() != 0 && new_streams.size() == 0) {            
        it = waiting_streams_.begin();
        const auto& stream = *it;
        stream->setError("long prompt error, not scheduled");
        it = waiting_streams_.erase(it);
    }
    return new_streams;
}

int EmbeddingScheduler::waitingStreamsSize() {
    return waiting_streams_.size();
}

int EmbeddingScheduler::runningStreamsSize() {
    return running_streams_.size();
}

}  // namespace rtp_llm
