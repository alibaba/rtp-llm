#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "src/fastertransformer/utils/logger.h"
#include <memory>
#include <mutex>

using namespace std;
namespace rtp_llm {

FIFOScheduler::FIFOScheduler(const MagaInitParams& config, const std::shared_ptr<CacheManager>& cache_manager):
    cache_manager_(cache_manager), max_seq_len_(config.gpt_init_parameter->max_seq_len_) {}


FIFOScheduler::~FIFOScheduler() {
    (void)stop();
    FT_LOG_INFO("destory FIFOScheduler");
}

absl::Status FIFOScheduler::stop() {
    FT_LOG_INFO("stop FIFOScheduler");
    {
        lock_guard<mutex> lock(lock_);
        stop_ = true;
    }
    cond_.notify_all();
    return absl::OkStatus();
}

void FIFOScheduler::evictDoneStreams(list<GenerateStreamPtr>& streams) const {
    for (auto it = streams.begin(); it != streams.end();) {
        if ((*it)->stopped() || (*it)->finished()) {
            // Immediately free resources to run more streams
            (*it)->releaseResource();
            FT_LOG_DEBUG("evict stream [%ld]", (*it)->streamId());
            it = streams.erase(it);
        } else {
            ++it;
        }
    }
}

absl::Status FIFOScheduler::enqueue(const GenerateStreamPtr& stream) {
    {
        lock_guard<mutex> lock(lock_);
        waiting_streams_.emplace_back(stream);
    }
    cond_.notify_all();
    return absl::OkStatus();
}

int FIFOScheduler::runningNextBlockNum() const {
    int total_need_block_nums = 0;
    for (auto& stream : running_streams_) {
        total_need_block_nums += stream->nextNeedBlockNums() + reserve_block_num_;
    }
    return total_need_block_nums;
}

void FIFOScheduler::evaluateRunningNext() {
    if (running_streams_.empty()) {
        return;
    }

    int running_next_block_num = runningNextBlockNum();
    for (auto& stream : waiting_streams_) {
        int need_block_num = running_next_block_num - cache_manager_->freeBlockNums();
        if (need_block_num > 0) {
            stream->tryReleaseKVBlock(need_block_num);
        } else {
            break;
        }
    }
    while (!running_streams_.empty()) {
        int need_block_num = (int)runningNextBlockNum() - (int)cache_manager_->freeBlockNums();
        if (need_block_num <= 0) {
            break;
        }
        auto& last_stream = *(running_streams_.rbegin());
        last_stream->tryReleaseKVBlock(need_block_num);
        FT_LOG_DEBUG("stream [%ld] fall back", last_stream->streamId());
        waiting_streams_.emplace_front(last_stream);
        running_streams_.pop_back();
    }

    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        if (!(*it)->incrKVBlock()) {
            (*it)->setStop("incrKVBlock failed");
            (*it)->releaseResource();
            FT_LOG_DEBUG("stream [%ld] incr block failed", (*it)->streamId());
            it = running_streams_.erase(it);
        } else {
            it++;
        }
    }
}

bool FIFOScheduler::evaluateRunningMemory(int total_token_size) const {
    return total_token_size < max_seq_len_;
}

bool FIFOScheduler::evaluateKVCacheMemory(int block_num) const {
    return runningNextBlockNum() + block_num <= cache_manager_->freeBlockNums();
}

// TODO(xinfei.sxf) 在考虑reuse cache的情况下，评估不准确，应该立刻申请资源试试。
bool FIFOScheduler::evaluateNewStream(const list<GenerateStreamPtr>& streams,
                                      const GenerateStreamPtr&       new_stream) const {
    int total_token_size = new_stream->contextLength() + running_streams_.size();
    for (auto& stream : streams) {
        total_token_size += stream->contextLength();
    }
    return evaluateKVCacheMemory(new_stream->initalKVCacheCount() + reserve_block_num_ * streams.size())
           && evaluateRunningMemory(total_token_size) && new_stream->initKVBlock();
}

list<GenerateStreamPtr> FIFOScheduler::scheduleNew() {
    list<GenerateStreamPtr> new_streams;
    for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
        if (evaluateNewStream(new_streams, *it)) {
            FT_LOG_DEBUG("stream [%ld] add to new queue", (*it)->streamId());
            new_streams.emplace_back(*it);
            it = waiting_streams_.erase(it);
        } else if (running_streams_.empty() && new_streams.empty()) {
            // It is impossible for this stream to acquire enough resources
            FT_LOG_DEBUG("stream [%ld] can not add to new queue", (*it)->streamId());
            (*it)->setStop("can not be add input queue");
            it++;
        } else {
            // try to join new_streams in the next schedule cycle
            break;
        }
    }
    return new_streams;
}

absl::StatusOr<list<GenerateStreamPtr>> FIFOScheduler::schedule() {
    unique_lock<mutex> lock(lock_);
    cond_.wait(lock, [this]{
        return stop_ || !waiting_streams_.empty() || !running_streams_.empty();
    });
    evictDoneStreams(waiting_streams_);
    evictDoneStreams(running_streams_);
    // TODO(xinfei.sxf) 刚踢出running的可能马上又加入了running
    evaluateRunningNext();

    auto new_stream = scheduleNew();
    running_streams_.insert(running_streams_.end(), new_stream.begin(), new_stream.end());
    return running_streams_;
}

int FIFOScheduler::waitingStreamsSize() {
    return waiting_streams_.size();
}

int FIFOScheduler::runningStreamsSize() {
    return running_streams_.size();
}

}  // namespace rtp_llm
