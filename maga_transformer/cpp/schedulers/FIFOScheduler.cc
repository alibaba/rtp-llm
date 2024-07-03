#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "src/fastertransformer/utils/logger.h"
#include <memory>
#include <mutex>

using namespace std;
namespace rtp_llm {

FIFOScheduler::FIFOScheduler(const ft::GptInitParameter&          params,
                             const std::shared_ptr<CacheManager>& cache_manager,
                             const kmonitor::MetricsReporterPtr   metrics_reporter):
    cache_manager_(cache_manager),
    max_seq_len_(params.max_seq_len_),
    max_context_batch_size_(params.max_context_batch_size_),
    reserve_block_num_(params.scheduler_reserve_resource_ratio_ * cache_manager->freeBlockNums() / 100),
    enable_partial_fallback_(params.enable_partial_fallback_),
    metrics_reporter_(metrics_reporter) {}

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
        (*it)->checkTimeout();
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
        total_need_block_nums += stream->nextNeedBlockNums();
    }
    return total_need_block_nums;
}

// TODO(xinfei.sxf) Is there any situation where the request cannot be ended?
void FIFOScheduler::evaluateRunningNext() {
    int running_next_block_num = runningNextBlockNum();
    // Only in the case of partial fallback, the stream in the waiting queue may hold blocks resources.
    if (enable_partial_fallback_) {
        for (auto& stream : waiting_streams_) {
            int need_block_num = (int)runningNextBlockNum() - (int)cache_manager_->freeBlockNums();
            if (need_block_num <= 0) {
                break;
            }
            if (stream->maxBlockSize()) {
                FT_LOG_INFO("lack mem, stream [%ld] in watting queue try release blocks, "
                    "it's input_length:%d seq_length:%d, hold block size:%d, release block size:%d",
                    stream->streamId(), stream->inputLength(), stream->seqLength(), stream->maxBlockSize(), need_block_num);
                stream->tryReleaseKVBlock(need_block_num);
            }
        }
    }

    while (!running_streams_.empty()) {
        int need_block_num = (int)runningNextBlockNum() - (int)cache_manager_->freeBlockNums();
        if (need_block_num <= 0) {
            break;
        }
        auto& last_stream = *(running_streams_.rbegin());
        int need_release_blocks = enable_partial_fallback_ ? need_block_num : last_stream->maxBlockSize();
        FT_LOG_INFO("lack mem, stream [%ld] fallback to wait, it's input_length:%d seq_length:%d, hold block size:%d, release block size:%d",
            last_stream->streamId(), last_stream->inputLength(), last_stream->seqLength(), last_stream->maxBlockSize(), need_release_blocks);
        last_stream->tryReleaseKVBlock(need_release_blocks);
        last_stream->setPaused();
        waiting_streams_.emplace_front(last_stream);
        running_streams_.pop_back();
    }

    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        if (!(*it)->incrKVBlock()) {
            (*it)->setStop("incrKVBlock failed");
            (*it)->releaseResource();
            FT_LOG_WARNING("stream [%ld] incr block failed", (*it)->streamId());
            it = running_streams_.erase(it);
        } else {
            it++;
        }
    }
}

bool FIFOScheduler::evaluateRunningMemory(int total_token_size) const {
    return total_token_size < max_seq_len_ * max_context_batch_size_;
}

bool FIFOScheduler::evaluateKVCacheMemory(int new_block_num) const {
    FT_LOG_DEBUG("stream runningNextBlockNum = %d, new_block_num = %d, cache_manager_->freeBlockNums() = %d",
        runningNextBlockNum(), new_block_num, cache_manager_->freeBlockNums());
    return runningNextBlockNum() + new_block_num <= cache_manager_->freeBlockNums();
}

// TODO(xinfei.sxf) When considering reuse cache, the evaluation is not accurate, so we should try to apply for resources immediately.
bool FIFOScheduler::evaluateNewStream(const list<GenerateStreamPtr>& streams,
                                      const GenerateStreamPtr&       new_stream) const {
    int total_token_size = new_stream->contextLength() + running_streams_.size();
    for (auto& stream : streams) {
        total_token_size += stream->contextLength();
    }

    FT_LOG_DEBUG("stream [%d], new_stream->nextNeedBlockNums() = %d, reserve_block_num = %d, "
    "total_token_size = %d, evaluateRunningMemory(total_token_size) = %d",
    new_stream->streamId(), new_stream->nextNeedBlockNums(), reserve_block_num_, total_token_size, evaluateRunningMemory(total_token_size));

    return evaluateKVCacheMemory(new_stream->nextNeedBlockNums() + reserve_block_num_)
           && evaluateRunningMemory(total_token_size) && new_stream->initKVBlock();
}

list<GenerateStreamPtr> FIFOScheduler::scheduleNew() {
    list<GenerateStreamPtr> new_streams;
    for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
        // TODO(xinfei.sxf) set detail stop reason to stream
        if (evaluateNewStream(new_streams, *it)) {
            FT_LOG_DEBUG("stream [%ld %p] add to new queue", (*it)->streamId(), (*it).get());
            // if setRunning fails, it must be in stopped state, evict it in next iteration
            if ((*it)->setRunning()) {
                new_streams.emplace_back(*it);
                it = waiting_streams_.erase(it);
            }
        } else if (running_streams_.empty() && new_streams.empty()) {
            // TODO(xinfei.sxf) At this time, you can also release the blocks held by other waiting streams
            FT_LOG_DEBUG("stream [%ld] can not add to new queue", (*it)->streamId());
            // TODO(xinfei.sxf) Return some tokens...
            (*it)->setStop("LACK MEM", absl::StatusCode::kResourceExhausted);
            (*it)->releaseResource();
            it++;
        } else {
            // try to join new streams in the next schedule cycle
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
    // TODO(xinfei.sxf) Those who just kicked out of running may join running again immediately.
    auto running_stream_size = running_streams_.size();
    evaluateRunningNext();
    auto fallback_stream_size = running_stream_size - running_streams_.size();
    auto new_stream = scheduleNew();
    running_streams_.insert(running_streams_.end(), new_stream.begin(), new_stream.end());
    reportMetrics(fallback_stream_size);
    return running_streams_;
}

int FIFOScheduler::waitingStreamsSize() {
    return waiting_streams_.size();
}

int FIFOScheduler::runningStreamsSize() {
    return running_streams_.size();
}

void FIFOScheduler::reportMetrics(size_t fallback_stream_size) {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.fallback_stream_size = fallback_stream_size;
        collector.running_stream_size = running_streams_.size();
        collector.wait_stream_size = waiting_streams_.size();
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
