#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/utils/Logger.h"
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
    reserve_block_num_(params.scheduler_reserve_resource_ratio_ * cache_manager->availableBlockNums() / 100),
    enable_partial_fallback_(params.enable_partial_fallback_),
    // not support enable_whole_fallback_ when use pd_speration
    enable_whole_fallback_(params.use_cache_store_ == false),
    enable_fast_gen_(params.enable_fast_gen_),
    fast_gen_max_context_len_(params.fast_gen_max_context_len_),
    metrics_reporter_(metrics_reporter) {}

FIFOScheduler::~FIFOScheduler() {
    (void)stop();
    FT_LOG_INFO("destory FIFOScheduler");
}

bool FIFOScheduler::empty() {
    return waiting_streams_.empty() && running_streams_.empty();
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

void FIFOScheduler::evaluateRunningRemote() {
    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        if ((*it)->needRemoteGenerate()) {
            (*it)->setRemoteGenerate();
            remote_running_streams_.emplace_back(*it);
            FT_LOG_DEBUG("evict remote running stream [%ld]", (*it)->streamId());
            it = running_streams_.erase(it);
        } else {
            ++it;
        }
    }
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

int FIFOScheduler::runningNextBlockNum(size_t reserve_step) const {
    int total_need_block_nums = 0;
    for (auto& stream : running_streams_) {
        total_need_block_nums += stream->nextNeedBlockNums(reserve_step);
    }
    return total_need_block_nums;
}

// TODO(xinfei.sxf) Is there any situation where the request cannot be ended?
tuple<int, int> FIFOScheduler::evaluateRunningNext(size_t reserve_step) {
    // Only in the case of partial fallback, the stream in the waiting queue may hold blocks resources.
    int fallback_streams = 0;
    int error_streams = 0;

    if (enable_partial_fallback_) {
        for (auto& stream : waiting_streams_) {
            int need_block_num = (int)runningNextBlockNum(reserve_step) - (int)cache_manager_->availableBlockNums();
            if (need_block_num <= 0) {
                break;
            }
            if (stream->maxBlockSize()) {
                FT_LOG_INFO("lack mem, stream [%ld] in watting queue try release blocks, "
                    "it's input_length:%d seq_length:%d, hold block size:%d, release block size:%d",
                    stream->streamId(), stream->inputLength(), stream->seqLength(), stream->maxBlockSize(), need_block_num);
                stream->tryReleaseKVBlock(need_block_num);
                fallback_streams++;
            }
        }
    }

    if (enable_whole_fallback_) {
        while (!running_streams_.empty()) {
            int need_block_num = (int)runningNextBlockNum(reserve_step) - (int)cache_manager_->availableBlockNums();
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
            fallback_streams++;
        }
    }

    if (enable_fast_gen_) {
        token_capacity_ = fast_gen_max_context_len_;
        FT_LOG_DEBUG("initial token_capacity is %d", token_capacity_);
    }

    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        auto result = (*it)->incrKVBlock(token_capacity_, reserve_step);
        if (!result.ok()) {
            (*it)->stopAndRelease(ErrorCode::MALLOC_FAILED, "incrKVBlock failed");
            FT_LOG_WARNING("stream [%ld] incr block failed", (*it)->streamId());
            it = running_streams_.erase(it);
            error_streams++;
        } else {
            if (enable_fast_gen_) {
                token_capacity_ -= result.value();
                FT_LOG_DEBUG("after stream [%d] acquireCapacity, token_capacity is %d", (*it)->streamId(), token_capacity_);
            }
            it++;
        }
    }
    return {fallback_streams, error_streams};
}

bool FIFOScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams,
                                         const GenerateStreamPtr&       new_stream) const {
    if (!enable_fast_gen_) {
        int max_token_size = new_stream->contextLength();
        for (auto& stream : streams) {
            max_token_size = std::max(max_token_size, stream->contextLength());
        }
        return max_token_size * (streams.size() + 1) + running_streams_.size() < int(max_seq_len_ * max_context_batch_size_);
    } else {
        return true;
    }
}

bool FIFOScheduler::evaluateNewStream(const list<GenerateStreamPtr>& streams,
                                      const GenerateStreamPtr&       new_stream,
                                      size_t reserve_step) {
    if (!evaluateRunningMemory(streams, new_stream)) {
        return false;
    }

    auto result = new_stream->initKVBlock(token_capacity_, reserve_step);
    if (result.ok() && enable_fast_gen_) {
        token_capacity_ -= result.value();
        FT_LOG_DEBUG("after stream [%d] acquireCapacity, token_capacity is %d", new_stream->streamId(), token_capacity_);
    }
    return result.ok() && cache_manager_->availableBlockNums() >= reserve_block_num_;
}

list<GenerateStreamPtr> FIFOScheduler::scheduleNew(size_t reserve_step) {
    list<GenerateStreamPtr> new_streams;
    for (auto it = waiting_streams_.begin(); it != waiting_streams_.end();) {
        auto& stream = *it;
        if (evaluateNewStream(new_streams, *it, reserve_step)) {
            FT_LOG_DEBUG("stream [%ld] add to new queue", stream->streamId());
            // if setRunning fails, it must be in stopped state, evict it in next iteration
            if (stream->setRunning()) {
                new_streams.emplace_back(stream);
                it = waiting_streams_.erase(it);
            } else {
                FT_LOG_WARNING("stream [%ld] set running failed", stream->streamId());
                stream->releaseResource();
                it++;
            }
        } else if (running_streams_.empty() && new_streams.empty() && remote_running_streams_.empty()) {
            // TODO(xinfei.sxf) At this time, we can also release the blocks held by other waiting streams
            FT_LOG_WARNING("stream [%ld] can not add to new queue", stream->streamId());
            if (stream->inputLength() > cache_manager_->maxSeqLen()) {
                stream->stopAndRelease(ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN, 
                    "input len " + std::to_string(stream->inputLength()) +
                    " is greater than kv cache max seq len " + std::to_string(cache_manager_->maxSeqLen()));
            } else {
                stream->stopAndRelease(ErrorCode::MALLOC_FAILED, "LACK MEM");
            }
            it++;
        } else {
            // try to join new streams in the next schedule cycle
            break;
        }
    }
    return new_streams;
}

absl::StatusOr<list<GenerateStreamPtr>> FIFOScheduler::schedule(size_t reserve_step) {
    unique_lock<mutex> lock(lock_);
    cond_.wait(lock, [this]{
        return stop_ || !waiting_streams_.empty() || !running_streams_.empty() || !remote_running_streams_.empty();
    });
    evaluateRunningRemote();
    evictDoneStreams(waiting_streams_);
    evictDoneStreams(running_streams_);
    evictDoneStreams(remote_running_streams_);

    // TODO(xinfei.sxf) Those who just kicked out of running may join running again immediately.
    auto [fallback_streams, error_streams] = evaluateRunningNext(reserve_step);
    auto new_stream = scheduleNew(reserve_step);
    running_streams_.insert(running_streams_.end(), new_stream.begin(), new_stream.end());
    reportMetrics(fallback_streams);
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
