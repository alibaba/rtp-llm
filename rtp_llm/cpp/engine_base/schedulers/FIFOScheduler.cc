#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include <chrono>
#include <mutex>

using namespace std;
namespace rtp_llm {

FIFOScheduler::FIFOScheduler(const RuntimeConfig&                   runtime_config,
                             const ModelConfig&                     model_config,
                             const PDSepConfig&                     pd_sep_config,
                             const ParallelismConfig&               parallelism_config,
                             const ModelSpecificConfig&             model_specific_config,
                             const std::shared_ptr<KVCacheManager>& cache_manager,
                             const kmonitor::MetricsReporterPtr     metrics_reporter,
                             const int                              max_score_len):
    FIFOSchedulerBase(runtime_config,
                      model_config,
                      pd_sep_config,
                      parallelism_config,
                      model_specific_config,
                      cache_manager,
                      metrics_reporter) {
    RTP_LLM_LOG_INFO("max_generate_batch_size is [%zu], max_batch_tokens_size is [%zu]",
                     max_generate_batch_size_,
                     max_batch_tokens_size_);
}

FIFOScheduler::~FIFOScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destory FIFOScheduler");
}

bool FIFOScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams,
                                          const GenerateStreamPtr&       new_stream) const {
    RTP_LLM_PROFILE_FUNCTION();
    if (pd_sep_config_.role_type == RoleType::DECODE) {
        if (running_streams_.size() + streams.size() + 1 < max_generate_batch_size_) {
            return true;
        }
    }
    // prefill and decode not mixed together
    if (!running_streams_.empty()) {
        return false;
    }
    if (running_streams_.size() + streams.size() + 1 > max_generate_batch_size_) {
        return false;
    }

    int max_token_size = new_stream->contextLength();
    if (streams.empty() && max_token_size + running_streams_.size() < int(max_seq_len_)) {
        return true;
    }
    for (auto& stream : streams) {
        max_token_size = std::max(max_token_size, stream->contextLength());
    }
    // 这里的判断是要求当前调度轮所有请求参与计算的 token 数之和小于 max_batch_tokens_size_，loading_cache_streams
    // 这一轮实际不参与计算，不需要计入。
    return max_token_size * (streams.size() + 1) + running_streams_.size() < int(max_batch_tokens_size_);
}

void FIFOScheduler::accountBatchMetrics(const GenerateStreamPtr& new_stream) {
    for (auto& stream : running_streams_) {
        stream->incBatchWithPrefillTimes(1);
        stream->incBatchWithPrefillLen(new_stream->currentExecuteTokenSize());
    }
}

void FIFOScheduler::onRunningStream(const GenerateStreamPtr& stream) {
    accountBatchMetrics(stream);
}

bool FIFOScheduler::waitPredicate() {
    // Check streams directly without calling empty() which acquires lock_ (already held by schedule())
    return stop_ || schedule_trigger_ || !waiting_streams_.empty() || !loading_cache_streams_.empty()
           || !running_streams_.empty();
}

absl::StatusOr<list<GenerateStreamPtr>> FIFOScheduler::schedule() {
    unique_lock<mutex> lock(lock_);
    if (need_fill_fake_stream_) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }

    schedule_trigger_ = false;

    // LOADING_CACHE -> DONE/WAITING: error / load cache done
    evaluateAndUpdateStreams(loading_cache_streams_);
    // RUNNING -> DONE: error / finished
    evaluateAndUpdateStreams(running_streams_);

    // WAITING -> RUNNING: can run
    // WAITING -> LOADING_CACHE: load cache ok
    //
    // Two-phase state transition for WAITING streams:
    //   Phase 1 (evaluateWaitingStreams): Streams that pass memory check get CanRun event,
    //       but are NOT removed from waiting_streams_ yet. This is because evaluateWaitingStreams
    //       iterates over waiting_streams_ and removing elements during iteration would be unsafe.
    //   Phase 2 (evaluateAndUpdateStreams): Actually moves streams from waiting_streams_ to
    //       their new state (RUNNING or LOADING_CACHE) based on the events set in Phase 1.
    // This separation ensures safe iteration while deferring structural modifications.
    size_t prev_waiting_size = waiting_streams_.size();
    evaluateWaitingStreams(waiting_streams_);
    evaluateAndUpdateStreams(waiting_streams_);
    running_streams_.insert(running_streams_.end(), new_streams_.begin(), new_streams_.end());
    new_streams_.clear();

    // If streams were scheduled, trigger next scheduling round
    if (waiting_streams_.size() < prev_waiting_size) {
        schedule_trigger_ = true;
    }

    reportMetrics();
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return running_streams_;
}

}  // namespace rtp_llm
