#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/cache/PrefillCacheHitMetricsReporter.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <functional>
#include <strings.h>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>
#include <limits.h>
#include <cstdint>
#include <map>

using namespace std;
using namespace autil::legacy;

using grpc::Status;
using grpc::ClientContext;

namespace rtp_llm {

PrefillRpcServer::~PrefillRpcServer() {
    stopAsyncResponseWorkers();
    stopResponseRegistryGc();
    if (enqueue_worker_pool_) {
        enqueue_worker_pool_->stop();
        enqueue_worker_pool_.reset();
    }
    if (worker_lambda_pool_) {
        worker_lambda_pool_->stop();
        worker_lambda_pool_.reset();
    }
    if (slot_worker_pool_) {
        slot_worker_pool_->stop();
        slot_worker_pool_.reset();
    }
}

namespace {

bool envValueIsTrue(const char* value) {
    return value != nullptr
           && (strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "on") == 0
               || strcasecmp(value, "yes") == 0);
}

bool envValueIsFalse(const char* value) {
    return value != nullptr
           && (strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "off") == 0
               || strcasecmp(value, "no") == 0);
}

bool prefillTraceLogEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PREFILL_TRACE_LOG_ENABLE");
        if (value == nullptr) {
            value = std::getenv("PREFILL_CACHE_DEBUG_LOG");
        }
        if (value == nullptr) {
            value = std::getenv("KV_CACHE_DEBUG_LOG");
        }
        return envValueIsTrue(value);
    }();
    return enabled;
}

bool prefillCacheDebugLogEnabled() {
    const bool enabled = prefillTraceLogEnabled();
    return enabled;
}

bool prefillTheoryHitLogEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PREFILL_THEORY_HIT_LOG_ENABLED");
        if (value == nullptr || value[0] == 0) {
            value = std::getenv("PREFILL_THEORY_HIT_LOG_ENABLE");
        }
        if (value == nullptr || value[0] == 0) {
            return true;
        }
        return !envValueIsFalse(value);
    }();
    return enabled;
}

const char* prefillTheoryHitLogPath() {
    const char* value = std::getenv("PREFILL_THEORY_HIT_LOG_PATH");
    if (value == nullptr || value[0] == 0) {
        return "/home/admin/logs/prefill_theory_hit.log";
    }
    return value;
}

double theoryHitRatio(int64_t hit_count, int64_t total_count) {
    return total_count > 0 ? static_cast<double>(hit_count) / static_cast<double>(total_count) : 0.0;
}

struct TheoryHitWindowSnapshot {
    const char* label       = "";
    int64_t     window_ms   = 0;
    int64_t     hit_count   = 0;
    int64_t     total_count = 0;
    double      hit_ratio   = 0.0;
};

struct TheoryHitStatsSnapshot {
    int64_t                 now_ms              = 0;
    int64_t                 request_hit_count   = 0;
    int64_t                 request_total_count = 0;
    double                  request_hit_ratio   = 0.0;
    int64_t                 all_hit_count       = 0;
    int64_t                 all_total_count     = 0;
    double                  all_hit_ratio       = 0.0;
    TheoryHitWindowSnapshot window_1m;
    TheoryHitWindowSnapshot window_5m;
    TheoryHitWindowSnapshot window_10m;
    TheoryHitWindowSnapshot window_15m;
};

void markResponseEntryDone(const std::shared_ptr<ResponseBufferEntry>& entry, const grpc::Status& status) {
    if (!entry) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(entry->mu);
        if (!status.ok()) {
            entry->error_status = status;
        }
        entry->done.store(true);
        entry->last_activity_us = currentTimeUs();
        entry->cancel_producer  = nullptr;
    }
    entry->cv.notify_all();
}

grpc::Status statusFromErrorInfo(const ErrorInfo& error_info) {
    if (!error_info.hasError()) {
        return grpc::Status::OK;
    }
    const auto     error_msg       = error_info.ToString();
    auto           grpc_error_code = transErrorCodeToGrpc(error_info.code());
    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(error_info.code()));
    error_details.set_error_message(error_msg);
    std::string error_details_serialized;
    if (error_details.SerializeToString(&error_details_serialized)) {
        return grpc::Status(grpc_error_code, error_msg, error_details_serialized);
    }
    RTP_LLM_LOG_WARNING(
        "statusFromErrorInfo error details serialize to string failed, error code [%s], error message [%s]",
        ErrorCodeToString(error_info.code()).c_str(),
        error_msg.c_str());
    return grpc::Status(grpc_error_code, error_msg);
}

void addBatchSuccess(EnqueueBatchResponsePB* response, int64_t request_id) {
    auto* success = response->add_successes();
    success->set_request_id(request_id);
}

void addBatchError(EnqueueBatchResponsePB* response, int64_t request_id, int64_t code, const std::string& msg) {
    auto* error = response->add_errors();
    error->set_request_id(request_id);
    auto* error_info = error->mutable_error_info();
    error_info->set_error_code(code);
    error_info->set_error_message(msg);
}

// Helper to detect whether a Future (std::future or autil Future) is ready.
// Uses SFINAE: std::future::wait_for returns std::future_status; autil
// Future::wait_for also returns std::future_status.
template<typename FutureT>
bool futureIsReady(FutureT& f) {
    return f.valid() && f.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

template<typename FutureT>
void detachLeftoverFutures(std::vector<FutureT>& futures) {
    // Clear leftover futures naturally; no longer creates a throwaway thread.
    // Futures that are not yet ready are simply abandoned — the thread pool
    // owns the actual work; the future object is just a handle.
    futures.clear();
}

template<typename FutureT>
void drainReadyFutures(std::vector<FutureT>& futures, std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        bool all_done  = true;
        bool any_ready = false;
        for (auto& f : futures) {
            if (f.valid()) {
                if (futureIsReady(f)) {
                    try {
                        f.get();
                    } catch (...) {}
                    any_ready = true;
                } else {
                    all_done = false;
                }
            }
        }
        if (all_done)
            break;
        if (!any_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

template<typename FutureT, typename OnReady, typename OnTimeout>
void collectFutures(std::vector<FutureT>&                 futures,
                    std::chrono::steady_clock::time_point deadline,
                    OnReady&&                             on_ready,
                    OnTimeout&&                           on_timeout) {
    std::vector<bool> collected(futures.size(), false);
    size_t            remaining = futures.size();
    while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
        bool any_ready = false;
        for (size_t i = 0; i < futures.size(); ++i) {
            if (!collected[i] && futureIsReady(futures[i])) {
                collected[i] = true;
                --remaining;
                any_ready = true;
                on_ready(i);
            }
        }
        if (remaining > 0 && !any_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    for (size_t i = 0; i < futures.size(); ++i) {
        if (!collected[i]) {
            if (futureIsReady(futures[i])) {
                on_ready(i);
            } else {
                on_timeout(i);
            }
        }
    }
}

}  // namespace

struct AsyncProducerCancelState {
    std::atomic<bool>                  cancelled{false};
    std::mutex                         mu;
    std::weak_ptr<grpc::ClientContext> client_context;
    std::weak_ptr<GenerateStream>      stream;
};

std::function<void()> makeAsyncProducerCancelCallback(const std::shared_ptr<AsyncProducerCancelState>& state) {
    return [state] {
        bool expected = false;
        if (!state->cancelled.compare_exchange_strong(expected, true)) {
            return;
        }

        std::shared_ptr<grpc::ClientContext> client_context;
        std::shared_ptr<GenerateStream>      stream;
        {
            std::lock_guard<std::mutex> lock(state->mu);
            client_context = state->client_context.lock();
            stream         = state->stream.lock();
        }
        if (client_context) {
            client_context->TryCancel();
        }
        if (stream) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled");
        }
    };
}

void refreshAsyncProducerCancelState(const std::shared_ptr<AsyncProducerCancelState>& state,
                                     const std::shared_ptr<grpc::ClientContext>&      client_context,
                                     const std::shared_ptr<GenerateStream>&           stream) {
    bool should_cancel = false;
    {
        std::lock_guard<std::mutex> lock(state->mu);
        state->client_context = client_context;
        state->stream         = stream;
        should_cancel         = state->cancelled.load();
    }
    if (should_cancel) {
        if (client_context) {
            client_context->TryCancel();
        }
        if (stream) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled");
        }
    }
}

class ScopeExit {
public:
    explicit ScopeExit(std::function<void()> fn): fn_(std::move(fn)) {}
    ~ScopeExit() {
        if (fn_) {
            fn_();
        }
    }
    ScopeExit(const ScopeExit&)            = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

private:
    std::function<void()> fn_;
};

void cancelResponseEntry(const std::shared_ptr<ResponseBufferEntry>& entry) {
    if (!entry) {
        return;
    }
    std::function<void()> cancel_producer;
    {
        std::lock_guard<std::mutex> lock(entry->mu);
        entry->cancelled.store(true);
        entry->last_activity_us = currentTimeUs();
        cancel_producer         = entry->cancel_producer;
        entry->cancel_producer  = nullptr;
    }
    if (cancel_producer) {
        cancel_producer();
    }
    entry->cv.notify_all();
}

namespace {

class TheoryHitStats {
public:
    TheoryHitStats() {
        bucket_seconds_.fill(std::numeric_limits<int64_t>::min());
        bucket_hit_counts_.fill(0);
        bucket_total_counts_.fill(0);
    }

    TheoryHitStatsSnapshot record(int64_t hit_count, int64_t total_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t               now_ms         = autil::TimeUtility::currentTimeInMilliSeconds();
        const int64_t               current_second = now_ms / 1000;
        const int64_t               safe_hit       = std::max<int64_t>(0, hit_count);
        const int64_t               safe_total     = std::max<int64_t>(0, total_count);

        if (safe_total > 0) {
            const size_t index = static_cast<size_t>(current_second % kBucketCount);
            if (bucket_seconds_[index] != current_second) {
                bucket_seconds_[index]      = current_second;
                bucket_hit_counts_[index]   = 0;
                bucket_total_counts_[index] = 0;
            }
            bucket_hit_counts_[index] += safe_hit;
            bucket_total_counts_[index] += safe_total;
            all_hit_count_ += safe_hit;
            all_total_count_ += safe_total;
        }

        TheoryHitStatsSnapshot snapshot;
        snapshot.now_ms              = now_ms;
        snapshot.request_hit_count   = safe_hit;
        snapshot.request_total_count = safe_total;
        snapshot.request_hit_ratio   = theoryHitRatio(safe_hit, safe_total);
        snapshot.all_hit_count       = all_hit_count_;
        snapshot.all_total_count     = all_total_count_;
        snapshot.all_hit_ratio       = theoryHitRatio(all_hit_count_, all_total_count_);
        snapshot.window_1m           = windowSnapshot("1m", 60 * 1000, current_second);
        snapshot.window_5m           = windowSnapshot("5m", 5 * 60 * 1000, current_second);
        snapshot.window_10m          = windowSnapshot("10m", 10 * 60 * 1000, current_second);
        snapshot.window_15m          = windowSnapshot("15m", 15 * 60 * 1000, current_second);
        return snapshot;
    }

private:
    static constexpr size_t kBucketCount = 15 * 60 + 2;

    TheoryHitWindowSnapshot windowSnapshot(const char* label, int64_t window_ms, int64_t current_second) const {
        TheoryHitWindowSnapshot snapshot;
        snapshot.label               = label;
        snapshot.window_ms           = window_ms;
        const int64_t window_seconds = window_ms / 1000;
        for (size_t i = 0; i < kBucketCount; ++i) {
            const int64_t age_seconds = current_second - bucket_seconds_[i];
            if (age_seconds >= 0 && age_seconds < window_seconds) {
                snapshot.hit_count += bucket_hit_counts_[i];
                snapshot.total_count += bucket_total_counts_[i];
            }
        }
        snapshot.hit_ratio = theoryHitRatio(snapshot.hit_count, snapshot.total_count);
        return snapshot;
    }

private:
    std::array<int64_t, kBucketCount> bucket_seconds_;
    std::array<int64_t, kBucketCount> bucket_hit_counts_;
    std::array<int64_t, kBucketCount> bucket_total_counts_;
    int64_t                           all_hit_count_   = 0;
    int64_t                           all_total_count_ = 0;
    std::mutex                        mutex_;
};

std::string formatTheoryTimestampMs(int64_t timestamp_ms) {
    const time_t seconds = static_cast<time_t>(timestamp_ms / 1000);
    struct tm    local_time;
    if (localtime_r(&seconds, &local_time) == nullptr) {
        return std::to_string(timestamp_ms);
    }

    char date_buffer[64];
    if (strftime(date_buffer, sizeof(date_buffer), "%Y-%m-%dT%H:%M:%S", &local_time) == 0) {
        return std::to_string(timestamp_ms);
    }

    char offset_buffer[16];
    if (strftime(offset_buffer, sizeof(offset_buffer), "%z", &local_time) == 0) {
        offset_buffer[0] = '\0';
    }

    char output[96];
    snprintf(output, sizeof(output), "%s.%03ld%s", date_buffer, static_cast<long>(timestamp_ms % 1000), offset_buffer);
    return output;
}

void appendPrefillTheoryHitLogLine(const std::string& line) {
    if (!prefillTheoryHitLogEnabled()) {
        return;
    }
    static std::mutex    log_mutex;
    static std::ofstream log_file;
    static bool          open_failed = false;

    std::lock_guard<std::mutex> lock(log_mutex);
    if (!log_file.is_open() && !open_failed) {
        const char* path = prefillTheoryHitLogPath();
        log_file.open(path, std::ios::out | std::ios::app);
        if (!log_file.is_open()) {
            open_failed = true;
            RTP_LLM_LOG_WARNING("Failed to open prefill theory hit log path: %s", path);
            return;
        }
        RTP_LLM_LOG_INFO("Prefill theory hit log path: %s", path);
    }
    if (!log_file.is_open()) {
        return;
    }
    log_file << line << '\n';
    log_file.flush();
}

std::string formatPrefillTheoryHitLogLine(PrefillGenerateContext&       prefill_context,
                                          int64_t                       token_num,
                                          int                           seq_size_per_block,
                                          const TheoryHitStatsSnapshot& snapshot) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << "time=" << formatTheoryTimestampMs(snapshot.now_ms)
        << " ts_ms=" << snapshot.now_ms << " source=prefill"
        << " request_id=" << prefill_context.request_id << " request_key=" << prefill_context.request_key
        << " token_num=" << token_num << " seq_size_per_block=" << seq_size_per_block
        << " request_hit=" << snapshot.request_hit_count << " request_total=" << snapshot.request_total_count
        << " request_ratio=" << snapshot.request_hit_ratio << " all_hit=" << snapshot.all_hit_count
        << " all_total=" << snapshot.all_total_count << " all_ratio=" << snapshot.all_hit_ratio
        << " win1m_hit=" << snapshot.window_1m.hit_count << " win1m_total=" << snapshot.window_1m.total_count
        << " win1m_ratio=" << snapshot.window_1m.hit_ratio << " win5m_hit=" << snapshot.window_5m.hit_count
        << " win5m_total=" << snapshot.window_5m.total_count << " win5m_ratio=" << snapshot.window_5m.hit_ratio
        << " win10m_hit=" << snapshot.window_10m.hit_count << " win10m_total=" << snapshot.window_10m.total_count
        << " win10m_ratio=" << snapshot.window_10m.hit_ratio << " win15m_hit=" << snapshot.window_15m.hit_count
        << " win15m_total=" << snapshot.window_15m.total_count << " win15m_ratio=" << snapshot.window_15m.hit_ratio;
    return oss.str();
}

std::string cacheKeyPreview(const std::vector<CacheKeyType>& keys, size_t limit = 6) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < keys.size() && i < limit; ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << keys[i];
    }
    if (keys.size() > limit) {
        oss << ",...";
    }
    oss << "]";
    return oss.str();
}

std::string cacheKeysToString(const std::vector<CacheKeyType>& keys) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << keys[i];
    }
    oss << "]";
    return oss.str();
}

std::string cacheKeyDigest(const std::vector<CacheKeyType>& keys) {
    uint64_t digest = 14695981039346656037ULL;
    for (const auto cache_key : keys) {
        uint64_t value = static_cast<uint64_t>(cache_key);
        digest ^= value;
        digest *= 1099511628211ULL;
        digest ^= value >> 32;
        digest *= 1099511628211ULL;
    }
    return std::to_string(digest);
}

const char* prefillStageName(PrefillStatInfo::ExecuteStage stage) {
    switch (stage) {
        case PrefillStatInfo::start:
            return "start";
        case PrefillStatInfo::getRpcConnection:
            return "getRpcConnection";
        case PrefillStatInfo::multimodalProcess:
            return "multimodalProcess";
        case PrefillStatInfo::remoteAllocateResource:
            return "remoteAllocateResource";
        case PrefillStatInfo::enqueueRequest:
            return "enqueueRequest";
        case PrefillStatInfo::remoteLoadCacheStart:
            return "remoteLoadCacheStart";
        case PrefillStatInfo::pollLocalOutput:
            return "pollLocalOutput";
        case PrefillStatInfo::remoteLoadCacheEnd:
            return "remoteLoadCacheEnd";
        case PrefillStatInfo::RemoteGenerate:
            return "RemoteGenerate";
        case PrefillStatInfo::pollRemoteOutput:
            return "pollRemoteOutput";
        case PrefillStatInfo::finish:
            return "finish";
        default:
            return "unknown";
    }
}

void logPrefillFailureTrace(const char* event, PrefillGenerateContext& prefill_context) {
    if (!prefillTraceLogEnabled()) {
        return;
    }
    RTP_LLM_LOG_WARNING("Prefill request trace: event=%s request_id=%ld request_key=%s stage=%s retry_times=%ld "
                        "retry_cost_time_ms=%ld execute_time_ms=%ld decode_addr=%s grpc_code=%d grpc_message=%s "
                        "error_code=%d error_message=%s",
                        event,
                        prefill_context.request_id,
                        prefill_context.request_key.c_str(),
                        prefillStageName(prefill_context.stat_info.stage),
                        prefill_context.retry_times,
                        prefill_context.retry_cost_time_ms,
                        prefill_context.executeTimeMs(),
                        prefill_context.decode_addr.c_str(),
                        static_cast<int>(prefill_context.error_status.error_code()),
                        prefill_context.error_status.error_message().c_str(),
                        static_cast<int>(prefill_context.error_info.code()),
                        prefill_context.error_info.ToString().c_str());
}

std::vector<CacheKeyType> buildFullBlockCacheKeys(torch::Tensor input_ids, int seq_size_per_block) {
    std::vector<CacheKeyType> cache_keys;
    if (seq_size_per_block <= 0 || !input_ids.defined() || input_ids.numel() <= 0) {
        return cache_keys;
    }

    if (!input_ids.device().is_cpu()) {
        input_ids = input_ids.cpu();
    }
    if (!input_ids.is_contiguous()) {
        input_ids = input_ids.contiguous();
    }
    if (input_ids.scalar_type() != torch::kInt32) {
        input_ids = input_ids.to(torch::kInt32);
    }

    const int64_t token_num   = input_ids.numel();
    const int64_t block_count = token_num / seq_size_per_block;
    if (block_count <= 0) {
        return cache_keys;
    }
    cache_keys.reserve(static_cast<size_t>(block_count));

    auto*   token_ids    = input_ids.data_ptr<int32_t>();
    int64_t rolling_hash = 0;
    for (int64_t block_idx = 0; block_idx < block_count; ++block_idx) {
        const int64_t pos = block_idx * seq_size_per_block;
        rolling_hash      = rtp_llm::hashInt64Array(
            rolling_hash, token_ids + pos, token_ids + pos + static_cast<int64_t>(seq_size_per_block));
        cache_keys.push_back(static_cast<CacheKeyType>(rolling_hash));
    }
    return cache_keys;
}

void fillPrefillRecentCacheKeyMetricsCollector(PrefillRecentCacheKeyMetricsCollector& collector,
                                               const RecentCacheKeyWindow::Snapshot&  snapshot) {
    collector.has_value                  = true;
    collector.request_count              = true;
    collector.empty_request_count        = snapshot.request_occurrences == 0;
    collector.hit_count                  = snapshot.request_hit_occurrences;
    collector.total_count                = snapshot.request_occurrences;
    collector.hit_ratio                  = snapshot.request_hit_ratio;
    collector.retained_occurrences       = snapshot.retained_occurrences;
    collector.retained_unique_cache_keys = static_cast<int64_t>(snapshot.retained_unique_cache_keys);
    collector.time_window_ms             = snapshot.time_window_ms;
}

void fillPrefillTheoryHitMetricsCollector(PrefillRecentCacheKeyMetricsCollector& collector,
                                          const TheoryHitStatsSnapshot&          snapshot) {
    if (snapshot.all_total_count <= 0) {
        return;
    }
    collector.theory_has_value       = true;
    collector.theory_all_hit_count   = snapshot.all_hit_count;
    collector.theory_all_total_count = snapshot.all_total_count;
    collector.theory_all_hit_ratio   = snapshot.all_hit_ratio;
}

}  // namespace

#define CLIENT_GRPC_RET_IF_ERROR(prefill_context, state, error_code_value)                                             \
    if (!(state)) {                                                                                                    \
        auto   new_error_code = error_code_value;                                                                      \
        string new_error_msg  = "decode addr is " + prefill_context.decode_addr + ", ";                                \
        new_error_msg += "execute time is " + std::to_string(prefill_context.executeTimeMs()) + "ms, ";                \
        new_error_msg += "request timeout is " + std::to_string(prefill_context.request_timeout_ms) + "ms, ";          \
        new_error_msg += "rpc connection pointer is "                                                                  \
                         + std::to_string((int64_t)prefill_context.grpc_connection.channel.get()) + ", ";              \
        if (prefill_context.getStream()) {                                                                             \
            auto first_token_rt_ms = prefill_context.getStream()->getTimeInfo().first_token_rt_us / 1000;              \
            if (first_token_rt_ms) {                                                                                   \
                new_error_msg += "stream first token rt is " + std::to_string(first_token_rt_ms) + "ms, ";             \
            }                                                                                                          \
            auto wait_time_ms = prefill_context.getStream()->getTimeInfo().wait_time_us / 1000;                        \
            if (wait_time_ms) {                                                                                        \
                new_error_msg += "stream wait time is " + std::to_string(wait_time_ms) + "ms, ";                       \
            }                                                                                                          \
        }                                                                                                              \
        auto status = prefill_context.closeGrpcStream();                                                               \
        if (!status.ok()) {                                                                                            \
            const auto& error_msg = status.error_message();                                                            \
            if (error_msg.find("Connect Failed") != std::string::npos) {                                               \
                new_error_code = ErrorCode::CONNECT_FAILED;                                                            \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("No route to host") != std::string::npos) {                                      \
                new_error_code = ErrorCode::CONNECT_FAILED;                                                            \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("Connection reset by peer") != std::string::npos) {                              \
                new_error_code = ErrorCode::CONNECTION_RESET_BY_PEER;                                                  \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("Connection timed out") != std::string::npos) {                                  \
                new_error_code = ErrorCode::CONNECT_TIMEOUT;                                                           \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("Deadline Exceeded") != std::string::npos) {                                     \
                new_error_code = ErrorCode::DEADLINE_EXCEEDED;                                                         \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("keepalive watchdog timeout") != std::string::npos) {                            \
                new_error_code = ErrorCode::KEEP_ALIVE_TIMEOUT;                                                        \
                prefill_context.closeGrpcConnection();                                                                 \
            }                                                                                                          \
            new_error_msg += error_msg;                                                                                \
            if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {                                         \
                new_error_code = ErrorCode::DECODE_MALLOC_FAILED;                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            if (prefill_context.client_stream) {                                                                       \
                new_error_msg += "server disconnected with status::ok";                                                \
            }                                                                                                          \
        }                                                                                                              \
        if (prefill_context.getStream()) {                                                                             \
            prefill_context.getStream()->reportEvent(StreamEvents::Error, new_error_code, new_error_msg);              \
        }                                                                                                              \
        prefill_context.error_info = ErrorInfo(new_error_code, new_error_msg);                                         \
        prefill_context.error_status =                                                                                 \
            serializeErrorMsg(prefill_context.request_key, prefill_context.request_info, prefill_context.error_info);  \
        logPrefillFailureTrace("client_grpc_error", prefill_context);                                                  \
        return;                                                                                                        \
    }

void PrefillRpcServer::startResponseRegistryGc() {
    if (response_gc_thread_.joinable()) {
        return;
    }
    response_gc_stop_.store(false);
    response_gc_thread_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(response_gc_mu_);
        int                          gc_counter = 0;
        while (!response_gc_stop_.load()) {
            response_gc_cv_.wait_for(lock, std::chrono::seconds(10), [this] { return response_gc_stop_.load(); });
            if (response_gc_stop_.load()) {
                break;
            }
            lock.unlock();
            reportPoolMetrics();
            gc_counter++;
            if (gc_counter >= 6) {  // GC every 60 seconds
                response_registry_.gc(std::chrono::minutes(10));
                gc_counter = 0;
            }
            lock.lock();
        }
    });
}

void PrefillRpcServer::stopResponseRegistryGc() {
    response_gc_stop_.store(true);
    response_gc_cv_.notify_all();
    if (response_gc_thread_.joinable()) {
        response_gc_thread_.join();
    }
}

bool PrefillRpcServer::tryStartAsyncResponseWorker() {
    std::lock_guard<std::mutex> lock(response_worker_mu_);
    if (response_worker_stop_.load()) {
        return false;
    }
    ++response_worker_count_;
    return true;
}

void PrefillRpcServer::finishAsyncResponseWorker() {
    {
        std::lock_guard<std::mutex> lock(response_worker_mu_);
        if (response_worker_count_ > 0) {
            --response_worker_count_;
        }
    }
    response_worker_cv_.notify_all();
}

void PrefillRpcServer::stopAsyncResponseWorkers() {
    {
        std::lock_guard<std::mutex> lock(response_worker_mu_);
        response_worker_stop_.store(true);
    }
    response_registry_.cancelAll();

    static constexpr auto        kStopTimeout = std::chrono::seconds(30);
    std::unique_lock<std::mutex> lock(response_worker_mu_);
    bool all_done = response_worker_cv_.wait_for(lock, kStopTimeout, [this] { return response_worker_count_ == 0; });

    if (!all_done) {
        RTP_LLM_LOG_WARNING("stopAsyncResponseWorkers: timeout after %lds, still %zu workers active. Force resetting.",
                            kStopTimeout.count(),
                            response_worker_count_);
        response_worker_count_ = 0;
        // Notify other waiters that we've force-reset
        response_worker_cv_.notify_all();
    }
}

std::string PrefillRpcServer::batchTargetAddrForDpRank(int dp_rank) const {
    if (dp_rank < 0 || dp_rank >= maga_init_params_.parallelism_config.dp_size) {
        return "";
    }
    const auto&   all_workers = maga_init_params_.runtime_config.all_worker_grpc_addrs;
    const int64_t tp_size     = std::max<int64_t>(1, maga_init_params_.parallelism_config.tp_size);
    const int64_t world_rank  = static_cast<int64_t>(dp_rank) * tp_size;
    if (world_rank >= 0 && world_rank < static_cast<int64_t>(all_workers.size())) {
        return all_workers[world_rank];
    }
    if (dp_rank == maga_init_params_.parallelism_config.dp_rank
        && !maga_init_params_.runtime_config.worker_grpc_addrs.empty()) {
        return maga_init_params_.runtime_config.worker_grpc_addrs.front();
    }
    return "";
}

grpc::Status PrefillRpcServer::init(const EngineInitParams&                                maga_init_params,
                                    py::object                                             mm_process_engine,
                                    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    RTP_LLM_CHECK_WITH_INFO(maga_init_params.pd_sep_config.role_type == RoleType::PREFILL,
                            "prefill's role_type must be PREFILL");
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        return ret;
    }
    initThreadPools();
    if (PrefillCacheHitMetricsReporter::enabled()) {
        prefill_recent_cache_key_window_ = std::make_unique<RecentCacheKeyWindow>();
    } else {
        RTP_LLM_LOG_INFO("prefill recent-cache-key metrics disabled by PREFILL_CACHE_HIT_METRIC_ENABLE");
    }
    startResponseRegistryGc();
    return grpc::Status::OK;
}

void PrefillRpcServer::initThreadPools() {
    const auto& pd_sep_config     = maga_init_params_.pd_sep_config;
    const int   concurrency_limit = std::max(1, maga_init_params_.concurrency_config.concurrency_limit);

    // enqueue pool: L1 DP dispatch only (fast, ms-level, must never block)
    // Configurable via pd_sep_config.prefill_enqueue_pool_size (0 = use formula default)
    const int enqueue_threads = pd_sep_config.prefill_enqueue_pool_size > 0 ?
                                    static_cast<int>(pd_sep_config.prefill_enqueue_pool_size) :
                                    std::max(4, concurrency_limit * 4);
    const int enqueue_queue   = enqueue_threads * 2;

    enqueue_worker_pool_ =
        std::make_shared<autil::LockFreeThreadPool>(enqueue_threads, enqueue_queue, nullptr, "PrefillEnqueuePool");
    RTP_LLM_CHECK_WITH_INFO(enqueue_worker_pool_->start(), "PrefillRpcServer enqueue thread pool start failed");
    RTP_LLM_LOG_INFO("PrefillRpcServer enqueue pool started: threads=%d queue=%d (concurrency_limit=%d)",
                     enqueue_threads,
                     enqueue_queue,
                     concurrency_limit);
    enqueue_pool_metrics_.thread_max = static_cast<size_t>(enqueue_threads);
    enqueue_pool_metrics_.queue_max  = static_cast<size_t>(enqueue_queue);

    // worker lambda pool: heavy EnqueueGroup coordination (I/O-bound, ~12s per batch)
    // Configurable via pd_sep_config.prefill_worker_lambda_pool_size (0 = use formula default)
    const int worker_lambda_threads = pd_sep_config.prefill_worker_lambda_pool_size > 0 ?
                                          static_cast<int>(pd_sep_config.prefill_worker_lambda_pool_size) :
                                          std::max(4, concurrency_limit * 4);
    const int worker_lambda_queue   = worker_lambda_threads * 4;

    worker_lambda_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        worker_lambda_threads, worker_lambda_queue, nullptr, "PrefillWorkerPool");
    RTP_LLM_CHECK_WITH_INFO(worker_lambda_pool_->start(), "PrefillRpcServer worker lambda pool start failed");
    RTP_LLM_LOG_INFO("PrefillRpcServer worker lambda pool started: threads=%d queue=%d (concurrency_limit=%d)",
                     worker_lambda_threads,
                     worker_lambda_queue,
                     concurrency_limit);
    worker_lambda_pool_metrics_.thread_max = static_cast<size_t>(worker_lambda_threads);
    worker_lambda_pool_metrics_.queue_max  = static_cast<size_t>(worker_lambda_queue);

    // slot pool: L2 Prepare + L3 Load + L4 Finish
    // Configurable via pd_sep_config.prefill_slot_pool_size (0 = use formula default)
    const int slot_threads = pd_sep_config.prefill_slot_pool_size > 0 ?
                                 static_cast<int>(pd_sep_config.prefill_slot_pool_size) :
                                 std::max(16, concurrency_limit * 8);
    const int slot_queue   = slot_threads * 8;

    slot_worker_pool_ =
        std::make_shared<autil::LockFreeThreadPool>(slot_threads, slot_queue, nullptr, "PrefillSlotPool");
    RTP_LLM_CHECK_WITH_INFO(slot_worker_pool_->start(), "PrefillRpcServer slot thread pool start failed");
    RTP_LLM_LOG_INFO("PrefillRpcServer slot pool started: threads=%d queue=%d (concurrency_limit=%d)",
                     slot_threads,
                     slot_queue,
                     concurrency_limit);
    slot_pool_metrics_.thread_max = static_cast<size_t>(slot_threads);
    slot_pool_metrics_.queue_max  = static_cast<size_t>(slot_queue);
}

void PrefillRpcServer::reportOnePoolToKmonitor(const std::string& pool_name, const PoolMetrics& metrics) {
    if (!metrics_reporter_) {
        return;
    }
    PrefillPoolMetricsCollector collector;
    collector.active     = static_cast<int64_t>(metrics.active.load());
    collector.queued     = static_cast<int64_t>(metrics.queued.load());
    collector.completed  = static_cast<int64_t>(metrics.completed.load());
    collector.rejected   = static_cast<int64_t>(metrics.rejected.load());
    collector.fallback   = static_cast<int64_t>(metrics.fallback.load());
    collector.thread_max = static_cast<int64_t>(metrics.thread_max);
    collector.queue_max  = static_cast<int64_t>(metrics.queue_max);
    kmonitor::MetricsTags tags("pool_name", pool_name);
    metrics_reporter_->report<PrefillPoolMetrics, PrefillPoolMetricsCollector>(&tags, &collector);
}

void PrefillRpcServer::reportPoolMetrics() {
    // Report to kmonitor (called every 10s from GC thread)
    reportOnePoolToKmonitor("enqueue", enqueue_pool_metrics_);
    reportOnePoolToKmonitor("worker_lambda", worker_lambda_pool_metrics_);
    reportOnePoolToKmonitor("slot", slot_pool_metrics_);

    // Debug-level log for troubleshooting (avoid INFO noise on production)
    RTP_LLM_LOG_DEBUG("PoolMetrics enqueue: active=%zu queued=%zu completed=%zu rejected=%zu fallback=%zu "
                      "thread_max=%zu queue_max=%zu",
                      enqueue_pool_metrics_.active.load(),
                      enqueue_pool_metrics_.queued.load(),
                      enqueue_pool_metrics_.completed.load(),
                      enqueue_pool_metrics_.rejected.load(),
                      enqueue_pool_metrics_.fallback.load(),
                      enqueue_pool_metrics_.thread_max,
                      enqueue_pool_metrics_.queue_max);
    RTP_LLM_LOG_DEBUG("PoolMetrics worker_lambda: active=%zu queued=%zu completed=%zu rejected=%zu fallback=%zu "
                      "thread_max=%zu queue_max=%zu",
                      worker_lambda_pool_metrics_.active.load(),
                      worker_lambda_pool_metrics_.queued.load(),
                      worker_lambda_pool_metrics_.completed.load(),
                      worker_lambda_pool_metrics_.rejected.load(),
                      worker_lambda_pool_metrics_.fallback.load(),
                      worker_lambda_pool_metrics_.thread_max,
                      worker_lambda_pool_metrics_.queue_max);
    RTP_LLM_LOG_DEBUG(
        "PoolMetrics slot: active=%zu queued=%zu completed=%zu rejected=%zu fallback=%zu response_workers=%zu "
        "thread_max=%zu queue_max=%zu",
        slot_pool_metrics_.active.load(),
        slot_pool_metrics_.queued.load(),
        slot_pool_metrics_.completed.load(),
        slot_pool_metrics_.rejected.load(),
        slot_pool_metrics_.fallback.load(),
        response_worker_count_,
        slot_pool_metrics_.thread_max,
        slot_pool_metrics_.queue_max);
}

ErrorInfo PrefillRpcServer::waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream) {
    static int max_wait_timeout_us = maga_init_params_.pd_sep_config.prefill_max_wait_timeout_ms * 1000;
    auto       begin_time_us       = currentTimeUs();
    while (!stream->hasError() && stream->getStatus() == StreamState::WAITING) {
        usleep(100);
        auto current_time_us = currentTimeUs();
        auto cost_time_us    = current_time_us - begin_time_us;
        if (cost_time_us > max_wait_timeout_us) {
            string new_error_msg = "wait to run timeout, timeout is " + std::to_string(max_wait_timeout_us) + " us";
            stream->reportEvent(StreamEvents::Error, ErrorCode::WAIT_TO_RUN_TIMEOUT, new_error_msg);
            return ErrorInfo(ErrorCode::WAIT_TO_RUN_TIMEOUT, new_error_msg);
        }
    }
    if (stream->hasError()) {
        return stream->statusInfo();
    }
    return ErrorInfo::OkStatus();
}

void PrefillRpcServer::getRpcConnection(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] trans query", prefill_context.request_id);
    auto input                   = QueryConverter::transQuery(prefill_context.rpc_context.request);
    prefill_context.request_info = input->request_info;
    if (applyTimelineGate(prefill_context.request_key,
                          input->generate_config->gen_timeline,
                          input->generate_config->profile_step,
                          input->generate_config->profile_trace_name)) {
        input->generate_config->gen_timeline = true;
    }
    input->generate_config->pd_separation = true;
    if (engine_->isMTPEagle()) {
        input->generate_config->force_disable_sp_run = false;
    } else {
        input->generate_config->force_disable_sp_run = true;
    }
    prefill_context.generate_input = input;

    RTP_LLM_LOG_DEBUG("request [%ld] get rpc connection", prefill_context.request_id);

    auto&                       role_addrs = prefill_context.generate_input->generate_config->role_addrs;
    std::shared_ptr<const Host> host;

    // Check if request specifies host for DECODE role
    for (auto& role_addr : role_addrs) {
        if (role_addr.role == RoleType::DECODE) {
            host = std::make_shared<const Host>(role_addr.ip, role_addr.grpc_port, role_addr.http_port);
            break;
        }
    }

    // If no host specified in request, check if there's a master role
    char* remote_rpc_server_ip_env = std::getenv("REMOTE_RPC_SERVER_IP");
    bool  has_master_role          = (remote_rpc_server_ip_env != nullptr && strlen(remote_rpc_server_ip_env) > 0);

    // If no host specified in request and no master role, this is a direct prefill request
    // In this case, we still need to select decode machines as specified in the requirements
    if (!host && !has_master_role) {
        // For direct prefill requests without master role, we still need to select decode machines
        // The current logic will fail as expected since no host is available
        RTP_LLM_LOG_DEBUG(
            "request [%ld] no host specified in request and no master role, need to select decode machines",
            prefill_context.request_id);
    }

    if (!host || host->ip.empty()) {
        prefill_context.error_info =
            ErrorInfo(ErrorCode::GET_HOST_FAILED, "get host for decode cluster " + decode_cluster_name_ + " failed");
        prefill_context.error_status =
            serializeErrorMsg(prefill_context.request_key, prefill_context.request_info, prefill_context.error_info);
        logPrefillFailureTrace("get_rpc_connection_no_decode_host", prefill_context);
        return;
    }
    auto decode_addr    = host->ip + ":" + std::to_string(host->rpc_port);
    auto connect_status = resource_.rpc_pool.getConnection(decode_addr);
    if (!connect_status.ok()) {
        prefill_context.error_info = ErrorInfo(ErrorCode::GET_CONNECTION_FAILED,
                                               "get grpc connection for decode addr " + decode_addr + " failed");
        prefill_context.error_status =
            serializeErrorMsg(prefill_context.request_key, prefill_context.request_info, prefill_context.error_info);
        prefill_context.decode_addr = decode_addr;
        logPrefillFailureTrace("get_rpc_connection_failed", prefill_context);
        return;
    }
    prefill_context.decode_addr     = decode_addr;
    prefill_context.grpc_connection = connect_status.value();

    RTP_LLM_LOG_DEBUG("request [%ld] get rpc connection done", prefill_context.request_id);
}

void PrefillRpcServer::multimodalProcess(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    auto& input = prefill_context.generate_input;
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto result = mm_processor_->updateMultimodalFeatures(input);
        CLIENT_GRPC_RET_IF_ERROR(prefill_context, result.ok(), result.code());

        auto mutable_request = const_cast<GenerateInputPB*>(prefill_context.rpc_context.request);
        mutable_request->clear_token_ids();
        // TODO(xinfei.sxf) optimize copy
        auto* ids_ptr = input->input_ids.data_ptr<int32_t>();
        for (size_t i = 0; i < input->input_ids.numel(); i++) {
            mutable_request->add_token_ids(ids_ptr[i]);
        }
    }
    if (prefill_context.hasError()) {
        logPrefillFailureTrace("multimodal_process_failed", prefill_context);
    }
}

void PrefillRpcServer::remoteAllocateResource(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to remote allocate resource", prefill_context.request_id);
    prefill_context.client_context.reset(new ClientContext());
    auto    request_timeout_ms = prefill_context.request_timeout_ms;
    auto    max_rpc_timeout_ms = maga_init_params_.pd_sep_config.max_rpc_timeout_ms;
    int64_t final_timeout_ms   = request_timeout_ms > 0 ? request_timeout_ms : max_rpc_timeout_ms;
    if (final_timeout_ms > 0) {
        auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(final_timeout_ms);
        prefill_context.client_context->set_deadline(deadline);
    }
    if (prefill_context.cancel_state) {
        refreshAsyncProducerCancelState(
            prefill_context.cancel_state, prefill_context.client_context, prefill_context.getStream());
    }
    // final_timeout_ms <= 0: skip set_deadline; gRPC treats it as no deadline.
    prefill_context.client_stream =
        std::move(prefill_context.grpc_connection.stub->RemoteGenerate(prefill_context.client_context.get()));
    auto&             client_stream = prefill_context.client_stream;
    GenerateRequestPB alloc_request;
    alloc_request.set_stage(RemoteStage::ALLOCATE);
    alloc_request.set_client_id(process_id_);
    alloc_request.set_request_id(prefill_context.request_id);
    // TODO(xinfei.sxf) reduce copy
    GenerateInputPB* new_request = new GenerateInputPB(*prefill_context.rpc_context.request);
    new_request->clear_group_size();
    new_request->clear_group_id();
    new_request->mutable_generate_config()->clear_group_timeout();
    alloc_request.set_allocated_input(new_request);
    for (auto& addrs : prefill_context.prefill_worker_cache_store_addrs) {
        alloc_request.add_peer_addrs(addrs);
    }

    // Propagate CP size so decode knows prefill used context-parallel page-RR.
    const auto& cp_cfg = maga_init_params_.parallelism_config.prefill_cp_config;
    if (cp_cfg.kv_cache_sharded && maga_init_params_.parallelism_config.tp_size > 1) {
        alloc_request.set_prefill_cp_size(static_cast<int32_t>(maga_init_params_.parallelism_config.tp_size));
    }

    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, client_stream->Write(alloc_request), ErrorCode::REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED);
    GenerateOutputsPB allocate_response;
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, client_stream->Read(&allocate_response), ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED);
    if (prefillTraceLogEnabled() && allocate_response.has_error_info()
        && allocate_response.error_info().error_code() != 0) {
        RTP_LLM_LOG_WARNING("Prefill request trace: event=remote_allocate_response_error request_id=%ld "
                            "decode_addr=%s remote_error_code=%d remote_error_message=%s",
                            prefill_context.request_id,
                            prefill_context.decode_addr.c_str(),
                            allocate_response.error_info().error_code(),
                            allocate_response.error_info().error_message().c_str());
    }
    RTP_LLM_LOG_DEBUG("request [%ld] remote allocate resource done", prefill_context.request_id);
}

void PrefillRpcServer::enqueueRequest(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] trans query", prefill_context.request_id);
    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", prefill_context.request_id);
    auto stream = engine_->enqueue(prefill_context.generate_input);
    prefill_context.setStream(stream);
    if (prefill_context.cancel_state) {
        refreshAsyncProducerCancelState(
            prefill_context.cancel_state, prefill_context.client_context, prefill_context.getStream());
    }
    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", prefill_context.request_id);
}

void PrefillRpcServer::remoteLoadCacheStart(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] remote load cache", prefill_context.request_id);
    auto start_time_us         = currentTimeUs();
    prefill_context.error_info = waitStreamBeforeRun(prefill_context.getStream());
    prefill_context.stat_info.remote_load_cache_wait_stream_rt_us += currentTimeUs() - start_time_us;
    if (prefill_context.error_info.hasError()) {
        prefill_context.error_status =
            serializeErrorMsg(prefill_context.request_key, prefill_context.request_info, prefill_context.error_info);
        logPrefillFailureTrace("wait_stream_before_run_failed", prefill_context);
        return;
    }
    AtomicGuard       request_guard(loading_cache_requests_);
    GenerateRequestPB load_request;
    load_request.set_client_id(process_id_);
    load_request.set_request_id(prefill_context.request_id);
    load_request.set_start_time(currentTimeUs());
    start_time_us = currentTimeUs();
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Write(load_request), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    prefill_context.stat_info.remote_load_cache_write_request_rt_us += currentTimeUs() - start_time_us;
}

void PrefillRpcServer::pollLocalOutput(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to poll local output", prefill_context.request_id);
    auto first_status = pollStreamOutput(prefill_context.server_context,
                                         prefill_context.request_key,
                                         prefill_context.rpc_context.writer,
                                         prefill_context.getStream());
    if (!first_status.ok()) {
        prefill_context.error_status = first_status;
        logPrefillFailureTrace("poll_local_output_failed", prefill_context);
        return;
    }
    RTP_LLM_LOG_DEBUG("request [%ld] poll local output end", prefill_context.request_id);

    auto stream = prefill_context.getStream();
    if (stream->hasError()) {
        prefill_context.finished     = true;
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, stream->statusInfo().ToString());
        logPrefillFailureTrace("local_stream_failed", prefill_context);
    }
}

void PrefillRpcServer::remoteLoadCacheEnd(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    GenerateOutputsPB load_response;
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Read(&load_response), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    auto error_code = transRPCErrorCode(load_response.error_info().error_code());

    // Decode has finished loading cache, now safe to release KV cache blocks.
    // This is called after cache store transfer is complete.
    if (prefill_context.generate_input->generate_config->pd_separation) {
        prefill_context.getStream()->releaseKVCacheForPDSep();
    }

    CLIENT_GRPC_RET_IF_ERROR(prefill_context, error_code == ErrorCode::NONE_ERROR, error_code);
    RTP_LLM_LOG_DEBUG("request [%ld] remote load cache done", prefill_context.request_id);

    meta_->dequeue(prefill_context.request_id, prefill_context.getStream());
    if (!prefill_context.getStream()->hasEvent(StreamEvents::NeedRemoteGenerate)) {
        RTP_LLM_LOG_DEBUG("request [%ld] pd-sep prefill finished locally without remote generate, "
                          "skipping remote generate stages",
                          prefill_context.request_id);
        // Exit here to keep the remote load-cache completion and release ordering intact.
        prefill_context.finished = true;
    }
}

void PrefillRpcServer::remoteGenerate(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to remote generate", prefill_context.request_id);
    std::shared_ptr<GenerateStream> stream = prefill_context.getStream();
    RTP_LLM_LOG_DEBUG("remote generate stream[%s]: %s", stream->streamLogTag().c_str(), stream->debugString().c_str());
    vector<int> all_token   = stream->currentExecuteTokens();
    int         first_token = all_token[all_token.size() - 1];
    RTP_LLM_LOG_DEBUG("first token token id %d", first_token);
    GenerateRequestPB generate_request;
    generate_request.set_client_id(process_id_);
    generate_request.set_request_id(prefill_context.request_id);
    generate_request.set_first_generate_token_id(first_token);
    auto context_position_ids = stream->getContextPositionIds();
    if (context_position_ids.defined()) {
        generate_request.mutable_position_ids()->CopyFrom(
            {context_position_ids.data_ptr<int32_t>(),
             context_position_ids.data_ptr<int32_t>() + context_position_ids.numel()});
    }
    if (engine_->isMTPEagle()) {
        RTP_LLM_CHECK_WITH_INFO(stream->getProposeToken().size() > 0,
                                "mtp remote generate propose token should not be empty");
    }
    generate_request.mutable_propose_token_ids()->CopyFrom(
        {stream->getProposeToken().begin(), stream->getProposeToken().end()});

    auto sp_output_buffer = stream->getSPOutputBuffer();

    if (sp_output_buffer) {
        auto all_probs_cpu =
            sp_output_buffer->all_probs.is_cuda() ? sp_output_buffer->all_probs.cpu() : sp_output_buffer->all_probs;
        torch::Tensor hidden_states_cpu;
        if (!sp_output_buffer->hidden_states.defined()) {
            // dummy hidden states, so datatype is not important
            hidden_states_cpu = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16));
        } else {
            hidden_states_cpu = sp_output_buffer->hidden_states.is_cuda() ? sp_output_buffer->hidden_states.cpu() :
                                                                            sp_output_buffer->hidden_states;
        }
        QueryConverter::transTensorPB(generate_request.mutable_propose_probs(), all_probs_cpu);
        QueryConverter::transTensorPB(generate_request.mutable_propose_hidden(), hidden_states_cpu);
    }

    generate_request.set_stage(RemoteStage::GENERATE);

    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Write(generate_request), ErrorCode::REMOTE_GENERATE_FAILED);
}

void PrefillRpcServer::pollRemoteOutput(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to poll remote output", prefill_context.request_id);
    auto&             request_id = prefill_context.request_id;
    GenerateOutputsPB response;
    auto              prefill_total_reuse_len  = prefill_context.getStream()->initialReuseLength();
    auto              prefill_local_reuse_len  = prefill_context.getStream()->localReuseLength();
    auto              prefill_remote_reuse_len = prefill_context.getStream()->remoteReuseLength();
    auto              prefill_memory_reuse_len = prefill_context.getStream()->memoryReuseLength();

    auto first_token_rt_us = prefill_context.getStream()->getTimeInfo().first_token_rt_us;
    while (prefill_context.client_stream->Read(&response)) {
        if (prefill_context.server_context && prefill_context.server_context->IsCancelled()) {
            RTP_LLM_LOG_WARNING("request [%ld] cancel by user", request_id);
            prefill_context.error_status = grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
            return;
        }
        if (response.flatten_output().aux_info_size() == 0) {
            RTP_LLM_LOG_ERROR("request [%ld] generate output size is 0", request_id);
            break;
        }
        for (size_t i = 0; i < response.flatten_output().aux_info_size(); i++) {
            response.mutable_flatten_output()->mutable_aux_info(i)->set_pd_sep(true);
        }
        int64_t cost_time_us = currentTimeUs() - prefill_context.request_begin_time_us;
        for (size_t i = 0; i < response.flatten_output().aux_info_size(); i++) {
            auto decode_total_reuse_len  = response.flatten_output().aux_info(i).total_reuse_len();
            auto decode_local_reuse_len  = response.flatten_output().aux_info(i).local_reuse_len();
            auto decode_remote_reuse_len = response.flatten_output().aux_info(i).remote_reuse_len();
            auto decode_memory_reuse_len = response.flatten_output().aux_info(i).memory_reuse_len();

            response.mutable_flatten_output()->mutable_aux_info(i)->set_first_token_cost_time_us(first_token_rt_us);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_cost_time_us(cost_time_us);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_total_reuse_len(prefill_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_local_reuse_len(prefill_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_remote_reuse_len(prefill_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_memory_reuse_len(prefill_memory_reuse_len);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_total_reuse_len(
                prefill_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_local_reuse_len(
                prefill_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_remote_reuse_len(
                prefill_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_memory_reuse_len(
                prefill_memory_reuse_len);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_total_reuse_len(decode_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_local_reuse_len(decode_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_remote_reuse_len(
                decode_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_memory_reuse_len(
                decode_memory_reuse_len);
        }
        if (!prefill_context.rpc_context.writer->Write(response)) {
            RTP_LLM_LOG_WARNING("request [%ld] write outputs pb failed", request_id);
            prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
            return;
        }
    }
    auto status = prefill_context.closeGrpcStream();
    if (!status.ok() && status.error_code() != grpc::StatusCode::CANCELLED) {
        CLIENT_GRPC_RET_IF_ERROR(prefill_context, false, ErrorCode::REMOTE_GENERATE_FAILED);
    }
}

grpc::Status PrefillRpcServer::prepareAllocateResource(PrefillGenerateContext& prefill_context) {
    EXECUTE_STAGE_FUNC(getRpcConnection, prefill_context);
    EXECUTE_STAGE_FUNC(multimodalProcess, prefill_context);
    EXECUTE_STAGE_FUNC(remoteAllocateResource, prefill_context);
    return grpc::Status::OK;
}

void PrefillRpcServer::reportPrefillRecentCacheKeyMetricsOnce(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (prefill_context.recent_cache_key_metric_reported) {
        return;
    }
    if (!PrefillCacheHitMetricsReporter::enabled()) {
        return;
    }
    if (!prefill_recent_cache_key_window_) {
        return;
    }
    if (!prefill_context.generate_input) {
        return;
    }
    prefill_context.recent_cache_key_metric_reported = true;

    const int seq_size_per_block = maga_init_params_.kv_cache_config.seq_size_per_block;
    auto      cache_keys = buildFullBlockCacheKeys(prefill_context.generate_input->input_ids, seq_size_per_block);
    auto      snapshot   = prefill_recent_cache_key_window_->record(cache_keys);
    static TheoryHitStats theory_stats;
    auto theory_snapshot = theory_stats.record(snapshot.request_hit_occurrences, snapshot.request_occurrences);
    if (theory_snapshot.request_total_count > 0) {
        appendPrefillTheoryHitLogLine(formatPrefillTheoryHitLogLine(
            prefill_context, prefill_context.generate_input->input_ids.numel(), seq_size_per_block, theory_snapshot));
    }

    if (metrics_reporter_) {
        PrefillRecentCacheKeyMetricsCollector collector;
        fillPrefillRecentCacheKeyMetricsCollector(collector, snapshot);
        fillPrefillTheoryHitMetricsCollector(collector, theory_snapshot);
        metrics_reporter_->report<PrefillRecentCacheKeyMetrics, PrefillRecentCacheKeyMetricsCollector>(nullptr,
                                                                                                       &collector);
    }

    if (prefillCacheDebugLogEnabled()) {
        auto key_digest = cacheKeyDigest(cache_keys);
        auto key_text   = cacheKeysToString(cache_keys);
        RTP_LLM_LOG_INFO("Prefill cache-key trace: request_id=%ld request_key=%s token_num=%ld seq_size_per_block=%d "
                         "key_count=%zu hit_count=%ld total_count=%ld hit_ratio=%.6f cache_key_digest=%s "
                         "retained_occurrences=%ld retained_unique_cache_keys=%zu window_ms=%ld cache_keys=%s",
                         prefill_context.request_id,
                         prefill_context.request_key.c_str(),
                         prefill_context.generate_input->input_ids.numel(),
                         seq_size_per_block,
                         cache_keys.size(),
                         snapshot.request_hit_occurrences,
                         snapshot.request_occurrences,
                         snapshot.request_hit_ratio,
                         key_digest.c_str(),
                         snapshot.retained_occurrences,
                         snapshot.retained_unique_cache_keys,
                         snapshot.time_window_ms,
                         key_text.c_str());
        RTP_LLM_LOG_INFO("Prefill cache-key preview trace: request_id=%ld cache_key_digest=%s keys_preview=%s",
                         prefill_context.request_id,
                         key_digest.c_str(),
                         cacheKeyPreview(cache_keys).c_str());
    }
}

grpc::Status PrefillRpcServer::syncPrefix(PrefillGenerateContext& prefill_context) {
    auto max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;
    int  retry_interval_ms    = 1;

    EXECUTE_WITH_RETRY(
        prepareAllocateResource, prefill_context, max_retry_times, max_retry_timeout_ms, retry_interval_ms);
    if (prefill_context.hasError()) {
        logPrefillFailureTrace("prepare_allocate_failed", prefill_context);
        RTP_LLM_LOG_WARNING(
            "request [%ld] prepare allocate resource failed after retry [%d] times, cost time ms [%ld], "
            "max retry time [%ld], max retry timeout ms [%ld]",
            prefill_context.request_id,
            prefill_context.retry_times,
            prefill_context.retry_cost_time_ms,
            max_retry_times + 1,
            max_retry_timeout_ms);
        return prefill_context.error_status;
    }
    EXECUTE_STAGE_FUNC(enqueueRequest, prefill_context);
    EXECUTE_STAGE_FUNC(remoteLoadCacheStart, prefill_context);
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::finishStream(PrefillGenerateContext& prefill_context) {
    EXECUTE_STAGE_FUNC(pollLocalOutput, prefill_context);
    EXECUTE_STAGE_FUNC(remoteLoadCacheEnd, prefill_context);
    EXECUTE_STAGE_FUNC(remoteGenerate, prefill_context);
    EXECUTE_STAGE_FUNC(pollRemoteOutput, prefill_context);
    prefill_context.stat_info.nextStage();
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                  const GenerateInputPB*                 request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start generate stream call", request->request_id());
    auto pd_separation = request->generate_config().max_new_tokens() > 1 && request->generate_config().num_beams() <= 1
                         && request->generate_config().variable_num_beams().size() == 0
                         && request->generate_config().num_return_sequences() <= 1
                         && request->generate_config().can_use_pd_separation();
    if (prefillTraceLogEnabled()) {
        RTP_LLM_LOG_INFO(
            "Prefill request trace: event=recv request_id=%ld pd_separation=%d token_ids=%d "
            "max_new_tokens=%d num_beams=%d num_return_sequences=%d can_use_pd_separation=%d timeout_ms=%ld",
            request->request_id(),
            pd_separation,
            request->token_ids_size(),
            request->generate_config().max_new_tokens(),
            request->generate_config().num_beams(),
            request->generate_config().num_return_sequences(),
            request->generate_config().can_use_pd_separation(),
            request->generate_config().timeout_ms());
    }
    if (!pd_separation) {
        if (prefillTraceLogEnabled()) {
            RTP_LLM_LOG_INFO("Prefill request trace: event=bypass_local request_id=%ld token_ids=%d",
                             request->request_id(),
                             request->token_ids_size());
        }
        return LocalRpcServer::GenerateStreamCall(server_context, request, writer);
    }

    AtomicGuardPtr request_guard = make_shared<AtomicGuard>(onflight_requests_);
    RPCContext     rpc_context{request, writer};
    auto           prefill_context         = PrefillGenerateContext(&this->resource(),
                                                  rpc_context,
                                                  request->generate_config().timeout_ms(),
                                                  server_context,
                                                  metrics_reporter_,
                                                  meta_,
                                                  maga_init_params_.pd_sep_config.prefill_stop_stream_wait_timeout_ms);
    prefill_context.onflight_requests      = onflight_requests_;
    prefill_context.loading_cache_requests = loading_cache_requests_;

    try {
        auto status = syncPrefix(prefill_context);
        if (!status.ok()) {
            return status;
        }
        status = finishStream(prefill_context);
        if (!status.ok()) {
            return status;
        }
    } catch (const std::exception& e) {
        auto error_msg = "request [" + prefill_context.request_key + "] catch exception [" + e.what() + "]";
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        logPrefillFailureTrace("catch_exception", prefill_context);
        return prefill_context.error_status;
    } catch (...) {
        auto error_msg               = "request [" + prefill_context.request_key + "] catch unknown exception";
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        logPrefillFailureTrace("catch_unknown_exception", prefill_context);
        return prefill_context.error_status;
    }

    RTP_LLM_LOG_DEBUG("request [%ld] all done", prefill_context.request_id);

    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::EnqueueBatch(grpc::ServerContext*         context,
                                            const EnqueueBatchRequestPB* request,
                                            EnqueueBatchResponsePB*      response) {
    RTP_LLM_PROFILE_FUNCTION();
    response->set_batch_id(request->batch_id());

    struct TargetBatch {
        int                                 dp_rank = 0;
        std::vector<const GenerateInputPB*> inputs;
    };

    std::map<int, TargetBatch>          targets;
    std::vector<const GenerateInputPB*> all_inputs;
    std::unordered_set<int64_t>         seen_request_ids;
    bool                                duplicate_request_id = false;

    for (const auto& slot : request->dp_slots()) {
        auto& target   = targets[slot.dp_rank()];
        target.dp_rank = slot.dp_rank();
        for (const auto& external_input : slot.requests()) {
            if (!external_input.has_input()) {
                addBatchError(response,
                              /*request_id=*/0,
                              grpc::StatusCode::INVALID_ARGUMENT,
                              "EnqueueBatch external request missing input");
                continue;
            }
            const auto& input = external_input.input();
            all_inputs.push_back(&input);
            target.inputs.push_back(&input);
            if (!seen_request_ids.insert(input.request_id()).second) {
                duplicate_request_id = true;
            }
        }
    }

    response->mutable_successes()->Reserve(static_cast<int>(all_inputs.size()));
    response->mutable_errors()->Reserve(static_cast<int>(all_inputs.size()));

    auto add_error_for_inputs = [](EnqueueBatchResponsePB*                    response,
                                   const std::vector<const GenerateInputPB*>& inputs,
                                   int64_t                                    code,
                                   const std::string&                         message) {
        for (const auto* input : inputs) {
            if (input) {
                addBatchError(response, input->request_id(), code, message);
            }
        }
    };

    if (duplicate_request_id) {
        response->clear_errors();
        add_error_for_inputs(
            response, all_inputs, grpc::StatusCode::ALREADY_EXISTS, "duplicate request_id in EnqueueBatch");
        return grpc::Status::OK;
    }

    if (context && context->IsCancelled()) {
        add_error_for_inputs(response, all_inputs, grpc::StatusCode::CANCELLED, "EnqueueBatch cancelled by caller");
        return grpc::Status(grpc::StatusCode::CANCELLED, "EnqueueBatch cancelled by caller");
    }

    struct DispatchTarget {
        int                   dp_rank = 0;
        std::string           addr;
        EnqueueGroupRequestPB request;
    };

    const int                   local_dp_rank = static_cast<int>(maga_init_params_.parallelism_config.dp_rank);
    std::vector<DispatchTarget> dispatch_targets;
    dispatch_targets.reserve(targets.size());
    for (const auto& pair : targets) {
        const auto& target = pair.second;
        if (target.inputs.empty()) {
            continue;
        }
        DispatchTarget dispatch_target;
        dispatch_target.dp_rank = target.dp_rank;
        dispatch_target.request.set_batch_id(request->batch_id());
        dispatch_target.request.set_dp_rank(target.dp_rank);
        for (const auto* input : target.inputs) {
            auto* dp_input = dispatch_target.request.add_requests();
            dp_input->mutable_input()->CopyFrom(*input);
        }
        if (target.dp_rank != local_dp_rank) {
            dispatch_target.addr = batchTargetAddrForDpRank(target.dp_rank);
            if (dispatch_target.addr.empty()) {
                add_error_for_inputs(response,
                                     target.inputs,
                                     grpc::StatusCode::INVALID_ARGUMENT,
                                     "invalid EnqueueBatch dp_rank " + std::to_string(target.dp_rank));
                continue;
            }
        }
        dispatch_targets.push_back(std::move(dispatch_target));
    }

    struct DispatchResult {
        grpc::Status           status;
        EnqueueBatchResponsePB dp_response;
    };

    const auto dispatch_timeout_ms = maga_init_params_.pd_sep_config.batch_dispatch_timeout_ms;
    const auto dispatch_deadline   = std::chrono::steady_clock::now() + std::chrono::milliseconds(dispatch_timeout_ms);
    std::vector<autil::ThreadPoolBase::Future<DispatchResult>> dispatch_futures;
    dispatch_futures.reserve(dispatch_targets.size());
    for (auto& target : dispatch_targets) {
        enqueue_pool_metrics_.queued++;
        dispatch_futures.push_back(enqueue_worker_pool_->async([this, target = std::move(target)]() -> DispatchResult {
            enqueue_pool_metrics_.queued--;
            enqueue_pool_metrics_.active++;
            ScopeExit      enqueue_task_guard([this] {
                enqueue_pool_metrics_.active--;
                enqueue_pool_metrics_.completed++;
            });
            DispatchResult result;
            if (target.dp_rank == static_cast<int>(maga_init_params_.parallelism_config.dp_rank)) {
                result.status = EnqueueGroup(/*context=*/nullptr, &target.request, &result.dp_response);
                return result;
            }

            try {
                auto connect_status = resource_.rpc_pool.getConnection(target.addr);
                if (!connect_status.ok()) {
                    result.status = grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                                 "get EnqueueGroup connection failed: "
                                                     + std::string(connect_status.status().message()));
                } else {
                    grpc::ClientContext client_context;
                    auto                timeout_ms = maga_init_params_.pd_sep_config.max_rpc_timeout_ms;
                    if (timeout_ms > 0) {
                        client_context.set_deadline(std::chrono::system_clock::now()
                                                    + std::chrono::milliseconds(timeout_ms));
                    }
                    result.status =
                        connect_status.value().stub->EnqueueGroup(&client_context, target.request, &result.dp_response);
                }
            } catch (const std::exception& e) {
                result.status = grpc::Status(grpc::StatusCode::INTERNAL,
                                             "EnqueueGroup forward exception: " + std::string(e.what()));
            } catch (...) {
                result.status = grpc::Status(grpc::StatusCode::INTERNAL, "EnqueueGroup forward unknown exception");
            }
            return result;
        }));
    }

    auto merge_response = [&](const EnqueueGroupRequestPB&  dp_request,
                              const grpc::Status&           status,
                              const EnqueueBatchResponsePB& dp_response) {
        if (!status.ok()) {
            for (const auto& dp_input : dp_request.requests()) {
                if (dp_input.has_input()) {
                    addBatchError(response, dp_input.input().request_id(), status.error_code(), status.error_message());
                }
            }
            return;
        }

        std::unordered_set<int64_t> returned_request_ids;
        std::unordered_set<int64_t> error_request_ids;
        for (const auto& error : dp_response.errors()) {
            addBatchError(
                response, error.request_id(), error.error_info().error_code(), error.error_info().error_message());
            returned_request_ids.insert(error.request_id());
            error_request_ids.insert(error.request_id());
        }
        for (const auto& success : dp_response.successes()) {
            if (error_request_ids.find(success.request_id()) != error_request_ids.end()) {
                continue;
            }
            addBatchSuccess(response, success.request_id());
            returned_request_ids.insert(success.request_id());
        }
        for (const auto& dp_input : dp_request.requests()) {
            if (!dp_input.has_input()) {
                continue;
            }
            const auto request_id = dp_input.input().request_id();
            if (returned_request_ids.find(request_id) == returned_request_ids.end()) {
                addBatchError(
                    response, request_id, grpc::StatusCode::INTERNAL, "EnqueueGroup missing result for request");
            }
        }
    };

    collectFutures(
        dispatch_futures,
        dispatch_deadline,
        [&](size_t i) {
            auto result = dispatch_futures[i].get();
            merge_response(dispatch_targets[i].request, result.status, result.dp_response);
        },
        [&](size_t i) {
            merge_response(dispatch_targets[i].request,
                           grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                                        "EnqueueBatch dispatch timeout for dp_rank "
                                            + std::to_string(dispatch_targets[i].dp_rank)),
                           EnqueueBatchResponsePB());
            for (const auto& dp_input : dispatch_targets[i].request.requests()) {
                if (dp_input.has_input()) {
                    auto entry = response_registry_.get(dp_input.input().request_id());
                    cancelResponseEntry(entry);
                }
            }
        });
    detachLeftoverFutures(dispatch_futures);

    response_registry_.gc(std::chrono::minutes(10));
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::EnqueueGroup(grpc::ServerContext*         context,
                                            const EnqueueGroupRequestPB* request,
                                            EnqueueBatchResponsePB*      response) {
    RTP_LLM_PROFILE_FUNCTION();
    response->set_batch_id(request->batch_id());

    struct LocalSlot {
        std::shared_ptr<GenerateInputPB>          input;
        std::shared_ptr<ResponseBufferEntry>      entry;
        std::shared_ptr<RPCContext>               rpc_context;
        std::shared_ptr<PrefillGenerateContext>   prefill_context;
        std::shared_ptr<AsyncProducerCancelState> cancel_state;
        AtomicGuardPtr                            request_guard;
        int64_t                                   request_id   = 0;
        bool                                      prepared     = false;
        grpc::Status                              stage_status = grpc::Status::OK;
    };

    std::vector<const GenerateInputPB*> all_inputs;
    all_inputs.reserve(request->requests_size());
    std::unordered_set<int64_t> seen_request_ids;
    bool                        duplicate_request_id = false;
    for (const auto& dp_input : request->requests()) {
        if (!dp_input.has_input()) {
            addBatchError(response,
                          /*request_id=*/0,
                          grpc::StatusCode::INVALID_ARGUMENT,
                          "EnqueueGroup request missing input");
            continue;
        }
        all_inputs.push_back(&dp_input.input());
        if (!seen_request_ids.insert(dp_input.input().request_id()).second) {
            duplicate_request_id = true;
        }
    }

    response->mutable_successes()->Reserve(static_cast<int>(all_inputs.size()));
    response->mutable_errors()->Reserve(static_cast<int>(all_inputs.size()));

    auto add_error_for_all = [&](int64_t code, const std::string& message) {
        for (const auto* input : all_inputs) {
            addBatchError(response, input->request_id(), code, message);
        }
    };

    const int local_dp_rank = static_cast<int>(maga_init_params_.parallelism_config.dp_rank);
    if (request->dp_rank() != local_dp_rank) {
        add_error_for_all(grpc::StatusCode::INVALID_ARGUMENT,
                          "EnqueueGroup dp_rank mismatch, request dp_rank " + std::to_string(request->dp_rank())
                              + ", local dp_rank " + std::to_string(local_dp_rank));
        return grpc::Status::OK;
    }
    if (duplicate_request_id) {
        response->clear_errors();
        add_error_for_all(grpc::StatusCode::ALREADY_EXISTS, "duplicate request_id in EnqueueGroup");
        return grpc::Status::OK;
    }
    if (context && context->IsCancelled()) {
        add_error_for_all(grpc::StatusCode::CANCELLED, "EnqueueGroup cancelled by caller");
        return grpc::Status(grpc::StatusCode::CANCELLED, "EnqueueGroup cancelled by caller");
    }

    std::vector<LocalSlot> slots;
    slots.reserve(all_inputs.size());
    const int group_size = static_cast<int>(all_inputs.size());
    for (const auto* input : all_inputs) {
        auto input_copy = std::make_shared<GenerateInputPB>(*input);
        input_copy->set_group_size(group_size);
        input_copy->mutable_group_id()->set_value(request->batch_id());

        auto entry = response_registry_.reserve(input_copy->request_id());
        if (!entry) {
            addBatchError(
                response, input_copy->request_id(), grpc::StatusCode::ALREADY_EXISTS, "request already enqueued");
            continue;
        }
        slots.push_back({input_copy, entry, nullptr, nullptr, nullptr, nullptr, input_copy->request_id()});
    }

    if (slots.empty()) {
        return grpc::Status::OK;
    }

    for (const auto& slot : slots) {
        int64_t batch_id = (slot.input && slot.input->has_group_id()) ? slot.input->group_id().value() : -1;
        RTP_LLM_LOG_DEBUG("request [%ld] EnqueueGroup: has_group_id=%d, batch_id=%ld, request_batch_id=%ld",
                          slot.request_id,
                          slot.input ? slot.input->has_group_id() : 0,
                          batch_id,
                          request->batch_id());
        meta_->enqueuePending(slot.request_id, slot.input ? slot.input->token_ids_size() : 0, batch_id);
    }

    auto erase_reserved_slots = [this](const std::vector<LocalSlot>& slots) {
        for (const auto& slot : slots) {
            cancelResponseEntry(slot.entry);
            response_registry_.erase(slot.request_id);
        }
    };

    auto finish_pending_before_ack = [this](const LocalSlot& slot, const grpc::Status& status) {
        meta_->finishTask(slot.request_id,
                          slot.input ? slot.input->token_ids_size() : 0,
                          /*prefix_length=*/0,
                          status.ok() ? 0 : static_cast<int64_t>(status.error_code()),
                          status.ok() ? "" : std::string(status.error_message()));
    };

    if (!tryStartAsyncResponseWorker()) {
        auto status = grpc::Status(grpc::StatusCode::UNAVAILABLE, "EnqueueGroup server is stopping");
        for (const auto& slot : slots) {
            finish_pending_before_ack(slot, status);
            addBatchError(response, slot.request_id, status.error_code(), status.error_message());
        }
        erase_reserved_slots(slots);
        return grpc::Status::OK;
    }

    std::vector<int64_t> accepted_request_ids;
    accepted_request_ids.reserve(slots.size());
    for (const auto& slot : slots) {
        accepted_request_ids.push_back(slot.request_id);
    }

    auto slots_ptr = std::make_shared<std::vector<LocalSlot>>(std::move(slots));
    try {
        worker_lambda_pool_metrics_.queued++;
        auto worker_error = worker_lambda_pool_->pushTask(
            [this,
             slots_ptr,
             max_retry_times          = maga_init_params_.pd_sep_config.prefill_retry_times,
             max_retry_timeout_ms     = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms,
             batch_prepare_timeout_ms = maga_init_params_.pd_sep_config.batch_prepare_timeout_ms,
             batch_load_timeout_ms    = maga_init_params_.pd_sep_config.batch_load_timeout_ms]() mutable {
                worker_lambda_pool_metrics_.queued--;
                worker_lambda_pool_metrics_.active++;
                ScopeExit worker_task_guard([this] {
                    worker_lambda_pool_metrics_.active--;
                    worker_lambda_pool_metrics_.completed++;
                });
                ScopeExit controller_finish_guard([this] { finishAsyncResponseWorker(); });
                auto&     slots = *slots_ptr;

                auto entry_cancelled = [](const LocalSlot& slot) {
                    return !slot.entry || slot.entry->cancelled.load();
                };
                auto grpc_status_to_stream_error = [](const grpc::Status& status) {
                    return status.error_code() == grpc::StatusCode::CANCELLED ? ErrorCode::CANCELLED :
                                                                                ErrorCode::UNKNOWN_ERROR;
                };
                auto fail_slot = [this, grpc_status_to_stream_error](LocalSlot& slot, const grpc::Status& status) {
                    int64_t input_length  = slot.input ? slot.input->token_ids_size() : 0;
                    int64_t prefix_length = 0;
                    if (slot.prefill_context && slot.prefill_context->getStream()) {
                        auto stream   = slot.prefill_context->getStream();
                        input_length  = stream->inputLength();
                        prefix_length = stream->prefixLength();
                        if (!stream->hasError()) {
                            stream->reportError(grpc_status_to_stream_error(status),
                                                std::string(status.error_message()));
                        }
                    }
                    meta_->finishTask(slot.request_id,
                                      input_length,
                                      prefix_length,
                                      status.ok() ? 0 : static_cast<int64_t>(status.error_code()),
                                      status.ok() ? "" : std::string(status.error_message()));
                    markResponseEntryDone(slot.entry, status);
                    slot.prefill_context.reset();
                    slot.rpc_context.reset();
                    slot.request_guard.reset();
                    slot.cancel_state.reset();
                    slot.input.reset();
                    slot.entry.reset();
                };

                auto start_finish_worker = [this, fail_slot](LocalSlot& slot) {
                    auto entry                               = slot.entry;
                    auto writer                              = std::make_shared<ResponseBufferWriter>(entry);
                    slot.prefill_context->rpc_context.writer = writer.get();

                    if (!tryStartAsyncResponseWorker()) {
                        fail_slot(slot, grpc::Status(grpc::StatusCode::UNAVAILABLE, "EnqueueGroup server is stopping"));
                        return;
                    }

                    try {
                        auto finish_lambda = [this,
                                              pfx_ctx = slot.prefill_context,
                                              rpc_ctx = slot.rpc_context,
                                              input   = slot.input,
                                              writer,
                                              entry,
                                              guard        = slot.request_guard,
                                              cancel_state = slot.cancel_state,
                                              request_id   = slot.request_id]() mutable {
                            (void)rpc_ctx;
                            (void)input;
                            (void)writer;
                            (void)guard;
                            (void)cancel_state;
                            slot_pool_metrics_.active++;
                            ScopeExit    slot_finish_task_guard([this] {
                                slot_pool_metrics_.active--;
                                slot_pool_metrics_.completed++;
                            });
                            ScopeExit    worker_finish_guard([this] { finishAsyncResponseWorker(); });
                            ScopeExit    release_captures_guard([&] {
                                pfx_ctx.reset();
                                rpc_ctx.reset();
                                input.reset();
                                writer.reset();
                                entry.reset();
                                guard.reset();
                                cancel_state.reset();
                            });
                            grpc::Status finish_status;
                            try {
                                finish_status = finishStream(*pfx_ctx);
                                RTP_LLM_LOG_DEBUG("request [%ld] finishStream returned, ok=%d, has_stream=%d",
                                                  request_id,
                                                  finish_status.ok(),
                                                  pfx_ctx->getStream() ? 1 : 0);
                                // Record finished task for FlexLB calibration
                                if (finish_status.ok() && pfx_ctx->getStream()) {
                                    RTP_LLM_LOG_DEBUG("request [%ld] calling dequeue for FlexLB calibration",
                                                      request_id);
                                    meta_->dequeue(request_id, pfx_ctx->getStream());
                                } else if (!finish_status.ok()) {
                                    RTP_LLM_LOG_DEBUG("request [%ld] calling finishTask due to error, code=%d, msg=%s",
                                                      request_id,
                                                      static_cast<int>(finish_status.error_code()),
                                                      finish_status.error_message().c_str());
                                    meta_->finishTask(request_id,
                                                      input ? input->token_ids_size() : 0,
                                                      /*prefix_length=*/0,
                                                      static_cast<int64_t>(finish_status.error_code()),
                                                      finish_status.error_message());
                                }
                            } catch (const std::exception& e) {
                                auto error_msg =
                                    "request [" + pfx_ctx->request_key + "] finishStream exception [" + e.what() + "]";
                                finish_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
                                meta_->finishTask(request_id,
                                                  input ? input->token_ids_size() : 0,
                                                  /*prefix_length=*/0,
                                                  static_cast<int64_t>(finish_status.error_code()),
                                                  error_msg);
                            } catch (...) {
                                finish_status =
                                    grpc::Status(grpc::StatusCode::INTERNAL, "finishStream unknown exception");
                                meta_->finishTask(request_id,
                                                  input ? input->token_ids_size() : 0,
                                                  /*prefix_length=*/0,
                                                  static_cast<int64_t>(finish_status.error_code()),
                                                  "finishStream unknown exception");
                            }
                            markResponseEntryDone(entry, finish_status);
                            RTP_LLM_LOG_DEBUG(
                                "EnqueueGroup request [%ld] finishStream done, ok=%d", request_id, finish_status.ok());
                        };

                        // Non-blocking submit: if pool is full, fail the slot with UNAVAILABLE.
                        auto error = slot_worker_pool_->pushTask(std::move(finish_lambda));
                        if (error != autil::ThreadPoolBase::ERROR_NONE) {
                            slot_pool_metrics_.rejected++;
                            slot_pool_metrics_.fallback++;
                            // Pool saturated: fail the slot cleanly instead of detaching a thread.
                            // Previously used std::thread::detach() which risks UAF if the server
                            // is destroyed while the thread is still running (it captures `this`).
                            // tryStartAsyncResponseWorker() already incremented response_worker_count_,
                            // so we must call finishAsyncResponseWorker() to decrement it.
                            finishAsyncResponseWorker();
                            fail_slot(slot, grpc::Status(grpc::StatusCode::UNAVAILABLE, "slot worker pool saturated"));
                        }
                    } catch (const std::exception& e) {
                        finishAsyncResponseWorker();
                        fail_slot(slot,
                                  grpc::Status(grpc::StatusCode::INTERNAL,
                                               "start async response worker exception: " + std::string(e.what())));
                    } catch (...) {
                        finishAsyncResponseWorker();
                        fail_slot(
                            slot,
                            grpc::Status(grpc::StatusCode::INTERNAL, "start async response worker unknown exception"));
                    }
                };

                for (auto& slot : slots) {
                    auto rpc_ctx = std::make_shared<RPCContext>(RPCContext{slot.input.get(), nullptr});
                    auto pfx_ctx = std::make_shared<PrefillGenerateContext>(
                        &this->resource(),
                        *rpc_ctx,
                        slot.input->generate_config().timeout_ms(),
                        /*server_context=*/nullptr,
                        metrics_reporter_,
                        meta_,
                        maga_init_params_.pd_sep_config.prefill_stop_stream_wait_timeout_ms);
                    pfx_ctx->onflight_requests      = onflight_requests_;
                    pfx_ctx->loading_cache_requests = loading_cache_requests_;
                    auto guard                      = std::make_shared<AtomicGuard>(onflight_requests_);
                    auto cancel_state               = std::make_shared<AsyncProducerCancelState>();
                    {
                        std::lock_guard<std::mutex> lock(slot.entry->mu);
                        cancel_state->cancelled.store(slot.entry->cancelled.load());
                        slot.entry->cancel_producer = makeAsyncProducerCancelCallback(cancel_state);
                    }
                    slot.rpc_context      = rpc_ctx;
                    slot.prefill_context  = pfx_ctx;
                    slot.cancel_state     = cancel_state;
                    slot.request_guard    = guard;
                    pfx_ctx->cancel_state = cancel_state;
                }

                const auto prepare_deadline =
                    std::chrono::steady_clock::now() + std::chrono::milliseconds(batch_prepare_timeout_ms);
                std::vector<autil::ThreadPoolBase::Future<void>> prepare_futures;
                prepare_futures.reserve(slots.size());
                for (auto& slot : slots) {
                    auto* slot_ptr = &slot;
                    slot_pool_metrics_.queued++;
                    prepare_futures.push_back(slot_worker_pool_->async(
                        [this, slot_ptr, slots_ptr, entry_cancelled, max_retry_times, max_retry_timeout_ms] {
                            slot_pool_metrics_.queued--;
                            slot_pool_metrics_.active++;
                            ScopeExit slot_prepare_guard([this] {
                                slot_pool_metrics_.active--;
                                slot_pool_metrics_.completed++;
                            });
                            auto&     slot = *slot_ptr;
                            if (entry_cancelled(slot)) {
                                slot.stage_status =
                                    grpc::Status(grpc::StatusCode::CANCELLED, "EnqueueGroup request cancelled");
                                return;
                            }
                            try {
                                int64_t begin_time_us = currentTimeUs();
                                auto    stage         = slot.prefill_context->stat_info.saveStage();
                                for (int attempt = 0; attempt <= max_retry_times; ++attempt) {
                                    if (entry_cancelled(slot)) {
                                        slot.stage_status =
                                            grpc::Status(grpc::StatusCode::CANCELLED, "EnqueueGroup request cancelled");
                                        return;
                                    }
                                    slot.prefill_context->reset();
                                    slot.prefill_context->stat_info.restoreStage(stage);
                                    slot.prefill_context->retry_times++;
                                    prepareAllocateResource(*slot.prefill_context);
                                    if (slot.prefill_context->ok()) {
                                        slot.prepared = true;
                                        return;
                                    }
                                    auto cost_time_us                        = currentTimeUs() - begin_time_us;
                                    slot.prefill_context->retry_cost_time_ms = cost_time_us / 1000;
                                    if (max_retry_timeout_ms > 0 && cost_time_us >= max_retry_timeout_ms * 1000) {
                                        break;
                                    }
                                    usleep(1000);
                                }
                                slot.stage_status = slot.prefill_context->error_status.ok() ?
                                                        statusFromErrorInfo(slot.prefill_context->error_info) :
                                                        slot.prefill_context->error_status;
                                if (slot.stage_status.ok()) {
                                    slot.stage_status =
                                        grpc::Status(grpc::StatusCode::INTERNAL, "prepareAllocateResource failed");
                                }
                            } catch (const std::exception& e) {
                                slot.stage_status =
                                    grpc::Status(grpc::StatusCode::INTERNAL,
                                                 "prepareAllocateResource exception: " + std::string(e.what()));
                            } catch (...) {
                                slot.stage_status = grpc::Status(grpc::StatusCode::INTERNAL,
                                                                 "prepareAllocateResource unknown exception");
                            }
                        }));
                }
                collectFutures(
                    prepare_futures,
                    prepare_deadline,
                    [&](size_t i) { prepare_futures[i].get(); },
                    [&](size_t i) {
                        cancelResponseEntry(slots[i].entry);
                        slots[i].stage_status =
                            grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "EnqueueGroup prepare timeout");
                    });
                drainReadyFutures(prepare_futures, std::chrono::milliseconds(2000));
                detachLeftoverFutures(prepare_futures);

                std::vector<LocalSlot*> ready_slots;
                ready_slots.reserve(slots.size());
                for (auto& slot : slots) {
                    if (entry_cancelled(slot)) {
                        fail_slot(slot, grpc::Status(grpc::StatusCode::CANCELLED, "EnqueueGroup request cancelled"));
                    } else if (!slot.prepared) {
                        fail_slot(slot, slot.stage_status);
                    } else {
                        ready_slots.push_back(&slot);
                    }
                }
                if (ready_slots.empty()) {
                    return;
                }

                const int                                   local_group_size = static_cast<int>(ready_slots.size());
                std::vector<std::shared_ptr<GenerateInput>> generate_inputs;
                generate_inputs.reserve(ready_slots.size());
                for (auto* slot : ready_slots) {
                    slot->input->set_group_size(local_group_size);
                    slot->prefill_context->generate_input->group_size = local_group_size;
                    slot->prefill_context->stat_info.nextStage();
                    generate_inputs.push_back(slot->prefill_context->generate_input);
                }

                std::vector<GenerateStreamPtr> streams;
                try {
                    streams = engine_->enqueueMultiple(generate_inputs);
                } catch (const std::exception& e) {
                    for (auto* slot : ready_slots) {
                        fail_slot(*slot,
                                  grpc::Status(grpc::StatusCode::INTERNAL,
                                               "enqueueMultiple exception: " + std::string(e.what())));
                    }
                    return;
                } catch (...) {
                    for (auto* slot : ready_slots) {
                        fail_slot(*slot, grpc::Status(grpc::StatusCode::INTERNAL, "enqueueMultiple unknown exception"));
                    }
                    return;
                }

                std::unordered_map<int64_t, GenerateStreamPtr> stream_by_id;
                for (auto& stream : streams) {
                    if (stream) {
                        stream_by_id[stream->streamId()] = stream;
                    }
                }
                std::vector<LocalSlot*> stream_ready_slots;
                stream_ready_slots.reserve(ready_slots.size());
                for (auto* slot : ready_slots) {
                    auto it = stream_by_id.find(slot->request_id);
                    if (it == stream_by_id.end()) {
                        fail_slot(*slot, grpc::Status(grpc::StatusCode::INTERNAL, "EnqueueGroup stream not enqueued"));
                        continue;
                    }
                    slot->prefill_context->setStream(it->second);
                    refreshAsyncProducerCancelState(
                        slot->cancel_state, slot->prefill_context->client_context, slot->prefill_context->getStream());
                    stream_ready_slots.push_back(slot);
                }

                const auto load_deadline =
                    std::chrono::steady_clock::now() + std::chrono::milliseconds(batch_load_timeout_ms);
                std::vector<autil::ThreadPoolBase::Future<void>> load_futures;
                load_futures.reserve(stream_ready_slots.size());
                for (auto* slot : stream_ready_slots) {
                    slot_pool_metrics_.queued++;
                    load_futures.push_back(slot_worker_pool_->async(
                        [this, slot, slots_ptr, entry_cancelled, fail_slot, start_finish_worker] {
                            slot_pool_metrics_.queued--;
                            slot_pool_metrics_.active++;
                            ScopeExit slot_load_guard([this] {
                                slot_pool_metrics_.active--;
                                slot_pool_metrics_.completed++;
                            });
                            if (entry_cancelled(*slot)) {
                                fail_slot(*slot,
                                          grpc::Status(grpc::StatusCode::CANCELLED, "EnqueueGroup request cancelled"));
                                return;
                            }
                            try {
                                slot->prefill_context->stat_info.nextStage();
                                remoteLoadCacheStart(*slot->prefill_context);
                                refreshAsyncProducerCancelState(slot->cancel_state,
                                                                slot->prefill_context->client_context,
                                                                slot->prefill_context->getStream());
                                if (entry_cancelled(*slot)) {
                                    fail_slot(
                                        *slot,
                                        grpc::Status(grpc::StatusCode::CANCELLED, "EnqueueGroup request cancelled"));
                                    return;
                                }
                                if (slot->prefill_context->hasError()) {
                                    auto status = slot->prefill_context->error_status.ok() ?
                                                      statusFromErrorInfo(slot->prefill_context->error_info) :
                                                      slot->prefill_context->error_status;
                                    fail_slot(*slot, status);
                                    return;
                                }
                                start_finish_worker(*slot);
                            } catch (const std::exception& e) {
                                fail_slot(*slot,
                                          grpc::Status(grpc::StatusCode::INTERNAL,
                                                       "remoteLoadCacheStart exception: " + std::string(e.what())));
                            } catch (...) {
                                fail_slot(
                                    *slot,
                                    grpc::Status(grpc::StatusCode::INTERNAL, "remoteLoadCacheStart unknown exception"));
                            }
                        }));
                }
                collectFutures(
                    load_futures,
                    load_deadline,
                    [&](size_t i) { load_futures[i].get(); },
                    [&](size_t i) { cancelResponseEntry(stream_ready_slots[i]->entry); });
                drainReadyFutures(load_futures, std::chrono::milliseconds(2000));
                detachLeftoverFutures(load_futures);
            });

        if (worker_error != autil::ThreadPoolBase::ERROR_NONE) {
            worker_lambda_pool_metrics_.queued--;
            worker_lambda_pool_metrics_.rejected++;
            // Pool saturated: the lambda was NOT enqueued, so ScopeExit guards
            // inside the lambda did not run. We must manually finish the worker.
            finishAsyncResponseWorker();

            auto status = grpc::Status(grpc::StatusCode::UNAVAILABLE, "EnqueueGroup enqueue pool saturated");
            for (auto& slot : *slots_ptr) {
                finish_pending_before_ack(slot, status);
                addBatchError(response, slot.request_id, status.error_code(), status.error_message());
            }
            erase_reserved_slots(*slots_ptr);
            return grpc::Status::OK;
        }
    } catch (const std::exception& e) {
        finishAsyncResponseWorker();
        auto status = grpc::Status(grpc::StatusCode::INTERNAL,
                                   "start EnqueueGroup accept worker exception: " + std::string(e.what()));
        for (const auto& slot : *slots_ptr) {
            finish_pending_before_ack(slot, status);
            addBatchError(response, slot.request_id, status.error_code(), status.error_message());
        }
        erase_reserved_slots(*slots_ptr);
        return grpc::Status::OK;
    } catch (...) {
        finishAsyncResponseWorker();
        auto status = grpc::Status(grpc::StatusCode::INTERNAL, "start EnqueueGroup accept worker unknown exception");
        for (const auto& slot : *slots_ptr) {
            finish_pending_before_ack(slot, status);
            addBatchError(response, slot.request_id, status.error_code(), status.error_message());
        }
        erase_reserved_slots(*slots_ptr);
        return grpc::Status::OK;
    }

    for (const auto request_id : accepted_request_ids) {
        addBatchSuccess(response, request_id);
    }
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::FetchResponse(grpc::ServerContext*                   context,
                                             const FetchRequestPB*                  request,
                                             grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_FUNCTION();
    const auto request_id = request->request_id();
    auto       entry      = response_registry_.get(request_id);
    if (!entry) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND,
                            "request [" + std::to_string(request_id) + "] not found in response registry");
    }

    while (true) {
        if (context && context->IsCancelled()) {
            cancelResponseEntry(entry);
            response_registry_.erase(request_id);
            return grpc::Status(grpc::StatusCode::CANCELLED, "fetch response cancelled by client");
        }

        std::deque<GenerateOutputsPB> drained;
        grpc::Status                  terminal_status = grpc::Status::OK;
        bool                          terminal        = false;
        {
            std::unique_lock<std::mutex> lock(entry->mu);
            entry->cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                return !entry->queue.empty() || entry->done.load() || entry->cancelled.load()
                       || entry->error_status.has_value();
            });
            drained.swap(entry->queue);
            if (entry->cancelled.load()) {
                terminal        = true;
                terminal_status = grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
            } else if (entry->error_status.has_value()) {
                terminal        = true;
                terminal_status = *entry->error_status;
            } else if (entry->done.load()) {
                terminal = true;
            }
        }

        for (auto& output : drained) {
            if (!writer->Write(output)) {
                cancelResponseEntry(entry);
                response_registry_.erase(request_id);
                return grpc::Status(grpc::StatusCode::CANCELLED, "client writer closed");
            }
        }

        if (terminal) {
            response_registry_.erase(request_id);
            return terminal_status;
        }
    }
}

grpc::Status PrefillRpcServer::Cancel(grpc::ServerContext* context, const CancelRequestPB* request, EmptyPB* response) {
    (void)context;
    (void)response;
    RTP_LLM_PROFILE_FUNCTION();
    const auto request_id = request->request_id();
    auto       entry      = response_registry_.get(request_id);
    if (!entry) {
        return grpc::Status::OK;
    }
    cancelResponseEntry(entry);
    response_registry_.erase(request_id);
    return grpc::Status::OK;
}

grpc::Status
PrefillRpcServer::RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    auto request_id = request->request_id();
    // In mock mode, resource_.cache_store is nullptr (MockCacheStore is not a NormalCacheStore).
    // markRequestEnd is a NormalCacheStore-specific cleanup method that is not needed in mock mode.
    if (resource_.cache_store) {
        resource_.cache_store->markRequestEnd(std::to_string(request_id));
    }
    return grpc::Status::OK;
}

}  // namespace rtp_llm
