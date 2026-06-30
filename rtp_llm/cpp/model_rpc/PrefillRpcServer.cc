#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <strings.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <unistd.h>
#include <limits.h>
#include <cstdint>

using namespace std;
using namespace autil::legacy;

using grpc::Status;
using grpc::ClientContext;

namespace rtp_llm {

namespace {

bool envValueIsFalse(const char* value) {
    return value != nullptr
           && (strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "off") == 0
               || strcasecmp(value, "no") == 0);
}

bool envValueIsTrue(const char* value) {
    return value != nullptr
           && (strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "on") == 0
               || strcasecmp(value, "yes") == 0);
}

bool prefillCacheHitMetricEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PREFILL_CACHE_HIT_METRIC_ENABLE");
        if (value == nullptr || value[0] == 0) {
            return true;
        }
        return !envValueIsFalse(value);
    }();
    return enabled;
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
    collector.theory_1m_hit_count    = snapshot.window_1m.hit_count;
    collector.theory_1m_total_count  = snapshot.window_1m.total_count;
    collector.theory_1m_hit_ratio    = snapshot.window_1m.hit_ratio;
    collector.theory_5m_hit_count    = snapshot.window_5m.hit_count;
    collector.theory_5m_total_count  = snapshot.window_5m.total_count;
    collector.theory_5m_hit_ratio    = snapshot.window_5m.hit_ratio;
    collector.theory_10m_hit_count   = snapshot.window_10m.hit_count;
    collector.theory_10m_total_count = snapshot.window_10m.total_count;
    collector.theory_10m_hit_ratio   = snapshot.window_10m.hit_ratio;
    collector.theory_15m_hit_count   = snapshot.window_15m.hit_count;
    collector.theory_15m_total_count = snapshot.window_15m.total_count;
    collector.theory_15m_hit_ratio   = snapshot.window_15m.hit_ratio;
}

// REBASE CONFLICT CONTEXT(cdc1b18b6): keep the new base prefill cache-hit
// metric helpers above and also retain the source branch remote-RPC role check
// used by GLM5 MTP prefill.
const bool kHasRemoteRpcServerIp = []() {
    const char* env = std::getenv("REMOTE_RPC_SERVER_IP");
    return env != nullptr && std::strlen(env) > 0;
}();

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

grpc::Status PrefillRpcServer::init(const EngineInitParams&                                maga_init_params,
                                    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                    py::object                                             mm_process_engine) {
    RTP_LLM_CHECK_WITH_INFO(maga_init_params.pd_sep_config.role_type == RoleType::PREFILL,
                            "prefill's role_type must be PREFILL");
    auto ret = RemoteRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        return ret;
    }
    if (prefillCacheHitMetricEnabled()) {
        prefill_recent_cache_key_window_ = std::make_unique<RecentCacheKeyWindow>();
    } else {
        RTP_LLM_LOG_INFO("prefill recent-cache-key metrics disabled by PREFILL_CACHE_HIT_METRIC_ENABLE");
    }
    return grpc::Status::OK;
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
    bool has_master_role = kHasRemoteRpcServerIp;

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
        if (!result.ok()) {
            prefill_context.error_info = result;
            prefill_context.error_status =
                serializeErrorMsg(prefill_context.request_key, prefill_context.request_info, result);
            return;
        }

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
    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", prefill_context.request_id);
}

void PrefillRpcServer::remoteLoadCacheStart(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] remote load cache", prefill_context.request_id);
    prefill_context.error_info = waitStreamBeforeRun(prefill_context.getStream());
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
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Write(load_request), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
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
        if (prefill_context.server_context->IsCancelled()) {
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
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.closeGrpcStream().ok(), ErrorCode::REMOTE_GENERATE_FAILED);
}

grpc::Status PrefillRpcServer::prepareAllocateResource(PrefillGenerateContext& prefill_context) {
    EXECUTE_STAGE_FUNC(getRpcConnection, prefill_context);
    EXECUTE_STAGE_FUNC(multimodalProcess, prefill_context);
    reportPrefillRecentCacheKeyMetricsOnce(prefill_context);
    EXECUTE_STAGE_FUNC(remoteAllocateResource, prefill_context);
    return grpc::Status::OK;
}

void PrefillRpcServer::reportPrefillRecentCacheKeyMetricsOnce(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (prefill_context.recent_cache_key_metric_reported) {
        return;
    }
    if (!prefillCacheHitMetricEnabled()) {
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
                                                  meta_);
    prefill_context.onflight_requests      = onflight_requests_;
    prefill_context.loading_cache_requests = loading_cache_requests_;

    auto max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;
    int  retry_interval_ms    = 1;

    try {
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
        EXECUTE_STAGE_FUNC(pollLocalOutput, prefill_context);
        EXECUTE_STAGE_FUNC(remoteLoadCacheEnd, prefill_context);
        EXECUTE_STAGE_FUNC(remoteGenerate, prefill_context);
        EXECUTE_STAGE_FUNC(pollRemoteOutput, prefill_context);
        prefill_context.stat_info.nextStage();
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

grpc::Status
PrefillRpcServer::RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    auto request_id = request->request_id();
    resource_.cache_store->markRequestEnd(std::to_string(request_id));
    return grpc::Status::OK;
}

}  // namespace rtp_llm
