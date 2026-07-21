#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <curl/curl.h>
#include <exception>
#include <mutex>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <thread>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/events/KVCacheEventQueue.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

bool jsonCodeIsOk(const rapidjson::Value& code) {
    if (code.IsString()) {
        return std::string(code.GetString()) == "OK" || std::string(code.GetString()) == "1";
    }
    return code.IsInt() && code.GetInt() == 1;
}

bool kvcmResponseIsOk(const std::string& response) {
    rapidjson::Document document;
    document.Parse(response.c_str());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("header")) {
        return false;
    }
    const auto& header = document["header"];
    if (!header.IsObject() || !header.HasMember("status")) {
        return false;
    }
    const auto& status = header["status"];
    if (!status.IsObject() || !status.HasMember("code") || !jsonCodeIsOk(status["code"])) {
        return false;
    }
    const char* item_results_key = document.HasMember("item_results") ?
                                       "item_results" :
                                       (document.HasMember("itemResults") ? "itemResults" : nullptr);
    if (item_results_key != nullptr) {
        const auto& item_results = document[item_results_key];
        if (!item_results.IsArray()) {
            return false;
        }
        for (const auto& item : item_results.GetArray()) {
            if (!jsonCodeIsOk(item)) {
                return false;
            }
        }
    }
    return true;
}

size_t appendCurlResponse(char* data, size_t size, size_t count, void* user_data) {
    const size_t bytes = size * count;
    try {
        static_cast<std::string*>(user_data)->append(data, bytes);
        return bytes;
    } catch (...) {
        return 0;
    }
}

class CurlKVCacheEventReporter final: public KVCacheEventReporter {
public:
    CurlKVCacheEventReporter(std::string endpoint, int request_timeout_ms):
        endpoint_(std::move(endpoint)), request_timeout_ms_(std::max(request_timeout_ms, 1)) {
        while (!endpoint_.empty() && endpoint_.back() == '/') {
            endpoint_.pop_back();
        }
        if (endpoint_.find("://") == std::string::npos) {
            endpoint_ = "http://" + endpoint_;
        }
        static std::once_flag curl_init_once;
        std::call_once(curl_init_once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    }

    bool post(const std::string& route, const std::string& request, std::string& response) noexcept override {
        try {
            return postImpl(route, request, response);
        } catch (...) {
            return false;
        }
    }

private:
    bool postImpl(const std::string& route, const std::string& request, std::string& response) {
        const std::string url  = endpoint_ + route;
        CURL*             curl = curl_easy_init();
        if (curl == nullptr) {
            return false;
        }

        char error_buffer[CURL_ERROR_SIZE] = {0};
        auto headers                       = curl_slist_append(nullptr, "Content-Type: application/json");
        if (headers == nullptr) {
            curl_easy_cleanup(curl);
            return false;
        }
        auto headers_with_accept = curl_slist_append(headers, "Accept: application/json");
        if (headers_with_accept == nullptr) {
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            return false;
        }
        headers = headers_with_accept;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.data());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.size()));
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(request_timeout_ms_));
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(request_timeout_ms_));
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, appendCurlResponse);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);

        const CURLcode result      = curl_easy_perform(curl);
        long           status_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (result != CURLE_OK || status_code < 200 || status_code >= 300) {
            RTP_LLM_LOG_WARNING("KVCM event request failed, route=%s curl_code=%d http_status=%ld error=%s",
                                route.c_str(),
                                static_cast<int>(result),
                                status_code,
                                error_buffer);
            return false;
        }
        if (!kvcmResponseIsOk(response)) {
            RTP_LLM_LOG_WARNING(
                "KVCM event request returned failure, route=%s response=%s", route.c_str(), response.c_str());
            return false;
        }
        return true;
    }

    std::string endpoint_;
    int         request_timeout_ms_;
};

void writeString(JsonWriter& writer, const char* key, const std::string& value) {
    writer.Key(key);
    writer.String(value.c_str(), static_cast<rapidjson::SizeType>(value.size()));
}

std::string buildRegisterInstanceRequest(const KVCacheEventPublisherContext& context, const std::string& trace_id) {
    rapidjson::StringBuffer buffer;
    JsonWriter              writer(buffer);
    writer.StartObject();
    writeString(writer, "trace_id", trace_id);
    writeString(writer, "instance_group", context.instance_group);
    writeString(writer, "instance_id", context.instance_id);
    writer.Key("block_size");
    writer.Int(context.block_size_tokens);

    writer.Key("model_deployment");
    writer.StartObject();
    writeString(writer, "model_name", context.model_name);
    writeString(writer, "dtype", context.dtype);
    writer.Key("use_mla");
    writer.Bool(context.use_mla);
    writer.Key("tp_size");
    writer.Int(context.tp_size);
    writer.Key("dp_size");
    writer.Int(context.dp_size);
    writer.Key("pp_size");
    writer.Int(context.pp_size);
    writer.EndObject();

    writer.Key("location_spec_infos");
    writer.StartArray();
    writer.StartObject();
    writeString(writer, "name", context.spec_name);
    writer.Key("size");
    writer.Int64(context.spec_size_bytes);
    writer.EndObject();
    writer.EndArray();

    writer.Key("location_spec_groups");
    writer.StartArray();
    writer.StartObject();
    writeString(writer, "name", "default");
    writer.Key("spec_names");
    writer.StartArray();
    writer.String(context.spec_name.c_str(), static_cast<rapidjson::SizeType>(context.spec_name.size()));
    writer.EndArray();
    writer.EndObject();
    writer.EndArray();
    writer.EndObject();
    return buffer.GetString();
}

void writeReportHeader(JsonWriter& writer, const KVCacheEventPublisherContext& context, const std::string& trace_id) {
    writer.StartObject();
    writeString(writer, "trace_id", trace_id);
    writeString(writer, "instance_id", context.instance_id);
    writeString(writer, "host_ip_port", context.host_ip_port);
    writer.Key("events");
    writer.StartArray();
}

void writeReportFooter(JsonWriter& writer) {
    writer.EndArray();
    writeString(writer, "storage_type", "ST_EVENT_REPORT");
    writer.EndObject();
}

void writeLocationSpecs(JsonWriter& writer, const KVCacheEventPublisherContext& context) {
    writer.Key("specs");
    writer.StartArray();
    writer.StartObject();
    writeString(writer, "name", context.spec_name);
    writeString(writer, "uri", context.location_uri);
    writer.EndObject();
    writer.EndArray();
}

std::string buildMutationReport(const KVCacheEventPublisherContext& context,
                                const std::string&                  trace_id,
                                const std::vector<KVCacheEvent>&    events) {
    rapidjson::StringBuffer buffer;
    JsonWriter              writer(buffer);
    writeReportHeader(writer, context, trace_id);
    for (const auto& event : events) {
        const std::string block_key = std::to_string(event.block_key);
        writer.StartObject();
        if (event.type == KVCacheEventType::BLOCK_ADD) {
            writeString(writer, "event_type", "EVENT_BLOCK_ADD");
            writer.Key("block_add");
            writer.StartObject();
            writeString(writer, "block_key", block_key);
            writeString(writer, "medium", "hbm");
            writeLocationSpecs(writer, context);
            writer.EndObject();
        } else {
            writeString(writer, "event_type", "EVENT_BLOCK_DELETE");
            writer.Key("block_delete");
            writer.StartObject();
            writeString(writer, "block_key", block_key);
            writeString(writer, "medium", "hbm");
            writer.Key("spec_names");
            writer.StartArray();
            writer.String(context.spec_name.c_str(), static_cast<rapidjson::SizeType>(context.spec_name.size()));
            writer.EndArray();
            writer.EndObject();
        }
        writer.EndObject();
    }
    writeReportFooter(writer);
    return buffer.GetString();
}

std::vector<KVCacheEvent> coalesceMutations(const std::vector<KVCacheEvent>& events) {
    // KVCM aggregates all ADDs before all DELETEs inside one ReportEvent
    // request. The final transition for each key is therefore the only one
    // this unsent batch may carry without inverting DELETE -> ADD.
    std::vector<KVCacheEvent>           coalesced;
    std::unordered_map<int64_t, size_t> key_to_index;
    coalesced.reserve(events.size());
    key_to_index.reserve(events.size());
    for (const auto& event : events) {
        const auto [it, inserted] = key_to_index.emplace(event.block_key, coalesced.size());
        if (inserted) {
            coalesced.push_back(event);
        } else {
            coalesced[it->second] = event;
        }
    }
    return coalesced;
}

enum class ControlEventType {
    HOST_DOWN,
    NODE_REGISTER,
    HEARTBEAT,
};

std::string
buildControlReport(const KVCacheEventPublisherContext& context, const std::string& trace_id, ControlEventType type) {
    rapidjson::StringBuffer buffer;
    JsonWriter              writer(buffer);
    writeReportHeader(writer, context, trace_id);
    writer.StartObject();
    switch (type) {
        case ControlEventType::HOST_DOWN:
            writeString(writer, "event_type", "EVENT_HOST_DOWN");
            writer.Key("host_down");
            writer.StartObject();
            writer.EndObject();
            break;
        case ControlEventType::NODE_REGISTER:
            writeString(writer, "event_type", "EVENT_NODE_REGISTER");
            writer.Key("node_register");
            writer.StartObject();
            writer.Key("mediums");
            writer.StartArray();
            writer.String("hbm");
            writer.EndArray();
            writer.EndObject();
            break;
        case ControlEventType::HEARTBEAT:
            writeString(writer, "event_type", "EVENT_HEARTBEAT");
            writer.Key("heartbeat");
            writer.StartObject();
            writer.Key("system_status");
            writer.StartObject();
            writeString(writer, "engine", "rtp-llm");
            writeString(writer, "dp_rank", std::to_string(context.dp_rank));
            writer.EndObject();
            writer.EndObject();
            break;
    }
    writer.EndObject();
    writeReportFooter(writer);
    return buffer.GetString();
}

std::string buildSnapshotReport(const KVCacheEventPublisherContext& context,
                                const std::string&                  trace_id,
                                const KVCacheSnapshot&              snapshot) {
    rapidjson::StringBuffer buffer;
    JsonWriter              writer(buffer);
    writeReportHeader(writer, context, trace_id);
    writer.StartObject();
    writeString(writer, "event_type", "EVENT_BLOCK_SNAPSHOT");
    writer.Key("block_snapshot");
    writer.StartObject();
    writeString(writer, "medium", "hbm");
    writer.Key("blocks");
    writer.StartArray();
    for (const auto block_key_value : snapshot.block_keys) {
        writer.StartObject();
        writeString(writer, "block_key", std::to_string(block_key_value));
        writeLocationSpecs(writer, context);
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
    writer.EndObject();
    writeReportFooter(writer);
    return buffer.GetString();
}

}  // namespace

class KVCMPublisher::Impl {
public:
    Impl(KVCacheEventPublisherConfig           config,
         KVCacheEventPublisherContext          context,
         KVCacheSnapshotProvider               snapshot_provider,
         std::shared_ptr<KVCacheEventReporter> reporter):
        config_(std::move(config)),
        context_(std::move(context)),
        snapshot_provider_(std::move(snapshot_provider)),
        reporter_(std::move(reporter)),
        queue_(config_.queue_capacity) {
        if (reporter_) {
            snapshot_reporter_ = reporter_;
        } else if (!config_.manager_endpoint.empty()) {
            reporter_ =
                std::make_shared<CurlKVCacheEventReporter>(config_.manager_endpoint, config_.request_timeout_ms);
            snapshot_reporter_ =
                std::make_shared<CurlKVCacheEventReporter>(config_.manager_endpoint, config_.snapshot_timeout_ms);
        }
    }

    ~Impl() {
        stop();
    }

    bool start() noexcept {
        if (!isConfigValid()) {
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("KVCMPublisher is disabled by invalid config: endpoint=%s instance_group=%s "
                                "instance_id=%s host_ip_port=%s snapshot_provider=%d",
                                config_.manager_endpoint.c_str(),
                                context_.instance_group.c_str(),
                                context_.instance_id.c_str(),
                                context_.host_ip_port.c_str(),
                                static_cast<int>(static_cast<bool>(snapshot_provider_)));
            return false;
        }
        bool expected = false;
        if (!started_.compare_exchange_strong(expected, true)) {
            return true;
        }
        state_.store(PublisherState::STARTING, std::memory_order_relaxed);
        try {
            worker_ = std::thread(&Impl::workerLoop, this);
        } catch (const std::exception& e) {
            started_.store(false, std::memory_order_relaxed);
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("start KVCMPublisher failed: %s", e.what());
            return false;
        }
        return true;
    }

    PublishResult tryPublish(KVCacheEvent event) noexcept {
        if (!started_.load(std::memory_order_relaxed) || stopping_.load(std::memory_order_relaxed)) {
            return PublishResult::NOT_RUNNING;
        }
        const auto result = queue_.tryPush(std::move(event));
        if (result == detail::QueuePushResult::ACCEPTED) {
            accepted_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            dirty_generation_.fetch_add(1, std::memory_order_relaxed);
            queue_.wake();
        }
        return detail::toPublishResult(result);
    }

    void stop() noexcept {
        if (!started_.load(std::memory_order_relaxed)) {
            return;
        }
        stopping_.store(true, std::memory_order_relaxed);
        queue_.stop();
        if (worker_.joinable()) {
            worker_.join();
        }
        started_.store(false, std::memory_order_relaxed);
        state_.store(PublisherState::STOPPED, std::memory_order_relaxed);
    }

    PublisherStatus status() const noexcept {
        return {state_.load(std::memory_order_relaxed),
                queue_.size(),
                accepted_count_.load(std::memory_order_relaxed),
                dropped_count_.load(std::memory_order_relaxed)};
    }

private:
    bool isConfigValid() const {
        return reporter_ && snapshot_reporter_ && snapshot_provider_ && !context_.instance_group.empty()
               && !context_.instance_id.empty()
               && !context_.host_ip_port.empty() && !context_.model_name.empty() && !context_.dtype.empty()
               && !context_.spec_name.empty() && !context_.location_uri.empty() && context_.block_size_tokens > 0
               && context_.spec_size_bytes > 0;
    }

    std::string nextTraceId(const char* operation) {
        return "rtp-kv-event-" + std::to_string(context_.dp_rank) + '-' + operation + '-'
               + std::to_string(next_request_id_++);
    }

    bool post(const std::string& route, const std::string& request) {
        std::string response;
        return reporter_->post(route, request, response);
    }

    bool registerNode() {
        state_.store(PublisherState::REGISTERING, std::memory_order_relaxed);
        auto trace_id = nextTraceId("register");
        if (!post("/api/registerInstance", buildRegisterInstanceRequest(context_, trace_id))) {
            return false;
        }
        trace_id = nextTraceId("node-register");
        return post("/api/reportEvent", buildControlReport(context_, trace_id, ControlEventType::NODE_REGISTER));
    }

    bool reconcile(uint64_t generation) {
        state_.store(PublisherState::RESYNCING, std::memory_order_relaxed);
        queue_.discardPending();

        KVCacheSnapshot snapshot;
        try {
            snapshot = snapshot_provider_();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("KVCMPublisher snapshot failed: %s", e.what());
            return false;
        } catch (...) {
            RTP_LLM_LOG_WARNING("KVCMPublisher snapshot failed with unknown exception");
            return false;
        }

        const auto trace_id = nextTraceId("snapshot");
        std::string response;
        if (!snapshot_reporter_->post(
                "/api/reportEvent", buildSnapshotReport(context_, trace_id, snapshot), response)) {
            return false;
        }

        reconciled_generation_ = generation;
        RTP_LLM_LOG_INFO("KVCMPublisher snapshot committed, instance_id=%s host=%s dp_rank=%d version=%lld keys=%zu "
                         "generation=%llu",
                         context_.instance_id.c_str(),
                         context_.host_ip_port.c_str(),
                         context_.dp_rank,
                         static_cast<long long>(snapshot.version),
                         snapshot.block_keys.size(),
                         static_cast<unsigned long long>(generation));
        return true;
    }

    bool reportMutations(const std::vector<KVCacheEvent>& batch) {
        if (batch.empty()) {
            return true;
        }
        const auto coalesced = coalesceMutations(batch);
        const auto trace_id  = nextTraceId("mutation");
        return post("/api/reportEvent", buildMutationReport(context_, trace_id, coalesced));
    }

    bool heartbeat() {
        const auto trace_id = nextTraceId("heartbeat");
        return post("/api/reportEvent", buildControlReport(context_, trace_id, ControlEventType::HEARTBEAT));
    }

    void waitBeforeRetry() {
        queue_.waitForStop(std::chrono::milliseconds(std::max(config_.retry_interval_ms, 1)));
    }

    void workerLoop() noexcept {
        bool registered     = false;
        auto next_heartbeat = std::chrono::steady_clock::now();
        auto next_snapshot =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(config_.snapshot_interval_ms, 1));
        try {
            while (!stopping_.load(std::memory_order_relaxed)) {
                if (!registered) {
                    if (!registerNode()) {
                        state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                        waitBeforeRetry();
                        continue;
                    }
                    registered = true;
                    dirty_generation_.fetch_add(1, std::memory_order_relaxed);
                    next_heartbeat = std::chrono::steady_clock::now()
                                     + std::chrono::milliseconds(std::max(config_.heartbeat_interval_ms, 1));
                }

                const auto now = std::chrono::steady_clock::now();
                if (now >= next_snapshot) {
                    dirty_generation_.fetch_add(1, std::memory_order_relaxed);
                    next_snapshot = now + std::chrono::milliseconds(std::max(config_.snapshot_interval_ms, 1));
                }

                const uint64_t dirty_generation = dirty_generation_.load(std::memory_order_relaxed);
                if (reconciled_generation_ != dirty_generation) {
                    if (!reconcile(dirty_generation)) {
                        registered = false;
                        state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                        waitBeforeRetry();
                        continue;
                    }
                    state_.store(PublisherState::READY, std::memory_order_relaxed);
                    continue;
                }

                auto batch = queue_.waitPop(config_.report_batch_size,
                                            std::chrono::milliseconds(std::max(config_.flush_interval_ms, 1)));
                if (!batch.empty() && !reportMutations(batch)) {
                    dirty_generation_.fetch_add(1, std::memory_order_relaxed);
                    registered = false;
                    state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                    waitBeforeRetry();
                    continue;
                }

                if (std::chrono::steady_clock::now() >= next_heartbeat) {
                    if (!heartbeat()) {
                        dirty_generation_.fetch_add(1, std::memory_order_relaxed);
                        registered = false;
                        state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                        waitBeforeRetry();
                        continue;
                    }
                    next_heartbeat = std::chrono::steady_clock::now()
                                     + std::chrono::milliseconds(std::max(config_.heartbeat_interval_ms, 1));
                }
            }

            if (registered) {
                const auto trace_id = nextTraceId("shutdown");
                (void)post("/api/reportEvent", buildControlReport(context_, trace_id, ControlEventType::HOST_DOWN));
            }
        } catch (const std::exception& e) {
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("KVCMPublisher worker stopped after exception: %s", e.what());
        } catch (...) {
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("KVCMPublisher worker stopped after unknown exception");
        }
    }

private:
    KVCacheEventPublisherConfig           config_;
    KVCacheEventPublisherContext          context_;
    KVCacheSnapshotProvider               snapshot_provider_;
    std::shared_ptr<KVCacheEventReporter> reporter_;
    std::shared_ptr<KVCacheEventReporter> snapshot_reporter_;
    detail::KVCacheEventQueue             queue_;
    std::thread                           worker_;
    std::atomic<bool>                     started_{false};
    std::atomic<bool>                     stopping_{false};
    std::atomic<PublisherState>           state_{PublisherState::DISABLED};
    std::atomic<uint64_t>                 accepted_count_{0};
    std::atomic<uint64_t>                 dropped_count_{0};
    std::atomic<uint64_t>                 dirty_generation_{1};
    uint64_t                              reconciled_generation_ = 0;
    uint64_t                              next_request_id_       = 1;
};

KVCMPublisher::KVCMPublisher(KVCacheEventPublisherConfig           config,
                             KVCacheEventPublisherContext          context,
                             KVCacheSnapshotProvider               snapshot_provider,
                             std::shared_ptr<KVCacheEventReporter> reporter):
    impl_(std::make_unique<Impl>(
        std::move(config), std::move(context), std::move(snapshot_provider), std::move(reporter))) {}

KVCMPublisher::~KVCMPublisher() = default;

bool KVCMPublisher::start() noexcept {
    return impl_->start();
}

PublishResult KVCMPublisher::tryPublish(KVCacheEvent event) noexcept {
    return impl_->tryPublish(std::move(event));
}

void KVCMPublisher::stop() noexcept {
    impl_->stop();
}

PublisherStatus KVCMPublisher::status() const noexcept {
    return impl_->status();
}

bool KVCMPublisher::enabled() const noexcept {
    return true;
}

}  // namespace rtp_llm
