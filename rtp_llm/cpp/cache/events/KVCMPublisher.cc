#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <curl/curl.h>
#include <exception>
#include <grpcpp/create_channel.h>
#include <grpcpp/generic/generic_stub.h>
#include <grpcpp/security/credentials.h>
#include <map>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <thread>
#include <utility>

#include "rtp_llm/cpp/cache/events/KVCacheEventQueue.h"
#include "rtp_llm/cpp/cache/events/KVCMPublisherUtils.h"
#include "rtp_llm/cpp/cache/events/kvcm_discovery.pb.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DirectSubscriber.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/VIPServerSubscriber.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

bool jsonCodeMatches(const rapidjson::Value& code, int number, const char* name) {
    if (code.IsInt()) {
        return code.GetInt() == number;
    }
    if (code.IsString()) {
        const std::string value = code.GetString();
        return value == name || value == std::to_string(number);
    }
    if (code.IsObject()) {
        if (code.HasMember("code")) {
            return jsonCodeMatches(code["code"], number, name);
        }
        if (code.HasMember("status")) {
            return jsonCodeMatches(code["status"], number, name);
        }
    }
    return false;
}

bool jsonCodeIsOk(const rapidjson::Value& code) {
    return jsonCodeMatches(code, 1, "OK");
}

bool jsonCodeRequiresRegistration(const rapidjson::Value& code) {
    return jsonCodeMatches(code, 8, "INSTANCE_NOT_EXIST")
           || jsonCodeMatches(code, 9, "SERVER_NOT_LEADER")
           || jsonCodeMatches(code, 10, "NODE_NOT_REGISTERED");
}

}  // namespace

namespace detail {

KVCMReportFeedback parseKVCMReportFeedback(const std::string& response) {
    KVCMReportFeedback feedback;
    rapidjson::Document document;
    document.Parse(response.c_str());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("header")) {
        return feedback;
    }
    const auto& header = document["header"];
    if (!header.IsObject() || !header.HasMember("status")) {
        return feedback;
    }
    const auto& status = header["status"];
    if (!status.IsObject() || !status.HasMember("code")) {
        return feedback;
    }
    feedback.valid = true;
    feedback.ok = jsonCodeIsOk(status["code"]);
    const auto& code = status["code"];
    if (code.IsInt() || code.IsString() || code.IsObject()) {
        feedback.registration_required = jsonCodeRequiresRegistration(code);
        feedback.snapshot_required = jsonCodeMatches(code, 14, "SNAPSHOT_REQUIRED");
    }
    const char* snapshot_key = document.HasMember("snapshot_required") ? "snapshot_required" :
                               (document.HasMember("snapshotRequired") ? "snapshotRequired" : nullptr);
    if (snapshot_key != nullptr) {
        if (!document[snapshot_key].IsBool()) {
            return {};
        }
        feedback.snapshot_required = feedback.snapshot_required || document[snapshot_key].GetBool();
    }
    const char* item_results_key = document.HasMember("item_results") ?
                                       "item_results" :
                                       (document.HasMember("itemResults") ? "itemResults" : nullptr);
    if (item_results_key != nullptr) {
        const auto& item_results = document[item_results_key];
        if (!item_results.IsArray()) {
            return {};
        }
        for (const auto& item : item_results.GetArray()) {
            if (!jsonCodeIsOk(item)) {
                feedback.ok = false;
                feedback.snapshot_required = feedback.snapshot_required || jsonCodeMatches(item, 14, "SNAPSHOT_REQUIRED");
                feedback.registration_required = feedback.registration_required
                                                 || jsonCodeRequiresRegistration(item);
            }
        }
    }
    // extra_info is opaque to KVCM. A malformed control field must not turn
    // an otherwise valid event ACK into a registration/snapshot failure.
    const char* extra_key = document.HasMember("extra_info") ? "extra_info" :
                            (document.HasMember("extraInfo") ? "extraInfo" : nullptr);
    if (extra_key && document[extra_key].IsString()) {
        rapidjson::Document extra;
        extra.Parse(document[extra_key].GetString(), document[extra_key].GetStringLength());
        if (!extra.HasParseError() && extra.IsObject() && extra.HasMember("rtp_remote_cache_enabled")
            && extra["rtp_remote_cache_enabled"].IsBool()) {
            feedback.remote_cache_enabled = extra["rtp_remote_cache_enabled"].GetBool();
        }
    }
    return feedback;
}

bool kvcmResponseIsOk(const std::string& response) {
    return parseKVCMReportFeedback(response).ok;
}

std::string normalizeKVCacheEventEndpoint(std::string endpoint) {
    while (!endpoint.empty() && endpoint.back() == '/') {
        endpoint.pop_back();
    }
    if (!endpoint.empty() && endpoint.find("://") == std::string::npos) {
        endpoint = "http://" + endpoint;
    }
    return endpoint;
}

}  // namespace detail

namespace {

size_t appendCurlResponse(char* data, size_t size, size_t count, void* user_data) {
    const size_t bytes = size * count;
    try {
        static_cast<std::string*>(user_data)->append(data, bytes);
        return bytes;
    } catch (...) { return 0; }
}

int abortCancelledCurlTransfer(void* user_data, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    return static_cast<std::atomic<bool>*>(user_data)->load(std::memory_order_acquire) ? 1 : 0;
}

class KVCMEndpointResolver {
public:
    KVCMEndpointResolver(const KVCacheEventPublisherConfig& config, std::string instance_id):
        endpoint_(detail::normalizeKVCacheEventEndpoint(config.manager_endpoint)),
        discover_(endpoint_.empty()),
        timeout_ms_(std::max(config.request_timeout_ms, 1)),
        instance_id_(std::move(instance_id)) {
        if (discover_) {
            if (config.enable_vipserver) {
                subscriber_ = std::make_unique<kvcm::VIPServerSubscriber>();
                domains_ = {config.vipserver_domain};
            } else {
                subscriber_ = std::make_unique<kvcm::DirectSubscriber>();
                domains_ = config.server_addresses.empty() ? std::vector<std::string>{config.server_address} :
                                                            config.server_addresses;
            }
        }
    }

    std::string endpoint() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!endpoint_.empty()) {
            return endpoint_;
        }
        if (!subscriber_initialized_) {
            if (!subscriber_->init(domains_)) {
                return {};
            }
            subscriber_initialized_ = true;
        }
        std::vector<std::string> addresses;
        if (!subscriber_->getAddresses(addresses)) {
            return {};
        }
        for (const auto& address : addresses) {
            const auto endpoint = discoverLeader(address);
            if (!endpoint.empty()) {
                endpoint_ = endpoint;
                RTP_LLM_LOG_INFO("KVCM event leader Meta HTTP endpoint resolved: %s", endpoint_.c_str());
                return endpoint_;
            }
        }
        RTP_LLM_LOG_WARNING("KVCM event leader discovery failed for all %zu RPC addresses", addresses.size());
        return {};
    }

    void invalidate(const std::string& endpoint) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (discover_ && endpoint_ == endpoint) {
            endpoint_.clear();
        }
    }

private:
    std::string discoverLeader(const std::string& address) const {
        kvcm::discovery::GetClusterInfoRequest request;
        request.set_trace_id("rtp-kv-event-discovery");
        request.set_instance_id(instance_id_);
        const std::string request_data = request.SerializeAsString();
        grpc::Slice request_slice(request_data);
        grpc::ByteBuffer request_buffer(&request_slice, 1);
        grpc::ByteBuffer response_buffer;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms_));
        grpc::GenericStub stub(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
        grpc::CompletionQueue queue;
        auto call = stub.PrepareUnaryCall(
            &context, "/kv_cache_manager.proto.meta.MetaService/GetClusterInfo", request_buffer, &queue);
        if (!call) {
            queue.Shutdown();
            void* tag = nullptr;
            bool completed = false;
            while (queue.Next(&tag, &completed)) {}
            return {};
        }
        call->StartCall();
        grpc::Status status;
        int completion_tag = 0;
        call->Finish(&response_buffer, &status, &completion_tag);
        void* tag = nullptr;
        bool completed = false;
        const bool received = queue.Next(&tag, &completed);
        queue.Shutdown();
        void* remaining_tag = nullptr;
        bool remaining_ok = false;
        while (queue.Next(&remaining_tag, &remaining_ok)) {}
        if (!received || !completed || tag != &completion_tag || !status.ok()) {
            RTP_LLM_LOG_WARNING("KVCM event leader lookup failed, RPC address=%s grpc_code=%d error=%s",
                                address.c_str(), static_cast<int>(status.error_code()), status.error_message().c_str());
            return {};
        }
        std::vector<grpc::Slice> slices;
        if (!response_buffer.Dump(&slices).ok()) {
            return {};
        }
        std::string response_data;
        for (const auto& slice : slices) {
            response_data.append(reinterpret_cast<const char*>(slice.begin()), slice.size());
        }
        kvcm::discovery::GetClusterInfoResponse response;
        if (!response.ParseFromString(response_data) || response.header().status().code() != 1
            || !response.has_leader_endpoint()) {
            return {};
        }
        const auto& leader = response.leader_endpoint();
        if (response.leader_node_id().empty() || leader.node_id() != response.leader_node_id()
            || leader.host().empty() || leader.meta_http_port() <= 0 || leader.meta_http_port() > 65535) {
            return {};
        }
        const auto host = leader.host().find(':') == std::string::npos ? leader.host() : "[" + leader.host() + "]";
        return "http://" + host + ":" + std::to_string(leader.meta_http_port());
    }

    std::mutex mutex_;
    std::string endpoint_;
    bool discover_;
    int timeout_ms_;
    std::string instance_id_;
    std::unique_ptr<kvcm::Subscriber> subscriber_;
    std::vector<std::string> domains_;
    bool subscriber_initialized_ = false;
};

class CurlKVCacheEventReporter final: public KVCacheEventReporter {
public:
    CurlKVCacheEventReporter(std::shared_ptr<KVCMEndpointResolver> resolver, int request_timeout_ms):
        resolver_(std::move(resolver)),
        request_timeout_ms_(std::max(request_timeout_ms, 1)) {
        // Keep this publisher's process-wide curl initialization idempotent.
        // Other curl users remain responsible for following libcurl's global
        // initialization contract before starting concurrent transfers.
        static std::once_flag curl_init_once;
        std::call_once(curl_init_once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    }

    bool post(const std::string& route, const std::string& request, std::string& response) noexcept override {
        if (cancelled_.load(std::memory_order_acquire)) {
            return false;
        }
        try {
            response.clear();
            const auto endpoint = resolver_->endpoint();
            if (endpoint.empty()) {
                return false;
            }
            const bool success = postImpl(endpoint, route, request, response);
            if (!success) {
                resolver_->invalidate(endpoint);
            }
            return success;
        } catch (...) { return false; }
    }

    void cancel() noexcept override {
        cancelled_.store(true, std::memory_order_release);
    }

private:
    bool postImpl(const std::string& endpoint, const std::string& route,
                  const std::string& request, std::string& response) {
        const std::string url  = endpoint + route;
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
        curl_easy_setopt(curl, CURLOPT_EXPECT_100_TIMEOUT_MS, 0L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, appendCurlResponse);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, abortCancelledCurlTransfer);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &cancelled_);

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
        if (!detail::kvcmResponseIsOk(response)) {
            constexpr size_t kMaxLoggedResponseBytes = 256;
            const auto       response_prefix         = response.substr(0, kMaxLoggedResponseBytes);
            RTP_LLM_LOG_WARNING("KVCM event request returned failure, route=%s response_size=%zu response_prefix=%s%s",
                                route.c_str(),
                                response.size(),
                                response_prefix.c_str(),
                                response.size() > kMaxLoggedResponseBytes ? "..." : "");
            return false;
        }
        return true;
    }

    std::shared_ptr<KVCMEndpointResolver> resolver_;
    int               request_timeout_ms_;
    std::atomic<bool> cancelled_{false};
};

void writeString(JsonWriter& writer, const char* key, const std::string& value) {
    writer.Key(key);
    writer.String(value.c_str(), static_cast<rapidjson::SizeType>(value.size()));
}

std::string buildRegisterInstanceRequest(const KVCacheEventPublisherContext& context, const std::string& trace_id) {
    if (!context.register_instance_request.empty()) {
        rapidjson::Document request;
        request.Parse(context.register_instance_request.c_str());
        auto& allocator = request.GetAllocator();
        if (request.HasMember("trace_id")) {
            request["trace_id"].SetString(trace_id.c_str(), allocator);
        } else {
            request.AddMember("trace_id", rapidjson::Value(trace_id.c_str(), allocator), allocator);
        }
        rapidjson::StringBuffer buffer;
        JsonWriter writer(buffer);
        request.Accept(writer);
        return buffer.GetString();
    }
    rapidjson::StringBuffer buffer;
    JsonWriter              writer(buffer);
    writer.StartObject();
    writeString(writer, "trace_id", trace_id);
    writeString(writer, "instance_group", context.instance_group);
    writeString(writer, "instance_id", context.instance_id);
    writer.Key("block_size");
    writer.Int(context.block_size_tokens);
    writeString(writer, "default_query_type", "QT_PREFIX_MATCH");

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
    for (const auto& spec : context.specs) {
        writer.StartObject();
        writeString(writer, "name", spec.name);
        writer.EndObject();
    }
    writer.EndArray();

    writer.Key("location_spec_groups");
    writer.StartArray();
    writer.StartObject();
    writeString(writer, "name", "default");
    writer.Key("spec_names");
    writer.StartArray();
    for (const auto& spec : context.specs) {
        writer.String(spec.name.c_str(), static_cast<rapidjson::SizeType>(spec.name.size()));
    }
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
    writeString(writer, "storage_type", "ST_EVENT_REPORT_L1P5");
    writer.EndObject();
}

void writeLocationSpec(JsonWriter& writer, const KVCacheEventSpec& spec) {
    writer.StartObject();
    writeString(writer, "name", spec.name);
    writeString(writer, "uri", spec.location_uri);
    writer.EndObject();
}

std::string buildMutationReport(const KVCacheEventPublisherContext& context,
                                const std::string&                  trace_id,
                                const std::vector<KVCacheEvent>&    events) {
    rapidjson::StringBuffer buffer;
    JsonWriter              writer(buffer);
    writeReportHeader(writer, context, trace_id);
    for (const auto& event : events) {
        const std::string block_key = std::to_string(event.block_key);
        const auto&       spec      = context.specs.at(event.group_index);
        writer.StartObject();
        if (event.type == KVCacheEventType::BLOCK_ADD) {
            writeString(writer, "event_type", "EVENT_BLOCK_ADD");
            writer.Key("block_add");
            writer.StartObject();
            writeString(writer, "block_key", block_key);
            writeString(writer, "medium", "hbm");
            writer.Key("specs");
            writer.StartArray();
            writeLocationSpec(writer, spec);
            writer.EndArray();
            writer.EndObject();
        } else {
            writeString(writer, "event_type", "EVENT_BLOCK_DELETE");
            writer.Key("block_delete");
            writer.StartObject();
            writeString(writer, "block_key", block_key);
            writeString(writer, "medium", "hbm");
            writer.Key("spec_names");
            writer.StartArray();
            writer.String(spec.name.c_str(), static_cast<rapidjson::SizeType>(spec.name.size()));
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
    // request. The final transition for each (key, group) is therefore the only one
    // this unsent batch may carry without inverting DELETE -> ADD.
    std::vector<KVCacheEvent>                   coalesced;
    std::map<std::pair<int64_t, size_t>, size_t> component_to_index;
    coalesced.reserve(events.size());
    for (const auto& event : events) {
        const auto [it, inserted] =
            component_to_index.emplace(std::make_pair(event.block_key, event.group_index), coalesced.size());
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
buildControlReport(const KVCacheEventPublisherContext& context, const std::string& trace_id, ControlEventType type,
                   bool publish_cache_events = true) {
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
            if (publish_cache_events) {
                writer.String("hbm");
            }
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
    writer.Key("blocks");
    writer.StartArray();
    for (const auto& [block_key_value, group_indices] : snapshot.block_groups) {
        writer.StartObject();
        writeString(writer, "block_key", std::to_string(block_key_value));
        writeString(writer, "medium", "hbm");
        writer.Key("specs");
        writer.StartArray();
        for (const auto group_index : group_indices) {
            writeLocationSpec(writer, context.specs.at(group_index));
        }
        writer.EndArray();
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
    struct SnapshotReport {
        uint64_t    generation = 0;
        int64_t     version    = -1;
        size_t      key_count  = 0;
        std::string request;
    };

public:
    Impl(KVCacheEventPublisherConfig           config,
         KVCacheEventPublisherContext          context,
         KVCacheSnapshotProvider               snapshot_provider,
         std::shared_ptr<KVCacheEventReporter> reporter,
         RemoteCacheControl                   remote_cache_control):
        config_(std::move(config)),
        context_(std::move(context)),
        snapshot_provider_(std::move(snapshot_provider)),
        reporter_(std::move(reporter)),
        remote_cache_control_(std::move(remote_cache_control)),
        queue_(config_.publish_cache_events ? config_.queue_capacity : 1) {
        if (reporter_) {
            snapshot_reporter_ = reporter_;
        } else if (!config_.manager_endpoint.empty()
                   || (config_.enable_vipserver ? !config_.vipserver_domain.empty() :
                       (!config_.server_address.empty() || !config_.server_addresses.empty()))) {
            auto resolver = std::make_shared<KVCMEndpointResolver>(config_, context_.instance_id);
            reporter_ =
                std::make_shared<CurlKVCacheEventReporter>(resolver, config_.request_timeout_ms);
            if (config_.publish_cache_events) {
                snapshot_reporter_ =
                    std::make_shared<CurlKVCacheEventReporter>(resolver, config_.snapshot_timeout_ms);
            }
        }
    }

    ~Impl() {
        stop();
    }

    bool start() noexcept {
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        if (stopped_permanently_) {
            RTP_LLM_LOG_WARNING("KVCMPublisher cannot restart after stop");
            return false;
        }
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
        if (started_.load(std::memory_order_relaxed)) {
            return true;
        }
        stopping_.store(false, std::memory_order_relaxed);
        registered_.store(false, std::memory_order_relaxed);
        registered_once_.store(false, std::memory_order_relaxed);
        state_.store(PublisherState::STARTING, std::memory_order_relaxed);
        try {
            worker_           = std::thread(&Impl::workerLoop, this);
            heartbeat_worker_ = std::thread(&Impl::heartbeatLoop, this);
            if (config_.publish_cache_events) {
                snapshot_worker_  = std::thread(&Impl::snapshotLoop, this);
                snapshot_sender_  = std::thread(&Impl::snapshotSendLoop, this);
            }
        } catch (const std::exception& e) {
            {
                std::lock_guard<std::mutex> lock(snapshot_mu_);
                stopping_.store(true, std::memory_order_relaxed);
            }
            queue_.stop();
            snapshot_cv_.notify_all();
            if (worker_.joinable()) {
                worker_.join();
            }
            if (heartbeat_worker_.joinable()) {
                heartbeat_worker_.join();
            }
            if (snapshot_worker_.joinable()) {
                snapshot_worker_.join();
            }
            if (snapshot_sender_.joinable()) {
                snapshot_sender_.join();
            }
            // queue_.stop() is terminal; a failed thread launch cannot reuse it.
            stopped_permanently_ = true;
            started_.store(false, std::memory_order_relaxed);
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("start KVCMPublisher failed: %s", e.what());
            return false;
        }
        started_.store(true, std::memory_order_relaxed);
        return true;
    }

    PublishResult tryPublish(KVCacheEvent event) noexcept {
        if (!config_.publish_cache_events) {
            return PublishResult::DISABLED;
        }
        if (!started_.load(std::memory_order_relaxed) || stopping_.load(std::memory_order_relaxed)) {
            return PublishResult::NOT_RUNNING;
        }
        const auto result = queue_.tryPush(std::move(event));
        if (result == detail::QueuePushResult::ACCEPTED) {
            accepted_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            if (result == detail::QueuePushResult::FULL) {
                queue_full_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return detail::toPublishResult(result);
    }

    void stop() noexcept {
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        if (!started_.load(std::memory_order_relaxed) && !worker_.joinable()) {
            stopping_.store(true, std::memory_order_relaxed);
            stopped_permanently_ = true;
            state_.store(PublisherState::STOPPED, std::memory_order_relaxed);
            return;
        }
        // Pair shutdown with both snapshot wait predicates to avoid lost wakeups.
        {
            std::lock_guard<std::mutex> lock(snapshot_mu_);
            stopping_.store(true, std::memory_order_relaxed);
            snapshot_report_.reset();
        }
        queue_.stop();
        snapshot_cv_.notify_all();
        // Keep a shared test reporter usable for the final HOST_DOWN.
        if (snapshot_reporter_ && snapshot_reporter_ != reporter_) {
            snapshot_reporter_->cancel();
        }
        if (heartbeat_worker_.joinable()) {
            heartbeat_worker_.join();
        }
        if (worker_.joinable()) {
            worker_.join();
        }
        if (snapshot_worker_.joinable()) {
            snapshot_worker_.join();
        }
        if (snapshot_sender_.joinable()) {
            snapshot_sender_.join();
        }
        // Registration may have completed while stop was requested. Join every
        // producer/sender before HOST_DOWN so it is always the final request.
        const bool was_registered = registered_once_.load(std::memory_order_relaxed);
        if (was_registered && reporter_) {
            const auto trace_id = nextTraceId("shutdown");
            std::string response;
            (void)reporter_->post("/api/reportEvent",
                                   buildControlReport(context_, trace_id, ControlEventType::HOST_DOWN), response);
        }
        started_.store(false, std::memory_order_relaxed);
        stopped_permanently_ = true;
        state_.store(PublisherState::STOPPED, std::memory_order_relaxed);
    }

    PublisherStatus status() const noexcept {
        return {state_.load(std::memory_order_relaxed),
                queue_.size(),
                accepted_count_.load(std::memory_order_relaxed),
                dropped_count_.load(std::memory_order_relaxed),
                queue_full_count_.load(std::memory_order_relaxed),
                report_failed_count_.load(std::memory_order_relaxed)};
    }

private:
    bool isConfigValid() const {
        if (!config_.publish_cache_events) {
            rapidjson::Document request;
            request.Parse(context_.register_instance_request.c_str());
            return reporter_ && !context_.instance_group.empty() && !context_.instance_id.empty()
                   && !context_.host_ip_port.empty() && !request.HasParseError() && request.IsObject()
                   && request.HasMember("instance_id") && request["instance_id"].IsString()
                   && context_.instance_id == request["instance_id"].GetString()
                   && request.HasMember("instance_group") && request["instance_group"].IsString()
                   && context_.instance_group == request["instance_group"].GetString()
                   && request.HasMember("location_spec_infos") && request["location_spec_infos"].IsArray()
                   && !request["location_spec_infos"].Empty();
        }
        for (auto it = context_.specs.begin(); it != context_.specs.end(); ++it) {
            if (it->name.empty() || it->location_uri.empty()
                || std::any_of(context_.specs.begin(), it, [&](const auto& spec) { return spec.name == it->name; })) {
                return false;
            }
        }
        return reporter_ && snapshot_reporter_ && snapshot_provider_ && !context_.instance_group.empty()
               && !context_.instance_id.empty() && !context_.host_ip_port.empty() && !context_.model_name.empty()
               && !context_.dtype.empty() && context_.block_size_tokens > 0 && !context_.specs.empty()
               && config_.snapshot_interval_ms > 0;
    }

    std::string nextTraceId(const char* operation) {
        return "rtp-kv-event-" + std::to_string(context_.dp_rank) + '-' + operation + '-'
               + std::to_string(next_request_id_.fetch_add(1, std::memory_order_relaxed));
    }

    void refreshNormalAckDeadline() {
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        const int64_t deadline = now_ms + std::max(config_.ack_recovery_timeout_ms, 1);
        auto previous = normal_ack_deadline_ms_.load(std::memory_order_relaxed);
        while (previous < deadline && !normal_ack_deadline_ms_.compare_exchange_weak(
                   previous, deadline, std::memory_order_relaxed)) {}
    }

    void requestSnapshot(uint64_t generation) {
        if (!config_.publish_cache_events) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(snapshot_mu_);
            if (stopping_.load(std::memory_order_relaxed) || !registered_.load(std::memory_order_acquire)
                || generation != registration_generation_.load(std::memory_order_acquire)) {
                return;
            }
            snapshot_requested_ = true;
        }
        snapshot_cv_.notify_all();
    }

    void applyRemoteCacheControl(const detail::KVCMReportFeedback& feedback) noexcept {
        if (feedback.remote_cache_enabled.has_value() && remote_cache_control_) {
            try {
                remote_cache_control_(*feedback.remote_cache_enabled);
            } catch (...) {
                RTP_LLM_LOG_WARNING("KVCMPublisher remote cache control callback failed");
            }
        }
    }

    bool observeReportResponse(const detail::KVCMReportFeedback& feedback, bool success, uint64_t generation,
                               bool normal_ack = false, bool apply_remote_control = false) {
        const bool acknowledged = success && feedback.valid && feedback.ok;
        {
            std::lock_guard<std::mutex> lock(snapshot_mu_);
            if (stopping_.load(std::memory_order_relaxed)
                || generation != registration_generation_.load(std::memory_order_acquire)) {
                return false;
            }
            // An unknown transport outcome does not prove that registration was lost.
            if (feedback.registration_required) {
                normal_ack_deadline_ms_.store(0, std::memory_order_relaxed);
                registered_.store(false, std::memory_order_release);
            } else if (acknowledged && normal_ack) {
                refreshNormalAckDeadline();
            }
            if (acknowledged && apply_remote_control) {
                applyRemoteCacheControl(feedback);
            }
        }
        if (feedback.registration_required) {
            queue_.wake();
            snapshot_cv_.notify_all();
        }
        return acknowledged;
    }

    bool registerNode() {
        // Drain an old in-flight snapshot before starting a new registration.
        // Normal incremental sends never acquire this lock.
        std::lock_guard<std::mutex> send_lock(snapshot_send_mu_);
        if (stopping_.load(std::memory_order_relaxed)) {
            return false;
        }
        state_.store(PublisherState::REGISTERING, std::memory_order_relaxed);
        auto trace_id = nextTraceId("register");
        std::string response;
        if (!reporter_->post("/api/registerInstance", buildRegisterInstanceRequest(context_, trace_id), response)) {
            return false;
        }
        const auto registration_feedback = detail::parseKVCMReportFeedback(response);
        if (!registration_feedback.valid || !registration_feedback.ok) {
            return false;
        }
        trace_id = nextTraceId("node-register");
        const bool success = reporter_->post(
            "/api/reportEvent",
            buildControlReport(context_, trace_id, ControlEventType::NODE_REGISTER, config_.publish_cache_events),
            response);
        const auto node_feedback = detail::parseKVCMReportFeedback(response);
        if (!observeReportResponse(node_feedback, success,
                                   registration_generation_.load(std::memory_order_acquire))) {
            return false;
        }
        if (!registered_once_.load(std::memory_order_relaxed)) {
            // Bootstrap the first window only. Recovery ACKs cannot rearm it.
            refreshNormalAckDeadline();
        }
        {
            std::lock_guard<std::mutex> lock(snapshot_mu_);
            if (stopping_.load(std::memory_order_relaxed)) {
                return false;
            }
            registration_generation_.fetch_add(1, std::memory_order_release);
            // NODE_REGISTER is newer than RegisterInstance. Prefer its value
            // when present, and publish before the heartbeat worker can run.
            applyRemoteCacheControl(node_feedback.remote_cache_enabled.has_value() ? node_feedback :
                                                                                     registration_feedback);
            snapshot_report_.reset();
            snapshot_requested_ = config_.publish_cache_events;
            registered_.store(true, std::memory_order_release);
        }
        registered_once_.store(true, std::memory_order_relaxed);
        state_.store(PublisherState::READY, std::memory_order_relaxed);
        snapshot_cv_.notify_all();
        return true;
    }

    void snapshotLoop() noexcept {
        pthread_setname_np(pthread_self(), "kvcm_snapshot");
        const auto interval = std::chrono::milliseconds(config_.snapshot_interval_ms);
        auto next_snapshot = std::chrono::steady_clock::now();
        while (!stopping_.load(std::memory_order_relaxed)) {
            uint64_t generation;
            {
                std::unique_lock<std::mutex> lock(snapshot_mu_);
                snapshot_cv_.wait(lock, [this] {
                    return stopping_.load(std::memory_order_relaxed) || registered_.load(std::memory_order_acquire);
                });
                if (stopping_.load(std::memory_order_relaxed)) {
                    break;
                }
                snapshot_cv_.wait_until(lock, next_snapshot, [this] {
                    return stopping_.load(std::memory_order_relaxed) || !registered_.load(std::memory_order_acquire)
                           || snapshot_requested_;
                });
                if (stopping_.load(std::memory_order_relaxed)) {
                    break;
                }
                if (!registered_.load(std::memory_order_acquire)) {
                    continue;
                }
                snapshot_requested_ = false;
                generation = registration_generation_.load(std::memory_order_acquire);
            }
            try {
                const auto snapshot = snapshot_provider_();
                SnapshotReport report{generation, snapshot.version, snapshot.block_groups.size(),
                                      buildSnapshotReport(context_, nextTraceId("snapshot"), snapshot)};
                {
                    std::lock_guard<std::mutex> lock(snapshot_mu_);
                    if (!stopping_.load(std::memory_order_relaxed) && registered_.load(std::memory_order_acquire)
                        && generation == registration_generation_.load(std::memory_order_acquire)) {
                        // One bounded handoff slot: a slow upload keeps only the
                        // latest unsent capture. Failed uploads are never requeued.
                        if (snapshot_report_) {
                            RTP_LLM_LOG_WARNING("KVCMPublisher discarding superseded unsent snapshot, version=%lld",
                                                static_cast<long long>(snapshot_report_->version));
                        }
                        snapshot_report_ = std::move(report);
                    }
                }
                snapshot_cv_.notify_all();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("KVCMPublisher snapshot capture failed; discarding this round: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("KVCMPublisher snapshot capture failed; discarding this round");
            }
            // Start the interval after capture/handoff, independently of the ACK.
            next_snapshot = std::chrono::steady_clock::now() + interval;
        }
    }

    void snapshotSendLoop() noexcept {
        pthread_setname_np(pthread_self(), "kvcm_snap_send");
        while (!stopping_.load(std::memory_order_relaxed)) {
            SnapshotReport report;
            {
                std::unique_lock<std::mutex> lock(snapshot_mu_);
                snapshot_cv_.wait(lock, [this] {
                    return stopping_.load(std::memory_order_relaxed) || snapshot_report_.has_value();
                });
                if (stopping_.load(std::memory_order_relaxed)) {
                    break;
                }
                report = std::move(*snapshot_report_);
                snapshot_report_.reset();
            }
            std::lock_guard<std::mutex> send_lock(snapshot_send_mu_);
            if (stopping_.load(std::memory_order_relaxed) || !registered_.load(std::memory_order_acquire)
                || report.generation != registration_generation_.load(std::memory_order_acquire)) {
                continue;
            }
            try {
                std::string response;
                const bool success = snapshot_reporter_->post("/api/reportEvent", report.request, response);
                if (!observeReportResponse(detail::parseKVCMReportFeedback(response), success, report.generation)) {
                    RTP_LLM_LOG_WARNING("KVCMPublisher snapshot report failed; discarding this round, "
                                        "version=%lld keys=%zu",
                                        static_cast<long long>(report.version), report.key_count);
                    continue;
                }
                RTP_LLM_LOG_INFO("KVCMPublisher snapshot committed, instance_id=%s host=%s dp_rank=%d "
                                 "version=%lld keys=%zu",
                                 context_.instance_id.c_str(), context_.host_ip_port.c_str(), context_.dp_rank,
                                 static_cast<long long>(report.version), report.key_count);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("KVCMPublisher snapshot report failed; discarding this round: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("KVCMPublisher snapshot report failed; discarding this round");
            }
        }
    }

    bool reportMutations(const std::vector<KVCacheEvent>& batch) {
        if (batch.empty()) {
            return true;
        }
        const auto generation = registration_generation_.load(std::memory_order_acquire);
        if (!registered_.load(std::memory_order_acquire) || stopping_.load(std::memory_order_relaxed)) {
            return false;
        }
        const auto coalesced = coalesceMutations(batch);
        const auto trace_id  = nextTraceId("mutation");
        std::string response;
        const bool success = reporter_->post(
            "/api/reportEvent", buildMutationReport(context_, trace_id, coalesced), response);
        const auto feedback = detail::parseKVCMReportFeedback(response);
        if (!observeReportResponse(feedback, success, generation, /*normal_ack=*/true)) {
            return false;
        }
        if (feedback.snapshot_required) {
            requestSnapshot(generation);
        }
        return true;
    }

    bool heartbeat() {
        const auto generation = registration_generation_.load(std::memory_order_acquire);
        if (heartbeat_generation_ != generation) {
            heartbeat_generation_ = generation;
            heartbeat_snapshot_signal_delivered_ = false;
        }
        const auto trace_id = nextTraceId("heartbeat");
        std::string response;
        const bool success = reporter_->post(
            "/api/reportEvent", buildControlReport(context_, trace_id, ControlEventType::HEARTBEAT), response);
        const auto feedback = detail::parseKVCMReportFeedback(response);
        if (!observeReportResponse(feedback, success, generation, /*normal_ack=*/true, /*apply_remote_control=*/true)) {
            return false;
        }
        // Deliver a repeated heartbeat advisory once per consecutive streak,
        // as in the Subscriber. A false response or new registration rearms it.
        if (config_.publish_cache_events && feedback.snapshot_required && !heartbeat_snapshot_signal_delivered_) {
            requestSnapshot(generation);
        }
        heartbeat_snapshot_signal_delivered_ = feedback.snapshot_required;
        return true;
    }

    void heartbeatLoop() noexcept {
        pthread_setname_np(pthread_self(), "kvcm_heartbeat");
        const auto heartbeat_interval = std::chrono::milliseconds(std::max(config_.heartbeat_interval_ms, 1));
        size_t consecutive_failures = 0;
        while (!stopping_.load(std::memory_order_relaxed)) {
            queue_.waitForStop(heartbeat_interval);
            if (stopping_.load(std::memory_order_relaxed)) {
                break;
            }
            if (!registered_.load(std::memory_order_relaxed)) {
                continue;
            }
            const auto generation = registration_generation_.load(std::memory_order_acquire);
            if (!heartbeat()) {
                if (generation != registration_generation_.load(std::memory_order_acquire)) {
                    continue;
                }
                if (stopping_.load(std::memory_order_relaxed)) {
                    break;
                }
                // Heartbeats carry no cache mutations. Explicit recovery signals
                // have already been handled by observeReportResponse.
                ++consecutive_failures;
                if ((consecutive_failures & (consecutive_failures - 1)) == 0) {
                    RTP_LLM_LOG_WARNING("KVCMPublisher heartbeat failed %zu consecutive time(s)",
                                        consecutive_failures);
                }
                bool recover_registration = false;
                {
                    std::lock_guard<std::mutex> lock(snapshot_mu_);
                    if (generation != registration_generation_.load(std::memory_order_acquire)) {
                        continue;
                    }
                    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    auto deadline = normal_ack_deadline_ms_.load(std::memory_order_relaxed);
                    // Claim only if no newer ACK has refreshed this deadline.
                    // Zero stays disarmed across registration/snapshot ACKs.
                    if (deadline != 0 && now_ms >= deadline
                        && normal_ack_deadline_ms_.compare_exchange_strong(
                            deadline, 0, std::memory_order_relaxed)) {
                        registered_.store(false, std::memory_order_release);
                        recover_registration = true;
                    }
                }
                if (recover_registration) {
                    queue_.wake();
                    snapshot_cv_.notify_all();
                    RTP_LLM_LOG_WARNING("KVCMPublisher received no heartbeat or mutation ACK for %d ms; "
                                        "requesting one registration recovery",
                                        std::max(config_.ack_recovery_timeout_ms, 1));
                }
            } else if (consecutive_failures != 0) {
                RTP_LLM_LOG_INFO("KVCMPublisher heartbeat recovered after %zu consecutive failure(s)",
                                 consecutive_failures);
                consecutive_failures = 0;
            }
        }
    }

    void waitBeforeRetry(std::chrono::milliseconds delay, bool require_registration = true) {
        queue_.waitForStopOrInterrupt(delay, [this, require_registration] {
            return require_registration && !registered_.load(std::memory_order_relaxed);
        });
    }

    void workerLoop() noexcept {
        pthread_setname_np(pthread_self(), "kvcm_events");
        const auto    base_retry_interval   = std::chrono::milliseconds(std::max(config_.retry_interval_ms, 1));
        const int64_t max_retry_interval_ms = base_retry_interval.count() >= 30000 ?
                                                  base_retry_interval.count() :
                                                  std::min<int64_t>(base_retry_interval.count() * 32, 30000);
        const auto    max_retry_interval    = std::chrono::milliseconds(max_retry_interval_ms);
        auto          retry_interval        = base_retry_interval;
        auto          waitWithBackoff       = [&] {
            // A failure that already invalidated registration must still back off.
            // Only a new invalidation can interrupt a wait started while registered.
            const bool was_registered = registered_.load(std::memory_order_relaxed);
            waitBeforeRetry(retry_interval, was_registered);
            retry_interval = std::min(retry_interval * 2, max_retry_interval);
        };

        size_t consecutive_mutation_failures = 0;
        try {
            while (!stopping_.load(std::memory_order_relaxed)) {
                if (!registered_.load(std::memory_order_acquire)) {
                    // Drop the old registration's backlog. New captures provide
                    // the current cache view; ordinary snapshot rounds never drain deltas.
                    queue_.discardPending();
                    if (!registerNode()) {
                        state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                        RTP_LLM_LOG_WARNING("KVCMPublisher registration failed; retrying in %lld ms",
                                            static_cast<long long>(retry_interval.count()));
                        waitWithBackoff();
                        continue;
                    }
                    retry_interval = base_retry_interval;
                }
                if (!config_.publish_cache_events) {
                    queue_.waitForStopOrInterrupt(
                        std::chrono::milliseconds(std::max(config_.heartbeat_interval_ms, 1)),
                        [this] { return !registered_.load(std::memory_order_acquire); });
                    continue;
                }
                auto batch = queue_.waitPop(config_.report_batch_size,
                                            std::chrono::milliseconds(std::max(config_.flush_interval_ms, 1)));
                if (!reportMutations(batch)) {
                    dropped_count_.fetch_add(batch.size(), std::memory_order_relaxed);
                    report_failed_count_.fetch_add(batch.size(), std::memory_order_relaxed);
                    state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                    ++consecutive_mutation_failures;
                    if ((consecutive_mutation_failures & (consecutive_mutation_failures - 1)) == 0) {
                        RTP_LLM_LOG_WARNING("KVCMPublisher mutation report failed %zu consecutive time(s); "
                                            "dropping %zu event(s)", consecutive_mutation_failures, batch.size());
                    }
                } else if (registered_.load(std::memory_order_acquire)) {
                    if (!batch.empty()) {
                        consecutive_mutation_failures = 0;
                    }
                    state_.store(PublisherState::READY, std::memory_order_relaxed);
                }
            }

        } catch (const std::exception& e) {
            started_.store(false, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(snapshot_mu_);
                stopping_.store(true, std::memory_order_relaxed);
            }
            queue_.stop();
            snapshot_cv_.notify_all();
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("KVCMPublisher worker stopped after exception: %s", e.what());
        } catch (...) {
            started_.store(false, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(snapshot_mu_);
                stopping_.store(true, std::memory_order_relaxed);
            }
            queue_.stop();
            snapshot_cv_.notify_all();
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("KVCMPublisher worker stopped after unknown exception");
        }
    }

private:
    KVCacheEventPublisherConfig           config_;
    KVCacheEventPublisherContext          context_;
    KVCacheSnapshotProvider               snapshot_provider_;
    std::shared_ptr<KVCacheEventReporter> reporter_;
    RemoteCacheControl                   remote_cache_control_;
    std::shared_ptr<KVCacheEventReporter> snapshot_reporter_;
    detail::KVCacheEventQueue             queue_;
    std::thread                           worker_;
    std::thread                           heartbeat_worker_;
    std::thread                           snapshot_worker_;
    std::thread                           snapshot_sender_;
    std::mutex                            lifecycle_mu_;
    std::mutex                            snapshot_mu_;
    std::condition_variable               snapshot_cv_;
    std::mutex                            snapshot_send_mu_;
    std::atomic<bool>                     started_{false};
    std::atomic<bool>                     stopping_{false};
    std::atomic<bool>                     registered_{false};
    std::atomic<bool>                     registered_once_{false};
    bool                                  stopped_permanently_{false};
    std::atomic<PublisherState>           state_{PublisherState::DISABLED};
    std::atomic<uint64_t>                 accepted_count_{0};
    std::atomic<uint64_t>                 dropped_count_{0};
    std::atomic<uint64_t>                 queue_full_count_{0};
    std::atomic<uint64_t>                 report_failed_count_{0};
    std::atomic<uint64_t>                 registration_generation_{0};
    std::atomic<uint64_t>                 next_request_id_{1};
    // Zero means this outage already consumed its registration recovery.
    std::atomic<int64_t>                  normal_ack_deadline_ms_{0};
    bool                                  snapshot_requested_ = false;
    std::optional<SnapshotReport>         snapshot_report_;
    // Owned by the heartbeat worker.
    uint64_t                              heartbeat_generation_ = 0;
    bool                                  heartbeat_snapshot_signal_delivered_ = false;
};

KVCMPublisher::KVCMPublisher(KVCacheEventPublisherConfig           config,
                             KVCacheEventPublisherContext          context,
                             KVCacheSnapshotProvider               snapshot_provider,
                             std::shared_ptr<KVCacheEventReporter> reporter,
                             RemoteCacheControl                   remote_cache_control):
    impl_(std::make_unique<Impl>(
        std::move(config), std::move(context), std::move(snapshot_provider), std::move(reporter),
        std::move(remote_cache_control))) {}

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

}  // namespace rtp_llm
