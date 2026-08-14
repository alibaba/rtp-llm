#include "rtp_llm/cpp/cache/events/KVCMRequestBuilder.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <rapidjson/writer.h>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace rtp_llm::detail {
namespace {

class BoundedJsonBuffer {
public:
    using Ch = char;

    explicit BoundedJsonBuffer(size_t max_bytes, const std::atomic<bool>* cancelled = nullptr):
        max_bytes_(max_bytes), cancelled_(cancelled) {
        value_.reserve(std::min<size_t>(max_bytes_, 4096));
    }

    void Put(char value) {
        // Checking once per small chunk keeps cancellation prompt without
        // adding an atomic load for every byte emitted by RapidJSON.
        if ((value_.size() & 4095) == 0 && cancelled_ && cancelled_->load(std::memory_order_acquire)) {
            throw SnapshotBuildCancelled{};
        }
        if (value_.size() >= max_bytes_) {
            throw JsonPayloadLimitExceeded{};
        }
        value_.push_back(value);
    }

    void Flush() noexcept {}

    std::string takeString() noexcept {
        return std::move(value_);
    }

private:
    size_t                   max_bytes_;
    const std::atomic<bool>* cancelled_;
    std::string              value_;
};

rapidjson::SizeType checkedJsonStringSize(std::string_view value) {
    if (value.size() > static_cast<size_t>(std::numeric_limits<rapidjson::SizeType>::max())) {
        throw JsonPayloadLimitExceeded{};
    }
    return static_cast<rapidjson::SizeType>(value.size());
}

template<typename Writer>
void writeString(Writer& writer, const char* key, std::string_view value) {
    writer.Key(key);
    writer.String(value.data(), checkedJsonStringSize(value));
}

template<typename Writer>
void writeStringValue(Writer& writer, std::string_view value) {
    writer.String(value.data(), checkedJsonStringSize(value));
}

template<typename Writer>
void writeInt64String(Writer& writer, const char* key, int64_t value) {
    char       buffer[32];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    writer.Key(key);
    writer.String(buffer, static_cast<rapidjson::SizeType>(result.ptr - buffer));
}

template<typename Writer>
void writeReportHeader(Writer& writer, const KVCacheEventPublisherContext& context, const std::string& trace_id) {
    writer.StartObject();
    writeString(writer, "trace_id", trace_id);
    writeString(writer, "instance_id", context.instance_id);
    writeString(writer, "host_ip_port", context.host_ip_port);
    writer.Key("events");
    writer.StartArray();
}

template<typename Writer>
void writeReportFooter(Writer& writer) {
    writer.EndArray();
    writeString(writer, "storage_type", "ST_EVENT_REPORT_L1P5");
    writer.EndObject();
}

template<typename Writer>
void writeLocationSpecs(Writer& writer, const KVCacheEventPublisherContext& context) {
    writer.Key("specs");
    writer.StartArray();
    writer.StartObject();
    writeString(writer, "name", context.spec_name);
    writeString(writer, "uri", context.location_uri);
    writer.EndObject();
    writer.EndArray();
}

}  // namespace

std::string buildRegisterInstanceRequest(const KVCacheEventPublisherContext& context,
                                         const std::string&                  trace_id,
                                         size_t                              max_bytes) {
    BoundedJsonBuffer                    buffer(max_bytes);
    rapidjson::Writer<BoundedJsonBuffer> writer(buffer);
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
    writeStringValue(writer, context.spec_name);
    writer.EndArray();
    writer.EndObject();
    writer.EndArray();
    writer.EndObject();
    return buffer.takeString();
}

std::string buildMutationReport(const KVCacheEventPublisherContext& context,
                                const std::string&                  trace_id,
                                const std::vector<KVCacheEvent>&    events,
                                size_t                              max_bytes) {
    BoundedJsonBuffer                    buffer(max_bytes);
    rapidjson::Writer<BoundedJsonBuffer> writer(buffer);
    writeReportHeader(writer, context, trace_id);
    for (const auto& event : events) {
        writer.StartObject();
        if (event.type == KVCacheEventType::BLOCK_ADD) {
            writeString(writer, "event_type", "EVENT_BLOCK_ADD");
            writer.Key("block_add");
            writer.StartObject();
            writeInt64String(writer, "block_key", event.block_key);
            writeString(writer, "medium", "hbm");
            writeLocationSpecs(writer, context);
            writer.EndObject();
        } else {
            writeString(writer, "event_type", "EVENT_BLOCK_DELETE");
            writer.Key("block_delete");
            writer.StartObject();
            writeInt64String(writer, "block_key", event.block_key);
            writeString(writer, "medium", "hbm");
            writer.Key("spec_names");
            writer.StartArray();
            writeStringValue(writer, context.spec_name);
            writer.EndArray();
            writer.EndObject();
        }
        writer.EndObject();
    }
    writeReportFooter(writer);
    return buffer.takeString();
}

std::vector<KVCacheEvent> coalesceMutations(const std::vector<KVCacheEvent>& events) {
    // Only the final transition for one key changes its logical state. Fold
    // locally to bound payload size and remain correct with both current KVCM
    // (which preserves the final per-spec operation) and older deployments
    // that grouped ADD and DELETE writes inside one request.
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

std::string buildControlReport(const KVCacheEventPublisherContext& context,
                               const std::string&                  trace_id,
                               ControlEventType                    type,
                               size_t                              max_bytes) {
    BoundedJsonBuffer                    buffer(max_bytes);
    rapidjson::Writer<BoundedJsonBuffer> writer(buffer);
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
    return buffer.takeString();
}

std::string buildSnapshotReport(const KVCacheEventPublisherContext& context,
                                const std::string&                  trace_id,
                                const KVCacheSnapshot&              snapshot,
                                size_t                              max_bytes,
                                const std::atomic<bool>*            cancelled) {
    BoundedJsonBuffer                    buffer(max_bytes, cancelled);
    rapidjson::Writer<BoundedJsonBuffer> writer(buffer);
    writeReportHeader(writer, context, trace_id);
    writer.StartObject();
    writeString(writer, "event_type", "EVENT_BLOCK_SNAPSHOT");
    writer.Key("block_snapshot");
    writer.StartObject();
    writer.Key("blocks");
    writer.StartArray();
    for (const auto block_key_value : snapshot.block_keys) {
        writer.StartObject();
        writeInt64String(writer, "block_key", block_key_value);
        writeString(writer, "medium", "hbm");
        writeLocationSpecs(writer, context);
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
    writer.EndObject();
    writeReportFooter(writer);
    return buffer.takeString();
}

}  // namespace rtp_llm::detail
