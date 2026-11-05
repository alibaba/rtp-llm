#include "rtp_llm/cpp/cache/events/KVCMEventReport.h"

#include <map>
#include <tuple>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace rtp_llm::detail {
using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

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
        JsonWriter              writer(buffer);
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

void writeLocationSpec(JsonWriter&                         writer,
                       const KVCacheEventPublisherContext& context,
                       const KVCacheEventSpec&             spec,
                       KVCacheEventMedium                  medium) {
    writer.StartObject();
    writeString(writer, "name", spec.name);
    // Preserve the existing HBM URI. Lower tiers refer to this engine's local
    // cache, not a remote storage allocation or a physical block address.
    writeString(writer,
                "uri",
                medium == KVCacheEventMedium::HBM ?
                    spec.location_uri :
                    "event_report://" + context.host_ip_port + "/" + cacheEventMediumName(medium));
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
            writeString(writer, "medium", cacheEventMediumName(event.medium));
            writer.Key("specs");
            writer.StartArray();
            writeLocationSpec(writer, context, spec, event.medium);
            writer.EndArray();
            writer.EndObject();
        } else {
            writeString(writer, "event_type", "EVENT_BLOCK_DELETE");
            writer.Key("block_delete");
            writer.StartObject();
            writeString(writer, "block_key", block_key);
            writeString(writer, "medium", cacheEventMediumName(event.medium));
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
    // request. The final transition for each (key, medium, group) is therefore the only one
    // this unsent batch may carry without inverting DELETE -> ADD.
    std::vector<KVCacheEvent>                                         coalesced;
    std::map<std::tuple<int64_t, KVCacheEventMedium, size_t>, size_t> component_to_index;
    coalesced.reserve(events.size());
    for (const auto& event : events) {
        const auto [it, inserted] = component_to_index.emplace(
            std::make_tuple(event.block_key, event.medium, event.group_index), coalesced.size());
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
                               bool                                publish_cache_events) {
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
                for (const auto medium : context.enabled_mediums) {
                    writer.String(cacheEventMediumName(medium));
                }
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
    // One snapshot replaces ALL media for this reporter. Never split it into
    // per-medium requests, including when one of the tiers is empty.
    for (const auto medium : kKVCacheEventMediums) {
        for (const auto& [block_key_value, group_indices] : snapshot.groups(medium)) {
            writer.StartObject();
            writeString(writer, "block_key", std::to_string(block_key_value));
            writeString(writer, "medium", cacheEventMediumName(medium));
            writer.Key("specs");
            writer.StartArray();
            for (const auto group_index : group_indices) {
                writeLocationSpec(writer, context, context.specs.at(group_index), medium);
            }
            writer.EndArray();
            writer.EndObject();
        }
    }
    writer.EndArray();
    writer.EndObject();
    writer.EndObject();
    writeReportFooter(writer);
    return buffer.GetString();
}

}  // namespace rtp_llm::detail
