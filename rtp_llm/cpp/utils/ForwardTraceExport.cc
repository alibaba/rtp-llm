#include "rtp_llm/cpp/utils/ForwardTrace.h"

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>

namespace rtp_llm {
namespace {
using autil::legacy::Any;
using autil::legacy::AnyCast;
using autil::legacy::json::JsonArray;
using autil::legacy::json::JsonMap;

struct Span { double begin; double end; };
using SpanGroups = std::map<std::string, std::vector<Span>>;

void addCpuSpan(const JsonMap& event, bool chunk, SpanGroups& spans) {
    auto cat = event.find("cat");
    const auto* category = cat == event.end() ? nullptr : AnyCast<std::string>(&cat->second);
    if (!category || (*category != "cpu_op" && *category != "user_annotation")) return;
    for (const auto* key : {"pid", "tid", "ts", "dur"}) {
        if (!event.count(key)) return;
    }
    const auto start = autil::legacy::json::JsonNumberCast<double>(event.at("ts"));
    const auto duration = autil::legacy::json::JsonNumberCast<double>(event.at("dur"));
    auto key = autil::legacy::json::ToString(event.at("pid"), true) + ":"
        + autil::legacy::json::ToString(event.at("tid"), true) + (chunk ? ":chunk" : ":model");
    spans[key].push_back({start, start + duration});
}

int64_t uncoveredForwards(const SpanGroups& forwards, SpanGroups& wrappers) {
    int64_t missing = 0;
    for (auto& group : wrappers) {
        std::sort(group.second.begin(), group.second.end(), [](const Span& a, const Span& b) {
            return a.begin < b.begin;
        });
    }
    for (const auto& group : forwards) {
        auto& candidates = wrappers[group.first];
        for (const auto& forward : group.second) {
            auto it = std::upper_bound(candidates.begin(), candidates.end(), forward.begin,
                [](double value, const Span& span) { return value < span.begin; });
            if (it == candidates.begin() || (--it)->end < forward.end) ++missing;
        }
    }
    return missing;
}

JsonArray array(const std::vector<int64_t>& values) {
    JsonArray result;
    result.reserve(values.size());
    for (auto value : values) result.emplace_back(value);
    return result;
}


JsonMap metadata(ForwardTraceRecord& record, bool& valid) {
    JsonMap result;
    for (const auto& item : record.integers) result[item.first] = item.second;
    for (const auto& item : record.strings) result[item.first] = item.second;
    result["forward_id"] = record.id;
    result["parent_forward_id"] = record.parent_id;
    JsonMap raw;
    valid = record.strings["status"] == "ok";
    for (const auto& item : record.arrays) {
        if (item.second.valid) raw[item.first] = array(item.second.host);
        else {
            raw[item.first] = Any();
            valid = false;
        }
    }
    result["recorded_inputs"] = raw;
    std::vector<int64_t> q, prefix;
    if (normalizeForwardTraceLengths(record, q, prefix)) {
        std::vector<int64_t> kv(q.size());
        for (size_t i = 0; i < q.size(); ++i) kv[i] = q[i] + prefix[i];
        result["q_lens"] = array(q);
        result["prefix_lens"] = array(prefix);
        result["kv_lens"] = array(kv);
        result["total_q_tokens"] = std::accumulate(q.begin(), q.end(), int64_t{0});
        result["logical_sequences"] = static_cast<int64_t>(q.size());
    } else {
        valid = false;
    }
    result["lengths_complete"] = valid;
    return result;
}
}  // namespace

void enrichForwardTrace(const std::string& path, ForwardTraceSession& session) {
    session.materialize();
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read profiler trace: " + path);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();
    Any document;
    autil::legacy::json::ParseJson(text, document);
    text.clear();
    text.shrink_to_fit();
    auto* root = AnyCast<JsonMap>(&document);
    if (!root) throw std::runtime_error("profiler trace must be a JSON object");
    auto events_it = root->find("traceEvents");
    auto* events = events_it == root->end() ? nullptr : AnyCast<JsonArray>(&events_it->second);
    if (!events) throw std::runtime_error("profiler trace lacks traceEvents");

    std::map<std::string, JsonMap> by_name;
    JsonArray manifest;
    int64_t incomplete = 0;
    for (auto& record : session.records) {
        if (record->strings["kind"] == "chunk" && record->parent_id > 0) {
            const auto& parent = *session.records.at(record->parent_id - 1);
            auto count = parent.integers.find("request_count");
            auto logical = parent.integers.find("logical_sequences");
            if (count != parent.integers.end() && logical != parent.integers.end()
                && count->second == logical->second) {
                record->integers["request_count"] = record->arrays.at("q_lens").host.size();
            }
        }
        bool valid;
        auto data = metadata(*record, valid);
        if (!valid) ++incomplete;
        by_name.emplace("RTP::model_forward(id=" + std::to_string(record->id) + ")", data);
        manifest.emplace_back(std::move(data));
    }
    std::set<std::string> found;
    SpanGroups existing_forwards, annotated_forwards;
    int64_t unknown_events = 0;
    for (auto& event : *events) {
        auto* map = AnyCast<JsonMap>(&event);
        if (!map) continue;
        auto name_it = map->find("name");
        const auto* name = name_it == map->end() ? nullptr : AnyCast<std::string>(&name_it->second);
        if (!name) continue;
        if (*name == "py_model.forward") addCpuSpan(*map, false, existing_forwards);
        if (name->find("RTP::kimi_k3.chunk_prefill.target_forward(") == 0) {
            addCpuSpan(*map, true, existing_forwards);
        }
        if (name->find("RTP::model_forward(id=") != 0) continue;
        auto data = by_name.find(*name);
        if (data == by_name.end()) {
            ++unknown_events;
            continue;
        }
        found.insert(*name);
        const auto kind = data->second.find("kind");
        const auto* kind_name = kind == data->second.end() ? nullptr : AnyCast<std::string>(&kind->second);
        addCpuSpan(*map, kind_name && *kind_name == "chunk", annotated_forwards);
        auto& args_any = (*map)["args"];
        if (!AnyCast<JsonMap>(&args_any)) args_any = JsonMap{};
        (*AnyCast<JsonMap>(&args_any))["rtp_forward"] = data->second;
        // Format only in the save worker, never on the model execution path.
        // Keep unbounded per-sequence arrays in args instead of a truncated name.
        auto integer = [&](const char* key) {
            const auto it = data->second.find(key);
            const auto* value = it == data->second.end() ? nullptr : AnyCast<int64_t>(&it->second);
            return value && *value >= 0 ? std::to_string(*value) : std::string("unknown");
        };
        std::string label = name->substr(0, name->size() - 1);
        const auto phase_it = data->second.find("phase");
        if (phase_it != data->second.end()) {
            if (const auto* phase = AnyCast<std::string>(&phase_it->second)) label += ",phase=" + *phase;
        }
        label += ",requests=" + integer("request_count");
        label += ",sequences=" + integer("logical_sequences");
        label += ",tokens=" + integer("total_q_tokens") + ")";
        (*map)["name"] = std::move(label);
    }
    const int64_t missing_events = by_name.size() - found.size();
    const int64_t unannotated = uncoveredForwards(existing_forwards, annotated_forwards);
    JsonMap envelope;
    envelope["schema_version"] = int64_t{1};
    envelope["kv_length_convention"] = std::string("prefix_plus_current_q");
    envelope["replay_scope"] = std::string("shape_only; token/KV contents and routing are not captured");
    envelope["complete"] = incomplete == 0 && session.dropped == 0 && unknown_events == 0
        && missing_events == 0 && unannotated == 0;
    envelope["unannotated_forward_events"] = unannotated;
    envelope["incomplete_records"] = incomplete;
    envelope["dropped_records_or_arrays"] = session.dropped;
    envelope["missing_scope_events"] = missing_events;
    envelope["unmatched_scope_events"] = unknown_events;
    envelope["forwards"] = manifest;
    (*root)["rtp_forward_metadata"] = envelope;

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot enrich profiler trace: " + path);
    output << autil::legacy::json::ToString(document, true);
    output.close();
    if (!output) throw std::runtime_error("cannot finish profiler trace: " + path);
}
}  // namespace rtp_llm
