#include "rtp_llm/cpp/utils/AsyncProfileTrace.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace rtp_llm {
namespace {
// Wire-compatible subset of Perfetto's Trace/TracePacket/TrackDescriptor/
// TrackEvent/DebugAnnotation protos. No SDK, global tracing state or thread
// descriptors are installed. Field numbers are from protos/perfetto/trace/.
std::string varint(uint64_t value) {
    std::string out;
    while (value >= 128) {
        out += static_cast<char>((value & 127) | 128);
        value >>= 7;
    }
    return out + static_cast<char>(value);
}
std::string number(unsigned field, uint64_t value) {
    return varint(field << 3) + varint(value);
}
std::string bytes(unsigned field, const std::string& value) {
    return varint((field << 3) | 2) + varint(value.size()) + value;
}
std::string flow(uint64_t id, bool terminate = false) {
    auto out = varint(((terminate ? 48 : 47) << 3) | 1);
    for (int byte = 0; byte < 8; ++byte) {
        out += static_cast<char>(id & 255);
        id >>= 8;
    }
    return out;
}
std::string annotation(const char* name, const std::string& value) {
    return bytes(4, bytes(10, name) + bytes(6, value));
}
const char* kindName(AsyncProfileKind kind) {
    switch (kind) {
        case AsyncProfileKind::Execution:
            return "execution";
        case AsyncProfileKind::Lifecycle:
            return "lifetime";
        case AsyncProfileKind::Wait:
            return "wait";
        default:
            return "derived";
    }
}
struct Timeline {
    uint64_t    next_uuid{0x5254500000000000ULL};
    std::string output;
    uint64_t    track(const std::string& name, uint64_t parent = 0) {
        const auto uuid = ++next_uuid;
        // Native sibling merging gives the UI one named track, while each
        // overlap lane retains a valid BEGIN/END stack and exact timestamps.
        auto descriptor = number(1, uuid) + bytes(2, name) + number(15, 1);
        if (parent)
            descriptor += number(5, parent);
        output += bytes(1, bytes(60, descriptor));
        return uuid;
    }
    void event(int64_t            timestamp,
               uint64_t           track,
               unsigned           type,
               const std::string& name     = {},
               const std::string& category = {},
               const std::string& args     = {}) {
        if (timestamp < 0)
            throw std::runtime_error("logical event precedes Kineto time base");
        auto event = number(9, type) + number(11, track);
        if (!name.empty())
            event += bytes(23, name);
        if (!category.empty())
            event += bytes(22, category);
        // Default BOOTTIME is deliberately used for both JSON and proto import:
        // these timestamps are already in Kineto's exported relative time base.
        output += bytes(1, number(8, timestamp) + number(10, 1) + bytes(11, event + args));
    }
};
std::string eventArgs(const AsyncProfileEvent& e) {
    const auto& r = e.record;
    return annotation("request_id", std::to_string(r.request_id))
           + annotation("client_request_id", std::to_string(r.client_request_id))
           + annotation("workflow_id", std::to_string(r.workflow_id))
           + annotation("transfer_id", std::to_string(r.transfer_id))
           + annotation("parent_transfer_id", std::to_string(r.parent_transfer_id))
           + annotation("related_workflow_id", std::to_string(r.related_workflow_id))
           + annotation("session_id", std::to_string(e.session_id)) + annotation("span_id", std::to_string(e.span_id))
           + annotation("kind", kindName(detail::exportedAsyncProfileKind(e)))
           + annotation("declared_kind", kindName(r.kind)) + annotation("original_name", e.name)
           + annotation("stage", r.stage) + annotation("path", r.path) + annotation("operation", r.operation)
           + annotation("scope", r.scope) + annotation("rank", std::to_string(r.rank))
           + annotation("batch_begin", std::to_string(r.begin)) + annotation("batch_end", std::to_string(r.end))
           + annotation("origin_pid", std::to_string(e.pid)) + annotation("start_tid", std::to_string(r.start_tid))
           + annotation("end_tid", std::to_string(e.end_tid))
           + annotation("thread_mismatch", e.thread_mismatch ? "true" : "false")
           + annotation("incomplete", e.start_clipped || e.end_clipped ? "true" : "false")
           + annotation("start_clipped", e.start_clipped ? "true" : "false")
           + annotation("end_clipped", e.end_clipped ? "true" : "false");
}

// ZIP STORE keeps the native JSON bytes intact, needs no compression dependency
// and streams the large payload instead of copying it into an archive buffer.
uint32_t crc32(const std::string& data) {
    static const auto table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t n = 0; n < values.size(); ++n) {
            auto c = n;
            for (int bit = 0; bit < 8; ++bit)
                c = (c >> 1) ^ ((c & 1) ? 0xedb88320U : 0);
            values[n] = c;
        }
        return values;
    }();
    uint32_t crc = 0xffffffffU;
    for (unsigned char c : data)
        crc = table[(crc ^ c) & 255] ^ (crc >> 8);
    return ~crc;
}
void le(std::ostream& out, uint32_t value, unsigned size) {
    while (size--) {
        out.put(static_cast<char>(value & 255));
        value >>= 8;
    }
}
void zip(std::ostream& out, const std::string& json, const std::string& proto) {
    std::string        names[] = {"timeline.json", "requests.pftrace"};
    const std::string* data[]  = {&json, &proto};
    uint32_t           offsets[2], checksums[2];
    // An empty protobuf member is not a recognizable Perfetto trace. A capture
    // containing only request-owned work now needs just the physical JSON.
    const int count = proto.empty() ? 1 : 2;
    // Reject ZIP32 overflow before writing anything. The original JSON remains.
    if (json.size() + proto.size() + 1024ULL >= std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("request timeline exceeds ZIP32 limit");
    for (int i = 0; i < count; ++i) {
        offsets[i]   = static_cast<uint32_t>(out.tellp());
        checksums[i] = crc32(*data[i]);
        le(out, 0x04034b50, 4);
        le(out, 20, 2);
        le(out, 0, 2);
        le(out, 0, 2);
        le(out, 0, 2);
        le(out, 0, 2);
        le(out, checksums[i], 4);
        le(out, data[i]->size(), 4);
        le(out, data[i]->size(), 4);
        le(out, names[i].size(), 2);
        le(out, 0, 2);
        out << names[i] << *data[i];
    }
    const auto central = static_cast<uint32_t>(out.tellp());
    for (int i = 0; i < count; ++i) {
        le(out, 0x02014b50, 4);
        le(out, 20, 2);
        le(out, 20, 2);
        le(out, 0, 2);
        le(out, 0, 2);
        le(out, 0, 2);
        le(out, 0, 2);
        le(out, checksums[i], 4);
        le(out, data[i]->size(), 4);
        le(out, data[i]->size(), 4);
        le(out, names[i].size(), 2);
        le(out, 0, 2);
        le(out, 0, 2);
        le(out, 0, 2);
        le(out, 0, 2);
        le(out, 0, 4);
        le(out, offsets[i], 4);
        out << names[i];
    }
    const auto central_size = static_cast<uint32_t>(out.tellp()) - central;
    le(out, 0x06054b50, 4);
    le(out, 0, 2);
    le(out, 0, 2);
    le(out, count, 2);
    le(out, count, 2);
    le(out, central_size, 4);
    le(out, central, 4);
    le(out, 0, 2);
}
}  // namespace

std::string asyncProfileWorkflowTrace(const std::string& native, const std::vector<AsyncProfileEvent>& input) {
    auto       events    = detail::resolveAsyncProfileLinks(input);
    const auto base      = detail::asyncProfileBaseTimeNs(native);
    auto       timestamp = [base](int64_t us) {
        if (us < 0 || us > std::numeric_limits<int64_t>::max() / 1000)
            throw std::runtime_error("invalid request timestamp");
        return us * 1000 - base;
    };
    Timeline timeline;
    using WorkflowKey = std::tuple<int64_t, int64_t, uint64_t>;
    std::map<WorkflowKey, uint64_t>                              workflow_tracks;
    std::map<std::string, std::vector<const AsyncProfileEvent*>> slices;
    for (const auto& e : events) {
        const auto& r = e.record;
        // Request-owned logical projections are intentionally hidden. Their
        // real execution events remain on OS threads in timeline.json.
        if (r.stage == "task_link" || r.request_id >= 0 || r.path == "request")
            continue;
        const auto name = r.path == "profiler" ? "KV diagnostics" : "KV background";
        slices[name].push_back(&e);
    }
    // Only fixed shared track names are exposed. Crossing spans use sibling
    // lanes with the SAME name; Perfetto merges them into one visual track.
    // No request/workflow/stage titles or per-kind tracks are emitted.
    for (auto& group : slices) {
        auto& spans = group.second;
        std::stable_sort(spans.begin(), spans.end(), [](const auto* a, const auto* b) {
            return a->start_us != b->start_us ? a->start_us < b->start_us : a->end_us > b->end_us;
        });
        struct Lane {
            uint64_t                              track;
            std::vector<const AsyncProfileEvent*> stack;
        };
        std::vector<Lane> lanes{{timeline.track(group.first), {}}};
        auto              close = [&](Lane& lane) {
            timeline.event(timestamp(lane.stack.back()->end_us), lane.track, 2);
            lane.stack.pop_back();
        };
        for (const auto* e : spans) {
            if (e->end_us < e->start_us)
                throw std::runtime_error("negative logical duration");
            size_t index = 0;
            for (; index < lanes.size(); ++index) {
                auto& lane = lanes[index];
                while (!lane.stack.empty() && lane.stack.back()->end_us <= e->start_us)
                    close(lane);
                if (lane.stack.empty())
                    break;
                const auto& parent = *lane.stack.back();
                // Temporal containment does not make another request a child.
                if (e->end_us <= parent.end_us && e->pid == parent.pid
                    && e->record.request_id == parent.record.request_id
                    && e->record.workflow_id == parent.record.workflow_id)
                    break;
            }
            if (index == lanes.size())
                lanes.push_back({timeline.track(group.first), {}});
            auto&      lane = lanes[index];
            const auto kind = detail::exportedAsyncProfileKind(*e);
            auto       name = e->record.stage + " [" + kindName(kind) + "]";
            name += " / " + e->record.path;
            if (e->record.workflow_id)
                workflow_tracks.try_emplace(WorkflowKey{e->pid, e->record.request_id, e->record.workflow_id},
                                            lane.track);
            timeline.event(timestamp(e->start_us),
                           lane.track,
                           e->end_us == e->start_us ? 3 : 1,
                           name,
                           std::string("kv.") + kindName(kind),
                           eventArgs(*e));
            if (e->end_us != e->start_us)
                lane.stack.push_back(e);
        }
        for (auto& lane : lanes)
            while (!lane.stack.empty())
                close(lane);
    }
    // Shared work is executed once under its owner request. Dependency markers
    // connect the waiter to that owner without duplicating IO or inventing tids.
    uint64_t next_flow = 0x5254501000000000ULL;
    for (const auto& e : events) {
        const auto& r = e.record;
        if (!r.related_workflow_id)
            continue;
        const auto waiter = workflow_tracks.find({e.pid, r.request_id, r.workflow_id});
        auto       owner  = std::find_if(workflow_tracks.begin(), workflow_tracks.end(), [&](const auto& item) {
            return std::get<0>(item.first) == e.pid && std::get<2>(item.first) == r.related_workflow_id;
        });
        if (waiter == workflow_tracks.end() || owner == workflow_tracks.end())
            continue;
        const auto id = ++next_flow;
        timeline.event(timestamp(e.start_us),
                       owner->second,
                       3,
                       "shared load dependency",
                       "kv.dependency",
                       eventArgs(e) + flow(id));
        timeline.event(timestamp(e.start_us),
                       waiter->second,
                       3,
                       "await shared workflow",
                       "kv.dependency",
                       eventArgs(e) + flow(id, true));
    }
    return timeline.output;
}

void writeAsyncProfileBundle(const std::string&                    path,
                             const std::string&                    native,
                             const std::vector<AsyncProfileEvent>& events) {
    // Physical slices are retained exactly once on real threads. Only the new
    // logical view mirrors their span IDs, explicitly labelled execution.
    std::vector<AsyncProfileEvent> execution;
    for (const auto& e : events)
        if (detail::exportedAsyncProfileKind(e) == AsyncProfileKind::Execution)
            execution.push_back(e);
    const auto        json    = mergeAsyncProfileTrace(native, execution);
    const auto        proto   = asyncProfileWorkflowTrace(native, events);
    auto              pattern = path + ".XXXXXX";
    std::vector<char> temporary(pattern.begin(), pattern.end());
    temporary.push_back('\0');
    const int fd = mkstemp(temporary.data());
    if (fd < 0)
        throw std::runtime_error("cannot create request timeline bundle");
    close(fd);  // mkstemp retains private permissions (0600).
    try {
        std::ofstream out(temporary.data(), std::ios::binary | std::ios::trunc);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        zip(out, json, proto);
        out.close();
        if (std::rename(temporary.data(), path.c_str()) != 0)
            throw std::runtime_error("cannot commit request timeline bundle");
    } catch (...) {
        std::remove(temporary.data());
        throw;
    }
}
}  // namespace rtp_llm
