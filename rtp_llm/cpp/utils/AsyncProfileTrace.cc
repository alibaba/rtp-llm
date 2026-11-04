#include "rtp_llm/cpp/utils/AsyncProfileTrace.h"

#include <cstdio>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <set>
#include <map>
#include <string_view>
#include <sys/stat.h>

namespace rtp_llm {
namespace detail {

std::ostringstream asyncProfileOutputStream() {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    return stream;
}

}  // namespace detail

namespace {

bool whitespace(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

size_t skipSpace(std::string_view s, size_t pos) {
    while (pos < s.size() && whitespace(s[pos]))
        ++pos;
    return pos;
}

// A lexical scan locates insertion boundaries; it never parses/re-serializes
// native events (which can contain large IDs, fractional timestamps and flows).
size_t stringEnd(std::string_view s, size_t begin) {
    for (size_t pos = begin + 1; pos < s.size(); ++pos) {
        if (s[pos] == '\\')
            ++pos;
        else if (s[pos] == '"')
            return pos;
    }
    throw std::runtime_error("unterminated string in Kineto trace");
}

size_t rootValue(std::string_view s, const std::string& key) {
    int depth = 0;
    for (size_t pos = 0; pos < s.size(); ++pos) {
        if (s[pos] == '"') {
            auto end   = stringEnd(s, pos);
            auto colon = skipSpace(s, end + 1);
            if (depth == 1 && s.compare(pos + 1, end - pos - 1, key) == 0 && colon < s.size() && s[colon] == ':')
                return skipSpace(s, colon + 1);
            pos = end;
        } else if (s[pos] == '{' || s[pos] == '[') {
            ++depth;
        } else if (s[pos] == '}' || s[pos] == ']') {
            --depth;
        }
    }
    return std::string::npos;
}

size_t arrayEnd(const std::string& s, size_t begin) {
    if (begin >= s.size() || s[begin] != '[')
        throw std::runtime_error("missing Kineto traceEvents array");
    int depth = 0;
    for (size_t pos = begin; pos < s.size(); ++pos) {
        if (s[pos] == '"') {
            pos = stringEnd(s, pos);
        } else if (s[pos] == '[') {
            ++depth;
        } else if (s[pos] == ']' && --depth == 0) {
            return pos;
        }
    }
    throw std::runtime_error("unterminated Kineto traceEvents array");
}

size_t objectEnd(std::string_view s, size_t begin) {
    if (begin >= s.size() || s[begin] != '{')
        throw std::runtime_error("unsupported Kineto event layout");
    int depth = 0;
    for (size_t pos = begin; pos < s.size(); ++pos) {
        if (s[pos] == '"')
            pos = stringEnd(s, pos);
        else if (s[pos] == '{')
            ++depth;
        else if (s[pos] == '}' && --depth == 0)
            return pos;
    }
    throw std::runtime_error("unterminated Kineto event");
}

bool stringFieldIs(std::string_view s, const std::string& key, std::string_view value) {
    auto pos = rootValue(s, key);
    return pos < s.size() && s[pos] == '"' && stringEnd(s, pos) == pos + value.size() + 1
           && s.substr(pos + 1, value.size()) == value;
}

int64_t threadIdentityField(std::string_view s, const std::string& key) {
    auto pos = rootValue(s, key);
    if (pos >= s.size())
        return 0;
    const bool quoted = s[pos] == '"';
    pos += quoted;
    int64_t value  = 0;
    auto    parsed = std::from_chars(s.data() + pos, s.data() + s.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr == s.data() + s.size())
        return 0;
    if (quoted ? *parsed.ptr != '"' : (!whitespace(*parsed.ptr) && *parsed.ptr != ',' && *parsed.ptr != '}'))
        return 0;
    return value;
}

using ThreadIdentity = std::pair<int64_t, int64_t>;

std::set<ThreadIdentity> namedNativeThreads(const std::string& trace, size_t begin, size_t end) {
    std::set<ThreadIdentity> names;
    for (auto pos = skipSpace(trace, begin + 1); pos < end;) {
        auto close = objectEnd(trace, pos);
        auto event = std::string_view(trace).substr(pos, close - pos + 1);
        // Most native events need only the lexical scan, never JSON allocation.
        if (event.find("thread_name") != std::string_view::npos && stringFieldIs(event, "ph", "M")
            && stringFieldIs(event, "name", "thread_name")) {
            auto args_pos = rootValue(event, "args");
            if (args_pos < event.size() && event[args_pos] == '{') {
                auto args = event.substr(args_pos, objectEnd(event, args_pos) - args_pos + 1);
                auto name = rootValue(args, "name");
                if (name < args.size() && args[name] == '"' && stringEnd(args, name) > name + 1)
                    names.emplace(threadIdentityField(event, "pid"), threadIdentityField(event, "tid"));
            }
        }
        pos = skipSpace(trace, close + 1);
        if (pos < end && trace[pos] == ',')
            pos = skipSpace(trace, pos + 1);
    }
    return names;
}

int64_t baseTimeNs(const std::string& s) {
    auto pos = rootValue(s, "baseTimeNanoseconds");
    if (pos == std::string::npos)
        return 0;  // Older Kineto versions export absolute microseconds.
    const auto begin = pos;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9')
        ++pos;
    auto end = skipSpace(s, pos);
    if (pos == begin || end >= s.size() || (s[end] != ',' && s[end] != '}'))
        throw std::runtime_error("unsupported Kineto baseTimeNanoseconds");
    return std::stoll(s.substr(begin, pos - begin));
}

std::string quoted(const std::string& value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string    out   = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + '"';
}

const char* kindName(AsyncProfileKind kind) {
    switch (kind) {
        case AsyncProfileKind::Execution:
            return "execution";
        case AsyncProfileKind::Lifecycle:
            return "lifecycle";
        case AsyncProfileKind::Wait:
            return "wait";
        case AsyncProfileKind::Derived:
            return "derived";
    }
    return "derived";
}

AsyncProfileKind exportedKind(const AsyncProfileEvent& e) {
    if (e.record.kind == AsyncProfileKind::Execution
        && (e.record.start_tid <= 0 || e.thread_mismatch || (e.end_tid != 0 && e.record.start_tid != e.end_tid)))
        return AsyncProfileKind::Lifecycle;
    return e.record.kind;
}

std::string logicalTrackName(const AsyncProfileMetadata& r) {
    if (r.path == "profiler")
        return "KV diagnostics";
    return "KV background";
}

std::string timeUs(int64_t ns) {
    if (ns < 0)
        throw std::runtime_error("async event precedes Kineto time base");
    auto fraction = std::to_string(ns % 1000);
    return std::to_string(ns / 1000) + "." + std::string(3 - fraction.size(), '0') + fraction;
}

int64_t relativeNs(int64_t us, int64_t base) {
    if (us < 0 || us > std::numeric_limits<int64_t>::max() / 1000)
        throw std::runtime_error("invalid async timestamp");
    return us * 1000 - base;
}

void writeEvent(std::ostream& out, const AsyncProfileEvent& e, int64_t base, bool& first) {
    const auto& r               = e.record;
    const auto  kind            = exportedKind(e);
    const bool  thread_mismatch = e.thread_mismatch || (e.end_tid != 0 && r.start_tid != e.end_tid);
    const bool  execution       = kind == AsyncProfileKind::Execution;
    const auto  start           = relativeNs(e.start_us, base);
    const auto  end             = relativeNs(e.end_us, base);
    if (end < start)
        throw std::runtime_error("negative async event duration");
    auto args = detail::asyncProfileOutputStream();
    args << "{\"Backend\":\"RTP-LLM cache transfer\",\"kind\":" << quoted(kindName(kind))
         << ",\"declared_kind\":" << quoted(kindName(r.kind)) << ",\"original_name\":" << quoted(e.name)
         << ",\"request_id\":" << quoted(std::to_string(r.request_id))
         << ",\"transfer_id\":" << quoted(std::to_string(r.transfer_id))
         << ",\"workflow_id\":" << quoted(std::to_string(r.workflow_id))
         << ",\"parent_transfer_id\":" << quoted(std::to_string(r.parent_transfer_id))
         << ",\"related_workflow_id\":" << quoted(std::to_string(r.related_workflow_id))
         << ",\"client_request_id\":" << quoted(std::to_string(r.client_request_id))
         << ",\"session_id\":" << quoted(std::to_string(e.session_id))
         << ",\"span_id\":" << quoted(std::to_string(e.span_id)) << ",\"rank\":" << r.rank
         << ",\"batch_begin\":" << r.begin << ",\"batch_end\":" << r.end << ",\"stage\":" << quoted(r.stage)
         << ",\"path\":" << quoted(r.path) << ",\"scope\":" << quoted(r.scope)
         << ",\"operation\":" << quoted(r.operation) << ",\"start_tid\":" << r.start_tid << ",\"end_tid\":" << e.end_tid
         << ",\"start_clipped\":" << (e.start_clipped ? "true" : "false")
         << ",\"end_clipped\":" << (e.end_clipped ? "true" : "false")
         << ",\"incomplete\":" << (e.start_clipped || e.end_clipped ? "true" : "false")
         << ",\"thread_mismatch\":" << (thread_mismatch ? "true" : "false") << '}';
    auto common = [&] {
        if (!first)
            out << ',';
        first = false;
        // Async track names are stable across requests; identities belong in
        // args so a busy capture does not create one track per transfer.
        const auto name = execution ? e.name : logicalTrackName(r);
        out << "\n{\"cat\":" << quoted(execution ? "cache.transfer.execution" : "cache.transfer.logical")
            << ",\"name\":" << quoted(name) << ",\"pid\":" << e.pid << ",\"tid\":" << (execution ? r.start_tid : 0)
            << ",\"args\":" << args.str();
    };
    common();
    if (execution) {
        out << ",\"ph\":\"X\",\"ts\":" << timeUs(start) << ",\"dur\":" << timeUs(end - start) << '}';
    } else {
        const auto id    = quoted(std::to_string(e.session_id) + ":" + std::to_string(e.span_id));
        const auto scope = quoted("cache.transfer.logical");
        out << ",\"ph\":\"b\",\"ts\":" << timeUs(start) << ",\"id2\":{\"local\":" << id << "},\"scope\":" << scope
            << '}';
        common();
        out << ",\"ph\":\"e\",\"ts\":" << timeUs(end) << ",\"id2\":{\"local\":" << id << "},\"scope\":" << scope << '}';
    }
}

}  // namespace

namespace detail {
AsyncProfileKind exportedAsyncProfileKind(const AsyncProfileEvent& event) {
    return exportedKind(event);
}
int64_t asyncProfileBaseTimeNs(const std::string& trace) {
    return baseTimeNs(trace);
}
std::vector<AsyncProfileEvent> resolveAsyncProfileLinks(const std::vector<AsyncProfileEvent>& events) {
    // Task IDs are process-local. A remote rank can reuse the same numeric ID.
    using Key = std::pair<int64_t, uint64_t>;
    std::map<Key, const AsyncProfileMetadata*> links;
    for (const auto& e : events)
        if (e.record.stage == "task_link")
            links[{e.pid, e.record.transfer_id}] = &e.record;
    auto resolved = events;
    for (auto& e : resolved) {
        const auto found = links.find({e.pid, e.record.transfer_id});
        if (!e.record.workflow_id && found != links.end()) {
            e.record.workflow_id        = found->second->workflow_id;
            e.record.parent_transfer_id = found->second->parent_transfer_id;
        }
        if (!e.record.workflow_id && e.record.transfer_id)
            e.record.workflow_id = e.record.transfer_id;
    }
    return resolved;
}
}  // namespace detail

std::string mergeAsyncProfileTrace(const std::string&                    native_trace,
                                   const std::vector<AsyncProfileEvent>& input_events) {
    const auto events = detail::resolveAsyncProfileLinks(input_events);
    if (events.empty())
        return native_trace;
    auto       begin     = rootValue(native_trace, "traceEvents");
    auto       end       = arrayEnd(native_trace, begin);
    auto       base      = baseTimeNs(native_trace);
    bool       first     = skipSpace(native_trace, begin + 1) == end;
    auto       extra     = detail::asyncProfileOutputStream();
    const bool has_names = std::any_of(events.begin(), events.end(), [](const auto& e) {
        return exportedKind(e) == AsyncProfileKind::Execution && !e.record.start_thread_name.empty();
    });
    if (has_names) {
        auto named = namedNativeThreads(native_trace, begin, end);
        for (const auto& event : events) {
            const auto& r = event.record;
            if (exportedKind(event) != AsyncProfileKind::Execution || r.start_thread_name.empty()
                || !named.emplace(event.pid, r.start_tid).second)
                continue;
            if (!first)
                extra << ',';
            first = false;
            extra << "\n{\"cat\":\"cache.transfer.metadata\",\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":"
                  << event.pid << ",\"tid\":" << r.start_tid << ",\"args\":{\"name\":" << quoted(r.start_thread_name)
                  << "}}";
        }
    }
    for (const auto& event : events) {
        // Keep every real execution event, including its request identity.
        // Only the optional request-owned logical projection is removed.
        if (exportedKind(event) != AsyncProfileKind::Execution
            && (event.record.request_id >= 0 || event.record.path == "request"))
            continue;
        writeEvent(extra, event, base, first);
    }
    auto merged = native_trace;
    merged.insert(end, extra.str());
    return merged;
}

void appendAsyncProfileTrace(const std::string& path, const std::vector<AsyncProfileEvent>& events) {
    if (events.empty())
        return;
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read native Kineto export");
    const std::string original{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad())
        throw std::runtime_error("cannot read complete Kineto export");
    auto        merged = mergeAsyncProfileTrace(original, events);
    struct stat native_stat;
    if (stat(path.c_str(), &native_stat) != 0)
        throw std::runtime_error("cannot stat native Kineto export");
    auto              pattern = path + ".async.XXXXXX";
    std::vector<char> temporary(pattern.begin(), pattern.end());
    temporary.push_back('\0');
    int fd = mkstemp(temporary.data());
    if (fd < 0)
        throw std::runtime_error("cannot create async trace export");
    if (fchmod(fd, native_stat.st_mode & 0777) != 0) {
        close(fd);
        std::remove(temporary.data());
        throw std::runtime_error("cannot preserve native trace permissions");
    }
    auto file = fdopen(fd, "wb");
    if (!file) {
        close(fd);
        std::remove(temporary.data());
        throw std::runtime_error("cannot open async trace export");
    }
    bool written = std::fwrite(merged.data(), 1, merged.size(), file) == merged.size();
    bool closed  = std::fclose(file) == 0;
    if (!written || !closed || std::rename(temporary.data(), path.c_str()) != 0) {
        std::remove(temporary.data());
        throw std::runtime_error("cannot commit async trace export; native trace retained");
    }
    // The compatibility JSON is already committed. Sidecar failure must never
    // destroy it; callers report the failed additional export.
    merged.clear();
    merged.shrink_to_fit();
    writeAsyncProfileBundle(path + ".zip", original, events);
}

}  // namespace rtp_llm
