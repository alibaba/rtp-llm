#include "rtp_llm/cpp/engine_base/awi_runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace rtp_llm {
namespace awi {

namespace {

bool isHexSha(const char* value) {
    if (value == nullptr) {
        return false;
    }
    const size_t len = std::strlen(value);
    if (len != 40) {
        return false;
    }
    for (size_t i = 0; i < len; ++i) {
        const char c = value[i];
        const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!hex) {
            return false;
        }
    }
    return true;
}

std::string readEnvSha(const char* name) {
    const char* value = std::getenv(name);
    if (isHexSha(value)) {
        return std::string(value);
    }
    return {};
}

std::string jsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string quote(const std::string& value) {
    return "\"" + jsonEscape(value) + "\"";
}

std::string boolString(bool value) {
    return value ? "true" : "false";
}

std::string eventToJson(const Event& event, const std::string& operation_id) {
    std::ostringstream out;
    out << "{";
    out << "\"type\":" << quote(event.type);
    if (!operation_id.empty()) {
        out << ",\"operation_id\":" << quote(operation_id);
    } else if (!event.operation_id.empty()) {
        out << ",\"operation_id\":" << quote(event.operation_id);
    }
    if (!event.request_id.empty()) {
        out << ",\"request_id\":" << quote(event.request_id);
    }
    if (event.step >= 0 && event.type == "decode") {
        out << ",\"step\":" << event.step;
    }
    if (event.input_tokens > 0) {
        out << ",\"input_tokens\":" << event.input_tokens;
    }
    if (event.start_ns > 0) {
        out << ",\"start_ns\":" << event.start_ns;
    }
    if (event.end_ns > 0) {
        out << ",\"end_ns\":" << event.end_ns;
    }
    if (event.elapsed_ms > 0.0) {
        out << ",\"elapsed_ms\":" << event.elapsed_ms;
    }
    if (!event.backend.empty()) {
        out << ",\"backend\":" << quote(event.backend);
    }
    if (event.type == "decode") {
        out << ",\"graph_replay\":" << boolString(event.graph_replay);
    }
    if (event.hit_tokens > 0 || event.type == "reuse") {
        out << ",\"hit_tokens\":" << event.hit_tokens;
    }
    if (event.concurrency > 0) {
        out << ",\"concurrency\":" << event.concurrency;
    }
    if (event.context_tokens > 0) {
        out << ",\"context_tokens\":" << event.context_tokens;
    }
    if (!event.failure_kind.empty()) {
        out << ",\"failure_kind\":" << quote(event.failure_kind);
    }
    if (!event.sequences.empty()) {
        out << ",\"sequences\":[";
        for (size_t i = 0; i < event.sequences.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto& seq = event.sequences[i];
            out << "{\"request_id\":" << quote(seq.request_id) << ",\"input_tokens\":" << seq.input_tokens
                << ",\"kv_tokens\":" << seq.kv_tokens << ",\"reserved_tokens\":" << seq.reserved_tokens << "}";
        }
        out << "]";
    }
    out << ",\"rank\":" << event.rank << ",\"world_size\":" << event.world_size;
    if (!event.source.empty()) {
        out << ",\"source\":" << quote(event.source);
    }
    out << "}";
    return out.str();
}

}  // namespace

AwiRuntime& AwiRuntime::instance() {
    static AwiRuntime runtime;
    return runtime;
}

void AwiRuntime::setParallel(int32_t rank, int32_t world_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    rank_       = rank;
    world_size_ = std::max<int32_t>(1, world_size);
}

const std::string& AwiRuntime::graphBackend() {
#if defined(USING_ROCM) && USING_ROCM
    static const std::string backend = "hip";
#elif defined(USING_CUDA) && USING_CUDA
    static const std::string backend = "cuda";
#else
    static const std::string backend = "unknown";
#endif
    return backend;
}

bool AwiRuntime::graphBackendIsHip() {
    return graphBackend() == "hip";
}

namespace {
std::string gitHead() {
    FILE* pipe = popen("git rev-parse HEAD 2>/dev/null", "r");
    if (pipe == nullptr) {
        return {};
    }
    char buffer[64] = {0};
    char* line      = fgets(buffer, sizeof(buffer), pipe);
    pclose(pipe);
    if (line == nullptr) {
        return {};
    }
    std::string value(buffer);
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
        value.pop_back();
    }
    return isHexSha(value.c_str()) ? value : std::string();
}
}  // namespace

std::string AwiRuntime::internalCommit() {
    static std::string cached = []() {
        auto value = readEnvSha("RTP_INTERNAL_COMMIT");
#if defined(RTP_AWI_INTERNAL_COMMIT)
        if (value.empty()) {
            value = RTP_AWI_INTERNAL_COMMIT;
            if (!isHexSha(value.c_str())) {
                value.clear();
            }
        }
#endif
        if (value.empty()) {
            value = gitHead();
        }
        return value;
    }();
    return cached;
}

std::string AwiRuntime::publicCommit() {
    static std::string cached = []() {
        auto value = readEnvSha("RTP_PUBLIC_COMMIT");
#if defined(RTP_AWI_PUBLIC_COMMIT)
        if (value.empty()) {
            value = RTP_AWI_PUBLIC_COMMIT;
            if (!isHexSha(value.c_str())) {
                value.clear();
            }
        }
#endif
        if (value.empty()) {
            value = gitHead();
        }
        return value;
    }();
    return cached;
}

void AwiRuntime::resetBuildCacheForTest() {
    // No-op: the build SHA is process-immutable.  Kept for test translation
    // units that include the header.
}

void AwiRuntime::bindRequest(int64_t stream_id, const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_to_request_[stream_id] = request_id;
    request_to_stream_[request_id] = stream_id;
    auto& state = request_state_[request_id];
    state.bound = true;
    recording_  = true;
}

void AwiRuntime::unbindRequest(int64_t stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stream_to_request_.find(stream_id);
    if (it != stream_to_request_.end()) {
        // Keep the request state for the final observations even after the
        // stream is released; only the live id mapping is dropped.
        stream_to_request_.erase(it);
    }
    recording_ = !stream_to_request_.empty() || !barrier_operation_.empty();
}

std::string AwiRuntime::requestIdForStream(int64_t stream_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stream_to_request_.find(stream_id);
    return it == stream_to_request_.end() ? std::string() : it->second;
}

std::string AwiRuntime::operationForRequestLocked(const std::string& request_id) const {
    auto it = request_operation_.find(request_id);
    return it == request_operation_.end() ? std::string() : it->second;
}

void AwiRuntime::appendEventLocked(const Event& event) {
    events_.push_back(event);
    // Keep the evaluator response bounded even for long capacity searches.
    constexpr size_t kMaxEvents = 200000;
    if (events_.size() > kMaxEvents) {
        events_.erase(events_.begin(), events_.begin() + (events_.size() - kMaxEvents));
    }
}

void AwiRuntime::recordCapture(const std::string& backend, const std::string& source) {
    std::lock_guard<std::mutex> lock(mutex_);
    Event event;
    event.type         = "capture";
    event.backend      = backend.empty() ? graphBackend() : backend;
    event.graph_replay = false;
    event.source       = source;
    event.rank         = rank_;
    event.world_size   = world_size_;
    appendEventLocked(event);
}

void AwiRuntime::recordGraphForward(const std::string& backend, bool replayed, const std::string& source) {
    observeForwardGraphReplay(backend, replayed, source);
}

void AwiRuntime::recordPrefillChunk(const std::string& request_id,
                                    int64_t            input_tokens,
                                    int64_t            start_ns,
                                    int64_t            end_ns,
                                    bool               final_chunk,
                                    int32_t            rank,
                                    int32_t            world_size,
                                    const std::string& source) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Aggregate chunked prefill under the same request id.  The first chunk
    // owns start_ns, the final chunk publishes the complete prefill window.
    auto it = std::find_if(events_.begin(), events_.end(), [&](const Event& event) {
        return event.type == "prefill" && event.request_id == request_id && event.step < 0;
    });
    if (it != events_.end()) {
        it->end_ns       = std::max(it->end_ns, end_ns);
        it->input_tokens = std::max(it->input_tokens, input_tokens);
        if (final_chunk) {
            it->step = 0;
        }
        return;
    }
    Event event;
    event.type         = "prefill";
    event.request_id   = request_id;
    event.operation_id = operationForRequestLocked(request_id);
    event.input_tokens = input_tokens;
    event.start_ns     = start_ns;
    event.end_ns       = end_ns;
    event.source       = source;
    event.rank         = rank >= 0 ? rank : rank_;
    event.world_size   = world_size > 0 ? world_size : world_size_;
    // A provisional (chunked) event has step < 0 so a later chunk extends it
    // instead of appending a second prefill record.
    event.step = final_chunk ? 0 : -2;
    appendEventLocked(event);
}

void AwiRuntime::recordDecode(const std::string& request_id,
                              int64_t            step,
                              int64_t            elapsed_ns,
                              bool               graph_replay,
                              const std::string& backend,
                              int32_t            rank,
                              int32_t            world_size,
                              const std::string& source) {
    std::lock_guard<std::mutex> lock(mutex_);
    Event event;
    event.type         = "decode";
    event.request_id   = request_id;
    event.operation_id = operationForRequestLocked(request_id);
    event.step         = step;
    event.elapsed_ms   = static_cast<double>(elapsed_ns) / 1e6;
    event.graph_replay = graph_replay;
    event.backend      = backend.empty() ? graphBackend() : backend;
    event.source       = source;
    event.rank         = rank >= 0 ? rank : rank_;
    event.world_size   = world_size > 0 ? world_size : world_size_;
    appendEventLocked(event);
}

void AwiRuntime::recordResident(const ResidentSnapshot& snapshot,
                                const std::string&      operation_id,
                                const std::string&      source) {
    std::lock_guard<std::mutex> lock(mutex_);
    Event event;
    event.type           = "resident";
    event.operation_id   = operation_id.empty() ? barrier_operation_ : operation_id;
    event.sequences      = snapshot.sequences;
    event.source         = source;
    event.rank           = rank_;
    event.world_size     = world_size_;
    event.context_tokens = barrier_context_tokens_;
    appendEventLocked(event);
}

void AwiRuntime::recordReuse(const std::string& request_id,
                             int64_t            hit_tokens,
                             int64_t            input_tokens,
                             const std::string& source) {
    std::lock_guard<std::mutex> lock(mutex_);
    Event event;
    event.type         = "reuse";
    event.request_id   = request_id;
    event.operation_id = operationForRequestLocked(request_id);
    event.hit_tokens   = hit_tokens;
    event.input_tokens = input_tokens;
    event.source       = source;
    event.rank         = rank_;
    event.world_size   = world_size_;
    appendEventLocked(event);
}

void AwiRuntime::recordCapacityFailure(const std::string& operation_id,
                                       int64_t            concurrency,
                                       int64_t            context_tokens,
                                       const std::string& failure_kind,
                                       const std::string& detail,
                                       const std::string& source) {
    std::lock_guard<std::mutex> lock(mutex_);
    Event event;
    event.type           = "capacity_failure";
    event.operation_id   = operation_id;
    event.concurrency    = concurrency;
    event.context_tokens = context_tokens;
    event.failure_kind   = failure_kind;
    event.source         = detail.empty() ? source : (source + ": " + detail);
    event.rank           = rank_;
    event.world_size     = world_size_;
    appendEventLocked(event);
}

void AwiRuntime::armBarrier(const std::string&              operation_id,
                            const std::vector<std::string>& request_ids,
                            int64_t                         context_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    barrier_operation_      = operation_id;
    barrier_requests_.clear();
    barrier_context_tokens_ = context_tokens;
    resident_snapshot_recorded_ = false;
    for (const auto& id : request_ids) {
        if (id.empty()) {
            continue;
        }
        barrier_requests_.insert(id);
        request_operation_[id] = operation_id;
        auto& state            = request_state_[id];
        state.bound            = true;
        // /awi/generate sets the exact input+output target before the stream
        // can be observed.  Never fall back to the operation-wide context here:
        // that would demand the full context from a single request.
    }
}

void AwiRuntime::clearBarrier() {
    std::lock_guard<std::mutex> lock(mutex_);
    barrier_operation_.clear();
    barrier_requests_.clear();
    barrier_context_tokens_ = 0;
}

bool AwiRuntime::barrierActive() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !barrier_operation_.empty() && !barrier_requests_.empty();
}

bool AwiRuntime::isBarrierRequest(int64_t stream_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (barrier_requests_.empty()) {
        return false;
    }
    auto it = stream_to_request_.find(stream_id);
    if (it == stream_to_request_.end()) {
        return false;
    }
    return barrier_requests_.count(it->second) != 0;
}

void AwiRuntime::setRequestTarget(int64_t stream_id, int64_t target_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stream_to_request_.find(stream_id);
    if (it == stream_to_request_.end()) {
        return;
    }
    auto& state            = request_state_[it->second];
    state.bound            = true;
    state.target_tokens    = target_tokens;
}

bool AwiRuntime::barrierReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return barrierReadyLocked();
}

bool AwiRuntime::barrierReadyLocked() const {
    if (barrier_operation_.empty() || barrier_requests_.empty()) {
        return true;
    }
    int64_t total_reserved = 0;
    for (const auto& id : barrier_requests_) {
        auto it = request_state_.find(id);
        if (it == request_state_.end() || !it->second.seen) {
            return false;
        }
        const auto& state = it->second;
        const auto  target = state.target_tokens > 0 ? state.target_tokens : state.input_tokens;
        if (target > 0 && state.reserved_tokens < target) {
            return false;
        }
        if (state.kv_tokens < state.input_tokens) {
            return false;
        }
        total_reserved += state.reserved_tokens;
    }
    // The operation requests the whole context to be resident at once.  This
    // is a real allocator-derived sum, not the requested value echoed back.
    if (barrier_context_tokens_ > 0 && total_reserved < barrier_context_tokens_) {
        return false;
    }
    return true;
}

bool AwiRuntime::observeResidentAndCheckReady(int64_t stream_id,
                                              int64_t input_tokens,
                                              int64_t kv_tokens,
                                              int64_t reserved_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stream_to_request_.find(stream_id);
    if (it == stream_to_request_.end()) {
        return true;
    }
    auto& state              = request_state_[it->second];
    state.input_tokens       = std::max<int64_t>(state.input_tokens, input_tokens);
    state.kv_tokens          = std::max<int64_t>(state.kv_tokens, kv_tokens);
    state.reserved_tokens    = std::max<int64_t>(state.reserved_tokens, reserved_tokens);
    state.seen               = true;
    return barrierReadyLocked();
}

bool AwiRuntime::consumeResidentSnapshotRequest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!barrierReadyLocked() || resident_snapshot_recorded_) {
        return false;
    }
    resident_snapshot_recorded_ = true;
    return true;
}

int64_t AwiRuntime::barrierContextTokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return barrier_context_tokens_;
}

std::string AwiRuntime::barrierOperationForObservation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return barrier_operation_;
}

size_t AwiRuntime::barrierRequestCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return barrier_requests_.size();
}

bool AwiRuntime::recordingEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recording_;
}

std::string AwiRuntime::eventsJson(const std::string& operation_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << "[";
    bool first = true;
    for (const auto& event : events_) {
        std::string op = event.operation_id;
        if (op.empty() && !event.request_id.empty()) {
            auto it = request_operation_.find(event.request_id);
            if (it != request_operation_.end()) {
                op = it->second;
            }
        }
        const bool include = operation_id.empty() || op.empty() || op == operation_id;
        if (!include) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << eventToJson(event, op);
    }
    out << "]";
    return out.str();
}

std::string AwiRuntime::observationsJson(const std::string& operation_id, const std::string& weights_json) const {
    std::ostringstream out;
    out << "{";
    out << "\"build\":{\"source\":{\"internal\":" << quote(internalCommit())
        << ",\"public\":" << quote(publicCommit()) << "}},";
    out << "\"configuration\":{\"context_tokens\":1048576"
        << ",\"memory_kv_reuse\":true"
        << ",\"graph_backend\":" << quote(graphBackend()) << "},";
    out << "\"startup_network_fetches\":0,";
    out << "\"weights\":" << (weights_json.empty() ? "[]" : weights_json) << ",";
    out << "\"events\":" << eventsJson(operation_id);
    out << "}";
    return out.str();
}

int64_t AwiRuntime::nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int64_t AwiRuntime::nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ---- per-forward graph observation -----------------------------------------

namespace {
thread_local bool        g_forward_graph_replayed = false;
thread_local std::string g_forward_graph_backend;
thread_local bool        g_forward_graph_seen     = false;
}  // namespace

void beginForwardGraphObservation() {
    g_forward_graph_replayed = false;
    g_forward_graph_backend.clear();
    g_forward_graph_seen = false;
}

void observeForwardGraphReplay(const std::string& backend, bool replayed, const std::string& /*source*/) {
    g_forward_graph_seen     = true;
    g_forward_graph_replayed = g_forward_graph_replayed || replayed;
    if (!backend.empty()) {
        g_forward_graph_backend = backend;
    }
}

bool takeForwardGraphObservation(std::string& backend, bool& replayed) {
    if (!g_forward_graph_seen) {
        return false;
    }
    backend  = g_forward_graph_backend.empty() ? AwiRuntime::graphBackend() : g_forward_graph_backend;
    replayed = g_forward_graph_replayed;
    // Reset so the next executor step starts clean.
    g_forward_graph_seen     = false;
    g_forward_graph_replayed = false;
    g_forward_graph_backend.clear();
    return true;
}

}  // namespace awi
}  // namespace rtp_llm