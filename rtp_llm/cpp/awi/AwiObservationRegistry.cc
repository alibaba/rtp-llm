#include "rtp_llm/cpp/awi/AwiObservationRegistry.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace rtp_llm {

AwiObservationRegistry& AwiObservationRegistry::instance() {
    static AwiObservationRegistry registry;
    return registry;
}

int64_t AwiObservationRegistry::nowNs() {
    // CLOCK_REALTIME: common across rank processes on the same host, matching
    // the engine's autil::TimeUtility wall clock (contract: common host clock).
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string AwiObservationRegistry::jsonEscape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
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
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

void AwiObservationRegistry::armBarrier(const std::string&              operation_id,
                                        std::vector<std::string>       request_ids,
                                        int64_t                        context_tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    AwiBarrierOp                op;
    op.operation_id   = operation_id;
    op.request_ids    = std::move(request_ids);
    op.context_tokens = context_tokens;
    op.arm_time_ns    = nowNs();
    for (const auto& req : op.request_ids) {
        request_to_op_[req] = operation_id;
    }
    ops_[operation_id] = std::move(op);
}

bool AwiObservationRegistry::isArmed(const std::string& request_id, std::string* operation_id_out) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = request_to_op_.find(request_id);
    if (it == request_to_op_.end()) {
        return false;
    }
    if (operation_id_out != nullptr) {
        *operation_id_out = it->second;
    }
    return true;
}

bool AwiObservationRegistry::belongsToOp(const std::string& operation_id, const std::string& request_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = request_to_op_.find(request_id);
    return it != request_to_op_.end() && it->second == operation_id;
}

size_t AwiObservationRegistry::opPeerCount(const std::string& operation_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = ops_.find(operation_id);
    return it == ops_.end() ? 0 : it->second.request_ids.size();
}

bool AwiObservationRegistry::opResolved(const std::string& operation_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = ops_.find(operation_id);
    return it == ops_.end() || it->second.released || it->second.failed;
}

void AwiObservationRegistry::releaseBarrier(const std::string& operation_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = ops_.find(operation_id);
    if (it != ops_.end()) {
        it->second.released = true;
    }
}

void AwiObservationRegistry::failBarrier(const std::string& operation_id,
                                         const std::string& failure_kind,
                                         int64_t            concurrency) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = ops_.find(operation_id);
    if (it == ops_.end()) {
        return;
    }
    it->second.failed = true;
    AwiCapacityFailureEvent ev;
    ev.operation_id   = operation_id;
    ev.concurrency    = concurrency;
    ev.context_tokens = it->second.context_tokens;
    ev.failure_kind   = failure_kind;
    capacity_events_.push_back(ev);
}

bool AwiObservationRegistry::markResidentRecorded(const std::string& operation_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = ops_.find(operation_id);
    if (it == ops_.end() || it->second.resident_recorded) {
        return false;
    }
    it->second.resident_recorded = true;
    return true;
}

void AwiObservationRegistry::recordPrefill(const AwiPrefillEvent& e) {
    std::lock_guard<std::mutex> lock(mu_);
    prefill_events_.push_back(e);
}

void AwiObservationRegistry::recordDecode(const AwiDecodeEvent& e) {
    std::lock_guard<std::mutex> lock(mu_);
    decode_events_.push_back(e);
}

void AwiObservationRegistry::recordResident(const std::vector<AwiSeqSnapshot>& sequences) {
    std::lock_guard<std::mutex> lock(mu_);
    resident_events_.push_back(sequences);
}

void AwiObservationRegistry::recordReuse(const AwiReuseEvent& e) {
    std::lock_guard<std::mutex> lock(mu_);
    reuse_events_.push_back(e);
}

void AwiObservationRegistry::recordCapture(const std::string& backend) {
    std::lock_guard<std::mutex> lock(mu_);
    capture_backends_.push_back(backend);
}

bool AwiObservationRegistry::isTracked(const std::string& request_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return request_to_op_.count(request_id) != 0;
}

void AwiObservationRegistry::prefillStep(const std::string& request_id,
                                         int64_t            input_tokens,
                                         int64_t            step_start_ns,
                                         int64_t            step_end_ns) {
    std::lock_guard<std::mutex> lock(mu_);
    if (request_to_op_.count(request_id) == 0) {
        return;
    }
    auto& pending = prefill_pending_[request_id];
    if (pending.start_ns == 0) {
        pending.start_ns = step_start_ns;
    }
    pending.input_tokens = input_tokens;
    pending.end_ns       = step_end_ns;
}

void AwiObservationRegistry::finalizePrefill(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = prefill_pending_.find(request_id);
    if (it == prefill_pending_.end() || it->second.finalized) {
        return;
    }
    it->second.finalized = true;
    AwiPrefillEvent ev;
    ev.request_id   = request_id;
    ev.input_tokens = it->second.input_tokens;
    ev.start_ns     = it->second.start_ns;
    ev.end_ns       = it->second.end_ns;
    prefill_events_.push_back(ev);
}

int64_t AwiObservationRegistry::nextDecodeStep(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    return ++decode_steps_[request_id];
}

void AwiObservationRegistry::noteGraphReplay() {
    graph_replay_count_.fetch_add(1, std::memory_order_relaxed);
}

uint64_t AwiObservationRegistry::graphReplayCount() const {
    return graph_replay_count_.load(std::memory_order_relaxed);
}

void AwiObservationRegistry::recordReuseOnce(const std::string& request_id, int64_t hit_tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    // Reuse hits are recorded for EVERY request (not only barrier-armed ones):
    // the evaluator's reuse smoke drives plain /v1/chat/completions cases whose
    // rids are never armed, yet their hits must appear as kind=reuse events.
    if (request_id.empty() || hit_tokens <= 0) {
        return;
    }
    if (!reuse_recorded_.insert(request_id).second) {
        return;
    }
    AwiReuseEvent ev;
    ev.request_id = request_id;
    ev.hit_tokens = hit_tokens;
    reuse_events_.push_back(ev);
}

bool AwiObservationRegistry::opAwaitingResident(std::string* operation_id_out) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& entry : ops_) {
        if (entry.second.released && !entry.second.failed && !entry.second.resident_recorded) {
            if (operation_id_out != nullptr) {
                *operation_id_out = entry.first;
            }
            return true;
        }
    }
    return false;
}

bool AwiObservationRegistry::opRequestIds(const std::string& operation_id, std::vector<std::string>* ids_out) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = ops_.find(operation_id);
    if (it == ops_.end() || ids_out == nullptr) {
        return false;
    }
    *ids_out = it->second.request_ids;
    return true;
}

void AwiObservationRegistry::recordDeviceOom(int64_t concurrency_hint) {
    std::lock_guard<std::mutex> lock(mu_);
    AwiCapacityFailureEvent ev;
    ev.failure_kind = "device_oom";
    ev.concurrency  = concurrency_hint;
    // Attribute to the most recently armed operation, if any.
    const AwiBarrierOp* newest = nullptr;
    for (const auto& entry : ops_) {
        if (newest == nullptr || entry.second.arm_time_ns > newest->arm_time_ns) {
            newest = &entry.second;
        }
    }
    if (newest != nullptr) {
        ev.operation_id   = newest->operation_id;
        ev.context_tokens = newest->context_tokens;
    }
    capacity_events_.push_back(ev);
}

void AwiObservationRegistry::recordStartupNetworkFetch() {
    std::lock_guard<std::mutex> lock(mu_);
    ++startup_network_fetches_;
}

void AwiObservationRegistry::setBuildInfo(const std::string& internal_sha, const std::string& public_sha) {
    std::lock_guard<std::mutex> lock(mu_);
    internal_sha_ = internal_sha;
    public_sha_   = public_sha;
}

void AwiObservationRegistry::setConfiguration(int64_t context_tokens, bool memory_kv_reuse, const std::string& graph_backend) {
    std::lock_guard<std::mutex> lock(mu_);
    cfg_context_tokens_  = context_tokens;
    cfg_memory_kv_reuse_ = memory_kv_reuse;
    cfg_graph_backend_   = graph_backend;
}

void AwiObservationRegistry::setWeightsPlacementJson(const std::string& weights_json_array) {
    std::lock_guard<std::mutex> lock(mu_);
    weights_json_ = weights_json_array;
}

void AwiObservationRegistry::setRank(int64_t world_rank) {
    std::lock_guard<std::mutex> lock(mu_);
    rank_world_rank_ = world_rank;
}

int64_t AwiObservationRegistry::rank() const {
    std::lock_guard<std::mutex> lock(mu_);
    return rank_world_rank_;
}

std::string AwiObservationRegistry::queryJson(const std::string& operation_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::unordered_set<std::string> op_requests;
    auto                            op_it = ops_.find(operation_id);
    if (op_it != ops_.end()) {
        for (const auto& r : op_it->second.request_ids) {
            op_requests.insert(r);
        }
    }
    // Evaluator request ids are all prefixed with the operation id (wave ids
    // "<op>-r<rep>-s<sample>-c<i>", smoke ids "<op>-smoke-<file>-<i>"). The
    // prefix match keeps non-armed smoke events (e.g. reuse hits) queryable,
    // while armed wave ids match exactly through op_requests.
    const bool include_all = operation_id.empty();
    auto       include     = [&include_all, &operation_id, &op_requests](const std::string& request_id) {
        return include_all || op_requests.count(request_id) != 0
               || request_id.compare(0, operation_id.size(), operation_id) == 0;
    };

    std::string out;
    out.reserve(4096);
    out += "{\"rank\":";
    out += std::to_string(rank_world_rank_);
    // Per-rank graph replay evidence: every rank replays its shard graphs in
    // lockstep, so cross-rank totals let the reviewer detect an eager fallback
    // on any participating rank (decode events aggregate on the scheduling rank).
    out += ",\"graph_replay_total\":";
    out += std::to_string(graph_replay_count_.load(std::memory_order_relaxed));
    out += ",\"build\":{\"source\":{\"internal\":\"" + jsonEscape(internal_sha_) + "\",\"public\":\""
           + jsonEscape(public_sha_) + "\"}}";
    out += ",\"configuration\":{\"context_tokens\":" + std::to_string(cfg_context_tokens_)
           + ",\"memory_kv_reuse\":" + (cfg_memory_kv_reuse_ ? "true" : "false") + ",\"graph_backend\":\""
           + jsonEscape(cfg_graph_backend_) + "\"}";
    out += ",\"startup_network_fetches\":" + std::to_string(startup_network_fetches_);
    out += ",\"weights\":" + (weights_json_.empty() ? std::string("[]") : weights_json_);
    out += ",\"events\":[";
    bool first = true;
    auto sep   = [&first, &out]() {
        if (!first) {
            out += ",";
        }
        first = false;
    };
    for (const auto& backend : capture_backends_) {
        sep();
        out += "{\"kind\":\"capture\",\"backend\":\"" + jsonEscape(backend) + "\"}";
    }
    for (const auto& e : prefill_events_) {
        if (!include(e.request_id)) {
            continue;
        }
        sep();
        out += "{\"kind\":\"prefill\",\"request_id\":\"" + jsonEscape(e.request_id) + "\",\"input_tokens\":"
               + std::to_string(e.input_tokens) + ",\"start_ns\":" + std::to_string(e.start_ns)
               + ",\"end_ns\":" + std::to_string(e.end_ns) + "}";
    }
    for (const auto& e : decode_events_) {
        if (!include(e.request_id)) {
            continue;
        }
        sep();
        out += "{\"kind\":\"decode\",\"request_id\":\"" + jsonEscape(e.request_id) + "\",\"step\":"
               + std::to_string(e.step) + ",\"backend\":\"" + jsonEscape(e.backend) + "\",\"graph_replay\":"
               + (e.graph_replay ? "true" : "false") + ",\"elapsed_ms\":" + std::to_string(e.elapsed_ms) + "}";
    }
    for (const auto& sequences : resident_events_) {
        bool intersects = include_all;
        for (const auto& s : sequences) {
            if (include(s.request_id)) {
                intersects = true;
                break;
            }
        }
        if (!intersects) {
            continue;
        }
        sep();
        out += "{\"kind\":\"resident\",\"sequences\":[";
        for (size_t i = 0; i < sequences.size(); ++i) {
            if (i > 0) {
                out += ",";
            }
            const auto& s = sequences[i];
            out += "{\"request_id\":\"" + jsonEscape(s.request_id) + "\",\"input_tokens\":"
                   + std::to_string(s.input_tokens) + ",\"kv_tokens\":" + std::to_string(s.kv_tokens)
                   + ",\"reserved_tokens\":" + std::to_string(s.reserved_tokens) + "}";
        }
        out += "]}";
    }
    for (const auto& e : reuse_events_) {
        if (!include(e.request_id)) {
            continue;
        }
        sep();
        out += "{\"kind\":\"reuse\",\"request_id\":\"" + jsonEscape(e.request_id) + "\",\"hit_tokens\":"
               + std::to_string(e.hit_tokens) + "}";
    }
    for (const auto& e : capacity_events_) {
        if (!include_all && e.operation_id != operation_id) {
            continue;
        }
        sep();
        out += "{\"kind\":\"capacity_failure\",\"operation_id\":\"" + jsonEscape(e.operation_id)
               + "\",\"concurrency\":" + std::to_string(e.concurrency) + ",\"context_tokens\":"
               + std::to_string(e.context_tokens) + ",\"failure_kind\":\"" + jsonEscape(e.failure_kind) + "\"}";
    }
    out += "]}";
    return out;
}

}  // namespace rtp_llm
