#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace rtp_llm {

// Per-process telemetry store for the AWI evaluator contract
// (evaluator-protocol.md). Engine components append one JSON event per
// real execution occurrence (prefill/decode step, graph capture, residency
// snapshot, reuse hit, capacity failure); nothing here is derived from
// requested settings. The AwiCommandInternal RPC drains the store.
//
// Events are stored as pre-serialized JSON objects together with the
// string request/operation identifiers they belong to, so collection can
// filter without a JSON parser on the hot path.
class AwiRankStore {
public:
    static AwiRankStore& instance();

    // json_event must be a single serialized JSON object. request_id is the
    // string scheduler observation id (X-AWI-Request-ID propagation via
    // RequestInfoPB.request_id); operation_id is non-empty only for events
    // produced inside an armed barrier operation.
    void appendEvent(std::string json_event, std::string request_id, std::string operation_id);

    void armBarrier(std::string operation_id, std::vector<std::string> request_ids, int64_t context_tokens);

    bool isBarrierRequest(const std::string& request_id) const;

    // Assembled JSON object:
    // {"events":[...], "barriers":[...], "dropped_events": N}
    // operation_id empty => every event; otherwise only events tagged with
    // that operation or whose request_id was armed by it. Capture events
    // carry their own operation tag when armed inside an operation.
    std::string collectJson(const std::string& operation_id) const;

    void setEventCapacity(size_t cap);

private:
    AwiRankStore();

    struct Event {
        std::string json;
        std::string request_id;
        std::string operation_id;
    };
    struct Barrier {
        std::string operation_id;
        std::vector<std::string> request_ids;
        int64_t context_tokens = 0;
        int64_t armed_ns = 0;
    };

    mutable std::mutex     mu_;
    std::deque<Event>      events_;
    std::vector<Barrier>   barriers_;
    size_t                 dropped_ = 0;
    size_t                 cap_;
};

}  // namespace rtp_llm
