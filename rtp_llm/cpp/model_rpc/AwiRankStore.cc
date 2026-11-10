#include "rtp_llm/cpp/model_rpc/AwiRankStore.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace rtp_llm {

namespace {

size_t eventCapFromEnv() {
    const char* env = getenv("AWI_EVENT_CAP");
    if (env != nullptr) {
        char* end = nullptr;
        long long value = strtoll(env, &end, 10);
        if (end != env && value > 0) {
            return static_cast<size_t>(value);
        }
    }
    return 2000000;
}

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void appendJsonString(std::string& out, const std::string& value) {
    out.push_back('"');
    for (char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out.push_back(c);
            }
        }
    }
    out.push_back('"');
}

}  // namespace

AwiRankStore::AwiRankStore(): cap_(eventCapFromEnv()) {}

AwiRankStore& AwiRankStore::instance() {
    static AwiRankStore store;
    return store;
}

void AwiRankStore::setEventCapacity(size_t cap) {
    std::lock_guard<std::mutex> lock(mu_);
    cap_ = cap > 0 ? cap : 1;
}

void AwiRankStore::appendEvent(std::string json_event, std::string request_id, std::string operation_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (events_.size() >= cap_) {
        events_.pop_front();
        ++dropped_;
    }
    events_.push_back(Event{std::move(json_event), std::move(request_id), std::move(operation_id)});
}

void AwiRankStore::armBarrier(std::string operation_id, std::vector<std::string> request_ids, int64_t context_tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& barrier : barriers_) {
        if (barrier.operation_id == operation_id) {
            barrier.request_ids = request_ids;
            barrier.context_tokens = context_tokens;
            barrier.armed_ns = nowNs();
            return;
        }
    }
    barriers_.push_back(Barrier{std::move(operation_id), std::move(request_ids), context_tokens, nowNs()});
}

bool AwiRankStore::isBarrierRequest(const std::string& request_id) const {
    if (request_id.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& barrier : barriers_) {
        for (const auto& armed : barrier.request_ids) {
            if (armed == request_id) {
                return true;
            }
        }
    }
    return false;
}

std::string AwiRankStore::collectJson(const std::string& operation_id) const {
    std::lock_guard<std::mutex> lock(mu_);

    std::vector<const std::string*> selected;
    selected.reserve(events_.size());
    if (operation_id.empty()) {
        for (const auto& event : events_) {
            selected.push_back(&event.json);
        }
    } else {
        const Barrier* barrier = nullptr;
        for (const auto& candidate : barriers_) {
            if (candidate.operation_id == operation_id) {
                barrier = &candidate;
                break;
            }
        }
        for (const auto& event : events_) {
            if (event.operation_id == operation_id) {
                selected.push_back(&event.json);
                continue;
            }
            if (barrier != nullptr && !event.request_id.empty()) {
                for (const auto& armed : barrier->request_ids) {
                    if (armed == event.request_id) {
                        selected.push_back(&event.json);
                        break;
                    }
                }
            }
        }
    }

    std::string out;
    out.reserve(selected.size() * 256 + 128);
    out += "{\"events\":[";
    for (size_t i = 0; i < selected.size(); ++i) {
        if (i) {
            out.push_back(',');
        }
        out += *selected[i];
    }
    out += "],\"barriers\":[";
    for (size_t i = 0; i < barriers_.size(); ++i) {
        if (i) {
            out.push_back(',');
        }
        const auto& barrier = barriers_[i];
        out += "{\"operation_id\":";
        appendJsonString(out, barrier.operation_id);
        out += ",\"context_tokens\":";
        out += std::to_string(barrier.context_tokens);
        out += ",\"armed_ns\":";
        out += std::to_string(barrier.armed_ns);
        out += ",\"request_ids\":[";
        for (size_t j = 0; j < barrier.request_ids.size(); ++j) {
            if (j) {
                out.push_back(',');
            }
            appendJsonString(out, barrier.request_ids[j]);
        }
        out += "]}";
    }
    out += "],\"dropped_events\":";
    out += std::to_string(dropped_);
    out += "}";
    return out;
}

}  // namespace rtp_llm
