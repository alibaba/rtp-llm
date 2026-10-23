#pragma once

#include "rtp_llm/cpp/utils/TimeUtil.h"

#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rtp_llm::p2p_internal {

// Far above any writeback RPC deadline, so an evicted client never has an RPC in flight.
constexpr int64_t kPeerClientIdleTtlMs = 10 * 60 * 1000;

// Returns call()'s pointer result; on an exception or a null result returns null and describes why in `failure`.
template<typename Call>
auto callForResult(Call&& call, std::string& failure) -> decltype(call()) {
    try {
        auto result = call();
        if (!result) {
            failure = "no result";
        }
        return result;
    } catch (const std::exception& error) {
        failure = error.what();
    } catch (...) {
        failure = "unknown exception";
    }
    return nullptr;
}

// One client per peer; a client nobody else holds is dropped once it has been idle past the TTL.
template<typename Key, typename Client>
class PeerClientCache {
public:
    explicit PeerClientCache(int64_t idle_ttl_ms = kPeerClientIdleTtlMs): idle_ttl_ms_(idle_ttl_ms) {}

    template<typename Create>
    std::shared_ptr<Client> get(const Key& key, Create&& create) {
        // Declared before the lock so evicted clients are destroyed unlocked: destruction joins their threads.
        std::vector<std::shared_ptr<Client>> evicted;
        std::lock_guard<std::mutex>          lock(mutex_);
        const auto                           now_ms = currentTimeMs();
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->second.client.use_count() == 1 && now_ms - it->second.last_used_ms >= idle_ttl_ms_) {
                evicted.push_back(std::move(it->second.client));
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            auto client = create();
            if (!client) {
                return nullptr;
            }
            it = entries_.emplace(key, Entry{std::move(client), now_ms}).first;
        }
        it->second.last_used_ms = now_ms;
        return it->second.client;
    }

private:
    struct Entry {
        std::shared_ptr<Client> client;
        int64_t                 last_used_ms = 0;
    };

    const int64_t        idle_ttl_ms_;
    std::mutex           mutex_;
    std::map<Key, Entry> entries_;
};

}  // namespace rtp_llm::p2p_internal
