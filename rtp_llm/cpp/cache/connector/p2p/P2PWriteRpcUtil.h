#pragma once

#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PNotification.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rtp_llm::p2p_internal {

inline int64_t writebackSteadyDeadline(int64_t wall_deadline_ms) {
    const auto remaining = std::max<int64_t>(0, wall_deadline_ms - currentTimeMs());
    const auto now       = P2PNotification::nowMs();
    return now + std::min(remaining, std::numeric_limits<int64_t>::max() - now);
}

inline int64_t writebackLocalDeadline(int64_t timeout_ms) {
    return timeout_ms > 0 && timeout_ms <= std::numeric_limits<int32_t>::max() ? currentTimeMs() + timeout_ms : 0;
}

// Idle retention window measured monotonically after all handles and in-flight RPCs finish.
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

namespace detail {

template<typename Client, typename = void>
struct ClientActivityTraits {
    static bool hasInflightRpcs(const Client&) {
        return false;
    }
    static int64_t lastActivityMs(const Client&) {
        return 0;
    }
};

template<typename Client>
struct ClientActivityTraits<Client,
                            std::void_t<decltype(std::declval<const Client&>().hasInflightRpcs()),
                                        decltype(std::declval<const Client&>().lastActivityMs())>> {
    static bool hasInflightRpcs(const Client& client) {
        return client.hasInflightRpcs();
    }
    static int64_t lastActivityMs(const Client& client) {
        return client.lastActivityMs();
    }
};

}  // namespace detail

// One client per peer; evicted only when no caller holds it, no async RPCs remain in flight,
// and monotonic idle time since last activity reaches the TTL.
template<typename Key, typename Client>
class PeerClientCache {
public:
    explicit PeerClientCache(int64_t idle_ttl_ms = kPeerClientIdleTtlMs): idle_ttl_ms_(idle_ttl_ms) {}

    template<typename Create>
    std::shared_ptr<Client> get(const Key& key, Create&& create) {
        // Declared before the lock so evicted clients are destroyed unlocked: destruction joins their threads.
        std::vector<std::shared_ptr<Client>> evicted;
        std::lock_guard<std::mutex>          lock(mutex_);
        const auto                           now_ms = P2PNotification::nowMs();
        for (auto it = entries_.begin(); it != entries_.end();) {
            const bool busy = it->second.client.use_count() > 1
                              || detail::ClientActivityTraits<Client>::hasInflightRpcs(*it->second.client);
            const int64_t last_active_ms =
                std::max(it->second.last_used_ms,
                         detail::ClientActivityTraits<Client>::lastActivityMs(*it->second.client));
            if (!busy && now_ms - last_active_ms >= idle_ttl_ms_) {
                evicted.push_back(std::move(it->second.client));
                it = entries_.erase(it);
            } else {
                if (busy) {
                    it->second.last_used_ms = now_ms;
                }
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
