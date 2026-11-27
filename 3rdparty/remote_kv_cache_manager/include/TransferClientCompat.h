#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "kvcm_client/transfer_client.h"

namespace rtp_llm::kvcm {
namespace detail {

template<typename Client, typename = void>
struct HasTransferDeadline: std::false_type {};

template<typename Client>
struct HasTransferDeadline<Client,
                           std::void_t<decltype(&Client::LoadKvCachesWithDeadline),
                                       decltype(&Client::SaveKvCachesWithDeadline)>>: std::true_type {};

template<typename Client, typename Trace, typename = void>
struct HasLateReadSafety: std::false_type {};

template<typename Client, typename Trace>
struct HasLateReadSafety<Client, Trace, std::void_t<decltype(&Client::QueryReadBufferSafety),
                                                   decltype(&Trace::read_execution_id)>>: std::true_type {};

template<typename Client, typename = void>
struct HasDiscardReadRecovery: std::false_type {};

template<typename Client>
struct HasDiscardReadRecovery<Client, std::void_t<decltype(&Client::DiscardReadRecovery)>>: std::true_type {};

template<typename Trace, typename = void>
struct HasReadRecoveryCancellation: std::false_type {};

template<typename Trace>
struct HasReadRecoveryCancellation<Trace, std::void_t<decltype(&Trace::read_recovery_cancelled)>>: std::true_type {};

}  // namespace detail

inline constexpr bool kSupportsReadRecoveryDiscard =
    detail::HasDiscardReadRecovery<kv_cache_manager::TransferClient>::value &&
    detail::HasLateReadSafety<kv_cache_manager::TransferClient, kv_cache_manager::TransferTraceInfo>::value;

inline constexpr bool kSupportsReadRecoveryCancellation =
    kSupportsReadRecoveryDiscard && detail::HasReadRecoveryCancellation<kv_cache_manager::TransferTraceInfo>::value;

template<typename Trace, typename Client = kv_cache_manager::TransferClient>
void setSdkReadExecutionId(std::shared_ptr<Trace>&                       trace,
                           const std::string&                            trace_id,
                           const std::shared_ptr<std::atomic<bool>>&     read_recovery_cancelled) {
    if constexpr (detail::HasLateReadSafety<Client, Trace>::value) {
        if (!trace)
            trace = std::make_shared<Trace>();
        trace->read_execution_id = trace_id;
        if constexpr (detail::HasReadRecoveryCancellation<Trace>::value)
            trace->read_recovery_cancelled = read_recovery_cancelled;
    }
}

template<typename Client>
bool querySdkReadBufferSafety(Client& client, const std::string& trace_id, int64_t deadline_ms) {
    if constexpr (detail::HasLateReadSafety<Client, kv_cache_manager::TransferTraceInfo>::value)
        return client.QueryReadBufferSafety(trace_id, deadline_ms);
    return false;
}

template<typename Client>
void discardSdkReadRecovery(Client& client, const std::string& trace_id) {
    if constexpr (detail::HasDiscardReadRecovery<Client>::value)
        client.DiscardReadRecovery(trace_id);
}

template<typename Client>
kv_cache_manager::ClientErrorCode
loadSdkKvCaches(Client&                                                     client,
                const kv_cache_manager::UriStrVec&                          uris,
                const kv_cache_manager::BlockBuffers&                       buffers,
                int64_t                                                     deadline_ms,
                const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace) {
    if constexpr (detail::HasTransferDeadline<Client>::value) {
        if (deadline_ms > 0)
            return client.LoadKvCachesWithDeadline(uris, buffers, deadline_ms, trace);
    }
    return client.LoadKvCaches(uris, buffers, trace);
}

template<typename Client>
std::pair<kv_cache_manager::ClientErrorCode, kv_cache_manager::UriStrVec>
saveSdkKvCaches(Client&                                                     client,
                const kv_cache_manager::UriStrVec&                          uris,
                const kv_cache_manager::BlockBuffers&                       buffers,
                int64_t                                                     deadline_ms,
                const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace) {
    if constexpr (detail::HasTransferDeadline<Client>::value) {
        if (deadline_ms > 0)
            return client.SaveKvCachesWithDeadline(uris, buffers, deadline_ms, trace);
    }
    return client.SaveKvCaches(uris, buffers, trace);
}

}  // namespace rtp_llm::kvcm
