#pragma once

#include <cstdint>
#include <memory>
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

}  // namespace detail

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
