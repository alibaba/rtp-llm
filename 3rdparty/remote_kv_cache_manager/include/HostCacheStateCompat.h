#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kvcm_client/meta_client.h"

namespace rtp_llm {
namespace kvcm {

template<typename Match, typename = void>
struct HasGlobalHostCacheMatch: std::false_type {};

template<typename Match>
struct HasGlobalHostCacheMatch<Match, std::void_t<decltype(std::declval<const Match&>().global)>>: std::true_type {};

// The legacy RTP request selects hosts for a P2P-only metadata estimate.
// New SDKs require explicit backends; an empty list would silently disable it.
template<typename Client>
auto getSdkHostCacheState(Client&                              client,
                         const std::string&                    trace_id,
                         kv_cache_manager::QueryType           query_type,
                         const std::vector<int64_t>&            keys,
                         const std::vector<std::string>&       medium,
                         int32_t                              p2p_host_count) {
    if constexpr (std::is_invocable_v<decltype(&Client::GetHostCacheState),
                                      Client&,
                                      const std::string&,
                                      kv_cache_manager::QueryType,
                                      const std::vector<int64_t>&,
                                      const std::vector<std::string>&,
                                      int32_t,
                                      const std::vector<kv_cache_manager::StorageType>&>) {
        return client.GetHostCacheState(trace_id,
                                        query_type,
                                        keys,
                                        medium,
                                        p2p_host_count,
                                        {kv_cache_manager::StorageType::ST_EVENT_REPORT_L2});
    } else {
        return client.GetHostCacheState(trace_id, query_type, keys, medium, p2p_host_count);
    }
}

}  // namespace kvcm
}  // namespace rtp_llm
