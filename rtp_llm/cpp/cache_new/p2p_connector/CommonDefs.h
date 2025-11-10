#pragma once

#include "rtp_llm/cpp/cache_new/p2p_connector/proto/service.pb.h"

namespace rtp_llm {
namespace cache_store {

struct CacheStoreServerWorker {
    std::string ip;
    uint32_t    port;
    uint32_t    rdma_port;
    CacheStoreServerWorker(const std::string& ip, uint32_t port, uint32_t rdma_port):
        ip(ip), port(port), rdma_port(rdma_port) {}
};

}  // namespace cache_store
}  // namespace rtp_llm