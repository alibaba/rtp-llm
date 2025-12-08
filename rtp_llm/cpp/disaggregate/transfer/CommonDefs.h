#pragma once

#include <string>
#include <cstdint>

namespace rtp_llm {

struct CacheStoreServerWorker {
    std::string ip;
    uint32_t    port;
    uint32_t    rdma_port;
    CacheStoreServerWorker(const std::string& ip, uint32_t port, uint32_t rdma_port):
        ip(ip), port(port), rdma_port(rdma_port) {}
};

}  // namespace rtp_llm
