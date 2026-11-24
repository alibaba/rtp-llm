#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockState.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/remote_mock_control.pb.h"

namespace rtp_llm {
struct RemoteMockServiceOptions {
    RemoteMockLimits limits;
    uint32_t         reads_per_connection   = 8;
    uint32_t         writes_per_connection  = 2;
    uint32_t         max_callback_endpoints = 128;
    uint32_t         max_timeout_ms         = 60000;
    std::string      event_path;
    uint64_t         max_event_bytes = 8ULL << 20;
};
struct RemoteMockServiceStats {
    bool event_capture_failed = false;
    // Response bytes handed to ARPC; no claim of TCP delivery acknowledgement.
    uint64_t read_payload_bytes = 0;
    // Actual decoded reverse-response contents, including failed shape/CRC/session checks.
    uint64_t write_payload_bytes = 0;
    uint64_t reverse_rpcs        = 0;
    uint64_t failed_operations   = 0;
    uint64_t read_admitted_ns    = 0;
    uint64_t read_payload_ns     = 0;
    uint64_t write_stored_ns     = 0;
    uint64_t write_ack_ns        = 0;
};
// Own this service until shutdown() has drained, and keep TcpServer alive until
// then. The server must subsequently stop before this object is destroyed.
class RemoteMockService: public KvCacheStoreService {
public:
    explicit RemoteMockService(RemoteMockServiceOptions options = {});
    ~RemoteMockService() override;
    bool                                   init();
    void                                   shutdown();
    remote_mock::RemoteMockControlService* controlService();
    RemoteMockServiceStats                 stats() const;
    void                                   load(google::protobuf::RpcController*,
                                                const CacheLoadRequest*,
                                                CacheLoadResponse*,
                                                google::protobuf::Closure*) override;
    void                                   transfer(google::protobuf::RpcController*,
                                                    const CacheTransferRequest*,
                                                    CacheTransferResponse*,
                                                    google::protobuf::Closure*) override;
    void                                   blockRead(google::protobuf::RpcController*,
                                                     const BlockReadRequest*,
                                                     BlockReadResponse*,
                                                     google::protobuf::Closure*) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace rtp_llm
