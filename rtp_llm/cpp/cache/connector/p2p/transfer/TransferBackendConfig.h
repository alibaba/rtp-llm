#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm {
namespace transfer {

enum class TransferBackend : uint8_t {
    kTcp,
    kBarexRdma,
    kMooncake,
};

struct MooncakeTransferEngineInitConfig {
    std::string metadata_conn_string = "";
    std::string local_server_name    = "";
    std::string ip_or_host_name      = "";
    std::string transport            = "tcp";
    uint16_t    rpc_port             = 12345;
};

struct MooncakeBackendConfig {
    MooncakeTransferEngineInitConfig classic;
    std::string                      location           = "*";
    bool                             remote_accessible  = true;
    bool                             update_metadata    = true;
    int64_t                          control_plane_port = 0;
};

struct TransferBackendConfig {
    bool                 cache_store_rdma_mode               = false;
    bool                 cache_store_mooncake_mode           = false;
    MooncakeBackendConfig mooncake;
    int64_t              rdma_transfer_wait_timeout_ms       = 180 * 1000;
    int                  messager_io_thread_count            = 2;
    int                  messager_worker_thread_count        = 16;
    int                  rdma_max_block_pairs_per_connection = 0;
    int64_t              cache_store_listen_port             = 0;
    int                  cache_store_tcp_anet_rpc_thread_num = 3;
    int                  cache_store_tcp_anet_rpc_queue_num  = 100;

    TransferBackend resolveBackend() const {
        enum class BackendSelection : uint8_t {
            kTcp,
            kBarexRdma,
            kMooncake,
        };

        BackendSelection selection = BackendSelection::kTcp;
        if (cache_store_mooncake_mode) {
            selection = BackendSelection::kMooncake;
        } else if (cache_store_rdma_mode) {
            selection = BackendSelection::kBarexRdma;
        }

        switch (selection) {
            case BackendSelection::kTcp:
                return TransferBackend::kTcp;
            case BackendSelection::kBarexRdma:
                return TransferBackend::kBarexRdma;
            case BackendSelection::kMooncake:
                return TransferBackend::kMooncake;
        }

        return TransferBackend::kTcp;
    }
};

}  // namespace transfer
}  // namespace rtp_llm
