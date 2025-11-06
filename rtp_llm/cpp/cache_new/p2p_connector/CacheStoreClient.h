#pragma once

#include "rtp_llm/cpp/cache_new/p2p_connector/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/proto/service.pb.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/LoadContext.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpClient.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpServer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreClientService.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CommonDefs.h"

namespace rtp_llm {

class CacheStoreClientClosure: public google::protobuf::Closure {
public:
    CacheStoreClientClosure(const std::shared_ptr<CacheLoadRequest>&  cache_load_request,
                            const std::shared_ptr<CacheLoadResponse>& cache_load_response,
                            arpc::ANetRPCController*                  controller,
                            const std::shared_ptr<LoadContext>&       load_context);
    ~CacheStoreClientClosure() {
        if (controller_) {
            delete controller_;
        }
    };

public:
    void Run() override;

private:
    std::shared_ptr<CacheLoadRequest>  cache_load_request_;
    std::shared_ptr<CacheLoadResponse> cache_load_response_;
    arpc::ANetRPCController*           controller_ = nullptr;
    std::shared_ptr<LoadContext>       load_context_;
};

class CacheStoreClient {
public:
    CacheStoreClient(const std::shared_ptr<TcpClient>& tcp_client, const std::shared_ptr<TcpServer>& tcp_server);
    ~CacheStoreClient();

public:
    bool init();

    std::vector<CacheStoreServerWorker> getPeerWorkerInfo(const std::string& ip, uint32_t port);

    std::shared_ptr<LoadContext> asyncLoad(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                           int64_t                                               timeout_ms,
                                           const std::string&                                    ip,
                                           uint32_t                                              port,
                                           int                                                   partition_count,
                                           int                                                   partition_id);

private:
    int64_t generateContextId();
    bool    generateCacheLoadRequest(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                     int64_t                                               deadline_ms,
                                     int64_t                                               context_id,
                                     int                                                   partition_count,
                                     int                                                   partition_id,
                                     const std::shared_ptr<CacheLoadRequest>&              cache_load_request);

private:
    std::shared_ptr<TcpClient> tcp_client_;
    std::shared_ptr<TcpServer> tcp_server_;

    std::shared_ptr<LoadContextStore> load_context_store_;

    std::unique_ptr<CacheStoreClientService> cache_store_client_service_;
};

}  // namespace rtp_llm