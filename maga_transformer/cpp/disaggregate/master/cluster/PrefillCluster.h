#pragma once
#include "absl/status/statusor.h"
#include "autil/LoopThread.h"
#include "maga_transformer/cpp/disaggregate/master/scheduler/Struct.h"
#include "maga_transformer/cpp/http_server/http_client/SimpleHttpClient.h"
#include "maga_transformer/cpp/disaggregate/master/cluster/PrefillWorkerInfo.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"

namespace rtp_llm {
namespace rtp_llm_master {

class PrefillCluster {
public:
    PrefillCluster(const SubscribeServiceConfig& config);
    bool                        init();
    absl::StatusOr<MachineInfo> getBestWorker(int input_length, std::vector<int64_t> token_block_hash);

protected:
    void updateWorkerStatusLoop();

private:
    std::unordered_map<std::string, PrefillWorkerInfo> worker_map_;
    SubscribeServiceConfig                             subscribe_config_;
    std::unique_ptr<rtp_llm::SubscribeServiceManager>  subscribe_service_manager_;
    std::shared_ptr<http_server::SimpleHttpClient>     http_client_;
    autil::LoopThreadPtr                               update_worker_thread_;
};

}  // namespace rtp_llm_master

}  // namespace rtp_llm
