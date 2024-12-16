#pragma once
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/disaggregate/master/scheduler/Struct.h"
#include "maga_transformer/cpp/disaggregate/master/cluster/PrefillWorkerInfo.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"

namespace rtp_llm {
namespace rtp_llm_master {

class PrefillCluster {
public:
    absl::StatusOr<MachineInfo> getBestWorker(int input_length, std::vector<int64_t> token_block_hash);

protected:
    void updateWorkerStatusLoop();
    void updateWorkerListLoop();

private:
    std::unordered_map<std::string, PrefillWorkerInfo> worker_map_;
    std::unique_ptr<rtp_llm::SubscribeServiceManager>  subscribe_service_manager_;
};

}  // namespace rtp_llm_master

}  // namespace rtp_llm
