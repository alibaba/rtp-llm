#include "maga_transformer/cpp/disaggregate/master/cluster/PrefillCluster.h"
#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {
namespace rtp_llm_master {
PrefillCluster::PrefillCluster(const SubscribeServiceConfig& config): subscribe_config_(config) {}

bool PrefillCluster::init() {
    subscribe_service_manager_.reset(new SubscribeServiceManager());
    if (!subscribe_service_manager_->init(subscribe_config_)) {
        FT_LOG_WARNING("PrefillCluster init failed, subscribe service manager init failed");
        return false;
    }
    FT_LOG_INFO("PrefillCluster init done");
    return true;
}

void PrefillCluster::updateWorkerStatusLoop() {
    
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm
