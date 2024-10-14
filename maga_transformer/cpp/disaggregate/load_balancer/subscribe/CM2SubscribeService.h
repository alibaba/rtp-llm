#pragma once

#include <shared_mutex>
#include "autil/Log.h"
#include "aios/apps/facility/cm2/cm_sub/cm_sub.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"

namespace rtp_llm {

class CM2SubscribeService: public SubscribeService {
public:
    CM2SubscribeService()  = default;
    ~CM2SubscribeService();

public:
    bool init(const CM2SubscribeServiceConfig& config);
    bool isReady() override;
    bool getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) override;

private:
    bool                                  inited_{false};
    std::shared_ptr<cm_sub::CMSubscriber> subscriber_;

private:
    AUTIL_LOG_DECLARE();
};

}  // namespace rtp_llm