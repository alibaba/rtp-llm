#pragma once

#include "autil/Log.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"

namespace rtp_llm {

class LocalSubscribeService: public SubscribeService {
public:
    LocalSubscribeService()  = default;
    ~LocalSubscribeService() = default;

public:
    bool init(const LocalSubscribeServiceConfig& config);
    bool isReady() override;
    bool getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) override;

private:
    bool                                         inited_{false};
    std::vector<std::shared_ptr<const TopoNode>> nodes_;

private:
    AUTIL_LOG_DECLARE();
};

}  // namespace rtp_llm