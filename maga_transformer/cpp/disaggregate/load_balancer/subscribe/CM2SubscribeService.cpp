#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/CM2SubscribeService.h"

#include "aios/apps/facility/cm2/cm_basic/basic_struct/cm_central_sub.h"
#include "aios/apps/facility/cm2/cm_basic/basic_struct/cm_cluster_wrapper.h"
#include "aios/apps/facility/cm2/cm_basic/basic_struct/cm_node_wrapper.h"
#include "aios/apps/facility/cm2/cm_sub/config/subscriber_config.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, CM2SubscribeService);

CM2SubscribeService::~CM2SubscribeService() {
    if (subscriber_) {
        subscriber_->release();
        subscriber_.reset();
    }
}

bool CM2SubscribeService::init(const CM2SubscribeServiceConfig& config) {
    if (!config.validate()) {
        AUTIL_LOG(WARN,
                  "cm2 subscribe service init failed, config is invalid, config is [%s]",
                  autil::legacy::ToJsonString(config).c_str());
        return false;
    }

    cm_sub::SubscriberConfig* sub_config = new cm_sub::SubscriberConfig();
    sub_config->_zkServer                = config.zk_host;
    sub_config->_zkPath                  = config.zk_path;
    sub_config->_timeout                 = config.zk_timeout_ms;

    sub_config->_serverType   = cm_sub::FromZK;
    sub_config->_compressType = cm_basic::CT_SNAPPY;
    sub_config->_subType      = cm_basic::SubReqMsg::ST_PART;

    for (auto& cluster : config.clusters) {
        sub_config->_subClusterSet.insert(cluster);
    }

    subscriber_ = std::make_shared<cm_sub::CMSubscriber>();
    if (subscriber_->init(sub_config, nullptr) != 0) {
        AUTIL_LOG(WARN,
                  "cm2 subscribe service init failed, subscriber init failed, config is [%s]",
                  autil::legacy::ToJsonString(config).c_str());
        return false;
    }

    if (subscriber_->subscriber() != 0) {
        AUTIL_LOG(WARN,
                  "cm2 subscribe service init failed, subscriber subscribe failed, config is [%s]",
                  autil::legacy::ToJsonString(config).c_str());
        return false;
    }

    inited_ = true;  // cm sub subcribe will connect and process first response
    return true;
}

bool CM2SubscribeService::isReady() {
    return inited_;
}

bool CM2SubscribeService::getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) {
    if (!subscriber_) {
        AUTIL_LOG(WARN, "cm2 subscribe service get topo nodes failed, subscriber is null");
        return false;
    }

    auto cm_central = subscriber_->getCMCentral();
    if (!cm_central) {
        AUTIL_LOG(WARN, "cm2 subscribe service get topo nodes failed, cm central is null");
        return false;
    }

    // parse every time call getTopoNodes, may use cluster version to optimize
    auto clusters = cm_central->getAllCluster();
    for (auto& cluster : clusters) {
        auto nodes = cluster->getNodeList();
        for (auto& node : nodes) {
            uint32_t arpc_port = 0;
            for (int i = 0; i < node->getProtoPortSize(); i++) {
                if (node->getProtoType(i) == cm_basic::ProtocolType::PT_TCP) {
                    arpc_port = node->getProtoPort(i);
                    break;
                }
            }

            if (arpc_port == 0) {
                AUTIL_LOG(WARN,
                          "cm2 subscribe service get topo nodes failed, arpc port is 0, cluster is %s node spec is %s",
                          cluster->getClusterName().c_str(),
                          node->getNodeSpec().c_str());
                continue;
            }

            auto toponode = std::make_shared<TopoNode>(cluster->getClusterName(), node->getNodeIP(), arpc_port);
            topo_nodes.push_back(toponode);
        }
    }
    return true;
}

}  // namespace rtp_llm