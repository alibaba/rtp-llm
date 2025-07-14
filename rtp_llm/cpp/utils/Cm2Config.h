#pragma once

#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

class Cm2ClusterConfig: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("cm2_server_cluster_name", cluster_name);
        json.Jsonize("cm2_server_leader_path", zk_path);
        json.Jsonize("cm2_server_zookeeper_host", zk_host);
    }

public:
    std::string cluster_name;
    std::string zk_path;
    std::string zk_host;
};

}  // namespace rtp_llm
