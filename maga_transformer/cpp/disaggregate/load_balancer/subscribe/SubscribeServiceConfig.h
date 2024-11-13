#pragma once

#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

class NacosSubscribeServiceConfig : public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    bool validate() const;

public:
    std::string server_host;
    std::vector<std::string> clusters;
};

class CM2SubscribeServiceConfig: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    bool validate() const;

public:
    std::string              zk_host;
    std::string              zk_path;
    uint32_t                 zk_timeout_ms{10 * 1000};
    std::vector<std::string> clusters;
};

class LocalNodeJsonize: public autil::legacy::Jsonizable {
public:
    LocalNodeJsonize() = default;
    LocalNodeJsonize(const std::string& biz, const std::string& ip, uint32_t arpc_port);
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    bool validate() const;

public:
    std::string biz;
    std::string ip;
    uint32_t    arpc_port;
};

class LocalSubscribeServiceConfig: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    bool validate() const;

public:
    std::vector<LocalNodeJsonize> nodes;
};

class SubscribeServiceConfig: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    bool validate() const;

public:
    std::vector<CM2SubscribeServiceConfig>   cm2_configs;
    std::vector<LocalSubscribeServiceConfig> local_configs;
    std::vector<NacosSubscribeServiceConfig> nacos_configs;
};

}  // namespace rtp_llm