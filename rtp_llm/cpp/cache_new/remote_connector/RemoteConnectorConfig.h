#pragma once

#include <vector>
#include <string>
#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

enum class DataStorageType : uint8_t {
    DATA_STORAGE_TYPE_UNKNOWN      = 0,
    DATA_STORAGE_TYPE_LOCAL        = 1,
    DATA_STORAGE_TYPE_3FS          = 2,
    DATA_STORAGE_TYPE_MOONCAKE     = 3,
    DATA_STORAGE_TYPE_TAIR_MEMPOOL = 4,
    DATA_STORAGE_TYPE_NFS          = 5,
};

class LocationSpecInfo: public autil::legacy::Jsonizable {
public:
    LocationSpecInfo() = default;
    LocationSpecInfo(const std::string& name, int64_t size);
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

private:
    std::string name_;
    int64_t     size_ = 0;
};

class SdkTimeoutConfig: public autil::legacy::Jsonizable {
public:
    SdkTimeoutConfig() = default;
    SdkTimeoutConfig(int put_timeout_ms, int get_timeout_ms):
        put_timeout_ms_(put_timeout_ms), get_timeout_ms_(get_timeout_ms) {}
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

private:
    int put_timeout_ms_{2000};
    int get_timeout_ms_{2000};
};

class SdkBackendConfig: public autil::legacy::Jsonizable {
public:
    SdkBackendConfig() = default;
    SdkBackendConfig(DataStorageType type);
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

protected:
    DataStorageType type_ = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    std::string     sdk_log_file_path_;
    std::string     sdk_log_level_;
};

class Hf3fsSdkConfig: public SdkBackendConfig {
public:
    Hf3fsSdkConfig();
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

private:
    bool   enable_async_write_{true};
    size_t write_thread_num_{4};
    size_t write_queue_size_{1000};

    size_t read_iov_block_size_{0};
    size_t read_iov_size_{1ULL << 32};         // 4GB
    size_t write_iov_block_size_{1ULL << 20};  // 1MB
    size_t write_iov_size_{1ULL << 32};        // 4GB
};

class MooncakeSdkConfig: public SdkBackendConfig {
public:
    MooncakeSdkConfig();
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

private:
    std::string location_{"*"};
    size_t      put_replica_num_{1};
};

class TairMempoolSdkConfig: public SdkBackendConfig {
public:
    TairMempoolSdkConfig();
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
};

class NfsSdkConfig: public SdkBackendConfig {
public:
    NfsSdkConfig();
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
};

class SdkWrapperConfig: public autil::legacy::Jsonizable {
public:
    SdkWrapperConfig() = default;
    SdkWrapperConfig(uint32_t thread_num, uint32_t queue_size, int put_timeout_ms, int get_timeout_ms):
        thread_num_(thread_num), queue_size_(queue_size), timeout_config_(put_timeout_ms, get_timeout_ms) {}
    void  Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    auto& sdk_backend_configs() {
        return sdk_backend_configs_;
    }

private:
    uint32_t                                       thread_num_{4};
    uint32_t                                       queue_size_{2000};
    std::vector<std::shared_ptr<SdkBackendConfig>> sdk_backend_configs_;
    SdkTimeoutConfig                               timeout_config_;
};

class ModelDeployment: public autil::legacy::Jsonizable {
public:
    ModelDeployment() = default;
    ModelDeployment(const std::string& model_name,
                    const std::string& dtype,
                    bool               use_mla,
                    int32_t            tp_size,
                    int32_t            dp_size,
                    const std::string& lora_name,
                    int32_t            pp_size,
                    const std::string& extra,
                    const std::string& user_data):
        model_name_(model_name),
        dtype_(dtype),
        use_mla_(use_mla),
        tp_size_(tp_size),
        dp_size_(dp_size),
        lora_name_(lora_name),
        pp_size_(pp_size),
        extra_(extra),
        user_data_(user_data) {}
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

private:
    std::string model_name_;
    std::string dtype_;
    bool        use_mla_ = false;
    int32_t     tp_size_ = 1;
    int32_t     dp_size_ = 1;
    std::string lora_name_;
    int32_t     pp_size_ = 1;
    std::string extra_;
    std::string user_data_;
};

class MetaChannelConfig: public autil::legacy::Jsonizable {
public:
    MetaChannelConfig() = default;
    MetaChannelConfig(uint32_t retry_time, uint32_t connection_timeout, uint32_t call_timeout):
        retry_time_(retry_time), connection_timeout_(connection_timeout), call_timeout_(call_timeout) {}
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

    inline uint32_t retry_time() const {
        return retry_time_;
    }

private:
    uint32_t retry_time_         = 3;
    uint32_t connection_timeout_ = 1000;  // ms
    uint32_t call_timeout_       = 100;   // ms
};

class RemoteConnectorConfig: public autil::legacy::Jsonizable {
public:
    using LocationSpecInfoMap = std::map<std::string, int64_t>;
    using LocationSpecGroups  = std::map<std::string, std::vector<std::string>>;
    enum class RoleType : uint8_t {
        UNKNOWN   = 0b00000000,
        WORKER    = 0b00000001,
        SCHEDULER = 0b00000010,
        HYBRID    = 0b00000011,
    };

    RemoteConnectorConfig() = default;
    RemoteConnectorConfig(bool                                        enable_vipserver,
                          const std::string&                          vipserver_domain,
                          int32_t                                     block_size,
                          const std::string&                          instance_group,
                          const std::string&                          instance_id,
                          const std::vector<std::string>&             addresses,
                          const std::shared_ptr<LocationSpecInfoMap>& location_spec_info_map,
                          const std::shared_ptr<MetaChannelConfig>&   meta_channel_config,
                          const std::shared_ptr<SdkWrapperConfig>&    sdk_wrapper_config,
                          const std::shared_ptr<LocationSpecGroups>&  location_spec_groups,
                          const ModelDeployment&                      model_deployment):
        enable_vipserver_(enable_vipserver),
        vipserver_domain_(vipserver_domain),
        block_size_(block_size),
        instance_group_(instance_group),
        instance_id_(instance_id),
        addresses_(addresses),
        location_spec_info_map_(location_spec_info_map),
        meta_channel_config_(meta_channel_config),
        sdk_wrapper_config_(sdk_wrapper_config),
        location_spec_groups_(location_spec_groups),
        model_deployment_(model_deployment) {}

    void        Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    inline bool enable_vipserver() const {
        return enable_vipserver_;
    }
    inline const std::string& vipserver_domain() const {
        return vipserver_domain_;
    }
    inline const std::vector<std::string>& addresses() const {
        return addresses_;
    }
    inline void set_addresses(const std::vector<std::string>& addresses) {
        addresses_ = addresses;
    }
    inline const auto& meta_channel_config() const {
        return meta_channel_config_;
    }

private:
    bool                                 enable_vipserver_ = false;
    std::string                          vipserver_domain_ = "";
    int32_t                              block_size_       = -1;
    std::string                          instance_group_;
    std::string                          instance_id_;
    std::vector<std::string>             addresses_;
    std::shared_ptr<LocationSpecInfoMap> location_spec_info_map_;
    std::shared_ptr<MetaChannelConfig>   meta_channel_config_;
    std::shared_ptr<SdkWrapperConfig>    sdk_wrapper_config_;
    std::shared_ptr<LocationSpecGroups>  location_spec_groups_;
    ModelDeployment                      model_deployment_;
};

using RemoteConnectorConfigPtr = std::shared_ptr<RemoteConnectorConfig>;

}  // namespace rtp_llm
