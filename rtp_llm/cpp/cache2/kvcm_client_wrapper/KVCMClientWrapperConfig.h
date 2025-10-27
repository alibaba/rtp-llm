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
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

private:
    size_t                                         thread_num_{4};
    size_t                                         queue_size_{2000};
    std::vector<std::shared_ptr<SdkBackendConfig>> sdk_backend_configs_;
    SdkTimeoutConfig                               timeout_config_;
};

class ModelDeployment: public autil::legacy::Jsonizable {
public:
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
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

    inline uint32_t retry_time() const {
        return retry_time_;
    }

private:
    uint32_t retry_time_         = 3;
    uint32_t connection_timeout_ = 1000;  // ms
    uint32_t call_timeout_       = 100;   // ms
};

class KVCMClientWrapperConfig: public autil::legacy::Jsonizable {
public:
    using LocationSpecInfoMap = std::map<std::string, int64_t>;
    enum class RoleType : uint8_t {
        UNKNOWN   = 0b00000000,
        WORKER    = 0b00000001,
        SCHEDULER = 0b00000010,
        HYBRID    = 0b00000011,
    };
    void        Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    inline void set_addresses(const std::vector<std::string>& addresses) {
        addresses_ = addresses;
    }
    inline bool enable_vipserver() const {
        return enable_vipserver_;
    }
    inline const std::string& vipserver_domain() const {
        return vipserver_domain_;
    }
    inline const std::vector<std::string>& addresses() const {
        return addresses_;
    }
    inline const MetaChannelConfig meta_channel_config() const {
        return meta_channel_config_;
    }

private:
    bool                              enable_vipserver_    = false;
    std::string                       vipserver_domain_    = "";
    int32_t                           block_size_          = -1;
    int32_t                           byte_size_per_block_ = -1;
    std::string                       instance_group_;
    std::string                       instance_id_;
    std::string                       self_location_spec_name_;
    LocationSpecInfoMap               location_spec_info_map_;
    std::vector<std::string>          addresses_;
    MetaChannelConfig                 meta_channel_config_;
    std::shared_ptr<SdkWrapperConfig> sdk_wrapper_config_;
    ModelDeployment                   model_deployment_;
};
}  // namespace rtp_llm
