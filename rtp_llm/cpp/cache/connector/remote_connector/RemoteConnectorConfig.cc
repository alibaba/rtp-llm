#include "RemoteConnectorConfig.h"

using namespace autil::legacy;

namespace rtp_llm {
namespace {
DataStorageType DataStorageTypeFromString(const std::string& type) {
    if (type == "local") {
        return DataStorageType::DATA_STORAGE_TYPE_LOCAL;
    } else if (type == "3fs") {
        return DataStorageType::DATA_STORAGE_TYPE_3FS;
    } else if (type == "mooncake") {
        return DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
    } else if (type == "pace") {
        return DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
    } else if (type == "file") {
        return DataStorageType::DATA_STORAGE_TYPE_NFS;
    } else {
        return DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    }
}

std::string DataStorageTypeToString(const DataStorageType& type) {
    switch (type) {
        case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN:
            return "unknown";
        case DataStorageType::DATA_STORAGE_TYPE_LOCAL:
            return "local";
        case DataStorageType::DATA_STORAGE_TYPE_3FS:
            return "3fs";
        case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
            return "mooncake";
        case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
            return "pace";
        case DataStorageType::DATA_STORAGE_TYPE_NFS:
            return "file";
        default:
            return "unrecognized";
    }
}
}  // namespace

LocationSpecInfo::LocationSpecInfo(const std::string& name, int64_t size): name_(name), size_(size) {}

void LocationSpecInfo::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("name", name_);
    json.Jsonize("size", size_);
}

void SdkTimeoutConfig::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("put_timeout_ms", put_timeout_ms_, put_timeout_ms_);
    json.Jsonize("get_timeout_ms", get_timeout_ms_, get_timeout_ms_);
}

SdkBackendConfig::SdkBackendConfig(DataStorageType type): type_(type) {}

void SdkBackendConfig::Jsonize(Jsonizable::JsonWrapper& json) {
    if (json.GetMode() == FastJsonizableBase::Mode::TO_JSON) {
        json.Jsonize("type", DataStorageTypeToString(type_));
    } else {
        std::string type_str;
        json.Jsonize("type", type_str);
        type_ = DataStorageTypeFromString(type_str);
    }
    json.Jsonize("sdk_log_file_path", sdk_log_file_path_, "");
    json.Jsonize("sdk_log_level", sdk_log_level_, "ERROR");
}

Hf3fsSdkConfig::Hf3fsSdkConfig(): SdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_3FS) {}

void Hf3fsSdkConfig::Jsonize(Jsonizable::JsonWrapper& json) {
    SdkBackendConfig::Jsonize(json);
    json.Jsonize("enable_async_write", enable_async_write_, true);
    json.Jsonize("write_thread_num", write_thread_num_, write_thread_num_);
    json.Jsonize("write_queue_size", write_queue_size_, write_queue_size_);
    json.Jsonize("read_iov_block_size", read_iov_block_size_, read_iov_block_size_);
    json.Jsonize("read_iov_size", read_iov_size_, read_iov_size_);
    json.Jsonize("write_iov_block_size", write_iov_block_size_, write_iov_block_size_);
    json.Jsonize("write_iov_size", write_iov_size_, write_iov_size_);
}

MooncakeSdkConfig::MooncakeSdkConfig(): SdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE) {}

void MooncakeSdkConfig::Jsonize(Jsonizable::JsonWrapper& json) {
    SdkBackendConfig::Jsonize(json);
    json.Jsonize("location", location_, location_);
    json.Jsonize("put_replica_num", put_replica_num_, put_replica_num_);
}

TairMempoolSdkConfig::TairMempoolSdkConfig(): SdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL) {}

void TairMempoolSdkConfig::Jsonize(Jsonizable::JsonWrapper& json) {
    SdkBackendConfig::Jsonize(json);
}

NfsSdkConfig::NfsSdkConfig(): SdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_NFS) {}

void NfsSdkConfig::Jsonize(Jsonizable::JsonWrapper& json) {
    SdkBackendConfig::Jsonize(json);
}

void SdkWrapperConfig::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("thread_num", thread_num_, thread_num_);
    json.Jsonize("queue_size", queue_size_, queue_size_);
    if (json.GetMode() == FastJsonizableBase::Mode::TO_JSON) {
        json.Jsonize("sdk_backend_configs", sdk_backend_configs_);
    } else {
        std::vector<Any> sdk_backend_configs;
        json.Jsonize("sdk_backend_configs", sdk_backend_configs);
        for (const auto& config_any : sdk_backend_configs) {
            assert(json::IsJsonMap(config_any));
            const json::JsonMap&              config_json_map = *(AnyCast<json::JsonMap>(&config_any));
            const Any&                        type_any        = config_json_map.at("type");
            const std::string&                type_str        = *(AnyCast<std::string>(&type_any));
            DataStorageType                   type            = DataStorageTypeFromString(type_str);
            std::shared_ptr<SdkBackendConfig> sdk_backend_config;
            switch (type) {
                case DataStorageType::DATA_STORAGE_TYPE_LOCAL:
                    sdk_backend_config = std::make_shared<SdkBackendConfig>();
                    break;
                case DataStorageType::DATA_STORAGE_TYPE_3FS:
                    sdk_backend_config = std::make_shared<Hf3fsSdkConfig>();
                    break;
                case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
                    sdk_backend_config = std::make_shared<MooncakeSdkConfig>();
                    break;
                case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
                    sdk_backend_config = std::make_shared<TairMempoolSdkConfig>();
                    break;
                case DataStorageType::DATA_STORAGE_TYPE_NFS:
                    sdk_backend_config = std::make_shared<NfsSdkConfig>();
                    break;
                default:
                    AUTIL_LEGACY_THROW(NotJsonizableException, "invalid sdk_backend type : " + type_str);
                    break;
            }
            FromJson(*sdk_backend_config, config_any);
            sdk_backend_configs_.push_back(std::move(sdk_backend_config));
        }
    }
    json.Jsonize("timeout_config", timeout_config_, timeout_config_);
}

void ModelDeployment::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("model_name", model_name_);
    json.Jsonize("dtype", dtype_);
    json.Jsonize("use_mla", use_mla_, use_mla_);
    json.Jsonize("tp_size", tp_size_);
    json.Jsonize("dp_size", dp_size_, dp_size_);
    json.Jsonize("lora_name", lora_name_, lora_name_);
    json.Jsonize("pp_size", pp_size_, pp_size_);
    json.Jsonize("extra", extra_, extra_);
    json.Jsonize("user_data", user_data_, user_data_);
}

void MetaChannelConfig::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("retry_time", retry_time_, retry_time_);
    json.Jsonize("connection_timeout", connection_timeout_, connection_timeout_);
    json.Jsonize("call_timeout", call_timeout_, call_timeout_);
}

void RemoteConnectorConfig::Jsonize(Jsonizable::JsonWrapper& json) {
    json.Jsonize("enable_vipserver", enable_vipserver_, false);
    json.Jsonize("vipserver_domain", vipserver_domain_, "");
    json.Jsonize("instance_group", instance_group_);
    json.Jsonize("instance_id", instance_id_);
    json.Jsonize("block_size", block_size_);
    json.Jsonize("location_spec_infos", location_spec_info_map_);
    json.Jsonize("address", addresses_);
    json.Jsonize("meta_channel_config", meta_channel_config_, meta_channel_config_);
    json.Jsonize("sdk_config", sdk_wrapper_config_);
    json.Jsonize("model_deployment", model_deployment_);
    json.Jsonize("location_spec_groups", location_spec_groups_, {});
}
}  // namespace rtp_llm