#pragma once

#include <optional>

#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class DistStorage {
public:
    enum StorageType {
        ST_LOCAL_MEM = 0,
        ST_3FS       = 1,
    };

    struct Iov {
        std::shared_ptr<void> data;
        size_t                len{0};
        bool                  gpu_mem{true};
        bool                  ignore{false};
    };

    struct Item {
        StorageType                        type = ST_LOCAL_MEM;
        std::string                        key;  // block direct key, 直接表示这个block
        std::vector<Iov>                   iovs;
        std::map<std::string, std::string> metas;  // block 关联的部署/模型等信息

        size_t size() const {
            size_t size = 0;
            for (const auto& iov : iovs) {
                size += iov.len;
            }
            return size;
        }
    };

public:
    virtual bool lookup(const DistStorage::Item& key) = 0;
    virtual bool get(const DistStorage::Item& item)   = 0;
    virtual bool put(const DistStorage::Item& item)   = 0;
    virtual bool del(const DistStorage::Item& item)   = 0;
};

struct DistStorageLocalMemInitParams {
    // TODO: not support for now
};

struct DistStorage3FSInitParams {
    bool   enable_async_write = true;
    size_t write_thread_num   = 4;
    size_t write_queue_size   = 1000;

    size_t read_iov_block_size  = 0;
    size_t read_iov_size        = 1ULL << 32;  // 4GB
    size_t write_iov_block_size = 1ULL << 20;  // 1MB
    size_t write_iov_size       = 1ULL << 32;  // 4GB

    std::string mountpoint{"/3fs/stage/3fs/"};
    std::string root_dir{"rtp_llm/"};
};

struct DistStorageManagerInitParams {
    std::optional<DistStorage3FSInitParams>      init_params_3fs;
    std::optional<DistStorageLocalMemInitParams> init_params_local_mem;
};

}  // namespace rtp_llm