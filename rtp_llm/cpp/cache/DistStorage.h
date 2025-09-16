#pragma once

#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

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
    std::string toString() const {
        std::ostringstream oss;
        oss << "enable_async_write: " << enable_async_write << ", write_thread_num: " << write_thread_num
            << ", write_queue_size: " << write_queue_size << ", read_iov_block_size: " << read_iov_block_size
            << ", read_iov_size: " << read_iov_size << ", write_iov_block_size: " << write_iov_block_size
            << ", write_iov_size: " << write_iov_size << ", read_timeout_ms: " << read_timeout_ms
            << ", write_timeout_ms: " << write_timeout_ms << ", mountpoint: " << mountpoint
            << ", root_dir: " << root_dir << ", file_cache_capacity: " << file_cache_capacity;
        return oss.str();
    }

    bool   enable_async_write = true;
    size_t write_thread_num   = 4;
    size_t write_queue_size   = 1000;

    size_t read_iov_block_size  = 0;
    size_t read_iov_size        = 1ULL << 32;  // 4GB
    size_t write_iov_block_size = 1ULL << 20;  // 1MB
    size_t write_iov_size       = 1ULL << 32;  // 4GB

    size_t read_timeout_ms  = 1000;
    size_t write_timeout_ms = 2000;

    std::string mountpoint{"/3fs/stage/3fs/"};
    std::string root_dir{"rtp_llm/"};

    size_t file_cache_capacity{20000};
};

struct DistStorageManagerInitParams {
    std::string toString() const {
        std::ostringstream oss;
        oss << "lookup_timeout_ms: " << lookup_timeout_ms << ", get_timeout_ms: " << get_timeout_ms
            << ", put_timeout_ms: " << put_timeout_ms << ", del_timeout_ms: " << del_timeout_ms;
        return oss.str();
    }

    std::optional<DistStorage3FSInitParams>      init_params_3fs;
    std::optional<DistStorageLocalMemInitParams> init_params_local_mem;

    size_t lookup_timeout_ms{1000};
    size_t get_timeout_ms{2000};
    size_t put_timeout_ms{2000};
    size_t del_timeout_ms{1000};
};

}  // namespace rtp_llm