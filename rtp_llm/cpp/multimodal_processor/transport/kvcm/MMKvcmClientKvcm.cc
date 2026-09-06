#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmClient.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "kvcm_client/kv_meta_object_client.h"

namespace rtp_llm {
namespace {

std::string errorText(const char* operation, kv_cache_manager::ClientErrorCode code) {
    std::ostringstream oss;
    oss << "KVCM " << operation << " failed with client error " << static_cast<int>(code);
    return oss.str();
}

void appendObject(const MMKvcmBuffer&             object,
                  std::vector<std::string>*       keys,
                  std::vector<uint64_t>*          sizes,
                  kv_cache_manager::BlockBuffers* buffers) {
    keys->push_back(object.key);
    sizes->push_back(object.nbytes);
    kv_cache_manager::BlockBuffer block;
    kv_cache_manager::Iov         iov;
    iov.type = object.is_cuda ? kv_cache_manager::MemoryType::GPU : kv_cache_manager::MemoryType::CPU;
    iov.base = object.data;
    iov.size = static_cast<size_t>(object.nbytes);
    block.iovs.push_back(iov);
    buffers->push_back(std::move(block));
}

class MMKvcmClientImpl final: public MMKvcmClient {
public:
    MMKvcmClientImpl(std::unique_ptr<kv_cache_manager::KvMetaObjectClient> client,
                     uint64_t                                              max_object_bytes,
                     uint64_t                                              max_receipt_bytes):
        client_(std::move(client)), max_object_bytes_(max_object_bytes), max_receipt_bytes_(max_receipt_bytes) {}

    std::string save(const std::string& trace_id, const std::vector<MMKvcmBuffer>& objects) override {
        if (const auto error = validateMMKvcmObjects(objects, max_object_bytes_, max_receipt_bytes_); !error.empty()) {
            return error;
        }
        std::lock_guard<std::timed_mutex> lock(mutex_);
        for (size_t begin = 0; begin < objects.size();) {
            const size_t                   end = nextMMKvcmBatchEnd(objects, begin);
            std::vector<std::string>       keys;
            std::vector<uint64_t>          sizes;
            kv_cache_manager::BlockBuffers buffers;
            keys.reserve(end - begin);
            sizes.reserve(end - begin);
            buffers.reserve(end - begin);
            for (size_t i = begin; i < end; ++i) {
                appendObject(objects[i], &keys, &sizes, &buffers);
            }
            const auto code = client_->SaveObjects(trace_id + ":save", keys, sizes, buffers);
            if (code != kv_cache_manager::ER_OK) {
                // RTP generates fresh UUID keys for every receipt, so removing
                // prior batches and uncertain writes cannot delete an existing
                // application object.
                std::vector<std::string> rollback_keys;
                rollback_keys.reserve(end);
                for (size_t i = 0; i < end; ++i) {
                    rollback_keys.push_back(objects[i].key);
                }
                const auto rollback = removeUnlocked(trace_id + ":rollback", rollback_keys);
                auto       error    = errorText("save", code);
                if (!rollback.empty()) {
                    error += "; rollback outcome is uncertain: " + rollback;
                }
                return error;
            }
            begin = end;
        }
        return {};
    }

    std::string
    load(const std::string& trace_id, const std::vector<MMKvcmBuffer>& objects, int64_t timeout_ms) override {
        if (const auto error = validateMMKvcmObjects(objects, max_object_bytes_, max_receipt_bytes_); !error.empty()) {
            return error;
        }
        if (timeout_ms <= 0) {
            return "KVCM load request deadline is exhausted";
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
        if (!lock.try_lock_until(deadline)) {
            return "KVCM load request deadline expired waiting for the object client";
        }
        for (size_t begin = 0; begin < objects.size();) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return "KVCM load request deadline expired between object batches";
            }
            const size_t                   end = nextMMKvcmBatchEnd(objects, begin);
            std::vector<std::string>       keys;
            std::vector<uint64_t>          sizes;
            kv_cache_manager::BlockBuffers buffers;
            keys.reserve(end - begin);
            sizes.reserve(end - begin);
            buffers.reserve(end - begin);
            for (size_t i = begin; i < end; ++i) {
                appendObject(objects[i], &keys, &sizes, &buffers);
            }
            const auto code = client_->LoadObjects(trace_id + ":load", keys, sizes, buffers);
            if (code != kv_cache_manager::ER_OK) {
                return errorText("load", code);
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return "KVCM load request deadline expired during an object batch";
            }
            begin = end;
        }
        return {};
    }

    std::string remove(const std::string& trace_id, const std::vector<std::string>& keys) override {
        if (keys.empty()) {
            return {};
        }
        std::unordered_set<std::string> unique;
        unique.reserve(keys.size());
        for (const auto& key : keys) {
            if (key.empty() || key.size() > kMMKvcmMaxKeyBytes || !unique.insert(key).second) {
                return "KVCM remove keys must contain 1 to 512 bytes and be unique";
            }
        }
        std::lock_guard<std::timed_mutex> lock(mutex_);
        return removeUnlocked(trace_id, keys);
    }

private:
    std::string removeUnlocked(const std::string& trace_id, const std::vector<std::string>& keys) {
        std::string first_error;
        for (size_t begin = 0; begin < keys.size(); begin += kMMKvcmMaxBatchItems) {
            const size_t                   end = std::min(begin + kMMKvcmMaxBatchItems, keys.size());
            const std::vector<std::string> batch(keys.begin() + begin, keys.begin() + end);
            const auto                     code = client_->Remove(trace_id + ":remove", batch);
            if (code != kv_cache_manager::ER_OK && first_error.empty()) {
                first_error = errorText("remove", code);
            }
        }
        return first_error;
    }

    std::unique_ptr<kv_cache_manager::KvMetaObjectClient> client_;
    uint64_t                                              max_object_bytes_;
    uint64_t                                              max_receipt_bytes_;
    std::timed_mutex                                      mutex_;
};

}  // namespace

bool hasMMKvcmImplementation() {
    return true;
}

std::shared_ptr<MMKvcmClient> createMMKvcmClient(const MMKvcmConfig& config) {
    if (!validateMMKvcmConfig(config).empty()) {
        return nullptr;
    }
    try {
        kv_cache_manager::KvMetaObjectClientConfig kvcm_config;
        kvcm_config.metadata.addresses                           = config.addresses;
        kvcm_config.metadata.instance_id                         = config.instance_id;
        kvcm_config.metadata.call_timeout_ms                     = config.call_timeout_ms;
        kvcm_config.instance_group                               = config.instance_group;
        kvcm_config.user_data                                    = config.user_data;
        kvcm_config.transfer_client_config                       = config.transfer_client_config;
        kvcm_config.transfer_init_params.role_type               = kv_cache_manager::RoleType::WORKER;
        kvcm_config.transfer_init_params.self_location_spec_name = "value";
        kvcm_config.max_object_bytes                             = static_cast<uint64_t>(config.max_object_bytes);
        kvcm_config.write_timeout_seconds                        = config.write_timeout_seconds;
        auto [code, client] = kv_cache_manager::KvMetaObjectClient::Create("rtp-mm-kvcm-init", kvcm_config);
        if (code != kv_cache_manager::ER_OK || client == nullptr) {
            return nullptr;
        }
        return std::make_shared<MMKvcmClientImpl>(
            std::move(client), kvcm_config.max_object_bytes, static_cast<uint64_t>(config.max_receipt_bytes));
    } catch (...) {
        return nullptr;
    }
}

}  // namespace rtp_llm
