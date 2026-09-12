#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmClientKvcm.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "kvcm_client/kv_meta_object_client.h"

namespace rtp_llm {
namespace {

static_assert(kv_cache_manager::kKvMetaObjectClientApiVersion == 1,
              "RTP-LLM requires KVCM KVMeta object client API version 1");

std::string errorText(const char* operation, kv_cache_manager::ClientErrorCode code) {
    std::ostringstream oss;
    oss << "KVCM " << operation << " failed with client error " << static_cast<int>(code);
    return oss.str();
}

std::string exceptionText(const char* operation) {
    return std::string("KVCM ") + operation + " threw an exception";
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

struct ServiceBatch {
    std::vector<std::string>       keys;
    std::vector<uint64_t>          sizes;
    kv_cache_manager::BlockBuffers buffers;
};

std::string prepareObjectBatches(const std::vector<MMKvcmBuffer>& objects, std::vector<ServiceBatch>* batches) {
    batches->clear();
    batches->reserve((objects.size() + kMMKvcmMaxBatchItems - 1) / kMMKvcmMaxBatchItems);
    for (size_t begin = 0; begin < objects.size();) {
        const size_t end = nextMMKvcmBatchEnd(objects, begin);
        if (end == begin) {
            return "KVCM object batch cannot fit within service limits";
        }
        ServiceBatch batch;
        batch.keys.reserve(end - begin);
        batch.sizes.reserve(end - begin);
        batch.buffers.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            appendObject(objects[i], &batch.keys, &batch.sizes, &batch.buffers);
        }
        batches->push_back(std::move(batch));
        begin = end;
    }
    return {};
}

void prepareKeyBatches(const std::vector<std::string>& keys, std::vector<ServiceBatch>* batches) {
    batches->clear();
    batches->reserve((keys.size() + kMMKvcmMaxBatchItems - 1) / kMMKvcmMaxBatchItems);
    for (size_t begin = 0; begin < keys.size(); begin += kMMKvcmMaxBatchItems) {
        const size_t end = std::min(begin + kMMKvcmMaxBatchItems, keys.size());
        ServiceBatch batch;
        batch.keys.assign(keys.begin() + begin, keys.begin() + end);
        batches->push_back(std::move(batch));
    }
}

class MMKvcmClientImpl final: public MMKvcmClient {
public:
    MMKvcmClientImpl(std::unique_ptr<kv_cache_manager::KvMetaObjectClient> client,
                     uint64_t                                              max_object_bytes,
                     uint64_t                                              max_receipt_bytes):
        client_(std::move(client)), max_object_bytes_(max_object_bytes), max_receipt_bytes_(max_receipt_bytes) {}

    std::string save(const std::string& trace_id, const std::vector<MMKvcmBuffer>& objects) override {
        try {
            return saveImpl(trace_id, objects);
        } catch (...) {
            // The adapter is called across RTP worker boundaries whose callers
            // expect an error string, not a C++ exception. Provider-controlled
            // details must not escape into logs or RPC responses.
            return exceptionText("save");
        }
    }

    std::string
    load(const std::string& trace_id, const std::vector<MMKvcmBuffer>& objects, int64_t timeout_ms) override {
        try {
            return loadImpl(trace_id, objects, timeout_ms);
        } catch (...) {
            return exceptionText("load");
        }
    }

    std::string remove(const std::string& trace_id, const std::vector<std::string>& keys) override {
        try {
            return removeImpl(trace_id, keys);
        } catch (...) {
            return exceptionText("remove");
        }
    }

private:
    std::string saveImpl(const std::string& trace_id, const std::vector<MMKvcmBuffer>& objects) {
        if (const auto error = validateMMKvcmObjects(objects, max_object_bytes_, max_receipt_bytes_); !error.empty()) {
            return error;
        }
        // Prepare every key/size/iovec batch and both trace strings before the
        // first mutation. A later allocation or conversion failure can no
        // longer strand a prefix committed by an earlier service batch.
        std::vector<ServiceBatch> batches;
        if (const auto error = prepareObjectBatches(objects, &batches); !error.empty()) {
            return error;
        }
        const std::string                 save_trace            = trace_id + ":save";
        const std::string                 rollback_remove_trace = trace_id + ":rollback:remove";
        std::lock_guard<std::timed_mutex> lock(mutex_);
        for (size_t batch_index = 0; batch_index < batches.size(); ++batch_index) {
            const auto&                       batch = batches[batch_index];
            kv_cache_manager::ClientErrorCode code;
            try {
                code = client_->SaveObjects(save_trace, batch.keys, batch.sizes, batch.buffers);
            } catch (...) {
                // The native call may have committed any part of this batch
                // before throwing. UUID object keys make the full admitted
                // prefix safe to remove.
                const auto rollback = removeBatchesUnlocked(rollback_remove_trace, batches, batch_index + 1);
                auto       error    = exceptionText("save");
                if (!rollback.empty()) {
                    error += "; rollback outcome is uncertain: " + rollback;
                }
                return error;
            }
            if (code != kv_cache_manager::ER_OK) {
                // RTP generates fresh UUID keys for every receipt, so removing
                // prior batches and uncertain writes cannot delete an existing
                // application object.
                const auto rollback = removeBatchesUnlocked(rollback_remove_trace, batches, batch_index + 1);
                auto       error    = errorText("save", code);
                if (!rollback.empty()) {
                    error += "; rollback outcome is uncertain: " + rollback;
                }
                return error;
            }
        }
        return {};
    }

    std::string loadImpl(const std::string& trace_id, const std::vector<MMKvcmBuffer>& objects, int64_t timeout_ms) {
        if (const auto error = validateMMKvcmObjects(objects, max_object_bytes_, max_receipt_bytes_); !error.empty()) {
            return error;
        }
        if (timeout_ms <= 0) {
            return "KVCM load request deadline is exhausted";
        }
        const auto now = std::chrono::steady_clock::now();
        const auto max_timeout_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::time_point::max() - now)
                .count();
        const auto deadline = timeout_ms >= max_timeout_ms ? std::chrono::steady_clock::time_point::max()
                                                           : now + std::chrono::milliseconds(timeout_ms);
        std::vector<ServiceBatch> batches;
        if (const auto error = prepareObjectBatches(objects, &batches); !error.empty()) {
            return error;
        }
        const std::string                  load_trace = trace_id + ":load";
        std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
        if (!lock.try_lock_until(deadline)) {
            return "KVCM load request deadline expired waiting for the object client";
        }
        for (const auto& batch : batches) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return "KVCM load request deadline expired between object batches";
            }
            kv_cache_manager::ClientErrorCode code;
            try {
                code = client_->LoadObjects(load_trace, batch.keys, batch.sizes, batch.buffers);
            } catch (...) {
                return exceptionText("load");
            }
            if (code != kv_cache_manager::ER_OK) {
                return errorText("load", code);
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return "KVCM load request deadline expired during an object batch";
            }
        }
        return {};
    }

    std::string removeImpl(const std::string& trace_id, const std::vector<std::string>& keys) {
        if (keys.empty()) {
            return {};
        }
        if (keys.size() > kMMKvcmMaxObjectsPerReceipt) {
            return "KVCM remove exceeds the receipt object limit";
        }
        std::unordered_set<std::string> unique;
        unique.reserve(keys.size());
        for (const auto& key : keys) {
            if (key.empty() || key.size() > kMMKvcmMaxKeyBytes || !unique.insert(key).second) {
                return "KVCM remove keys must contain 1 to 512 bytes and be unique";
            }
        }
        std::vector<ServiceBatch> batches;
        prepareKeyBatches(keys, &batches);
        const std::string                 remove_trace = trace_id + ":remove";
        std::lock_guard<std::timed_mutex> lock(mutex_);
        return removeBatchesUnlocked(remove_trace, batches, batches.size());
    }

    std::string removeBatchesUnlocked(const std::string&               remove_trace,
                                      const std::vector<ServiceBatch>& batches,
                                      size_t                           batch_count) {
        bool                              saw_exception = false;
        kv_cache_manager::ClientErrorCode first_code    = kv_cache_manager::ER_OK;
        for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
            try {
                const auto code = client_->Remove(remove_trace, batches[batch_index].keys);
                if (code != kv_cache_manager::ER_OK && first_code == kv_cache_manager::ER_OK && !saw_exception) {
                    first_code = code;
                }
            } catch (...) {
                // Batches are disjoint UUID keys. Continue cleanup after one
                // provider exception. Defer all error-string construction
                // until every batch has been attempted so formatting failure
                // cannot strand the remaining committed prefix.
                if (first_code == kv_cache_manager::ER_OK) {
                    saw_exception = true;
                }
            }
        }
        if (first_code != kv_cache_manager::ER_OK) {
            return errorText("remove", first_code);
        }
        return saw_exception ? exceptionText("remove") : std::string{};
    }

    std::unique_ptr<kv_cache_manager::KvMetaObjectClient> client_;
    uint64_t                                              max_object_bytes_;
    uint64_t                                              max_receipt_bytes_;
    std::timed_mutex                                      mutex_;
};

}  // namespace

namespace detail {

std::shared_ptr<MMKvcmClient>
createMMKvcmClientAdapter(std::unique_ptr<kv_cache_manager::KvMetaObjectClient> client,
                          std::uint64_t                                         max_object_bytes,
                          std::uint64_t                                         max_receipt_bytes) {
    if (client == nullptr
        || kv_cache_manager::GetKvMetaObjectClientApiVersion()
               != kv_cache_manager::kKvMetaObjectClientApiVersion
        || max_object_bytes == 0 || max_object_bytes > kMMKvcmMaxObjectBytes
        || max_receipt_bytes < max_object_bytes
        || max_receipt_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return nullptr;
    }
    return std::make_shared<MMKvcmClientImpl>(std::move(client), max_object_bytes, max_receipt_bytes);
}

}  // namespace detail

bool hasMMKvcmImplementation() {
    return true;
}

std::shared_ptr<MMKvcmClient> createMMKvcmClient(const MMKvcmConfig& config) {
    if (!validateMMKvcmConfig(config).empty()) {
        return nullptr;
    }
    try {
        // Check the linked client library before Create constructs metadata,
        // transfer, or registered-memory state.  The adapter repeats this
        // check for dependency-injected clients used by tests and embedders.
        if (kv_cache_manager::GetKvMetaObjectClientApiVersion()
            != kv_cache_manager::kKvMetaObjectClientApiVersion) {
            return nullptr;
        }
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
        return detail::createMMKvcmClientAdapter(
            std::move(client), kvcm_config.max_object_bytes, static_cast<uint64_t>(config.max_receipt_bytes));
    } catch (...) {
        return nullptr;
    }
}

}  // namespace rtp_llm
