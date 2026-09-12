#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmClient.h"

#include <cstddef>
#include <limits>
#include <unordered_set>

namespace rtp_llm {

std::size_t nextMMKvcmBatchEnd(const std::vector<MMKvcmBuffer>& objects, std::size_t begin) {
    std::size_t end         = begin;
    uint64_t    batch_bytes = 0;
    while (end < objects.size() && end - begin < kMMKvcmMaxBatchItems
           && objects[end].nbytes <= kMMKvcmMaxBatchBytes - batch_bytes) {
        batch_bytes += objects[end].nbytes;
        ++end;
    }
    return end;
}

std::string
validateMMKvcmObjects(const std::vector<MMKvcmBuffer>& objects, uint64_t max_object_bytes, uint64_t max_receipt_bytes) {
    if (max_object_bytes == 0 || max_object_bytes > kMMKvcmMaxObjectBytes
        || max_receipt_bytes < max_object_bytes
        || max_receipt_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return "KVCM object byte limits are invalid";
    }
    if (objects.empty()) {
        return "KVCM object batch is empty";
    }
    if (objects.size() > kMMKvcmMaxObjectsPerReceipt) {
        return "KVCM object batch exceeds the receipt object limit";
    }

    std::unordered_set<std::string> keys;
    keys.reserve(objects.size());
    uint64_t total_bytes = 0;
    for (const auto& object : objects) {
        if (object.key.empty() || object.key.size() > kMMKvcmMaxKeyBytes || !keys.insert(object.key).second) {
            return "KVCM object keys must contain 1 to 512 bytes and be unique";
        }
        const auto address = reinterpret_cast<std::uintptr_t>(object.data);
        if (object.data == nullptr || object.nbytes == 0 || object.nbytes > max_object_bytes
            || object.nbytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
            || object.nbytes > std::numeric_limits<std::uintptr_t>::max() - address) {
            return "KVCM object buffer has an invalid size or address";
        }
        if (object.nbytes > max_receipt_bytes - total_bytes) {
            return "KVCM object batch exceeds the receipt byte limit";
        }
        total_bytes += object.nbytes;
    }
    return {};
}

std::string validateMMKvcmConfig(const MMKvcmConfig& config) {
    if (config.addresses.empty() || config.addresses.size() > 64) {
        return "KVCM KVMeta address count must be between 1 and 64";
    }
    std::unordered_set<std::string> unique_addresses;
    for (const auto& address : config.addresses) {
        if (address.empty() || address.size() > 1024) {
            return "KVCM KVMeta address must contain 1 to 1024 bytes";
        }
        if (!unique_addresses.insert(address).second) {
            return "KVCM KVMeta addresses contain a duplicate";
        }
    }
    if (config.instance_id.empty() || config.instance_id.size() > kMMKvcmMaxInstanceIdBytes) {
        return "KVCM instance_id must contain 1 to 512 bytes";
    }
    if (config.instance_group.empty() || config.instance_group.size() > kMMKvcmMaxInstanceGroupBytes) {
        return "KVCM instance_group must contain 1 to 512 bytes";
    }
    if (config.user_data.size() > kMMKvcmMaxUserDataBytes) {
        return "KVCM user_data exceeds 65536 bytes";
    }
    if (config.transfer_client_config.empty()) {
        return "KVCM transfer_client_config is empty";
    }
    if (config.call_timeout_ms == 0 || config.call_timeout_ms > 600000 || config.write_timeout_seconds <= 0
        || config.write_timeout_seconds > kMMKvcmMaxWriteTimeoutSecs || config.object_gc_timeout_ms <= 0
        || config.max_object_bytes <= 0 || config.max_receipt_bytes <= 0) {
        return "KVCM timeouts and byte limits are outside their valid ranges";
    }
    if (static_cast<uint64_t>(config.max_object_bytes) > kMMKvcmMaxObjectBytes) {
        return "KVCM max_object_bytes exceeds the KVMeta service limit";
    }
    if (static_cast<uint64_t>(config.max_receipt_bytes) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return "KVCM max_receipt_bytes exceeds the local address space";
    }
    if (config.max_receipt_bytes < config.max_object_bytes) {
        return "KVCM max_receipt_bytes is smaller than max_object_bytes";
    }
    return {};
}

}  // namespace rtp_llm
