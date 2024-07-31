#include "LoraManager.h"

#include <chrono>

namespace ft = fastertransformer;
namespace rtp_llm {
namespace lora {

void LoraManager::addLora(int64_t lora_id,
                          const ft::lora::loraLayerWeightsMap& lora_a_weights,
                          const ft::lora::loraLayerWeightsMap& lora_b_weights)
{
    FT_CHECK_WITH_INFO((lora_id >= 0), "add lora need lora id[%ld] be greater than 0", lora_id);
    FT_CHECK_WITH_INFO(!hasLora(lora_id),
        "Lora id[%ld] is globally unique and cannot be added repeatedly", lora_id);
    std::unique_lock<std::shared_mutex> scoped_lock(mutex_);
    lora_map_[lora_id] = LoraResource({true, std::make_shared<ft::lora::LoraModel>(lora_a_weights, lora_b_weights)});
}

void LoraManager::removeLora(int64_t lora_id) {
    FT_CHECK_WITH_INFO(hasLora(lora_id),
        "Lora id[%ld] need exits when remove lora", lora_id);
    ft::lora::LoraModelPtr resource = nullptr;
    bool is_timeout = true;
    {
        std::unique_lock<std::shared_mutex> scoped_lock(mutex_);
        FT_CHECK_WITH_INFO(lora_map_[lora_id].alive_,
            "Lora id[%ld] need alive when remove lora", lora_id);
        lora_map_[lora_id].alive_ = false;
        resource = lora_map_[lora_id].resource_;

    }
    {
        std::unique_lock<std::mutex> scoped_lock(remove_mutex_);
        if (wait_remove_timeout_ == 0) {
            // one for var resource, another for lora_map_
            cv_.wait(scoped_lock, [&resource]{ return resource.use_count() == 2; });
            is_timeout = false;
        } else {
            auto end_time = std::chrono::high_resolution_clock::now() + std::chrono::seconds(wait_remove_timeout_);
            is_timeout = !cv_.wait_until(scoped_lock, end_time, [&resource]{ return resource.use_count() == 2; });
        }
    }

    {
        std::unique_lock<std::shared_mutex> scoped_lock(mutex_);
        if (is_timeout) {
            FT_CHECK_WITH_INFO(false, "remove lora[%d] timeout.", lora_id);
        } else {
            lora_map_.erase(lora_id);
        }
    }
    return;
}

ft::lora::LoraModelPtr LoraManager::getLora(int64_t lora_id) {
    std::shared_lock<std::shared_mutex> scoped_lock(mutex_);
    auto it = lora_map_.find(lora_id);
    if (it == lora_map_.end()) {
        return nullptr;
    }
    return it->second.resource_;
}

bool LoraManager::hasLora(int64_t lora_id) {
    std::shared_lock<std::shared_mutex> scoped_lock(mutex_);
    return (lora_map_.find(lora_id) != lora_map_.end());
}

bool LoraManager::isLoraAlive(int64_t lora_id) {
    std::shared_lock<std::shared_mutex> scoped_lock(mutex_);
    auto it = lora_map_.find(lora_id);
    if (it == lora_map_.end()) {
        return false;
    }
    if (it->second.alive_) {
        return true;
    } else {
        return false;
    }
}

// std::vector<ft::LoraResource> LoraManager::getLoraResource(ft::BufferPtr lora_ids) {
//     if (lora_ids == nullptr) {
//         return std::vector<ft::LoraResource>();
//     }
//     size_t batch_size = lora_ids->shape()[0];
//     std::vector<ft::LoraResource> result(batch_size);
//     int32_t* lora_ids_ptr = lora_ids->data<int32_t>();
//     for (int i = 0; i < batch_size; i++) {
//         result[i] = getLora(lora_ids_ptr[i]);
//     }
//     return result;
// }

}  // namespace lora
}  // namespace rtp_llm