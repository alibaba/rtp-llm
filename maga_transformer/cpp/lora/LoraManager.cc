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
    std::unique_lock<std::mutex> scoped_lock(mutex_);
    lora_map_[lora_id] = std::make_shared<ft::lora::LoraModel>(lora_a_weights, lora_b_weights);
}

void LoraManager::removeLora(int64_t lora_id) {
    FT_CHECK_WITH_INFO(hasLora(lora_id),
        "Lora id[%ld] need exits when remove lora", lora_id);
    {
        std::unique_lock<std::mutex> scoped_lock(mutex_);
        ft::lora::LoraModelPtr resource = lora_map_[lora_id];
        // one for var resource, another for lora_map_
        cv_.wait(scoped_lock, [&resource]{ return resource.use_count() == 2; });
    }

    {
        std::unique_lock<std::mutex> scoped_lock(mutex_);
        lora_map_.erase(lora_id);
    }
    return;
}

ft::lora::LoraModelPtr LoraManager::getLora(int64_t lora_id) {
    std::unique_lock<std::mutex> scoped_lock(mutex_);
    auto it = lora_map_.find(lora_id);
    if (it == lora_map_.end()) {
        return nullptr;
    }
    return it->second;
}

bool LoraManager::hasLora(int64_t lora_id) {
    std::unique_lock<std::mutex> scoped_lock(mutex_);
    return (lora_map_.find(lora_id) != lora_map_.end());
}


ft::lora::LoraModelInputPtr LoraManager::makeLoraModelInput(ft::BufferPtr lora_ids,
                                                            ft::BufferPtr lora_input_lengths)
{
    if (lora_ids == nullptr || lora_input_lengths == nullptr) {
        return nullptr;
    }
    FT_CHECK_WITH_INFO((lora_ids->dim() == 1 && lora_input_lengths->dim() == 1),
        "lora_ids dim[%d] and lora_input_lengths dim[%d] must be equal to 1.",
        lora_ids->dim(), lora_input_lengths->dim());

    FT_CHECK_WITH_INFO((lora_ids->shape()[0] == lora_input_lengths->shape()[0]),
        "lora_ids [%d] and lora_input_lengths [%d] must has same batch_size.",
        lora_ids->shape()[0], lora_input_lengths->shape()[0]);

    size_t batch_size = lora_ids->shape()[0];
    std::vector<ft::lora::LoraModelPtr> result(batch_size);
    int32_t* lora_ids_ptr = lora_ids->data<int32_t>();
    for (int i = 0; i < batch_size; i++) {
        result[i] = getLora(lora_ids_ptr[i]);
    }
    return std::make_shared<ft::lora::LoraModelInput>(lora_input_lengths, result);
}

}  // namespace lora
}  // namespace rtp_llm