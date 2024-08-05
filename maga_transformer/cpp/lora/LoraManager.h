#pragma once
#include "src/fastertransformer/devices/LoraWeights.h"
#include <condition_variable>

namespace ft = fastertransformer;
namespace rtp_llm {

namespace lora {


class LoraManager {
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<int64_t, ft::lora::LoraModelPtr> lora_map_;

public:

    LoraManager() = default;
    ~LoraManager() = default;
    LoraManager(LoraManager& other) = delete;
    LoraManager(LoraManager&& other) = delete;

    void addLora(int64_t lora_id,
                 const ft::lora::loraLayerWeightsMap& lora_a_weights,
                 const ft::lora::loraLayerWeightsMap& lora_b_weights);

    void removeLora(int64_t lora_id);

    ft::lora::LoraModelPtr getLora(int64_t lora_id);

    bool hasLora(int64_t lora_id);

    void releaseSignal() {
        cv_.notify_all();
    }


    // helper function
    ft::lora::LoraModelInputPtr makeLoraModelInput(ft::BufferPtr lora_ids, ft::BufferPtr lora_input_lengths);

};


struct LoraResourceGuard {
    std::shared_ptr<LoraManager> lora_manager_;
    ft::lora::LoraModelPtr lora_ptr_;

    LoraResourceGuard(std::shared_ptr<LoraManager> lora_manager, int lora_id) {
        lora_manager_ = lora_manager;
        lora_ptr_ = lora_manager_->getLora(lora_id);
    }

    ~LoraResourceGuard() {
        lora_ptr_ = nullptr;
        lora_manager_->releaseSignal();
    }
};

} // namespace lora

}  // namespace rtp_llm