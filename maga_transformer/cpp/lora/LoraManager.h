#pragma once
#include "src/fastertransformer/devices/LoraWeights.h"
#include <condition_variable>

namespace ft = fastertransformer;
namespace rtp_llm {

namespace lora {

struct LoraResource {
    bool alive_;
    ft::lora::LoraModelPtr resource_;
};

class LoraManager {
private:
    std::shared_mutex mutex_;
    std::mutex remove_mutex_;
    std::condition_variable cv_;
    std::unordered_map<int64_t, LoraResource> lora_map_;
    size_t wait_remove_timeout_ = 0;

public:

    LoraManager() = default;
    LoraManager(int wait_remove_timeout) {
        wait_remove_timeout_ = wait_remove_timeout;
    };
    ~LoraManager() = default;
    LoraManager(LoraManager& other) = delete;
    LoraManager(LoraManager&& other) = delete;

    void addLora(int64_t lora_id,
                 const ft::lora::loraLayerWeightsMap& lora_a_weights,
                 const ft::lora::loraLayerWeightsMap& lora_b_weights);

    void removeLora(int64_t lora_id);

    ft::lora::LoraModelPtr getLora(int64_t lora_id);

    bool hasLora(int64_t lora_id);

    bool isLoraAlive(int64_t lora_id);

    void releaseSignal() {
        cv_.notify_all();
    }

    // std::vector<ft::LoraResource>  getLoraResource(ft::BufferPtr lora_ids);

};

} // namespace lora

}  // namespace rtp_llm