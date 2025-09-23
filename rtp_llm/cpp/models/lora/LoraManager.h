#pragma once

#include "rtp_llm/cpp/devices/LoraWeights.h"
#include <condition_variable>

namespace rtp_llm {

namespace lora {

class LoraManager {
private:
    std::mutex                                               mutex_;
    std::condition_variable                                  cv_;
    std::unordered_map<int64_t, rtp_llm::lora::LoraModelPtr> lora_map_;
    std::unordered_map<std::string, int64_t>                 adapter_map_;
    int64_t                                                  count_               = 0;
    int32_t                                                  max_lora_model_size_ = -1;

public:
    LoraManager(const int max_lora_model_size = -1);
    virtual ~LoraManager()           = default;
    LoraManager(LoraManager& other)  = delete;
    LoraManager(LoraManager&& other) = delete;

    // virtual for test
    virtual void addLora(const std::string&                        adapter_name,
                         const rtp_llm::lora::loraLayerWeightsMap& lora_a_weights,
                         const rtp_llm::lora::loraLayerWeightsMap& lora_b_weights);
    virtual void removeLora(const std::string& adapter_name);

    int getLoraId(const std::string& adapter_name);

    rtp_llm::lora::LoraModelPtr getLora(int64_t lora_id);

    rtp_llm::lora::LoraModelPtr getLora(const std::string& adapter_name);

    bool hasLora(const std::string& adapter_name);

    void releaseSignal() {
        cv_.notify_all();
    }

    // helper function
    rtp_llm::lora::LoraModelInputPtr makeLoraModelInput(rtp_llm::BufferPtr lora_ids,
                                                        rtp_llm::BufferPtr lora_input_lengths);

    std::optional<std::string> checkLoraInfoSize(const std::map<std::string, std::string>& lora_infos) const;
};

struct LoraResourceGuard {
    std::shared_ptr<LoraManager> lora_manager_;
    rtp_llm::lora::LoraModelPtr  lora_ptr_;

    LoraResourceGuard(std::shared_ptr<LoraManager> lora_manager, const std::string& adapter_name) {
        lora_manager_ = lora_manager;
        lora_ptr_     = lora_manager_->getLora(adapter_name);
    }

    ~LoraResourceGuard() {
        lora_ptr_ = nullptr;
        lora_manager_->releaseSignal();
    }
};

}  // namespace lora

}  // namespace rtp_llm
