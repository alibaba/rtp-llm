#include "LoraManager.h"
#include <chrono>

namespace rtp_llm {
namespace lora {

LoraManager::LoraManager(const int max_lora_model_size): max_lora_model_size_(max_lora_model_size) {}

rtp_llm::lora::LoraModelPtr LoraManager::getLora(int64_t lora_id) {
    std::unique_lock<std::mutex> scoped_lock(mutex_);
    auto                         it = lora_map_.find(lora_id);
    if (it == lora_map_.end()) {
        return nullptr;
    }
    return it->second;
}

rtp_llm::lora::LoraModelPtr LoraManager::getLora(const std::string& adapter_name) {
    std::unique_lock<std::mutex> scoped_lock(mutex_);
    auto                         it = adapter_map_.find(adapter_name);
    if (it != adapter_map_.end()) {
        auto lora = lora_map_.find(it->second);
        if (lora != lora_map_.end()) {
            return lora->second;
        }
    }
    return nullptr;
}

bool LoraManager::hasLora(const std::string& adapter_name) {
    std::unique_lock<std::mutex> scoped_lock(mutex_);
    auto                         it = adapter_map_.find(adapter_name);
    if (it == adapter_map_.end()) {
        return false;
    } else {
        return (lora_map_.find(it->second) != lora_map_.end());
    }
}

void LoraManager::addLora(const std::string&                        adapter_name,
                          const rtp_llm::lora::loraLayerWeightsMap& lora_a_weights,
                          const rtp_llm::lora::loraLayerWeightsMap& lora_b_weights) {
    std::unique_lock<std::mutex> scoped_lock(mutex_);
    auto                         lora_id = count_;
    RTP_LLM_CHECK_WITH_INFO((lora_id >= 0), "add lora need lora id[%ld] be greater than 0", lora_id);
    RTP_LLM_CHECK_WITH_INFO((lora_map_.find(lora_id) == lora_map_.end()),
                            "Lora id[%ld] is globally unique and cannot be added repeatedly",
                            lora_id);
    lora_map_[lora_id]         = std::make_shared<rtp_llm::lora::LoraModel>(lora_a_weights, lora_b_weights);
    adapter_map_[adapter_name] = lora_id;
    count_++;
}

void LoraManager::removeLora(const std::string& adapter_name) {
    int lora_id = -1;
    {
        std::unique_lock<std::mutex> scoped_lock(mutex_);
        auto                         it = adapter_map_.find(adapter_name);
        RTP_LLM_CHECK_WITH_INFO(
            it != adapter_map_.end(), "adapter name[%s] need exits when remove lora", adapter_name.c_str());
        lora_id = it->second;
        RTP_LLM_CHECK_WITH_INFO(
            (lora_map_.find(lora_id) != lora_map_.end()), "Lora id[%ld] need exits when remove lora", lora_id);
        rtp_llm::lora::LoraModelPtr resource = lora_map_[lora_id];
        // must remove adapter map before wait.
        adapter_map_.erase(adapter_name);
        // one for var resource, another for lora_map_
        cv_.wait(scoped_lock, [&resource] { return resource.use_count() == 2; });
    }
    {
        std::unique_lock<std::mutex> scoped_lock(mutex_);
        lora_map_.erase(lora_id);
    }
    return;
}

int LoraManager::getLoraId(const std::string& adapter_name) {
    std::unique_lock<std::mutex> scoped_lock(mutex_);
    auto                         it = adapter_map_.find(adapter_name);
    if (it == adapter_map_.end()) {
        return -1;
    } else {
        return it->second;
    }
}

rtp_llm::lora::LoraModelInputPtr LoraManager::makeLoraModelInput(rtp_llm::BufferPtr lora_ids,
                                                                 rtp_llm::BufferPtr lora_input_lengths) {
    if (lora_ids == nullptr || lora_input_lengths == nullptr) {
        return nullptr;
    }
    RTP_LLM_CHECK_WITH_INFO((lora_ids->dim() == 1 && lora_input_lengths->dim() == 1),
                            "lora_ids dim[%d] and lora_input_lengths dim[%d] must be equal to 1.",
                            lora_ids->dim(),
                            lora_input_lengths->dim());

    RTP_LLM_CHECK_WITH_INFO((lora_ids->shape()[0] == lora_input_lengths->shape()[0]),
                            "lora_ids [%d] and lora_input_lengths [%d] must has same batch_size.",
                            lora_ids->shape()[0],
                            lora_input_lengths->shape()[0]);

    size_t                                   batch_size = lora_ids->shape()[0];
    std::vector<rtp_llm::lora::LoraModelPtr> result(batch_size);
    int32_t*                                 lora_ids_ptr  = lora_ids->data<int32_t>();
    bool                                     use_same_lora = true;
    bool                                     has_lora      = false;
    for (int i = 0; i < batch_size; i++) {
        result[i]     = getLora(lora_ids_ptr[i]);
        has_lora      = result[i] || has_lora;
        use_same_lora = use_same_lora && (lora_ids_ptr[i] == lora_ids_ptr[0]);
    }
    if (!has_lora) {
        return nullptr;
    }
    return std::make_shared<rtp_llm::lora::LoraModelInput>(lora_input_lengths, result, use_same_lora);
}

std::optional<std::string> LoraManager::checkLoraInfoSize(const std::map<std::string, std::string>& lora_infos) const {
    bool err = max_lora_model_size_ != -1 && lora_infos.size() > max_lora_model_size_;
    if (!err) {
        return std::nullopt;
    } else {
        auto formatMapToString = [](const std::map<std::string, std::string>& map) {
            std::ostringstream oss;
            bool               first = true;
            for (const auto& pair : map) {
                if (!first) {
                    oss << ", ";
                }
                first = false;
                oss << "'" << pair.first << "': '" << pair.second << "'";
            }
            return oss.str();
        };
        std::ostringstream err;
        err << "lora_infos[{" << formatMapToString(lora_infos) << "}]'s size exceed MAX_LORA_MODEL_SIZE["
            << max_lora_model_size_ << "]";
        RTP_LLM_LOG_WARNING("%s", err.str().c_str());
        return err.str();
    }
}

}  // namespace lora
}  // namespace rtp_llm
