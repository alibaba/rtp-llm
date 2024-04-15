#pragma once

#include <assert.h>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/Query.h"

namespace rtp_llm {

enum class PrefixType {
    PromptTuning = 0,
    PTuningV2    = 1,
    KVCacheReuse = 2,
    NoPrefix     = 3
};

struct PrefixParams {
    PrefixParams() {}
    PrefixParams(PrefixType _prefix_type, size_t _prefix_length,
                 const std::vector<int>& _block_cache, std::optional<torch::Tensor> _prefix_tensor, const std::vector<int>& _prefix_prompt) {
        prefix_type = _prefix_type;
        prefix_length = _prefix_length;
        block_cache = _block_cache;
        prefix_tensor = _prefix_tensor;
        prefix_prompt = _prefix_prompt;
    }
    
    PrefixType                      prefix_type;
    size_t                          prefix_length;
    std::vector<int>                block_cache;
    // TODO(xinfei.sxf) 区分好 prefix tensor和tokens id
    std::optional<torch::Tensor>    prefix_tensor;
    std::vector<int>                prefix_prompt;
};

struct PrefixInfo {
    bool ptuning = false;
    bool count_length = true;
    bool count_prefix_length = true;
    std::optional<torch::Tensor> prefix_tensors;
    std::vector<int> prefix_prompt;
};

class PtuningBase {
public:
    PrefixType prefix_type_;

    virtual std::tuple<PrefixType, std::optional<torch::Tensor>, std::vector<int>> getPrefixParams(const GenerateConfig& generate_config) = 0;
    virtual std::tuple<bool, std::vector<int>, int> getBlockIndice(int blockNum, const GenerateConfig& generate_config) = 0;
    virtual size_t calcPrefixBlockNum(const GenerateConfig& generate_config) = 0;

    PrefixInfo getPtuningInfo(const GenerateConfig& generate_config) {
        auto [prefix_type, prefix_tensors, prefix_prompt] = getPrefixParams(generate_config);
        return PrefixInfo{
            .ptuning = true,
            .count_length = prefix_type_ == PrefixType::PromptTuning,
            .count_prefix_length = prefix_type_ != PrefixType::PTuningV2,
            .prefix_tensors = prefix_tensors,
            .prefix_prompt = prefix_prompt
        };
    }
};

class Ptuning : public PtuningBase {
public:    
    Ptuning(CacheManager& cache_manager,
            const PrefixParams& prefix_params, bool insert_resident_cache = false)
    : prefix_params_(prefix_params),
      cache_manager_(cache_manager),
      prefix_block_indice_(prefix_params_.block_cache),
      prefix_additional_block_(-1)
    {
        prefix_type_ = prefix_params_.prefix_type;
        maybeInsertPrefixCache(insert_resident_cache);
        if (prefix_params_.prefix_length % cache_manager.cacheConfig().seq_size_per_block != 0) {
            prefix_additional_block_ = prefix_block_indice_.back();
            prefix_block_indice_.pop_back();
        }
    }

    size_t calcPrefixBlockNum(const GenerateConfig& generate_config) override {
        return prefix_block_indice_.size();
    }

    std::tuple<PrefixType, std::optional<torch::Tensor>, std::vector<int>> getPrefixParams(const GenerateConfig& generate_config) override {
        return {prefix_type_, prefix_params_.prefix_tensor, prefix_params_.prefix_prompt};
    }

    std::tuple<bool, std::vector<int>, int> getBlockIndice(int blockNum, const GenerateConfig& generate_config) override {
        auto [success, block_indice] = cache_manager_.mallocIndex(blockNum);
        if (!success) {
            return {false, {}, 0};
        }
        if (prefix_additional_block_ > 0) {
            auto [success2, additional_block] = cache_manager_.mallocIndex(1);
            if (!success2) {
                return {false, {}, 0};
            }
            cache_manager_.blockCopy(prefix_additional_block_, additional_block[0]);
            block_indice.insert(block_indice.end(), additional_block.begin(), additional_block.end());
        }
        block_indice.insert(block_indice.begin(), prefix_block_indice_.begin(), prefix_block_indice_.end());
        return {true, block_indice, prefix_params_.prefix_length};
    }

private:
    void maybeInsertPrefixCache(bool insertResidentCache) {
        if (insertResidentCache) {
            cache_manager_.insertResidentCache(prefix_params_.block_cache, prefix_params_.prefix_prompt);
        }
    }

    PrefixParams     prefix_params_;
    CacheManager&    cache_manager_;
    std::vector<int> prefix_block_indice_;
    int              prefix_additional_block_;
};

class MultiTaskPtuning: public PtuningBase {
public:
    MultiTaskPtuning(CacheManager& cache_manager, const std::unordered_map<int, PrefixParams>& prefix_params_map) 
    : cache_manager_(cache_manager), prefix_type_(PrefixType::PromptTuning) {
        for (const auto& item : prefix_params_map) {
            int id = item.first;
            const PrefixParams& params = item.second;
            ptunings_[id] = std::make_unique<Ptuning>(cache_manager, params, true);
        }
    }

    std::tuple<bool, std::vector<int>, int> getBlockIndice(int blockNum, const GenerateConfig& generate_config) override {
        auto task_id = generate_config.task_id;
        if (task_id != std::nullopt) {
            auto it = ptunings_.find(task_id.value());
            if (it == ptunings_.end()) {
                auto [success, block_indices] = cache_manager_.mallocIndex(blockNum);
                return {success, block_indices, 0};
            }
            return it->second->getBlockIndice(blockNum, generate_config);
        } else {
            auto [success, block_indices] = cache_manager_.mallocIndex(blockNum);
            return {success, block_indices, 0};
        }
    }

    std::tuple<PrefixType, std::optional<torch::Tensor>, std::vector<int>>
    getPrefixParams(const GenerateConfig& generate_config) override {
        auto task_id = generate_config.task_id;
        if (task_id != std::nullopt) {
            auto it = ptunings_.find(task_id.value());
            if (it == ptunings_.end()) {
                return {PrefixType::NoPrefix, torch::Tensor(), {}};
            }
            return it->second->getPrefixParams(generate_config);
        } else {
            return {PrefixType::NoPrefix, torch::Tensor(), {}};
        }
    }

    size_t calcPrefixBlockNum(const GenerateConfig& generate_config) override {
        auto task_id = generate_config.task_id;
        if (task_id != std::nullopt) {
            auto it = ptunings_.find(task_id.value());
            if (it == ptunings_.end()) {
                return 0;
            }
            return it->second->calcPrefixBlockNum(generate_config);
        } else {
            return 0;
        }
    }

private:
    CacheManager&                                     cache_manager_;
    std::unordered_map<int, std::unique_ptr<Ptuning>> ptunings_;
    PrefixType                                        prefix_type_;
};

}  // namespace rtp_llm
