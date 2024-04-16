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
    PrefixType                   prefixType;
    int                          prefixLength;
    std::vector<int>             blockCache;
    std::optional<torch::Tensor> prefixTensor;
};

struct PrefixInfo {
    bool                         ptuning           = false;
    bool                         countLength       = true;
    bool                         countPrefixLength = true;
    std::optional<torch::Tensor> prefixTensors;
};

class PtuningBase {
public:
    PrefixType prefixType_;

    virtual std::tuple<PrefixType, std::optional<torch::Tensor>>
                                              getPrefixParams(const GenerateConfig& generateConfig)              = 0;
    virtual std::tuple<std::vector<int>, int> getBlockIndice(int blockNum, const GenerateConfig& generateConfig) = 0;
    virtual size_t                            calcPrefixBlockNum(const GenerateConfig& generateConfig)           = 0;

    PrefixInfo getPtuningInfo(const GenerateConfig& generateConfig) {
        auto [prefixType, prefixTensors] = getPrefixParams(generateConfig);
        return PrefixInfo{.ptuning           = true,
                          .countLength       = prefixType_ == PrefixType::PromptTuning,
                          .countPrefixLength = prefixType_ != PrefixType::PTuningV2,
                          .prefixTensors     = prefixTensors};
    }
};

class Ptuning: public PtuningBase {
public:
    Ptuning(CacheManager& cacheManager, const PrefixParams& prefixParams, bool insertResidentCache = false):
        prefixParams_(prefixParams),
        cacheManager_(cacheManager),
        prefixBlockIndice_(prefixParams_.blockCache),
        prefixAdditionalBlock_(-1) {
        prefixType_ = prefixParams_.prefixType;
        maybeInsertPrefixCache(insertResidentCache);
        if (prefixParams_.prefixLength % cacheManager.cacheConfig().seq_size_per_block != 0) {
            prefixAdditionalBlock_ = prefixBlockIndice_.back();
            prefixBlockIndice_.pop_back();
        }
    }

    size_t calcPrefixBlockNum(const GenerateConfig& generateConfig) override {
        return prefixBlockIndice_.size();
    }

    std::tuple<PrefixType, std::optional<torch::Tensor>>
    getPrefixParams(const GenerateConfig& generateConfig) override {
        return {prefixType_, prefixParams_.prefixTensor};
    }

    std::tuple<std::vector<int>, int> getBlockIndice(int blockNum, const GenerateConfig& generateConfig) override {
        std::vector<int> blockIndice = cacheManager_.mallocIndex(blockNum);
        if (prefixAdditionalBlock_ > 0) {
            std::vector<int> additionalBlock = cacheManager_.mallocIndex(1);
            cacheManager_.blockCopy(prefixAdditionalBlock_, additionalBlock[0]);
            blockIndice.insert(blockIndice.end(), additionalBlock.begin(), additionalBlock.end());
        }
        blockIndice.insert(blockIndice.begin(), prefixBlockIndice_.begin(), prefixBlockIndice_.end());
        return {blockIndice, prefixParams_.prefixLength};
    }

private:
    void maybeInsertPrefixCache(bool insertResidentCache) {
        if (insertResidentCache) {
            assert(prefixParams_.prefixTensor.has_value());
            // TODO(xinfei.sxf) convert prefixParams_.prefixTensor to std::vector<int>& token_ids
            std::vector<int> token_ids;
            cacheManager_.insertResidentCache(prefixParams_.blockCache, token_ids);
        }
    }

    PrefixParams     prefixParams_;
    CacheManager&    cacheManager_;
    std::vector<int> prefixBlockIndice_;
    int              prefixAdditionalBlock_;
};

class MultiTaskPtuning: public PtuningBase {
public:
    MultiTaskPtuning(CacheManager& cacheManager, const std::unordered_map<int, PrefixParams>& prefixParamsMap):
        cacheManager_(cacheManager), prefixType_(PrefixType::PromptTuning) {
        for (const auto& item : prefixParamsMap) {
            int                 id     = item.first;
            const PrefixParams& params = item.second;
            ptunings_[id]              = std::make_unique<Ptuning>(cacheManager, params, true);
        }
    }

    std::tuple<std::vector<int>, int> getBlockIndice(int blockNum, const GenerateConfig& generateConfig) override {
        auto task_id = generateConfig.task_id;
        if (task_id != std::nullopt) {
            auto it = ptunings_.find(task_id.value());
            if (it == ptunings_.end()) {
                return {cacheManager_.mallocIndex(blockNum), 0};
            }
            return it->second->getBlockIndice(blockNum, generateConfig);
        } else {
            return {cacheManager_.mallocIndex(blockNum), 0};
        }
    }

    std::tuple<PrefixType, std::optional<torch::Tensor>>
    getPrefixParams(const GenerateConfig& generateConfig) override {
        auto task_id = generateConfig.task_id;
        if (task_id != std::nullopt) {
            auto it = ptunings_.find(task_id.value());
            if (it == ptunings_.end()) {
                return {PrefixType::NoPrefix, torch::Tensor()};
            }
            return it->second->getPrefixParams(generateConfig);
        } else {
            return {PrefixType::NoPrefix, torch::Tensor()};
        }
    }

    size_t calcPrefixBlockNum(const GenerateConfig& generateConfig) override {
        auto task_id = generateConfig.task_id;
        if (task_id != std::nullopt) {
            auto it = ptunings_.find(task_id.value());
            if (it == ptunings_.end()) {
                return 0;
            }
            return it->second->calcPrefixBlockNum(generateConfig);
        } else {
            return 0;
        }
    }

private:
    CacheManager&                                     cacheManager_;
    std::unordered_map<int, std::unique_ptr<Ptuning>> ptunings_;
    PrefixType                                        prefixType_;
};

}  // namespace rtp_llm
