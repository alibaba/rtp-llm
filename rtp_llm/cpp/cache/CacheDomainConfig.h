#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/ModuleGroupKey.h"

namespace rtp_llm {

// One allocator and BlockTree domain with independent, unmerged module leaves.
// Dense member indices are private iteration coordinates; ownership is always
// resolved by ModuleGroupKey and model layers remain local to their module.
class CacheDomainConfig {
public:
    explicit CacheDomainConfig(std::vector<CacheConfig> modules, size_t active_module_count = 0):
        modules_(std::move(modules)),
        active_module_count_(active_module_count == 0 ? modules_.size() : active_module_count) {
        RTP_LLM_CHECK_WITH_INFO(!modules_.empty(), "cache domain requires a main module");
        RTP_LLM_CHECK_WITH_INFO(active_module_count_ > 0 && active_module_count_ <= modules_.size(),
                                "cache domain active module count=%zu exceeds module count=%zu",
                                active_module_count_,
                                modules_.size());
        const auto& main = modules_.front();
        RTP_LLM_CHECK_WITH_INFO(main.seq_size_per_block > 0, "cache domain requires a positive base key stride");
        for (size_t index = 0; index < modules_.size(); ++index) {
            const auto& leaf = modules_[index];
            RTP_LLM_CHECK_WITH_INFO(leaf.layer_num == leaf.topology().layers().size(),
                                    "cache domain module=%zu layer count=%u topology layers=%zu",
                                    index,
                                    leaf.layer_num,
                                    leaf.topology().layers().size());
            // Reuse admission uses the main module's granularity. A LINEAR-only
            // leaf can have a different reuse_unit_tokens on this same key grid.
            RTP_LLM_CHECK_WITH_INFO(
                leaf.seq_size_per_block == main.seq_size_per_block && leaf.storage_cp_size == main.storage_cp_size
                    && leaf.storage_cp_rank == main.storage_cp_rank && leaf.linear_step == main.linear_step,
                "cache domain module=%zu has an incompatible base key/CP grid",
                index);
            for (const auto& group : leaf.groups()) {
                RTP_LLM_CHECK_WITH_INFO(!leaf.layerIdsForGroup(group.tag).empty(),
                                        "cache domain module=%zu tag=%s has no owning layers",
                                        index,
                                        group.tag.c_str());
                (void)leaf.groupKeyScale(group.tag);
                group_keys_.push_back({CacheModuleId(index), group.tag});
            }
        }
    }

    explicit CacheDomainConfig(CacheConfig main): CacheDomainConfig(std::vector<CacheConfig>{std::move(main)}) {}

    size_t moduleCount() const {
        return modules_.size();
    }

    size_t activeModuleCount() const {
        return active_module_count_;
    }

    bool isActive(CacheModuleId module_id) const {
        (void)module(module_id);
        return module_id.value() < active_module_count_;
    }

    const CacheConfig& module(CacheModuleId module_id) const {
        RTP_LLM_CHECK_WITH_INFO(module_id.value() < modules_.size(),
                                "cache domain module=%zu out of range [0,%zu)",
                                module_id.value(),
                                modules_.size());
        return modules_[module_id.value()];
    }

    const CacheConfig& main() const {
        return module(CacheModuleId::main());
    }

    const std::vector<ModuleGroupKey>& groupKeys() const {
        return group_keys_;
    }

    const GroupBase& group(const ModuleGroupKey& key) const {
        const auto& leaf = module(key.module_id);
        const auto& tags = leaf.groupTags();
        RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), key.tag) != tags.end(),
                                "cache domain unknown module=%zu tag=%s",
                                key.module_id.value(),
                                key.tag.c_str());
        return leaf.group(key.tag);
    }

    const GroupBase& groupForLayer(CacheModuleId module_id, int local_layer_id, const std::string& tag) const {
        const auto& leaf = module(module_id);
        RTP_LLM_CHECK_WITH_INFO(local_layer_id >= 0 && static_cast<size_t>(local_layer_id) < leaf.layer_num,
                                "cache domain module=%zu layer=%d out of range [0,%u) tag=%s",
                                module_id.value(),
                                local_layer_id,
                                leaf.layer_num,
                                tag.c_str());
        (void)group({module_id, tag});
        const auto& tags = leaf.topology().layer(local_layer_id).group_tags;
        RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), tag) != tags.end(),
                                "cache domain module=%zu layer=%d does not own tag=%s",
                                module_id.value(),
                                local_layer_id,
                                tag.c_str());
        return leaf.group(tag);
    }

    size_t groupIndex(const ModuleGroupKey& key) const {
        (void)group(key);
        return static_cast<size_t>(
            std::distance(group_keys_.begin(), std::find(group_keys_.begin(), group_keys_.end(), key)));
    }

    void finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
        for (auto& leaf : modules_) {
            leaf.finalizeBlockNums(global_block_num, runtime_config);
        }
    }

private:
    std::vector<CacheConfig>    modules_;
    std::vector<ModuleGroupKey> group_keys_;
    size_t                      active_module_count_;
};

}  // namespace rtp_llm
