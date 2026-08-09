#pragma once

#include <optional>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

struct KVCacheBlockBudget {
    size_t explicit_pool_reserve_bytes = 0;
    size_t paged_block_bytes           = 0;
    size_t swa_block_bytes             = 0;
};

uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget, int linear_step);

class CacheConfigCreator {
public:
    static CacheConfig createBasicConfig(const ModelConfig&       model_config,
                                         const ParallelismConfig& parallelism_config,
                                         bool                     is_mtp,
                                         int                      gen_num_per_cycle);
    static CacheConfig createConfig(const ModelConfig&                               model_config,
                                    const ParallelismConfig&                         parallelism_config,
                                    const RuntimeConfig&                             runtime_config,
                                    const KVCacheConfig&                             kv_cache_config,
                                    const std::optional<WarmUpResult>&               warm_up_result = std::nullopt,
                                    const std::optional<SpeculativeExecutionConfig>& sp_config      = std::nullopt);
    static CacheConfig createSpConfig(const ModelConfig&                 score_model_config,
                                      const ModelConfig&                 propose_model_config,
                                      const ParallelismConfig&           parallelism_config,
                                      const RuntimeConfig&               runtime_config,
                                      const KVCacheConfig&               kv_cache_config,
                                      const SpeculativeExecutionConfig&  sp_config,
                                      const std::optional<WarmUpResult>& warm_up_result,
                                      bool                               is_mtp,
                                      bool                               is_eagle);

private:
    static uint32_t localBlockNum(const KVCacheBlockBudget& budget,
                                  size_t                    total_budget_bytes,
                                  int                       test_block_num,
                                  int                       linear_step,
                                  bool                      sentinel_only);
    static void     publishLocalBlockNum(
            int* block_nums, size_t world_size, int64_t world_rank, uint32_t local_block_num, bool sentinel_only);
    static uint32_t
    selectConvergedBlockNum(const int* block_nums, size_t world_size, uint32_t local_block_num, bool sentinel_only);
    static uint32_t
    convergeBlockNum(uint32_t local_block_num, const ParallelismConfig& parallelism_config, bool sentinel_only);
};

}  // namespace rtp_llm
