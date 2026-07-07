#pragma once

#include <cstdint>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

struct TestKVCacheSpec: public KVCacheSpec {
    DataType dtype = DataType::TYPE_INVALID;
    size_t   k_block_bytes = 0;
    size_t   v_block_bytes = 0;
    size_t   k_scale_bytes     = 0;
    size_t   v_scale_bytes     = 0;
    uint32_t local_kv_head_num = 1;

    size_t block_size() const override {
        return k_block_size() + v_block_size();
    }
    size_t k_block_size() const override {
        return k_block_bytes / getTypeSize(dtype);
    }
    size_t v_block_size() const override {
        return v_block_bytes / getTypeSize(dtype);
    }
    size_t block_size_bytes() const override {
        return k_block_bytes + v_block_bytes;
    }
    size_t k_block_size_bytes() const override {
        return k_block_bytes;
    }
    size_t v_block_size_bytes() const override {
        return v_block_bytes;
    }
    size_t scale_block_size_bytes() const override {
        return k_scale_bytes + v_scale_bytes;
    }
    size_t k_scale_block_size_bytes() const override {
        return k_scale_bytes;
    }
    size_t v_scale_block_size_bytes() const override {
        return v_scale_bytes;
    }
    DataType memoryLayoutDType() const override {
        return dtype;
    }
    KVCacheSpecPtr clone() const override {
        return std::make_shared<TestKVCacheSpec>(*this);
    }
    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "TestKVCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

inline KVCacheSpecPtr createTestKvCacheSpec(uint32_t          layer_num,
                                            rtp_llm::DataType dtype,
                                            uint32_t          local_head_num_kv,
                                            uint32_t          seq_size_per_block,
                                            size_t            k_block_stride_bytes,
                                            size_t            v_block_stride_bytes) {
    (void)layer_num;
    RTP_LLM_CHECK_WITH_INFO(local_head_num_kv > 0, "local_head_num_kv must be > 0");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be > 0");
    const size_t type_sz = rtp_llm::getTypeSize(dtype);
    RTP_LLM_CHECK_WITH_INFO(type_sz > 0, "invalid dtype=%d", static_cast<int>(dtype));
    RTP_LLM_CHECK_WITH_INFO(k_block_stride_bytes % type_sz == 0,
                            "k_block_stride_bytes=%zu must be divisible by type size=%zu",
                            k_block_stride_bytes,
                            type_sz);
    RTP_LLM_CHECK_WITH_INFO(v_block_stride_bytes % type_sz == 0,
                            "v_block_stride_bytes=%zu must be divisible by type size=%zu",
                            v_block_stride_bytes,
                            type_sz);

    auto spec                = std::make_shared<TestKVCacheSpec>();
    spec->tag                = "default";
    spec->type               = k_block_stride_bytes == v_block_stride_bytes ? KVCacheSpecType::MultiHeadAttention :
                                                                             KVCacheSpecType::MultiHeadLatentAttention;
    spec->seq_size_per_block = seq_size_per_block;
    spec->dtype              = dtype;
    spec->k_block_bytes      = k_block_stride_bytes;
    spec->v_block_bytes      = v_block_stride_bytes;
    spec->local_kv_head_num = local_head_num_kv;
    return spec;
}

inline BlockPoolConfig createTestConfig(size_t            k_block_stride_bytes = 512,
                                        size_t            v_block_stride_bytes = 512,
                                        size_t            k_scale_stride_bytes = 0,
                                        size_t            v_scale_stride_bytes = 0,
                                        rtp_llm::DataType dtype                = rtp_llm::DataType::TYPE_FP16,
                                        uint32_t          local_head_num_kv    = 1,
                                        uint32_t          seq_size_per_block   = 1) {
    constexpr uint32_t kLayerNum = 4;
    constexpr uint32_t kBlockNum = 10;

    auto spec = createTestKvCacheSpec(
        kLayerNum, dtype, local_head_num_kv, seq_size_per_block, k_block_stride_bytes, v_block_stride_bytes);
    auto test_spec           = std::dynamic_pointer_cast<TestKVCacheSpec>(spec);
    test_spec->k_scale_bytes = k_scale_stride_bytes;
    test_spec->v_scale_bytes = v_scale_stride_bytes;

    rtp_llm::CacheConfig cache_config;
    cache_config.layer_num             = kLayerNum;
    cache_config.layer_all_num         = kLayerNum;
    cache_config.block_num             = kBlockNum;
    cache_config.dtype                 = dtype;
    cache_config.seq_size_per_block    = seq_size_per_block;
    cache_config.kv_block_stride_bytes = k_block_stride_bytes + v_block_stride_bytes;
    cache_config.kv_scale_stride_bytes = k_scale_stride_bytes + v_scale_stride_bytes;

    std::vector<int> layer_ids(kLayerNum);
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
    cache_config.groups[0].local_kv_head_num = test_spec->local_kv_head_num;

    return BlockPoolConfigHelper::createConfig(cache_config);
}

inline void createDevice() {
    torch::manual_seed(114514);
    rtp_llm::initRuntime(/*device_id=*/0,
                         /*trace_memory=*/false,
                         /*enable_comm_overlap=*/false,
                         rtp_llm::MlaOpsType::AUTO);
}

BlockPoolPtr createBlockPool() {
    createDevice();
    auto config     = createTestConfig();
    auto block_pool = std::make_shared<BlockPool>(config);
    return block_pool;
}

}  // namespace rtp_llm
