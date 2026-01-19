#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  bool                     is_mtp) {
    int        local_head_num_kv = (model_config.attn_config.kv_head_num > 1) ?
                                       model_config.attn_config.kv_head_num / parallelism_config.tp_size :
                                       model_config.attn_config.kv_head_num;
    const auto device_prop       = rtp_llm::DeviceFactory::getDefaultDevice()->getDeviceProperties();
    auto       dtype =
        model_config.attn_config.kv_cache_dtype == KvCacheDataType::INT8 ?
                  rtp_llm::DataType::TYPE_INT8 :
                  (model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8 ? rtp_llm::DataType::TYPE_FP8_E4M3 :
                                                                                     model_config.data_type);
    if (device_prop.type == rtp_llm::DeviceType::ArmCpu) {
        // Arm attention operator support FP32 data type only
        dtype =
            model_config.attn_config.kv_cache_dtype == KvCacheDataType::INT8 ? rtp_llm::TYPE_INT8 : rtp_llm::TYPE_FP32;
    }
    auto layer_num = model_config.num_layers;
    if (is_mtp) {
        layer_num = 1;
    }

    std::vector<int> all_layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        all_layer_ids[i] = i;
    }

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);

    config.use_mla   = model_config.attn_config.use_mla;
    config.is_sparse = model_config.attn_config.is_sparse;
    config.dtype     = dtype;

    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        auto spec                = std::make_shared<MLAKVCacheSpec>();
        spec->type               = KVCacheType::MultiHeadLatentAttention;
        spec->dtype              = dtype;
        spec->kv_lora_rank       = static_cast<uint32_t>(model_config.attn_config.kv_lora_rank);
        spec->rope_head_dim      = static_cast<uint32_t>(model_config.attn_config.rope_head_dim);
        spec->local_head_num_kv  = 1;  // mla set local_head_num_kv to 1
        spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
        spec->is_sparse          = model_config.attn_config.is_sparse;

        config.cache_specs.push_back(spec);
        config.kv_block_stride       = spec->block_size();
        config.kv_block_stride_bytes = spec->block_size_bytes();
        config.kv_block_size         = static_cast<size_t>(config.layer_num) * config.kv_block_stride;
        config.kv_block_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_block_stride_bytes;
    } else {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->type               = KVCacheType::MultiHeadAttention;
        spec->dtype              = dtype;
        spec->local_head_num_kv  = static_cast<uint32_t>(std::max(1, local_head_num_kv));
        spec->size_per_head      = static_cast<uint32_t>(model_config.attn_config.size_per_head);
        spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);

        config.cache_specs.push_back(spec);
        config.kv_block_stride       = spec->block_size();
        config.kv_block_stride_bytes = spec->block_size_bytes();
        config.kv_block_size         = static_cast<size_t>(config.layer_num) * config.kv_block_stride;
        config.kv_block_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_block_stride_bytes;
    }

    // kv scale stride (K+V scales together) for int8/fp8
    if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
        const size_t local_head_num_kv        = static_cast<size_t>(config.cache_specs[0]->local_head_num_kv);
        const size_t seq_size_per_block       = static_cast<size_t>(config.seq_size_per_block);
        const size_t kv_scale_kv_stride       = local_head_num_kv * seq_size_per_block;
        const size_t kv_scale_kv_stride_bytes = kv_scale_kv_stride * sizeof(float);
        config.kv_scale_stride                = 2 * kv_scale_kv_stride;
        config.kv_scale_stride_bytes          = 2 * kv_scale_kv_stride_bytes;
        config.kv_scale_size                  = static_cast<size_t>(config.layer_num) * config.kv_scale_stride;
        config.kv_scale_size_bytes            = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;
    } else {
        config.kv_scale_stride       = 0;
        config.kv_scale_stride_bytes = 0;
        config.kv_scale_size         = 0;
        config.kv_scale_size_bytes   = 0;
    }

    config.block_stride       = config.kv_block_stride + config.kv_scale_stride;
    config.block_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.block_size         = config.kv_block_size + config.kv_scale_size;
    config.block_size_bytes   = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    // Global layer ids are the indices used by BlockPool::convertIndexToAddr (0..N-1 in a single-model case).
    config.global_layer_ids.push_back(all_layer_ids);
    config.layer_ids.push_back(all_layer_ids);
    return config;
}

size_t CacheConfigCreator::getDefaultRuntimeMemorySize(const RuntimeConfig&     runtime_config,
                                                       const ParallelismConfig& parallelism_config,
                                                       const ModelConfig&       model_config,
                                                       const std::optional<SpeculativeExecutionConfig>& sp_config) {
    auto reserve_runtime_mem_bytes = runtime_config.reserve_runtime_mem_mb * 1024 * 1024;
    RTP_LLM_LOG_INFO("RuntimeConfig has reserve_runtime_mem_mb=%ld", runtime_config.reserve_runtime_mem_mb);

    const auto minimal_runtime_bytes = 256L * 1024 * 1024 * std::max(4, 8 / (int)parallelism_config.tp_size);
    if (reserve_runtime_mem_bytes < minimal_runtime_bytes) {
        RTP_LLM_LOG_INFO("tp_size %d needs at least %d MiB memory for runtime by default, "
                         "but only %ld MiB reserved memory set by config. adjust to minimal value.",
                         parallelism_config.tp_size,
                         minimal_runtime_bytes / 1024 / 1024,
                         reserve_runtime_mem_bytes / 1024 / 1024);
        reserve_runtime_mem_bytes = minimal_runtime_bytes;
    }

    if (model_config.mm_model_config.is_multimodal) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        if (reserve_runtime_mem_bytes < minimal_runtime_required) {
            reserve_runtime_mem_bytes = minimal_runtime_required;
            RTP_LLM_LOG_INFO("multimodal needs at least %ld MiB memory for runtime by default, "
                             "but only %ld MiB memory reserved. adjust to minimal value.",
                             minimal_runtime_required / 1024 / 1024,
                             reserve_runtime_mem_bytes / 1024 / 1024);
        }
    }

    if (sp_config && sp_config->type != SP_TYPE_NONE) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        if (reserve_runtime_mem_bytes < minimal_runtime_required) {
            reserve_runtime_mem_bytes = minimal_runtime_required;
            RTP_LLM_LOG_INFO("speculative decoding  needs at least %ld MiB memory for runtime by default, "
                             "but only %ld MiB memory reserved. adjust to minimal value.",
                             minimal_runtime_required / 1024 / 1024,
                             reserve_runtime_mem_bytes / 1024 / 1024);
        }
    }

    return reserve_runtime_mem_bytes;
}

size_t CacheConfigCreator::getKVCacheMemorySize(const RuntimeConfig&                             runtime_config,
                                                const KVCacheConfig&                             kv_cache_config,
                                                const ModelConfig&                               model_config,
                                                const ParallelismConfig&                         parallelism_config,
                                                const std::optional<WarmUpResult>&               warm_up_result,
                                                const std::optional<SpeculativeExecutionConfig>& sp_config) {
    const auto device                       = rtp_llm::DeviceFactory::getDefaultDevice();
    size_t     device_reserved_memory_bytes = device->getDeviceStatus().device_memory_status.preserved_bytes;
    size_t     runtime_required_bytes       = 0;

    if (kv_cache_config.kv_cache_mem_mb > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache memory size %ld MiB",
                         kv_cache_config.kv_cache_mem_mb);
        return kv_cache_config.kv_cache_mem_mb * 1024 * 1024;
    }

    // Unified call to getDefaultRuntimeMemorySize
    size_t env_runtime_required_bytes =
        getDefaultRuntimeMemorySize(runtime_config, parallelism_config, model_config, sp_config);

    if (warm_up_result) {
        if (device_reserved_memory_bytes != warm_up_result->device_reserved_bytes) {
            RTP_LLM_LOG_WARNING("device reserved memory bytes %ld when create config does not equal to "
                                "the amount when warm up %ld. take min value.",
                                device_reserved_memory_bytes,
                                warm_up_result->device_reserved_bytes);
            device_reserved_memory_bytes =
                std::min(device_reserved_memory_bytes, warm_up_result->device_reserved_bytes);
        }

        runtime_required_bytes = std::max(env_runtime_required_bytes, warm_up_result->max_used_memory);

        RTP_LLM_LOG_INFO(
            "devices reserved %ld MiB memory, warm up consumed %ld MiB max memory, env runtime memory %ld MiB, final runtime memory %ld MiB",
            device_reserved_memory_bytes / 1024 / 1024,
            warm_up_result->max_used_memory / 1024 / 1024,
            env_runtime_required_bytes / 1024 / 1024,
            runtime_required_bytes / 1024 / 1024);
    } else {
        runtime_required_bytes = env_runtime_required_bytes;
        RTP_LLM_LOG_INFO("warm up result not available, use default runtime memory size %ld MiB",
                         runtime_required_bytes / 1024 / 1024);
    }

    size_t sample_need_mem =
        (size_t)runtime_config.max_generate_batch_size * model_config.vocab_size * 4 * 8;  // just estimated value
    RTP_LLM_LOG_INFO("sampler needs %ld MiB memory, model runtime needs %ld MiB memory, take max value.",
                     sample_need_mem / 1024 / 1024,
                     runtime_required_bytes / 1024 / 1024);
    runtime_required_bytes = std::max(sample_need_mem, runtime_required_bytes);

    RTP_LLM_CHECK_WITH_INFO(device_reserved_memory_bytes > runtime_required_bytes,
                            "device reserved memory %ld  MiB is less than runtime required memory %ld MiB",
                            device_reserved_memory_bytes / 1024 / 1024,
                            runtime_required_bytes / 1024 / 1024);

    const auto kv_cache_mem_size = device_reserved_memory_bytes - runtime_required_bytes;
    RTP_LLM_LOG_INFO("cache config final decided kv cache memory size %ld MiB", kv_cache_mem_size / 1024 / 1024);
    return kv_cache_mem_size;
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config    = CacheConfigCreator::createBasicConfig(model_config, parallelism_config);
    uint32_t    block_num = 0;

    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = getKVCacheMemorySize(
            runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
        block_num = kv_cache_mem_size / config.block_size_bytes;
    }
    RTP_LLM_CHECK_WITH_INFO(block_num > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_num,
                            static_cast<long>(config.block_size_bytes / 1024 / 1024));

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    config.block_num            = static_cast<int>(block_num);
    RTP_LLM_LOG_INFO("kv cache block nums is %u, allows storing %ld tokens", block_num, kv_cache_seq_len);
    if (kv_cache_seq_len < model_config.max_seq_len) {
        RTP_LLM_LOG_WARNING("kv cache block nums %u can only store %ld tokens, less than max_seq_len %ld, "
                            "this is dangerous, consider decrease max_seq_len",
                            block_num,
                            kv_cache_seq_len,
                            model_config.max_seq_len);
    }
    return config;
}

CacheConfig CacheConfigCreator::createSpConfig(const ModelConfig&                 score_model_config,
                                               const ModelConfig&                 propose_model_config,
                                               const ParallelismConfig&           parallelism_config,
                                               const RuntimeConfig&               runtime_config,
                                               const KVCacheConfig&               kv_cache_config,
                                               const SpeculativeExecutionConfig&  sp_config,
                                               const std::optional<WarmUpResult>& warm_up_result,
                                               bool                               is_mtp,
                                               bool                               is_eagle) {
    CacheConfig score_config = CacheConfigCreator::createBasicConfig(score_model_config, parallelism_config, false);
    CacheConfig propose_config =
        CacheConfigCreator::createBasicConfig(propose_model_config, parallelism_config, is_mtp);

    int num_mtp_modules = 1;
    if (is_mtp) {
        num_mtp_modules = sp_config.gen_num_per_cycle;
        if (is_eagle) {
            num_mtp_modules = 1;
        }
    }

    uint32_t total_layer_num = score_config.layer_num;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_layer_num += propose_config.layer_num;
    }

    size_t total_block_size_bytes = score_config.block_size_bytes;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_block_size_bytes += propose_config.block_size_bytes;
    }

    size_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = CacheConfigCreator::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);

        block_num = kv_cache_mem_size
                    / (static_cast<size_t>(score_config.block_size_bytes)
                       + static_cast<size_t>(propose_config.block_size_bytes) * static_cast<size_t>(num_mtp_modules));
    }

    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    CacheConfig config      = score_config;
    config.layer_all_num    = total_layer_num;
    config.block_size_bytes = total_block_size_bytes;
    config.block_size       = config.block_size_bytes / rtp_llm::getTypeSize(config.dtype);
    config.block_num        = block_num;

    // Record global layer ids for BlockPool address lookup.
    // - Main model global_layer_ids[0] covers all layers across main + mtp modules: [0 .. total_layer_num-1].
    // - Each mtp_sub_config has its own global_layer_ids[0] range for its local layers.
    config.global_layer_ids.clear();
    config.global_layer_ids.resize(1);
    config.global_layer_ids[0].resize(total_layer_num);
    for (uint32_t i = 0; i < total_layer_num; ++i) {
        config.global_layer_ids[0][i] = static_cast<int>(i);
    }

    const uint32_t main_layer_num = score_config.layer_num;
    const uint32_t mtp_layer_num  = propose_config.layer_num;

    // Each sub-model needs an independent CacheConfig because global_layer_ids differs per module.
    config.mtp_sub_configs.clear();
    config.mtp_sub_configs.reserve(num_mtp_modules);
    for (int m = 0; m < num_mtp_modules; ++m) {
        auto sub_cfg           = std::make_shared<CacheConfig>(propose_config);
        sub_cfg->block_num     = block_num;
        sub_cfg->layer_all_num = sub_cfg->layer_num;

        sub_cfg->global_layer_ids.clear();
        sub_cfg->global_layer_ids.resize(1);
        sub_cfg->global_layer_ids[0].resize(mtp_layer_num);
        for (uint32_t l = 0; l < mtp_layer_num; ++l) {
            sub_cfg->global_layer_ids[0][l] = static_cast<int>(main_layer_num + m * mtp_layer_num + l);
        }
        config.mtp_sub_configs.push_back(sub_cfg);
    }

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    RTP_LLM_LOG_INFO("CacheConfig created: is_mtp=%d, total_layers=%u, num_mtp_modules=%d, block_num=%zu, "
                     "allows storing %zu tokens, total_block_size=%zu bytes (main=%zu + %d*propose=%zu)",
                     is_mtp,
                     total_layer_num,
                     num_mtp_modules,
                     block_num,
                     kv_cache_seq_len,
                     total_block_size_bytes,
                     score_config.block_size_bytes,
                     num_mtp_modules,
                     propose_config.block_size_bytes);

    RTP_LLM_LOG_INFO("CacheConfig debugString(main_score_model):\n%s", score_config.debugString().c_str());
    for (size_t i = 0; i < config.mtp_sub_configs.size(); ++i) {
        const auto& sub = config.mtp_sub_configs[i];
        RTP_LLM_LOG_INFO("CacheConfig debugString(sub_propose_model[%zu]):\n%s", i, sub->debugString().c_str());
    }

    return config;
}

}  // namespace rtp_llm
