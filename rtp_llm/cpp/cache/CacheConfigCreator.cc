#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <numeric>

#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

inline std::vector<std::vector<int>> splitIntoGroups(const std::vector<int>& ids, int group_size) {
    std::vector<std::vector<int>> groups;
    if (ids.empty()) {
        return groups;
    }
    const int n = static_cast<int>(ids.size());
    const int s = std::max(group_size, 1);
    groups.reserve((n + s - 1) / s);
    for (int i = 0; i < n; i += s) {
        const int end = std::min(i + s, n);
        groups.emplace_back(ids.begin() + i, ids.begin() + end);
    }
    return groups;
}

}  // namespace

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  bool                     is_mtp) {
    if (model_config.hybrid_attention_config.enable_hybrid_attention) {
        return CacheConfigCreator::createHybridConfig(model_config, parallelism_config, is_mtp);
    } else {
        return CacheConfigCreator::createSingleConfig(model_config, parallelism_config, is_mtp);
    }
}

CacheConfig CacheConfigCreator::createSingleConfig(const ModelConfig&       model_config,
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

    config.use_mla = model_config.attn_config.use_mla;
    config.dtype   = dtype;

    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        auto spec                = std::make_shared<MLAKVCacheSpec>();
        spec->type               = KVCacheType::MultiHeadLatentAttention;
        spec->dtype              = dtype;
        spec->kv_lora_rank       = static_cast<uint32_t>(model_config.attn_config.kv_lora_rank);
        spec->rope_head_dim      = static_cast<uint32_t>(model_config.attn_config.rope_head_dim);
        spec->local_head_num_kv  = 1;  // mla set local_head_num_kv to 1
        spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);

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
    config.group_size         = layer_num;  // only 1 group for SingleConfig

    // Global layer ids are the indices used by BlockPool::convertIndexToAddr (0..N-1 in a single-model case).
    config.global_layer_ids.push_back(all_layer_ids);
    config.layer_ids.push_back(all_layer_ids);
    return config;
}

CacheConfig CacheConfigCreator::createHybridConfig(const ModelConfig&       model_config,
                                                   const ParallelismConfig& parallelism_config,
                                                   bool                     is_mtp) {
    // Hybrid config currently supports main model only (no MTP). Keep behavior consistent with createBasicConfig.
    RTP_LLM_CHECK_WITH_INFO(!is_mtp, "createHybridConfig does not support is_mtp=true yet");

    const auto device_prop = rtp_llm::DeviceFactory::getDefaultDevice()->getDeviceProperties();
    auto       dtype =
        model_config.attn_config.kv_cache_dtype == KvCacheDataType::INT8 ?
                  rtp_llm::DataType::TYPE_INT8 :
                  (model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8 ? rtp_llm::DataType::TYPE_FP8_E4M3 :
                                                                                     model_config.data_type);
    if (device_prop.type == rtp_llm::DeviceType::ArmCpu) {
        dtype =
            model_config.attn_config.kv_cache_dtype == KvCacheDataType::INT8 ? rtp_llm::TYPE_INT8 : rtp_llm::TYPE_FP32;
    }

    int64_t layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);

    // Split layers by attention type from model_config.hybrid_attention_config.hybrid_attention_types.
    std::vector<int> linear_layers;
    std::vector<int> full_layers;
    linear_layers.reserve(layer_num);
    full_layers.reserve(layer_num);

    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(types.size()) == layer_num,
                            "hybrid_attention_types size mismatch: got=%zu expect=%ld",
                            types.size(),
                            layer_num);

    for (int i = 0; i < static_cast<int>(layer_num); ++i) {
        if (types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            linear_layers.push_back(i);
        } else {
            // Treat NONE / SLIDING_WINDOW as "full attention" group for KV cache purpose.
            full_layers.push_back(i);
        }
    }

    // Build per-attention-type ids (semantic required by user).
    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;

    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.global_layer_ids.push_back(linear_layers);
    config.global_layer_ids.push_back(full_layers);
    config.layer_ids.push_back(linear_layers);
    config.layer_ids.push_back(full_layers);

    // 1) Build full-attention spec (MHA / MLA).
    KVCacheSpecPtr full_spec;
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        auto spec                = std::make_shared<MLAKVCacheSpec>();
        spec->type               = KVCacheType::MultiHeadLatentAttention;
        spec->dtype              = dtype;
        spec->kv_lora_rank       = static_cast<uint32_t>(model_config.attn_config.kv_lora_rank);
        spec->rope_head_dim      = static_cast<uint32_t>(model_config.attn_config.rope_head_dim);
        spec->local_head_num_kv  = 1;  // mla set local_head_num_kv to 1
        spec->seq_size_per_block = config.seq_size_per_block;
        full_spec                = spec;
    } else {
        int local_head_num_kv =
            (model_config.attn_config.kv_head_num > 1) ?
                static_cast<int>(model_config.attn_config.kv_head_num / parallelism_config.tp_size) :
                static_cast<int>(model_config.attn_config.kv_head_num);
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->type               = KVCacheType::MultiHeadAttention;
        spec->dtype              = dtype;
        spec->local_head_num_kv  = static_cast<uint32_t>(std::max(1, local_head_num_kv));
        spec->size_per_head      = static_cast<uint32_t>(model_config.attn_config.size_per_head);
        spec->seq_size_per_block = config.seq_size_per_block;
        full_spec                = spec;
    }

    // 2) Build linear-attention spec.
    // Match Python Qwen3NextGatedDeltaNetBase:
    // - ssm_state_size = local_num_v_heads * head_k_dim * head_v_dim
    // - conv_state_size = (kernel_dim - 1) * qkv_size
    // - qkv_size = head_k_dim * local_num_k_heads * 2 + head_v_dim * local_num_v_heads
    const auto& la = model_config.linear_attention_config;
    RTP_LLM_CHECK_WITH_INFO(la.linear_key_head_dim > 0 && la.linear_value_head_dim > 0, "invalid linear head dim");
    RTP_LLM_CHECK_WITH_INFO(
        la.linear_conv_kernel_dim > 1, "invalid linear_conv_kernel_dim=%d", la.linear_conv_kernel_dim);
    RTP_LLM_CHECK_WITH_INFO(la.linear_num_key_heads > 0 && la.linear_num_value_heads > 0, "invalid linear heads");

    const int tp                = std::max(1, parallelism_config.tp_size);
    const int local_num_k_heads = la.linear_num_key_heads / tp;
    const int local_num_v_heads = la.linear_num_value_heads / tp;
    RTP_LLM_CHECK_WITH_INFO(local_num_k_heads > 0 && local_num_v_heads > 0,
                            "invalid local heads for linear attention: k=%d v=%d tp=%d",
                            local_num_k_heads,
                            local_num_v_heads,
                            tp);
    RTP_LLM_CHECK_WITH_INFO(la.linear_key_head_dim == la.linear_value_head_dim,
                            "linear head dims must match (current impl): k=%d v=%d",
                            la.linear_key_head_dim,
                            la.linear_value_head_dim);

    const int head_dim        = la.linear_key_head_dim;
    const int ssm_state_size  = local_num_v_heads * head_dim * head_dim;
    const int qkv_size        = head_dim * local_num_k_heads * 2 + head_dim * local_num_v_heads;
    const int conv_state_size = (la.linear_conv_kernel_dim - 1) * qkv_size;

    auto linear_spec                 = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type                = KVCacheType::LinearAttention;
    linear_spec->dtype               = dtype;
    linear_spec->local_head_num_kv   = 1;
    linear_spec->seq_size_per_block  = config.seq_size_per_block;
    linear_spec->conv_state_size     = static_cast<uint32_t>(conv_state_size);
    linear_spec->temporal_state_size = static_cast<uint32_t>(ssm_state_size);

    // Partition KVCacheGroups by gcd(total_linear_layers, total_full_layers).
    const int linear_cnt = static_cast<int>(linear_layers.size());
    const int full_cnt   = static_cast<int>(full_layers.size());
    int       group_size = 0;
    if (linear_cnt > 0 && full_cnt > 0) {
        group_size = std::gcd(linear_cnt, full_cnt);
    } else {
        group_size = std::max(linear_cnt, full_cnt);
    }
    group_size        = std::max(group_size, 1);
    config.group_size = group_size;

    const auto linear_groups = splitIntoGroups(linear_layers, group_size);
    const auto full_groups   = splitIntoGroups(full_layers, group_size);

    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.cache_specs.clear();

    // Keep order: all linear groups first, then full groups.
    for (const auto& g : linear_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(linear_spec);
    }
    for (const auto& g : full_groups) {
        config.global_layer_ids.push_back(g);
        config.layer_ids.push_back(g);
        config.cache_specs.push_back(full_spec);
    }
    config.linear_group_num = static_cast<int>(linear_groups.size());
    config.full_group_num   = static_cast<int>(full_groups.size());

    // Decide the physical KV block/scale sizes by taking max between full and linear specs.
    // NOTE: Hybrid allocator currently shares one BlockPool layout for all groups, so the largest block wins.
    const size_t full_kv_block_stride         = full_spec->block_size();
    const size_t full_kv_block_stride_bytes   = full_spec->block_size_bytes();
    const size_t linear_kv_block_stride       = linear_spec->block_size();
    const size_t linear_kv_block_stride_bytes = linear_spec->block_size_bytes();

    config.kv_block_stride       = std::max(full_kv_block_stride, linear_kv_block_stride);
    config.kv_block_stride_bytes = std::max(full_kv_block_stride_bytes, linear_kv_block_stride_bytes);
    config.kv_block_size         = static_cast<size_t>(config.group_size) * config.kv_block_stride;
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_size) * config.kv_block_stride_bytes;

    // kv scale stride (K+V scales together) for int8/fp8: take max between two "layouts" (full vs linear).
    size_t full_scale_stride       = 0;
    size_t full_scale_stride_bytes = 0;
    if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
        const size_t local_head_num_kv        = static_cast<size_t>(full_spec->local_head_num_kv);
        const size_t seq_size_per_block       = static_cast<size_t>(config.seq_size_per_block);
        const size_t kv_scale_kv_stride       = local_head_num_kv * seq_size_per_block;
        const size_t kv_scale_kv_stride_bytes = kv_scale_kv_stride * sizeof(float);
        full_scale_stride                     = 2 * kv_scale_kv_stride;
        full_scale_stride_bytes               = 2 * kv_scale_kv_stride_bytes;
    }
    // linear attention kv scale is treated as 0 (currently not used).
    config.kv_scale_stride       = full_scale_stride;
    config.kv_scale_stride_bytes = full_scale_stride_bytes;
    config.kv_scale_size         = static_cast<size_t>(config.group_size) * config.kv_scale_stride;
    config.kv_scale_size_bytes   = static_cast<size_t>(config.group_size) * config.kv_scale_stride_bytes;

    config.block_stride       = config.kv_block_stride + config.kv_scale_stride;
    config.block_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.block_size         = config.kv_block_size + config.kv_scale_size;
    config.block_size_bytes   = config.kv_block_size_bytes + config.kv_scale_size_bytes;

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

    config.linear_step = kv_cache_config.linear_step;
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
    config.linear_step      = std::max(1, kv_cache_config.linear_step);
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
