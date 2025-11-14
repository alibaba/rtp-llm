#pragma once

#include <random>
#include <thread>

#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/cache/DistKvCache.h"
#include "rtp_llm/cpp/cache/perf_test/KVCacheOptionBase.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

extern std::atomic<bool> g_stop_flag;

namespace rtp_llm {

class KVCacheWriteOption final: public KVCacheOptionBase {
public:
    KVCacheWriteOption(): KVCacheOptionBase() {
        addInt32Option("meta_num", meta_num);
        addInt32Option("write_num_per_meta", write_num_per_meta);
        addInt32Option("write_interval_ms", write_interval_ms);
        addBoolOption("increase_meta", increase_meta);
        addInt32Option("max_meta_num", max_meta_num);
        addStringOption("last_cache_key", std::to_string(last_cache_key));
    }

public:
    bool parseOptions(int argc, char* argv[]) {
        if (!KVCacheOptionBase::parseOptions(argc, argv)) {
            RTP_LLM_LOG_ERROR("base parse options failed");
            return false;
        }

        getOptionValue("meta_num", meta_num);
        getOptionValue("write_num_per_meta", write_num_per_meta);
        getOptionValue("write_interval_ms", write_interval_ms);
        getOptionValue("increase_meta", increase_meta);
        getOptionValue("max_meta_num", max_meta_num);

        std::string last_cache_key_str;
        getOptionValue("last_cache_key", last_cache_key_str);
        last_cache_key = std::stoll(last_cache_key_str);

        return true;
    }

    std::string toString() const {
        std::ostringstream ss;
        ss << "meta_num: " << meta_num << ", write_num_per_meta: " << write_num_per_meta
           << ", write_interval_ms: " << write_interval_ms << ", "
           << "increase_meta: " << increase_meta << ", "
           << "max_meta_num: " << max_meta_num << ", "
           << "last_cache_key: " << last_cache_key << ", " << KVCacheOptionBase::toString();
        return ss.str();
    }

public:
    int     meta_num{1};              // the number of cache key per kvcache file
    int     write_num_per_meta{1};    // write times per cache key
    int     write_interval_ms{1000};  // sleep time ms between every write
    bool    increase_meta{false};     // whether increase meta num each write
    int     max_meta_num{1};          // max meta num
    int64_t last_cache_key{-1};       // set this if write to a fixed file
};

class KVCacheWritePerfTest final {
public:
    void startTest(const KVCacheWriteOption& option) const {
        if (option.meta_num <= 0) {
            RTP_LLM_LOG_ERROR("meta_num must be greater than 0");
            return;
        }

        CacheConfig cache_config;
        cache_config.layer_num          = option.layer_num;
        cache_config.block_nums         = option.block_num;
        cache_config.block_size         = option.block_size;
        cache_config.local_head_num_kv  = option.local_head_num_kv;
        cache_config.size_per_head      = option.size_per_head;
        cache_config.seq_size_per_block = option.seq_size_per_block;
        cache_config.dtype              = getDataType(option.data_type);
        cache_config.k_block_stride     = option.block_stride;
        cache_config.v_block_stride     = option.block_stride;

        auto             device   = createDevice();
        auto             reporter = createMetricsReporter();
        KVCacheConfig kv_cache_config;
        ParallelismConfig parallelism_config;
        RuntimeConfig runtime_config;
        runtime_config.model_name = "TestModel";
        auto cache_manager = std::make_shared<CacheManager>(cache_config, device, false, reporter, kv_cache_config, parallelism_config, runtime_config);

        if (g_stop_flag.load()) {
            return;
        }
        // wait for 3fs iov mr
        RTP_LLM_LOG_INFO("sleep for 3fs iov mr");
        usleep(option.wait_mr_time_sec * 1000 * 1000);
        if (g_stop_flag.load()) {
            return;
        }

        // fill kv cache
        fillKVCache(cache_manager, cache_config);

        DistStorage3FSInitParams storage_3fs_init_params;
        storage_3fs_init_params.root_dir = "test/";
        DistKvCacheInitParams dist_kvcache_init_params;
        dist_kvcache_init_params.storage_manager_params.init_params_3fs = storage_3fs_init_params;

        auto dist_kvcache = std::make_shared<DistKvCache>(cache_manager.get(), parallelism_config, runtime_config, reporter);
        if (!dist_kvcache->init(dist_kvcache_init_params)) {
            RTP_LLM_LOG_ERROR("dist kvcache init failed");
            return;
        }

        std::string   filename = "cache_key_list.txt";
        std::ofstream file(filename);
        if (!file.is_open()) {
            RTP_LLM_LOG_WARNING("open cache key file failed, filename: %s", filename.c_str());
        }

        int64_t request_id = 0;
        int     meta_num   = option.meta_num;
        while (true) {
            if (g_stop_flag.load()) {
                break;
            }

            for (int write_times = 0; write_times < option.write_num_per_meta; ++write_times) {
                if (g_stop_flag.load()) {
                    break;
                }

                std::vector<int32_t> block_indices(meta_num);
                for (int i = 0; i < meta_num; ++i) {
                    block_indices[i] = i + 1;
                }

                std::vector<int64_t> cache_keys(meta_num);
                for (int i = 0; i < meta_num; ++i) {
                    cache_keys[i] = generateCacheKey();
                }
                if (option.last_cache_key != -1) {
                    cache_keys.back() = option.last_cache_key;
                }

                size_t ignore_block_num = 0;
                RTP_LLM_LOG_DEBUG("put cache, request: %ld, cache key: %lu", request_id, cache_keys.back());

                if (!dist_kvcache->put(cache_keys, block_indices, ignore_block_num, request_id, {})) {
                    RTP_LLM_LOG_ERROR(
                        "put cache to 3fs failed, request: %ld, cache key: %lu", request_id, cache_keys.back());
                } else {
                    RTP_LLM_LOG_INFO(
                        "put cache success, request: %ld, cache key num: %zu", request_id, cache_keys.size());
                    if (file.is_open()) {
                        file << vectorToString(cache_keys) << std::endl;
                        file.flush();
                    } else {
                        RTP_LLM_LOG_INFO("cache key list: %s", vectorToString(cache_keys).c_str());
                    }
                }

                ++request_id;
                std::this_thread::sleep_for(std::chrono::milliseconds(option.write_interval_ms));
            }

            if (!option.increase_meta) {
                break;
            }

            if (meta_num >= option.max_meta_num) {
                break;
            }
            ++meta_num;
        }

        file.close();
    }

private:
    int64_t generateCacheKey() const {
        // 定义下界和上界
        const int64_t lower_bound = 1111111111111111111LL;
        const int64_t upper_bound = 9223372036854775807LL;

        // 初始化随机数生成器
        std::random_device                     rd;
        std::mt19937_64                        gen(rd());
        std::uniform_int_distribution<int64_t> dist(lower_bound, upper_bound);

        // 生成随机整数
        int64_t value = dist(gen);
        return value;
    }

    DeviceBase* createDevice() const {
        ParallelismConfig parallelism_config;
        ModelConfig model_config;
        EPLBConfig eplb_config;
        FMHAConfig fmha_config;
        DeviceResourceConfig device_resource_config;
        MoeConfig moe_config;
        SpeculativeExecutionConfig sp_config;
        MiscellaneousConfig misc_config;
        ProfilingDebugLoggingConfig profiling_debug_logging_config;
        HWKernelConfig hw_kernel_config;
        ConcurrencyConfig concurrency_config;
        FfnDisAggregateConfig ffn_disaggregate_config;
        RuntimeConfig runtime_config;
        DeviceFactory::initDevices(
            parallelism_config,
            model_config,
            eplb_config,
            fmha_config,
            device_resource_config,
            moe_config,
            sp_config,
            misc_config,
            profiling_debug_logging_config,
            hw_kernel_config,
            concurrency_config,
            ffn_disaggregate_config,
            runtime_config);
        return DeviceFactory::getDefaultDevice();
    }

    kmonitor::MetricsReporterPtr createMetricsReporter() const {
        rtp_llm::initKmonitorFactory();
        auto kmon_tags = kmonitor::MetricsTags();
        kmon_tags.AddTag("dp_rank", "-1");
        return std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);
    }

    void fillKVCache(std::shared_ptr<CacheManager> cache_manager, const CacheConfig& cache_config) const {
        // 设置随机数生成器
        std::random_device                           rd;
        std::mt19937                                 gen(rd());
        std::uniform_int_distribution<unsigned char> dist(0, 255);

        // 申请内存大小
        const size_t size   = cache_config.k_block_stride;
        auto         memory = std::make_unique<unsigned char[]>(size);

        // 填充随机值
        for (size_t i = 0; i < size; ++i) {
            memory[i] = dist(gen);
        }

        for (int layer_index = 0; layer_index < cache_config.layer_num; ++layer_index) {
            for (int block_index = 0; block_index < cache_config.block_nums; ++block_index) {
                auto cpu_buffer = rtp_llm::Buffer(rtp_llm::MemoryType::MEMORY_CPU,
                                                  cache_config.dtype,
                                                  {size / rtp_llm::getTypeSize(cache_config.dtype)},
                                                  memory.get());
                cache_manager->setKVBlockValue(block_index, layer_index, cpu_buffer, cpu_buffer);
            }
        }
    }
};

}  // namespace rtp_llm