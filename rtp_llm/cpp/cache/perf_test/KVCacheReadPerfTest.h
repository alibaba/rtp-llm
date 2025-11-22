#pragma once

#include <random>
#include <filesystem>

#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/cache/DistKvCache.h"
#include "rtp_llm/cpp/cache/perf_test/KVCacheOptionBase.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

extern std::atomic<bool> g_stop_flag;

namespace rtp_llm {

class KVCacheReadOption final: public KVCacheOptionBase {
public:
    KVCacheReadOption(): KVCacheOptionBase() {
        addStringOption("file", file);
        addInt32Option("read_num_per_file", read_num_per_file);
        addInt32Option("read_interval_ms", read_interval_ms);
    }

public:
    bool parseOptions(int argc, char* argv[]) {
        if (!KVCacheOptionBase::parseOptions(argc, argv)) {
            RTP_LLM_LOG_ERROR("base parse options failed");
            return false;
        }

        getOptionValue("file", file);
        getOptionValue("read_num_per_file", read_num_per_file);
        getOptionValue("read_interval_ms", read_interval_ms);

        if (!std::filesystem::exists(file)) {
            RTP_LLM_LOG_ERROR("cache key file not exist: %s, must set a valid filename!", file.c_str());
            return false;
        }

        return true;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "file: " << file << ", read_num_per_file: " << read_num_per_file
            << ", read_interval_ms: " << read_interval_ms << ", " << KVCacheOptionBase::toString();
        return oss.str();
    }

public:
    std::string file{"cache_key_list.txt"};  // cache key filename
    int         read_num_per_file{1};        // read num per kvcache file
    int         read_interval_ms{1000};      // sleep time ms between every read
};

class KVCacheReadPerfTest final {
public:
    void startTest(const KVCacheReadOption& option) const {
        if (option.file.empty()) {
            RTP_LLM_LOG_ERROR("read failed, cache key file not set!");
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
        const auto kvcache = cache_manager->kvCacheBuffer();

        DistStorage3FSInitParams storage_3fs_init_params;
        storage_3fs_init_params.root_dir = "test/";
        DistKvCacheInitParams dist_kvcache_init_params;
        dist_kvcache_init_params.storage_manager_params.init_params_3fs = storage_3fs_init_params;

        auto dist_kvcache = std::make_shared<DistKvCache>(cache_manager.get(), parallelism_config, runtime_config, reporter);
        if (!dist_kvcache->init(dist_kvcache_init_params)) {
            RTP_LLM_LOG_ERROR("dist kvcache init failed");
            return;
        }

        if (g_stop_flag.load()) {
            return;
        }
        // wait for 3fs iov mr
        RTP_LLM_LOG_INFO("sleep for 3fs iov mr");
        usleep(option.wait_mr_time_sec * 1000 * 1000);
        if (g_stop_flag.load()) {
            return;
        }

        int64_t request_id      = 0;
        auto    cache_key_lists = getCacheKeyFromFile(option.file);
        for (const auto& cache_keys : cache_key_lists) {
            if (g_stop_flag.load()) {
                break;
            }
            if (cache_keys.empty()) {
                continue;
            }

            for (int i = 0; i < option.read_num_per_file; ++i) {
                if (g_stop_flag.load()) {
                    break;
                }

                std::vector<int32_t> block_indices(cache_keys.size());
                for (int i = 0; i < cache_keys.size(); ++i) {
                    block_indices[i] = i + 1;
                }

                size_t     ignore_block_num = 0;
                const auto matched_len = dist_kvcache->matchForAllRank(cache_keys, ignore_block_num, request_id, {});
                if (matched_len != cache_keys.size()) {
                    RTP_LLM_LOG_ERROR("not fully match, request: %ld, cache key: %ld, matched len: %d|%lu",
                                      request_id,
                                      cache_keys.back(),
                                      matched_len,
                                      cache_keys.size());
                    ++request_id;
                    continue;
                }

                RTP_LLM_LOG_INFO("get cache, request id: %ld, cache key num: %d", request_id, matched_len);
                if (!dist_kvcache->get(cache_keys, block_indices, ignore_block_num, request_id, {})) {
                    RTP_LLM_LOG_ERROR(
                        "get cache from 3fs failed, request: %ld, cache key: %ld", request_id, cache_keys.back());
                }

                ++request_id;
                usleep(option.read_interval_ms * 1000);
            }
        }
    }

private:
    std::string trim(const std::string& str) const {
        // 去除首尾空格
        auto line = str;
        line.erase(line.begin(),
                   std::find_if(line.begin(), line.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
                   line.end());
        return line;
    }

    std::vector<std::vector<int64_t>> getCacheKeyFromFile(const std::string& filename) const {
        const std::string filepath = (std::filesystem::current_path() / filename).string();
        std::ifstream     file(filepath);
        if (!file.is_open()) {
            RTP_LLM_LOG_ERROR("open file failed, filepath: %s", filepath.c_str());
            return {};
        }

        std::vector<std::vector<int64_t>> result;
        std::string                       line;

        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty()) {
                continue;
            }

            std::vector<int64_t> row;
            std::istringstream   iss(line);
            std::string          token;

            while (std::getline(iss, token, ',')) {
                token = trim(token);
                if (token.empty()) {
                    continue;
                }

                char* endptr;
                errno           = 0;
                long long value = std::strtoll(token.c_str(), &endptr, 10);

                if (errno == ERANGE || value > INT64_MAX || value < INT64_MIN) {
                    RTP_LLM_LOG_ERROR("数值超出 int64_t 范围: %s", token.c_str());
                    continue;
                }
                if (endptr == token.c_str() || *endptr != '\0') {
                    RTP_LLM_LOG_ERROR("无效数字格式: %s", token.c_str());
                    continue;
                }

                row.push_back(static_cast<int64_t>(value));
            }

            if (!row.empty()) {
                result.push_back(std::move(row));
            }
        }

        file.close();
        return result;
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
};

}  // namespace rtp_llm