#pragma once

#include "rtp_llm/cpp/config/ConfigModules.h"
#include <string>
#include <vector>
#include <cstdint>

namespace rtp_llm {

struct AsymmetricTPContext {
    std::string decode_ip;
    uint32_t    decode_port;
    int         local_partition_count;
    int         local_partition_id;
    int         remote_partition_count;
    int         remote_partition_id;

    AsymmetricTPContext(const std::string& decode_ip,
                        uint32_t           decode_port,
                        int                local_partition_count,
                        int                local_partition_id,
                        int                remote_partition_count,
                        int                remote_partition_id):
        decode_ip(decode_ip),
        decode_port(decode_port),
        local_partition_count(local_partition_count),
        local_partition_id(local_partition_id),
        remote_partition_count(remote_partition_count),
        remote_partition_id(remote_partition_id) {}
};

/// @brief 不对称TP工具类
class AsymmetricTpUtil {
public:
    AsymmetricTpUtil(const ParallelismConfig& parallelism_config);
    ~AsymmetricTpUtil();

public:
    std::vector<AsymmetricTPContext>
    handleAsymmetricTP(const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers);

private:
    /// @brief 处理 N Prefill -> 1 Decode 的情况（Prefill TP 数量大于 Decode TP 数量）
    std::vector<AsymmetricTPContext>
    handleNP1D(const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers);

    /// @brief 处理 1 Prefill -> N Decode 的情况（Decode TP 数量大于 Prefill TP 数量）
    std::vector<AsymmetricTPContext>
    handleND1P(const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers);

private:
    const ParallelismConfig& parallelism_config_;
};

}  // namespace rtp_llm
