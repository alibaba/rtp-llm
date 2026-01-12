#pragma once

#include <memory>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// 管理 reuse 相关信息的结构体
struct ReuseInfo {
    ReuseInfo(int seq_size_per_block): seq_size_per_block(seq_size_per_block) {
        RTP_LLM_LOG_INFO("ReuseInfo init, seq_size_per_block %d", seq_size_per_block);
    }

    // 计算 reuse blocks 数量 (通过 reuse_length / seq_size_per_block)
    size_t reuseBlocksNum() const {
        return seq_size_per_block > 0 ? reuse_length / seq_size_per_block : 0;
    }

    // 通过 blocks 数量设置 reuse_length
    void setReuseBlocksNum(size_t num) {
        reuse_length = num * seq_size_per_block;
    }

    // 初始化 cache 配置 (全局配置 && 请求配置)
    void init(bool resource_reuse_cache,
              bool resource_enable_3fs,
              bool resource_enable_memory_block_cache,
              bool request_reuse_cache,
              bool request_enable_3fs,
              bool request_enable_memory_block_cache) {
        reuse_cache               = resource_reuse_cache && request_reuse_cache;
        enable_3fs                = resource_enable_3fs && request_enable_3fs;
        enable_memory_block_cache = resource_enable_memory_block_cache && request_enable_memory_block_cache;
    }

    // reuse length 相关
    int initial_reuse_length = 0;  // 请求初始复用长度（用于统计）
    int reuse_length         = 0;  // 当前可复用的 KV cache token 长度
    int local_reuse_length   = 0;  // 本地缓存复用长度
    int remote_reuse_length  = 0;  // 远程缓存复用长度
    int reuse_mm_length      = 0;  // 多模态特征复用数量

    // cache 配置相关
    bool reuse_cache               = false;  // 是否启用 cache 复用
    bool enable_3fs                = false;  // 是否启用 3FS
    bool enable_memory_block_cache = false;  // 是否启用内存 block cache

    int seq_size_per_block = 1;  // 每个 block 的序列长度

    // prefill reuse 信息 (从 prefill 返回给 decode)
    int64_t prefill_total_reuse_len  = 0;  // prefill 总复用长度
    int64_t prefill_local_reuse_len  = 0;  // prefill 本地复用长度
    int64_t prefill_remote_reuse_len = 0;  // prefill 远程复用长度
};

using ReuseInfoPtr = std::shared_ptr<ReuseInfo>;

}  // namespace rtp_llm
