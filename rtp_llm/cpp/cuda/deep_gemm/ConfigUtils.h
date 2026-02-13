#pragma once

#include <map>
#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/deep_gemm/utils.h"

namespace rtp_llm {

template<class T>
static inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct DeepGemmConfigKey {
    int          m;
    int          n;
    int          k;
    int          num_groups;
    int          num_sms;
    DeepGemmType gemm_type;
    int          expected_m;

    bool operator==(const DeepGemmConfigKey& other) const {
        return (m == other.m && n == other.n && k == other.k && num_groups == other.num_groups
                && num_sms == other.num_sms && gemm_type == other.gemm_type && expected_m == other.expected_m);
    }
};

struct DeepGemmConfigKeyHasher {
    std::size_t operator()(const DeepGemmConfigKey& key) const {
        uint64_t hash = 0;
        hash_combine(hash, key.m);
        hash_combine(hash, key.n);
        hash_combine(hash, key.k);
        hash_combine(hash, key.num_groups);
        hash_combine(hash, key.num_sms);
        hash_combine(hash, uint64_t(key.gemm_type));
        hash_combine(hash, key.expected_m);
        return hash;
    }
};

static bool gemm_swap_ab_heuristic(size_t m, DeepGemmType gemm_type) {
    switch (gemm_type) {
        case DeepGemmType::Normal:
            return m < 64;  // based on rough empirical results
        case DeepGemmType::GroupedContiguous:
            return false;  // currently not supported
        case DeepGemmType::GroupedMasked:
            return m < 128;  // based on rough empirical results
        default:
            return false;
    }
}

inline int ceil_div(int a, int b) {
    RTP_LLM_CHECK_WITH_INFO(b != 0, "division cannot be zero");
    return (a + b - 1) / b;
}

inline int getTmaAlignedSize(int x, int data_size) {
    int tma_alignment_bytes = 16, alignment;
    RTP_LLM_CHECK_WITH_INFO(tma_alignment_bytes % data_size == 0,
                            "TMA alignment bytes 16 must be divisible by data size");
    alignment = tma_alignment_bytes / data_size;
    return ceil_div(x, alignment) * alignment;
}

inline int fixWaveSaturate(int x, int num_sms) {
    return (x == 0) ? num_sms : x;
}

inline int getNumWaves(int m, int n, int bm, int bn, int num_groups, int num_sms) {
    auto m_w = ceil_div(m, bm), n_w = ceil_div(n, bn);
    return ceil_div(m_w * n_w * num_groups, num_sms);
}

inline int getLastWaveUtil(int m, int n, int bm, int bn, int num_groups, int num_sms) {
    auto m_w = ceil_div(m, bm), n_w = ceil_div(n, bn);
    return fixWaveSaturate(m_w * n_w * num_groups % num_sms, num_sms);
}

inline int getMaxSmem() {
    static int max_smem_per_block = -1;
    if (max_smem_per_block != -1) {
        return max_smem_per_block;
    }
    int device_idx = 0;
    check_cuda_value(cudaGetDevice(&device_idx));
    check_cuda_value(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_idx));
    return max_smem_per_block;
}

inline bool isTmaMulticastLegal(int shape_dim, int block_dim, int num_tma_multicast, int num_sms) {
    if (num_tma_multicast == 1) {
        return true;
    }
    return (shape_dim % (block_dim * num_tma_multicast) == 0) && (num_sms % num_tma_multicast) == 0;
}

inline int getSmemSize(int num_stages, int k, int bm, int bn, int bk, bool swap_ab) {
    if (swap_ab) {
        RTP_LLM_CHECK_WITH_INFO(bk % bm == 0, "invalid block k (%d) and block m (%d)", bk, bm);
        int smem_d                  = bm * bn * 2;
        int smem_a_per_stage        = bm * bk;
        int smem_scales_a_per_stage = ceil_div(k, bk) * 4;
        int smem_b_per_stage        = bn * bk;
        int smem_scales_b           = ceil_div(bn * 4, 128) * 128;
        int smem_barrier            = num_stages * 8 * 2;

        int smem_size = 0;
        smem_size += smem_d;
        smem_size += num_stages * smem_a_per_stage;
        smem_size += num_stages * smem_scales_b;
        smem_size += num_stages * smem_b_per_stage;
        smem_size += ceil_div(smem_scales_a_per_stage, 8) * 8;
        smem_size += smem_barrier;
        return smem_size;
    } else {
        int smem_d                  = bm * bn * 2;
        int smem_a_per_stage        = bm * bk;
        int smem_scales_a_per_stage = bm * 4;
        int smem_b_per_stage        = bn * bk;
        int smem_scales_b           = ceil_div(k, bk) * 4;
        int smem_barrier            = num_stages * 8 * 2;

        int smem_size = 0;
        smem_size += smem_d;
        smem_size += num_stages * smem_a_per_stage;
        smem_size += num_stages * smem_scales_a_per_stage;
        smem_size += num_stages * smem_b_per_stage;
        int scaler = (bk % bn == 0) ? 1 : 2;
        smem_size += ceil_div(smem_scales_b * scaler, 8) * 8;
        smem_size += smem_barrier;
        return smem_size;
    }
}

class DeepGemmConfig {
public:
    uint32_t block_m, block_n, num_stages, num_tma_multicast, smem_size, swap_ab;

    DeepGemmConfig(uint32_t block_m,
                   uint32_t block_n,
                   uint32_t num_stages,
                   uint32_t num_tma_multicast,
                   uint32_t smem_size,
                   bool     swap_ab):
        block_m(block_m),
        block_n(block_n),
        num_stages(num_stages),
        num_tma_multicast(num_tma_multicast),
        smem_size(smem_size),
        swap_ab(swap_ab) {}
};

inline DeepGemmConfig getBestConfig(int          m,
                                    int          n,
                                    int          k,
                                    int          num_groups,
                                    int          num_sms,
                                    DeepGemmType gemm_type,
                                    int          expected_m     = -1,
                                    bool         use_64_padding = false) {
    static std::unordered_map<DeepGemmConfigKey, DeepGemmConfig, DeepGemmConfigKeyHasher> best_configs;

    DeepGemmConfigKey key{m, n, k, num_groups, num_sms, gemm_type, expected_m};
    auto              it = best_configs.find(key);
    if (it != best_configs.end()) {
        return it->second;
    }

    int original_m = m, original_n = n;
    if (expected_m == -1) {
        expected_m = m;
    }
    int expected_n = n;

    bool swap_ab = gemm_swap_ab_heuristic(expected_m, gemm_type);
    if (swap_ab) {
        std::swap(m, n);
        std::swap(expected_m, expected_n);
    }

    int best_block_m = -1, best_block_n = -1;
    {
        struct BlockConfig {
            int block_m   = -1;
            int block_n   = -1;
            int num_waves = std::numeric_limits<int>::max();
            int util      = std::numeric_limits<int>::min();

            bool operator>(const BlockConfig& other) {
                if (num_waves != other.num_waves) {
                    return num_waves < other.num_waves;
                }
                if (util != other.util) {
                    return util > other.util;
                }
                if (block_m != other.block_m) {
                    return block_m > other.block_m;
                }
                if (block_n != other.block_n) {
                    return block_n < other.block_n;
                }
                return false;
            }
        };

        BlockConfig valid_best, best;
        int         block_m;
        if ((gemm_type == DeepGemmType::GroupedContiguous && m <= 64) || expected_m <= 64) {
            block_m = 64;
        } else {
            block_m = 128;
        }

        if (gemm_type == DeepGemmType::GroupedContiguous && use_64_padding) {
            block_m = 64;
        }

        // For some reason, m64n8k32 is not used in deep gemm.
        // It might be worth it to activate the shape
        // for some small m in swap ab variant.
        static int block_ns[] = {16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128};
        for (int block_n : block_ns) {
            int num_waves = getNumWaves(expected_m, expected_n, block_m, block_n, num_groups, num_sms);
            int util      = getLastWaveUtil(expected_m, expected_n, block_m, block_n, num_groups, num_sms);

            BlockConfig current{block_m, block_n, num_waves, util};

            if (current > best) {
                best = current;
            }
            if (current > valid_best) {
                if (gemm_type == DeepGemmType::GroupedMasked && num_groups > 1) {
                    if (swap_ab) {
                        if (n < block_n || n % block_n != 0)
                            continue;
                    } else {
                        if (m < block_m || m % block_m != 0)
                            continue;
                    }
                }
                valid_best = current;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(valid_best.block_m != -1, "block m size cannot be None in best config");
        RTP_LLM_CHECK_WITH_INFO(valid_best.block_n != -1, "block n size cannot be None in best config");
        if (valid_best.block_m != best.block_m || valid_best.block_n != best.block_n) {
            RTP_LLM_LOG_WARNING("best block shape for %sdeep gemm (%d, %d) is not valid, fallback to (%d, %d), "
                                "consider changing the shape m%dn%dk%d for better performance",
                                swap_ab ? "swap ab " : "",
                                best.block_m,
                                best.block_n,
                                valid_best.block_m,
                                valid_best.block_n,
                                original_m,
                                original_n,
                                k);
        }

        best_block_m = valid_best.block_m;
        best_block_n = valid_best.block_n;
    }

    int best_num_stages = -1, best_smem_size = -1;
    {
        const int        sm90_capacitty = getMaxSmem();
        std::vector<int> num_stages_vec;
        if (128 % best_block_n) {
            num_stages_vec = std::vector<int>({6, 5, 4});
        } else {
            num_stages_vec = std::vector<int>({8, 7, 6, 5, 4});
        }
        for (int num_stages : num_stages_vec) {
            best_smem_size = getSmemSize(num_stages, k, best_block_m, best_block_n, 128, swap_ab);
            if (best_smem_size <= sm90_capacitty) {
                best_num_stages = num_stages;
                break;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(best_num_stages != -1, "stages num cannot be None in best config");
    }

    int best_num_tma_multicast = 1;
    if (swap_ab) {
        if (n >= 1024 && isTmaMulticastLegal(m, best_block_m, 2, num_sms) && num_groups == 1) {
            best_num_tma_multicast = 2;
        }
    } else {
        if (m >= 1024 && isTmaMulticastLegal(n, best_block_n, 2, num_sms) && num_groups == 1) {
            best_num_tma_multicast = 2;
        }
    }

    DeepGemmConfig value =
        DeepGemmConfig(best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size, swap_ab);
    best_configs.emplace(key, value);
    return value;
}

}  // namespace rtp_llm
