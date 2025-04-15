#include <vector>
#include <map>
#include <algorithm>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include "src/fastertransformer/deep_gemm/utils.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/core/QBuffer.h"
#include "src/fastertransformer/deep_gemm/DeepGemmPlugin.h"

using namespace std;

namespace fastertransformer {
#ifdef ENABLE_FP8
template<uint32_t N, uint32_t K, uint32_t GROUP_NUM, DeepGemmType GEMM_TYPE>
void dispatchBlockNK(__nv_bfloat16*         output,
                     __nv_fp8_e4m3*         lhs,
                     float*                 lhs_scale,
                     __nv_fp8_e4m3*         rhs,
                     float*                 rhs_scale,
                     int*                   grouped_layout,
                     uint32_t               m,
                     uint32_t               bm,
                     uint32_t               bn,
                     uint32_t               bk,
                     uint32_t               num_stages,
                     uint32_t               num_tma_multicast,
                     bool                   is_tma_multicast_on_a,
                     cudaStream_t           stream,
                     uint32_t               num_sms,
                     uint32_t               smem_size);

void runDeepGemm(__nv_bfloat16*         output,
                 __nv_fp8_e4m3*         lhs,
                 float*                 lhs_scale,
                 __nv_fp8_e4m3*         rhs,
                 float*                 rhs_scale,
                 int*                   grouped_layout,
                 uint32_t               m,
                 uint32_t               n,
                 uint32_t               k,
                 uint32_t               bm,
                 uint32_t               bn,
                 uint32_t               bk,
                 uint32_t               num_groups,
                 uint32_t               num_stages,
                 uint32_t               num_tma_multicast,
                 bool                   is_is_tma_multicast_on_a,
                 DeepGemmType           gemm_type,
                 cudaStream_t           stream,
                 uint32_t               num_sms,
                 uint32_t               smem_size);

inline int DeepGemmPlugin::getNumSms() {
    static int num_sms = -1;
    if (num_sms != -1) {
        return num_sms;
    }
    cudaDeviceProp properties;
    int            device_idx;
    check_cuda_error(cudaGetDevice(&device_idx));
    check_cuda_error(cudaGetDeviceProperties(&properties, device_idx));

    num_sms = properties.multiProcessorCount;
    return num_sms;
}

inline int ceil_div(int a, int b) {
    FT_CHECK_WITH_INFO(b != 0, "division cannot be zero");
    return (a + b - 1) / b;
}

inline int getTmaAlignedSize(int x, int data_size) {
    int tma_alignment_bytes = 16, alignment;
    FT_CHECK_WITH_INFO(tma_alignment_bytes % data_size == 0, "TMA alignment bytes 16 must be divisible by data size");
    alignment = tma_alignment_bytes / data_size;
    return ceil_div(x, alignment) * alignment;
}

inline int fixWaveSaturate(int x, int num_sms) {
    return (x == 0)? num_sms: x;
}

inline int getNumWaves(int m, int n, int bm, int bn, int num_groups, int num_sms) {
    auto m_w = ceil_div(m, bm), n_w = ceil_div(n, bn);
    return ceil_div(m_w * n_w * num_groups, num_sms);
}

inline int getLastWaveUtil(int m, int n, int bm, int bn, int num_groups, int num_sms) {
    auto m_w = ceil_div(m, bm), n_w = ceil_div(n, bn);
    return fixWaveSaturate(m_w * n_w * num_groups % num_sms, num_sms);
}

inline bool isTmaMulticastLegal(int shape_dim, int block_dim, int num_tma_multicast, int num_sms) {
    if (num_tma_multicast == 1) {
        return true;
    }
    return (shape_dim % (block_dim * num_tma_multicast) == 0) && (num_sms % num_tma_multicast) == 0;
}

inline int getSmemSize(int num_stages, int k, int bm, int bn, int bk = 128) {
    int smem_d = bm * bn * 2;
    int smem_a_per_stage = bm * bk;
    int smem_scales_a_per_stage = bm * 4;
    int smem_b_per_stage = bn * bk;
    int smem_scales_b = ceil_div(k, bk) * 4;
    int smem_barrier = num_stages * 8 * 2;

    int smem_size = 0;
    smem_size += smem_d;
    smem_size += num_stages * smem_a_per_stage;
    smem_size += num_stages * smem_scales_a_per_stage;
    smem_size += num_stages * smem_b_per_stage;
    int scaler = (bk % bn == 0)? 1: 2;
    smem_size += ceil_div(smem_scales_b * scaler, 8) * 8;
    smem_size += smem_barrier;
    return smem_size;
}


torch::Tensor getColMajorTmaAlignedTensor(Buffer lhs_scale) {
    FT_CHECK_WITH_INFO(lhs_scale.dim() == 2 || lhs_scale.dim() == 3, "lhs scale must be dim 2 or 3");
    FT_CHECK_WITH_INFO(lhs_scale.type() == DataType::TYPE_FP32, "lhs scale must be fp32");
    int remove_dim = 0;
    if (lhs_scale.dim() == 2) {
        remove_dim = 1;
    }

    size_t g, m, k;
    g = remove_dim? 1: lhs_scale.shape()[0];
    m = lhs_scale.shape()[1 - remove_dim];
    k = lhs_scale.shape()[2 - remove_dim];

    int aligned_m = getTmaAlignedSize(m, lhs_scale.typeSize());
	auto col_major_lhs_scale = torch::transpose(torch::empty({int(g), int(k), int(aligned_m)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)), 1, 2);
    col_major_lhs_scale.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(0, m), torch::indexing::Slice()},
        torch::from_blob(
            lhs_scale.data(), {int(g), int(m), int(k)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
    );
    if (remove_dim) {
        return col_major_lhs_scale.squeeze(0);
    } else {
        return col_major_lhs_scale;
    }
}

class DeepGemmConfig {
public:
    uint32_t num_sms, block_m, block_n, num_stages, num_tma_multicast, smem_size;
    bool is_tma_multicast_on_a;

    DeepGemmConfig(uint32_t num_sms, uint32_t block_m, uint32_t block_n, uint32_t num_stages, uint32_t num_tma_multicast,
                   bool is_tma_multicast_on_a, uint32_t smem_size):
        num_sms(num_sms),
        block_m(block_m),
        block_n(block_n),
        num_stages(num_stages),
        num_tma_multicast(num_tma_multicast),
        smem_size(smem_size),
        is_tma_multicast_on_a(is_tma_multicast_on_a) {}
};

DeepGemmConfig getBestConfig(int m, int n, int k, int num_groups, int num_sms, bool is_grouped_contiguous = false) {
    static unordered_map<uint64_t, DeepGemmConfig> best_configs;
    uint64_t key = ((uint64_t)m << 44) | ((uint64_t)(n & 0xffff) << 28) | ((uint64_t)(k & 0xffff) << 12) |  ((uint64_t)num_sms << 4) | ((uint64_t)is_grouped_contiguous);
    auto it = best_configs.find(key);
    if (it != best_configs.end()) {
        return it->second;
    }

    int block_m;
    if (!is_grouped_contiguous && m <= 64) {
        block_m = 64;
    } else {
        block_m = 128;
    }

    int best_block_m = -1, best_block_n = -1;
    for (int block_n: std::vector<int>({16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160})) {
        bool success = false;
        if (best_block_m == -1 || best_block_n == -1) {
            success = true;
        } else {
            int num_waves = getNumWaves(m, n, block_m, block_n, num_groups, num_sms);
            int best_num_waves = getNumWaves(m, n, best_block_m, best_block_n, num_groups, num_sms);
            if (num_waves < best_num_waves) {
                success = true;
            } else if (num_waves == best_num_waves) {
                int util = getLastWaveUtil(m, n, block_m, block_n, num_groups, num_sms);
                int best_util = getLastWaveUtil(m, n, best_block_m, best_block_n, num_groups, num_sms);
                success = tie(util, block_m, best_block_n) > tie(best_util, best_block_m, block_n);
            }
        }
        if (success) {
            best_block_m = block_m; best_block_n = block_n;
        }
    }
    FT_CHECK_WITH_INFO(best_block_m != -1, "block m size cannot be None in best config");
    FT_CHECK_WITH_INFO(best_block_n != -1, "block n size cannot be None in best config");
    int best_num_stages = -1, best_smem_size = -1;
    const int sm90_capacitty = 232448;
    vector<int> num_stages_vec;
    if ((128 % best_block_n) && (128 / __gcd(128, best_block_n) <= 4)) {
        num_stages_vec = vector<int>({4});
    } else {
        num_stages_vec = vector<int>({8, 7, 6, 5, 4});
    }
    for (auto& num_stages: num_stages_vec) {
        best_smem_size = getSmemSize(num_stages, k, best_block_m, best_block_n);
        if (best_smem_size <= sm90_capacitty) {
            best_num_stages = num_stages;
            break;
        }
    }
    FT_CHECK_WITH_INFO(best_num_stages != -1, "stages num cannot be None in best config");

    int best_num_tma_multicast = 1; bool is_tma_multicast_on_a = true;

    vector<bool> is_multicast_legal = best_block_m > best_block_n ?
        vector<bool>({isTmaMulticastLegal(n, best_block_n, 2, num_sms), isTmaMulticastLegal(m, best_block_m, 2, num_sms)}):
        vector<bool>({isTmaMulticastLegal(m, best_block_m, 2, num_sms), isTmaMulticastLegal(n, best_block_n, 2, num_sms)});

    for (int index = 0; index < 2; ++index) {
        if (m >= 512 && is_multicast_legal[index] && num_groups == 1) {
            best_num_tma_multicast = 2; is_tma_multicast_on_a = (index == (best_block_m > best_block_n? 0: 1));
            break;
        }
    }

    auto num_waves = getNumWaves(m, n, best_block_m, best_block_n, num_groups, num_sms);
    auto num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves);
    num_min_sms = ceil_div(max(num_min_sms, num_sms - 8), best_num_tma_multicast) * best_num_tma_multicast;

    FT_CHECK_WITH_INFO(num_min_sms <= num_sms, "num_min_sms(%d) should not less than num_sms(%d)", num_min_sms, num_sms);
    DeepGemmConfig value = DeepGemmConfig(num_min_sms, best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, is_tma_multicast_on_a, best_smem_size);
    best_configs.emplace(key, value);
    return value;
}

#define DISPATCH_DEEP_GEMM(N, K, GROUP_NUM, GEMM_TYPE) \
    if (n == N && k == K && num_groups == GROUP_NUM && gemm_type == GEMM_TYPE) { \
        dispatchBlockNK<N, K, GROUP_NUM, GEMM_TYPE>(output, lhs, lhs_scale, rhs, rhs_scale, grouped_layout, m, bm, bn, bk, num_stages, num_tma_multicast, is_tma_multicast_on_a, stream, num_sms, smem_size); \
        return; \
    }

void runDeepGemm(__nv_bfloat16*         output,
                 __nv_fp8_e4m3*         lhs,
                 float*                 lhs_scale,
                 __nv_fp8_e4m3*         rhs,
                 float*                 rhs_scale,
                 int*                   grouped_layout,
                 uint32_t               m,
                 uint32_t               n,
                 uint32_t               k,
                 uint32_t               bm,
                 uint32_t               bn,
                 uint32_t               bk,
                 uint32_t               num_groups,
                 uint32_t               num_stages,
                 uint32_t               num_tma_multicast,
                 bool                   is_tma_multicast_on_a,
                 DeepGemmType           gemm_type,
                 cudaStream_t           stream,
                 uint32_t               num_sms,
                 uint32_t               smem_size)
{
    FT_LOG_DEBUG("m:%u, n:%u, k:%u , bm:%u, bn:%u, bk:%u, num_groups:%u, num_stages:%u, num_tma_multicast:%u\n, is_tma_multicast_on_a:%u", m, n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, is_tma_multicast_on_a);

    // Normal Gemm
    DISPATCH_DEEP_GEMM(2112, 7168, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(4096, 7168, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(7168, 2048, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(2048, 7168, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(16384, 512, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(24576, 1536, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(7168, 16384, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(18432, 7168, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(7168, 18432, 1, DeepGemmType::Normal)

    // tp 8
    DISPATCH_DEEP_GEMM(3072, 1536, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(2048, 512, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(2304, 7168, 1, DeepGemmType::Normal)
    DISPATCH_DEEP_GEMM(7168, 2304, 1, DeepGemmType::Normal)

    // Grouped Contiguous
    DISPATCH_DEEP_GEMM(4096, 7168, 256, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 4096, 256, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 2048, 256, DeepGemmType::GroupedContiguous)

    DISPATCH_DEEP_GEMM(4096, 7168, 128, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 4096, 128, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 2048, 128, DeepGemmType::GroupedContiguous)

    DISPATCH_DEEP_GEMM(4096, 7168, 8, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 4096, 8, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 2048, 8, DeepGemmType::GroupedContiguous)

    DISPATCH_DEEP_GEMM(4096, 7168, 64, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 4096, 64, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 2048, 64, DeepGemmType::GroupedContiguous)

    DISPATCH_DEEP_GEMM(4096, 7168, 32, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 4096, 32, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 2048, 32, DeepGemmType::GroupedContiguous)

    DISPATCH_DEEP_GEMM(4096, 7168, 2, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 4096, 2, DeepGemmType::GroupedContiguous)
    DISPATCH_DEEP_GEMM(7168, 2048, 2, DeepGemmType::GroupedContiguous)

    // masked
    DISPATCH_DEEP_GEMM(4096, 7168, 256, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 4096, 256, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 2048, 256, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(4096, 7168, 128, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 4096, 128, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 2048, 128, DeepGemmType::GroupedMasked)

    DISPATCH_DEEP_GEMM(4096, 7168, 32, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 4096, 32, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 2048, 32, DeepGemmType::GroupedMasked)

    DISPATCH_DEEP_GEMM(4096, 7168, 64, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 4096, 64, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 2048, 64, DeepGemmType::GroupedMasked)

    DISPATCH_DEEP_GEMM(4096, 7168, 2, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 4096, 2, DeepGemmType::GroupedMasked)
    DISPATCH_DEEP_GEMM(7168, 2048, 2, DeepGemmType::GroupedMasked)
    FT_FAIL("DISPATCH_DEEP_GEMM(N=%u, K=%u, NUM_GROUPS=%u, GEMM_TYPE=%u) no template found", n, k, num_groups, gemm_type);
}
#endif

void DeepGemmPlugin::gemmFp8(const Buffer &lhs, const Buffer &rhs, Buffer &output, cudaStream_t stream) {
#ifdef ENABLE_FP8
    // lhs.fp8 e4m3, [m, k]; scales -> fp32, [m, k / 128]
    // rhs.fp8 e4m3, [n, k]; scales -> fp32, [n / 128, k / 128]
    // output.bf16, [m, n]
    size_t m, n, k;
    m = lhs.shape()[0]; k = lhs.shape()[1]; n = rhs.size() / k;
    FT_CHECK_WITH_INFO(n % 64 == 0 && k % 128 == 0, "n(%d) % 64 or k(%d) % 128 != 0", n, k);
    FT_LOG_DEBUG("lhs:%s, scale:%s, rhs:%s, scale:%s out:%s",
		lhs.debugString().c_str(),
		reinterpret_cast<const QBuffer&>(lhs).scales().debugString().c_str(),
		rhs.debugString().c_str(),
		reinterpret_cast<const QBuffer&>(rhs).scales().debugString().c_str(),
		output.debugString().c_str());
    int num_sms = getNumSms();
    auto best_config = getBestConfig(m, n, k, 1, num_sms);

    runDeepGemm(output.data<__nv_bfloat16>(),
                reinterpret_cast<const QBuffer&>(lhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(lhs).scales().data<float>(),
                reinterpret_cast<const QBuffer&>(rhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(rhs).scalesData<float>(),
                nullptr, // grouped_layout
                m,
                n,
                k,
                best_config.block_m,
                best_config.block_n,
                128, // block_k
                1,   // num_groups
                best_config.num_stages,
                best_config.num_tma_multicast,
                best_config.is_tma_multicast_on_a,
                DeepGemmType::Normal,
                stream,
                best_config.num_sms,
                best_config.smem_size);
#endif
}

void DeepGemmPlugin::groupedGemmFp8Contiguous(const Buffer &lhs, const Buffer &rhs, Buffer &output, const Buffer &m_indices, cudaStream_t stream) {
#ifdef ENABLE_FP8
    // lhs.fp8 e4m3, [m_sum, k]; scales -> fp32, [m_sum, k / 128]
    // rhs.fp8 e4m3, [num_groups, n, k]; scales -> fp32, [num_groups, n / 128, k / 128]
    // output.bf16, [m_sum, n]
    // m_indices -> int32, [m_sum]
    size_t m, n, k;
    m = lhs.shape()[0]; k = lhs.shape()[1]; n = rhs.shape()[1];
    int num_groups = rhs.shape()[0];
    FT_CHECK_WITH_INFO(n % 64 == 0 && k % 128 == 0, "n(%d) % 64 or k(%d) % 128 != 0", n, k);

    auto lhs_scales = getColMajorTmaAlignedTensor(reinterpret_cast<const QBuffer&>(lhs).scales());
    int num_sms = getNumSms();

    auto best_config = getBestConfig(m, n, k, 1, num_sms, true);

    runDeepGemm(output.data<__nv_bfloat16>(),
                reinterpret_cast<const QBuffer&>(lhs).kernel().data<__nv_fp8_e4m3>(),
                (float*)lhs_scales.data_ptr(),
                reinterpret_cast<const QBuffer&>(rhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(rhs).scalesData<float>(),
                m_indices.data<int>(), // grouped_layout
                m,
                n,
                k,
                best_config.block_m,
                best_config.block_n,
                128, // block_k
                num_groups,   // num_groups
                best_config.num_stages,
                best_config.num_tma_multicast,
                best_config.is_tma_multicast_on_a,
                DeepGemmType::GroupedContiguous,
                stream,
                best_config.num_sms,
                best_config.smem_size);
#endif
}

void DeepGemmPlugin::groupedGemmFp8Masked(const Buffer &lhs, const Buffer &rhs, Buffer &output, const Buffer &masked_m, int expected_m, cudaStream_t stream) {
#ifdef ENABLE_FP8
    // lhs.fp8 e4m3, [num_groups, m_max, k]; scales -> fp32, [num_groups, m_max, k / 128]
    // rhs.fp8 e4m3, [num_groups, n, k]; scales -> fp32, [num_groups, n / 128, k / 128]
    // output.bf16, [m, n]
    // masked_m -> int32, [num_groups]
    size_t m, n, k;
    m = lhs.shape()[1]; k = lhs.shape()[2]; n = rhs.shape()[1];
    int num_groups = rhs.shape()[0];
    FT_CHECK_WITH_INFO(n % 64 == 0 && k % 128 == 0, "n(%ld) % 64 or k(%ld) % 128 != 0", n, k);

    int num_sms = getNumSms();

    auto best_config = getBestConfig(m, n, k, num_groups, num_sms);

    runDeepGemm(output.data<__nv_bfloat16>(),
                reinterpret_cast<const QBuffer&>(lhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(lhs).scalesData<float>(),
                reinterpret_cast<const QBuffer&>(rhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(rhs).scalesData<float>(),
                masked_m.data<int>(), // grouped_layout
                m,
                n,
                k,
                best_config.block_m,
                best_config.block_n,
                128, // block_k
                num_groups,   // num_groups
                best_config.num_stages,
                best_config.num_tma_multicast,
                best_config.is_tma_multicast_on_a,
                DeepGemmType::GroupedMasked,
                stream,
                best_config.num_sms,
                best_config.smem_size);
#endif
}
} // namespace fastertransformer
