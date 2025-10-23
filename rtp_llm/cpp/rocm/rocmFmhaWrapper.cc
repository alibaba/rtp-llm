#include "rocmFmhaWrapper.h"
#include "mha_fwd.h"
#include "torch/mha_fwd.h"
#include "ck_tile/host.hpp"
#include "utils.hpp"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/01_fmha/mask.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/01_fmha/fmha_fwd.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/01_fmha/utils.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/01_fmha/bias.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/include/ck_tile/host.hpp"

namespace rtp_llm {
// ===== helpers: dump & checks (put once in this file top if reused) =====
#include <hip/hip_runtime.h>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

// ---- FP8 解码：E4M3fnuz ----
static inline float fp8_e4m3fnuz_to_float(uint8_t x) {
    if (x == 0) return 0.0f;
    uint8_t sign = x >> 7;
    uint8_t exp  = (x >> 3) & 0xF;
    uint8_t mant = x & 0x7;
    // fnuz: 没有 NaN/Inf，最大指数位按最大有限值处理
    if (exp == 0xF) { exp = 0xE; mant = 0x7; }
    const int bias = 7;
    float val;
    if (exp == 0) {
        int e = 1 - bias;                 // -6
        float m = float(mant) / 8.0f;     // 无隐含1
        val = std::ldexp(m, e);
    } else {
        int e = int(exp) - bias;
        float m = 1.0f + float(mant) / 8.0f; // 有隐含1
        val = std::ldexp(m, e);
    }
    return sign ? -val : val;
}

// ---- 可选：FP8 解码 E5M2fnuz ----
static inline float fp8_e5m2fnuz_to_float(uint8_t x) {
    if (x == 0) return 0.0f;
    uint8_t sign = x >> 7;
    uint8_t exp  = (x >> 2) & 0x1F;
    uint8_t mant = x & 0x3;
    if (exp == 0x1F) { exp = 0x1E; mant = 0x3; }
    const int bias = 15;
    float val;
    if (exp == 0) {
        int e = 1 - bias;                 // -14
        float m = float(mant) / 4.0f;
        val = std::ldexp(m, e);
    } else {
        int e = int(exp) - bias;
        float m = 1.0f + float(mant) / 4.0f;
        val = std::ldexp(m, e);
    }
    return sign ? -val : val;
}

// ---- f32 <-> bf16 位级转换（RNE 舍入）----
static inline uint16_t f32_to_bf16_bits(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    // round-to-nearest-even: add 0x7FFF + lsb of high 16 bits
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    uint16_t b = uint16_t((u + rounding_bias) >> 16);
    return b;
}
static inline float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = (uint32_t(b) << 16);
    float f; std::memcpy(&f, &u, 4);
    return f;
}

enum class Fp8Kind { E4M3FNUZ, E5M2FNUZ };

// ---- 主函数：从 device 取 FP8 字节 → 解码 → 量化为 BF16 → 再转 FP32 打印 ----
static inline void dump_fp8_as_f32_via_bf16_dev(const void* dptr,
                                                size_t n_bytes,
                                                const char* tag,
                                                Fp8Kind kind = Fp8Kind::E4M3FNUZ,
                                                float dequant_scale = 1.0f) {
    size_t k = std::min(n_bytes, size_t(32));  // 打印前 32B
    std::vector<uint8_t> u8(k);
    hipMemcpy(u8.data(), dptr, k, hipMemcpyDeviceToHost);

    // 原始 u8
    {
        std::ostringstream oss;
        oss << "[" << tag << " u8 first " << k << "B]: ";
        for (size_t i = 0; i < k; ++i)
            oss << std::hex << std::setw(2) << std::setfill('0') << (int)u8[i] << " ";
        std::cout << oss.str() << std::dec << std::endl;
    }

    // 解码 -> *scale -> 量化到 BF16 位型 -> 还原为 FP32（bf16->fp32）用于打印/统计
    std::vector<uint16_t> bf16_bits(k);
    std::vector<float>    f32_from_bf16(k);

    for (size_t i = 0; i < k; ++i) {
        float f = (kind == Fp8Kind::E4M3FNUZ)
                    ? fp8_e4m3fnuz_to_float(u8[i])
                    : fp8_e5m2fnuz_to_float(u8[i]);
        f *= dequant_scale;                 // 应用 per-tensor scale
        uint16_t b = f32_to_bf16_bits(f);   // 量化到 BF16（RNE）
        bf16_bits[i]   = b;
        f32_from_bf16[i] = bf16_bits_to_f32(b); // bf16 -> fp32 便于输出和计算
    }

    // 打印 bf16 十六进制位型
    {
        std::ostringstream oss;
        oss << "[" << tag << " bf16 hex first " << k << "]: ";
        for (size_t i = 0; i < k; ++i)
            oss << "0x" << std::hex << std::setw(4) << std::setfill('0') << bf16_bits[i] << " ";
        std::cout << oss.str() << std::dec << std::endl;
    }
    // 打印 bf16->fp32 数值
    {
        std::ostringstream oss;
        oss << "[" << tag << " bf16->f32 first " << k << "]: ";
        for (size_t i = 0; i < k; ++i)
            oss << f32_from_bf16[i] << " ";
        std::cout << oss.str() << std::endl;
    }

    // 统计
    float mn = +INFINITY, mx = -INFINITY, sum = 0.f;
    for (float v : f32_from_bf16) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
    std::cout << "[" << tag << " decoded(bf16->f32) stats] "
              << "min=" << mn << " max=" << mx
              << " mean=" << (sum / float(k))
              << " (scale=" << dequant_scale
              << ", kind=" << (kind==Fp8Kind::E4M3FNUZ?"E4M3fnuz":"E5M2fnuz")
              << ")\n";
}



static inline bool isAligned(const void* p, size_t a) {
    return (reinterpret_cast<uintptr_t>(p) % a) == 0;
}
static inline float bf16_to_f32(uint16_t u) {
    uint32_t t = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &t, sizeof(f));
    return f;
}
static inline void dump_bytes_u8_dev(const void* dptr, size_t n, const char* tag) {
    size_t k = std::min(n, size_t(32));
    std::vector<uint8_t> h(k);
    hipMemcpy(h.data(), dptr, k, hipMemcpyDeviceToHost);
    std::ostringstream oss;
    oss << "[" << tag << " u8 first " << k << "B]: ";
    for (size_t i = 0; i < k; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)h[i] << " ";
    }
    std::cout << oss.str() << std::dec << std::endl;

    uint8_t mn=255, mx=0; double sum=0.0;
    for (auto v: h){ if(v<mn) mn=v; if(v>mx) mx=v; sum+=v; }
    std::cout << "[" << tag << " stats] min=" << (int)mn
              << " max=" << (int)mx << " mean=" << (sum / (double)k) << std::endl;
}
static inline void dump_bf16_dev(const void* dptr, size_t n_elts, const char* tag) {
    size_t k = std::min(n_elts, size_t(8));
    std::vector<uint16_t> h(k);
    hipMemcpy(h.data(), dptr, k * sizeof(uint16_t), hipMemcpyDeviceToHost);
    std::ostringstream oss_hex, oss_f;
    oss_hex << "[" << tag << " bf16 hex first " << k << "]: ";
    oss_f   << "[" << tag << " bf16->f32 first " << k << "]: ";
    for (size_t i = 0; i < k; ++i) {
        oss_hex << "0x" << std::hex << std::setw(4) << std::setfill('0') << h[i] << " ";
        oss_f   << std::dec << bf16_to_f32(h[i]) << " ";
    }
    std::cout << oss_hex.str() << std::dec << std::endl;
    std::cout << oss_f.str() << std::endl;
}
// ======================================================================

static void throwCKError(const char* const file, int const line, std::string const& info = "") {
    auto error_msg =
        std::string("[CK][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    std::printf("%s", error_msg.c_str());
    fflush(stdout);
    fflush(stderr);
    abort();
    throw std::exception();
}
#define CK_FAIL(info, ...) throwCKError(__FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__))
uint32_t rocmFmhaWrapper::runCKFmha(void*  q,
                                    void*  k,
                                    void*  v,
                                    void*  output,
                                    void*  softmax_lse_,
                                    size_t batch_size,
                                    size_t seq_len,
                                    size_t max_prefix_prompt_length,
                                    void*  seqstart_q,
                                    void*  seqstart_k,
                                    void*  lse_acc_buf,
                                    void*  linear_bias_slopes,
                                    void*  biasBuffer,
                                    bool   i_perm_,  // if true, will be batch * nhead * seqlen * hdim
                                    bool   o_perm_) {  // if false, will be batch * seqlen * nhead * hdim

    // map parms from FT to CK
    mode_enum mode      = mode_enum::group;
    auto      data_type = getDataTypeStr(dtype_);
    auto      batch     = static_cast<ck_tile::index_t>(batch_size);
    auto      nhead     = static_cast<ck_tile::index_t>(head_num_);
    auto      nhead_k   = static_cast<ck_tile::index_t>(kv_head_num_);
    if (nhead_k < 0)
        nhead_k = nhead;

    if (nhead % nhead_k != 0) {
        std::cerr << "nhead:" << nhead << " must be multiple of nhead_k:" << nhead_k << std::endl;
        return false;
    }

    ck_tile::index_t seqlen_q = seq_len;
    ck_tile::index_t seqlen_k = seq_len + max_prefix_prompt_length;
    if (seqlen_k < 0)
        seqlen_k = seqlen_q;
    // auto [seqlen_qs, seqlen_ks, seqlen_kpads] = decode_seqlen(mode_enum::batch,
    //                                                           batch,
    //                                                           std::to_string(seqlen_q),
    //                                                           std::to_string(seqlen_k),
    //                                                           "-1");

    ck_tile::index_t hdim_q = size_per_head_;
    ck_tile::index_t hdim_v = size_per_head_;
    if (hdim_v < 0)
        hdim_v = hdim_q;

    // the output of add_fusedQKV_bias_transpose_kernel:
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.
    bool  i_perm  = i_perm_;  // if true, will be batch * nhead * seqlen * hdim
    bool  o_perm  = o_perm_;  // if false, will be batch * seqlen * nhead * hdim
    float scale_s = 0.f;
    if (scale_s == .0f)
        scale_s = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q));  // TODO: q ? v ?

    auto  squant          = false;
    float scale_p         = 1.f;
    float scale_o         = 1.f;
    float logits_soft_cap = 0.f;
    // if (squant) {
    //     scale_s = scale_s * (range_q / dtype_max) * (range_k / dtype_max);
    //     scale_p = dtype_max / range_p;
    //     // scale_p = [max(fp8_t)/range_o] * [range_p/max(fp8_t)] * [range_v/max(fp8_t)]
    //     scale_o = range_p * range_v / range_o / dtype_max;
    // }

    bool is_v_rowmajor = true;
    auto lse           = softmax_lse_ ? true : false;

    std::string msk_str;
    if (mtype_ == AttentionMaskType::noMask) {
        msk_str = "0";
    } else if (mtype_ == AttentionMaskType::causalMask) {
        msk_str = "b";
        // RTP_LLM_LOG_INFO("Using causal_bottom_right Mask");
    } else {
        RTP_LLM_LOG_ERROR("Mask type not supported");
    }

    bias_info bias = bias_info::decode(linear_bias_slopes ? "a" : "n");
    mask_info mask = mask_info::decode(msk_str, seqlen_q, seqlen_k);  // TODO: we don't need x/y anymore

    float    p_drop      = 0.;
    uint64_t drop_seed   = 1.;
    uint64_t drop_offset = 0.;
    bool     drop_prefs  = false;
    if (p_drop < 0.0f || p_drop > 1.0f) {
        std::cerr << "The value of p_drop should be 0~1" << std::endl;
        return false;
    }

    bool s_randval = false;
    if (p_drop > 0.0f) {
        s_randval = true;
    }

    int                    num_splits   = 1;
    const ck_tile::index_t max_seqlen_q = seqlen_q;  // max of all batch
    const ck_tile::index_t max_seqlen_k = seqlen_k;

    // host memory for storing all the tensor elements
    const ck_tile::index_t shape_batch    = (mode == mode_enum::batch ? batch : 1);
    const ck_tile::index_t shape_seqlen_q = (mode == mode_enum::batch ? seqlen_q : seqlen_q);
    const ck_tile::index_t shape_seqlen_k = (mode == mode_enum::batch ? seqlen_k : seqlen_k);
    const ck_tile::index_t seqlen_knew    = 0;

    ck_tile::HostTensor<ck_tile::half_t> lse_acc_host(
        1 < num_splits ? std::array<ck_tile::index_t, 4>{num_splits, batch, nhead, max_seqlen_q} :
                         std::array<ck_tile::index_t, 4>{1, 1, 1, 1});
    if (lse_acc_buf == nullptr) {
        // printf("[CK] size = %d\n", lse_acc_host.get_element_space_size_in_bytes());
        return lse_acc_host.get_element_space_size_in_bytes();
    }
    // ck_tile::DeviceMem lse_acc_buf(lse_acc_host.get_element_space_size_in_bytes());
    auto has_logits_soft_cap = false;

    // auto fmha_traits = fmha_fwd_traits{hdim_q,
    //                                    hdim_v,
    //                                    data_type,
    //                                    mode == mode_enum::group,
    //                                    is_v_rowmajor,
    //                                    has_logits_soft_cap,
    //                                    mask.type,
    //                                    bias.type,
    //                                    lse,
    //                                    p_drop > 0.0f,
    //                                    squant};

    auto fmha_args = [&]() {
        assert(nhead % nhead_k == 0);
        // QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head]
        const ck_tile::index_t stride_q    = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
        const ck_tile::index_t stride_k    = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
        const ck_tile::index_t stride_knew = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
        const ck_tile::index_t stride_v    = [&]() {
            if (is_v_rowmajor)
                return i_perm ? hdim_v : (nhead + 2 * nhead_k) * hdim_v;
            else
                return i_perm ? shape_seqlen_k : (nhead + 2 * nhead_k) * shape_seqlen_k;
        }();
        const ck_tile::index_t stride_vnew = [&]() {
            if (is_v_rowmajor)
                return i_perm ? hdim_v : (nhead + 2 * nhead_k) * hdim_v;
            else
                return i_perm ? seqlen_knew : (nhead + 2 * nhead_k) * seqlen_knew;
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_o_acc   = hdim_v;
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q    = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k    = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_knew = (i_perm ? seqlen_knew * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v    = [&]() {
            if (is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_vnew = [&]() {
            if (is_v_rowmajor)
                return i_perm ? seqlen_knew * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * seqlen_knew : seqlen_knew;
        }();
        const ck_tile::index_t nhead_stride_bias = (i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_lse_acc = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_o_acc   = (max_seqlen_q * hdim_v);
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q       = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k       = (nhead_k * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_knew    = (nhead_k * seqlen_knew * hdim_q);
        const ck_tile::index_t batch_stride_v       = (nhead_k * hdim_v * shape_seqlen_k);
        const ck_tile::index_t batch_stride_vnew    = (nhead_k * hdim_v * seqlen_knew);
        const ck_tile::index_t batch_stride_bias    = (0 * nhead * shape_seqlen_q * shape_seqlen_k);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_lse_acc = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_o_acc   = (nhead * max_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_o       = (nhead * shape_seqlen_q * hdim_v);
        // setup split_stride_* arguments (only used in split-kv kernel)
        const ck_tile::index_t split_stride_lse_acc = (shape_batch * nhead * shape_seqlen_q);
        const ck_tile::index_t split_stride_o_acc   = (batch * nhead * max_seqlen_q * hdim_v);
        const ck_tile::index_t min_seqlen_q         = 0;

// // === DEBUG: dump fmha_fwd_args effective arguments (exactly matches return order) ===
// {
//     // 指针类实参（与 return 顺序一致）
//     const void* arg_q                  = q;
//     const void* arg_k                  = k;
//     const void* arg_v                  = v;
//     const void* arg_bias               = (bias.type == bias_enum::alibi) ? linear_bias_slopes : biasBuffer;
//     const void* arg_dropout_randval    = nullptr;                 // 你 return 里就是 nullptr
//     const void* arg_softmax_lse        = softmax_lse_;            // 参数名就是 softmax_lse_
//     const void* arg_out                = output;
//     const void* arg_seqstart_q         = nullptr;                 // 你 return 里传的是 nullptr（cu_seqlen_q_ptr）
//     const void* arg_seqstart_k         = nullptr;                 // 你 return 里传的是 nullptr（cu_seqlen_kv_ptr）
//     const void* arg_linear_bias_slopes = nullptr;                 // 你 return 里对应位置传的是 seqstart_q（已上移到前面），此处保持 nullptr 便于区分

//     // 长度/批次
//     const ck_tile::index_t arg_seqlen_q     = shape_seqlen_q;
//     const ck_tile::index_t arg_seqlen_k     = shape_seqlen_k;
//     const ck_tile::index_t arg_batch        = batch;
//     const ck_tile::index_t arg_max_seqlen_q = max_seqlen_q;

//     // 维度/头
//     const ck_tile::index_t arg_hdim_q  = hdim_q;
//     const ck_tile::index_t arg_hdim_v  = hdim_v;
//     const ck_tile::index_t arg_nhead   = nhead;
//     const ck_tile::index_t arg_nhead_k = nhead_k;

//     // scale / soft-cap
//     const float arg_scale_s         = scale_s;        // 你上面算的 1/sqrt(hdim_q)
//     const float arg_scale_p         = 1.f;
//     const float arg_scale_o         = 1.f;
//     const float arg_logits_soft_cap = logits_soft_cap;

//     // strides（与 return 的顺序保持一致）
//     const ck_tile::index_t arg_stride_q        = stride_q;
//     const ck_tile::index_t arg_stride_k        = stride_k;
//     const ck_tile::index_t arg_stride_v        = stride_v;
//     const ck_tile::index_t arg_stride_bias     = (bias.type == bias_enum::alibi) ? ((bias.rank_info == 0) ? 0 : nhead) : stride_bias;
//     const ck_tile::index_t arg_stride_randval  = stride_randval;
//     const ck_tile::index_t arg_stride_o        = stride_o;

//     const ck_tile::index_t arg_nhead_stride_q      = nhead_stride_q;
//     const ck_tile::index_t arg_nhead_stride_k      = nhead_stride_k;
//     const ck_tile::index_t arg_nhead_stride_v      = nhead_stride_v;
//     const ck_tile::index_t arg_nhead_stride_bias   = nhead_stride_bias;     // 你这里本来就算成 0
//     const ck_tile::index_t arg_nhead_stride_randval= nhead_stride_randval;
//     const ck_tile::index_t arg_nhead_stride_lse    = nhead_stride_lse;
//     const ck_tile::index_t arg_nhead_stride_o      = nhead_stride_o;

//     const ck_tile::index_t arg_batch_stride_q      = batch_stride_q;
//     const ck_tile::index_t arg_batch_stride_k      = batch_stride_k;
//     const ck_tile::index_t arg_batch_stride_v      = batch_stride_v;
//     const ck_tile::index_t arg_batch_stride_bias   = batch_stride_bias;     // 你这里本来就算成 0
//     const ck_tile::index_t arg_batch_stride_randval= batch_stride_randval;
//     const ck_tile::index_t arg_batch_stride_lse    = batch_stride_lse;
//     const ck_tile::index_t arg_batch_stride_o      = batch_stride_o;

//     // mask / dropout
//     const ck_tile::index_t arg_mask_left   = mask.left;
//     const ck_tile::index_t arg_mask_right  = mask.right;
//     const ck_tile::index_t arg_mask_type   = static_cast<ck_tile::index_t>(mask.type);
//     const ck_tile::index_t arg_min_seqlen_q= min_seqlen_q;        // 你上面算的是 0

//     const float arg_p_dropout           = p_drop;                  // 变量名在本函数里是 p_drop
//     const bool  arg_has_dropout_randval = s_randval;               // 变量名在本函数里是 s_randval
//     const unsigned long long arg_drop_seed_val   = static_cast<unsigned long long>(drop_seed);
//     const unsigned long long arg_drop_offset_val = static_cast<unsigned long long>(drop_offset);

//     // === 打印（严格对齐 return 的语义与顺序）===
//     std::cout << "[fmha_fwd_args] ptrs: "
//               << "q=" << arg_q
//               << " k=" << arg_k
//               << " v=" << arg_v
//               << " bias=" << arg_bias
//               << " dropout_randval=" << arg_dropout_randval
//               << " softmax_lse=" << arg_softmax_lse
//               << " out=" << arg_out
//               << " seqstart_q=" << arg_seqstart_q
//               << " seqstart_k=" << arg_seqstart_k
//               << " linear_bias_slopes=" << arg_linear_bias_slopes
//               << std::endl;

//     std::cout << "[fmha_fwd_args] lens/batch: "
//               << "seqlen_q=" << arg_seqlen_q
//               << " seqlen_k=" << arg_seqlen_k
//               << " batch=" << arg_batch
//               << " max_seqlen_q=" << arg_max_seqlen_q
//               << std::endl;

//     std::cout << "[fmha_fwd_args] dims/heads: "
//               << "hdim_q=" << arg_hdim_q
//               << " hdim_v=" << arg_hdim_v
//               << " nhead=" << arg_nhead
//               << " nhead_k=" << arg_nhead_k
//               << std::endl;

//     std::cout << "[fmha_fwd_args] scales: "
//               << "scale_s=" << arg_scale_s
//               << " scale_p=" << arg_scale_p
//               << " scale_o=" << arg_scale_o
//               << " logits_soft_cap=" << arg_logits_soft_cap
//               << std::endl;

//     std::cout << "[fmha_fwd_args] strides: "
//               << "q=" << arg_stride_q
//               << " k=" << arg_stride_k
//               << " v=" << arg_stride_v
//               << " bias=" << arg_stride_bias
//               << " randval=" << arg_stride_randval
//               << " o=" << arg_stride_o
//               << std::endl;

//     std::cout << "[fmha_fwd_args] nhead_strides: "
//               << "q=" << arg_nhead_stride_q
//               << " k=" << arg_nhead_stride_k
//               << " v=" << arg_nhead_stride_v
//               << " bias=" << arg_nhead_stride_bias
//               << " randval=" << arg_nhead_stride_randval
//               << " lse=" << arg_nhead_stride_lse
//               << " o=" << arg_nhead_stride_o
//               << std::endl;

//     std::cout << "[fmha_fwd_args] batch_strides: "
//               << "q=" << arg_batch_stride_q
//               << " k=" << arg_batch_stride_k
//               << " v=" << arg_batch_stride_v
//               << " bias=" << arg_batch_stride_bias
//               << " randval=" << arg_batch_stride_randval
//               << " lse=" << arg_batch_stride_lse
//               << " o=" << arg_batch_stride_o
//               << std::endl;

//     std::cout << "[fmha_fwd_args] mask/dropout: "
//               << "mask.left=" << arg_mask_left
//               << " mask.right=" << arg_mask_right
//               << " mask.type=" << arg_mask_type
//               << " min_seqlen_q=" << arg_min_seqlen_q
//               << " p_drop=" << arg_p_dropout
//               << " has_randval=" << (arg_has_dropout_randval ? 1 : 0)
//               << " drop_seed=" << arg_drop_seed_val
//               << " drop_offset=" << arg_drop_offset_val
//               << std::endl;
// }
// // === DEBUG END ===

// // === EXTRA DEBUG: dtype/layout/alignment + sample values ===
// {
//     // 1) 元信息：dtype / layout / mode
//     std::cout << "[fmha_meta] dtype=" << data_type
//               << " i_perm=" << (i_perm ? 1 : 0)
//               << " o_perm=" << (o_perm ? 1 : 0)
//               << " is_v_rowmajor=" << (is_v_rowmajor ? 1 : 0)
//               << " mode=" << (mode == mode_enum::group ? "group" : "batch")
//               << " lse=" << (softmax_lse_ ? 1 : 0)
//               << std::endl;

//     // 2) 指针对齐
//     auto print_align = [&](const char* name, const void* p){
//         std::cout << "[align] " << name
//                   << " align16=" << (isAligned(p,16)?1:0)
//                   << " align32=" << (isAligned(p,32)?1:0)
//                   << " align64=" << (isAligned(p,64)?1:0)
//                   << std::endl;
//     };
//     print_align("q", q); print_align("k", k); print_align("v", v);
//     print_align("out", output); if(softmax_lse_) print_align("lse", softmax_lse_);

//     // 3) stride 期望 vs 实际（帮助识别维度排列是否对齐）
//     //    这里对 q 的期望 stride 进行对比，你也可以按需添加 k/v/out
//     ck_tile::index_t expect_stride_q = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
//     if (expect_stride_q != stride_q) {
//         std::cout << "[stride_mismatch] q: expect=" << expect_stride_q
//                   << " actual=" << stride_q << std::endl;
//     }
//     ck_tile::index_t expect_stride_v = 0;
//     if (is_v_rowmajor)
//         expect_stride_v = (i_perm ? hdim_v : (nhead + 2 * nhead_k) * hdim_v);
//     else
//         expect_stride_v = (i_perm ? shape_seqlen_k : (nhead + 2 * nhead_k) * shape_seqlen_k);
//     if (expect_stride_v != stride_v) {
//         std::cout << "[stride_mismatch] v: expect=" << expect_stride_v
//                   << " actual=" << stride_v << std::endl;
//     }

//     // 4) 采样打印数据（前若干元素）
//     //    - BF16: 打印 hex + 转 float
//     //    - FP8 : 打印 raw u8 的 hex + min/max/mean
//     const bool is_bf16 = (data_type.find("bf16") != std::string::npos);
//     const bool is_fp8  = (data_type.find("fp8")  != std::string::npos);

//     if (is_bf16 && !is_fp8) {
//         // 每个 head 的向量维度 = hdim_q/v，取第 1 个 batch、第 1 个 head，第 1 行的起始位置
//         dump_bf16_dev(q, 8, "q");
//         dump_bf16_dev(k, 8, "k");
//         dump_bf16_dev(v, 8, "v");
//         dump_bf16_dev(output, 8, "out");
//     } else if (is_fp8) {
//         dump_bytes_u8_dev(q, 32, "q");
//         dump_bytes_u8_dev(k, 32, "k");
//         dump_bytes_u8_dev(v, 32, "v");
//         dump_bf16_dev(output, 32, "out");

//         dump_fp8_as_f32_via_bf16_dev(q,      32, "q",   Fp8Kind::E4M3FNUZ, 1.0f);
//         dump_fp8_as_f32_via_bf16_dev(k,      32, "k",   Fp8Kind::E4M3FNUZ, 1.0f);
//         dump_fp8_as_f32_via_bf16_dev(v,      32, "v",   Fp8Kind::E4M3FNUZ, 1.0f);
//         dump_bf16_dev(output, 32, "out");

//     } else {
//         // 其他类型：至少给出 raw bytes
//         dump_bytes_u8_dev(q, 32, "q");
//         dump_bytes_u8_dev(k, 32, "k");
//         dump_bytes_u8_dev(v, 32, "v");
//         dump_bytes_u8_dev(output, 32, "out");
//     }
// }

        return fmha_fwd_args{q,
                             k,
                             v,
                             bias.type == bias_enum::alibi ? linear_bias_slopes : biasBuffer,
                             nullptr,  // randval_buf.GetDeviceBuffer(),
                             // lse_acc_buf.GetDeviceBuffer(),  // lse_acc_buf.GetDeviceBuffer(),
                             // lse_acc_buf,  // lse_acc_buf.GetDeviceBuffer(),
                             // nullptr,                        // o_acc_buf.GetDeviceBuffer(),
                             softmax_lse_,
                             output,
                             nullptr, //cu_seqlen_q_ptr
                             nullptr, //cu_seqlen_kv_ptr
                             seqstart_q,
                             seqstart_k,
                             nullptr,  // seqlen_kpads
                             nullptr, //seqstart_padded_q_ptr
                             nullptr, //seqstart_padded_k_ptr
                             shape_seqlen_q,
                             shape_seqlen_k,
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             // num_splits,
                             scale_s,
                             scale_p,
                             scale_o,
                             logits_soft_cap,
                             stride_q,
                             stride_k,
                             stride_v,
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias,
                             stride_randval,
                             // stride_o_acc,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_randval,
                             nhead_stride_lse,
                             // nhead_stride_lse_acc,
                             // nhead_stride_o_acc,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_randval,
                             batch_stride_lse,
                             // batch_stride_lse_acc,
                             // batch_stride_o_acc,
                             batch_stride_o,
                             // split_stride_lse_acc,
                             // split_stride_o_acc,
                             mask.left,
                             mask.right,
                             static_cast<ck_tile::index_t>(mask.type),
                             min_seqlen_q,
                             p_drop,
                             s_randval,
                             std::make_pair(drop_seed, drop_offset)};
    }();

    ck_tile::stream_config stream_config{
        stream_,  // stream_id_
        false,    // time_kernel_
        0,        // log_level_
        0,        // cold_niters_
        1,        // nrepeat_
        // false     //
    };
    float run_time;
    // if (data_type == "bf16" && size_per_head_ == 128 && msk_str == "b")
    //     run_time = aiter::mha_fwd(
    //         fmha_args, stream_config, data_type, mode == mode_enum::group, mask.type, bias.type, lse, true);
    // else
        run_time = aiter::mha_fwd(
            fmha_args, stream_config, data_type, mode == mode_enum::group, mask.type, bias.type, lse, false);
    // std::cout << "\nrun_time for ck fmha_fwd: " << run_time << std::endl;
    if (run_time < 0) {
        CK_FAIL("fmha_fwd faild");
    } else {
        return 1;
    }
}

uint32_t rocmFmhaWrapper::runCKFmhaV2(void*  q,
                                      void*  k,
                                      void*  v,
                                      void*  output,
                                      void*  softmax_lse_,
                                      size_t batch_size,
                                      size_t seq_len,
                                      size_t max_prefix_prompt_length,
                                      void*  seqstart_q,
                                      void*  seqstart_k,
                                      void*  lse_acc_buf,
                                      void*  linear_bias_slopes,
                                      void*  biasBuffer,
                                      size_t token_num,
                                      bool   i_perm_,  // if true, will be batch * nhead * seqlen * hdim
                                      bool   o_perm_) {  // if false, will be batch * seqlen * nhead * hdim

    // map parms from FT to CK
    mode_enum mode      = mode_enum::group;
    auto      data_type = getDataTypeStr(dtype_);
    auto      batch     = static_cast<ck_tile::index_t>(batch_size);
    auto      nhead     = static_cast<ck_tile::index_t>(head_num_);
    auto      nhead_k   = static_cast<ck_tile::index_t>(kv_head_num_);
    if (nhead_k < 0)
        nhead_k = nhead;

    if (nhead % nhead_k != 0) {
        std::cerr << "nhead:" << nhead << " must be multiple of nhead_k:" << nhead_k << std::endl;
        return false;
    }

    ck_tile::index_t seqlen_q = seq_len;
    ck_tile::index_t seqlen_k = seq_len + max_prefix_prompt_length;
    if (seqlen_k < 0)
        seqlen_k = seqlen_q;

    ck_tile::index_t hdim_q = size_per_head_;
    ck_tile::index_t hdim_v = size_per_head_;
    if (hdim_v < 0)
        hdim_v = hdim_q;

    // the output of add_fusedQKV_bias_transpose_kernel:
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.
    bool  i_perm  = i_perm_;  // if true, will be batch * nhead * seqlen * hdim
    bool  o_perm  = o_perm_;  // if false, will be batch * seqlen * nhead * hdim
    float scale_s = 0.f;
    if (scale_s == .0f)
        scale_s = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q));  // TODO: q ? v ?

    auto  squant          = false;
    float scale_p         = 1.f;
    float scale_o         = 1.f;
    float logits_soft_cap = 0.f;
    // if (squant) {
    //     scale_s = scale_s * (range_q / dtype_max) * (range_k / dtype_max);
    //     scale_p = dtype_max / range_p;
    //     // scale_p = [max(fp8_t)/range_o] * [range_p/max(fp8_t)] * [range_v/max(fp8_t)]
    //     scale_o = range_p * range_v / range_o / dtype_max;
    // }

    bool is_v_rowmajor = true;
    auto lse           = softmax_lse_ ? true : false;

    std::string msk_str;
    if (mtype_ == AttentionMaskType::noMask) {
        msk_str = "0";
    } else if (mtype_ == AttentionMaskType::causalMask) {
        msk_str = "b";
        // RTP_LLM_LOG_INFO("Using causal_bottom_right Mask");
    } else {
        RTP_LLM_LOG_ERROR("Mask type not supported");
    }

    bias_info bias = bias_info::decode(linear_bias_slopes ? "a" : "n");
    mask_info mask = mask_info::decode(msk_str, seqlen_q, seqlen_k);  // TODO: we don't need x/y anymore

    float    p_drop      = 0.;
    uint64_t drop_seed   = 1.;
    uint64_t drop_offset = 0.;
    bool     drop_prefs  = false;
    if (p_drop < 0.0f || p_drop > 1.0f) {
        std::cerr << "The value of p_drop should be 0~1" << std::endl;
        return false;
    }

    bool s_randval = false;
    if (p_drop > 0.0f) {
        s_randval = true;
    }

    int                    num_splits   = 1;
    const ck_tile::index_t max_seqlen_q = seqlen_q;  // max of all batch
    const ck_tile::index_t max_seqlen_k = seqlen_k;

    // host memory for storing all the tensor elements
    const ck_tile::index_t               shape_batch    = (mode == mode_enum::batch ? batch : 1);
    const ck_tile::index_t               shape_seqlen_q = (mode == mode_enum::batch ? seqlen_q : seqlen_q * batch);
    const ck_tile::index_t               shape_seqlen_k = (mode == mode_enum::batch ? seqlen_k : seqlen_k * batch);
    const ck_tile::index_t               seqlen_knew    = 0;
    ck_tile::HostTensor<ck_tile::half_t> lse_acc_host(
        1 < num_splits ? std::array<ck_tile::index_t, 4>{num_splits, batch, nhead, shape_seqlen_q} :
                         std::array<ck_tile::index_t, 4>{1, 1, 1, 1});
    if (lse_acc_buf == nullptr) {
        return lse_acc_host.get_element_space_size_in_bytes();
    }
    // ck_tile::DeviceMem lse_acc_buf(lse_acc_host.get_element_space_size_in_bytes());
    auto has_logits_soft_cap = false;

    auto fmha_traits = fmha_fwd_traits{hdim_q,
                                       hdim_v,
                                       data_type,
                                       mode == mode_enum::group,
                                       is_v_rowmajor,
                                       has_logits_soft_cap,
                                       mask.type,
                                       bias.type,
                                       lse,
                                       p_drop > 0.0f,
                                       squant};

    auto fmha_args = [&]() {
        assert(nhead % nhead_k == 0);
        // QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head]
        const ck_tile::index_t stride_q    = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
        const ck_tile::index_t stride_k    = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
        const ck_tile::index_t stride_knew = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
        const ck_tile::index_t stride_v    = [&]() {
            if (is_v_rowmajor)
                return i_perm ? hdim_v : (nhead + 2 * nhead_k) * hdim_v;
            else
                return i_perm ? shape_seqlen_k : (nhead + 2 * nhead_k) * shape_seqlen_k;
        }();
        const ck_tile::index_t stride_vnew = [&]() {
            if (is_v_rowmajor)
                return i_perm ? hdim_v : (nhead + 2 * nhead_k) * hdim_v;
            else
                return i_perm ? seqlen_knew : (nhead + 2 * nhead_k) * seqlen_knew;
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
        const ck_tile::index_t stride_randval = 0;  //(max_seqlen_k);
        const ck_tile::index_t stride_o_acc   = 0;  // hdim_v;
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q    = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k    = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_knew = (i_perm ? seqlen_knew * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v    = [&]() {
            if (is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_vnew = [&]() {
            if (is_v_rowmajor)
                return i_perm ? seqlen_knew * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * seqlen_knew : seqlen_knew;
        }();
        const ck_tile::index_t nhead_stride_bias =
            0;  //(i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = 0;  //(shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = 0;  // shape_seqlen_q;
        const ck_tile::index_t nhead_stride_lse_acc = 0;  // shape_seqlen_q;
        const ck_tile::index_t nhead_stride_o_acc   = 0;  //(shape_seqlen_q * hdim_v);
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q       = 0;  //(nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k       = 0;  //(nhead_k * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_knew    = 0;  //(nhead_k * seqlen_knew * hdim_q);
        const ck_tile::index_t batch_stride_v       = 0;  //(nhead_k * hdim_v * shape_seqlen_k);
        const ck_tile::index_t batch_stride_vnew    = 0;  //(nhead_k * hdim_v * seqlen_knew);
        const ck_tile::index_t batch_stride_bias    = 0;  //(0 * nhead * shape_seqlen_q * shape_seqlen_k);
        const ck_tile::index_t batch_stride_randval = 0;  //(nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = 0;  //(nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_lse_acc = 0;  //(nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_o_acc   = 0;  //(nhead * max_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_o       = (nhead * shape_seqlen_q * hdim_v);
        // setup split_stride_* arguments (only used in split-kv kernel)
        const ck_tile::index_t split_stride_lse_acc = 0;  //(shape_batch * nhead * shape_seqlen_q);
        const ck_tile::index_t split_stride_o_acc   = 0;  //(batch * nhead * max_seqlen_q * hdim_v);
        const ck_tile::index_t min_seqlen_q         = 0;

        return fmha_fwd_args{q,
                             k,
                             v,
                             bias.type == bias_enum::alibi ? linear_bias_slopes : biasBuffer,
                             nullptr,  // randval_buf.GetDeviceBuffer(),
                             // lse_acc_buf.GetDeviceBuffer(),  // lse_acc_buf.GetDeviceBuffer(),
                             // lse_acc_buf,  // lse_acc_buf.GetDeviceBuffer(),
                             // nullptr,                        // o_acc_buf.GetDeviceBuffer(),
                             softmax_lse_,
                             output,
                             nullptr, //cu_seqlen_q_ptr
                             nullptr, //cu_seqlen_kv_ptr
                             seqstart_q,
                             seqstart_k,
                             nullptr,  // seqlen_kpads
                             nullptr, //seqstart_padded_q_ptr
                             nullptr, //seqstart_padded_k_ptr
                             shape_seqlen_q,
                             shape_seqlen_k,
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             // num_splits,
                             scale_s,
                             scale_p,
                             scale_o,
                             logits_soft_cap,
                             stride_q,
                             stride_k,
                             stride_v,
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias,
                             stride_randval,
                             // stride_o_acc,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_randval,
                             nhead_stride_lse,
                             // nhead_stride_lse_acc,
                             // nhead_stride_o_acc,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_randval,
                             batch_stride_lse,
                             // batch_stride_lse_acc,
                             // batch_stride_o_acc,
                             batch_stride_o,
                             // split_stride_lse_acc,
                             // split_stride_o_acc,
                             mask.left,
                             mask.right,
                             static_cast<ck_tile::index_t>(mask.type),
                             min_seqlen_q,
                             p_drop,
                             s_randval,
                             std::make_pair(drop_seed, drop_offset)};
    }();

    ck_tile::stream_config stream_config{
        stream_,  // stream_id_
        false,    // time_kernel_
        0,        // log_level_
        0,        // cold_niters_
        1,        // nrepeat_
        // false     //
    };

    float run_time = fmha_fwd(fmha_traits, fmha_args, stream_config);
    // std::cout << "\nrun_time for ck fmha_fwd: " << run_time << std::endl;
    if (run_time < 0) {
        CK_FAIL("fmha_fwd faild");
    } else {
        return 1;
    }
}

uint32_t rocmFmhaWrapper::runCKFmhaMLA(void*  q,
                                       void*  k,
                                       void*  v,
                                       void*  output,
                                       void*  softmax_lse_,
                                       size_t batch_size,
                                       size_t seq_len,
                                       float  softmax_extra_scale,
                                       void*  seqstart_q,
                                       void*  seqstart_k,
                                       void*  lse_acc_buf,
                                       void*  linear_bias_slopes,
                                       void*  biasBuffer) {

    // map parms from FT to CK
    mode_enum mode      = mode_enum::group;
    auto      data_type = getDataTypeStr(dtype_);
    auto      batch     = static_cast<ck_tile::index_t>(batch_size);
    auto      nhead     = static_cast<ck_tile::index_t>(head_num_);
    auto      nhead_k   = static_cast<ck_tile::index_t>(kv_head_num_);
    if (nhead_k < 0)
        nhead_k = nhead;

    if (nhead % nhead_k != 0) {
        std::cerr << "nhead:" << nhead << " must be multiple of nhead_k:" << nhead_k << std::endl;
        return false;
    }

    ck_tile::index_t seqlen_q = seq_len;
    ck_tile::index_t seqlen_k = seq_len;
    if (seqlen_k < 0)
        seqlen_k = seqlen_q;
    // auto [seqlen_qs, seqlen_ks, seqlen_kpads] = decode_seqlen(mode_enum::batch,
    //                                                           batch,
    //                                                           std::to_string(seqlen_q),
    //                                                           std::to_string(seqlen_k),
    //                                                           "-1");

    ck_tile::index_t hdim_q = size_per_head_;
    ck_tile::index_t hdim_v = size_per_head_;
    if (hdim_v < 0)
        hdim_v = hdim_q;

    // the output of add_fusedQKV_bias_transpose_kernel:
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.
    bool i_perm = false;  // if true, will be batch * nhead * seqlen * hdim
    bool o_perm = false;  // if false, will be batch * seqlen * nhead * hdim

    float scale_s = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q)) * softmax_extra_scale;  // TODO: q ? v ?

    auto  squant          = false;
    float scale_p         = 1.f;
    float scale_o         = 1.f;
    float logits_soft_cap = 0.f;
    // if (squant) {
    //     scale_s = scale_s * (range_q / dtype_max) * (range_k / dtype_max);
    //     scale_p = dtype_max / range_p;
    //     // scale_p = [max(fp8_t)/range_o] * [range_p/max(fp8_t)] * [range_v/max(fp8_t)]
    //     scale_o = range_p * range_v / range_o / dtype_max;
    // }

    bool is_v_rowmajor = true;
    auto lse           = softmax_lse_ ? true : false;

    std::string msk_str;
    if (mtype_ == AttentionMaskType::noMask) {
        msk_str = "0";
    } else if (mtype_ == AttentionMaskType::causalMask) {
        msk_str = "b";
        // RTP_LLM_LOG_INFO("Using causal_bottom_right Mask");
    } else {
        RTP_LLM_LOG_ERROR("Mask type not supported");
    }

    bias_info bias = bias_info::decode(linear_bias_slopes ? "a" : "n");
    mask_info mask = mask_info::decode(msk_str, seqlen_q, seqlen_k);  // TODO: we don't need x/y anymore

    float    p_drop      = 0.;
    uint64_t drop_seed   = 1.;
    uint64_t drop_offset = 0.;
    bool     drop_prefs  = false;
    if (p_drop < 0.0f || p_drop > 1.0f) {
        std::cerr << "The value of p_drop should be 0~1" << std::endl;
        return false;
    }

    bool s_randval = false;
    if (p_drop > 0.0f) {
        s_randval = true;
    }

    int                    num_splits   = 1;
    const ck_tile::index_t max_seqlen_q = seq_len;  // max of all batch
    const ck_tile::index_t max_seqlen_k = seq_len;

    // host memory for storing all the tensor elements
    const ck_tile::index_t shape_batch    = (mode == mode_enum::batch ? batch : 1);
    const ck_tile::index_t shape_seqlen_q = (mode == mode_enum::batch ? seq_len : seq_len);
    const ck_tile::index_t shape_seqlen_k = (mode == mode_enum::batch ? seq_len : seq_len);
    const ck_tile::index_t seqlen_knew    = 0;

    ck_tile::HostTensor<ck_tile::half_t> lse_acc_host(
        1 < num_splits ? std::array<ck_tile::index_t, 4>{num_splits, batch, nhead, max_seqlen_q} :
                         std::array<ck_tile::index_t, 4>{1, 1, 1, 1});
    if (lse_acc_buf == nullptr) {
        // printf("[CK] size = %d\n", lse_acc_host.get_element_space_size_in_bytes());
        return lse_acc_host.get_element_space_size_in_bytes();
    }
    // ck_tile::DeviceMem lse_acc_buf(lse_acc_host.get_element_space_size_in_bytes());
    auto has_logits_soft_cap = false;

    // auto fmha_traits = fmha_fwd_traits{hdim_q,
    //                                    hdim_v,
    //                                    data_type,
    //                                    mode == mode_enum::group,
    //                                    is_v_rowmajor,
    //                                    has_logits_soft_cap,
    //                                    mask.type,
    //                                    bias.type,
    //                                    lse,
    //                                    p_drop > 0.0f,
    //                                    squant};

    auto fmha_args = [&]() {
        assert(nhead % nhead_k == 0);
        // QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head]
        const ck_tile::index_t stride_q    = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
        const ck_tile::index_t stride_k    = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
        const ck_tile::index_t stride_knew = (i_perm ? hdim_q : (nhead + 2 * nhead_k) * hdim_q);
        const ck_tile::index_t stride_v    = [&]() {
            if (is_v_rowmajor)
                return i_perm ? hdim_v : (nhead + 2 * nhead_k) * hdim_v;
            else
                return i_perm ? shape_seqlen_k : (nhead + 2 * nhead_k) * shape_seqlen_k;
        }();
        const ck_tile::index_t stride_vnew = [&]() {
            if (is_v_rowmajor)
                return i_perm ? hdim_v : (nhead + 2 * nhead_k) * hdim_v;
            else
                return i_perm ? seqlen_knew : (nhead + 2 * nhead_k) * seqlen_knew;
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_o_acc   = hdim_v;
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q    = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k    = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_knew = (i_perm ? seqlen_knew * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v    = [&]() {
            if (is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_vnew = [&]() {
            if (is_v_rowmajor)
                return i_perm ? seqlen_knew * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * seqlen_knew : seqlen_knew;
        }();
        const ck_tile::index_t nhead_stride_bias = (i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_lse_acc = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_o_acc   = (max_seqlen_q * hdim_v);
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q       = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k       = (nhead_k * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_knew    = (nhead_k * seqlen_knew * hdim_q);
        const ck_tile::index_t batch_stride_v       = (nhead_k * hdim_v * shape_seqlen_k);
        const ck_tile::index_t batch_stride_vnew    = (nhead_k * hdim_v * seqlen_knew);
        const ck_tile::index_t batch_stride_bias    = (0 * nhead * shape_seqlen_q * shape_seqlen_k);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_lse_acc = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_o_acc   = (nhead * max_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_o       = (nhead * shape_seqlen_q * hdim_v);
        // setup split_stride_* arguments (only used in split-kv kernel)
        const ck_tile::index_t split_stride_lse_acc = (shape_batch * nhead * shape_seqlen_q);
        const ck_tile::index_t split_stride_o_acc   = (batch * nhead * max_seqlen_q * hdim_v);
        const ck_tile::index_t min_seqlen_q         = 0;

        return fmha_fwd_args{q,
                             k,
                             v,
                             bias.type == bias_enum::alibi ? linear_bias_slopes : biasBuffer,
                             nullptr,  // randval_buf.GetDeviceBuffer(),
                             // lse_acc_buf.GetDeviceBuffer(),  // lse_acc_buf.GetDeviceBuffer(),
                             // lse_acc_buf,  // lse_acc_buf.GetDeviceBuffer(),
                             // nullptr,                        // o_acc_buf.GetDeviceBuffer(),
                             softmax_lse_,
                             output,
                             nullptr, //cu_seqlen_q_ptr
                             nullptr, //cu_seqlen_kv_ptr
                             seqstart_q,
                             seqstart_k,
                             nullptr,  // seqlen_kpads
                             nullptr, //seqstart_padded_q_ptr
                             nullptr, //seqstart_padded_k_ptr
                             shape_seqlen_q,
                             shape_seqlen_k,
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             // num_splits,
                             scale_s,
                             scale_p,
                             scale_o,
                             logits_soft_cap,
                             stride_q,
                             stride_k,
                             stride_v,
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias,
                             stride_randval,
                             // stride_o_acc,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_randval,
                             nhead_stride_lse,
                             // nhead_stride_lse_acc,
                             // nhead_stride_o_acc,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_randval,
                             batch_stride_lse,
                             // batch_stride_lse_acc,
                             // batch_stride_o_acc,
                             batch_stride_o,
                             // split_stride_lse_acc,
                             // split_stride_o_acc,
                             mask.left,
                             mask.right,
                             static_cast<ck_tile::index_t>(mask.type),
                             min_seqlen_q,
                             p_drop,
                             s_randval,
                             std::make_pair(drop_seed, drop_offset)};
    }();

    ck_tile::stream_config stream_config{
        stream_,  // stream_id_
        false,    // time_kernel_
        0,        // log_level_
        0,        // cold_niters_
        1,        // nrepeat_
        // false     //
    };
    float run_time;
    if (data_type == "bf16" && size_per_head_ == 128 && msk_str == "b")
        run_time = aiter::mha_fwd(
            fmha_args, stream_config, data_type, mode == mode_enum::group, mask.type, bias.type, lse, true);
    else
        run_time = aiter::mha_fwd(
            fmha_args, stream_config, data_type, mode == mode_enum::group, mask.type, bias.type, lse, false);
    // std::cout << "\nrun_time for ck fmha_fwd: " << run_time << std::endl;
    if (run_time < 0) {
        CK_FAIL("fmha_fwd faild");
    } else {
        return 1;
    }
}

}  // namespace rtp_llm
