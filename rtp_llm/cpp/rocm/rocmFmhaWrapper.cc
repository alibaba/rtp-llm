#include "rocmFmhaWrapper.h"
#include "mha_fwd.h"
#include "ck_tile/host.hpp"
#include "utils.hpp"
#include "rtp_llm/cpp/utils/Logger.h"

// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/01_fmha/mask.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/01_fmha/fmha_fwd.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/01_fmha/utils.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/example/ck_tile/01_fmha/bias.hpp"
// #include "aiter_meta/3rdparty/composable_kernel/include/ck_tile/host.hpp"

namespace rtp_llm {

inline std::string getFmhaDataTypeStr(const DataType& data_type) {
    std::string type_str;
    if (data_type == TYPE_FP16) {
        type_str = "fp16";
    } else if (data_type == TYPE_BF16) {
        type_str = "bf16";
    } else if (data_type == TYPE_FP32) {
        type_str = "fp32";
    } else if (data_type == TYPE_FP8_E4M3) {
        type_str = "fp8bf16";
    } else {
        throw std::runtime_error("wrong data type");
    }
    return type_str;
}
    
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
    auto      data_type = getFmhaDataTypeStr(dtype_);
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
    
    quant_scale_enum qscale_type = quant_scale_enum::no_scale;
    bool do_fp8_static_quant = false;
    bool skip_min_seqlen_q   = false;        

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
    if (!is_causal_) {
        msk_str = "0";
    } else {
        msk_str = "b";
        // RTP_LLM_LOG_INFO("Using causal_bottom_right Mask");
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
                                       qscale_type,
                                       do_fp8_static_quant,
                                       skip_min_seqlen_q};

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

        return fmha_fwd_args{
                             q,
                             k,
                             v,
                             bias.type == bias_enum::alibi ? linear_bias_slopes : biasBuffer,
                             nullptr, //  q_descale_ptr,
                             nullptr, //  k_descale_ptr,
                             nullptr, //  v_descale_ptr,
                             nullptr,  // has_dropout_randval
                             softmax_lse_,
                             output,
                             seqstart_q, //seqstart_q_ptr
                             seqstart_k, //seqstart_k_ptr
                             nullptr, // seqlen_q_ptr
                             nullptr, // seqlen_k_ptr
                             nullptr, //cu_seqlen_q_ptr
                             nullptr, // cu_seqlen_k_ptr
                             reinterpret_cast<const void*>(&shape_seqlen_q),
                             reinterpret_cast<const void*>(&shape_seqlen_k),
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             static_cast<ck_tile::index_t>(scale_s),
                             static_cast<ck_tile::index_t>(logits_soft_cap),
                             static_cast<float>(stride_q),
                             static_cast<float>(stride_k),
                             static_cast<float>(stride_v),
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias,
                             stride_randval,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_randval,
                             nhead_stride_lse,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_randval,
                             batch_stride_lse,
                             batch_stride_o,
                             mask.left,
                             mask.right,
                             mask.sink,
                             static_cast<ck_tile::index_t>(mask.type),
                             min_seqlen_q,
                             static_cast<ck_tile::index_t>(p_drop),
                             static_cast<float>(s_randval),
                             drop_seed,
                             drop_offset};
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
    auto      data_type = getFmhaDataTypeStr(dtype_);
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

    quant_scale_enum qscale_type = quant_scale_enum::no_scale;
    bool do_fp8_static_quant = false;
    bool skip_min_seqlen_q   = false;      

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
    if (!is_causal_) {
        msk_str = "0";
    } else {
        msk_str = "b";
        // RTP_LLM_LOG_INFO("Using causal_bottom_right Mask");
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
                                       qscale_type,
                                       do_fp8_static_quant,
                                       skip_min_seqlen_q};

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

        return fmha_fwd_args{
                             q,
                             k,
                             v,
                             bias.type == bias_enum::alibi ? linear_bias_slopes : biasBuffer,
                             nullptr, //  q_descale_ptr,
                             nullptr, //  k_descale_ptr,
                             nullptr, //  v_descale_ptr,
                             nullptr,  // has_dropout_randval
                             softmax_lse_,
                             output,
                             seqstart_q, //seqstart_q_ptr
                             seqstart_k, //seqstart_k_ptr
                             nullptr, // seqlen_q_ptr
                             nullptr, // seqlen_k_ptr
                             nullptr, //cu_seqlen_q_ptr
                             nullptr, // cu_seqlen_k_ptr
                             reinterpret_cast<const void*>(&shape_seqlen_q),
                             reinterpret_cast<const void*>(&shape_seqlen_k),
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             static_cast<ck_tile::index_t>(scale_s),
                             static_cast<ck_tile::index_t>(logits_soft_cap),
                             static_cast<float>(stride_q),
                             static_cast<float>(stride_k),
                             static_cast<float>(stride_v),
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias,
                             stride_randval,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_randval,
                             nhead_stride_lse,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_randval,
                             batch_stride_lse,
                             batch_stride_o,
                             mask.left,
                             mask.right,
                             mask.sink,
                             static_cast<ck_tile::index_t>(mask.type),
                             min_seqlen_q,
                             static_cast<ck_tile::index_t>(p_drop),
                             static_cast<float>(s_randval),
                             drop_seed,
                             drop_offset};
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
    auto      data_type = getFmhaDataTypeStr(dtype_);
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
    if (!is_causal_) {
        msk_str = "0";
    } else {
        msk_str = "b";
        // RTP_LLM_LOG_INFO("Using causal_bottom_right Mask");
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

    quant_scale_enum qscale_type = quant_scale_enum::no_scale;
    bool do_fp8_static_quant = false;
    bool skip_min_seqlen_q   = false; 

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
                                       qscale_type,
                                       do_fp8_static_quant,
                                       skip_min_seqlen_q};

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

        return fmha_fwd_args{
                             q,
                             k,
                             v,
                             bias.type == bias_enum::alibi ? linear_bias_slopes : biasBuffer,
                             nullptr, //  q_descale_ptr,
                             nullptr, //  k_descale_ptr,
                             nullptr, //  v_descale_ptr,
                             nullptr,  // has_dropout_randval
                             softmax_lse_,
                             output,
                             seqstart_q, //seqstart_q_ptr
                             seqstart_k, //seqstart_k_ptr
                             nullptr, // seqlen_q_ptr
                             nullptr, // seqlen_k_ptr
                             nullptr, //cu_seqlen_q_ptr
                             nullptr, // cu_seqlen_k_ptr
                             reinterpret_cast<const void*>(&shape_seqlen_q),
                             reinterpret_cast<const void*>(&shape_seqlen_k),
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             static_cast<ck_tile::index_t>(scale_s),
                             static_cast<ck_tile::index_t>(logits_soft_cap),
                             static_cast<float>(stride_q),
                             static_cast<float>(stride_k),
                             static_cast<float>(stride_v),
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias,
                             stride_randval,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_randval,
                             nhead_stride_lse,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_randval,
                             batch_stride_lse,
                             batch_stride_o,
                             mask.left,
                             mask.right,
                             mask.sink,
                             static_cast<ck_tile::index_t>(mask.type),
                             min_seqlen_q,
                             static_cast<ck_tile::index_t>(p_drop),
                             static_cast<float>(s_randval),
                             drop_seed,
                             drop_offset};
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

}  // namespace rtp_llm
