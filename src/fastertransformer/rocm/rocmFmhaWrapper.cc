#include "rocmFmhaWrapper.h"
#include "fmha_fwd.hpp"
#include "ck_tile/host.hpp"
#include "mask.hpp"
#include "utils.hpp"
#include "bias.hpp"

namespace fastertransformer {

bool rocmFmhaWrapper::runCKFmha(void*  q,
                                void*  k,
                                void*  v,
                                void*  output,
                                void*  softmax_lse_,
                                size_t batch_size,
                                size_t seq_len,
                                void*  linear_bias_slopes,
                                void*  biasBuffer) {

    // TODO: for batch mode only
    mode_enum  mode            = mode_enum::batch;
    auto       seqlen_qs       = std::vector<ck_tile::index_t>(batch_size, seq_len);
    auto       seqlen_ks       = std::vector<ck_tile::index_t>(batch_size, seq_len);
    auto       seqlen_kpads    = std::vector<ck_tile::index_t>(batch_size, -1);  // TODO: batch not support k_padding
    const auto seqstart_q_host = to_seqstarts(seqlen_qs);
    const auto seqstart_k_host = to_seqstarts(seqlen_ks);
    const auto seqstart_k_with_padding_host = to_seqstarts(seqlen_kpads);
    ck_tile::DeviceMem     seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem     seqstart_k(seqstart_k_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem     seqlen_k_buf(seqlen_kpads[0] < 0 ? 0 : seqlen_ks.size() * sizeof(int32_t));
    const ck_tile::index_t shape_seqlen_q = seqlen_qs[0];
    const ck_tile::index_t shape_seqlen_k = seqlen_ks[0];
    const ck_tile::index_t max_seqlen_q   = seq_len;
    const ck_tile::index_t max_seqlen_k   = seq_len;
    int                    hdim_q         = size_per_head_;
    int                    hdim_v         = size_per_head_;
    ck_tile::index_t       nhead          = {static_cast<ck_tile::index_t>(head_num_)};

    assert(mtype_ == AttentionMaskType::noMask);
    mask_info mask = mask_info::decode("0", seqlen_qs[0], seqlen_ks[0]);  // TODO: we don't need x/y anymore
    bias_info bias = bias_info::decode(linear_bias_slopes ? "a" : "n");

    // the output of add_fusedQKV_bias_transpose_kernel:
    // shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].
    bool  i_perm        = true;  // if true, will be batch * nhead * seqlen * hdim
    bool  o_perm        = true;  // if false, will be batch * seqlen * nhead * hdim
    bool  is_v_rowmajor = true;
    float scale_s       = 0.f;
    float scale_p       = 1.f;
    float scale_o       = 1.f;
    float p_drop        = 0.f;

    bool s_randval = false;

    auto fmha_traits = fmha_fwd_traits{
        hdim_q,
        hdim_v,
        dtype_,
        mode == mode_enum::group,
        is_v_rowmajor,
        mask.type,
        bias.type,
        softmax_lse_ ? true : false,  // bool has_lse;
        // false,                        // 0~1 probability of dropout
        false                         // do_fp8_static_quant; only fp8 will default use squant
    };

    auto fmha_args = [&, k_paddings_ = seqlen_kpads]() {
        assert(head_num_ % kv_head_num_ == 0);
        /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
        ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
        ///       'nhead_stride_bias' are 0.
        // setup stride_* arguments
        const ck_tile::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k = (i_perm ? hdim_q : kv_head_num_ * hdim_q);
        const ck_tile::index_t stride_v = [&]() {
            if (is_v_rowmajor)
                return i_perm ? hdim_v : kv_head_num_ * hdim_v;
            else
                return i_perm ? shape_seqlen_k : kv_head_num_ * shape_seqlen_k;
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v = [&]() {
            if (is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_bias = (i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = max_seqlen_q;
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q       = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k       = (kv_head_num_ * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_v       = (kv_head_num_ * hdim_v * shape_seqlen_k);
        const ck_tile::index_t batch_stride_bias    = (0 * nhead * shape_seqlen_q * shape_seqlen_k);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = (nhead * max_seqlen_q);
        const ck_tile::index_t batch_stride_o       = (nhead * shape_seqlen_q * hdim_v);

        return fmha_fwd_args{
            q,
            k,
            v,
            bias.type == bias_enum::alibi ? linear_bias_slopes : biasBuffer,
            // nullptr,  // randval_buf.GetDeviceBuffer(),
            softmax_lse_,
            output,
            seqstart_q.GetDeviceBuffer(),
            seqstart_k.GetDeviceBuffer(),
            k_paddings_[0] < 0 ? nullptr : seqlen_k_buf.GetDeviceBuffer(),
            shape_seqlen_q,
            shape_seqlen_k,
            {static_cast<ck_tile::index_t>(batch_size)},
            max_seqlen_q,
            hdim_q,
            hdim_v,
            nhead,
            {static_cast<ck_tile::index_t>(kv_head_num_)},
            scale_s,
            scale_p,
            scale_o,
            stride_q,
            stride_k,
            stride_v,
            bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias,
            //  stride_randval,
            stride_o,
            nhead_stride_q,
            nhead_stride_k,
            nhead_stride_v,
            nhead_stride_bias,
            //  nhead_stride_randval,
            nhead_stride_lse,
            nhead_stride_o,
            batch_stride_q,
            batch_stride_k,
            batch_stride_v,
            batch_stride_bias,
            //  batch_stride_randval,
            batch_stride_lse,
            batch_stride_o,
            mask.left,
            mask.right,
            static_cast<ck_tile::index_t>(mask.type),
            // p_drop,
            // s_randval,
            // {1, 0}  //{drop_seed, drop_offset}};
        };  
    }();

    ck_tile::stream_config stream_config{
        nullptr,  // stream_id_
        false,    // time_kernel_
        0,        // log_level_
        0,        // cold_niters_
        0,        // nrepeat_
        // false     // 
    };

    float run_time = fmha_fwd(fmha_traits, fmha_args, stream_config);
    if (run_time > 0) {
        return true;
    } else {
        return false;
    }
}
}  // namespace fastertransformer