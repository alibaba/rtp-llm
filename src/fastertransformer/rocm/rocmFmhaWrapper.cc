#include "rocmFmhaWrapper.h"
#include "fmha_fwd.hpp"
#include "ck_tile/host.hpp"
#include "mask.hpp"
#include "utils.hpp"
#include "bias.hpp"
#include "src/fastertransformer/utils/logger.h"

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
    // map parms from FT to CK
    mode_enum mode         = mode_enum::batch;
    auto      data_type    = getDataTypeStr(dtype_);
    auto      batch        = static_cast<ck_tile::index_t>(batch_size);
    auto      nhead        = static_cast<ck_tile::index_t>(head_num_);
    auto      nhead_k      = static_cast<ck_tile::index_t>(kv_head_num_);
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
    auto [seqlen_qs, seqlen_ks, seqlen_kpads] = decode_seqlen(mode,
                                                              batch,
                                                              std::to_string(seqlen_q),
                                                              std::to_string(seqlen_k),
                                                              "-1");

    ck_tile::index_t hdim_q = size_per_head_;
    ck_tile::index_t hdim_v = size_per_head_;
    if (hdim_v < 0)
        hdim_v = hdim_q;

    // the output of add_fusedQKV_bias_transpose_kernel:
    // shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].
    bool  i_perm        = true;  // if true, will be batch * nhead * seqlen * hdim
    bool  o_perm        = false;  // if false, will be batch * seqlen * nhead * hdim
    float scale_s       = 0.f;
    if (scale_s == .0f)
        scale_s = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q));  // TODO: q ? v ?

    auto squant        = false;
    float scale_p       = 1.f;
    float scale_o       = 1.f;
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
        msk_str="0";
    }
    else if (mtype_ == AttentionMaskType::causalMask)
    {
        msk_str = "b";
        FT_LOG_INFO("Using causal_bottom_right Mask");
    }
    else
    {
        FT_LOG_ERROR("Mask type not supported");
    }
    
    

    bias_info bias = bias_info::decode(linear_bias_slopes ? "a" : "n");
    mask_info mask = mask_info::decode(msk_str, seqlen_qs[0], seqlen_ks[0]);  // TODO: we don't need x/y anymore

    float    p_drop      = 0.;
    uint64_t drop_seed   = 1.;
    uint64_t drop_offset = 0.;
    if (p_drop < 0.0f || p_drop > 1.0f) {
        std::cerr << "The value of p_drop should be 0~1" << std::endl;
        return false;
    }

    bool s_randval = false;
    if(p_drop > 0.0f)
    {
        s_randval = true;
    }

    int num_splits = 1;
    const ck_tile::index_t max_seqlen_q = seq_len;
    const ck_tile::index_t max_seqlen_k = seq_len;

    // host memory for storing all the tensor elements
    const ck_tile::index_t shape_batch     = (mode == mode_enum::batch ? batch : 1);
    const ck_tile::index_t shape_seqlen_q  = seqlen_qs[0];
    const ck_tile::index_t shape_seqlen_k  = seqlen_ks[0];
    const auto seqstart_q_host = generate_seqstarts(mode, batch, seqlen_q);
    const auto seqstart_k_host = generate_seqstarts(mode, batch, seqlen_k);
    ck_tile::DeviceMem     seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem     seqstart_k(seqstart_k_host.size() * sizeof(int32_t));

    auto fmha_traits = fmha_fwd_traits{hdim_q,
                                       hdim_v,
                                       data_type,
                                       mode == mode_enum::group,
                                       is_v_rowmajor,
                                       mask.type,
                                       bias.type,
                                       lse,
                                       p_drop > 0.0f,
                                       squant};

    auto fmha_args = [&]() {
        assert(nhead % nhead_k == 0);
        /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
        ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
        ///       'nhead_stride_bias' are 0.
        // setup stride_* arguments
        const ck_tile::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k;
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_o_acc   = hdim_v;
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_bias =
            (i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = max_seqlen_q;
        const ck_tile::index_t nhead_stride_lse_acc = max_seqlen_q;
        const ck_tile::index_t nhead_stride_o_acc   = (max_seqlen_q * hdim_v);
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q       = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k       = (nhead_k * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_v       = (nhead_k * hdim_v * shape_seqlen_k);
        const ck_tile::index_t batch_stride_bias    = (0 * nhead * shape_seqlen_q * shape_seqlen_k);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = (nhead * max_seqlen_q);
        const ck_tile::index_t batch_stride_lse_acc = (nhead * max_seqlen_q);
        const ck_tile::index_t batch_stride_o_acc   = (nhead * max_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_o       = (nhead * shape_seqlen_q * hdim_v);
        // setup split_stride_* arguments (only used in split-kv kernel)
        const ck_tile::index_t split_stride_lse_acc = (batch * nhead * max_seqlen_q);
        const ck_tile::index_t split_stride_o_acc   = (batch * nhead * max_seqlen_q * hdim_v);

        return fmha_fwd_args{q,
                             k,
                             v,
                             bias.type == bias_enum::alibi ? linear_bias_slopes : biasBuffer,
                             nullptr,  // randval_buf.GetDeviceBuffer(),
                             nullptr,  // lse_acc_buf.GetDeviceBuffer(),
                             nullptr,  // o_acc_buf.GetDeviceBuffer(),
                             softmax_lse_,
                             output,
                             seqstart_q.GetDeviceBuffer(),
                             seqstart_k.GetDeviceBuffer(),
                             nullptr,
                             shape_seqlen_q,
                             shape_seqlen_k,
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             num_splits,
                             scale_s,
                             scale_p,
                             scale_o,
                             stride_q,
                             stride_k,
                             stride_v,
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias,
                             stride_randval,
                             stride_o_acc,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_randval,
                             nhead_stride_lse,
                             nhead_stride_lse_acc,
                             nhead_stride_o_acc,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_randval,
                             batch_stride_lse,
                             batch_stride_lse_acc,
                             batch_stride_o_acc,
                             batch_stride_o,
                             split_stride_lse_acc,
                             split_stride_o_acc,
                             mask.left,
                             mask.right,
                             static_cast<ck_tile::index_t>(mask.type),
                             p_drop,
                             s_randval,
                             {drop_seed, drop_offset}};
    }();

    ck_tile::stream_config stream_config{
        nullptr,  // stream_id_
        false,    // time_kernel_
        1,        // log_level_
        0,        // cold_niters_
        1,        // nrepeat_
        // false     // 
    };

    float run_time = fmha_fwd(fmha_traits, fmha_args, stream_config);
    std::cout << "\nrun_time for ck fmha_fwd: " << run_time << std::endl;
    if (run_time < 0) {
        return false;
    } else {
        return true;
    }
}


}  // namespace fastertransformer