#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

namespace rtp_llm {

LoraLinearOutput DeviceBase::loraLinear(const LoraLinearParams& params) {
    const auto& A = params.gemm_params.A;
    const auto& B = params.gemm_params.B;

    if (initParams().profile_debug_logging_config.check_nan) {
        if (A.isQBuffer()) {
            const auto& qbuffer = reinterpret_cast<const QBuffer&>(A);
            checkNAN(qbuffer.kernel(), "loraLinear_A_kernel_dump", nullptr, true);
            checkNAN(qbuffer.scales(), "loraLinear_A_scales_dump", nullptr, true);
        } else {
            checkNAN(A, "loraLinear_A_dump", nullptr, true);
        }
        if (B.isQBuffer()) {
            const auto& qbuffer = reinterpret_cast<const QBuffer&>(B);
            checkNAN(qbuffer.kernel(), "loraLinear_B_kernel_dump", nullptr, true);
            checkNAN(qbuffer.scales(), "loraLinear_B_scales_dump", nullptr, true);
        } else {
            checkNAN(B, "loraLinear_B_dump", nullptr, true);
        }
    }

    auto output = gemm(params.gemm_params);

    if (initParams().profile_debug_logging_config.check_nan) {
        if (output->isQBuffer()) {
            const auto& qbuffer = reinterpret_cast<const QBuffer&>(*output);
            checkNAN(qbuffer.kernel(), "loraLinear_gemm_output_kernel_dump", nullptr, true);
            checkNAN(qbuffer.scales(), "loraLinear_gemm_output_scales_dump", nullptr, true);
        } else {
            checkNAN(*output, "loraLinear_gemm_output_dump", nullptr, true);
        }
    }

    if (params.lora_input) {
        auto&                  lora_input_lengths     = *params.lora_input->lora_input_lengths_;
        auto&                  lora_a                 = params.lora_input->lora_a_;
        auto&                  lora_b                 = params.lora_input->lora_b_;
        int32_t*               lora_input_lengths_ptr = lora_input_lengths.data<int32_t>();
        std::vector<BufferPtr> inputs;
        std::vector<BufferPtr> outputs;
        std::vector<BufferPtr> lora_as;
        std::vector<BufferPtr> lora_bs;
        size_t                 start = 0;
        for (int i = 0; i < lora_input_lengths.shape()[0]; i++) {
            if (lora_a[i] != nullptr && lora_b[i] != nullptr) {
                auto input_tmp  = params.gemm_params.A.slice(start, lora_input_lengths_ptr[i]);
                auto output_tmp = output->slice(start, lora_input_lengths_ptr[i]);
                inputs.push_back(input_tmp);
                outputs.push_back(output_tmp);
                lora_as.push_back(std::const_pointer_cast<Buffer>(lora_a[i]));
                lora_bs.push_back(std::const_pointer_cast<Buffer>(lora_b[i]));
            }
            start = start + lora_input_lengths_ptr[i];
        }
        if (inputs.size() > 0) {
            if (params.lora_input->use_same_lora_) {
                RTP_LLM_LOG_DEBUG("use same lora");
                auto tmp    = gemm({params.gemm_params.A,
                                    *lora_as[0],
                                    std::nullopt,
                                    nullptr,
                                    DataType::TYPE_INVALID,
                                    params.gemm_params.D_type});
                auto result = gemm({*tmp,
                                    *lora_bs[0],
                                    std::nullopt,
                                    output,
                                    DataType::TYPE_INVALID,
                                    params.gemm_params.D_type,
                                    TransposeOperation::NONE,
                                    TransposeOperation::NONE,
                                    ActivationType::Identity,
                                    1.0f,
                                    1.0f});
            } else {
                // M = X * A
                auto tmp = groupedGemm({inputs, lora_as});
                // Y = M * B + Y
                auto result = groupedGemm({tmp.output, lora_bs, outputs});
            }
        }
    }
    return LoraLinearOutput({std::move(output)});
}

ReduceScatterLoraLinearOutput DeviceBase::loraLinearReduceScatter(const LoraLinearReduceScatterParams& params) {
    const LoraLinearParams& linear_params     = params.lora_linear_params;
    const auto&             gemm_a            = linear_params.gemm_params.A;
    auto                    output            = linear_params.gemm_params.D;
    int                     m_split           = init_params_.m_split;
    const auto              m                 = gemm_a.shape()[0];
    size_t                  overlap_comm_type = init_params_.device_resource_config.overlap_comm_type;
    if (overlap_comm_type == 1 && (m_split > 0) && ((!linear_params.lora_input) || linear_params.lora_input->isEmpty())
        && init_params_.tp_size > 1) {
        RTP_LLM_CHECK_WITH_INFO(params.mode == ParallelMode::TP || params.mode == ParallelMode::FFN_TP,
                                "For overlap_comm_type 1, mode must be TP or FFN_TP.");
        size_t token_idx    = 0;
        size_t rs_token_idx = 0;
        size_t tp_size      = params.mode == ParallelMode::FFN_TP ? init_params_.ffn_tp_size : init_params_.tp_size;
        size_t m_chunk      = m / m_split;
        if (m > 128) {
            m_chunk = (m / m_split + 127) & ~127;
        }
        while (token_idx < m) {
            const auto micro_batch_tokens    = std::min(m - token_idx, m_chunk);
            const auto rs_micro_batch_tokens = micro_batch_tokens / tp_size;
            BufferPtr  input_a_chunk         = nullptr;
            BufferPtr  output_d_chunk        = output->slice(token_idx, micro_batch_tokens);
            BufferPtr  rs_recv_chunk         = params.rs_recv_buffer->slice(rs_token_idx, rs_micro_batch_tokens);

            if (params.qscheme == NoQuantize) {
                input_a_chunk = gemm_a.slice(token_idx, micro_batch_tokens);
            } else if (params.qscheme == Qint8PerToken) {
                input_a_chunk = reinterpret_cast<const QBuffer&>(gemm_a).qslice(token_idx, micro_batch_tokens);
            } else if (params.qscheme == Qfp8PerTensor) {
                input_a_chunk = reinterpret_cast<const QBuffer&>(gemm_a).qslicePerTensor(token_idx, micro_batch_tokens);
            } else {
                RTP_LLM_FAIL("unsupported qscheme");
            }
            auto micro_batch_gemm_params = GemmParams(*input_a_chunk,
                                                      linear_params.gemm_params.B,
                                                      linear_params.gemm_params.C,
                                                      output_d_chunk,
                                                      linear_params.gemm_params.compute_type,
                                                      linear_params.gemm_params.D_type,
                                                      linear_params.gemm_params.transA,
                                                      linear_params.gemm_params.transB,
                                                      linear_params.gemm_params.activationType,
                                                      linear_params.gemm_params.alpha,
                                                      linear_params.gemm_params.beta,
                                                      init_params_.device_resource_config.overlap_math_sm_count);
            loraLinear({micro_batch_gemm_params});
            reduceScatter({output_d_chunk, rs_recv_chunk, ReduceOp::Sum, params.mode, true});
            token_idx += micro_batch_tokens;
            rs_token_idx += rs_micro_batch_tokens;
        }
        overlappedCommBarrier();
        return ReduceScatterLoraLinearOutput(
            {std::move(params.rs_recv_buffer), std::move(linear_params.gemm_params.D)});
    }

    // by default
    output = loraLinear(linear_params).output;
    reduceScatter({linear_params.gemm_params.D, params.rs_recv_buffer, ReduceOp::Sum, params.mode});
    return ReduceScatterLoraLinearOutput({std::move(params.rs_recv_buffer), std::move(linear_params.gemm_params.D)});
}

AllGatherLoraLinearOutput DeviceBase::allGatherloraLinear(const AllGatherLoraLinearParams& params) {
    const LoraLinearParams& linear_params = params.lora_linear_params;
    const auto&             gemm_a        = linear_params.gemm_params.A;
    const auto&             gemm_b        = linear_params.gemm_params.B;

    BufferPtr output = nullptr;
    if (linear_params.gemm_params.D) {
        output = linear_params.gemm_params.D;
    } else {
        output                      = allocateBuffer({params.output_type, {gemm_a.shape()[0], gemm_b.shape()[1]}});
        linear_params.gemm_params.D = output;
    }
    int m_split = init_params_.m_split;

    size_t overlap_comm_type = init_params_.device_resource_config.overlap_comm_type;

    const size_t m = gemm_a.shape()[0];

    if (overlap_comm_type == 1 && (m_split > 0) && ((!linear_params.lora_input) || linear_params.lora_input->isEmpty())
        && init_params_.tp_size > 1) {
        overlappedComputeBarrier();
        size_t token_idx    = 0;
        size_t ag_token_idx = 0;
        size_t m_chunk      = m / m_split;
        RTP_LLM_CHECK_WITH_INFO(params.mode == ParallelMode::TP || params.mode == ParallelMode::FFN_TP,
                                "For overlap_comm_type 1, mode must be TP or FFN_TP.");
        size_t tp_size = params.mode == ParallelMode::FFN_TP ? init_params_.ffn_tp_size : init_params_.tp_size;
        while (token_idx < m) {
            const auto micro_batch_tokens    = std::min(m - token_idx, m_chunk);
            const auto ag_micro_batch_tokens = micro_batch_tokens / tp_size;
            BufferPtr  input_a_chunk         = gemm_a.slice(token_idx, micro_batch_tokens);
            BufferPtr  output_d_chunk        = output->slice(token_idx, micro_batch_tokens);
            BufferPtr  ag_send_buffer        = params.ag_send_buffer->slice(ag_token_idx, ag_micro_batch_tokens);
            allGather({{input_a_chunk}, params.mode, {ag_send_buffer}, false, true});
            if (params.qscheme == NoQuantize) {
            } else if (params.qscheme == Qint8PerToken) {
                Buffer    send_scale                = reinterpret_cast<const QBuffer&>(*params.ag_send_buffer).scales();
                Buffer    recevie_scale             = reinterpret_cast<const QBuffer&>(gemm_a).scales();
                BufferPtr micro_batch_send_scale    = send_scale.slice(ag_token_idx, ag_micro_batch_tokens);
                BufferPtr mirco_batch_recevie_scale = recevie_scale.slice(token_idx, micro_batch_tokens);
                allGather({{mirco_batch_recevie_scale}, params.mode, {micro_batch_send_scale}, false, true});
                input_a_chunk = reinterpret_cast<const QBuffer&>(gemm_a).qslice(token_idx, micro_batch_tokens);
            } else if (params.qscheme == Qfp8PerTensor) {
                input_a_chunk = reinterpret_cast<const QBuffer&>(gemm_a).qslicePerTensor(token_idx, micro_batch_tokens);
            } else {
                RTP_LLM_FAIL("unsupported qscheme");
            }
            overlappedCommBarrier();
            auto micro_batch_gemm_params = GemmParams(*input_a_chunk,
                                                      linear_params.gemm_params.B,
                                                      linear_params.gemm_params.C,
                                                      output_d_chunk,
                                                      linear_params.gemm_params.compute_type,
                                                      linear_params.gemm_params.D_type,
                                                      linear_params.gemm_params.transA,
                                                      linear_params.gemm_params.transB,
                                                      linear_params.gemm_params.activationType,
                                                      linear_params.gemm_params.alpha,
                                                      linear_params.gemm_params.beta,
                                                      init_params_.device_resource_config.overlap_math_sm_count);
            loraLinear({micro_batch_gemm_params});
            token_idx += micro_batch_tokens;
            ag_token_idx += ag_micro_batch_tokens;
        }
        return AllGatherLoraLinearOutput({std::move(output), std::move(params.ag_recv_buffer)});
    }

    if (params.qscheme == NoQuantize || params.qscheme == Qfp8PerTensor) {
        allGather({{gemm_a.slice(0, gemm_a.shape()[0])}, params.mode, {params.ag_send_buffer}, false});
    } else if (params.qscheme == Qint8PerToken) {
        allGather({{gemm_a.slice(0, gemm_a.shape()[0])}, params.mode, {params.ag_send_buffer}, false});
        Buffer send_scale    = reinterpret_cast<const QBuffer&>(*params.ag_send_buffer).scales();
        Buffer recevie_scale = reinterpret_cast<const QBuffer&>(gemm_a).scales();
        allGather({{recevie_scale.slice(0, recevie_scale.shape()[0])},
                   params.mode,
                   {send_scale.slice(0, send_scale.shape()[0])},
                   false});
    } else {
        RTP_LLM_FAIL("unsupported qscheme");
    }
    output = loraLinear(linear_params).output;
    return AllGatherLoraLinearOutput({std::move(output), std::move(params.ag_recv_buffer)});
}

};  // namespace rtp_llm
