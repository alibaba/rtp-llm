#include "rocmMoeWrapper.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

#include "ck/ck.hpp"
#include "fused_moe.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_permute.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"

#include "ck/utility/data_type.hpp"
#include "ck/library/utility/device_memory.hpp"

namespace rtp_llm {

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

void* add_offset(void* ptr, std::size_t offset, std::size_t element_size) {
    char* char_ptr = static_cast<char*>(ptr);
    char_ptr += offset * element_size;
    return static_cast<void*>(char_ptr);
}

using Silu    = ck::tensor_operation::element_wise::Silu;
using Gelu    = ck::tensor_operation::element_wise::Gelu;
using Relu    = ck::tensor_operation::element_wise::Relu;
using Sigmoid = ck::tensor_operation::element_wise::Sigmoid;

uint32_t rocmMoeWrapper::runCKMoe(const rocmMoeParams& params,
                                  DataType             dtype,
                                  DataType             wtype,
                                  ActivationType       activation_type,
                                  int                  fused_quant,
                                  int                  gate_only) {
    uint32_t rslt = 0;

    // Using CK fused_moe
    auto prec_i = "fp16";
    auto prec_w = "fp16";
    auto prec_o = "fp16";
    if (dtype == DataType::TYPE_FP16 && wtype == DataType::TYPE_FP16) {
        prec_i = "fp16";
        prec_w = "fp16";
        prec_o = "fp16";
    } else if (dtype == DataType::TYPE_BF16 && wtype == DataType::TYPE_BF16) {
        prec_i = "bf16";
        prec_w = "bf16";
        prec_o = "bf16";
    } else {
        CK_FAIL("input type %d and weights type %d not supported by CK", dtype, wtype);
        return rslt;
    }
    // TODO: temporarily fixed scale type
    auto prec_st = "fp32";
    auto prec_sw = "fp32";
    auto prec_sq = "fp32";
    auto prec_kw = "fp32";

    int act_type = 0;  // default to gelu.
    switch (activation_type) {
        case ActivationType::Gelu:
        case ActivationType::Geglu:
            act_type = 0;
            break;
        case ActivationType::Silu:
        case ActivationType::Swiglu:
            act_type = 1;
            break;
        default:
            CK_FAIL("not support activation type");
    }

    fused_moe_traits traits{prec_i,
                            prec_w,
                            prec_o,
                            prec_st,
                            prec_sw,
                            prec_sq,
                            prec_kw,
                            static_cast<int>(params.block_m),
                            act_type,
                            gate_only,
                            fused_quant};

    fused_moe_args args{params.input,
                        params.input_scale_ptr,
                        params.gate_ptr,
                        params.down_ptr,
                        params.gate_scale_ptr,
                        params.down_scale_ptr,
                        params.smooth_scale_ptr,
                        nullptr,  // local_expert_mask_ptr
                        params.output_ptr,
                        nullptr,  // ws_ptr
                        params.topk_ids_ptr,
                        params.topk_weight_ptr,
                        params.sorted_token_ids_ptr,
                        params.sorted_weight_ptr,
                        params.sorted_expert_ids_ptr,
                        params.num_sorted_tiles_ptr,
                        static_cast<int>(params.block_m),
                        static_cast<int>(params.hidden_size),
                        static_cast<int>(params.intermediate_size),
                        static_cast<int>(params.num_tokens),
                        static_cast<int>(params.num_experts),
                        static_cast<int>(params.topk),
                        static_cast<int>(params.stride_token)};
    // printf("[Debug]h: %i; inter: %i, token: %i, expert: %i, topk: %i, stride: %i, go: %i, quant: %i.\n",
    // args.hidden_size, args.intermediate_size, args.num_tokens, args.num_experts, args.topk, args.stride_token,
    // gate_only, fused_quant);
    float ave_time = fused_moe(traits, args, ck_tile::stream_config{params.stream, false, 0, 0, 1});
    if (ave_time < 0) {
        CK_FAIL("Fused Moe fail");
    }
    return rslt;
}

}  // namespace rtp_llm