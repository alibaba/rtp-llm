#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/utils/ShapeCheck.h"

#include <optional>
#include <functional>
#include <algorithm>
#include <sstream>

namespace rtp_llm {

std::string GptModelInputs::debugString() const {
    if (!Logger::getEngineLogger().isDebugMode()) {
        return "";
    }
    std::stringstream debug_string;
    debug_string << "GptModelInputs { "
                 << "combo_tokens: " << combo_tokens->debugStringWithData<int32_t>()
                 << ", input_lengths: " << input_lengths->debugStringWithData<int32_t>()
                 << ", sequence_lengths: " << sequence_lengths->debugStringWithData<int32_t>()
                 << ", prefix_lengths: " << prefix_lengths->debugStringWithData<int32_t>();
    if (combo_position_ids) {
        debug_string << ", combo_position_ids: " << combo_position_ids->debugStringWithData<int32_t>();
    }
    if (lora_ids) {
        debug_string << ", lora_ids: " << lora_ids->debugStringWithData<int32_t>();
    }
    if (lora_input_lengths) {
        debug_string << ", lora_input_lengths: " << lora_input_lengths->debugStringWithData<int32_t>();
    }
    if (kv_cache_block_id) {
        debug_string << ", kv_cache_block_id: " << kv_cache_block_id->debugStringWithData<int32_t>();
    }
    if (attention_mask) {
        debug_string << ", attention_mask: " << attention_mask->debugString();
    }
    if (request_id) {
        debug_string << ", request_id: " << request_id->debugStringWithData<int64_t>();
    }
    if (request_pd_separation) {
        debug_string << ", request_pd_separation: " << request_pd_separation->debugStringWithData<bool>();
    }
    if (cache_keys) {
        debug_string << ", cache_keys: " << cache_keys->debugStringWithData<int64_t>();
    }
    debug_string << ", k block_size: " << k_block_size;
    debug_string << ", v block_size: " << v_block_size;
    debug_string << ", pd_separation: " << pd_separation;
    debug_string << "}";
    return debug_string.str();
}

// target independence params check
void GemmParams::check() const {

    // check dim
    auto dim = A.dim();
    RTP_LLM_CHECK_WITH_INFO((dim >= 1), "Gemm op param A dim %d should greater than 2.", A.dim());

    RTP_LLM_CHECK_WITH_INFO((B.dim() == dim), "Gemm op param B dim %d should be equal to A", B.dim());

    if (C != std::nullopt) {
        auto c_dim = C.value().get().dim();
        //  c_dim == 1: do broadcast
        RTP_LLM_CHECK_WITH_INFO(
            (c_dim == dim || c_dim == 1), "Gemm op param C dim %d should be equal to A and B", c_dim);
    }

    if (dim > 2) {
        bool batch_dim_same =
            std::equal(A.shape().begin(), A.shape().end() - 2, B.shape().begin(), B.shape().end() - 2);

        RTP_LLM_CHECK_WITH_INFO(batch_dim_same,
                                "Batch Gemm op A [%s] and B [%s] need batch shape same!",
                                ShapeStringView(A.shape()).c_str(),
                                ShapeStringView(B.shape()).c_str());

        if (C != std::nullopt) {
            bool batch_dim_same = std::equal(A.shape().begin(),
                                             A.shape().end() - 2,
                                             C.value().get().shape().begin(),
                                             C.value().get().shape().end() - 2);
            RTP_LLM_CHECK_WITH_INFO(batch_dim_same,
                                    "Batch Gemm op C [%s] need batch shape same!",
                                    ShapeStringView(C.value().get().shape()).c_str());
        }
    }

    auto m_a = (transA == TransposeOperation::NONE) ? A.shape()[dim - 2] : A.shape()[dim - 1];
    auto k_a = (transA == TransposeOperation::NONE) ? A.shape()[dim - 1] : A.shape()[dim - 2];

    auto k_b = (transB == TransposeOperation::NONE) ? B.shape()[dim - 2] : B.shape()[dim - 1];
    auto n_b = (transB == TransposeOperation::NONE) ? B.shape()[dim - 1] : B.shape()[dim - 2];

    RTP_LLM_CHECK_WITH_INFO((k_a == k_b),
                            "Gemm op A (%s) [%s] need compact with B (%s) [%s]!",
                            enumToString(transA).c_str(),
                            ShapeStringView(A.shape()).c_str(),
                            enumToString(transB).c_str(),
                            ShapeStringView(B.shape()).c_str());

    if (C != std::nullopt) {
        auto c_dim = C.value().get().dim();
        auto n_c   = C.value().get().shape()[c_dim - 1];

        RTP_LLM_CHECK_WITH_INFO((n_c == n_b),
                                "Gemm op B (%s) [%s] need compact with C [%s]!",
                                enumToString(transB).c_str(),
                                ShapeStringView(B.shape()).c_str(),
                                ShapeStringView(C.value().get().shape()).c_str());
        if (c_dim > 1) {
            auto m_c = C.value().get().shape()[c_dim - 2];
            RTP_LLM_CHECK_WITH_INFO((m_c == m_a),
                                    "Gemm op A (%s) [%s] need compact with C [%s]!",
                                    enumToString(transA).c_str(),
                                    ShapeStringView(A.shape()).c_str(),
                                    ShapeStringView(C.value().get().shape()).c_str());
        }
    }
}

GemmType GemmParams::dispatch() const {

    bool a_is_qbuffer = A.isQBuffer();
    bool b_is_qbuffer = B.isQBuffer();
    bool d_is_qbuffer = (D == nullptr) ? false : D->isQBuffer();

    if (A.dim() == 2) {
        if (!a_is_qbuffer && !b_is_qbuffer && !d_is_qbuffer) {
            return GemmType::BufferA_BufferB_BufferC_2DGemm;
        }
        if (a_is_qbuffer && !b_is_qbuffer && !d_is_qbuffer) {
            return GemmType::QBufferA_BufferB_BufferC_2DGemm;
        }
        if (!a_is_qbuffer && b_is_qbuffer && !d_is_qbuffer) {
            return GemmType::BufferA_QBufferB_BufferC_2DGemm;
        }
        if (a_is_qbuffer && b_is_qbuffer && !d_is_qbuffer) {
            return GemmType::QBufferA_QBufferB_BufferC_2DGemm;
        }

    } else if (A.dim() > 2) {
        if (!a_is_qbuffer && !b_is_qbuffer && !d_is_qbuffer) {
            return GemmType::BufferA_BufferB_BufferC_3DGemm;
        }
    }

    return GemmType::InvalidGemm;
}

// target independence params check
void GroupedGemmParams::check() const {

    auto a_size = A.size();
    auto b_size = B.size();
    auto c_size = (C.has_value()) ? C.value().size() : a_size;
    RTP_LLM_CHECK_WITH_INFO((a_size == b_size && b_size == c_size),
                            "group gemm needs all arguments to have same size.");

    for (int i = 0; i < int(a_size); i++) {
        auto a_dim = A[i]->dim();
        auto b_dim = B[i]->dim();
        auto c_dim = (C.has_value()) ? C.value()[i]->dim() : a_dim;
        RTP_LLM_CHECK_WITH_INFO((a_dim == 2 && b_dim == 2 && c_dim == 2), "group gemm needs A, B, C dim equal to 2.");

        auto a_type = A[i]->type();
        auto b_type = B[i]->type();
        auto c_type = (C.has_value()) ? C.value()[i]->type() : a_type;
        RTP_LLM_CHECK_WITH_INFO((a_type == b_type && b_type == c_type), "group gemm needs A, B, C has same dtype.");

        auto m_a = A[i]->shape()[0];
        auto k_a = A[i]->shape()[1];
        auto k_b = B[i]->shape()[0];
        auto n_b = B[i]->shape()[1];
        auto m_c = (C.has_value()) ? C.value()[i]->shape()[0] : m_a;
        auto n_c = (C.has_value()) ? C.value()[i]->shape()[1] : n_b;
        RTP_LLM_CHECK_WITH_INFO((m_a == m_c && k_a == k_b && n_b == n_c),
                                "group gemm[%d] A, B, C (%d, %d, %d, %d, %d, %d) valid.",
                                i,
                                m_a,
                                k_a,
                                m_c,
                                n_b,
                                k_b,
                                n_c);
    }
}

}  // namespace rtp_llm
