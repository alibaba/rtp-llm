#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/core/ShapeCheck.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

#include <optional>
#include <functional>
#include <algorithm>
#include <sstream>

namespace rtp_llm {

std::string combineStrings(const std::vector<std::string>& vec) {
    std::string result = "\" ";
    for (const auto& s : vec) {
        result += s + ", ";
    }
    result += "\"";
    return result;
}

OpException::OpException(const OpStatus& status): status_(status) {
    std::stringstream ss;
    ss << "OpException[" << (int32_t)status_.error_type << "]: " << status_.error_message << std::endl;
    RTP_LLM_LOG_INFO("%s", ss.str().c_str());
    const auto stack = rtp_llm::getStackTrace();
    RTP_LLM_STACKTRACE_LOG_INFO("%s", stack.c_str());
    ss << stack;
    detail_str_ = ss.str();
    if (StaticConfig::user_ft_core_dump_on_exception) {
        fflush(stdout);
        fflush(stderr);
        abort();
    }
}

std::string GptModelInputs::debugString(bool force) const {
    if (!Logger::getEngineLogger().isDebugMode() && !force) {
        return "";
    }
    auto              tb = [](const torch::Tensor& t) -> std::string { return tensorDebugString(t); };
    std::stringstream debug_string;
    debug_string << "GptModelInputs { "
                 << "trace_ids: " << combineStrings(trace_ids) << ", combo_tokens: " << tb(combo_tokens)
                 << ", input_lengths: " << tb(input_lengths) << ", sequence_lengths: " << tb(sequence_lengths)
                 << ", prefix_lengths: " << tb(prefix_lengths);
    if (combo_position_ids.defined()) {
        debug_string << ", combo_position_ids: " << tb(combo_position_ids);
    }
    if (kv_cache_kernel_block_id.defined()) {
        debug_string << ", kv_cache_kernel_block_id: " << tb(kv_cache_kernel_block_id);
    }
    if (kv_cache_block_id.defined()) {
        debug_string << ", kv_cache_block_id: " << tb(kv_cache_block_id);
    }
    if (attention_mask.defined()) {
        debug_string << ", attention_mask: " << tb(attention_mask);
    }
    if (request_id.defined()) {
        debug_string << ", request_id: " << tb(request_id);
    }
    if (request_pd_separation.defined()) {
        debug_string << ", request_pd_separation: " << tb(request_pd_separation);
    }
    if (cache_keys.defined()) {
        debug_string << ", cache_keys: " << tb(cache_keys);
    }
    debug_string << ", kv_block_stride_bytes: " << kv_block_stride_bytes;
    debug_string << ", pd_separation: " << pd_separation;
    debug_string << "}";
    return debug_string.str();
}

namespace {
inline std::vector<size_t> torchSizesToVec(c10::IntArrayRef sizes) {
    return std::vector<size_t>(sizes.begin(), sizes.end());
}
}  // namespace

// target independence params check
void GemmParams::check() const {
    auto dim = A.dim();
    RTP_LLM_CHECK_WITH_INFO((dim >= 1), "Gemm op param A dim %ld should greater than 1.", (long)A.dim());
    RTP_LLM_CHECK_WITH_INFO((B.dim() == dim), "Gemm op param B dim %ld should be equal to A", (long)B.dim());

    if (C.has_value()) {
        auto c_dim = C->dim();
        RTP_LLM_CHECK_WITH_INFO(
            (c_dim == dim || c_dim == 1), "Gemm op param C dim %ld should be equal to A and B", (long)c_dim);
    }

    auto a_sizes = torchSizesToVec(A.sizes());
    auto b_sizes = torchSizesToVec(B.sizes());

    if (dim > 2) {
        bool batch_dim_same = std::equal(a_sizes.begin(), a_sizes.end() - 2, b_sizes.begin(), b_sizes.end() - 2);

        RTP_LLM_CHECK_WITH_INFO(batch_dim_same,
                                "Batch Gemm op A [%s] and B [%s] need batch shape same!",
                                ShapeStringView(a_sizes).c_str(),
                                ShapeStringView(b_sizes).c_str());

        if (C.has_value()) {
            auto c_sizes        = torchSizesToVec(C->sizes());
            bool batch_dim_same = std::equal(a_sizes.begin(), a_sizes.end() - 2, c_sizes.begin(), c_sizes.end() - 2);
            RTP_LLM_CHECK_WITH_INFO(
                batch_dim_same, "Batch Gemm op C [%s] need batch shape same!", ShapeStringView(c_sizes).c_str());
        }
    }

    auto m_a = (transA == TransposeOperation::NONE) ? a_sizes[dim - 2] : a_sizes[dim - 1];
    auto k_a = (transA == TransposeOperation::NONE) ? a_sizes[dim - 1] : a_sizes[dim - 2];

    auto k_b = (transB == TransposeOperation::NONE) ? b_sizes[dim - 2] : b_sizes[dim - 1];
    auto n_b = (transB == TransposeOperation::NONE) ? b_sizes[dim - 1] : b_sizes[dim - 2];

    RTP_LLM_CHECK_WITH_INFO((k_a == k_b),
                            "Gemm op A (%s) [%s] need compact with B (%s) [%s]!",
                            enumToString(transA).c_str(),
                            ShapeStringView(a_sizes).c_str(),
                            enumToString(transB).c_str(),
                            ShapeStringView(b_sizes).c_str());

    if (C.has_value()) {
        auto c_sizes = torchSizesToVec(C->sizes());
        auto c_dim   = C->dim();
        auto n_c     = c_sizes[c_dim - 1];

        RTP_LLM_CHECK_WITH_INFO((n_c == n_b),
                                "Gemm op B (%s) [%s] need compact with C [%s]!",
                                enumToString(transB).c_str(),
                                ShapeStringView(b_sizes).c_str(),
                                ShapeStringView(c_sizes).c_str());
        if (c_dim > 1) {
            auto m_c = c_sizes[c_dim - 2];
            RTP_LLM_CHECK_WITH_INFO((m_c == m_a),
                                    "Gemm op A (%s) [%s] need compact with C [%s]!",
                                    enumToString(transA).c_str(),
                                    ShapeStringView(a_sizes).c_str(),
                                    ShapeStringView(c_sizes).c_str());
        }
    }
}

// target independence params check
void GroupedGemmParams::check() const {
    auto a_size = A.size();
    auto b_size = B.size();
    auto c_size = (C.has_value()) ? C.value().size() : a_size;
    RTP_LLM_CHECK_WITH_INFO((a_size == b_size && b_size == c_size),
                            "group gemm needs all arguments to have same size.");

    for (int i = 0; i < int(a_size); i++) {
        auto a_dim = A[i].dim();
        auto b_dim = B[i].dim();
        auto c_dim = (C.has_value()) ? C.value()[i].dim() : a_dim;
        RTP_LLM_CHECK_WITH_INFO((a_dim == 2 && b_dim == 2 && c_dim == 2), "group gemm needs A, B, C dim equal to 2.");

        auto a_type = A[i].scalar_type();
        auto b_type = B[i].scalar_type();
        auto c_type = (C.has_value()) ? C.value()[i].scalar_type() : a_type;
        RTP_LLM_CHECK_WITH_INFO((a_type == b_type && b_type == c_type), "group gemm needs A, B, C has same dtype.");

        auto m_a = A[i].size(0);
        auto k_a = A[i].size(1);
        auto k_b = B[i].size(0);
        auto n_b = B[i].size(1);
        auto m_c = (C.has_value()) ? C.value()[i].size(0) : m_a;
        auto n_c = (C.has_value()) ? C.value()[i].size(1) : n_b;
        RTP_LLM_CHECK_WITH_INFO((m_a == m_c && k_a == k_b && n_b == n_c),
                                "group gemm[%d] A, B, C (%ld, %ld, %ld, %ld, %ld, %ld) valid.",
                                i,
                                (long)m_a,
                                (long)k_a,
                                (long)m_c,
                                (long)n_b,
                                (long)k_b,
                                (long)n_c);
    }
}

BatchCopyParams::CopyType BatchCopyParams::get_copy_type(MemoryType dst_type, MemoryType src_type) {
    if (src_type == MEMORY_GPU) {
        if (dst_type == MEMORY_GPU) {
            return CopyType::D2D;
        } else {
            return CopyType::D2H;
        }
    } else {
        if (dst_type == MEMORY_GPU) {
            return CopyType::H2D;
        } else {
            return CopyType::H2H;
        }
    }
}

BatchCopyParams& BatchCopyParams::reserve(CopyType copy_type, size_t size) {
    RTP_LLM_CHECK_WITH_INFO(copy_type < TYPE_SIZE, "unexpected CopyType %d", copy_type);
    copy_buffers[copy_type].dst_ptr.reserve(size);
    copy_buffers[copy_type].src_ptr.reserve(size);
    copy_buffers[copy_type].sizes.reserve(size);

    return *this;
}

BatchCopyParams& BatchCopyParams::add(void* dst, const void* src, size_t size, CopyType copy_type) {
    RTP_LLM_CHECK_WITH_INFO(copy_type < TYPE_SIZE, "unexpected CopyType %d", copy_type);

    if (size > 0) {
        auto& buffers = copy_buffers[copy_type];
        buffers.dst_ptr.push_back(dst);
        buffers.src_ptr.push_back(src);
        buffers.sizes.push_back(size);
    }

    return *this;
}

}  // namespace rtp_llm
