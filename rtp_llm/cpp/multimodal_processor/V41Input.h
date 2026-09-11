#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct V41ImageInput {
    int32_t       start   = 0;
    int32_t       n_vit_h = 0;
    int32_t       n_vit_w = 0;
    torch::Tensor patches;
    torch::Tensor types;
    std::string   content_sha256;
    std::string   processor_identity;
};

struct V41RequestInputs {
    torch::Tensor              token_types;  // CPU int32 [canonical tokens], text=-1, image types=0..3.
    torch::Tensor              image_mask;   // CPU bool [canonical tokens], including all three delimiters.
    std::vector<V41ImageInput> images;

    void validateChunk(int64_t begin, int64_t end) const {
        RTP_LLM_CHECK_WITH_INFO(begin >= 0 && end >= begin && end <= token_types.numel(),
                                "V4.1 prefill chunk must lie within the canonical prompt");
        for (const auto& image : images) {
            const int64_t image_end = static_cast<int64_t>(image.start) + image.types.numel();
            RTP_LLM_CHECK_WITH_INFO(image.start >= end || image_end <= begin
                                        || (image.start >= begin && image_end <= end),
                                    "V4.1 global prefill chunks and prefix reuse must not split an image span");
        }
    }

    static bool isDigest(const std::string& digest) {
        return digest.size() == 64 && std::all_of(digest.begin(), digest.end(), [](char c) {
                   return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
               });
    }

    void validate(const torch::Tensor& ids) const {
        RTP_LLM_CHECK_WITH_INFO(ids.defined() && ids.device().is_cpu() && ids.scalar_type() == torch::kInt32
                                    && ids.dim() == 1 && ids.is_contiguous() && ids.numel() <= 1048576,
                                "V4.1 IDs must be contiguous canonical CPU int32 within the 1M context limit");
        RTP_LLM_CHECK_WITH_INFO(token_types.defined() && token_types.device().is_cpu()
                                    && token_types.scalar_type() == torch::kInt32 && token_types.dim() == 1
                                    && token_types.is_contiguous() && token_types.numel() == ids.numel(),
                                "V4.1 token types must be contiguous CPU int32 with one value per canonical token");
        RTP_LLM_CHECK_WITH_INFO(image_mask.defined() && image_mask.device().is_cpu()
                                    && image_mask.scalar_type() == torch::kBool && image_mask.dim() == 1
                                    && image_mask.is_contiguous() && image_mask.numel() == ids.numel(),
                                "V4.1 image mask must be contiguous CPU bool with one value per canonical token");
        const auto*       tokens = ids.data_ptr<int32_t>();
        const auto*       kinds  = token_types.data_ptr<int32_t>();
        const auto*       mask   = image_mask.data_ptr<bool>();
        std::vector<bool> covered(ids.numel(), false);
        int64_t           previous_end = 0;
        for (const auto& image : images) {
            RTP_LLM_CHECK_WITH_INFO(image.n_vit_h > 0 && image.n_vit_w > 0 && image.n_vit_h <= 3066
                                        && image.n_vit_w <= 3066,
                                    "V4.1 prepared image must have a bounded positive ViT grid");
            const int64_t rows   = (image.n_vit_h + 2) / 3;
            const int64_t cols   = (image.n_vit_w + 2) / 3;
            const int64_t length = rows * (cols + 1) + 2;
            const int64_t end    = static_cast<int64_t>(image.start) + length;
            RTP_LLM_CHECK_WITH_INFO(length <= 1024 && image.start >= previous_end && end <= ids.numel(),
                                    "V4.1 image spans must be complete, sorted and disjoint");
            RTP_LLM_CHECK_WITH_INFO(image.patches.device().is_cpu() && image.patches.scalar_type() == torch::kBFloat16
                                        && image.patches.is_contiguous() && image.patches.dim() == 4
                                        && image.patches.size(0) == static_cast<int64_t>(image.n_vit_h) * image.n_vit_w
                                        && image.patches.size(1) == 3 && image.patches.size(2) == 14
                                        && image.patches.size(3) == 14,
                                    "V4.1 patches must contain a complete contiguous CPU BF16 image grid");
            RTP_LLM_CHECK_WITH_INFO(image.types.device().is_cpu() && image.types.scalar_type() == torch::kInt32
                                        && image.types.dim() == 1 && image.types.is_contiguous()
                                        && image.types.numel() == length,
                                    "V4.1 image types must describe the complete image");
            RTP_LLM_CHECK_WITH_INFO(isDigest(image.content_sha256) && isDigest(image.processor_identity),
                                    "V4.1 image cache identity requires content and processor SHA256");
            const auto* image_types = image.types.data_ptr<int32_t>();
            for (int64_t offset = 0; offset < length; ++offset) {
                const int32_t expected = offset == 0          ? 0 :
                                         offset == length - 1 ? 3 :
                                                                ((offset - 1) % (cols + 1) == cols ? 2 : 1);
                const int64_t index    = image.start + offset;
                RTP_LLM_CHECK_WITH_INFO(tokens[index] == 129264 && kinds[index] == expected
                                            && image_types[offset] == expected,
                                        "V4.1 image rows must retain canonical ID 129264 and row-major delimiters");
                covered[index] = true;
            }
            previous_end = end;
        }
        for (int64_t index = 0; index < ids.numel(); ++index) {
            RTP_LLM_CHECK_WITH_INFO(tokens[index] >= 0 && tokens[index] < 129280 && kinds[index] >= -1
                                        && kinds[index] <= 3 && mask[index] == (kinds[index] != -1)
                                        && covered[index] == mask[index] && (tokens[index] != 129264 || mask[index]),
                                    "V4.1 canonical tokens, image types and image mask disagree");
        }
    }
};

}  // namespace rtp_llm
