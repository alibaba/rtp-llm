#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <torch/torch.h>

namespace rtp_llm {

// V4.1 prepared-image request metadata transported through the typed
// v41_inputs channel. Filled by the renderer's V41PreparedInputs and
// consumed by MultimodalProcessor (local ViT) and the engine input path.
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
};

}  // namespace rtp_llm