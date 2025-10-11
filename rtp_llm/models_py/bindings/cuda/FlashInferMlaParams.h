#pragma once

#include <torch/extension.h>
#include <vector>
#include <memory>
#include "rtp_llm/models_py/bindings/OpDefs.h"
using namespace torch_ext;

namespace rtp_llm {

/**
 * @param page_size The size of each page in the KV cache
 * @param attention_inputs The attention inputs containing sequence information
 * @param device The torch device to create tensors on
 * @return FlashInferMlaParams structure containing all the computed parameters
 */
FlashInferMlaParams FillFlashInferMlaParams(int page_size, 
                                           const PyAttentionInputs& attention_inputs, 
                                           const torch::Device& device);

}