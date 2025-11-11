#pragma once

#include <torch/extension.h>
#include <vector>
#include <memory>
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/MlaParamsBase.h"

using namespace torch_ext;

namespace rtp_llm {

class FlashInferMlaAttnParams: public MlaParamsBase {
public:
    MlaParams fillParams(torch::Tensor t_prefix_lengths,
                         torch::Tensor t_sequence_lengths,
                         torch::Tensor t_input_lengths,
                         torch::Tensor t_kv_cache_block_id_host,
                         int           seq_size_per_block);
};
void registerPyFlashInferMlaParams(pybind11::module& m);

}  // namespace rtp_llm