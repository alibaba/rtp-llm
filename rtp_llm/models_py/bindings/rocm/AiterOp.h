#pragma once
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
namespace rtp_llm {
class AiterAttnPyParams: public ParamsBase {
public:
    AiterAttnPyParams();
    AiterAttnPyParams(torch::Tensor input_lengths, bool is_prefill);
    AiterAttnPyParams(torch::Tensor input_lengths,
                      torch::Tensor sequence_lengths,
                      torch::Tensor kv_cache_block_id_host,
                      torch::Tensor kv_cache_block_id_device,
                      bool          enable_cuda_graph);

    void update() override;

    bool check_recycle() override;

    // public:
    bool is_prefill_{true};
    bool enable_cuda_graph_{false};
    //     int  batch_size_{0};
    //     int  max_seq_len_{0};
    //     int  max_seqlen_q_{0};
    //     int  max_seqlen_k_{0};
};

// 独立函数声明：将 AiterAttnPyParams 实例转换为 ParamsBasePtr
ParamsBasePtr create_prefill_params(std::shared_ptr<AiterAttnPyParams> params);
ParamsBasePtr create_decode_params(std::shared_ptr<AiterAttnPyParams> params);

}  // namespace rtp_llm

namespace rtp_llm {
void registerAiterOp(const pybind11::module& m);
}  // namespace rtp_llm