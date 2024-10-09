#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"

namespace fastertransformer {

LoraLinearOutput DeviceBase::loraLinear(const LoraLinearParams& params) {
    auto output = gemm(params.gemm_params);
    if (params.lora_input) {
        auto& lora_input_lengths = *params.lora_input->lora_input_lengths_;
        auto& lora_a = params.lora_input->lora_a_;
        auto& lora_b = params.lora_input->lora_b_;
        int32_t* lora_input_lengths_ptr = lora_input_lengths.data<int32_t>();
        std::vector<BufferPtr> inputs;
        std::vector<BufferPtr> outputs;
        std::vector<BufferPtr> lora_as;
        std::vector<BufferPtr> lora_bs;
        size_t start = 0;
        for (int i = 0; i < lora_input_lengths.shape()[0]; i++) {
            if (lora_a[i] != nullptr && lora_b[i] != nullptr) {
                auto input_tmp = params.gemm_params.A.slice(start, lora_input_lengths_ptr[i]);
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
                FT_LOG_DEBUG("use same lora");
                auto tmp = gemm({params.gemm_params.A, *lora_as[0]});
                auto result = gemm({*tmp,
                                    *lora_bs[0],
                                    std::nullopt,
                                    output,
                                    DataType::TYPE_INVALID,
                                    TransposeOperation::NONE,
                                    TransposeOperation::NONE,
                                    ActivationType::Identity,
                                    1.0f,
                                    1.0f});
            } else if (useGroupGemm()) {
                FT_LOG_DEBUG("use group gemm");
                // M = X * A
                auto tmp = groupedGemm({inputs, lora_as});
                // Y = M * B + Y
                auto result = groupedGemm({tmp.output, lora_bs, outputs});
            } else {
                FT_LOG_DEBUG("use for gemm");
                for (int i = 0; i < inputs.size(); i++) {
                    auto tmp = gemm({*inputs[i], *lora_as[i]});
                    auto result = gemm({*tmp,
                                        *lora_bs[i],
                                        std::nullopt,
                                        outputs[i],
                                        DataType::TYPE_INVALID,
                                        TransposeOperation::NONE,
                                        TransposeOperation::NONE,
                                        ActivationType::Identity,
                                        1.0f,
                                        1.0f});
                }
            }

        }
    }
    return LoraLinearOutput({std::move(output)});
}

}; // namespace fastertransformer

