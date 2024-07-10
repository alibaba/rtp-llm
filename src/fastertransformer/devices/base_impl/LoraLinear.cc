#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"

namespace fastertransformer {

LoraLinearOutput DeviceBase::loraLinear(const LoraLinearParams& params) {
    auto output = gemm(params.gemm_params);

    if (params.lora_input != std::nullopt && params.lora_map != std::nullopt)
    {
        auto& lora_ids = *params.lora_input.value().lora_ids;
        auto& lora_input_lengths = *params.lora_input.value().lora_input_lengths;
        printBufferData(lora_ids, "lora_ids");
        printBufferData(lora_input_lengths, "lora_input_lengths");
        int32_t* lora_ids_ptr = lora_ids.data<int32_t>();
        int32_t* lora_input_lengths_ptr = lora_input_lengths.data<int32_t>();
        std::vector<BufferPtr> inputs;
        std::vector<BufferPtr> outputs;
        std::vector<BufferPtr> lora_as;
        std::vector<BufferPtr> lora_bs;
        size_t start = 0;
        for (int i = 0; i < lora_ids.shape()[0]; i++) {
            if (lora_ids_ptr[i] >= 0 && params.lora_map.value().get().hasLoraWeight(lora_ids_ptr[i])) {
                auto input_tmp = params.gemm_params.A.slice(start, lora_input_lengths_ptr[i]);
                auto output_tmp = output->slice(start, lora_input_lengths_ptr[i]);
                auto lora_a_tmp = params.lora_map.value().get().getLoraWeight(lora_ids_ptr[i]).A;
                auto lora_b_tmp = params.lora_map.value().get().getLoraWeight(lora_ids_ptr[i]).B;
                inputs.push_back(input_tmp);
                outputs.push_back(output_tmp);
                lora_as.push_back(std::const_pointer_cast<Buffer>(lora_a_tmp));
                lora_bs.push_back(std::const_pointer_cast<Buffer>(lora_b_tmp));
            }
            start = start + lora_input_lengths_ptr[i];
        }
        if (inputs.size() > 0) {
            // M = X * A
            auto tmp = groupedGemm({inputs, lora_as});
            // Y = M * B + Y
            auto result = groupedGemm({tmp.output, lora_bs, outputs});
        }

    }

    return LoraLinearOutput({std::move(output)});
}

}; // namespace fastertransformer

