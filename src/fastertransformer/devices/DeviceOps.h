#pragma once

#include "src/fastertransformer/devices/OpData.h"

namespace fastertransformer {

class DeviceOps {
public:
    DeviceOps();
    virtual ~DeviceOps();

public:
    // tensor ops
    virtual void copy(const CopyParams& params);
    virtual CloneOutput clone(const CloneParams& params);
    virtual TransposeOutput transpose(const TransposeParams& params);
    virtual ConvertOutput convert(const ConvertParams& params);
    virtual SelectOutput select(const SelectParams& params);
    virtual ConcatOutput concat(const ConcatParams& params);
    virtual SplitOutput split(const SplitParams& params);
    virtual void         bufMemset(Buffer& buf, int val);

    // basic compuation ops
    virtual LayernormOutput layernorm(const LayernormParams& params);
    virtual AddBiasOutput addbias(const AddBiasParams& params);
    virtual BufferPtr gemm(const GemmParams& params);
    virtual GroupedGemmOutput groupedGemm(const GroupedGemmParams& params);
    virtual MultiplyOutput multiply(const MultiplyParams& params);
    virtual BufferPtr embeddingLookup(const EmbeddingLookupParams& params);
    virtual BufferPtr multimodalEmbedding(const MultimodalEmbeddingParams& params);
    virtual BufferPtr activation(const ActivationParams& params);
    virtual BufferPtr softmax(const SoftmaxParams& params);
    virtual LossOutput loss(const LossParams& params);
    virtual MaskOutput attentionMask(const MaskParams& params);
    virtual BufferPtr loraLinearWithActivation(const LoraLinearWithActivationParams& params);

    // QKV ops
    virtual BufferPtr mhaQKVGemm(const AttentionLayerParams& params);
    virtual BufferPtr mlaQKVGemm(const AttentionLayerParams& params);

    // dedicated attention ops
    virtual AttentionModuleOutput contextAttention(const AttentionModuleParams& params);
    virtual AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params);
    virtual AttentionModuleOutput mlaContextAttention(const MlaAttentionModuleParams& params);
    virtual AttentionModuleOutput mlaDecoderSelfAttention(const MlaDecoderAttentionParams& params);
    virtual void mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params);

    // Top level model ops
    virtual AttentionLayerOutput attentionLayer(const AttentionLayerParams& params);
    virtual AttentionLayerOutput mlaAttentionLayer(const AttentionLayerParams& params);
    virtual FfnLayerOutput ffnLayer(const FfnLayerParams& params);
    virtual FfnLayerOutput microBatchedFfnLayer(const FfnLayerParams& params);
    virtual FfnLayerOutput moeFfnLayer(const FfnLayerParams& params);
    virtual FfnLayerOutput moeSharedExpert(const FfnLayerParams& params);
    virtual MoeGateSelectOutput moeGateSelect(const FfnLayerParams& params);
    virtual FfnLayerOutput moeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs);
    virtual FfnLayerOutput moeFfnAndCombine(const FfnLayerParams& params, const MoeDispatchOutput& dispatched_output);
    virtual LoraLinearOutput loraLinear(const LoraLinearParams& params);
    virtual LoraLinearOutput loraLinearWithAllReduce(const LoraLinearParams& params);
    virtual MoeDispatchOutput epDispatch(const MoeDispatchParams& params);
    virtual FfnLayerOutput epCombine(const MoeCombineParams& params);

    // for sampler
    virtual GreedyOutput sampleGreedy(const GreedyParams& params);
    virtual void sampleBeamSearch(const BeamSearchParams& params);

    // for device communication
    virtual void broadcast(const BroadcastParams& params);
    virtual AllReduceOutput allReduce(const AllReduceParams& params);
    virtual void allGather(const AllGatherParams& params);
    virtual AllToAllOutput allToAll(const AllToAllParams& params);
    virtual PrepareAllReduceOutput prepareAllReduce(const PrepareAllReduceParams& params);

    // for quantization
    virtual BufferPtr quantize(const QuantizeParams& params);

    // for multi thread no block copy
    virtual void noBlockCopy(const CopyParams& params);

    // for perf
    virtual void perfRangePush(const std::string& name) const;
    virtual void perfRangePop() const;

    // for device-specific weights preprocess
    static torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight);
    static torch::Tensor packInt8TensorToPackedInt4(torch::Tensor weight);
    static torch::Tensor preprocessWeightsForMixedGemm(torch::Tensor weight, torch::ScalarType quant_type, const std::string &arch);
    static std::vector<torch::Tensor> symmetricQuantizeLastAxisOfBatchedMatrix(torch::Tensor weight, torch::ScalarType quant_type);
    static std::vector<torch::Tensor> symmetricQuantizeLastAxisOfBatchedMatrix(torch::Tensor weight, torch::ScalarType quant_type, const std::string &arch);
    static torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale);
};

}  // namespace fastertransformer
