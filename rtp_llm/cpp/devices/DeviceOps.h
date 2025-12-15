#pragma once

#include "rtp_llm/cpp/devices/OpData.h"
#include "torch/extension.h"

namespace rtp_llm {

class DeviceOps {
public:
    DeviceOps();
    virtual ~DeviceOps();

public:
    // tensor ops
    virtual void            copy(const CopyParams& params);
    virtual void            multiMergeCopy(const MultiMergeCopyParams& params);
    virtual void            multiCopy(const MultiCopyParams& params);
    virtual void            batchCopy(const BatchCopyParams& params);
    virtual CloneOutput     clone(const CloneParams& params);
    virtual TransposeOutput transpose(const TransposeParams& params);
    virtual ConvertOutput   convert(const ConvertParams& params);
    virtual SelectOutput    select(const SelectParams& params);
    virtual ConcatOutput    concat(const ConcatParams& params);
    virtual SplitOutput     split(const SplitParams& params);
    virtual SliceOutput     slice(const SliceParams& params);
    virtual void            bufMemset(Buffer& buf, int val, DeviceStream stream = DeviceStream::DEFAULT);

    // basic compuation ops
    virtual LayernormOutput   layernorm(const LayernormParams& params);
    virtual LayernormOutput   layernormWithStride(const LayernormWithStrideParams& params);
    virtual QkRmsNormOutput   qkRmsNorm(const QkRmsNormParams& params);
    virtual AddBiasOutput     addbias(const AddBiasParams& params);
    virtual BufferPtr         gemm(const GemmParams& params);
    virtual GroupedGemmOutput groupedGemm(const GroupedGemmParams& params);
    virtual MultiplyOutput    multiply(const MultiplyParams& params);
    virtual BufferPtr         embeddingLookup(const EmbeddingLookupParams& params);
    virtual BufferPtr         multimodalEmbedding(const MultimodalEmbeddingParams& params);
    virtual BufferPtr         inputEmbedding(const InputEmbeddingParams& params);
    virtual BufferPtr         activation(const ActivationParams& params);
    virtual BufferPtr         softmax(const SoftmaxParams& params);
    virtual LossOutput        loss(const LossParams& params);
    virtual MaskOutput        attentionMask(const MaskParams& params);
    virtual BufferPtr         loraLinearWithActivation(const LoraLinearWithActivationParams& params);
    virtual void              maskLogits(Buffer& logits, const Buffer& mask);
    virtual void              weightLogits(Buffer& logits, const Buffer& weigth);

    // QKV ops
    virtual BufferPtr mhaQKVGemm(const AttentionLayerParams& params);
    virtual BufferPtr mlaQKVGemm(const AttentionLayerParams& params);
    virtual BufferPtr attentionQKVGemm(const AttentionLayerParams& params);
    virtual BufferPtr attentionAttn(const AttentionLayerParams& params);
    virtual BufferPtr attentionOutGemm(const AttentionLayerParams& params);

    // dedicated attention ops
    virtual AttentionModuleOutput contextAttention(const AttentionModuleParams& params);
    virtual AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params);
    virtual AttentionModuleOutput mlaContextAttention(const MlaAttentionModuleParams& params);
    virtual AttentionModuleOutput mlaAbsorbAttention(const MlaAttentionModuleParams& params);
    virtual void                  mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params);

    // Top level model ops
    virtual AttentionLayerOutput attentionLayer(const AttentionLayerParams& params);
    virtual AttentionLayerOutput mlaAttentionLayer(const AttentionLayerParams& params);
    virtual FfnLayerOutput       ffnLayer(const FfnLayerParams& params);
    virtual FfnLayerOutput       microBatchedFfnLayer(const FfnLayerParams& params);
    virtual FfnLayerOutput       moeFfnLayer(const FfnLayerParams& params);
    virtual FfnLayerOutput       epMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output);
    virtual FfnLayerOutput       moeSharedExpert(const FfnLayerParams& params);
    virtual MoeGateSelectOutput  moeGateSelect(const FfnLayerParams& params);
    virtual FfnLayerOutput       moeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs);
    virtual FfnLayerOutput       moeFfnFp8(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs);
    virtual FfnLayerOutput moeFfnFp8Contiguous(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs);
    virtual FfnLayerOutput moeFfnFp8Masked(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs);
    virtual FfnLayerOutput deepEpMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs);
    virtual FfnLayerOutput deepEpLLMoeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs);

    virtual LoraLinearOutput              loraLinear(const LoraLinearParams& params);
    virtual LoraLinearOutput              loraLinearWithAllReduce(const LoraLinearParams& params);
    virtual MoeDispatchOutput             epDispatch(const MoeDispatchParams& params);
    virtual MoeCombineOutput              epCombine(const MoeCombineParams& params);
    virtual FfnLayerOutput                gatherCombineOutput(const MoeCombineOutput& params);
    virtual ReduceScatterLoraLinearOutput loraLinearReduceScatter(const LoraLinearReduceScatterParams& params);
    virtual AllGatherLoraLinearOutput     allGatherloraLinear(const AllGatherLoraLinearParams& params);
    virtual void                          chainSpeculativeSampling(const SpeculativeSamplingParams& params);

    // for sampler
    virtual GreedyOutput     sampleGreedy(const GreedyParams& params);
    virtual BeamSearchOutput sampleBeamSearch(const BeamSearchParams& params);

    // for device communication
    virtual void                   broadcast(const BroadcastParams& params);
    virtual void                   batchSendRecv(const BatchSendRecvParams& params, const ParallelMode& mode);
    virtual AllReduceOutput        allReduce(const AllReduceParams& params);
    virtual void                   allGather(const AllGatherParams& params);
    virtual AllToAllOutput         allToAll(const AllToAllParams& params);
    virtual void                   reduceScatter(const ReduceScatterParams& params);
    virtual PrepareAllReduceOutput prepareAllReduce(const PrepareAllReduceParams& params);

    // for quantization
    virtual BufferPtr quantize(const QuantizeParams& params);

    // for multi thread no block copy
    virtual void noBlockCopy(const CopyParams& params);
    virtual void noBlockCopy(const MultiCopyParams& params);

    // for perf
    virtual void perfRangePush(const std::string& name) const;
    virtual void perfRangePop() const;

    // for check
    virtual bool checkNAN(const Buffer& input);

    // for device-specific weights preprocess
    static torch::Tensor
    preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool user_arm_gemm_use_kai);
    static torch::Tensor              preprocessWeightScale(torch::Tensor weight, torch::Tensor scale);

    virtual void prepareCommBuffer(const PrepareCommBufferParams& params);
};

}  // namespace rtp_llm
