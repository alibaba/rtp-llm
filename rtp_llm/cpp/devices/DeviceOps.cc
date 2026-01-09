#include "rtp_llm/cpp/devices/DeviceOps.h"
#include "OpData.h"

namespace rtp_llm {

DeviceOps::DeviceOps() {}

DeviceOps::~DeviceOps() {}

void DeviceOps::copy(const CopyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::batchCopy(const BatchCopyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

CloneOutput DeviceOps::clone(const CloneParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

TransposeOutput DeviceOps::transpose(const TransposeParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::batchSendRecv(const BatchSendRecvParams& params, const ParallelMode& mode) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

ConvertOutput DeviceOps::convert(const ConvertParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

SelectOutput DeviceOps::select(const SelectParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

ConcatOutput DeviceOps::concat(const ConcatParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

SplitOutput DeviceOps::split(const SplitParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

LayernormOutput DeviceOps::layernorm(const LayernormParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

LayernormOutput DeviceOps::layernormWithStride(const LayernormWithStrideParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

QkRmsNormOutput DeviceOps::qkRmsNorm(const QkRmsNormParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

SliceOutput DeviceOps::slice(const SliceParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AddBiasOutput DeviceOps::addbias(const AddBiasParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::loraLinearWithActivation(const LoraLinearWithActivationParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::gemm(const GemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

GroupedGemmOutput DeviceOps::groupedGemm(const GroupedGemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

MultiplyOutput DeviceOps::multiply(const MultiplyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::embeddingLookup(const EmbeddingLookupParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::multimodalEmbedding(const MultimodalEmbeddingParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::inputEmbedding(const InputEmbeddingParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::activation(const ActivationParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::softmax(const SoftmaxParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

LossOutput DeviceOps::loss(const LossParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

MaskOutput DeviceOps::attentionMask(const MaskParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::mhaQKVGemm(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::mlaQKVGemm(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::attentionQKVGemm(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::attentionAttn(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::attentionOutGemm(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput DeviceOps::contextAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput DeviceOps::decoderSelfAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionLayerOutput DeviceOps::mlaAttentionLayer(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput DeviceOps::mlaContextAttention(const MlaAttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput DeviceOps::mlaAbsorbAttention(const MlaAttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionLayerOutput DeviceOps::attentionLayer(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::ffnLayer(const FfnLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::microBatchedFfnLayer(const FfnLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::moeFfnLayer(const FfnLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::moeSharedExpert(const FfnLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

MoeGateSelectOutput DeviceOps::moeGateSelect(const FfnLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::moeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::moeFfnFp8(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::moeFfnFp8Contiguous(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::moeFfnFp8Masked(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::epMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::deepEpMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::deepEpLLMoeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

LoraLinearOutput DeviceOps::loraLinear(const LoraLinearParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

LoraLinearOutput DeviceOps::loraLinearWithAllReduce(const LoraLinearParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

MoeDispatchOutput DeviceOps::epDispatch(const MoeDispatchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

MoeCombineOutput DeviceOps::epCombine(const MoeCombineParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput DeviceOps::gatherCombineOutput(const MoeCombineOutput& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AllGatherLoraLinearOutput DeviceOps::allGatherloraLinear(const AllGatherLoraLinearParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

ReduceScatterLoraLinearOutput DeviceOps::loraLinearReduceScatter(const LoraLinearReduceScatterParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

GreedyOutput DeviceOps::sampleGreedy(const GreedyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BeamSearchOutput DeviceOps::sampleBeamSearch(const BeamSearchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::broadcast(const BroadcastParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AllReduceOutput DeviceOps::allReduce(const AllReduceParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::allGather(const AllGatherParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AllToAllOutput DeviceOps::allToAll(const AllToAllParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::reduceScatter(const ReduceScatterParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr DeviceOps::quantize(const QuantizeParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

PrepareAllReduceOutput DeviceOps::prepareAllReduce(const PrepareAllReduceParams& params) {
    return PrepareAllReduceOutput{params.buffer};
}

void DeviceOps::bufMemset(Buffer& buf, int val, DeviceStream stream) {
    if (buf.where() == MemoryType::MEMORY_CPU || buf.where() == MemoryType::MEMORY_CPU_PINNED) {
        std::memset(buf.data(), val, buf.sizeBytes());
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

void DeviceOps::noBlockCopy(const CopyParams& params) {
    copy(params);
}

void DeviceOps::noBlockCopy(const MultiCopyParams& params) {
    multiCopy(params);
}

void DeviceOps::multiMergeCopy(const MultiMergeCopyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::multiCopy(const MultiCopyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

torch::Tensor
DeviceOps::preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool user_arm_gemm_use_kai) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

torch::Tensor DeviceOps::packInt8TensorToPackedInt4(torch::Tensor weight) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

torch::Tensor
DeviceOps::preprocessWeightsForMixedGemm(torch::Tensor weight, torch::ScalarType quant_type, const std::string& arch) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

std::vector<torch::Tensor> DeviceOps::symmetricQuantizeLastAxisOfBatchedMatrix(torch::Tensor      weight,
                                                                               torch::ScalarType  quant_type,
                                                                               const std::string& arch) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

torch::Tensor DeviceOps::preprocessWeightScale(torch::Tensor weight, torch::Tensor scale) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::maskLogits(Buffer& logits, const Buffer& mask) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::perfRangePush(const std::string& name) const {}

void DeviceOps::perfRangePop() const {}

void DeviceOps::prepareCommBuffer(const PrepareCommBufferParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

bool DeviceOps::checkNAN(const Buffer& input, const std::string& name, std::function<void()> on_nan, bool force_print) {
    return false;
}

}  // namespace rtp_llm
