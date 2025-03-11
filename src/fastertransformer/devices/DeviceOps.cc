#include "src/fastertransformer/devices/DeviceOps.h"

namespace fastertransformer {

DeviceOps::DeviceOps() {}

DeviceOps::~DeviceOps() {}

void DeviceOps::copy(const CopyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

CloneOutput DeviceOps::clone(const CloneParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

TransposeOutput DeviceOps::transpose(const TransposeParams& params) {
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

LayernormOutput DeviceOps::layernorm(const LayernormParams& params) {
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

AttentionModuleOutput DeviceOps::contextAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput DeviceOps::decoderSelfAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionLayerOutput DeviceOps::mlaAttentionLayer(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput DeviceOps::mlaContextAttention(const MlaAttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput DeviceOps::mlaDecoderSelfAttention(const MlaDecoderAttentionParams& params) {
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

LoraLinearOutput DeviceOps::loraLinear(const LoraLinearParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

LoraLinearOutput DeviceOps::loraLinearWithAllReduce(const LoraLinearParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

GreedyOutput DeviceOps::sampleGreedy(const GreedyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::sampleBeamSearch(const BeamSearchParams& params) {
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

BufferPtr DeviceOps::quantize(const QuantizeParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

PrepareAllReduceOutput DeviceOps::prepareAllReduce(const PrepareAllReduceParams& params) {
    return PrepareAllReduceOutput{params.buffer};
};

void DeviceOps::bufMemset(Buffer& buf, int val) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::noBlockCopy(const CopyParams& params) {
    copy(params);
}

torch::Tensor DeviceOps::preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

torch::Tensor DeviceOps::packInt8TensorToPackedInt4(torch::Tensor weight) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

torch::Tensor DeviceOps::preprocessWeightsForMixedGemm(torch::Tensor weight, torch::ScalarType quant_type, const std::string &arch) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

std::vector<torch::Tensor> DeviceOps::symmetricQuantizeLastAxisOfBatchedMatrix(torch::Tensor weight, torch::ScalarType quant_type, const std::string &arch) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

torch::Tensor DeviceOps::preprocessWeightScale(torch::Tensor weight, torch::Tensor scale) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceOps::perfRangePush(const std::string& name) const {}

void DeviceOps::perfRangePop() const {}

} // namespace fastertransformer
