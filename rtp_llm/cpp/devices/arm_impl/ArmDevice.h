#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Scheduler.h"
#include "gemm_opt/ArmGemmKernel.h"
#include "rtp_llm/cpp/devices/utils/Timer.h"

namespace rtp_llm {

class ArmCpuDevice : public DeviceBase {
public:
    ArmCpuDevice(const DeviceInitParams& params);
    ~ArmCpuDevice();

public:
    DeviceProperties getDeviceProperties() override;
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return allocator_.get(); }
    torch::Device getTorchDevice() override { return torch::Device(torch::kCPU);};

public:
    void copy(const CopyParams& params) override;
    LayernormOutput layernorm(const LayernormParams& params) override;
    LayernormOutput layernormWithStride(const LayernormWithStrideParams& params) override;
    BufferPtr gemm(const GemmParams& params) override;
    BufferPtr gemm_acl(const GemmParams& params);
    BufferPtr gemm_opt(const GemmParams& params);
    BufferPtr gemm_kai_bf16(const GemmParams& params);
    BufferPtr gemm_kai_a8w4(const GemmParams& params);
    GroupedGemmOutput groupedGemm(const GroupedGemmParams& params) override;
    BufferPtr embeddingLookup(const EmbeddingLookupParams& params) override;
    BufferPtr activation(const ActivationParams& params) override;
    BufferPtr softmax(const SoftmaxParams& params) override;
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params) override;
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params) override;
    GreedyOutput sampleGreedy(const GreedyParams& params) override;
    void sampleBeamSearch(const BeamSearchParams& params) override;
    BufferPtr mlaQKVGemm(const AttentionLayerParams& params) override;
    void mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) override;
    void prepareMoEGate(const FfnLayerParams& params, BufferPtr gate);
    void mlaAbsorbAttention(const MlaAttentionModuleParams& params) override;
    void mlaContextAttention(const MlaAttentionModuleParams& params) override;
    FfnLayerOutput moeFfnLayer(const FfnLayerParams& params) override;
    void broadcast(const BroadcastParams& params) override;
    void allReduceSum(const AllReduceParams& params);
    DevicePrepOutput prepareModelRun(const DevicePrepParams& params) override;
    void printStat();
    MemoryStatus getDeviceMemoryStatus() override;
    SliceOutput slice(const SliceParams& params) override;
#ifdef GEMM_DEBUG
    static void print_time();
#endif
    static torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight);

    static torch::Tensor packInt8TensorToPackedInt4(torch::Tensor weight);
    static torch::Tensor preprocessWeightsForMixedGemm(torch::Tensor row_major_quantized_weight, torch::ScalarType quant_type, const std::string &arch);
    static torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale, const std::string& key);

private:
    std::unique_ptr<IAllocator> allocator_;
    arm_compute::DataType getAclDataType(DataType type);
    void runOneBatch(const AttentionModuleParams& params, size_t past_seq, int batch, size_t seq_len, size_t step);
    void runOneBatchStride(const AttentionModuleParams& params, size_t past_seq, int batch, size_t seq_len, size_t step);
    void runOneBatchFlash(const AttentionModuleParams& params, size_t past_seq, int batch, size_t seq_len, size_t step);
    void runOneBatchFlashDecoding(const AttentionModuleParams& params, size_t past_seq, int batch, size_t seq_len, size_t step);
    std::unordered_map<int, std::tuple<int, float *, float *>> ropeCosSin;
    template<typename T>
    void halfRopeQK(void *qkv, int batch, int seq_len, int num_heads, int kv_num_heads, int head_size, size_t step, const RopeConfig* rope_config);
    void biasAddRopeWriteKVCache(const AttentionModuleParams& params, size_t past_seq, int batch, size_t seq_len, size_t step);
    void logTime(std::chrono::microseconds diff, size_t index);
    uint64_t  a_cnt_[16] = {0};
    uint64_t a_tmin_[16] = {999999999, 999999999, 999999999, 999999999, 999999999, 999999999, 999999999, 999999999,
                            999999999, 999999999, 999999999, 999999999, 999999999, 999999999, 999999999, 999999999};
    uint64_t a_tmax_[16] = {0};
    uint64_t a_tave_[16] = {0};
    GemmKernel gemm_kernel_;

    FfnLayerOutput moe_ffn_a8w4(const BufferPtr expert_indices, const BufferPtr expert_weights, const BufferPtr output, const FfnLayerParams& params);

    BufferPtr (ArmCpuDevice::*gemmFunc)(const GemmParams& params);
    bool isKAIenabled;
    bool isFAenabled;

#ifdef GEMM_DEBUG
    static TimerRecorder timer_recorder_;
#endif
};

extern ConstBufferPtr (*armPrepareWeightFunc)(ConstBufferPtr input, bool isTranspose, bool isForceF32Out);
extern float32x4_t vexpq_f32(float32x4_t x);
extern float vMax(int n, const float* a);
} // namespace rtp_llm
