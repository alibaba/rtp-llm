#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Scheduler.h"

namespace fastertransformer {

#include <omp.h>
template <typename T0, typename F>
void parallel_for(const T0& D0, const F& func) {
    int nthr = omp_get_max_threads();
#pragma omp parallel for num_threads(nthr)
    for (T0 d0 = 0; d0 < D0; ++d0) func(d0);
}

class ArmCpuDevice : public DeviceBase {
public:
    ArmCpuDevice(const DeviceInitParams& params);
    ~ArmCpuDevice();

public:
    DeviceProperties getDeviceProperties() override;
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return allocator_.get(); }

public:
    void copy(const CopyParams& params) override;
    LayernormOutput layernorm(const LayernormParams& params) override;
    BufferPtr gemm(const GemmParams& params) override;
    GroupedGemmOutput groupedGemm(const GroupedGemmParams& params) override;
    BufferPtr embeddingLookup(const EmbeddingLookupParams& params) override;
    BufferPtr activation(const ActivationParams& params) override;
    BufferPtr softmax(const SoftmaxParams& params) override;
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params) override;
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params) override;
    void sampleGreedy(const GreedyParams& params) override;
    void sampleBeamSearch(const BeamSearchParams& params) override;
    void broadcast(const BroadcastParams& params) override;
    void allReduceSum(const AllReduceParams& params);
    void printStat();

private:
    std::unique_ptr<IAllocator> allocator_;
    arm_compute::DataType getAclDataType(DataType type);
    void contextAttentionStride(const AttentionModuleParams& params);
    void decoderSelfAttentionStride(const AttentionModuleParams& params);
    void contextAttentionFallback(const AttentionModuleParams& params);
    void decoderSelfAttentionFallback(const AttentionModuleParams& params);
    void logTime(std::chrono::microseconds diff, size_t index);
    uint64_t  a_cnt_[16] = {0};
    uint64_t a_tmin_[16] = {999999999, 999999999, 999999999, 999999999, 999999999, 999999999, 999999999, 999999999,
                            999999999, 999999999, 999999999, 999999999, 999999999, 999999999, 999999999, 999999999};
    uint64_t a_tmax_[16] = {0};
    uint64_t a_tave_[16] = {0};
};

} // namespace fastertransformer

