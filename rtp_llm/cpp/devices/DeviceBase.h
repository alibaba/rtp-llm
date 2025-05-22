#pragma once

#include "rtp_llm/cpp/devices/DeviceOps.h"
#include "rtp_llm/cpp/devices/DeviceData.h"
#include "rtp_llm/cpp/devices/BufferManager.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/stats/ExpertStats.h"

namespace rtp_llm {

class DeviceBase : public DeviceOps {
public:
    DeviceBase(const DeviceInitParams& params);

    virtual void init();
    std::shared_ptr<rtp_llm::CacheStore> cacheStore();

    // Init and preRun(NormalEngine::loop()) are executed in two different threads, some environments
    // needs to be reset again in a new thread(such as cudaSetDevice,
    // otherwise it will be executed in default cudaDevice 0) so we provide a preRun() to do this.
    virtual void preRun() {}
    virtual DeviceProperties getDeviceProperties() = 0;
    virtual MemoryStatus getDeviceMemoryStatus();
    DeviceStatus getDeviceStatus();
    virtual torch::Device getTorchDevice() {
        throw std::runtime_error("getTorchDevice() is not implemented");
    }

    void traceMemoryUsage();
    virtual void printDebugInfo() {};
    bool enableDevicePerf() const { return enable_device_perf_; }
    void setTraceMemory(bool trace_memory);
    void holdBufferRecycle();
    void releaseBufferRecycleHold();
    BufferPtr allocateBuffer(const BufferParams& params, const BufferHints& hints = {});
    BufferPtr allocateBufferLike(const Buffer& buffer,
                                 const AllocationType atype = AllocationType::DEVICE,
                                 const BufferHints& hints = {});
    virtual void syncAndCheck();
    virtual void syncDeviceStream(DeviceStream stream);
    virtual void syncCommunication(bool timeout = true);
    virtual void syncCommunication(ParallelMode mode, bool timeout = true);
    virtual void overlappedCommBarrier();
    virtual DeviceHookPtr createCommHook();
    virtual void overlappedComputeBarrier();
    virtual DevicePrepOutput prepareModelRun(const DevicePrepParams& params);
    virtual DeviceEventPtr createEvent();
    virtual DeviceEventPtr createTorchEvent();
    void setCacheStore(std::shared_ptr<rtp_llm::CacheStore> cache_store);

    void writeCacheStore(const WriteCacheParams& params);

    void writeHiddenStatesStore(const WriteMTPHiddenStatesParams& params);

    DeviceInitParams initParams() {
        return init_params_;
    }

    // for record moe expert stats
    virtual OverallExpertStats createMoeExpertStates(const ExpertStatsParams& params);
    virtual void cleanMoeExpertStates(const OverallExpertStats& stats);

    // for deepseek micro batching
    virtual void setMoEInsertion(const MoEInsertionParams& params);
    virtual std::unique_ptr<MoEInsertionReturns> stealMoEInsertionRet();
    virtual const std::unique_ptr<MoEInsertionReturns>& getMoEInsertionRet();
    virtual void computeInsertedMoE();

    virtual void
    updateExpertGpuLoads(const MoeConfigs& moe_conf, const OptionalExpertStats& expert_stats, BufferPtr expert_ids);

public:
    // device-independence op implementations
    CloneOutput clone(const CloneParams& params) override;
    SelectOutput select(const SelectParams& params) override;
    ConcatOutput concat(const ConcatParams& params) override;
    SplitOutput split(const SplitParams& params) override;
    AttentionLayerOutput attentionLayer(const AttentionLayerParams& params) override;
    FfnLayerOutput ffnLayer(const FfnLayerParams& params) override;
    FfnLayerOutput moeFfnLayer(const FfnLayerParams& params) override;
    FfnLayerOutput epMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output) override;
    FfnLayerOutput moeSharedExpert(const FfnLayerParams& params) override;
    LoraLinearOutput loraLinear(const LoraLinearParams& params) override;
    AllReduceOutput allReduce(const AllReduceParams& params) override;
    LossOutput loss(const LossParams& params) override;
    MaskOutput attentionMask(const MaskParams& params) override;
    BufferPtr loraLinearWithActivation(const LoraLinearWithActivationParams& params) override;
    ReduceScatterLoraLinearOutput loraLinearReduceScatter(const LoraLinearReduceScatterParams& params) override;
    AllGatherLoraLinearOutput allGatherloraLinear(const AllGatherLoraLinearParams& params) override;
    BufferPtr mhaQKVGemm(const AttentionLayerParams& params) override;
    GroupedGemmOutput groupedGemm(const GroupedGemmParams& params) override;
    MultimodalEmbeddingOutput multimodalEmbedding(const MultimodalEmbeddingParams& params) override;

    //mla
    AttentionLayerOutput mlaAttentionLayer(const AttentionLayerParams& params) override;

    void prepareCommBuffer(const PrepareCommBufferParams& params) override;
protected:
    BufferStatus queryBufferStatus();
    AllocationType getMemAllocationType(const MemoryType type);

private:
    DeviceBase(const DeviceBase&) = delete;
    DeviceBase& operator=(const DeviceBase&) = delete;
    DeviceBase(DeviceBase&&)                 = delete;
    DeviceBase& operator=(DeviceBase&&) = delete;

private:
    virtual IAllocator* getAllocator() = 0;
    virtual IAllocator* getHostAllocator() = 0;

protected:
    int device_id_;
    DeviceInitParams init_params_;
    std::shared_ptr<rtp_llm::CacheStore> cache_store_;
    bool enable_device_perf_ = false;

    std::unique_ptr<MoEInsertionParams> moe_insertion_params_;
    std::unique_ptr<MoEInsertionReturns> moe_insertion_ret_;

private:
    std::unique_ptr<BufferManager> buffer_manager_;

public:
    MlaOpsType mla_ops_type = MlaOpsType::AUTO;
};

};  // namespace rtp_llm
