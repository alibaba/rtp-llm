#pragma once

#include "rtp_llm/cpp/devices/DeviceOps.h"
#include "rtp_llm/cpp/devices/DeviceData.h"
#include "rtp_llm/cpp/devices/BufferManager.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"
#include "rtp_llm/cpp/devices/GraphBase.h"
#include "rtp_llm/cpp/devices/NativeGraphRunnerBase.h"

namespace rtp_llm {

#define CACHED_BUF(dtype, atype, ...)                                                                                  \
    [&]() {                                                                                                            \
        static std::deque<rtp_llm::BufferPtr> buffers;                                                                 \
        rtp_llm::BufferPtr                    buffer;                                                                  \
        std::vector<size_t>                   shape = __VA_ARGS__;                                                     \
        auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());                            \
        if (!buffers.empty()) {                                                                                        \
            buffer = std::move(buffers.back());                                                                        \
            buffers.pop_back();                                                                                        \
            if (buffer->size() < size) {                                                                               \
                buffer = nullptr;                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        if (!buffer) {                                                                                                 \
            buffer = device_->allocateBuffer({rtp_llm::DataType::dtype, shape, atype}, {});                            \
        }                                                                                                              \
        return std::make_shared<rtp_llm::Buffer>(                                                                      \
            buffer->where(), buffer->type(), shape, buffer->data(), [buffer](rtp_llm::Buffer* buf) {                   \
                buffers.emplace_back(std::move(buffer));                                                               \
            });                                                                                                        \
    }()

#define CACHED_HOST_BUF(dtype, ...) CACHED_BUF(dtype, rtp_llm::AllocationType::HOST, __VA_ARGS__)

#define CACHED_DEVICE_BUF(dtype, ...) CACHED_BUF(dtype, rtp_llm::AllocationType::DEVICE, __VA_ARGS__)

#define SAFE_CACHED_HOST_BUF(dtype, ...)                                                                               \
    [&]() {                                                                                                            \
        static std::deque<rtp_llm::BufferPtr> buffers;                                                                 \
        static std::mutex                     mu;                                                                      \
        rtp_llm::BufferPtr                    buffer;                                                                  \
        std::vector<size_t>                   shape = __VA_ARGS__;                                                     \
        auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());                            \
        {                                                                                                              \
            std::unique_lock lock(mu);                                                                                 \
            if (!buffers.empty()) {                                                                                    \
                buffer = std::move(buffers.back());                                                                    \
                buffers.pop_back();                                                                                    \
                if (buffer->size() < size) {                                                                           \
                    buffer = nullptr;                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        if (!buffer) {                                                                                                 \
            auto atype = rtp_llm::AllocationType::HOST;                                                                \
            buffer     = device_->allocateBuffer({rtp_llm::DataType::dtype, shape, atype}, {});                        \
        }                                                                                                              \
        return std::make_shared<rtp_llm::Buffer>(                                                                      \
            buffer->where(), buffer->type(), shape, buffer->data(), [buffer](rtp_llm::Buffer* buf) {                   \
                std::unique_lock lock(mu);                                                                             \
                buffers.emplace_back(std::move(buffer));                                                               \
            });                                                                                                        \
    }()

using NativeGraphRunner = NativeGraphRunnerBase<GptModelInputs, GptModelOutputs>;

class DeviceBase: public DeviceOps {
public:
    DeviceBase(const DeviceInitParams& params);

    virtual void                         init();
    std::shared_ptr<rtp_llm::CacheStore> cacheStore();

    // Init and preRun(NormalEngine::loop()) are executed in two different threads, some environments
    // needs to be reset again in a new thread(such as cudaSetDevice,
    // otherwise it will be executed in default cudaDevice 0) so we provide a preRun() to do this.
    virtual void             preRun() {}
    virtual DeviceProperties getDeviceProperties() = 0;
    virtual MemoryStatus     getDeviceMemoryStatus();
    DeviceStatus             getDeviceStatus();
    virtual torch::Device    getTorchDevice() {
        throw std::runtime_error("getTorchDevice() is not implemented");
    }

    void         traceMemoryUsage();
    virtual void printDebugInfo() {};
    bool         enableDevicePerf() const {
        return enable_device_perf_;
    }
    void                     setTraceMemory(bool trace_memory);
    void                     holdBufferRecycle();
    void                     releaseBufferRecycleHold();
    BufferPtr                allocateBuffer(const BufferParams& params, const BufferHints& hints = {});
    BufferPtr                allocateBufferLike(const Buffer&        buffer,
                                                const AllocationType atype = AllocationType::DEVICE,
                                                const BufferHints&   hints = {});
    virtual void             checkError();
    virtual void             syncAndCheck();
    virtual void             syncDeviceStream(DeviceStream stream);
    virtual void             syncCommunication(bool timeout = true);
    virtual void             syncCommunication(ParallelMode mode, bool timeout = true);
    virtual void             overlappedCommBarrier();
    virtual DeviceHookPtr    createCommHook();
    virtual void             overlappedComputeBarrier();
    virtual DevicePrepOutput prepareModelRun(const DevicePrepParams& params);
    virtual DeviceEventPtr   createEvent();
    virtual DeviceEventPtr   createTorchEvent();
    virtual void             updateCurrentTorchStream();
    virtual GraphBase*       getDeviceGraphRunner(const DeviceInitParams& params,
                                                  py::object              py_instance,
                                                  int                     kv_cache_block_offset,
                                                  bool                    is_prefill_cuda_graph_mode = false) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    void setCacheStore(std::shared_ptr<rtp_llm::CacheStore> cache_store);

    void writeCacheStore(const WriteCacheParams& params);

    void writeCacheStore(const CacheStoreInputs& cache_store_inputs, const KvCacheInfo& kv_cache, bool mla_kvcache);

    DeviceInitParams initParams() {
        return init_params_;
    }

    DeviceInitParams& initParamsRef() {
        return init_params_;
    }

    // for record moe expert stats
    virtual OverallExpertStats createMoeExpertStates(const ExpertStatsParams& params);
    virtual void               cleanMoeExpertStates(const OverallExpertStats& stats);

    // for deepseek micro batching
    virtual void                                        setMoEInsertion(const MoEInsertionParams& params);
    virtual std::unique_ptr<MoEInsertionReturns>        stealMoEInsertionRet();
    virtual const std::unique_ptr<MoEInsertionReturns>& getMoEInsertionRet();
    virtual void                                        computeInsertedMoE();

    // for cuda profiler
    virtual void profileStart();
    virtual void profileStop();

    virtual void
    updateExpertGpuLoads(const MoeConfigs& moe_conf, const OptionalExpertStats& expert_stats, BufferPtr expert_ids);

    virtual std::shared_ptr<NativeGraphRunner> getNativeGraphRunner() {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    void nativeGraphBeginCapture() {
        native_graph_capturing_ = true;
    }
    void nativeGraphEndCapture() {
        native_graph_capturing_ = false;
    }
    bool nativeGraphCapturing() {
        return native_graph_capturing_;
    }

    virtual void getRopeCacheOnce(const RopeConfig& rope_config, int max_position_embeddings);
    bool         useRopeCache() const {
        return use_rope_cache_;
    }
    torch::Tensor ropeCache() const {
        return rope_cache_;
    }

public:
    // device-independence op implementations
    void         batchCopy(const BatchCopyParams& params) override;
    CloneOutput  clone(const CloneParams& params) override;
    SelectOutput select(const SelectParams& params) override;
    ConcatOutput concat(const ConcatParams& params) override;
    SplitOutput  split(const SplitParams& params) override;
    // attention layer
    AttentionLayerOutput attentionLayer(const AttentionLayerParams& params) override;
    BufferPtr            attentionQKVGemm(const AttentionLayerParams& params) override;
    BufferPtr            attentionAttn(const AttentionLayerParams& params) override;
    BufferPtr            attentionOutGemm(const AttentionLayerParams& params) override;
    // ffn layer
    FfnLayerOutput   ffnLayer(const FfnLayerParams& params) override;
    FfnLayerOutput   moeFfnLayer(const FfnLayerParams& params) override;
    FfnLayerOutput   epMoeFfnLayer(const FfnLayerParams& params, const MoeGateSelectOutput& gate_output) override;
    FfnLayerOutput   moeSharedExpert(const FfnLayerParams& params) override;
    LoraLinearOutput loraLinear(const LoraLinearParams& params) override;
    AllReduceOutput  allReduce(const AllReduceParams& params) override;
    LossOutput       loss(const LossParams& params) override;
    MaskOutput       attentionMask(const MaskParams& params) override;
    BufferPtr        loraLinearWithActivation(const LoraLinearWithActivationParams& params) override;
    ReduceScatterLoraLinearOutput loraLinearReduceScatter(const LoraLinearReduceScatterParams& params) override;
    AllGatherLoraLinearOutput     allGatherloraLinear(const AllGatherLoraLinearParams& params) override;
    BufferPtr                     mhaQKVGemm(const AttentionLayerParams& params) override;
    GroupedGemmOutput             groupedGemm(const GroupedGemmParams& params) override;
    MultimodalEmbeddingOutput     multimodalEmbedding(const MultimodalEmbeddingParams& params) override;
    BufferPtr                     inputEmbedding(const InputEmbeddingParams& params) override;
    // mla
    AttentionLayerOutput mlaAttentionLayer(const AttentionLayerParams& params) override;

    void prepareCommBuffer(const PrepareCommBufferParams& params) override;
    void chainSpeculativeSampling(const SpeculativeSamplingParams& params) override;

protected:
    BufferStatus   queryBufferStatus();
    AllocationType getMemAllocationType(const MemoryType type);

private:
    DeviceBase(const DeviceBase&)            = delete;
    DeviceBase& operator=(const DeviceBase&) = delete;
    DeviceBase(DeviceBase&&)                 = delete;
    DeviceBase& operator=(DeviceBase&&)      = delete;

private:
    virtual IAllocator* getAllocator()     = 0;
    virtual IAllocator* getHostAllocator() = 0;

protected:
    int                                  device_id_;
    DeviceInitParams                     init_params_;
    std::shared_ptr<rtp_llm::CacheStore> cache_store_;
    bool                                 enable_device_perf_ = false;

    std::unique_ptr<MoEInsertionParams>  moe_insertion_params_;
    std::unique_ptr<MoEInsertionReturns> moe_insertion_ret_;

    std::once_flag rope_cache_flag_;
    bool           use_rope_cache_ = false;
    torch::Tensor  rope_cache_;

protected:
    std::unique_ptr<BufferManager> buffer_manager_;
    bool                           native_graph_capturing_ = false;

public:
    MlaOpsType mla_ops_type = MlaOpsType::AUTO;
};

};  // namespace rtp_llm
