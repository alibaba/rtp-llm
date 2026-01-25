#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/cache/connector/p2p/AsymmetricTpUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/PrefillWorkerLoadContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/StoreWaitContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferServer.h"
#include "rtp_llm/cpp/model_rpc/TpBroadcastManager.h"
#include "autil/LoopThread.h"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

class P2PConnectorWorker {
public:
    P2PConnectorWorker(const KVCacheConfig&                        cache_config,
                       const CacheStoreConfig&                     cache_store_config,
                       const ParallelismConfig&                    parallelism_config,
                       const PDSepConfig&                          pd_sep_config,
                       const ModelConfig&                          model_config,
                       uint32_t                                    layer_all_num,
                       const std::shared_ptr<LayerBlockConvertor>& layer_block_convertor,
                       const kmonitor::MetricsReporterPtr&         metrics_reporter);
    ~P2PConnectorWorker();

public:
    bool init(int64_t store_wait_timeout_ms = 10 * 1000);

public:
    bool writeByLayer(int layer_id, const KVCacheResourcePtr& resource, int64_t request_id, DeviceEventPtr event);

    ErrorInfo handleRead(int64_t                                              request_id,
                         const std::string&                                   unique_key,
                         int64_t                                              deadline_ms,
                         const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers);

    ErrorInfo read(int64_t                                               request_id,
                   const std::string&                                    unique_key,
                   int64_t                                               deadline_ms,
                   const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers);

    // 取消 read 请求，根据 unique_key 找到对应的 task 并设置为 cancelled
    bool cancelRead(const std::string& unique_key);

    // 取消 handleRead 请求，根据 unique_key 找到对应的 load_context 并设置为 cancelled
    bool cancelHandleRead(const std::string& unique_key);

public:
    // ================== 内部状态访问 ==================

    std::shared_ptr<ComputedLayerCacheBufferStore> getComputedBuffersStore() const {
        return computed_buffers_;
    }

    std::shared_ptr<PrefillWorkerLoadContextStore> getLoadContexts() const {
        return load_contexts_;
    }

    void setStoreWaitTimeoutMs(int64_t store_wait_timeout_ms) {
        store_wait_timeout_ms_ = store_wait_timeout_ms;
    }

private:
    // Store wait 线程处理
    void loopCheckProc();

private:
    // 配置
    const KVCacheConfig&                        cache_config_;
    const CacheStoreConfig&                     cache_store_config_;
    const ParallelismConfig&                    parallelism_config_;
    const PDSepConfig&                          pd_sep_config_;
    const ModelConfig&                          model_config_;
    const uint32_t                              layer_all_num_;
    const std::shared_ptr<LayerBlockConvertor>& layer_block_convertor_;
    kmonitor::MetricsReporterPtr                metrics_reporter_;

    // Prefill 端组件（发送数据）
    std::shared_ptr<TransferClient>                transfer_client_;
    std::shared_ptr<AsymmetricTpUtil>              asymmetric_tp_util_;
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;
    std::shared_ptr<PrefillWorkerLoadContextStore> load_contexts_;
    int64_t                                        store_wait_timeout_ms_ = 10 * 1000;

    // Store wait 检查器（Prefill 端）
    std::shared_ptr<StoreWaitContextChecker> store_wait_context_checker_;

    // 清理线程（定期清理过期缓存和上报状态）
    autil::LoopThreadPtr cleanup_thread_;

    // Decode 端组件（接收数据）
    std::shared_ptr<TransferServer>    transfer_server_;
    std::shared_ptr<TransferTaskStore> transfer_task_store_;
};

}  // namespace rtp_llm
