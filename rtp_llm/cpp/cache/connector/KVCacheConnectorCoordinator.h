#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "autil/LoopThread.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/IKVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/KVCacheHandshake.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class KVCacheAllocator;
class KVCacheMemoryConnector;
class RemoteConnector;
class P2PConnector;
class KVCacheConnectorReadWriteContext;

class KVCacheConnectorCoordinator: public IKVCacheConnectorCoordinator {
public:
    KVCacheConnectorCoordinator(const CacheConfig&                       cache_config,
                                const KVCacheConfig&                     kv_cache_config,
                                const RuntimeConfig&                     runtime_config,
                                const ParallelismConfig&                 parallelism_config,
                                const SpeculativeExecutionConfig&        sp_config,
                                const std::shared_ptr<KVCacheAllocator>& allocator,
                                const kmonitor::MetricsReporterPtr&      metrics_reporter   = nullptr,
                                const PDSepConfig&                       pd_sep_config      = PDSepConfig{},
                                const CacheStoreConfig&                  cache_store_config = CacheStoreConfig{});
    virtual ~KVCacheConnectorCoordinator();

public:
    bool init();

    bool hasActiveConnectors() const override;
    bool hasP2PConnector() const override;

    // virtual for test
    virtual std::shared_ptr<AsyncContext>
    asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) override;
    virtual std::shared_ptr<AsyncContext>
    asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) override;
    virtual std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) override;

    virtual bool              executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response);
    std::vector<CacheKeyType> memoryCacheKeys() const;

    uint32_t convertToGlobalLayerId(int model_id, int layer_id) const override {
        return allocator_->convertToGlobalLayerId(model_id, layer_id);
    }

    /// Prefill-side StartLoad path; P2P connector wiring fills this in when enabled.
    virtual void handleRead(const P2PConnectorStartLoadRequestPB& request,
                            P2PConnectorStartLoadResponsePB&      response,
                            std::function<bool()>                 is_cancelled = nullptr);

    // ----- M04 PR-3: PD pair startup handshake (REQ-D1 / REQ-D2) ------
    //
    // ``localHandshakeInfo()`` is computed once at init() time from the
    // local CacheConfig and cached.  Under legacy path
    // (super_block_layout.enabled == false) the result is all-zero — legacy
    // peers hand-shake trivially.
    //
    // ``validatePeerHandshake(peer)`` MUST be called by the per-connector
    // peer-pairing path once the peer's HandshakeInfo arrives over the
    // wire.  On any mismatch it raises RTP_LLM_FAIL with a diagnostic
    // describing the offending field (mixed-mode, hash drift, salt drift).
    // PR-3 wires the validator helper; the actual cross-the-wire exchange
    // lives in the per-connector init path and is added alongside the
    // peer-list-acquisition site (one site per connector type — memory,
    // remote, p2p).  Until the per-connector wiring lands the validator
    // is invoked by unit tests only; default-off behaviour is preserved.
    const HandshakeInfo& localHandshakeInfo() const {
        return local_handshake_info_;
    }

    // F01-PR2-followup: returns true if the pair may proceed (legacy↔legacy
    // OR same-version unified pair).  On mismatch, logs WARN, increments
    // the process-wide ``pd.cache.salt_mismatch_skipped`` counter and
    // returns false — callers MUST treat false as "do NOT publish this
    // peer's cache_keys into our reuse lookup" (legacy-only fallback at
    // worst, refusal at best).  The function deliberately does NOT raise
    // RTP_LLM_FAIL: a mixed-mode peer is a misconfiguration, not a fatal
    // crash; the counter + WARN + cache_key salt divergence are the
    // user-visible signal (Risk 9.6 silent reuse-miss path).
    bool validatePeerHandshake(const HandshakeInfo& peer_info) const;

    // Wire-side overload: construct the peer HandshakeInfo from proto fields
    // 103/104/105 (``cache_store_service.proto`` CacheLoadRequest /
    // CacheTransferRequest) and run ``validatePeerHandshake``.  Peer's
    // ``pool_descriptor_hash`` is NOT yet on the wire (deferred to the next
    // PR adding field 106), so this overload passes 0 — legacy↔legacy and
    // matching-bitmap cases still pass, hash-only drift falls back to the
    // cache_key XOR safety net (Risk 9.6).
    //
    // FIX-B HIGH-5: ``peer_salt_magic`` is the proto-103 ``salt_magic``
    // sentinel ({0,100}), NOT the {0,1} HandshakeInfo::protocol_magic.
    // The mapping into HandshakeInfo::protocol_magic uses
    // ``hash_salt_version > 0`` as the load-bearing gate so a legacy peer
    // (no field 103, version=0) is unambiguously accepted instead of
    // mis-routed into the unified↔unified REFUSE branch.
    bool acceptPeerHandshakeFields(uint32_t peer_salt_magic,
                                   uint32_t peer_hash_salt_version,
                                   uint32_t peer_hash_salt_nonzero_bitmap) const;

private:
    std::shared_ptr<KVCacheMemoryConnector> initMemoryConnector();
    std::shared_ptr<RemoteConnector>        initRemoteConnector();
    bool                                    initP2PConnectorInternal();
    // Returns CP size when page-level RR sharding is active; 1 otherwise.
    int  cpSize() const;
    void initUpdateThread();
    void updateOnce();
    void processReadContexts();
    void processWriteContexts();
    void asyncReadAfterMatch(std::shared_ptr<FusedAsyncReadContext> fused_read_context);

    bool isPdInvertMode() const;

private:
    const CacheConfig                 cache_config_;
    const KVCacheConfig               kv_cache_config_;
    const RuntimeConfig               runtime_config_;
    const ParallelismConfig           parallelism_config_;
    const SpeculativeExecutionConfig  sp_config_;
    std::shared_ptr<KVCacheAllocator> allocator_;
    kmonitor::MetricsReporterPtr      metrics_reporter_;
    PDSepConfig                       pd_sep_config_;
    CacheStoreConfig                  cache_store_config_;

    std::vector<std::shared_ptr<KVCacheConnector>>    connectors_;
    std::shared_ptr<KVCacheMemoryConnector>           memory_connector_;
    std::shared_ptr<RemoteConnector>                  remote_connector_;
    std::shared_ptr<P2PConnector>                     p2p_connector_;
    mutable std::mutex                                update_mutex_;
    std::list<std::shared_ptr<FusedAsyncReadContext>> fused_async_read_context_list_;
    std::list<std::shared_ptr<FusedAsyncContext>>     fused_async_write_context_list_;
    autil::LoopThreadPtr                              update_thread_;
    const int                                         update_interval_ms_{1};
    std::atomic<bool>                                 stop_{false};

    // M04 PR-3: cached at init() from cache_config_; consulted on every
    // peer pairing.
    HandshakeInfo                                     local_handshake_info_{};
};

}  // namespace rtp_llm
