// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rtp_llm/cpp/disaggregate/cache_store/Messager.h"

namespace rtp_llm {
class HixlMemoryUtil;
class HixlSourceGrants;
class HixlManifestService;
class TcpServer;
class HixlClientState;

class HixlMessager final: public Messager {
public:
    HixlMessager(const CacheStoreInitParams&                     params,
                 std::shared_ptr<HixlMemoryUtil>                 memory,
                 const std::shared_ptr<RequestBlockBufferStore>& store,
                 const kmonitor::MetricsReporterPtr&             metrics);
    ~HixlMessager() override;
    bool init(MessagerInitParams params) override;
    void finishRegistration() override;
    void load(const std::shared_ptr<LoadRequest>&                          request,
              const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) override;
    void transfer(const std::shared_ptr<TransferRequest>& request) override;

protected:
    bool generateBlockInfo(::BlockBufferInfo*, const std::shared_ptr<BlockBuffer>&, uint32_t, uint32_t) override {
        return false;
    }

private:
    std::shared_ptr<HixlSourceGrants>    grants_;
    std::shared_ptr<HixlManifestService> service_;
    std::shared_ptr<TcpServer>           server_;
    std::shared_ptr<HixlClientState>     client_;
};
}  // namespace rtp_llm
