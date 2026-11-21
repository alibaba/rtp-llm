// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <mutex>
#include <utility>

#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlManifest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlTimerQueue.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/proto/hixl_cache_store.pb.h"

namespace rtp_llm {

// Only control metadata crosses ARPC. Published tensors remain in their
// registered pools; the reader releases a grant after its physical READ.
class HixlManifestService final: public hixl_proto::HixlCacheStoreService {
public:
    HixlManifestService(std::string                              engine_id,
                        std::string                              source_instance,
                        std::shared_ptr<MemoryUtil>              memory,
                        std::shared_ptr<RequestBlockBufferStore> store,
                        std::shared_ptr<HixlSourceGrants>        grants,
                        std::shared_ptr<HixlTimerQueue>          timers):
        engine_id_(std::move(engine_id)),
        source_instance_(std::move(source_instance)),
        memory_(std::move(memory)),
        store_(std::move(store)),
        grants_(std::move(grants)),
        timers_(std::move(timers)) {}

    void setReady(bool ready) {
        std::lock_guard<std::mutex> lock(admission_mutex_);
        ready_ = ready;
    }
    void manifest(google::protobuf::RpcController*   controller,
                  const hixl_proto::ManifestRequest* request,
                  hixl_proto::ManifestResponse*      response,
                  google::protobuf::Closure*         done) override;
    void release(google::protobuf::RpcController*  controller,
                 const hixl_proto::ReleaseRequest* request,
                 hixl_proto::ReleaseResponse*      response,
                 google::protobuf::Closure*        done) override;

private:
    std::mutex                               admission_mutex_;
    bool                                     ready_{false};
    std::string                              engine_id_;
    std::string                              source_instance_;
    std::shared_ptr<MemoryUtil>              memory_;
    std::shared_ptr<RequestBlockBufferStore> store_;
    std::shared_ptr<HixlSourceGrants>        grants_;
    std::shared_ptr<HixlTimerQueue>          timers_;
};

}  // namespace rtp_llm
