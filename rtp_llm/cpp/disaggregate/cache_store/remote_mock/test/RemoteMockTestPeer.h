#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
namespace rtp_llm {
// A controllable peer for malformed replies and capacity ordering. The separate
// loopback target uses the real registered-Host CacheStore callback service.
class RemoteMockTestPeer: public KvCacheStoreService {
public:
    std::string                payload = std::string("A\0B\xff"
                                                     "0123456789xy!",
                                      16);
    std::atomic<int>           calls{0};
    std::atomic<int>           corruption{0};
    std::atomic<bool>          hold{false};
    std::mutex                 mutex;
    std::condition_variable    condition;
    google::protobuf::Closure* pending = nullptr;
    bool                       waitPull() {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, std::chrono::seconds(2), [&] { return pending != nullptr; });
    }
    void release() {
        google::protobuf::Closure* done;
        {
            std::lock_guard<std::mutex> lock(mutex);
            done    = pending;
            pending = nullptr;
        }
        if (done)
            done->Run();
    }

    void load(google::protobuf::RpcController*,
              const CacheLoadRequest*,
              CacheLoadResponse*         r,
              google::protobuf::Closure* d) override {
        r->set_error_code(EC_FAILED_INVALID_REQ);
        d->Run();
    }
    void transfer(google::protobuf::RpcController*,
                  const CacheTransferRequest*,
                  CacheTransferResponse*     r,
                  google::protobuf::Closure* d) override {
        r->set_error_code(EC_FAILED_INVALID_REQ);
        d->Run();
    }
    void blockRead(google::protobuf::RpcController*,
                   const BlockReadRequest*    q,
                   BlockReadResponse*         r,
                   google::protobuf::Closure* d) override {
        ++calls;
        r->set_error_code(EC_SUCCESS);
        for (const auto& b : q->blocks()) {
            auto out = r->add_blocks();
            out->set_key(b.key());
            out->set_len(b.len());
            out->set_content(payload.data(), b.len());
        }
        if (corruption == 1)
            r->mutable_blocks(0)->set_key("wrong-key");
        if (corruption == 2)
            r->mutable_blocks(0)->set_content("truncated");
        if (corruption == 3)
            r->mutable_blocks(0)->set_len(15);
        if (corruption == 4)
            r->mutable_blocks(0)->set_content(std::string(16, '!'));
        if (corruption == 5)
            r->clear_blocks();
        if (corruption == 6)
            r->GetReflection()->MutableUnknownFields(r)->AddLengthDelimited(99, std::string(7000, 'x'));
        if (hold) {
            std::lock_guard<std::mutex> lock(mutex);
            pending = d;
            condition.notify_all();
            return;
        }
        d->Run();
    }
};
}  // namespace rtp_llm
