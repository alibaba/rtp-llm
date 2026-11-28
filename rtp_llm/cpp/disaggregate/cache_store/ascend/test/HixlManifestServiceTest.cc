// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlManifestService.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlProtocol.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

class RegisteredMemory: public MemoryUtil {
public:
    bool regUserMr(void*, uint64_t, bool, uint64_t) override { return true; }
    bool deregUserMr(void*, bool) override { return true; }
    bool isMemoryMr(void*, uint64_t, bool, bool) override { return true; }
    bool findMemoryMr(void* out, void* ptr, uint64_t size, bool, bool) override {
        *static_cast<HixlMemoryRegion*>(out) = {reinterpret_cast<uintptr_t>(ptr), size, 1};
        return true;
    }
    bool isRdmaMode() override { return true; }
};

class Callback: public google::protobuf::Closure {
public:
    explicit Callback(std::function<void()> callback): callback_(std::move(callback)) {}
    void Run() override { callback_(); }

private:
    std::function<void()> callback_;
};

TEST(HixlManifestServiceTest, SynchronousCompletionCanReenterManifestAndRelease) {
    auto memory = std::make_shared<RegisteredMemory>();
    auto store = std::make_shared<RequestBlockBufferStore>(memory);
    auto grants = std::make_shared<HixlSourceGrants>();
    auto timers = std::make_shared<HixlTimerQueue>();
    HixlManifestService service("source-engine", "source-instance", memory, store, grants, timers);
    service.setReady(true);
    auto prepare = [&](const std::string& id, uint64_t transfer) {
        auto buffer = std::make_shared<RequestBlockBuffer>(id);
        buffer->addBlock("block", std::make_shared<int>(7), sizeof(int), true, true);
        EXPECT_TRUE(store->setRequestBlockBuffer(buffer));
        hixl_proto::ManifestRequest req;
        req.set_request_id(id);
        req.set_reader_id("reader");
        req.set_reader_engine_id("reader-engine");
        req.set_transfer_id(transfer);
        req.set_timeout_ms(30000);
        auto* block = req.add_blocks();
        block->set_key("block");
        block->set_bytes(sizeof(int));
        return req;
    };
    auto outer = prepare("outer", 1);
    auto inner = prepare("inner", 2);
    hixl_proto::ManifestResponse outer_response, inner_response;
    std::vector<uint64_t> completed;
    auto release = [&](uint64_t id) {
        hixl_proto::ReleaseRequest req;
        hixl_proto::ReleaseResponse res;
        req.set_reader_id("reader");
        req.set_reader_engine_id("reader-engine");
        req.set_transfer_id(id);
        Callback done([&] {
            EXPECT_TRUE(res.released());
            completed.push_back(id);
        });
        service.release(nullptr, &req, &res, &done);
    };
    Callback inner_done([&] {
        EXPECT_EQ(inner_response.status(), hixl_proto::OK);
        release(2);
    });
    Callback outer_done([&] {
        EXPECT_EQ(outer_response.status(), hixl_proto::OK);
        service.manifest(nullptr, &inner, &inner_response, &inner_done);
        release(1);
    });
    service.manifest(nullptr, &outer, &outer_response, &outer_done);
    EXPECT_EQ(completed, (std::vector<uint64_t>{2, 1}));
    EXPECT_EQ(grants->size(), 0);
    timers->stop();
    store->stop();
}

}  // namespace
}  // namespace rtp_llm
