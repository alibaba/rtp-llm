// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlManifestService.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlProtocol.h"

#include <utility>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace rtp_llm {
namespace {

class ManifestContext {
public:
    ManifestContext(const hixl_proto::ManifestRequest& request,
                    std::string                        engine,
                    std::shared_ptr<MemoryUtil>        memory,
                    std::shared_ptr<HixlSourceGrants>  grants,
                    hixl_proto::ManifestResponse*      response,
                    google::protobuf::Closure*         done):
        reader_(request.reader_id()),
        reader_engine_(request.reader_engine_id()),
        transfer_(request.transfer_id()),
        memory_(std::move(memory)),
        grants_(std::move(grants)),
        response_(response),
        done_(done) {
        response_->set_engine_id(std::move(engine));
        for (const auto& block : request.blocks()) {
            pending_.emplace(block.key(), block.bytes());
        }
    }

    ~ManifestContext() {
        // Store shutdown can discard a watch without notifying it.
        fail(hixl_proto::SOURCE_UNAVAILABLE);
    }

    void fail(hixl_proto::Status status) {
        google::protobuf::Closure* done;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!done_) {
                return;
            }
            done = completeLocked(status);
        }
        done->Run();
    }

    void onBlocks(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        if (!ok) {
            fail(hixl_proto::SOURCE_UNAVAILABLE);
            return;
        }
        google::protobuf::Closure* done;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!done_) {
                return;
            }
            auto status = collectLocked(blocks);
            if (status == hixl_proto::OK && !pending_.empty()) {
                return;
            }
            if (status == hixl_proto::OK && !grants_->acquireReserved(reader_, transfer_, ready_)) {
                status = hixl_proto::SOURCE_UNAVAILABLE;
            }
            done = completeLocked(status);
        }
        done->Run();
    }

private:
    hixl_proto::Status collectLocked(const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        for (const auto& block : blocks) {
            auto it = pending_.find(block->key);
            if (it == pending_.end()) {
                continue;
            }
            if (it->second != block->len) {
                return hixl_proto::INVALID_REQUEST;
            }
            HixlMemoryRegion region;
            if (!memory_->findMemoryMr(&region, block->addr.get(), block->len, block->gpu_mem, block->adopted)) {
                return hixl_proto::UNREGISTERED_MEMORY;
            }
            auto* descriptor = response_->add_blocks();
            descriptor->set_key(block->key);
            descriptor->set_bytes(block->len);
            descriptor->set_address(reinterpret_cast<uintptr_t>(block->addr.get()));
            descriptor->set_region_id(region.id);
            descriptor->set_region_address(region.address);
            descriptor->set_region_bytes(region.bytes);
            ready_.push_back(block);
            pending_.erase(it);
        }
        return hixl_proto::OK;
    }

    google::protobuf::Closure* completeLocked(hixl_proto::Status status) {
        response_->set_status(status);
        if (status != hixl_proto::OK) {
            grants_->retire(reader_, reader_engine_, transfer_);
            response_->clear_blocks();
        }
        pending_.clear();
        ready_.clear();
        return std::exchange(done_, nullptr);
    }

    std::mutex                                mutex_;
    const std::string                         reader_;
    const std::string                         reader_engine_;
    const uint64_t                            transfer_;
    std::shared_ptr<MemoryUtil>               memory_;
    std::shared_ptr<HixlSourceGrants>         grants_;
    hixl_proto::ManifestResponse*             response_;
    google::protobuf::Closure*                done_;
    std::unordered_map<std::string, uint64_t> pending_;
    std::vector<std::shared_ptr<BlockBuffer>> ready_;
};

bool validRequest(const hixl_proto::ManifestRequest& request) {
    if (request.request_id().empty() || request.reader_id().empty() || request.reader_engine_id().empty()
        || !request.transfer_id() || !request.timeout_ms() || !request.blocks_size()) {
        return false;
    }
    std::unordered_map<std::string, uint64_t> keys;
    for (const auto& block : request.blocks()) {
        if (block.key().empty() || !block.bytes() || block.bytes() > std::numeric_limits<uint32_t>::max()
            || !keys.emplace(block.key(), block.bytes()).second) {
            return false;
        }
    }
    return true;
}

}  // namespace

void HixlManifestService::manifest(google::protobuf::RpcController*,
                                   const hixl_proto::ManifestRequest* request,
                                   hixl_proto::ManifestResponse*      response,
                                   google::protobuf::Closure*         done) {
    if (!validRequest(*request)) {
        response->set_status(hixl_proto::INVALID_REQUEST);
        done->Run();
        return;
    }
    bool admitted;
    {
        std::lock_guard<std::mutex> lock(admission_mutex_);
        admitted = ready_ && grants_->reserve(
            request->reader_id(), request->reader_engine_id(), request->transfer_id(), request->retired_through());
    }
    if (!admitted) {
        response->set_status(hixl_proto::SOURCE_UNAVAILABLE);
        done->Run();
        return;
    }
    response->set_source_instance(source_instance_);
    auto context = std::make_shared<ManifestContext>(*request, engine_id_, memory_, grants_, response, done);
    std::weak_ptr<ManifestContext> weak_context = context;
    // A timeout can finish the RPC and destroy its request before registration returns.
    const auto request_id = request->request_id();
    // The timer holds only a weak reference. A completed watch contains no
    // tensors or response pointer; request removal destroys it normally.
    timers_->schedule(HixlTimerQueue::Clock::now() + std::chrono::milliseconds(request->timeout_ms()), [weak_context] {
        if (auto context = weak_context.lock()) {
            context->fail(hixl_proto::DEADLINE_EXCEEDED);
        }
    });
    if (!store_->setRequestBlockBufferWatchFunc(
            request_id, [context](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
                context->onBlocks(ok, blocks);
            })) {
        context->fail(hixl_proto::SOURCE_UNAVAILABLE);
    }
}

void HixlManifestService::release(google::protobuf::RpcController*,
                                  const hixl_proto::ReleaseRequest* request,
                                  hixl_proto::ReleaseResponse*      response,
                                  google::protobuf::Closure*        done) {
    response->set_released(
        !request->reader_id().empty() && !request->reader_engine_id().empty() && request->transfer_id() != 0
        && grants_->retire(
            request->reader_id(), request->reader_engine_id(), request->transfer_id(), request->retired_through()));
    done->Run();
}

}  // namespace rtp_llm
