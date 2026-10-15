#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteWorkerUtil.h"

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PKeyUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <exception>
#include <set>
#include <tuple>

namespace rtp_llm::p2p_internal {

void WriteTaskGroup::cancel() {
    std::vector<transfer::IKVCacheRecvTaskPtr> pending;
    {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled = true;
        if (lease->isStopped()) {
            return;
        }
        lease->seal();
        for (const auto& [key, task] : tasks) {
            pending.push_back(task);
        }
        cv.notify_all();
    }
    for (const auto& task : pending) {
        task->cancel();
    }
}

void WriteTaskGroup::watchTask(const std::string& key, const transfer::IKVCacheRecvTaskPtr& task) {
    task->setDoneCallback(
        [weak = weak_from_this(), weak_task = std::weak_ptr<transfer::IKVCacheRecvTask>(task), key]() {
            if (auto group = weak.lock()) {
                std::lock_guard<std::mutex> lock(group->mutex);
                if (auto completed = weak_task.lock(); completed && !completed->success() && group->error.ok()) {
                    group->error = ErrorInfo(transfer::toErrorCode(completed->errorCode()),
                                             "write transfer key=" + key + ": " + completed->errorMessage());
                }
                group->lease->onTransferFinished();
                group->cv.notify_all();
            }
        });
}

ErrorInfo WriteTaskGroup::waitDone() {
    std::unique_lock<std::mutex> lock(mutex);
    const auto deadline = std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms));
    if (!cv.wait_until(lock, deadline, [this]() { return lease->isStopped() || cancelled || error.hasError(); })
        && error.ok()) {
        error = ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "write transfer deadline exceeded");
    }
    if (error.ok() && cancelled) {
        error = ErrorInfo(ErrorCode::CANCELLED, "write transfer cancelled");
    }
    const auto result = error;
    lock.unlock();
    if (result.hasError()) {
        cancel();
    }
    return result;
}

void WriteTaskGroup::waitStopped() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() { return lease->isStopped(); });
}

void WriteTaskGroup::releaseStoppedTasks() {
    std::lock_guard<std::mutex> lock(mutex);
    if (lease->isStopped()) {
        tasks.clear();
    }
}

KVCacheResourcePtr WriteTaskGroup::sourceResource() {
    std::lock_guard<std::mutex> lock(mutex);
    return source_resource_;
}

bool WriteTaskGroup::beginSourceTransfer(int64_t deadline) {
    std::lock_guard<std::mutex> lock(mutex);
    if (source_submitted_ || cancelled || !source_resource_) {
        return false;
    }
    deadline_ms = std::min(deadline_ms, deadline);
    source_submitted_ = true;
    return true;
}

void WriteTaskGroup::releaseSourceResource() {
    std::lock_guard<std::mutex> lock(mutex);
    source_resource_.reset();
}

ErrorInfo buildWriteUnits(const std::string&                          unique_key,
                          int64_t                                     deadline_ms,
                          const P2PWorkerRoutePlan&                   worker_plan,
                          const P2PConnectorWorkerConfig&             config,
                          const std::shared_ptr<LayerBlockConverter>& converter,
                          bool                                        sending,
                          std::vector<WriteTransferUnit>&             units) {
    units.clear();
    const auto reject = [](const std::string& message) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, message);
    };
    if (!config.topology || !converter || unique_key.empty() || deadline_ms <= currentTimeMs()) {
        return reject("write request has no topology, key, converter, or live deadline");
    }
    std::set<int>                                         route_ids;
    std::set<std::string>                                 unit_keys;
    std::set<std::tuple<uint32_t, std::string, uint32_t>> recv_blocks;
    for (const auto& route : worker_plan.routes) {
        if (route.route_id < 0 || !route_ids.insert(route.route_id).second || route.layer_buffers.empty()) {
            return reject("write route has invalid/duplicate id or no layer blocks");
        }
        // Phase 1 accepts only whole-block, symmetric routes.
        if (route.partition.count != 1 || route.partition.id != 0 || route.slice.mode != CpBlockSliceMode::NONE
            || route.slice.count != 1 || route.slice.index != 0) {
            return reject("writeback currently requires symmetric whole-block routes");
        }
        if (sending && (route.dst_ip.empty() || route.dst_port == 0 || route.dst_port > 65535)) {
            return reject("write route has invalid peer endpoint");
        }
        for (const auto& buffer : route.layer_buffers) {
            if (!buffer || buffer->getLayerId() < 0
                || static_cast<size_t>(buffer->getLayerId()) >= config.topology->layers().size()
                || buffer->cacheTag() != route.cache_tag || buffer->blockIdMap().empty()) {
                return reject("write route has invalid layer, tag, or key/block pairs");
            }
            const auto& tags = config.topology->layer(buffer->getLayerId()).group_tags;
            if (std::find(tags.begin(), tags.end(), buffer->cacheTag()) == tags.end()) {
                return reject("write tag does not belong to layer");
            }
            const auto key = P2PKeyUtil::makeWriteBackRouteLayerKey(
                unique_key, buffer->getLayerId(), buffer->cacheTag(), route.route_id, worker_plan.plan_digest);
            if (!unit_keys.insert(key).second) {
                return reject("duplicate write layer/tag within route");
            }
            for (const auto& [cache_key, block_id] : buffer->blockIdMap()) {
                if (block_id <= 0) {
                    return reject("write request contains invalid block");
                }
                if (!sending && !recv_blocks.emplace(buffer->getLayerId(), buffer->cacheTag(), block_id).second) {
                    return reject("write routes overlap destination blocks");
                }
            }
            WriteTransferUnit unit;
            unit.key = key;
            try {
                auto key_block_infos = LayerCacheBufferUtil::buildKeyBlockInfos(converter, buffer);
                if (!key_block_infos.ok()) {
                    return key_block_infos.status();
                }
                unit.blocks = std::move(key_block_infos.value());
            } catch (const std::exception& e) {
                return reject(std::string("write block conversion failed: ") + e.what());
            }
            for (const auto& [cache_key, info] : unit.blocks) {
                if (!info || info->blocks.empty()) {
                    return reject("write block has no memory ranges");
                }
                for (const auto& block : info->blocks) {
                    if (!block.addr || block.size_bytes == 0) {
                        return reject("write block has an invalid memory range");
                    }
                }
            }
            if (unit.blocks.size() != buffer->blockIdMap().size()) {
                return reject("write block conversion omitted keys");
            }
            if (sending) {
                unit.ip   = route.dst_ip;
                unit.port = route.dst_port;
            }
            units.push_back(std::move(unit));
        }
    }
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm::p2p_internal
