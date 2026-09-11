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
    std::lock_guard<std::mutex> lock(mutex);
    if (terminal_success.has_value()) {
        return;
    }
    cancelled = true;
    for (const auto& [key, task] : tasks) {
        task->cancel();
    }
}

bool WriteTaskGroup::fillStatus(WriteTaskStatus& status) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!terminal_success.has_value()) {
        int  done_now = 0;
        bool success  = !cancelled && error.ok();
        for (const auto& [key, task] : tasks) {
            const bool done = task->done();
            done_now += done ? 1 : 0;
            success = success && done && task->success();
        }
        lease->updateFinishedOps(done_now);
        if (lease->isStopped()) {
            terminal_success = success;
        }
    }
    status.sealed        = lease->isSealed();
    status.started_ops   = lease->startedOps();
    status.finished_ops  = lease->finishedOps();
    status.stopped       = lease->isStopped();
    status.write_success = terminal_success.value_or(false);
    status.error         = error;
    return status.stopped;
}

void WriteTaskGroup::releaseStoppedTasks() {
    std::lock_guard<std::mutex> lock(mutex);
    if (terminal_success.has_value()) {
        tasks.clear();
    }
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
                unit.blocks = LayerCacheBufferUtil::buildKeyBlockInfos(converter, buffer);
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
