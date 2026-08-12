#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr size_t kTransferWorkerCount = 2;
constexpr size_t kTransferQueueSize   = 1000;

ErrorInfo transferStatusToErrorInfo(TransferStatus status) {
    switch (status) {
        case TransferStatus::OK:
            return ErrorInfo::OkStatus();
        case TransferStatus::INVALID_ARGS:
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request");
        case TransferStatus::DEVICE_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device block transfer failed");
        case TransferStatus::DISK_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "disk block transfer failed");
        case TransferStatus::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device-disk staging pool exhausted");
    }
    return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unknown block transfer status");
}

}  // namespace

struct PerRankBlockTransferEngine::EndpointRegistry:
    public std::enable_shared_from_this<PerRankBlockTransferEngine::EndpointRegistry> {
    struct Key {
        Tier             tier;
        uintptr_t        pool;
        BlockIdxType     block;

        bool operator<(const Key& other) const {
            return std::tie(tier, pool, block) < std::tie(other.tier, other.pool, other.block);
        }
    };

    struct Access {
        Key    key;
        bool   write;
        size_t descriptor_index;
    };

    struct InFlight {
        size_t readers{0};
        bool   writer{false};
    };

    std::pair<ErrorInfo, std::shared_ptr<void>> reserve(const std::vector<TransferDescriptor>& descriptors,
                                                        const std::vector<const GroupSet*>&    group_sets) {
        std::vector<Access> accesses;
        for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
            const auto& descriptor = descriptors[descriptor_index];
            const auto& group_set  = *group_sets[descriptor_index];
            append(accesses, descriptor.source_tier, descriptor.source_blocks, group_set, false, descriptor_index);
            append(accesses, descriptor.target_tier, descriptor.target_blocks, group_set, true, descriptor_index);
        }

        std::map<Key, Access> batch_accesses;
        for (const auto& access : accesses) {
            const auto [it, inserted] = batch_accesses.emplace(access.key, access);
            if (!inserted && (it->second.write || access.write)) {
                return {ErrorInfo(ErrorCode::INVALID_PARAMS,
                                  "transfer endpoint conflict inside batch, descriptor_index="
                                      + std::to_string(access.descriptor_index)),
                        nullptr};
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const auto& [_, access] : batch_accesses) {
                const auto it = in_flight.find(access.key);
                if (it != in_flight.end()
                    && (access.write ? it->second.writer || it->second.readers > 0 : it->second.writer)) {
                    return {ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                      "RESOURCE_EXHAUSTED: transfer endpoint conflict, descriptor_index="
                                          + std::to_string(access.descriptor_index)),
                            nullptr};
                }
            }
            for (const auto& [key, access] : batch_accesses) {
                auto& state = in_flight[key];
                if (access.write) {
                    state.writer = true;
                } else {
                    ++state.readers;
                }
            }
        }

        std::vector<Access> reserved;
        reserved.reserve(batch_accesses.size());
        for (const auto& [_, access] : batch_accesses) {
            reserved.push_back(access);
        }
        auto self = shared_from_this();
        return {ErrorInfo::OkStatus(),
                std::shared_ptr<void>(nullptr, [self, reserved = std::move(reserved)](void*) {
                    self->release(reserved);
                })};
    }

    static void append(std::vector<Access>&                 accesses,
                       Tier                                 tier,
                       const std::vector<BlockIdxType>&     blocks,
                       const GroupSet&                      group_set,
                       bool                                 write,
                       size_t                               descriptor_index) {
        if (tier == Tier::DEVICE) {
            for (size_t index = 0; index < blocks.size(); ++index) {
                accesses.push_back(Access{Key{tier,
                                              reinterpret_cast<uintptr_t>(group_set.devicePools()[index].get()),
                                              blocks[index]},
                                          write,
                                          descriptor_index});
            }
        } else if (tier == Tier::HOST) {
            accesses.push_back(Access{Key{tier,
                                          reinterpret_cast<uintptr_t>(group_set.hostPool().get()),
                                          blocks.front()},
                                      write,
                                      descriptor_index});
        } else if (tier == Tier::DISK) {
            accesses.push_back(Access{Key{tier,
                                          reinterpret_cast<uintptr_t>(group_set.diskPool().get()),
                                          blocks.front()},
                                      write,
                                      descriptor_index});
        }
    }

    void release(const std::vector<Access>& accesses) {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto& access : accesses) {
            auto it = in_flight.find(access.key);
            if (access.write) {
                it->second.writer = false;
            } else {
                --it->second.readers;
            }
            if (!it->second.writer && it->second.readers == 0) {
                in_flight.erase(it);
            }
        }
    }

    std::mutex              mutex;
    std::map<Key, InFlight> in_flight;
};

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       DeviceHostCopyOptions    device_host_options,
                                                       size_t                   device_disk_staging_block_count,
                                                       size_t                   max_descriptors_per_batch):
    group_sets_(std::move(group_sets)),
    device_host_executor_(std::make_unique<DeviceHostTransferExecutor>(std::move(device_host_options))),
    host_disk_executor_(std::make_unique<HostDiskTransferExecutor>()),
    endpoint_registry_(std::make_shared<EndpointRegistry>()),
    max_descriptors_per_batch_(max_descriptors_per_batch) {
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);
    device_to_host_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferWorkerCount, kTransferQueueSize, "BlockD2HTransfer");
    host_to_device_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferWorkerCount, kTransferQueueSize, "BlockH2DTransfer");
    host_to_disk_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferWorkerCount, kTransferQueueSize, "BlockH2DiskTransfer");
    disk_to_host_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferWorkerCount, kTransferQueueSize, "BlockDisk2HTransfer");
    RTP_LLM_CHECK(device_to_host_task_pool_->start());
    RTP_LLM_CHECK(host_to_device_task_pool_->start());
    RTP_LLM_CHECK(host_to_disk_task_pool_->start());
    RTP_LLM_CHECK(disk_to_host_task_pool_->start());

    const bool any_disk_pool = std::any_of(group_sets_.begin(), group_sets_.end(), [](const GroupSetPtr& group_set) {
        return group_set->diskPool() != nullptr;
    });
    if (any_disk_pool) {
        device_disk_executor_ = std::make_unique<DeviceDiskTransferExecutor>(
            *device_host_executor_, *host_disk_executor_, group_sets_, device_disk_staging_block_count);
    }
}

PerRankBlockTransferEngine::~PerRankBlockTransferEngine() = default;

std::shared_ptr<AsyncContext>
PerRankBlockTransferEngine::submit(const std::vector<TransferDescriptor>& descriptors) {
    if (descriptors.empty()) {
        return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
    }

    const Tier source = descriptors.front().source_tier;
    const Tier target = descriptors.front().target_tier;
    std::vector<const GroupSet*> group_sets;
    std::vector<HostBufferView>  hosts;
    group_sets.reserve(descriptors.size());
    hosts.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
        if (descriptor.source_tier != source || descriptor.target_tier != target) {
            return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
        }
        const auto* group_set = group_sets_[descriptor.group_set_id].get();
        group_sets.push_back(group_set);
        if (source == Tier::HOST || target == Tier::HOST) {
            hosts.push_back(resolveHostView(*group_set, descriptor.singleBlockAt(Tier::HOST)));
        }
    }

    if ((source == Tier::HOST || target == Tier::HOST)
        && (source == Tier::DISK || target == Tier::DISK)) {
        const auto* disk_pool = group_sets.front()->diskPool().get();
        if (std::any_of(group_sets.begin(), group_sets.end(),
                        [disk_pool](const GroupSet* group_set) { return group_set->diskPool().get() != disk_pool; })) {
            return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
        }
    }

    auto [reservation_error, reservation] = endpoint_registry_->reserve(descriptors, group_sets);
    if (!reservation_error.ok()) {
        RTP_LLM_LOG_WARNING("rejecting transfer batch: %s", reservation_error.ToString().c_str());
        return std::make_shared<CompletedAsyncContext>(reservation_error);
    }

    if (source == Tier::DISK && target == Tier::DEVICE) {
        return device_disk_executor_->execute(descriptors, group_sets, std::move(reservation));
    }

    if (source == Tier::DEVICE && target == Tier::DISK) {
        if (descriptors.size() != 1) {
            return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
        }
        return std::make_shared<CompletedAsyncContext>(
            transferStatusToErrorInfo(device_disk_executor_->execute(descriptors.front(), *group_sets.front())));
    }

    BlockTreeTaskPool* task_pool = taskPoolForDirection(source, target);
    if (task_pool == nullptr) {
        return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
    }

    auto context = std::make_shared<TransferBatchAsyncContext>(std::move(reservation));
    const bool accepted = task_pool->submit([this, descriptors, group_sets, hosts, context] {
        try {
            for (size_t begin = 0; begin < descriptors.size(); begin += max_descriptors_per_batch_) {
                const size_t end = std::min(begin + max_descriptors_per_batch_, descriptors.size());
                const std::vector<HostBufferView> sub_hosts(hosts.begin() + begin, hosts.begin() + end);
                const std::vector<TransferDescriptor> sub_descriptors(descriptors.begin() + begin,
                                                                      descriptors.begin() + end);
                const std::vector<const GroupSet*> sub_group_sets(group_sets.begin() + begin,
                                                                  group_sets.begin() + end);
                const TransferStatus status = execute(sub_hosts, sub_descriptors, sub_group_sets);
                if (status != TransferStatus::OK) {
                    for (size_t index = begin; index < end; ++index) {
                        RTP_LLM_LOG_WARNING("transfer batch item failed, index=%zu %s",
                                            index,
                                            descriptors[index].debugString().c_str());
                    }
                    const auto error = transferStatusToErrorInfo(status);
                    context->complete(ErrorInfo(error.code(),
                                                error.ToString() + ", descriptor_range=[" + std::to_string(begin)
                                                    + "," + std::to_string(end) + ")"));
                    return;
                }
            }
            context->complete(ErrorInfo::OkStatus());
        } catch (const std::exception& error) {
            context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
        } catch (...) {
            context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown transfer executor exception"));
        }
    });
    if (!accepted) {
        context->complete(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: transfer queue is full or stopped"));
    }
    return context;
}

TransferStatus
PerRankBlockTransferEngine::execute(const std::vector<HostBufferView>&       hosts,
                                    const std::vector<TransferDescriptor>& descriptors,
                                    const std::vector<const GroupSet*>&    group_sets) const {
    const Tier source = descriptors.front().source_tier;
    const Tier target = descriptors.front().target_tier;
    if ((source == Tier::DEVICE && target == Tier::HOST) || (source == Tier::HOST && target == Tier::DEVICE)) {
        return device_host_executor_->execute(hosts, descriptors, group_sets);
    }
    if ((source == Tier::HOST && target == Tier::DISK) || (source == Tier::DISK && target == Tier::HOST)) {
        return host_disk_executor_->execute(hosts, descriptors, group_sets);
    }
    return TransferStatus::INVALID_ARGS;
}

BlockTreeTaskPool* PerRankBlockTransferEngine::taskPoolForDirection(Tier source, Tier target) const {
    if (source == Tier::DEVICE && target == Tier::HOST) {
        return device_to_host_task_pool_.get();
    }
    if (source == Tier::HOST && target == Tier::DEVICE) {
        return host_to_device_task_pool_.get();
    }
    if (source == Tier::HOST && target == Tier::DISK) {
        return host_to_disk_task_pool_.get();
    }
    if (source == Tier::DISK && target == Tier::HOST) {
        return disk_to_host_task_pool_.get();
    }
    return nullptr;
}

HostBufferView PerRankBlockTransferEngine::resolveHostView(const GroupSet& group_set, BlockIdxType host_block) {
    const HostBlockBuffer buffer = group_set.hostPool()->blockBuffer(host_block);
    return HostBufferView{buffer.addr, buffer.payload_bytes, buffer.stride_bytes};
}

}  // namespace rtp_llm
