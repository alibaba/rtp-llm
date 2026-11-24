#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockWriteback.h"
#include <list>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockAdapter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/utils/BackgroundProfilingScope.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
constexpr size_t kAlignment = 16;
ErrorInfo        writeError(const std::string& text) {
    return {ErrorCode::EXECUTION_EXCEPTION, text};
}
void notify(const std::function<void()>& callback) {
    try {
        callback();
    } catch (...) {
        RTP_LLM_LOG_ERROR("remote writer drain callback threw");
    }
}
}  // namespace
struct RemoteMockWriteback::Impl {
    struct Pins {
        std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> values;
        ~Pins() {
            for (const auto& pin : values)
                pin.first->decTreeRef(pin.second, BlockTreeRefType::STORE);
        }
    };
    struct Operation {
        explicit Operation(TransferTask task, DoneCallback callback):
            task(std::move(task)), callback(std::move(callback)) {}
        TransferTask                       task;
        DoneCallback                       callback;
        std::shared_ptr<Pins>              pins;
        std::shared_ptr<AlignedHostMemory> staging;
        size_t                             record{0}, offset{0};
        std::atomic<bool>                  finished{false};
    };
    std::weak_ptr<RemoteMockAdapter>                           adapter;
    PerRankBlockTransferEngine*                                engine;
    std::vector<GroupSetPtr>                                   groups;
    RemoteMockSession                                          session;
    RemoteMockRankCredits                                      credits;
    bool                                                       owns_device_allocations;
    size_t                                                     max_descriptors, storage_bytes{0};
    uint64_t                                                   backing_bytes{0};
    size_t                                                     admitted{0};
    BlockTreeTaskPool                                          worker;
    mutable std::mutex                                         mutex;
    std::condition_variable                                    idle;
    std::atomic<bool>                                          stopping{false};
    std::unordered_map<Operation*, std::shared_ptr<Operation>> active;
    std::list<std::function<void()>>                           drained;
    RemoteMockWritebackStats                                   counters;

    Impl(const std::shared_ptr<RemoteMockAdapter>& adapter,
         PerRankBlockTransferEngine*               engine,
         std::vector<GroupSetPtr>                  groups,
         bool                                      owns,
         size_t                                    max_descriptors):
        adapter(adapter),
        engine(engine),
        groups(std::move(groups)),
        session(adapter->session()),
        credits(adapter->credits()),
        owns_device_allocations(owns),
        max_descriptors(max_descriptors),
        worker(1, credits.write_operations, "RemoteWriteback") {
        size_t physical_records = 0;
        for (const auto& group : this->groups) {
            storage_bytes  = std::max(storage_bytes, group->storageBytes());
            size_t records = max_descriptors;
            for (const auto& pool : group->devicePools())
                records = std::min(records, pool->totalBlocksNum());
            physical_records += std::min(records, max_descriptors - physical_records);
        }
        this->max_descriptors = std::min(max_descriptors, physical_records);
        if (!engine || !this->max_descriptors || !credits.write_operations || storage_bytes > SIZE_MAX - kAlignment)
            throw std::invalid_argument("invalid remote writeback geometry");
        backing_bytes = storage_bytes + kAlignment;
        if (backing_bytes >= credits.local_write_bytes / credits.write_operations || !worker.start())
            throw std::invalid_argument("remote writeback has no bounded record window");
    }
    void post(std::function<void()> callback) {
        RTP_LLM_CHECK(worker.submitCompletion(std::move(callback)));
    }
    void finish(const std::shared_ptr<Operation>& op, ErrorInfo error) {
        if (op->finished.exchange(true))
            return;
        // No CUDA or borrowed Host alias can remain at this continuation.
        op->staging.reset();
        op->pins.reset();
        {
            std::lock_guard<std::mutex> lock(mutex);
            // The window is released before a terminal may submit the next batch.
            // active still guards drain until that callback returns.
            --admitted;
            counters.retained_bytes -= backing_bytes;
            if (error.ok())
                ++counters.completed;
            else
                ++counters.failed;
        }
        auto callback = std::move(op->callback);
        try {
            callback(std::move(error));
        } catch (...) {
            RTP_LLM_LOG_ERROR("remote writer callback threw");
        }
        callback = nullptr;
        std::list<std::function<void()>> callbacks;
        {
            std::lock_guard<std::mutex> lock(mutex);
            active.erase(op.get());
            if (active.empty())
                callbacks.swap(drained);
        }
        idle.notify_all();
        for (auto& callback : callbacks)
            notify(callback);
    }
    bool canContinue(const std::shared_ptr<Operation>& op) {
        if (stopping || op->task.expired()) {
            finish(op, writeError(stopping ? "remote writeback stopped" : "remote writeback deadline exceeded"));
            return false;
        }
        return true;
    }
    void copy(const std::shared_ptr<Operation>& op) {
        if (!canContinue(op))
            return;
        if (op->record == op->task.descriptors().size()) {
            finish(op, ErrorInfo::OkStatus());
            return;
        }
        try {
            if (!op->staging)
                op->staging = std::make_shared<AlignedHostMemory>(storage_bytes, kAlignment, "remote_writeback");
            std::memset(op->staging->data(), 0, storage_bytes);
            const auto& desc    = op->task.descriptors()[op->record];
            const auto& group   = groups[desc.group_set_id];
            auto        capture = TransferDescriptor::deviceToHost(desc.group_set_id, desc.source_blocks, 0);
            BackgroundProfilingScope scope("cache.remote_mock.write.capture");
            // Retain the operation, not a second staging owner: after real D2H,
            // finish can release the allocation even while onDone wrappers unwind.
            engine->executeHostBuffers(
                op->task.subtask({capture}),
                {{op->staging->data(), group->payloadBytes(), group->storageBytes()}},
                op,
                [this, op, post_error = writeError("physical completion continuation allocation failed")](
                    TransferResult result) mutable noexcept {
                    try {
                        post([this, op, error = std::move(result.status)] {
                            if (!error.ok())
                                finish(op, error);
                            else
                                send(op);
                        });
                    } catch (...) {
                        // Every DMA in this context has drained. No continuation
                        // was queued, so this operation has one safe terminal.
                        finish(op, std::move(post_error));
                    }
                });
        } catch (const std::exception& error) {
            finish(op, writeError(error.what()));
        }
    }
    void send(const std::shared_ptr<Operation>& op) {
        if (!canContinue(op))
            return;
        const auto& desc  = op->task.descriptors()[op->record];
        const auto& group = groups[desc.group_set_id];
        if (op->offset == group->storageBytes()) {
            ++op->record;
            op->offset = 0;
            copy(op);
            return;
        }
        auto retained = adapter.lock();
        if (!retained) {
            finish(op, writeError("remote writeback adapter unavailable"));
            return;
        }
        const auto bytes =
            static_cast<uint32_t>(std::min<size_t>(credits.max_write_slice_bytes, group->storageBytes() - op->offset));
        RemoteMockHostTransfer transfer;
        DoneCallback           terminal;
        try {
            transfer.deadline       = op->task.deadline();
            transfer.retained_bytes = backing_bytes;
            transfer.slices.push_back({static_cast<uint32_t>(desc.group_set_id),
                                       desc.remote_target->record_index,
                                       op->offset,
                                       std::shared_ptr<void>(op, op->staging->data() + op->offset),
                                       bytes});
            terminal =
                [this, op, bytes, post_error = writeError("Host acknowledgement continuation allocation failed")](
                    ErrorInfo error) mutable noexcept {
                    try {
                        // The Host alias drain delivered the slice ACK. Account the
                        // wire bytes even if the next local continuation cannot run.
                        if (error.ok()) {
                            std::lock_guard<std::mutex> lock(mutex);
                            counters.acknowledged_payload_bytes += bytes;
                        }
                        post([this, op, bytes, error = std::move(error)]() mutable {
                            if (!error.ok()) {
                                finish(op, std::move(error));
                                return;
                            }
                            op->offset += bytes;
                            send(op);
                        });
                    } catch (...) {
                        // The adapter calls us after RPC drain and the last registered Host
                        // alias release. No continuation was queued. Settle outside its locks.
                        finish(op, std::move(post_error));
                    }
                };
        } catch (const std::exception& error) {
            // D2H has completed and nothing was published. Drop local aliases
            // before releasing the bounded window or invoking user callbacks.
            transfer.slices.clear();
            finish(op, writeError(error.what()));
            return;
        } catch (...) {
            transfer.slices.clear();
            finish(op, writeError("Host slice construction failed"));
            return;
        }
        // ID zero is not terminal. transferHost owns all postpublication error
        // settlement and invokes this continuation only after real alias drain.
        retained->transferHost(transfer, std::move(terminal));
    }
};
RemoteMockWriteback::RemoteMockWriteback(const std::shared_ptr<RemoteMockAdapter>& adapter,
                                         PerRankBlockTransferEngine*               engine,
                                         std::vector<GroupSetPtr>                  groups,
                                         bool                                      owns,
                                         size_t                                    max_descriptors):
    impl_(std::make_unique<Impl>(adapter, engine, std::move(groups), owns, max_descriptors)) {}
RemoteMockWriteback::~RemoteMockWriteback() {
    stopAccepting();
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        impl_->idle.wait(lock, [&] { return impl_->active.empty(); });
    }
    impl_->worker.waitForIdle();
}
bool RemoteMockWriteback::submit(TransferTask task, DoneCallback callback) {
    if (!callback)
        return false;
    std::string rejection;
    const auto& descriptors = task.descriptors();
    if (task.expired() || descriptors.empty() || descriptors.size() > impl_->max_descriptors)
        rejection = "remote writeback invalid batch size or deadline";
    for (const auto& desc : descriptors) {
        if (!rejection.empty())
            break;
        if (!desc.isExecutable() || !desc.isRemoteWrite() || desc.group_set_id >= impl_->groups.size()
            || !(desc.remote_target->session == impl_->session)) {
            rejection = "remote writeback invalid descriptor/session";
            break;
        }
        const auto& pools = impl_->groups[desc.group_set_id]->devicePools();
        if (pools.size() != desc.source_blocks.size()) {
            rejection = "remote writeback invalid source shape";
            break;
        }
        for (size_t member = 0; member < pools.size(); ++member)
            if (!pools[member]->validBlock(desc.source_blocks[member])
                || (impl_->owns_device_allocations && !pools[member]->isAllocated(desc.source_blocks[member])))
                rejection = "remote writeback source is unavailable";
    }
    auto setup_error = writeError("remote writeback setup failed");
    auto op          = std::make_shared<Impl::Operation>(std::move(task), std::move(callback));
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (rejection.empty() && (impl_->stopping || impl_->admitted >= impl_->credits.write_operations))
            rejection = "remote writeback admission stopped or full";
        if (rejection.empty()) {
            impl_->active.emplace(op.get(), op);
            ++impl_->admitted;
            ++impl_->counters.submitted;
            impl_->counters.retained_bytes += impl_->backing_bytes;
            impl_->counters.peak_retained_bytes =
                std::max(impl_->counters.peak_retained_bytes, impl_->counters.retained_bytes);
        } else
            ++impl_->counters.rejected;
    }
    if (!rejection.empty()) {
        op->callback(writeError(rejection));
        return false;
    }
    try {
        op->pins = std::make_shared<Impl::Pins>();
        if (impl_->owns_device_allocations) {
            size_t count = 0;
            for (const auto& desc : op->task.descriptors())
                count += desc.source_blocks.size();
            op->pins->values.reserve(count);
            for (const auto& desc : op->task.descriptors()) {
                const auto& pools = impl_->groups[desc.group_set_id]->devicePools();
                for (size_t member = 0; member < pools.size(); ++member) {
                    pools[member]->incTreeRef(desc.source_blocks[member], BlockTreeRefType::STORE);
                    op->pins->values.emplace_back(pools[member], desc.source_blocks[member]);
                }
            }
        }
        impl_->post([impl = impl_.get(), op] { impl->copy(op); });
    } catch (...) {
        // The initial work item never entered the queue and no DMA has started.
        impl_->finish(op, std::move(setup_error));
    }
    return true;
}
void RemoteMockWriteback::stopAccepting() {
    impl_->stopping = true;
}
std::function<void()> RemoteMockWriteback::prepareDrain(std::function<void()> callback) {
    std::list<std::function<void()>> node;
    node.emplace_back(std::move(callback));
    return [this, node = std::move(node)]() mutable {
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (!impl_->active.empty()) {
                impl_->drained.splice(impl_->drained.end(), node);
                return;
            }
        }
        notify(node.front());
    };
}
void RemoteMockWriteback::drain(std::function<void()> callback) {
    if (callback)
        prepareDrain(std::move(callback))();
}
size_t RemoteMockWriteback::maxDescriptors() const {
    return impl_->max_descriptors;
}
RemoteMockWritebackStats RemoteMockWriteback::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->counters;
}
}  // namespace rtp_llm
