#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaTaskContext.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto/rdma_service.pb.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <exception>
#include <future>
#include <limits>
#include <unordered_map>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

RdmaStagingBufferPool::RdmaStagingBufferPool(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                                             size_t block_count,
                                             size_t block_size_bytes):
    memory_manager_(memory_manager),
    pool_(std::make_shared<HostStagingBlockPool>(block_count, block_size_bytes)),
    block_count_(block_count),
    stride_bytes_(block_size_bytes) {}

std::shared_ptr<RdmaStagingBufferPool> RdmaStagingBufferPool::create(
    const std::shared_ptr<IRdmaMemoryManager>& memory_manager, size_t total_bytes, size_t block_size_bytes) {
    if (!memory_manager || total_bytes == 0 || block_size_bytes == 0
        || block_size_bytes > std::numeric_limits<size_t>::max() - (HostStagingBlockPool::kAlignment - 1)) {
        RTP_LLM_LOG_ERROR("invalid rdma staging pool config: total_bytes=%zu block_size=%zu alignment=%zu",
                          total_bytes,
                          block_size_bytes,
                          HostStagingBlockPool::kAlignment);
        return nullptr;
    }
    const size_t aligned_block_size =
        ((block_size_bytes + HostStagingBlockPool::kAlignment - 1) / HostStagingBlockPool::kAlignment)
        * HostStagingBlockPool::kAlignment;
    const size_t block_count = total_bytes / aligned_block_size;
    if (block_count == 0) {
        RTP_LLM_LOG_ERROR("rdma staging total bytes cannot hold one LLM block: total_bytes=%zu aligned_block_size=%zu",
                          total_bytes,
                          aligned_block_size);
        return nullptr;
    }
    try {
        auto result = std::shared_ptr<RdmaStagingBufferPool>(
            new RdmaStagingBufferPool(memory_manager, block_count, aligned_block_size));
        // HostStagingBlockPool's AlignedHostMemory requires pinned CPU memory.
        if (!result->init(block_count)) {
            return nullptr;
        }
        RTP_LLM_LOG_INFO("P2P RDMA H2D staging pool ready: configured_bytes=%zu usable_bytes=%zu blocks=%zu "
                         "block_size=%zu",
                         total_bytes,
                         block_count * aligned_block_size,
                         block_count,
                         aligned_block_size);
        return result;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("create rdma staging pool failed: %s", e.what());
        return nullptr;
    }
}

bool RdmaStagingBufferPool::init(size_t block_count) {
    std::vector<HostStagingBlockPool::HostStagingBlockLease> leases;
    leases.reserve(block_count);
    registered_blocks_.reserve(block_count);
    for (size_t i = 0; i < block_count; ++i) {
        auto lease_batch = mallocBatchWithBackoff(1, std::chrono::seconds(1));
        if (!lease_batch || lease_batch->empty()) {
            RTP_LLM_LOG_ERROR("acquire rdma staging block for registration failed: index=%zu", i);
            return false;
        }
        auto view = lease_batch->front().blockBuffer(stride_bytes_);
        BlockInfo block_info;
        block_info.addr = view.base;
        block_info.size_bytes = view.capacity_bytes;
        // Keep each staging slot in one MR so a packed sub-block never crosses
        // an internal MR boundary during RdmaConnection::findMemoryMr().
        if (!memory_manager_->regUserMr(block_info, block_info.size_bytes)) {
            RTP_LLM_LOG_ERROR("register rdma staging block failed: index=%zu addr=%p size=%zu",
                              i,
                              block_info.addr,
                              block_info.size_bytes);
            return false;
        }
        registered_blocks_.push_back(block_info);
        leases.push_back(std::move(lease_batch->front()));
    }
    return true;
}

RdmaStagingBufferPool::~RdmaStagingBufferPool() {
    pool_->cancelAllBatchWaiters();
    for (const auto& block_info : registered_blocks_) {
        if (!memory_manager_->deregUserMr(block_info)) {
            RTP_LLM_LOG_WARNING(
                "deregister rdma staging block failed: addr=%p size=%zu", block_info.addr, block_info.size_bytes);
        }
    }
}

std::optional<HostStagingBlockPool::HostStagingBlockBatch>
RdmaStagingBufferPool::mallocBatchWithBackoff(size_t count, std::chrono::milliseconds timeout) {
    using Batch = HostStagingBlockPool::HostStagingBlockBatch;
    auto ready = std::make_shared<std::promise<std::optional<Batch>>>();
    auto result = ready->get_future();
    const auto deadline = HostStagingBlockPool::Clock::now() + timeout;
    pool_->requestBatch(count, deadline, [ready](std::optional<Batch> batch) { ready->set_value(std::move(batch)); });
    // Pool deadlines are processed on release; bound this worker's wait even
    // when every slot remains in flight. The callback owns no stack references.
    if (result.wait_until(deadline) != std::future_status::ready) {
        return std::nullopt;
    }
    return result.get();
}

size_t RdmaStagingBufferPool::strideBytes() const {
    return stride_bytes_;
}

size_t RdmaStagingBufferPool::blockCount() const {
    return block_count_;
}

RdmaTaskContext::RdmaTaskContext(::google::protobuf::RpcController* controller,
                                 const ::rdma_transfer::RdmaLayerBlockTransferRequest* request,
                                 ::rdma_transfer::RdmaLayerBlockTransferResponse* response,
                                 ::google::protobuf::Closure* done,
                                 const kmonitor::MetricsReporterPtr& metrics_reporter):
    controller_(controller),
    request_(request),
    response_(response),
    done_(done),
    metrics_reporter_(metrics_reporter),
    unique_key_(request->unique_key()),
    collector_(std::make_shared<TransferServerMetricsCollector>()),
    start_time_us_(currentTimeUs()),
    deadline_ms_(request->timeout_ms() > 0 && request->timeout_ms() <= std::numeric_limits<int32_t>::max() ?
                     start_time_us_ / 1000 + request->timeout_ms() :
                     0) {}

RdmaTaskContext::~RdmaTaskContext() {
    if (done_) {
        run(false, TransferErrorCode::CONTEXT_DESTROYED, "RdmaTaskContext destroyed without completion");
    }
}

void RdmaTaskContext::setTask(const std::shared_ptr<TransferTask>& task) {
    task_                                = task;
    collector_->wait_task_run_latency_us = currentTimeUs() - start_time_us_;
}

bool RdmaTaskContext::startTransfer() {
    if (!task_) {
        RTP_LLM_LOG_WARNING("startTransfer: task is null, unique_key: %s", unique_key_.c_str());
        return false;
    }
    return task_->startTransfer();
}

std::string RdmaTaskContext::getUniqueKey() const {
    return unique_key_;
}

bool RdmaTaskContext::isTimeout() const {
    return currentTimeMs() > deadline_ms_;
}

uint64_t RdmaTaskContext::getDeadlineMs() const {
    return static_cast<uint64_t>(deadline_ms_);
}

std::string RdmaTaskContext::getRdmaServerIp() const {
    return request_->server_rdma_ip();
}

uint32_t RdmaTaskContext::getRdmaServerPort() const {
    return request_->server_rdma_port();
}

std::optional<torch::Tensor> RdmaTaskContext::findLocalBuffer(const KeyBlockInfo& target, int sub_block_idx) {
    if (sub_block_idx < 0 || static_cast<size_t>(sub_block_idx) >= target.blocks.size()) {
        return std::nullopt;
    }
    const BlockInfo& bi = target.blocks[sub_block_idx];
    if (bi.addr == nullptr || bi.size_bytes == 0) {
        return std::nullopt;
    }
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    if (bi.is_cuda) {
        options = options.device(torch::kCUDA, bi.device_index);
    }
    return torch::from_blob(bi.addr, {static_cast<int64_t>(bi.size_bytes)}, options);
}

RdmaBlockPairBuildResult
RdmaTaskContext::buildRdmaBlockPairs(const std::shared_ptr<RdmaStagingBufferPool>& staging_pool) {
    if (!task_) {
        return {false, {}, "rdma block validation failed: recv task is null"};
    }

    const bool has_batch_callback = task_->acceptsBatches();
    if (!has_batch_callback && task_->getBlockInfos().empty()) {
        return {false, {}, "rdma block validation failed: Decode expected block set is empty"};
    }

    std::unordered_map<int64_t, const ::rdma_transfer::RdmaCacheKeyBlockBufferInfo*> offered_blocks;
    offered_blocks.reserve(static_cast<size_t>(request_->blocks_size()));
    for (const auto& cache_key_block : request_->blocks()) {
        if (!cache_key_block.has_key()) {
            return {false, {}, "rdma block validation failed: Prefill offered a block without cache key"};
        }
        const auto insert_result = offered_blocks.emplace(cache_key_block.key(), &cache_key_block);
        if (!insert_result.second) {
            return {false,
                    {},
                    "rdma block validation failed: duplicate Prefill cache key="
                        + std::to_string(cache_key_block.key())};
        }
    }

    KeyBlockInfoMap batch_blocks;
    if (has_batch_callback) {
        const auto& targets = task_->getBlockInfos();
        batch_blocks.reserve(offered_blocks.size());
        for (const auto& offered : offered_blocks) {
            const auto target = targets.find(offered.first);
            if (target != targets.end()) {
                batch_blocks.emplace(target->first, target->second);
            }
        }
    }
    const auto& expected_blocks = has_batch_callback ? batch_blocks : task_->getBlockInfos();

    std::vector<RdmaBlockPair> result;

    auto build_remote_buffer = [](const ::rdma_transfer::RdmaBlockBufferInfo& info) {
        const auto& rdma_info = info.rdma_info();
        auto        nic_rkeys = std::make_shared<std::map<uint32_t, uint32_t>>();
        for (const auto& nic_rkey : rdma_info.nic_rkeys()) {
            nic_rkeys->insert({nic_rkey.nicid(), nic_rkey.rkey()});
        }
        return std::make_shared<RemoteBuffer>(rdma_info.addr(), info.len(), nic_rkeys);
    };

    for (const auto& [cache_key, expected_kbi] : expected_blocks) {
        if (!expected_kbi) {
            return {false,
                    {},
                    "rdma block validation failed: null Decode block info for cache key=" + std::to_string(cache_key)};
        }
        if (expected_kbi->blocks.empty()) {
            return {false,
                    {},
                    "rdma block validation failed: Decode sub-block set is empty for cache key="
                        + std::to_string(cache_key)};
        }
        const auto offered_it = offered_blocks.find(cache_key);
        if (offered_it == offered_blocks.end()) {
            return {false,
                    {},
                    "rdma block validation failed: Prefill did not offer Decode cache key="
                        + std::to_string(cache_key)};
        }
        batch_keys_.push_back(cache_key);
        const auto& offered_kbi = *offered_it->second;
        if (offered_kbi.blocks_size() < static_cast<int>(expected_kbi->blocks.size())) {
            return {false,
                    {},
                    "rdma block validation failed: missing Decode sub-block for cache key=" + std::to_string(cache_key)
                        + ", expected=" + std::to_string(expected_kbi->blocks.size())
                        + ", offered=" + std::to_string(offered_kbi.blocks_size())};
        }
        for (int i = 0; i < static_cast<int>(expected_kbi->blocks.size()); ++i) {
            const auto& block_info = offered_kbi.blocks(i);
            if (!block_info.has_rdma_info() || !block_info.rdma_info().has_addr()
                || block_info.rdma_info().addr() == 0) {
                return {false,
                        {},
                        "rdma block validation failed: missing Prefill RDMA address for cache key="
                            + std::to_string(cache_key) + ", sub_block=" + std::to_string(i)};
            }
            auto local_buf = findLocalBuffer(*expected_kbi, i);
            if (!local_buf) {
                return {false,
                        {},
                        "rdma block validation failed: invalid Decode buffer for cache key=" + std::to_string(cache_key)
                            + ", sub_block=" + std::to_string(i)};
            }
            const auto local_size = static_cast<uint64_t>(local_buf->numel());
            if (!block_info.has_len() || block_info.len() == 0 || block_info.len() != local_size) {
                return {false,
                        {},
                        "rdma block validation failed: block length mismatch for cache key=" + std::to_string(cache_key)
                            + ", sub_block=" + std::to_string(i) + ", expected=" + std::to_string(local_size)
                            + ", offered=" + std::to_string(block_info.len())};
            }
            auto remote_buf = build_remote_buffer(block_info);
            remote_buf->partition_layer_key = unique_key_;
            result.push_back({*local_buf, remote_buf});
            collector_->total_block_size += block_info.len();
        }
    }
    collector_->block_count = static_cast<int>(result.size());

    if (!staging_pool) {
        return {true, std::move(result), {}};
    }

    size_t required_staging_blocks = 0;
    size_t staged_bytes_in_block   = 0;
    const size_t staging_capacity = staging_pool->strideBytes();
    for (const auto& pair : result) {
        if (!pair.first.is_cuda()) {
            continue;
        }
        const size_t block_bytes = static_cast<size_t>(pair.first.numel());
        if (block_bytes > staging_capacity) {
            RdmaBlockPairBuildResult failure;
            failure.error_message = "rdma staging block is smaller than an LLM cache sub-block";
            failure.error_code = TransferErrorCode::BUFFER_MISMATCH;
            return failure;
        }
        if (staged_bytes_in_block == 0 || block_bytes > staging_capacity - staged_bytes_in_block) {
            ++required_staging_blocks;
            staged_bytes_in_block = 0;
        }
        staged_bytes_in_block += block_bytes;
    }
    if (required_staging_blocks == 0) {
        return {true, std::move(result), {}};
    }
    if (required_staging_blocks > staging_pool->blockCount()) {
        RdmaBlockPairBuildResult failure;
        failure.error_message = "rdma staging pool total bytes cannot hold all cache sub-blocks in one transfer";
        failure.error_code = TransferErrorCode::BUFFER_MISMATCH;
        return failure;
    }

    const int64_t now_ms = currentTimeMs();
    const int64_t remaining_ms = deadline_ms_ > now_ms ? deadline_ms_ - now_ms : 0;
    auto leases =
        staging_pool->mallocBatchWithBackoff(required_staging_blocks, std::chrono::milliseconds(remaining_ms));
    if (!leases) {
        RdmaBlockPairBuildResult failure;
        failure.error_message = "rdma staging pool acquire timed out";
        failure.error_code = TransferErrorCode::TIMEOUT;
        return failure;
    }
    size_t staging_index  = 0;
    size_t staging_offset = 0;
    std::vector<tcp::CopyTask> h2d_copy_tasks;
    h2d_copy_tasks.reserve(result.size());
    for (auto& pair : result) {
        if (!pair.first.is_cuda()) {
            continue;
        }
        const size_t block_bytes = static_cast<size_t>(pair.first.numel());
        void* device_addr = pair.first.data_ptr();
        if (staging_offset != 0 && block_bytes > staging_capacity - staging_offset) {
            ++staging_index;
            staging_offset = 0;
        }
        auto  staging_view = (*leases)[staging_index].blockBuffer(staging_capacity);
        void* host_addr = static_cast<uint8_t*>(staging_view.base) + staging_offset;
        pair.first = torch::from_blob(host_addr,
                                      {static_cast<int64_t>(block_bytes)},
                                      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        h2d_copy_tasks.push_back({host_addr, block_bytes, static_cast<char*>(device_addr)});
        staging_offset += block_bytes;
    }

    auto staging_transfer =
        std::make_shared<RdmaStagingTransfer>(staging_pool, std::move(*leases), std::move(h2d_copy_tasks));
    return {true, std::move(result), {}, std::move(staging_transfer)};
}

namespace {

inline ::rdma_transfer::RdmaTransferErrorCodePB toRdmaProtoErrorCode(TransferErrorCode ec) {
    switch (ec) {
        case TransferErrorCode::OK:              return ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR;
        case TransferErrorCode::BUFFER_MISMATCH: return ::rdma_transfer::RDMA_TRANSFER_BUFFER_MISMATCH;
        case TransferErrorCode::RDMA_FAILED:     return ::rdma_transfer::RDMA_TRANSFER_RDMA_FAILED;
        case TransferErrorCode::TIMEOUT:         return ::rdma_transfer::RDMA_TRANSFER_CONTEXT_TIMEOUT;
        case TransferErrorCode::CANCELLED:       return ::rdma_transfer::RDMA_TRANSFER_TASK_CANCELLED;
        case TransferErrorCode::CONNECTION_FAILED:
            return ::rdma_transfer::RDMA_TRANSFER_CONNECTION_FAILED;
        case TransferErrorCode::NOT_INITIALIZED:
            return ::rdma_transfer::RDMA_TRANSFER_NOT_INITIALIZED;
        case TransferErrorCode::QUEUE_REJECTED:
            return ::rdma_transfer::RDMA_TRANSFER_QUEUE_REJECTED;
        case TransferErrorCode::COPY_FAILED:
            return ::rdma_transfer::RDMA_TRANSFER_COPY_FAILED;
        case TransferErrorCode::CONTEXT_DESTROYED:
            return ::rdma_transfer::RDMA_TRANSFER_CONTEXT_DESTROYED;
        default:                                 return ::rdma_transfer::RDMA_TRANSFER_UNKNOWN_ERROR;
    }
}

}  // anonymous namespace

void RdmaTaskContext::run(bool               success,
                           TransferErrorCode  error_code,
                           const std::string& error_message) {
    notifyTransferDone(success, success ? TransferErrorCode::OK : error_code, error_message);
    std::string final_error_message = error_message;

    // notifyTransferDone() may override the result (e.g. cancel_requested_ was
    // set during transfer). Read back the authoritative result before replying.
    if (task_) {
        error_code = task_->errorCode();
        success = (error_code == TransferErrorCode::OK);
        if (!success) {
            final_error_message = task_->errorMessage();
        }
    }
    finishRpc(success, error_code, final_error_message);
}

void RdmaTaskContext::notifyTransferDone(bool success, TransferErrorCode error_code, const std::string& error_message) {
    if (task_) {
        if (success && task_->acceptsBatches()) {
            task_->notifyBatchDone(batch_keys_);
        } else {
            task_->notifyDone(success, success ? TransferErrorCode::OK : error_code, error_message);
        }
    }
}

void RdmaTaskContext::recordTransferError(TransferErrorCode error_code, const std::string& error_message) {
    if (task_) {
        task_->recordError(error_code, error_message);
    }
}

void RdmaTaskContext::finishRpc(bool success, TransferErrorCode error_code, const std::string& error_message) {
    if (!done_) {
        return;
    }

    collector_->total_cost_latency_us = currentTimeUs() - start_time_us_;

    std::string final_error_message = error_message;
    if (task_) {
        if (!success) {
            task_->recordError(error_code, error_message);
        }
        error_code = task_->errorCode();
        success = error_code == TransferErrorCode::OK;
        if (!success) {
            final_error_message = task_->errorMessage();
        }
    }
    auto proto_error_code = toRdmaProtoErrorCode(error_code);

    collector_->success = success;
    if (metrics_reporter_) {
        metrics_reporter_->report<TransferMetric, TransferServerMetricsCollector>(nullptr, collector_.get());
    }

    if (!success) {
        RTP_LLM_LOG_WARNING("P2P RDMA receive failed, partition_layer_key=%s transfer_code=%d transfer_name=%s "
                            "wire_code=%d deadline_ms=%ld error=%s",
                            unique_key_.c_str(),
                            static_cast<int>(error_code),
                            transferErrorCodeToString(error_code),
                            static_cast<int>(proto_error_code),
                            deadline_ms_,
                            final_error_message.c_str());
    }
    response_->set_error_code(proto_error_code);
    response_->set_error_message(final_error_message);
    done_->Run();
    done_ = nullptr;
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
