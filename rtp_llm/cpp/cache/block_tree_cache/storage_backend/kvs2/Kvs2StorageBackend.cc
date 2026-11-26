#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvs2/Kvs2StorageBackend.h"

#include <algorithm>
#include <limits>
#include <map>
#include <sstream>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {
namespace {

constexpr size_t kMaxParts        = 8;
constexpr size_t kMaxControlKeys  = 256;
constexpr int    kMaxTargets      = 256;
constexpr size_t kMaxIovs         = 1024;
constexpr int    kMaxTimeoutMs    = 1800000;
constexpr size_t kControlBytes    = 1u << 20;
constexpr size_t kPartReplyBudget = 8192;

struct Kvs2MatchMeta final: StorageBackendMatchMeta {
    kvs::BlockLocations locations;
};

const kvs::PartLocation* findLocation(const kvs::BlockLocation& location, const kvs::BlockPart& part) {
    const auto found = std::find_if(location.parts.begin(), location.parts.end(), [&](const auto& item) {
        return item.name == part.name && item.size_bytes == part.size_bytes && !item.uri.empty();
    });
    return found == location.parts.end() ? nullptr : &*found;
}

void checkStatus(const kvs::Status& status, const char* operation) {
    if (status.code == kvs::Code::Timeout) {
        throw StorageDeadlineExceeded(std::string(operation) + ": " + status.message);
    }
    if (!status.ok()) {
        throw StorageOperationBufferSafeFailure(std::string(operation) + ": " + status.message);
    }
}

void appendPart(RemoteOperationRequestPB& payload, const StorageBlockHandle& handle, const kvs::PartLocation& part) {
    payload.add_group_tags(handle.tag);
    payload.add_block_ids(handle.block);
    payload.add_uris(part.uri);
}

}  // namespace

Kvs2StorageBackend::Kvs2StorageBackend(const CacheConfig&                               cache_config,
                                       kvs::ClientOptions                               options,
                                       std::string                                      tenant,
                                       std::string                                      model_name,
                                       int64_t                                          tp_rank,
                                       int64_t                                          tp_size,
                                       int64_t                                          world_rank,
                                       std::shared_ptr<BroadcastManager>                 broadcast,
                                       std::shared_ptr<BlockTreeCacheMetricsReporter>    metrics):
    StorageBackend(nullptr, std::move(metrics), world_rank, std::chrono::milliseconds(options.timeout_ms)),
    cache_config_(cache_config),
    options_(std::move(options)),
    tenant_(std::move(tenant)),
    model_name_(std::move(model_name)),
    tp_rank_(tp_rank),
    tp_size_(tp_size),
    broadcast_(std::move(broadcast)) {}

Kvs2StorageBackend::~Kvs2StorageBackend() {
    shutdown();
}

bool Kvs2StorageBackend::initImpl() {
    if (tenant_.empty() || tp_size_ <= 0 || tp_rank_ < 0 || tp_rank_ >= tp_size_ || options_.timeout_ms <= 0
        || options_.timeout_ms > kMaxTimeoutMs
        || (tp_rank_ == 0 && tp_size_ > 1 && (!broadcast_ || broadcast_->workerNum() != static_cast<size_t>(tp_size_)))) {
        RTP_LLM_LOG_ERROR("KVS2 requires a tenant, positive deadline, and all TP workers on rank 0");
        return false;
    }
    std::ostringstream identity;
    identity << tenant_.size() << ':' << tenant_ << ':' << model_name_.size() << ':' << model_name_
             << ':' << cache_config_.seq_size_per_block << ':' << static_cast<int>(cache_config_.dtype) << ':'
             << tp_size_;
    for (const auto& group : topology().groups()) {
        if (!cacheGroupParticipatesInBlockTree(group.policy)) {
            continue;
        }
        const auto bytes = cache_config_.blockSizeBytesForGroup(group.tag);
        if (bytes == 0 || bytes > std::numeric_limits<uint64_t>::max() / tp_size_) {
            return false;
        }
        layout_.push_back({group.tag, static_cast<uint64_t>(bytes) * tp_size_});
        identity << ':' << group.tag.size() << ':' << group.tag << ':' << bytes
                 << ':' << static_cast<int>(group.policy.group_type) << ':' << group.policy.sliding_window_size;
        for (const int layer : topology().layerIdsForGroup(group.tag)) {
            const auto& physical = cache_config_.physicalGroupForLayer(layer, group.tag);
            const auto  spec     = physical.spec->fingerprint();
            identity << ':' << layer << ':' << spec.size() << ':' << spec;
        }
    }
    if (layout_.empty() || layout_.size() > kMaxParts) {
        RTP_LLM_LOG_ERROR("KVS2 requires 1 to %zu reusable cache groups, got %zu", kMaxParts, layout_.size());
        return false;
    }
    uint64_t fingerprint = 14695981039346656037ULL;
    for (const unsigned char byte : identity.str()) {
        fingerprint = (fingerprint ^ byte) * 1099511628211ULL;
    }
    layout_id_ = std::to_string(fingerprint);
    if (tp_rank_ == 0) {
        kvs::PoolInfo pool;
        checkStatus(kvs::MasterClient(options_).GetPool(options_.pool, &pool), "KVS2 get pool");
    }
    auto runtime = std::make_unique<kvs::Client>(options_);
    checkStatus(runtime->Open(), "KVS2 open");
    std::unique_lock lock(runtime_mutex_);
    runtime_client_ = std::move(runtime);
    return true;
}

void Kvs2StorageBackend::shutdownImpl() noexcept {
    std::unique_lock lock(runtime_mutex_);
    runtime_client_.reset();
}

kvs::ClientOptions Kvs2StorageBackend::optionsFor(TransferTask::Clock::time_point deadline) const {
    const auto remaining = std::chrono::floor<std::chrono::milliseconds>(deadline - TransferTask::Clock::now()).count();
    if (remaining <= 0) {
        throw StorageDeadlineExceeded("KVS2 task deadline expired");
    }
    auto options       = options_;
    options.timeout_ms = static_cast<int>(std::min<int64_t>({remaining, options_.timeout_ms, kMaxTimeoutMs}));
    return options;
}

std::string Kvs2StorageBackend::key(CacheKeyType key) const {
    return "rtp-v1/" + layout_id_ + "/" + std::to_string(key);
}

StorageMatchResult Kvs2StorageBackend::matchImpl(const StorageRequest& request) {
    const auto  deadline = request.transfer_task->deadline();
    const auto& keys     = *request.keys;
    const auto  local    = request.local_matched_blocks_num;
    auto        meta     = std::make_shared<Kvs2MatchMeta>();
    const size_t max_keys = kControlBytes / (layout_.size() * kPartReplyBudget);
    std::vector<size_t> consecutive(layout_.size(), 0);
    size_t matched = local;
    for (size_t begin = local; begin < keys.size();) {
        const size_t end = begin + std::min(max_keys, keys.size() - begin);
        kvs::BlockQuery query{tenant_, {}, layout_, {}};
        for (size_t pos = begin; pos < end; ++pos) {
            query.keys.push_back(key(keys[pos]));
        }
        kvs::BlockLocations locations;
        const auto status = kvs::MasterClient(optionsFor(deadline)).Lookup(query, &locations);
        if (!status.ok() || locations.size() != query.keys.size()) {
            break;
        }
        for (size_t pos = begin; pos < end; ++pos) {
            auto& location = locations[pos - begin];
            bool  complete = true;
            for (size_t part_index = 0; part_index < layout_.size(); ++part_index) {
                const auto& part = layout_[part_index];
                auto&       run  = consecutive[part_index];
                run = findLocation(location, part) ? run + 1 : 0;
                const auto& group = topology().group(part.name);
                const auto& handles = request.handles[pos];
                if (std::none_of(handles.begin(), handles.end(), [&](const auto& handle) {
                        return handle.tag == part.name;
                    })) {
                    continue;
                }
                complete = complete && run >= std::min(pos + 1 - local, group.reuseBlockCount(pos + 1));
            }
            meta->locations.push_back(std::move(location));
            if (complete) {
                matched = pos + 1;
            }
        }
        begin = end;
    }
    meta->locations.resize(matched - local);
    return {matched, matched > local ? std::move(meta) : nullptr};
}

FunctionRequestPB Kvs2StorageBackend::payloadRequest(bool write) {
    FunctionRequestPB request;
    auto* payload = request.mutable_remote_request();
    payload->set_op(write ? REMOTE_OPERATION_WRITE : REMOTE_OPERATION_READ);
    payload->set_trace_id(registry_.epoch() + "/kvs2/" + std::to_string(++sequence_));
    payload->set_kvs2_layout_id(layout_id_);
    return request;
}

void Kvs2StorageBackend::readImpl(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& meta) {
    const auto matched = std::dynamic_pointer_cast<Kvs2MatchMeta>(meta);
    if (!matched || matched->locations.size() < request.handles.size() - request.local_matched_blocks_num) {
        throw StorageOperationBufferSafeFailure("KVS2 read does not match the lookup");
    }
    auto payload = payloadRequest(false);
    const auto submit = [&] {
        const auto success = dispatch(request, payload);
        if (std::find(success.begin(), success.end(), false) != success.end()) {
            throw StorageOperationBufferSafeFailure("KVS2 read failed");
        }
    };
    for (size_t pos = request.local_matched_blocks_num; pos < request.handles.size(); ++pos) {
        const auto& location = matched->locations[pos - request.local_matched_blocks_num];
        for (const auto& handle : request.handles[pos]) {
            const auto part = std::find_if(layout_.begin(), layout_.end(), [&](const auto& item) {
                return item.name == handle.tag;
            });
            const auto* found = part == layout_.end() ? nullptr : findLocation(location, *part);
            if (!found || isNullBlockIdx(handle.block)) {
                throw StorageOperationBufferSafeFailure("KVS2 read has incomplete coverage");
            }
            appendPart(*payload.mutable_remote_request(), handle, *found);
            if (payload.remote_request().uris_size() == kMaxTargets) {
                submit();
                payload = payloadRequest(false);
            }
        }
    }
    submit();
}

void Kvs2StorageBackend::writeImpl(const StorageRequest& request) {
    const auto deadline = request.transfer_task->deadline();
    std::map<std::vector<std::string>, std::vector<size_t>> groups;
    for (size_t pos = 0; pos < request.handles.size(); ++pos) {
        std::vector<std::string> parts;
        for (const auto& handle : request.handles[pos]) {
            if (!isNullBlockIdx(handle.block)) {
                parts.push_back(handle.tag);
            }
        }
        if (!parts.empty()) {
            std::sort(parts.begin(), parts.end());
            groups[std::move(parts)].push_back(pos);
        }
    }

    std::exception_ptr failure;
    const size_t max_keys = std::min(kMaxControlKeys, kControlBytes / (layout_.size() * kPartReplyBudget));
    for (const auto& [parts, indices] : groups) {
        for (size_t begin = 0; begin < indices.size();) {
            if (request.transfer_task->expired() || !remoteCacheEnabled()) {
                throw StorageDeadlineExceeded("KVS2 write admission closed before the next session");
            }
            const size_t end = begin + std::min(max_keys, indices.size() - begin);
            try {
                kvs::StartWriteRequest start{tenant_, {}, layout_, parts};
                for (size_t pos = begin; pos < end; ++pos) {
                    start.keys.push_back(key(request.keys->at(indices[pos])));
                }
                kvs::WriteContext write;
                checkStatus(kvs::MasterClient(optionsFor(deadline)).StartWrite(start, &write), "KVS2 start write");
                block_tree_cache_detail::ScopeRollback abort([&]() noexcept {
                    try {
                        const auto status = kvs::MasterClient(options_).AbortWrite(write.write_id);
                        if (!status.ok()) {
                            RTP_LLM_LOG_WARNING("KVS2 abort failed for write_id=%s code=%d",
                                                write.write_id.c_str(), static_cast<int>(status.code));
                        }
                    } catch (...) {}
                });
                if (write.locations.size() != write.allocated_indices.size()) {
                    throw StorageOperationBufferSafeFailure("KVS2 write allocation indices mismatch");
                }
                auto payload = payloadRequest(true);
                std::vector<size_t> owners;
                for (size_t row = 0; row < write.locations.size(); ++row) {
                    const auto relative = write.allocated_indices[row];
                    if (relative >= end - begin || write.locations[row].parts.empty()) {
                        throw StorageOperationBufferSafeFailure("KVS2 write returned an invalid allocation");
                    }
                    const auto& handles = request.handles[indices[begin + relative]];
                    for (const auto& part : write.locations[row].parts) {
                        const auto handle = std::find_if(handles.begin(), handles.end(), [&](const auto& item) {
                            return item.tag == part.name && !isNullBlockIdx(item.block);
                        });
                        const auto layout = std::find_if(layout_.begin(), layout_.end(), [&](const auto& item) {
                            return item.name == part.name && item.size_bytes == part.size_bytes;
                        });
                        if (handle == handles.end() || layout == layout_.end() || part.uri.empty()) {
                            throw StorageOperationBufferSafeFailure("KVS2 write has an unexpected Part");
                        }
                        appendPart(*payload.mutable_remote_request(), *handle, part);
                        owners.push_back(row);
                    }
                }
                const auto delivered = dispatch(request, payload);
                std::vector<bool> success(write.locations.size(), true);
                for (size_t pos = 0; pos < owners.size(); ++pos) {
                    success[owners[pos]] = success[owners[pos]] && delivered[pos];
                }
                auto master = kvs::MasterClient(optionsFor(deadline));
                abort.dismiss();
                auto status = master.FinishWrite(write.write_id, success);
                if ((status.code == kvs::Code::Timeout || status.code == kvs::Code::Unavailable)
                    && !request.transfer_task->expired()) {
                    status = kvs::MasterClient(optionsFor(deadline)).FinishWrite(write.write_id, success);
                }
                checkStatus(status, "KVS2 finish write");
                if (std::find(success.begin(), success.end(), false) != success.end()) {
                    throw StorageOperationBufferSafeFailure("KVS2 write published only completed blocks");
                }
            } catch (...) {
                failure = std::current_exception();
            }
            begin = end;
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

std::vector<bool> Kvs2StorageBackend::dispatch(const StorageRequest& request, FunctionRequestPB& payload) {
    auto&      remote   = *payload.mutable_remote_request();
    const auto deadline = request.transfer_task->deadline();
    const bool write    = remote.op() == REMOTE_OPERATION_WRITE;
    if (remote.uris().empty()) {
        return {};
    }
    std::vector<bool> success(remote.uris_size(), true);
    auto collect = [&](const RemoteOperationResponsePB& response) {
        const auto& mask = response.succeed_block().bool_masks().values();
        for (size_t index = 0; index < success.size(); ++index) {
            success[index] = success[index] && static_cast<size_t>(mask.size()) == success.size() && mask.Get(index);
        }
        return response.payload_buffer_safe();
    };
    remote.set_timeout_ms(optionsFor(deadline).timeout_ms);
    if (!broadcast_) {
        RemoteOperationResponsePB response;
        execute(remote, response);
        if (!collect(response)) {
            throw StorageOperationCompletionUnknown("KVS2 local payload safety unknown",
                                                     StorageQuarantineReason::SDK_ERROR);
        }
        registry_.forget(remote.trace_id());
        return success;
    }

    FrozenTask::ExecutionInfo execution;
    execution.trace_id            = remote.trace_id();
    execution.operation           = write ? "write" : "read";
    execution.request_id          = request.request_id;
    execution.workflow_id         = request.transfer_task->workflowId();
    execution.parent_trace_id     = request.transfer_task->traceId();
    execution.remote              = true;
    execution.allow_force_release = false;
    TransferCompletionTarget target;
    target.remote = true;
    const auto& addresses = broadcast_->workerAddresses();
    for (size_t rank = 0; rank < addresses.size(); ++rank) {
        execution.expected_peers.push_back({rank, addresses[rank]});
        target.peers.push_back({rank, addresses[rank]});
    }
    execution.expected_rank_count = addresses.size();
    for (int index = 0; index < remote.block_ids_size(); ++index) {
        const auto& tag = remote.group_tags(index);
        FrozenTask::DescriptorInfo descriptor;
        descriptor.path_index = index;
        descriptor.group_tag  = tag;
        descriptor.targets.push_back({devicePool(tag), remote.block_ids(index)});
        execution.descriptors.push_back(std::move(descriptor));
    }
    auto prepared = FrozenTask::prepare(std::move(execution));
    std::list<std::shared_ptr<FrozenTask>> frozen_slot(1);
    if (!cleanup_) {
        throw StorageOperationBufferSafeFailure("KVS2 rank completion cleanup is unavailable");
    }
    auto cleanup = cleanup_->prepare(std::move(target), remote.trace_id());
    remote.set_timeout_ms(optionsFor(deadline).timeout_ms);
    bool submitted = true;
    bool safe = false;
    block_tree_cache_detail::ScopeRollback settle([&]() noexcept {
        if (!submitted) {
            return;
        }
        if (safe) {
            cleanup.submit();
            return;
        }
        auto frozen = std::make_shared<FrozenTask>(
            std::move(prepared),
            TransferResult(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "KVS2 rank completion unknown")),
            std::move(cleanup));
        frozen_slot.front() = frozen;
        frozen->freeze();
        std::lock_guard lock(frozen_mutex_);
        frozen_.splice(frozen_.end(), frozen_slot);
    });
    const auto result = broadcast_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        payload,
        static_cast<int>(remote.timeout_ms()),
        [deadline](const std::shared_ptr<RpcService::Stub>& stub,
                   const std::shared_ptr<grpc::ClientContext>& context,
                   const FunctionRequestPB& call,
                   grpc::CompletionQueue* queue) {
            context->set_deadline(std::chrono::system_clock::now() + (deadline - TransferTask::Clock::now()));
            return stub->AsyncExecuteFunction(context.get(), call, queue);
        },
        request.request_id,
        write ? "D2REMOTE" : "REMOTE2D",
        false);
    if (!result) {
        submitted = false;
        throw StorageOperationBufferSafeFailure("KVS2 rank dispatch rejected");
    }
    if (!result->done()) {
        const auto remaining = request.transfer_task->remainingTimeout();
        if (!remaining || !result->waitDone(static_cast<int>(std::min<int64_t>(remaining->count(), kMaxTimeoutMs)), false)) {
            throw StorageOperationTimeout("KVS2 rank RPC deadline expired",
                                          StorageQuarantineReason::RPC_TIMEOUT,
                                          StorageLocalTargetProtection::TARGETS_HELD);
        }
    }
    const auto responses = result->responses();
    if (responses.size() != addresses.size()) {
        throw StorageOperationCompletionUnknown("KVS2 rank reply count mismatch",
                                                 StorageQuarantineReason::INVALID_RESPONSE,
                                                 StorageLocalTargetProtection::TARGETS_HELD);
    }
    bool all_safe = true;
    for (size_t rank = 0; rank < responses.size(); ++rank) {
        const auto& response = responses[rank];
        all_safe = (result->rpcSucceeded(rank) && response.has_remote_response() && collect(response.remote_response()))
                   && all_safe;
    }
    safe = all_safe;
    if (!safe) {
        if (result->deadlineExceeded()) {
            throw StorageOperationTimeout("KVS2 rank RPC timed out",
                                          StorageQuarantineReason::RPC_TIMEOUT,
                                          StorageLocalTargetProtection::TARGETS_HELD);
        }
        throw StorageOperationCompletionUnknown("KVS2 rank I/O completion unknown",
                                                 StorageQuarantineReason::RPC_ERROR,
                                                 StorageLocalTargetProtection::TARGETS_HELD);
    }
    return success;
}

void Kvs2StorageBackend::execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
    const auto received = TransferTask::Clock::now();
    const std::shared_lock lock(runtime_mutex_);
    response.Clear();
    response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
    if (!runtime_client_ || request.kvs2_layout_id() != layout_id_ || request.trace_id().empty()
        || (request.op() != REMOTE_OPERATION_READ && request.op() != REMOTE_OPERATION_WRITE)
        || request.uris_size() != request.group_tags_size() || request.uris_size() != request.block_ids_size()) {
        response.set_payload_buffer_safe(true);
        return;
    }
    if (!registry_.begin(request.trace_id())) {
        const auto state = registry_.queryLocal(std::vector<std::string>{request.trace_id()}).front();
        response.set_payload_buffer_safe(state != TransferTaskRegistry::State::RUNNING);
        return;
    }
    try {
        if (request.timeout_ms() <= 0 || request.timeout_ms() > kMaxTimeoutMs) {
            throw StorageDeadlineExceeded("KVS2 invalid or exhausted worker budget");
        }
        executePayload(request, received + std::chrono::milliseconds(request.timeout_ms()), response);
    } catch (const StorageDeadlineExceeded&) {
        response.set_transfer_status(REMOTE_TRANSFER_STATUS_TIMEOUT);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("KVS2 worker transfer failed trace=%s: %s", request.trace_id().c_str(), error.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("KVS2 worker transfer failed trace=%s", request.trace_id().c_str());
    }
    registry_.finish(request.trace_id(), true);
    response.set_payload_buffer_safe(true);
}

void Kvs2StorageBackend::executePayload(const RemoteOperationRequestPB& request,
                                       TransferTask::Clock::time_point   deadline,
                                       RemoteOperationResponsePB&       response) {
    const bool write = request.op() == REMOTE_OPERATION_WRITE;
    const auto max_bytes = options_.transfer.max_request_bytes;
    std::vector<bool> success(request.uris_size(), true);
    std::vector<kvs::TransferTarget> targets;
    kvs::TransferBuffers buffers;
    std::vector<size_t> owners;
    uint64_t bytes = 0;
    uint64_t staged_bytes = 0;
    size_t iov_count = 0;
    bool timed_out = false;
    auto flush = [&] {
        if (targets.empty()) {
            return;
        }
        kvs::Client client(optionsFor(deadline));
        kvs::TransferHandle handle;
        if (write) {
            client.async()->Write(targets, buffers, {{}, [](kvs::TransferResult) {}}, &handle);
        } else {
            client.async()->Read(targets, buffers, [](kvs::TransferResult) {}, &handle);
        }
        kvs::TransferResult result;
        if (handle.valid()) {
            result.status = handle.Wait(&result);
        } else {
            result.status = {kvs::Code::Unavailable, "KVS2 SDK did not admit the payload"};
        }
        timed_out = timed_out || result.status.code == kvs::Code::Timeout;
        for (size_t index = 0; index < owners.size(); ++index) {
            const bool ok = result.target_status.size() == owners.size() ? result.target_status[index].ok()
                                                                        : result.status.ok();
            success[owners[index]] = success[owners[index]] && ok;
        }
        targets.clear();
        buffers.clear();
        owners.clear();
        bytes = 0;
        staged_bytes = 0;
        iov_count = 0;
    };

    for (int index = 0; index < request.uris_size(); ++index) {
        (void)optionsFor(deadline);
        const auto& tag = request.group_tags(index);
        const auto part = std::find_if(layout_.begin(), layout_.end(), [&](const auto& candidate) {
            return candidate.name == tag;
        });
        const auto block = request.block_ids(index);
        if (part == layout_.end() || !devicePool(tag)->validBlock(block)) {
            throw StorageOperationBufferSafeFailure("KVS2 worker has an invalid group block");
        }
        const auto rank_bytes = part->size_bytes / tp_size_;
        uint64_t size = 0;
        kvs::TransferBuffer source;
        for (const int layer : topology().layerIdsForGroup(tag)) {
            for (const auto& info : convertIndexToBuffer(layer, tag, block)) {
                if (!info.addr || !info.size_bytes || info.size_bytes > rank_bytes - size) {
                    throw StorageOperationBufferSafeFailure("KVS2 worker cache layout mismatch");
                }
                source.iovs.push_back({info.is_cuda ? kvs::MemoryType::GPU : kvs::MemoryType::CPU,
                                       info.addr, info.size_bytes, false});
                size += info.size_bytes;
            }
        }
        if (size != rank_bytes) {
            throw StorageOperationBufferSafeFailure("KVS2 worker cache block is incomplete");
        }
        uint64_t offset = rank_bytes * tp_rank_;
        for (auto iov : source.iovs) {
            while (iov.size != 0) {
                const bool new_target = owners.empty() || owners.back() != static_cast<size_t>(index);
                const bool stage_full = write && (staged_bytes > max_bytes || part->size_bytes > max_bytes - staged_bytes);
                if (bytes == max_bytes || iov_count == kMaxIovs
                    || (new_target && (targets.size() == kMaxTargets || stage_full))) {
                    flush();
                }
                if (owners.empty() || owners.back() != static_cast<size_t>(index)) {
                    targets.push_back({request.uris(index), offset});
                    buffers.emplace_back();
                    owners.push_back(index);
                    staged_bytes += part->size_bytes;
                }
                const auto size = std::min<uint64_t>(iov.size, max_bytes - bytes);
                buffers.back().iovs.push_back({iov.type, iov.base, static_cast<size_t>(size), false});
                iov.base = static_cast<char*>(iov.base) + size;
                iov.size -= size;
                offset += size;
                bytes += size;
                ++iov_count;
            }
        }
    }
    flush();
    auto* mask = response.mutable_succeed_block()->mutable_bool_masks();
    for (const bool ok : success) {
        mask->add_values(ok);
    }
    response.set_transfer_status(timed_out ? REMOTE_TRANSFER_STATUS_TIMEOUT
                                 : std::find(success.begin(), success.end(), false) == success.end()
                                     ? REMOTE_TRANSFER_STATUS_SUCCESS
                                     : REMOTE_TRANSFER_STATUS_FAILED);
}

void Kvs2StorageBackend::setCompletionCleanup(std::shared_ptr<TransferCompletionCleanup> cleanup) {
    cleanup_ = std::move(cleanup);
}

std::vector<std::shared_ptr<FrozenTask>> Kvs2StorageBackend::frozenTasks() const {
    std::list<std::shared_ptr<FrozenTask>> retired;
    std::vector<std::shared_ptr<FrozenTask>> active;
    {
        std::lock_guard lock(frozen_mutex_);
        for (auto it = frozen_.begin(); it != frozen_.end();) {
            const auto current = it++;
            if ((*current)->snapshot().state == FrozenTask::State::RELEASED) {
                retired.splice(retired.end(), frozen_, current);
            }
        }
        active.assign(frozen_.begin(), frozen_.end());
    }
    return active;
}

TransferTaskRegistry::State Kvs2StorageBackend::queryCompletion(const std::string& trace) {
    return registry_.queryAndBlockMissing({trace}).front();
}

TransferTaskRegistry::ForgetResult Kvs2StorageBackend::forgetCompletion(const std::string& trace) {
    return registry_.forget(trace);
}

void Kvs2StorageBackend::cleanupExpiredMissing() {
    registry_.cleanupExpiredMissing();
}

}  // namespace rtp_llm
