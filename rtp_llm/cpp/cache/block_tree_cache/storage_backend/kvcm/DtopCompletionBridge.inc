// EOS-only native GPU publication. No early payload copy or CPU snapshot.
// Metadata and payload transfer reuse the Prefill StorageBackend pipeline.

bool configureDecodePublication(const CacheTopology&                                                topology,
                                const StorageBackend::BufferResolver&                               resolver,
                                const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver) {
    const char* feature = std::getenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH");
    if (feature && std::string_view(feature) != "0" && std::string_view(feature) != "1") {
        RTP_LLM_LOG_ERROR("ENABLE_DECODE_PUBLISH must be 0 or 1");
        return false;
    }
    decode_publication_enabled_ = remoteCacheDecodePublicationEnabled();
    // Off must preserve the upstream Prefill backend contract. In particular,
    // do not validate Decode-only backend/layout/budget settings, retain its
    // resolver, or allocate publication accounting for an ordinary P cache.
    if (!decode_publication_enabled_)
        return true;
    std::string backend_config = kv_cache_config_.kvcm_model_sdk_config;
    backend_config.erase(
        std::remove_if(backend_config.begin(), backend_config.end(), [](unsigned char c) { return std::isspace(c); }),
        backend_config.end());
    // The enabled path selects its backend from existing PACE configuration.
    pace_backend_  = backend_config == "[{\"type\":\"pace\"}]" || backend_config == "[{\"type\":\"pace_ssd\"}]";
    const bool ssd = backend_config == "[{\"type\":\"pace_ssd\"}]";
    if (pace_backend_
        && ((ssd && (kv_cache_config_.kvcm_read_backend_type != 9 || !gdr_enabled_))
            || (!ssd && kv_cache_config_.kvcm_read_backend_type == 9))) {
        RTP_LLM_LOG_ERROR("PACE SSD requires backend9/pace_ssd and GDR; DRAM cannot use backend9");
        return false;
    }
    is_decode_role_         = parallelism_config_.role_type == RoleType::DECODE;
    const bool  prefill     = parallelism_config_.role_type == RoleType::PREFILL;
    const auto& cp          = parallelism_config_.prefill_cp_config;
    target_prefill_cp_      = cp.prefill_cp_size;
    target_prefill_tp_      = target_prefill_cp_;
    target_physical_shards_ = cp.kv_cache_sharded ? target_prefill_cp_ : 1;
    // CP is the target layout, DP is execution parallelism. They are independent.
    // Geometry checks remain in CPSlotMapper / group projection, never a CP2/4 whitelist.
    if (!pace_backend_ || (!is_decode_role_ && !prefill) || target_prefill_cp_ < 1
        || cache_config_.seq_size_per_block < 1
        || static_cast<size_t>(target_prefill_cp_)
               > static_cast<size_t>(std::numeric_limits<int>::max()) / cache_config_.seq_size_per_block
        || !cache_config_.mtp_sub_configs.empty() || sp_config_.type != SP_TYPE_NONE
        || autil::EnvUtil::getEnv("RTP_LLM_STREAM_ASYNC", false) || !kv_cache_config_.kvcm_client_config.empty()
        || kv_cache_config_.kvcm_instance_id_salt.empty()
        || (pace_backend_ && backend_config != (ssd ? "[{\"type\":\"pace_ssd\"}]" : "[{\"type\":\"pace\"}]"))
        || (is_decode_role_ && (parallelism_config_.tp_size != 1 || parallelism_config_.dp_size < 1))
        || (prefill && (parallelism_config_.tp_size != target_prefill_cp_ || parallelism_config_.dp_size != 1))) {
        RTP_LLM_LOG_ERROR(
            "Decode publication requires matching target P CP layout, positive D DP, synchronous single-token decoding, and native PACE");
        return false;
    }
    if (!gdr_enabled_) {
        RTP_LLM_LOG_ERROR("Decode publication requires GDR; CPU payload staging is not an alternative");
        return false;
    }
    RTP_LLM_LOG_INFO("DTOP_GPU layout target_cp=%d target_tp=%d shards=%d visible_tokens=%zu dp=%ld role=%d",
                     target_prefill_cp_,
                     target_prefill_tp_,
                     target_physical_shards_,
                     cache_config_.seq_size_per_block * target_physical_shards_,
                     parallelism_config_.dp_size,
                     static_cast<int>(parallelism_config_.role_type));
    RTP_LLM_LOG_INFO("DTOP_TRANSPORT backend=%s end_only=1 target_cp=%d target_tp=%d",
                     pace_backend_ ? "pace" : "file",
                     target_prefill_cp_,
                     target_prefill_tp_);
    if (is_decode_role_) {
        size_t                                     native_pool_bytes = 0;
        std::unordered_set<const DeviceBlockPool*> unique_pools;
        for (const auto& group : topology.groups()) {
            if (!group.policy.enable_prefix_reuse)
                continue;
            const auto& pool = pool_resolver(group.tag);
            if (!pool || pool->where() != MemoryType::MEMORY_GPU || pool->getTotalSizeBytes() == 0) {
                RTP_LLM_LOG_ERROR("Decode publication has no native pool for tag=%s", group.tag.c_str());
                return false;
            }
            if (unique_pools.insert(pool.get()).second) {
                if (pool->getTotalSizeBytes() > std::numeric_limits<size_t>::max() - native_pool_bytes) {
                    RTP_LLM_LOG_ERROR("Decode publication native pool capacity overflow");
                    return false;
                }
                native_pool_bytes += pool->getTotalSizeBytes();
            }
        }
        const auto limits = remoteCacheDecodePublicationLimits(native_pool_bytes);
        if (!limits) {
            RTP_LLM_LOG_ERROR("invalid Decode publication pending byte/job budget");
            return false;
        }
        completion_limits_ = *limits;
        RTP_LLM_LOG_INFO("DTOP_GPU budget native_pool_bytes=%zu pending_bytes_limit=%zu pending_jobs_limit=%zu",
                         native_pool_bytes,
                         completion_limits_.bytes,
                         completion_limits_.jobs);
        kvcm::DecodePublicationAccounting::SnapshotObserver snapshots;
        kvcm::DecodePublicationAccounting::EventObserver    events;
        if (metrics_reporter_) {
            // Retain only the reporter, not Impl or its block pools: quarantined
            // admission can outlive the executor until backend shutdown.
            const auto reporter = metrics_reporter_;
            snapshots           = [reporter](const auto& value) {
                RtpLLMDecodePublicationStateMetricsCollector collector;
                collector.pending_bytes    = value.bytes;
                collector.pending_jobs     = value.jobs;
                collector.queued_jobs      = value.queued;
                collector.running_jobs     = value.running;
                collector.quarantined_jobs = value.quarantined;
                collector.byte_limit       = value.byte_limit;
                collector.job_limit        = value.job_limit;
                collector.oldest_age_us    = value.oldest_age_us;
                reporter->report<RtpLLMDecodePublicationMetrics, RtpLLMDecodePublicationStateMetricsCollector>(
                    nullptr, &collector);
            };
            events = [reporter](const char* outcome, std::optional<int64_t> latency) {
                RtpLLMDecodePublicationEventMetricsCollector collector;
                collector.outcome          = outcome;
                collector.eos_to_commit_us = latency;
                reporter->report<RtpLLMDecodePublicationMetrics, RtpLLMDecodePublicationEventMetricsCollector>(
                    nullptr, &collector);
            };
        }
        completion_accounting_ = std::make_shared<kvcm::DecodePublicationAccounting>(
            completion_limits_.bytes, completion_limits_.jobs, std::move(snapshots), std::move(events));
        file_resolver_ = resolver;
    }
    buffer_safe_failure_ = std::make_exception_ptr(StorageOperationBufferSafeFailure(
        "KVCM operation failed before submission or after completed payload"));
    completion_unknown_failure_ = std::make_exception_ptr(StorageOperationCompletionUnknown(
        "KVCM payload completion unknown", StorageQuarantineReason::INVALID_RESPONSE));
    completion_timeout_failure_ = std::make_exception_ptr(StorageOperationTimeout(
        "KVCM payload transfer timed out", StorageQuarantineReason::SDK_TIMEOUT));
    return true;
}

size_t publicationWireBytes(const std::string& tag) const {
    if (!decode_publication_enabled_ || !is_decode_role_ || target_physical_shards_ == 1) {
        return cache_config_.blockSizeBytesForGroup(tag);
    }
    const auto& group = cache_config_.topology().group(tag);
    if (group.policy.group_type == CacheGroupType::FULL) {
        RTP_LLM_CHECK_WITH_INFO(group.policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                "DTOP_GPU requires FULL page RR, tag=%s",
                                tag.c_str());
        return cache_config_.blockSizeBytesForGroup(tag);
    }
    RTP_LLM_CHECK_WITH_INFO(group.policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK
                                && group.seqSizePerBlock()
                                       == cache_config_.seq_size_per_block * target_physical_shards_,
                            "DTOP_GPU requires matching canonical state/SWA span, tag=%s",
                            tag.c_str());
    size_t bytes = 0;
    for (const int layer : cache_config_.layerIdsForGroup(tag)) {
        const auto& physical = cache_config_.physicalGroupForLayer(layer, tag);
        RTP_LLM_CHECK_WITH_INFO(
            physical.kvScaleStrideBytes() == 0, "DTOP_GPU requires packed state/SWA, tag=%s", tag.c_str());
        size_t payload = physical.kvBlockStrideBytes();
        if (group.policy.cp_slice == CpBlockSliceMode::PAYLOAD_BYTES) {
            payload = group.spec->k_block_payload_bytes();
        } else {
            RTP_LLM_CHECK_WITH_INFO(group.policy.cp_slice == CpBlockSliceMode::EQUAL_BYTES,
                                    "DTOP_GPU unsupported state slice, tag=%s",
                                    tag.c_str());
        }
        RTP_LLM_CHECK_WITH_INFO(payload > 0 && payload % target_physical_shards_ == 0,
                                "DTOP_GPU invalid state payload, tag=%s",
                                tag.c_str());
        bytes += payload / target_physical_shards_;
    }
    return bytes;
}

std::string publicationPeerTag(const std::string& tag, int peer) const {
    return is_decode_role_ ? tag + "#cp" + std::to_string(peer) : tag;
}

// Select completed native checkpoints, not the final mutable slot. In
// particular, floor(E/unit) is only a candidate: all FULL pages and every
// required terminal STATE/SWA page must still be present at that exact key.
StorageRequest planCompletedDecode(const BatchKVCacheResource& resource, size_t computed_end) {
    StorageRequest request;
    if (!decode_publication_enabled_ || !is_decode_role_ || parallelism_config_.tp_rank != 0
        || completion_stopped_.load() || resource.decodeCompletionSubmitted())
        return request;
    if (resource.batchSize() != 1 || computed_end == 0)
        return request;
    const auto&  lifecycle      = resource.decodeCompletionLifecycle();
    const size_t base           = cache_config_.seq_size_per_block;
    const size_t canonical_unit = base * target_physical_shards_;
    const size_t completed_full = computed_end / base;
    const auto&  source_keys    = resource.cacheKeys(0);
    if (source_keys.size() < completed_full) {
        RTP_LLM_LOG_WARNING("DTOP_GPU incomplete key vector computed=%zu keys=%zu", computed_end, source_keys.size());
        return request;
    }
    // State/SWA slots must denote the same token boundary as the P key.
    size_t unit = canonical_unit;
    for (const auto& entry : group_policy_->groups()) {
        const auto& group = topology_->group(entry.second.tag);
        if (!entry.second.is_full)
            unit = std::lcm(unit, group.seqSizePerBlock());
    }
    CPSlotMapper mapper(0, target_physical_shards_, static_cast<int>(base));
    size_t       end = computed_end / unit * unit;
    for (; end > 0; end -= unit) {
        const size_t count = end / canonical_unit;
        bool         ready = true;
        for (const auto& entry : group_policy_->groups()) {
            const auto& spec   = entry.second;
            const auto& group  = topology_->group(spec.tag);
            const auto& blocks = resource.blocks(0, spec.tag);
            if (spec.is_full) {
                for (size_t i = 0; i < end / base; ++i)
                    if (i >= blocks.size() || isNullBlockIdx(blocks[i])) {
                        ready = false;
                        break;
                    }
            } else {
                const size_t required = group.reuseBlockCount(count);
                for (size_t key = count - std::min(count, required); key < count; ++key) {
                    const size_t tokens = (key + 1) * canonical_unit;
                    if (tokens % group.seqSizePerBlock()) {
                        ready = false;
                        break;
                    }
                    const size_t index = tokens / group.seqSizePerBlock() - 1;
                    if (index >= blocks.size() || isNullBlockIdx(blocks[index])) {
                        ready = false;
                        break;
                    }
                }
            }
            if (!ready)
                break;
        }
        if (ready)
            break;
    }
    if (end == 0) {
        RTP_LLM_LOG_INFO("DTOP_GPU no_complete_checkpoint computed=%zu", computed_end);
        return request;
    }
    if (const auto& handoff = lifecycle.prefillComputedEnd(); handoff && *handoff >= 0
                                                              && static_cast<uint64_t>(*handoff) <= computed_end
                                                              && end <= static_cast<uint64_t>(*handoff)) {
        // A partial D tail (or checkpoint backoff) did not add a complete
        // checkpoint beyond P's explicit computed span. Skip optional work
        // before copying keys, charging/holding pages or fencing the device.
        // Legacy/unknown source metadata must retain the compatible path.
        RTP_LLM_LOG_DEBUG(
            "DTOP_GPU no_new_checkpoint computed=%zu published=%zu prefill=%ld", computed_end, end, *handoff);
        return request;
    }
    CacheKeysType complete_keys(source_keys.begin(), source_keys.begin() + end / base);
    request.keys = std::make_shared<const CacheKeysType>(mapper.canonicalCacheKeys(complete_keys));
    request.handles.resize(request.keys->size());
    request.decode_completion = true;
    request.computed_tokens   = computed_end;
    request.published_tokens  = end;
    if (const auto& handoff = lifecycle.prefillComputedEnd();
        handoff && *handoff >= 0 && static_cast<uint64_t>(*handoff) <= computed_end) {
        request.prefill_computed_tokens = *handoff;
    }
    size_t                                                                       bytes = 0;
    std::unordered_map<const DeviceBlockPool*, std::unordered_set<BlockIdxType>> charged_blocks;
    size_t                                                                       unique_pages = 0;
    for (size_t key = 0; key < request.keys->size(); ++key) {
        for (const auto& entry : group_policy_->groups()) {
            const auto& spec  = entry.second;
            const auto& group = topology_->group(spec.tag);
            if (!spec.is_full && request.keys->size() - key > group.reuseBlockCount(request.keys->size()))
                continue;
            const auto&        blocks = resource.blocks(0, spec.tag);
            StorageBlockHandle handle;
            handle.tag = spec.tag;
            if (spec.is_full) {
                for (int peer = 0; peer < target_prefill_tp_; ++peer)
                    handle.peer_blocks.push_back(
                        blocks.at(target_physical_shards_ * key + (target_physical_shards_ > 1 ? peer : 0)));
                handle.block = handle.peer_blocks.front();
            } else {
                handle.block = blocks.at((key + 1) * canonical_unit / group.seqSizePerBlock() - 1);
            }
            auto charge = [&](BlockIdxType block) {
                const auto& pool = pools_by_tag_.at(spec.tag);
                if (charged_blocks[pool.get()].insert(block).second) {
                    const auto page_bytes = pool->blockSizeBytes();
                    RTP_LLM_CHECK_WITH_INFO(page_bytes <= std::numeric_limits<size_t>::max() - bytes,
                                            "EOS source-page budget charge overflow");
                    bytes += page_bytes;
                    ++unique_pages;
                }
            };
            charge(handle.block);
            for (auto peer : handle.peer_blocks)
                charge(peer);
            request.handles[key].push_back(std::move(handle));
        }
    }
    request.completion_admission = completion_accounting_->tryAcquire(bytes, lifecycle.eosAt());
    if (!request.completion_admission) {
        if (lifecycle.recordFirstBudgetSkip())
            completion_accounting_->recordBudgetSkip();
        const auto pending = completion_accounting_->snapshot();
        RTP_LLM_LOG_WARNING("DTOP_GPU admission=rejected retained_bytes=%zu pending_bytes=%zu pending_jobs=%zu",
                            bytes,
                            pending.bytes,
                            pending.jobs);
        return {};
    }
    // One fence at successful EOS. No decode-step fence or CPU capture. The
    // task cannot outlive source ownership until prepareWrite has pinned it.
    setCudaDevice();
    check_cuda_value(cudaDeviceSynchronize());
    RTP_LLM_LOG_INFO(
        "DTOP_GPU retained_plan computed=%zu published=%zu tail=%zu pages=%zu bytes=%zu CPU_snapshot_bytes=0",
        computed_end,
        end,
        computed_end - end,
        unique_pages,
        bytes);
    return request;
}

// The P transaction builds identical spec/tag batches and commits actual URIs
// with the same masks and FinishWrite rules. Only the source projection differs:
// D owns all P peers' FULL pages and the unsliced native STATE/SWA page locally.
std::vector<FunctionResponsePB> transferCompletedDecode(const std::vector<FunctionRequestPB>& requests,
                                                       bool& transfer_attempted,
                                                       bool& source_may_be_touched) {
    setCudaDevice();
    using Batch = CompletionResources::Operation;
    std::vector<std::shared_ptr<Batch>> batches;
    batches.reserve(requests.size());
    std::vector<FunctionResponsePB> responses(requests.size());
    CPSlotMapper mapper(0, target_physical_shards_, cache_config_.seq_size_per_block);
    // Prebuild error payloads and all source descriptors before the first SDK
    // operation. A failure after any peer begins is conservatively unknown.
    const auto unknown = completion_unknown_failure_;
    const auto timeout = completion_timeout_failure_;
    for (size_t peer = 0; peer < requests.size(); ++peer) {
        const auto& remote = requests[peer].remote_request();
        (void)responses[peer].mutable_remote_response();
        RTP_LLM_CHECK(remote.block_ids_size() == remote.group_tags_size()
                      && remote.block_ids_size() == remote.uris_size());
        auto operation = std::make_shared<Batch>();
        operation->client = client_wrapper_;
        for (const auto& pool : pools_by_tag_) operation->pools.push_back(pool.second);
        auto& batch = *operation;
        for (int i = 0; i < remote.block_ids_size(); ++i) {
            const auto& tag = remote.group_tags(i);
            const int block = remote.block_ids(i);
            RTP_LLM_CHECK_WITH_INFO(remote.uris(i).compare(0, 7, "pace://") == 0,
                                    "Decode DRAM publication received a non-PACE destination");
            batch.tags.push_back(publicationPeerTag(tag, peer));
            batch.blocks.push_back(block);
            batch.uris.push_back(remote.uris(i));
            batch.buffers.emplace_back();
            size_t bytes = 0;
            for (int layer : cache_config_.layerIdsForGroup(tag)) {
                const auto parts = mapper.sliceBlockForPeer(
                    cache_config_, tag, file_resolver_(layer, tag, block), peer);
                for (const auto& part : parts) {
                    RTP_LLM_CHECK(part.is_cuda && part.addr && part.size_bytes);
                    batch.buffers.back().iovs.push_back(
                        {kv_cache_manager::MemoryType::GPU, part.addr, part.size_bytes, false});
                    RTP_LLM_CHECK(part.size_bytes <= std::numeric_limits<size_t>::max() - bytes);
                    bytes += part.size_bytes;
                }
            }
            RTP_LLM_CHECK_WITH_INFO(bytes == publicationWireBytes(tag), "Decode wire size mismatch tag=%s", tag.c_str());
        }
        batches.push_back(std::move(operation));
    }
    try {
        for (size_t peer = 0; peer < batches.size(); ++peer) {
            const auto operation = batches[peer];
            auto& batch = *operation;
            if (batch.tags.empty()) {
                continue;
            }
            auto& response = *responses[peer].mutable_remote_response();
            completion_resources_->holdOperation(operation);
            block_tree_cache_detail::ScopeRollback hold_unknown_descriptors([&] {
                completion_resources_->completeOperation(operation, source_may_be_touched);
                if (source_may_be_touched) {
                    completion_stopped_.store(true);
                    if (completion_accounting_) completion_accounting_->stopAdmission();
                }
            });
            const auto result = executeTagTransfers(
                REMOTE_OPERATION_WRITE, batch.tags, batch.blocks, batch.uris, batch.buffers, response,
                &source_may_be_touched, &transfer_attempted);
            if (result == TransferResult::TIMEOUT) {
                std::rethrow_exception(timeout);
            }
            if (result != TransferResult::SUCCESS) {
                std::rethrow_exception(source_may_be_touched ? unknown : buffer_safe_failure_);
            }
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_SUCCESS);
        }
    } catch (const StorageOperationCompletionUnknown&) {
        if (source_may_be_touched) throw;
        std::rethrow_exception(buffer_safe_failure_);
    } catch (...) {
        std::rethrow_exception(source_may_be_touched ? unknown : buffer_safe_failure_);
    }
    source_may_be_touched = false;
    return responses;
}
