// EOS-only native GPU publication. No early payload copy or CPU snapshot.
// Metadata and payload transfer reuse the Prefill StorageBackend pipeline.

// Only the safety bound remains. The original StorageRequest/task owns this
// lease; no per-task ledger, phase callbacks, timers or sampler thread.
struct PublicationBudget: std::enable_shared_from_this<PublicationBudget> {
    struct Snapshot { size_t bytes, jobs; };
    struct Charge final: StorageCompletionAdmission {
        std::shared_ptr<PublicationBudget> owner;
        size_t bytes{0};
        ~Charge() override {
            if (!owner) return;
            std::lock_guard<std::mutex> lock(owner->mutex);
            owner->bytes -= bytes;
            --owner->jobs;
        }
    };
    explicit PublicationBudget(DecodePublicationLimits limits): limits(limits) {}
    std::shared_ptr<StorageCompletionAdmission> tryAcquire(size_t amount) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!accepting || amount > limits.bytes || bytes > limits.bytes - amount || jobs >= limits.jobs)
            return {};
        auto charge = std::make_shared<Charge>();
        charge->bytes = amount;
        charge->owner = shared_from_this();
        bytes += amount;
        ++jobs;
        return charge;
    }
    void stopAdmission() { std::lock_guard<std::mutex> lock(mutex); accepting = false; }
    Snapshot snapshot() const { std::lock_guard<std::mutex> lock(mutex); return {bytes, jobs}; }
    const DecodePublicationLimits limits;
    mutable std::mutex mutex;
    size_t bytes{0}, jobs{0};
    bool accepting{true};
};

bool configureDecodePublication(const CacheTopology&                                                topology,
                                const StorageBackend::BufferResolver&                               resolver,
                                const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver,
                                const kvcm::KVCMConfigPtr& custom_config) {
    if (!remoteCacheDecodePublicationConfigValid()) {
        if (parallelism_config_.role_type == RoleType::DECODE) {
            RTP_LLM_LOG_ERROR("ENABLE_DECODE_PUBLISH must be 0 or 1");
            return false;
        }
        RTP_LLM_LOG_WARNING("ENABLE_DECODE_PUBLISH is invalid; publication stays disabled on this non-Decode role");
    }
    is_decode_role_ = parallelism_config_.role_type == RoleType::DECODE;
    decode_publication_enabled_ = remoteCacheDecodePublicationEnabled() && is_decode_role_;
    // P/PDFUSION always use their original backend contract, independent of
    // the Decode switch. In particular,
    // do not validate Decode-only backend/layout/budget settings, retain its
    // resolver, or allocate publication accounting for an ordinary P cache.
    if (!decode_publication_enabled_)
        return true;
    std::string backend_type;
    try {
        // Use the same polymorphic parser as client creation. Inspect its wire
        // type without rewriting the caller's SDK options (including logging).
        autil::legacy::json::JsonArray backends;
        if (custom_config) {
            const auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                autil::legacy::ToJson(custom_config));
            const auto sdk = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(fields.at("sdk_config"));
            backends = autil::legacy::AnyCast<autil::legacy::json::JsonArray>(sdk.at("sdk_backend_configs"));
        } else {
            kvcm::SdkWrapperConfig sdk;
            sdk.parseBackendConfigs(kv_cache_config_.kvcm_model_sdk_config);
            backends = autil::legacy::AnyCast<autil::legacy::json::JsonArray>(
                autil::legacy::ToJson(sdk.sdk_backend_configs()));
        }
        if (backends.size() == 1) {
            const auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(backends.front());
            autil::legacy::FromJson(backend_type, fields.at("type"));
        }
    } catch (const autil::legacy::ExceptionBase& error) {
        RTP_LLM_LOG_ERROR("parse Decode publication SDK config failed: %s", error.what());
        return false;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("initialize Decode publication SDK config failed: %s", error.what());
        return false;
    }
    pace_backend_  = backend_type == "pace" || backend_type == "pace_ssd";
    const bool ssd = backend_type == "pace_ssd";
    if (pace_backend_
        && ((ssd && (kv_cache_config_.kvcm_read_backend_type != 9 || !gdr_enabled_))
            || (!ssd && kv_cache_config_.kvcm_read_backend_type == 9))) {
        RTP_LLM_LOG_ERROR("PACE SSD requires backend9/pace_ssd and GDR; DRAM cannot use backend9");
        return false;
    }
    const auto& cp          = parallelism_config_.prefill_cp_config;
    target_prefill_cp_      = cp.prefill_cp_size;
    target_prefill_tp_      = target_prefill_cp_;
    target_physical_shards_ = cp.kv_cache_sharded ? target_prefill_cp_ : 1;
    // CP is the target layout, DP is execution parallelism. They are independent.
    // Geometry checks remain in CPSlotMapper / group projection, never a CP2/4 whitelist.
    const auto& tags = cache_config_.groupTags();
    // Cache tags come from configure_v41_kv_cache; the L prefix belongs only
    // to remote location-spec names, not to the target/draft pool topology.
    const bool dsv41 = runtime_config_.model_name == "DeepSeekV41"
        && std::find(tags.begin(), tags.end(), "csa_state") != tags.end()
        && std::find(tags.begin(), tags.end(), "swa_kv") != tags.end();
    const bool supported_speculation = dsv41 && sp_config_.type == SP_TYPE_DSPARK
        && cache_config_.mtp_sub_configs.size() == 1 && cache_config_.mtp_sub_configs.front()
        && cache_config_.mtp_sub_configs.front()->layer_num > 0;
    if (!pace_backend_ || target_prefill_cp_ < 1
        || cache_config_.seq_size_per_block < 1
        || static_cast<size_t>(target_prefill_cp_)
               > static_cast<size_t>(std::numeric_limits<int>::max()) / cache_config_.seq_size_per_block
        || ((!cache_config_.mtp_sub_configs.empty() || sp_config_.type != SP_TYPE_NONE) && !supported_speculation)
        || (is_decode_role_ && (parallelism_config_.tp_size != 1 || parallelism_config_.dp_size < 1))) {
        RTP_LLM_LOG_ERROR(
            "Decode publication requires valid target CP, D TP1/positive DP, DSV4.1 complete target/draft endpoints, and native PACE: "
            "dsv41_tags=%d supported_speculation=%d sp_type=%d draft_modules=%zu role=%d tp=%ld dp=%ld target_cp=%d base=%zu pace=%d custom_client=%d",
            dsv41, supported_speculation, static_cast<int>(sp_config_.type), cache_config_.mtp_sub_configs.size(),
            static_cast<int>(parallelism_config_.role_type), parallelism_config_.tp_size, parallelism_config_.dp_size,
            target_prefill_cp_, cache_config_.seq_size_per_block, pace_backend_,
            !kv_cache_config_.kvcm_client_config.empty());
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
        RTP_LLM_LOG_INFO("DTOP_GPU budget native_pool_bytes=%zu pending_bytes_limit=%zu pending_jobs_limit=%zu",
                         native_pool_bytes, limits->bytes, limits->jobs);
        publication_budget_ = std::make_shared<PublicationBudget>(*limits);
        file_resolver_ = resolver;
    }
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
    return decode_publication_enabled_ && is_decode_role_ ? tag + "#cp" + std::to_string(peer) : tag;
}

// Select completed native checkpoints, not the final mutable slot. In
// particular, floor(E/unit) is only a candidate: all FULL pages and every
// required terminal STATE/SWA page must still be present at that exact key.
StorageRequest planCompletedDecode(const BatchKVCacheResource& resource, size_t computed_end,
                                   Clock::time_point deadline = Clock::time_point::max()) {
    StorageRequest request;
    if (!decode_publication_enabled_ || !is_decode_role_ || parallelism_config_.tp_rank != 0
        || completion_stopped_.load() || resource.decodeCompletionSubmitted())
        return request;
    if (Clock::now() >= deadline) return request;
    if (resource.batchSize() != 1 || computed_end == 0)
        return request;
    const size_t base           = cache_config_.seq_size_per_block;
    const size_t canonical_unit = base * target_physical_shards_;
    const size_t actual_computed_end = computed_end;
    const auto endpoint = resource.decode_endpoint;
    if (!endpoint && !resource.decode_round_endpoints.empty()) {
        RTP_LLM_LOG_WARNING("DTOP_GPU retained endpoints have no accepted latest endpoint");
        return request;
    }
    if (!cache_config_.mtp_sub_configs.empty()) {
        if (!endpoint || endpoint->end > computed_end) return request;
        computed_end = endpoint->end;
    }
    const auto& source_blocks = [&](const std::string& tag) -> const BlockIndicesType& {
        return endpoint ? endpoint->blocks.at(tag) : resource.blocks(0, tag);
    };
    const size_t completed_full = computed_end / base;
    const auto&  source_keys    = resource.cacheKeys(0);
    if (source_keys.size() < completed_full) {
        RTP_LLM_LOG_WARNING("DTOP_GPU incomplete key vector computed=%zu keys=%zu", computed_end, source_keys.size());
        return request;
    }
    // State/SWA slots must denote the same token boundary as the P key.
    size_t unit = canonical_unit;
    size_t full_prefix = completed_full;
    for (const auto& entry : group_policy_->groups()) {
        const auto& group = topology_->group(entry.second.tag);
        if (!entry.second.is_full)
            unit = std::lcm(unit, group.seqSizePerBlock());
        else if (!endpoint) {
            // Every lower ordinary candidate uses a subset of this FULL
            // prefix. Check it once, not again after each STATE/SWA backoff.
            const auto& blocks = source_blocks(entry.second.tag);
            const size_t limit = std::min(full_prefix, blocks.size());
            size_t prefix = 0;
            while (prefix < limit && !isNullBlockIdx(blocks[prefix])) ++prefix;
            full_prefix = prefix;
        }
    }
    CPSlotMapper mapper(0, target_physical_shards_, static_cast<int>(base));
    size_t       end = (endpoint ? computed_end : std::min(computed_end, full_prefix * base)) / unit * unit;
    if (endpoint && end != endpoint->end) return request;
    for (; end > 0; end -= unit) {
        const size_t count = end / canonical_unit;
        bool         ready = true;
        for (const auto& entry : group_policy_->groups()) {
            const auto& spec   = entry.second;
            const auto& group  = topology_->group(spec.tag);
            const auto& blocks = source_blocks(spec.tag);
            if (spec.is_full) {
                if (!endpoint) continue; // Proved above for this and every smaller candidate.
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
        if (ready) break;
        if (endpoint) return {}; // Never relabel one endpoint's STATE/SWA as a different H.
    }
    if (end == 0) {
        RTP_LLM_LOG_DEBUG("DTOP_GPU no_complete_checkpoint computed=%zu", computed_end);
        return request;
    }
    if (const auto& handoff = resource.prefillComputedEnd(); handoff && *handoff >= 0
                                                              && static_cast<uint64_t>(*handoff) <= actual_computed_end
                                                              && end <= static_cast<uint64_t>(*handoff)) {
        // A partial D tail (or checkpoint backoff) did not add a complete
        // checkpoint beyond P's explicit computed span. Skip optional work
        // before copying keys, charging/holding pages or fencing the device.
        // Legacy/unknown source metadata must retain the compatible path.
        RTP_LLM_LOG_DEBUG(
            "DTOP_GPU no_new_checkpoint computed=%zu published=%zu prefill=%ld", computed_end, end, *handoff);
        return request;
    }
    // Own the list itself, not an alias into a mutable/released stream. These
    // are native page references; no GPU payload is copied at EOS.
    std::vector<std::shared_ptr<const DecodeEndpoint>> retained_endpoints;
    if (endpoint) {
        retained_endpoints.push_back(endpoint);
        for (const auto& retained : resource.decode_round_endpoints) {
            if (!retained || retained->end > end) continue;
            if (!retained->end || retained->end % unit) {
                RTP_LLM_LOG_WARNING("DTOP_GPU invalid retained endpoint H=%zu", retained->end);
                return {};
            }
            if (std::find(retained_endpoints.begin(), retained_endpoints.end(), retained) == retained_endpoints.end())
                retained_endpoints.push_back(retained);
        }
    }
    const auto endpoints = std::make_shared<const std::vector<std::shared_ptr<const DecodeEndpoint>>>(
        std::move(retained_endpoints));
    if (!endpoints->empty()) request.source_owner = endpoints;
    CacheKeysType complete_keys(source_keys.begin(), source_keys.begin() + end / base);
    request.keys = std::make_shared<const CacheKeysType>(mapper.canonicalCacheKeys(complete_keys));
    request.handles.resize(request.keys->size());
    request.decode_completion = true;
    auto peer_blocks = std::make_shared<DecodePeerBlocks>();
    request.decode_peer_blocks = peer_blocks;
    size_t                                                                       bytes = 0;
    std::unordered_map<const DeviceBlockPool*, std::unordered_set<BlockIdxType>> charged_blocks;
    size_t                                                                       unique_pages = 0;
    // Group geometry, source tables and pools are immutable during planning.
    // Keep each key's group order while avoiding those lookups per key/peer.
    for (const auto& entry : group_policy_->groups()) {
        const auto& spec  = entry.second;
        const auto& group = topology_->group(spec.tag);
        BlockIndicesType state_blocks;
        if (!spec.is_full && !endpoints->empty()) {
            state_blocks.assign(request.keys->size(), NULL_BLOCK_IDX);
            for (const auto& retained : *endpoints) {
                const size_t count = retained->end / canonical_unit;
                const size_t first = count - std::min(count, group.reuseBlockCount(count));
                const auto source = retained->blocks.find(spec.tag);
                if (source == retained->blocks.end()) {
                    RTP_LLM_LOG_WARNING("DTOP_GPU missing retained group H=%zu tag=%s",
                                        retained->end, spec.tag.c_str());
                    return {};
                }
                const auto& blocks = source->second;
                for (size_t key = first; key < count; ++key) {
                    const size_t tokens = (key + 1) * canonical_unit;
                    if (tokens % group.seqSizePerBlock()
                        || tokens / group.seqSizePerBlock() - 1 >= blocks.size()) {
                        RTP_LLM_LOG_WARNING("DTOP_GPU incomplete retained group H=%zu tag=%s",
                                            retained->end, spec.tag.c_str());
                        return {};
                    }
                    const auto block = blocks[tokens / group.seqSizePerBlock() - 1];
                    auto& selected = state_blocks[key];
                    if (isNullBlockIdx(block) || (!isNullBlockIdx(selected) && selected != block)) {
                        RTP_LLM_LOG_WARNING("DTOP_GPU missing/conflicting retained source H=%zu key=%zu tag=%s",
                                            retained->end, key, spec.tag.c_str());
                        return {};
                    }
                    selected = block;
                }
            }
        }
        const size_t first = spec.is_full || !state_blocks.empty() ? 0 : request.keys->size()
            - std::min(request.keys->size(), group.reuseBlockCount(request.keys->size()));
        if (first == request.keys->size()) continue;
        const auto& blocks = source_blocks(spec.tag);
        const auto& pool = pools_by_tag_.at(spec.tag);
        auto& charged = charged_blocks[pool.get()];
        auto charge = [&](BlockIdxType block) {
            if (charged.insert(block).second) {
                const auto page_bytes = pool->blockSizeBytes();
                RTP_LLM_CHECK_WITH_INFO(page_bytes <= std::numeric_limits<size_t>::max() - bytes,
                                        "EOS source-page budget charge overflow");
                bytes += page_bytes;
                ++unique_pages;
            }
        };
        auto* projected = spec.is_full ? &(*peer_blocks)[spec.tag] : nullptr;
        if (projected) projected->resize(request.keys->size());
        for (size_t key = first; key < request.keys->size(); ++key) {
            BlockIdxType block;
            if (spec.is_full) {
                for (int peer = 0; peer < target_prefill_tp_; ++peer)
                    projected->at(key).push_back(
                        blocks.at(target_physical_shards_ * key + (target_physical_shards_ > 1 ? peer : 0)));
                block = projected->at(key).front();
            } else {
                block = state_blocks.empty() ?
                    blocks.at((key + 1) * canonical_unit / group.seqSizePerBlock() - 1) : state_blocks[key];
                if (isNullBlockIdx(block)) continue;
            }
            if (spec.is_full && target_physical_shards_ > 1)
                for (auto peer : projected->at(key)) charge(peer);
            else
                charge(block); // Replicated peers refer to this same physical page.
            // Same resolved-group append as P, without tree insertion or a
            // dense temporary GroupSetResource/tier-metadata matrix.
            BlockTreeStorer::appendDeviceGroup(request.handles[key], &spec.tag, &block, 1);
        }
    }
    // Production callers have already host-waited the writer: normal dispatch
    // waits sampler/worker dependencies; specUpdate waits the final draft event.
    // Release waits bookkeeping before freeing pages. Loads finish before Decode
    // is enqueued, and warmup returns before dispatch. Probe faults without
    // draining unrelated later GPU work; NotReady is not a quiescence proof.
    setCudaDevice();
#if USING_CUDA
    const auto writer_status = cudaStreamQuery(at::cuda::getCurrentCUDAStream().stream());
    if (writer_status != cudaSuccess && writer_status != cudaErrorNotReady)
        check_cuda_value(writer_status);
#endif
    if (Clock::now() >= deadline) return {};
    request.completion_admission = publication_budget_->tryAcquire(bytes);
    if (!request.completion_admission) {
        const auto pending = publication_budget_->snapshot();
        RTP_LLM_LOG_WARNING("DTOP_GPU admission=rejected retained_bytes=%zu pending_bytes=%zu pending_jobs=%zu",
                            bytes,
                            pending.bytes,
                            pending.jobs);
        return {};
    }
    RTP_LLM_LOG_DEBUG(
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
