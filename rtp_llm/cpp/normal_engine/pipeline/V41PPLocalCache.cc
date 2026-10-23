#include "rtp_llm/cpp/normal_engine/pipeline/V41PPLocalCache.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#if USING_CUDA
#include <c10/cuda/CUDAGraphsC10Utils.h>
#endif

namespace rtp_llm {

V41PPCacheBatch::Allocation::~Allocation() {
    if (resource) {
        manager->free(FreeInfo{resource, tokens, request_id});
    }
}

std::vector<KVCacheAllocationLeasePtr> V41PPCacheBatch::leases() const {
    std::vector<KVCacheAllocationLeasePtr> result;
    result.reserve(bindings_.size());
    for (const auto& binding : bindings_) {
        result.push_back(binding.lease);
    }
    return result;
}

torch::Tensor V41PPCacheBatch::kernelBlockTable() const {
    size_t columns = 0;
    for (const auto column : topology_->canonicalIndicesSnapshot()) {
        columns = std::max(columns, column + 1);
    }
    size_t pages = 1;
    for (const auto& binding : bindings_) {
        for (const auto& [tag, blocks] : binding.lease->blockTables()) {
            pages = std::max(pages, blocks.size());
        }
    }
    auto table = torch::zeros(
        {static_cast<int64_t>(bindings_.size()), static_cast<int64_t>(columns), static_cast<int64_t>(pages)},
        torch::TensorOptions(torch::kInt32));
    auto* values = table.data_ptr<int32_t>();
    for (size_t row = 0; row < bindings_.size(); ++row) {
        for (const auto& group : topology_->groups()) {
            const auto& blocks = bindings_[row].lease->blockTables().at(group.tag);
            std::copy(blocks.begin(), blocks.end(), values + (row * columns + group.canonical_idx) * pages);
        }
    }
    return table;
}

V41PPLocalCache::V41PPLocalCache(std::shared_ptr<KVCacheManager> manager,
                                 const ModelConfig&              model,
                                 const ParallelismConfig&        parallelism,
                                 bool                            allow_retained):
    manager_(std::move(manager)),
    borrow_root_(parallelism.pp_rank == 0 && parallelism.tp_rank == 0),
    max_initial_tokens_(std::min<int64_t>(model.max_seq_len, kV41MaxInitialPPPrefillTokens)),
    allow_retained_(allow_retained),
    max_retained_(parallelism.pp_size > 0 ? static_cast<size_t>(parallelism.pp_size) + 1 : 1) {
    if (!manager_ || model.model_type != "deepseek_v41" || model.num_layers != 40 || model.gen_num_per_cycle != 0
        || parallelism.pp_size != 2 || parallelism.dp_size != 1 || parallelism.tp_size != 4 || parallelism.ep_size != 4
        || parallelism.world_size != 8 || parallelism.pp_rank < 0 || parallelism.pp_rank > 1 || parallelism.tp_rank < 0
        || parallelism.tp_rank > 3 || parallelism.world_rank != parallelism.pp_rank * 4 + parallelism.tp_rank
        || parallelism.ep_rank != parallelism.tp_rank || parallelism.role_type != RoleType::PREFILL
        || !parallelism.local_cp_enabled() || parallelism.prefill_cp_config.kv_cache_sharded
        || !parallelism.pp_ep_enabled || parallelism.pp_ep_backend != "fork_nccl_mxfp8"
        || parallelism.pp_stage_layer_counts != std::vector<int64_t>{21, 19} || max_initial_tokens_ < 1) {
        throw std::invalid_argument("V4.1 local PP cache requires gen=0 CEP4PP2 PREFILL");
    }
    // The main-model layout projects its topology; leases use the actual
    // allocator topology. Keep this provenance rather than comparing views.
    topology_             = manager_->allLayerCacheBase().topologyPtr();
    const auto& config    = manager_->cacheConfig();
    const bool  stage_one = parallelism.pp_rank == 1;
    if (!config.use_independent_block_pools || config.layer_num != (stage_one ? 19u : 21u)
        || config.layer_all_num != config.layer_num || config.global_layer_begin != (stage_one ? 21u : 0u)
        || config.seq_size_per_block != 256 || config.kernel_seq_size_per_block != 256) {
        throw std::invalid_argument("V4.1 local PP cache requires native 21/19 allocation layouts");
    }
    std::map<std::string, std::pair<size_t, std::vector<int>>> expected;
    std::vector<int>                                           all_layers(config.layer_num);
    for (size_t layer = 0; layer < all_layers.size(); ++layer) {
        all_layers[layer] = static_cast<int>(layer);
    }
    expected.emplace("swa_kv", std::make_pair(67584, all_layers));
    expected.emplace("hca_kv", std::make_pair(73728, std::vector<int>{stage_one ? 0 : 20}));
    expected.emplace("indexer_kv",
                     std::make_pair(17408, stage_one ? std::vector<int>{0} : std::vector<int>{2, 8, 14, 20}));
    if (!stage_one) {
        expected.emplace("csa_kv", std::make_pair(36864, std::vector<int>{2, 8, 14}));
        expected.emplace("csa_state", std::make_pair(8192, std::vector<int>{2, 8, 14}));
    }
    if (topology_->groups().size() != expected.size() || topology_->layers().size() != config.layer_num) {
        throw std::invalid_argument("V4.1 local PP cache has an invalid group/layer roster");
    }
    std::set<size_t> columns;
    for (const auto& group : topology_->groups()) {
        const auto found   = expected.find(group.tag);
        const bool state   = group.tag == "swa_kv" || group.tag == "csa_state";
        const auto replica = stage_one && !state ? std::optional<uint32_t>(20) : std::nullopt;
        if (found == expected.end() || group.layer_ids != found->second.second
            || group.kv_block_stride_bytes != found->second.first || group.kv_scale_stride_bytes != 0
            || group.seq_size_per_block != 256 || group.kernel_seq_size_per_block != 256
            || group.read_replica_source_layer_id != replica || group.policy.single_block_per_request != state
            || (state && group.policy.enable_prefix_reuse)
            || group.policy.memory_placement != CacheMemoryPlacement::DEVICE
            || group.policy.cp_mapping != CpBlockMappingMode::NONE || group.policy.cp_slice != CpBlockSliceMode::NONE
            || group.canonical_idx >= 5 || !columns.insert(group.canonical_idx).second) {
            throw std::invalid_argument("V4.1 local PP cache has unqualified tagged allocation geometry");
        }
    }
}

void V41PPLocalCache::requireLease(const KVCacheAllocationLeasePtr& lease) const {
    if (!lease || lease->topologyPtr() != topology_ || lease->blockTables().size() != topology_->groups().size()) {
        throw std::invalid_argument("V4.1 local PP cache requires this rank's allocator provenance");
    }
    for (const auto& group : topology_->groups()) {
        const auto found     = lease->blockTables().find(group.tag);
        const auto max_pages = group.policy.single_block_per_request ? 1 : (max_initial_tokens_ + 255) / 256;
        if (found == lease->blockTables().end() || found->second.empty()
            || found->second.size() > static_cast<size_t>(max_pages)
            || (group.policy.single_block_per_request && found->second.size() != 1)) {
            throw std::invalid_argument("V4.1 local PP cache has invalid tagged page coverage");
        }
        std::set<BlockIdxType> unique;
        for (const auto page : found->second) {
            if (page <= 0 || static_cast<uint32_t>(page) >= group.block_num || !unique.insert(page).second) {
                throw std::invalid_argument("V4.1 local PP cache requires distinct live nonreserved pages per tag");
            }
        }
    }
}

std::shared_ptr<V41PPLocalCache::Allocation>
V41PPLocalCache::allocate(int64_t request_id, int64_t tokens, const KVCacheAllocationLeasePtr& root_lease) {
    if (local_generation_ == std::numeric_limits<int64_t>::max()) {
        throw std::overflow_error("V4.1 local PP allocation generations exhausted");
    }
    auto result              = std::make_shared<Allocation>();
    result->manager          = manager_;
    result->request_id       = request_id;
    result->local_generation = ++local_generation_;
    if (root_lease) {
        result->lease = root_lease;
    } else {
        auto input       = std::make_shared<GenerateInput>();
        tokens           = std::max<int64_t>(256, tokens);
        input->input_ids = torch::zeros({tokens}, torch::kInt32);
        result->tokens   = std::make_shared<CompleteTokenIds>(1, 1, tokens, tokens);
        result->tokens->init(input);
        result->tokens->setReserveStep(0);
        result->resource = std::make_shared<BatchKVCacheResource>();
        result->resource->resetBatchSize(1);
        result->resource->initGroups(topology_);
        MallocInfo info;
        info.batch_kv_cache_resource = result->resource;
        info.complete_token_ids      = result->tokens;
        info.request_id              = request_id;
        info.reuse_cache = info.enable_device_cache = info.enable_remove_skipped_blocks = false;
        const auto allocation                                                           = manager_->malloc(info);
        if (!allocation.success) {
            throw V41PPAllocationFailure("V4.1 local PP allocation failed with status "
                                         + std::to_string(static_cast<int>(allocation.status)));
        }
        result->lease = manager_->retainAllocation(result->resource->cacheResource(0));
    }
    result->allocated_tokens = tokens;
    requireLease(result->lease);
    return result;
}

void V41PPLocalCache::extend(const std::shared_ptr<Allocation>& allocation, int64_t tokens) {
    tokens = std::max<int64_t>(256, tokens);
    if (!allocation || !allocation->resource || allocation->allocated_tokens >= tokens) {
        return;
    }
    auto input       = std::make_shared<GenerateInput>();
    input->input_ids = torch::zeros({tokens}, torch::kInt32);
    auto grown       = std::make_shared<CompleteTokenIds>(1, 1, tokens, tokens);
    grown->init(input);
    grown->setReserveStep(0);
    MallocInfo info;
    info.batch_kv_cache_resource = allocation->resource;
    info.complete_token_ids      = grown;
    info.request_id              = allocation->request_id;
    info.reuse_cache = info.enable_device_cache = info.enable_remove_skipped_blocks = false;
    const auto result                                                               = manager_->malloc(info);
    if (!result.success) {
        throw V41PPAllocationFailure("V4.1 local PP allocation growth failed with status "
                                     + std::to_string(static_cast<int>(result.status)));
    }
    allocation->tokens           = grown;
    allocation->allocated_tokens = tokens;
    allocation->lease            = manager_->retainAllocation(allocation->resource->cacheResource(0));
    requireLease(allocation->lease);
}

V41PPCacheBatchPtr V41PPLocalCache::prepare(const torch::Tensor&                          spans,
                                            const std::vector<KVCacheAllocationLeasePtr>& root_leases) {
    pruneUncommitted();
    // Held for the whole call: the chained snapshot below aliases its request map.
    V41PPCacheBatchPtr retained;
    if (!uncommitted_.empty()) {
        if (!allow_retained_) {
            throw std::invalid_argument("V4.1 local PP cache already has a prepared dispatch");
        }
        // Asynchronous contract: preparation may follow a retained dispatch and the
        // retained set stays bounded by the pipeline slot ring. Whether the retained
        // dispatch already completed locally is an execution-lifecycle fact, so the
        // admission coordinator enforces it (requireRetainedLocallyComplete) rather
        // than the allocator.
        retained = uncommitted_.back().lock();
        if (uncommitted_.size() >= max_retained_) {
            throw std::invalid_argument("V4.1 local PP cache exceeded its retained dispatch ring bound");
        }
    }
    // Prospective state: chained from the retained dispatch when one exists, so a
    // retained batch's cursor progress is neither lost nor resurrected.
    const auto&   base            = retained ? retained->next_requests_ : requests_;
    const int64_t base_high_water = retained ? retained->source_high_water_ : source_high_water_;
    if (!spans.defined() || spans.device().type() != torch::kCPU || spans.scalar_type() != torch::kInt64
        || !spans.is_contiguous() || spans.dim() != 2 || spans.size(1) != 6 || spans.size(0) < 1 || spans.size(0) > 2
        || (borrow_root_ ? root_leases.size() != static_cast<size_t>(spans.size(0)) : !root_leases.empty())) {
        throw std::invalid_argument("V4.1 local PP cache requires one/two CPU int64 spans and root-only leases");
    }
    const auto*                                             values = spans.data_ptr<int64_t>();
    std::set<int64_t>                                       requests;
    std::set<int64_t>                                       generations;
    std::map<std::pair<std::string, BlockIdxType>, int64_t> page_owners;
    if (borrow_root_) {
        for (const auto& [id, request] : base) {
            for (const auto& [tag, pages] : request.allocation->lease->blockTables()) {
                for (const auto page : pages) {
                    page_owners.emplace(std::make_pair(tag, page), id);
                }
            }
        }
    }
    // Finish all dispatch and borrowed-ownership checks before new allocation.
    for (int64_t row = 0; row < spans.size(0); ++row) {
        const auto* s = values + row * 6;
        // Chunked (fast-gen) prefill continuations carry start>0 while remaining
        // context chunks. Bound them by the initial-prefill context like the
        // global admission does; the real guards here are the identity, chunk
        // and position checks below (fresh generation at token zero, then
        // next_chunk/next_position continuity) plus cumulative page coverage.
        const auto limit = max_initial_tokens_;
        if (s[0] < 0 || s[1] <= 0 || s[2] <= 0 || s[3] < 0 || s[4] < 0 || s[4] >= limit || s[5] <= 0
            || s[5] > limit - s[4] || !requests.insert(s[0]).second || !generations.insert(s[1]).second) {
            throw std::invalid_argument("V4.1 local PP cache has duplicate/invalid identities or an unqualified span");
        }
        const auto found = base.find(s[0]);
        if (found == base.end()) {
            if (s[1] <= base_high_water || s[3] != 0 || s[4] != 0) {
                throw std::invalid_argument("V4.1 local PP admission requires a fresh source generation at token zero");
            }
        } else if (found->second.source != std::array<int64_t, 3>{s[0], s[1], s[2]} || found->second.next_chunk != s[3]
                   || found->second.next_position != s[4]) {
            throw std::invalid_argument("V4.1 local PP cache rejects stale identities, replay and token gaps");
        }
        if (borrow_root_) {
            requireLease(root_leases[row]);
            for (const auto& group : topology_->groups()) {
                const auto& pages    = root_leases[row]->blockTables().at(group.tag);
                const auto  required = group.policy.single_block_per_request ? 1 : (s[4] + s[5] + 255) / 256;
                if (pages.size() < static_cast<size_t>(required)) {
                    throw std::invalid_argument("V4.1 local PP borrowed pages do not cover their request");
                }
                for (const auto page : pages) {
                    const auto inserted = page_owners.emplace(std::make_pair(group.tag, page), s[0]);
                    if (!inserted.second && inserted.first->second != s[0]) {
                        throw std::invalid_argument("V4.1 local PP borrowed pages alias another request");
                    }
                }
            }
            if (found != base.end()
                && found->second.allocation->lease->blockTables() != root_leases[row]->blockTables()) {
                throw std::invalid_argument("V4.1 local PP borrowed request changed its physical allocation");
            }
        }
    }
    auto batch                = std::make_shared<V41PPCacheBatch>();
    batch->domain_            = domain_;
    batch->topology_          = topology_;
    batch->next_requests_     = base;
    batch->source_high_water_ = base_high_water;
    batch->bindings_.reserve(spans.size(0));
    for (int64_t row = 0; row < spans.size(0); ++row) {
        std::array<int64_t, 6> s;
        std::copy_n(values + row * 6, 6, s.begin());
        auto       found = batch->next_requests_.find(s[0]);
        const bool fresh = found == batch->next_requests_.end();
        if (fresh) {
            // Size by the cumulative span end, not this chunk's count: they are
            // equal for a monolithic prefill (start == 0) and differ for every
            // FastGen continuation chunk.
            auto allocation = allocate(s[0], s[4] + s[5], borrow_root_ ? root_leases[row] : nullptr);
            found           = batch->next_requests_.emplace(s[0], Request{{s[0], s[1], s[2]}, 0, 0, allocation}).first;
            batch->source_high_water_ = std::max(batch->source_high_water_, s[1]);
        } else {
            extend(found->second.allocation, s[4] + s[5]);
        }
        found->second.next_chunk    = s[3] + 1;
        found->second.next_position = s[4] + s[5];
        batch->bindings_.push_back(
            {s, found->second.allocation->local_generation, found->second.allocation->lease, fresh});
    }
    trackUncommitted(batch);
    return batch;
}

bool V41PPLocalCache::ownsBatch(const V41PPCacheBatchPtr& batch) const {
    return batch && batch->domain_ == domain_ && !batch->bindings_.empty();
}

void V41PPLocalCache::requirePending(const V41PPCacheBatchPtr& batch) const {
    if (ownsBatch(batch) && !batch->committed_) {
        for (const auto& weak : uncommitted_) {
            if (weak.lock() == batch) {
                return;
            }
        }
    }
    throw std::invalid_argument("V4.1 local PP operation requires its current uncommitted dispatch");
}

V41PPLocalCache::InitializationOps V41PPLocalCache::prepareInitialization(const V41PPCacheBatchPtr& batch) const {
    requirePending(batch);
    if (!batch->worker_ready_ || batch->initialized_) {
        throw std::invalid_argument("V4.1 tagged initialization requires a new worker-ready dispatch");
    }
#if USING_CUDA
    if (c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None) {
        throw std::invalid_argument("V4.1 tagged initialization requires eager execution");
    }
#else
    throw std::invalid_argument("V4.1 tagged initialization requires CUDA");
#endif
    const auto layout = manager_->allLayerCacheBase();
    if (layout.topologyPtr() != topology_) {
        throw std::runtime_error("V4.1 tagged initialization lost allocator provenance");
    }
    std::map<std::string, std::set<int64_t>> pages;
    for (const auto& binding : batch->bindings_) {
        requireLease(binding.lease);
        if (binding.newly_admitted) {
            for (const auto& [tag, blocks] : binding.lease->blockTables()) {
                pages[tag].insert(blocks.begin(), blocks.end());
            }
        }
    }
    InitializationOps result;
    for (const auto& [tag, blocks] : pages) {
        const auto& group   = topology_->groupById(topology_->groupIdForTag(tag));
        auto        indices = torch::tensor(std::vector<int64_t>(blocks.begin(), blocks.end()), torch::kInt64);
        for (const auto layer_id : group.layer_ids) {
            const auto& view = layout.at(tag, layer_id);
            const auto& raw  = view.kv_addr;
            if (!raw.defined() || !raw.is_cuda() || raw.dim() < 2 || !raw.is_contiguous() || *blocks.begin() <= 0
                || *blocks.rbegin() >= raw.size(0)
                || raw.numel() * static_cast<int64_t>(raw.element_size())
                       != raw.size(0) * static_cast<int64_t>(group.kv_block_stride_bytes)
                || view.kv_scale_addr.defined()) {
                throw std::runtime_error("V4.1 tagged initialization requires complete native physical pages");
            }
            // These are physical per-tag pages, never another pool's IDs or
            // the subdivided attention-kernel table. Keep reserved page zero.
            result.emplace_back(raw, indices.to(raw.device()));
        }
    }
    return result;
}

void V41PPLocalCache::commit(const V41PPCacheBatchPtr& batch) {
    requirePending(batch);
    if (allow_retained_) {
        pruneUncommitted();
        if (uncommitted_.empty() || uncommitted_.front().lock() != batch) {
            // Chained snapshots must publish oldest-first. A newer snapshot is a
            // superset, so committing it first and the older one afterwards would
            // resurrect pre-chunk request state.
            throw std::invalid_argument("V4.1 local PP commits must publish retained dispatches in chunk order");
        }
    }
    requests_.swap(batch->next_requests_);
    source_high_water_ = batch->source_high_water_;
    batch->next_requests_.clear();
    batch->committed_ = true;
    untrackUncommitted(batch);
}

void V41PPLocalCache::abort(const V41PPCacheBatchPtr& batch) {
    requirePending(batch);
    if (allow_retained_) {
        pruneUncommitted();
        // Dispatches chained from this snapshot are unusable once it is aborted.
        // Require newest-first unwinding so none survives a poisoned prefix.
        if (uncommitted_.size() > 1 && uncommitted_.back().lock() != batch) {
            throw std::invalid_argument("V4.1 local PP abort must unwind retained dispatches newest first");
        }
    }
    batch->next_requests_.clear();
    batch->bindings_.clear();
    untrackUncommitted(batch);
}

void V41PPLocalCache::retire(const std::array<int64_t, 3>& source) {
    pruneUncommitted();
    if (!uncommitted_.empty()) {
        if (!allow_retained_) {
            throw std::invalid_argument("V4.1 local PP retirement requires a drained preparation");
        }
        // A retained dispatch still pins this request's pages and cursor progress;
        // it must commit or abort before the identity can be released.
        for (const auto& weak : uncommitted_) {
            const auto retained = weak.lock();
            if (!retained) {
                continue;
            }
            for (const auto& binding : retained->bindings_) {
                if (binding.span[0] == source[0]) {
                    throw std::invalid_argument(
                        "V4.1 local PP retirement requires its retained dispatch to commit or abort first");
                }
            }
        }
    }
    const auto found = requests_.find(source[0]);
    if (found == requests_.end() || found->second.source != source) {
        throw std::invalid_argument("V4.1 local PP retirement rejects stale source identities");
    }
    requests_.erase(found);
}

void V41PPLocalCache::pruneUncommitted() {
    std::vector<std::weak_ptr<V41PPCacheBatch>> live;
    live.reserve(uncommitted_.size());
    for (auto& weak : uncommitted_) {
        if (!weak.expired()) {
            live.push_back(std::move(weak));
        }
    }
    uncommitted_.swap(live);
}

void V41PPLocalCache::trackUncommitted(const V41PPCacheBatchPtr& batch) {
    uncommitted_.push_back(batch);
}

void V41PPLocalCache::untrackUncommitted(const V41PPCacheBatchPtr& batch) {
    for (auto entry = uncommitted_.begin(); entry != uncommitted_.end(); ++entry) {
        if (entry->lock() == batch) {
            uncommitted_.erase(entry);
            return;
        }
    }
}

size_t V41PPLocalCache::retainedLocal() {
    pruneUncommitted();
    return uncommitted_.size();
}

void V41PPLocalCache::requireRetainedLocallyComplete() const {
    for (const auto& weak : uncommitted_) {
        const auto retained = weak.lock();
        if (retained && !retained->execution_completed_) {
            throw std::invalid_argument(
                "V4.1 local PP cache requires its retained dispatch to complete locally before the next prepares");
        }
    }
}

}  // namespace rtp_llm
