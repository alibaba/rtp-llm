#include <torch/all.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <array>
#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);
}

namespace rtp_llm::test {
namespace {

torch::Tensor blockBytes(const KVCacheManager& manager, int layer, const std::string& tag, int block, bool scale) {
    const auto& group = manager.cacheConfig().group(tag);
    const auto addr = manager.convertIndexToAddr(layer, tag, block);
    const auto bytes = scale ? group.kvScaleStrideBytes() : group.kvBlockStrideBytes();
    auto* ptr = scale ? addr.kv_scale_addr : addr.kv_addr;
    RTP_LLM_CHECK_WITH_INFO(ptr != nullptr && bytes > 0, "CP completion fixture has no %s block plane",
                            scale ? "scale" : "payload");
    return torch::from_blob(ptr, {static_cast<int64_t>(bytes)},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

class NoDispatchProcessor final: public NormalBatchStreamProcessor {
public:
    NoDispatchProcessor(const ModelConfig& model, const CacheConfig& cache):
        NormalBatchStreamProcessor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache, false) {}

    absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& groups,
                                                    TensorHolder& holder) const override {
        auto result = NormalBatchStreamProcessor::gatherModelInput(groups, holder);
        if (result.ok() && !groups.contextStreams().empty()) {
            result->skip_lm_head = true;
        }
        return result;
    }

    absl::Status dispatchPrefillOnly(const StreamGroups&,
                                     GenerationPrefillCudaGraphStatus) const override {
        return absl::OkStatus();
    }
};

class DelayedWriterModel final: public ModelBase {
public:
    DelayedWriterModel(std::shared_ptr<KVCacheManager> manager,
                       GenerateStreamPtr stream,
                       py::function writer_barrier,
                       std::vector<std::pair<int, int>> required_copies,
                       int rank):
        manager_(std::move(manager)), stream_(std::move(stream)), writer_barrier_(std::move(writer_barrier)),
        required_copies_(std::move(required_copies)), rank_(rank) {}

    bool supportsSparseCacheSave() const override { return true; }

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        RTP_LLM_CHECK_WITH_INFO(inputs.cache_save_positions.defined() && inputs.cache_save_positions.numel() > 0,
                                "CP completion fixture lost sparse save positions");
        RTP_LLM_CHECK_WITH_INFO(inputs.cache_save_block_ids.defined() && inputs.cache_save_block_ids.numel() > 0,
                                "CP completion fixture lost sparse save destinations");
        RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_update_mapping.defined()
                                    && inputs.kv_cache_update_mapping.size(0) >= 1,
                                "CP completion fixture requires an owner copy mapping on each rank");
        const auto mapping = inputs.kv_cache_update_mapping.cpu().contiguous();
        const auto* rows = mapping.data_ptr<int32_t>();
        const auto& tags = inputs.kv_cache_group_tags;
        for (int64_t row = 0; row < mapping.size(0); ++row) {
            const int tag_row = rows[row * 3];
            const int src = rows[row * 3 + 1];
            const int dst = rows[row * 3 + 2];
            copy_rows_.push_back({tag_row, src, dst});
            RTP_LLM_CHECK_WITH_INFO(tag_row >= 0 && tag_row < static_cast<int>(tags.size()) && src != dst,
                                    "CP completion fixture has invalid A-to-D row");
            for (int layer = 0; layer < manager_->cacheConfig().layer_num; ++layer) {
                const auto& tag = tags[tag_row];
                // CP ranks use the same logical IDs but have different physical block contents.
                auto source = blockBytes(*manager_, layer, tag, src, false);
                auto copy = blockBytes(*manager_, layer, tag, dst, false);
                RTP_LLM_CHECK_WITH_INFO(torch::equal(source, copy),
                                        "rank %d consumed KV before local A-to-D copy", rank_);
                source_images_.push_back({layer, tag, src, source.clone()});
                if (manager_->cacheConfig().group(tag).kvScaleStrideBytes() > 0) {
                    RTP_LLM_CHECK_WITH_INFO(torch::equal(blockBytes(*manager_, layer, tag, src, true),
                                                         blockBytes(*manager_, layer, tag, dst, true)),
                                            "rank %d consumed scale before local A-to-D copy", rank_);
                }
            }
        }
        saw_copy_ = true;
        // G7 is locally full on rank 0 and partial on rank 1. Its required
        // A→C copy must still be broadcast to both. N9 belongs to the next
        // logical CP slot, so its independent D has no A→D history copy.
        for (const auto& [src, dst] : required_copies_) {
            const bool present = std::any_of(copy_rows_.begin(), copy_rows_.end(), [&](const auto& row) {
                return row[1] == src && row[2] == dst;
            });
            RTP_LLM_CHECK_WITH_INFO(present, "locally full rank 0 lost required A-to-C copy");
        }
        // Both ranks enter the same fixture barrier only after consuming their
        // own copied payload. Rank 0 then observes owner state on its executor
        // thread while rank 1 is still delayed in its writer.
        writer_barrier_();
        if (rank_ == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const auto& registration = *stream_->streamCacheResource().prefixCacheRegistration();
            RTP_LLM_CHECK_WITH_INFO(!registration.content_completed && !registration.source_pins.empty(),
                                    "owner released cache work before rank 1 writer");
            const auto& tail = registration.points.back();
            RTP_LLM_CHECK_WITH_INFO(tail.endpoint.pos == 9, "CP fixture lost frozen N=9");
            for (const auto& binding : tail.group_bindings) {
                RTP_LLM_CHECK_WITH_INFO(binding.state != PrefixCacheSaveState::CONTENT_COMPLETE,
                                        "owner marked N ready before rank 1 writer");
            }
            CacheKeysType path;
            for (const auto& point : registration.points) {
                path.push_back(point.endpoint.key);
            }
            const auto found = manager_->blockTreeCache()->tree()->findNode(path);
            RTP_LLM_CHECK_WITH_INFO(found.size() < path.size() || found.back()->token_end != 9,
                                    "owner published N before rank 1 writer");
            saw_pending_during_writer_ = true;
        }
        if (rank_ == 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(600));
        }
        const auto save_ids = inputs.cache_save_block_ids.cpu().contiguous();
        for (int64_t tag_row = 0; tag_row < save_ids.size(0); ++tag_row) {
            const auto& tag = tags[tag_row];
            for (int64_t col = 0; col < save_ids.size(2); ++col) {
                const int dst = save_ids.index({tag_row, 0, col}).item<int32_t>();
                if (dst < 0) continue;
                for (int layer = 0; layer < manager_->cacheConfig().layer_num; ++layer) {
                    blockBytes(*manager_, layer, tag, dst, false).fill_(200 + rank_);
                    if (manager_->cacheConfig().group(tag).kvScaleStrideBytes() > 0) {
                        blockBytes(*manager_, layer, tag, dst, true).fill_(220 + rank_);
                    }
                }
            }
        }
        saw_write_ = true;
        return {};
    }

    bool sawCopy() const { return saw_copy_; }
    bool sawWrite() const { return saw_write_; }
    bool sawPendingDuringWriter() const { return saw_pending_during_writer_; }
    const std::vector<std::array<int, 3>>& copyRows() const { return copy_rows_; }
    bool sourceUnchanged() const {
        for (const auto& image : source_images_) {
            if (!torch::equal(blockBytes(*manager_, image.layer, image.tag, image.block, false), image.bytes)) {
                return false;
            }
        }
        return true;
    }

private:
    struct SourceImage {
        int layer;
        std::string tag;
        int block;
        torch::Tensor bytes;
    };
    std::shared_ptr<KVCacheManager> manager_;
    GenerateStreamPtr stream_;
    py::function writer_barrier_;
    std::vector<std::pair<int, int>> required_copies_;
    int rank_;
    bool saw_copy_{false};
    bool saw_write_{false};
    bool saw_pending_during_writer_{false};
    std::vector<std::array<int, 3>> copy_rows_;
    std::vector<SourceImage> source_images_;
};

py::dict runCpCompletion(int rank, int world_rank, int world_size, py::function writer_barrier) {
    RTP_LLM_CHECK_WITH_INFO(rank == 0 || rank == 1, "CP completion fixture needs TP rank 0 or 1");
    RTP_LLM_CHECK_WITH_INFO(world_size == 2 || world_size == 4, "CP completion fixture needs one or two TP groups");
    RTP_LLM_CHECK_WITH_INFO(world_rank >= 0 && world_rank < world_size && world_rank % 2 == rank,
                            "CP completion fixture has inconsistent TP and world ranks");
    ModelConfig model;
    model.num_layers = 3;
    model.max_seq_len = 32;
    model.vocab_size = 16;
    model.input_vocab_size = 16;
    model.hidden_size = 2;
    model.attn_config.head_num = 2;
    model.attn_config.kv_head_num = 2;
    model.attn_config.size_per_head = 1;
    model.attn_config.tokens_per_block = 2;
    model.attn_config.kernel_tokens_per_block = 2;
    KVCacheSpecDesc desc{"default", KVCacheSpecType::MultiHeadAttention};
    desc.dtype = DataType::TYPE_INT8;
    desc.cp = CacheCpPolicyDesc{};
    desc.cp->mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    model.kv_cache_spec_descs.assign(3, {desc});

    ParallelismConfig parallelism;
    parallelism.tp_size = 2;
    parallelism.tp_rank = rank;
    parallelism.dp_size = world_size / 2;
    parallelism.dp_rank = world_rank / 2;
    parallelism.world_size = world_size;
    parallelism.world_rank = world_rank;
    parallelism.local_world_size = world_size;
    parallelism.local_rank = world_rank;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    KVCacheConfig kv_config;
    kv_config.reuse_cache = true;
    kv_config.enable_device_cache = true;
    kv_config.enable_prefill_partial_tail_reuse = true;
    PDSepConfig pd_config;
    pd_config.role_type = RoleType::PREFILL;
    auto config = CacheConfigCreator::createConfig(model, parallelism, kv_config);
    config.finalizeBlockNums(16, RuntimeConfig{});
    RTP_LLM_CHECK_WITH_INFO(config.request_projection_required, "CP completion fixture lacks CP projection");
    auto manager = std::make_shared<KVCacheManager>(config, false, nullptr, kv_config, parallelism,
                                                     RuntimeConfig{}, SpeculativeExecutionConfig{}, pd_config);
    RTP_LLM_CHECK_WITH_INFO(manager->init(), "CP completion fixture cache manager init failed");

    GenerateStreamPtr stream;
    std::vector<std::pair<int, int>> required_copies;
    if (rank == 0) {
        ResourceContext context;
        context.cache_manager = manager;
        context.reuse_cache = true;
        context.role_type = RoleType::PREFILL;
        context.ignore_request_cache_switches = true;
        context.enable_prefill_partial_tail_reuse = true;
        auto query = std::make_shared<GenerateInput>();
        query->input_ids = torch::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        stream = std::make_shared<NormalGenerateStream>(query, model, RuntimeConfig{}, context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        auto& resource = stream->streamCacheResource();
        resource.init(1);
        RTP_LLM_CHECK_WITH_INFO(resource.initKVBlock().ok(), "CP completion fixture stream allocation failed");
        const auto& tag = config.groupTags().front();
        const size_t slot = config.endpointBlockPosition(tag, 7);
        const auto& sets = manager->blockTreeCache()->groupSets();
        RTP_LLM_CHECK_WITH_INFO(!sets.empty(), "CP completion fixture has no GroupSet");
        for (size_t member = 0; member < sets.front()->devicePools().size(); ++member) {
            const auto src = resource.kvCache().blocks(member, tag).at(slot);
            sets.front()->devicePools()[member]->incTreeRef(src, BlockTreeRefType::CACHE);
        }
        stream->setInitialReuseLength(7);
        stream->setReuseLength(7);
        RTP_LLM_CHECK_WITH_INFO(resource.prepareRequiredContinuation().ok(),
                                "CP completion fixture continuation failed");
        RTP_LLM_CHECK_WITH_INFO(!resource.registeredSaveCopyMapping().empty(),
                                "CP completion fixture did not prepare owner copy IDs");
        for (const auto& copy : resource.registeredSaveCopyMapping()) {
            required_copies.emplace_back(copy.src, copy.dst);
        }
    }

    const auto& tag = config.groupTags().front();
    for (int block = 1; block < static_cast<int>(config.group(tag).block_num); ++block) {
        for (int layer = 0; layer < model.num_layers; ++layer) {
            blockBytes(*manager, layer, tag, block, false).fill_(block + rank * 17);
            if (config.group(tag).kvScaleStrideBytes() > 0) {
                blockBytes(*manager, layer, tag, block, true).fill_(block + rank * 17 + 64);
            }
        }
    }

    EngineInitParams params;
    params.model_config_ = model;
    params.parallelism_config = parallelism;
    params.kv_cache_config = kv_config;
    params.pd_sep_config = pd_config;
    params.py_model = py::none();
    NormalExecutor executor(params, manager);
    auto writer = std::make_unique<DelayedWriterModel>(manager, stream, std::move(writer_barrier),
                                                       std::move(required_copies), rank);
    auto* writer_ptr = writer.get();
    executor.setModel(std::move(writer));
    executor.setBatchProcessor(std::make_unique<NoDispatchProcessor>(model, config));
    const auto started = std::chrono::steady_clock::now();
    const auto status = executor.process(rank == 0 ? std::list<GenerateStreamPtr>{stream} :
                                            std::list<GenerateStreamPtr>{});
    RTP_LLM_CHECK_WITH_INFO(status.ok(), "CP completion fixture executor failed: %s",
                            status.ToString().c_str());
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    RTP_LLM_CHECK_WITH_INFO(writer_ptr->sawCopy() && writer_ptr->sawWrite(),
                            "CP completion fixture did not execute copy and writer");
    RTP_LLM_CHECK_WITH_INFO(writer_ptr->sourceUnchanged(),
                            "CP completion fixture modified an immutable source block");
    bool content_complete = false;
    bool source_pins_empty = false;
    bool tail_complete = false;
    bool tail_published = false;
    if (stream) {
        const auto& registration = *stream->streamCacheResource().prefixCacheRegistration();
        content_complete = registration.content_completed;
        source_pins_empty = registration.source_pins.empty();
        const auto& tail = registration.points.back();
        tail_complete = tail.endpoint.pos == 9
                        && std::all_of(tail.group_bindings.begin(), tail.group_bindings.end(), [](const auto& binding) {
                               return binding.state == PrefixCacheSaveState::CONTENT_COMPLETE;
                           });
        CacheKeysType path;
        for (const auto& point : registration.points) {
            path.push_back(point.endpoint.key);
        }
        const auto tail_key = tail.endpoint.key;
        // Match the product's terminal release path after the TP writer fence.
        stream->generate_status_->status = StreamState::FINISHED;
        stream->releaseResource();
        const auto found = manager->blockTreeCache()->tree()->findNode(path);
        tail_published = found.size() == path.size() && found.back()->token_end == 9
                         && manager->blockTreeCache()->tree()->isShortLeaf(found.back())
                         && std::all_of(found.back()->group_set_resources.begin(),
                                        found.back()->group_set_resources.end(), [](const auto& resource) {
                                            return resource.hasCompleteDeviceValue();
                                        });
        const auto full_keys = manager->blockTreeCache()->getKeySnapshot().keys;
        RTP_LLM_CHECK_WITH_INFO(std::find(full_keys.begin(), full_keys.end(), tail_key) == full_keys.end(),
                                "short N leaked into the full-key export");
    }
    py::dict result;
    result["rank"] = rank;
    result["world_rank"] = world_rank;
    result["world_size"] = world_size;
    result["elapsed_ms"] = elapsed_ms;
    result["content_complete"] = content_complete;
    result["source_pins_empty"] = source_pins_empty;
    result["tail_complete"] = tail_complete;
    result["tail_published"] = tail_published;
    result["pending_during_writer"] = writer_ptr->sawPendingDuringWriter();
    result["copy_rows"] = writer_ptr->copyRows();
    return result;
}

}  // namespace
}  // namespace rtp_llm::test

PYBIND11_MODULE(libth_tail_reuse_cp_completion_integration_test, m) {
    rtp_llm::registerExecCtxOps(m);
    m.def("run_cp_completion", &rtp_llm::test::runCpCompletion);
    m.def("confirm_tp_cache_work_completion", &rtp_llm::confirmTpCacheWorkCompletion);
}
