#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "kmonitor/client/MetricType.h"
#include "kmonitor/client/core/MetricsRecord.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);
}

namespace rtp_llm::test {
namespace {

constexpr int    kLayerId        = 0;
constexpr size_t kPhysicalBlocks = 8;

struct MetricSnapshot {
    std::string          name;
    double               value;
    kmonitor::MetricType type;
};

std::optional<MetricSnapshot> readMetric(kmonitor::MutableMetric* mutable_metric, const kmonitor::MetricsTags& tags) {
    auto* metric = mutable_metric->DeclareMetric(&tags);
    if (metric == nullptr) {
        throw std::runtime_error("failed to declare hidden-state capture metric");
    }
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    metric->Snapshot(&record, /*period=*/1000);
    mutable_metric->UndeclareMetric(metric);
    if (record.Values().empty()) {
        return std::nullopt;
    }
    if (record.Values().size() != 1) {
        throw std::runtime_error("expected one hidden-state capture metric snapshot value");
    }
    return MetricSnapshot{
        record.Values().front()->Name(), std::stod(record.Values().front()->Value()), mutable_metric->GetMetricType()};
}

class FakeMetricsReporter {
public:
    FakeMetricsReporter(): reporter_(std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{})) {}

    const kmonitor::MetricsReporterPtr& reporter() const {
        return reporter_;
    }

    py::tuple snapshotHiddenStateCaptureMetrics() const {
        auto*    metrics = reporter_->getMetricsGroup<RtpLLMHiddenStateCaptureMetrics>();
        py::dict values;
        py::dict types;
        addMetric(values, types, metrics->batch_qps_metric);
        addMetric(values, types, metrics->publish_success_qps_metric);
        addMetric(values, types, metrics->failure_qps_metric);
        addMetric(values, types, metrics->initialization_failure_qps_metric);
        addMetric(values, types, metrics->layout_failure_qps_metric);
        addMetric(values, types, metrics->prepare_failure_qps_metric);
        addMetric(values, types, metrics->quantize_failure_qps_metric);
        addMetric(values, types, metrics->store_failure_qps_metric);
        addMetric(values, types, metrics->shutdown_failure_qps_metric);
        addMetric(values, types, metrics->hard_contract_failure_qps_metric);
        addMetric(values, types, metrics->request_error_failure_qps_metric);
        addMetric(values, types, metrics->operational_failure_qps_metric);
        addMetric(values, types, metrics->duplicate_request_id_qps_metric);
        addMetric(values, types, metrics->fail_open_disable_qps_metric);
        addMetric(values, types, metrics->disabled_skip_qps_metric);
        addMetric(values, types, metrics->broken_rejection_qps_metric);
        addMetric(values, types, metrics->bf16_publish_qps_metric);
        addMetric(values, types, metrics->fp8_publish_qps_metric);
        addMetric(values, types, metrics->publish_latency_us_metric);
        addMetric(values, types, metrics->quantize_latency_us_metric);
        addMetric(values, types, metrics->store_put_latency_us_metric);
        addMetric(values, types, metrics->publish_request_count_metric);
        addMetric(values, types, metrics->publish_token_count_metric);
        addMetric(values, types, metrics->publish_payload_bytes_metric);
        addMetric(values, types, metrics->publish_input_ids_bytes_metric);
        addMetric(values, types, metrics->publish_aux_hidden_bytes_metric);
        addMetric(values, types, metrics->publish_last_hidden_bytes_metric);
        addMetric(values, types, metrics->publish_scale_bytes_metric);
        addMetric(values, types, metrics->capture_enabled_metric);
        addMetric(values, types, metrics->capture_broken_metric);
        addMetric(values, types, metrics->fail_open_enabled_metric);
        return py::make_tuple(std::move(values), std::move(types));
    }

private:
    void addMetric(py::dict& values, py::dict& types, kmonitor::MutableMetric* metric) const {
        if (auto snapshot = readMetric(metric, reporter_->getTags()); snapshot.has_value()) {
            const auto name        = py::str(snapshot->name);
            values[name]           = snapshot->value;
            types[std::move(name)] = static_cast<unsigned int>(snapshot->type);
        }
    }

private:
    kmonitor::MetricsReporterPtr reporter_;
};

struct TestCacheSpec: public KVCacheSpec {
    TestCacheSpec(std::string cache_tag, size_t tokens_per_block, size_t bytes): bytes_(bytes) {
        tag                = std::move(cache_tag);
        seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
        type               = KVCacheSpecType::OpaqueState;
    }

    size_t block_size() const override {
        return bytes_;
    }
    size_t k_block_size() const override {
        return bytes_ / 2;
    }
    size_t v_block_size() const override {
        return bytes_ - k_block_size();
    }
    size_t block_size_bytes() const override {
        return bytes_;
    }
    size_t k_block_size_bytes() const override {
        return k_block_size();
    }
    size_t v_block_size_bytes() const override {
        return v_block_size();
    }
    DataType memoryLayoutDType() const override {
        return DataType::TYPE_INT8;
    }
    KVCacheSpecPtr clone() const override {
        return std::make_shared<TestCacheSpec>(*this);
    }
    std::string debugString(size_t = 0) const override {
        return "TestCacheSpec{" + tag + "}";
    }

private:
    size_t bytes_;
};

struct GroupSpec {
    std::string tag;
    size_t      tokens_per_block;
    size_t      stride_bytes;
};

CacheConfig makeCacheConfig(const std::vector<GroupSpec>& groups) {
    CacheConfig config;
    config.dtype                          = DataType::TYPE_INT8;
    config.layer_num                      = 1;
    config.layer_all_num                  = 1;
    config.block_num                      = kPhysicalBlocks;
    config.seq_size_per_block             = groups.front().tokens_per_block;
    config.kernel_seq_size_per_block      = groups.front().tokens_per_block;
    config.kv_block_stride_bytes          = groups.front().stride_bytes;
    config.use_independent_block_pools    = true;
    config.use_opaque_kv_cache_store      = true;
    config.group_block_layout_initialized = true;

    std::vector<GroupBase>   topology_groups;
    std::vector<std::string> layer_tags;
    topology_groups.reserve(groups.size());
    layer_tags.reserve(groups.size());
    for (const auto& spec : groups) {
        GroupBase group;
        group.tag    = spec.tag;
        group.spec   = std::make_shared<TestCacheSpec>(spec.tag, spec.tokens_per_block, spec.stride_bytes);
        group.policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
        group.policy.explicit_block_num = kPhysicalBlocks;
        group.layer_ids                 = {kLayerId};
        group.block_num                 = kPhysicalBlocks;
        group.seq_size_per_block        = spec.tokens_per_block;
        group.kernel_seq_size_per_block = spec.tokens_per_block;
        group.kv_block_stride_bytes     = spec.stride_bytes;
        topology_groups.push_back(std::move(group));
        layer_tags.push_back(spec.tag);
    }

    LayerBase layer;
    layer.layer_id   = kLayerId;
    layer.group_tags = std::move(layer_tags);
    config.setTopology(std::move(topology_groups), {std::move(layer)});
    return config;
}

struct LayoutAndBases {
    GroupedCacheLayerLayout          layout;
    std::map<std::string, uintptr_t> base_addresses;
};

LayoutAndBases makeLayout(const CacheConfig& config) {
    GroupedCacheLayerLayout::GroupLayouts layouts;
    std::map<std::string, uintptr_t>      bases;
    for (const auto& group : config.topology().groups()) {
        auto storage =
            torch::zeros({static_cast<int64_t>(kPhysicalBlocks), static_cast<int64_t>(group.kv_block_stride_bytes)},
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        bases.emplace(group.tag, reinterpret_cast<uintptr_t>(storage.data_ptr()));
        layouts.emplace(group.tag,
                        CacheLayerLayout(std::vector<BlockBufferPtrInfo>{{std::move(storage), torch::Tensor()}}));
    }
    return {GroupedCacheLayerLayout(config.topologyPtr(), std::move(layouts)), std::move(bases)};
}

class RecordingCacheStore: public CacheStore {
public:
    struct BlockRecord {
        std::string key;
        uintptr_t   address{0};
        uint32_t    length{0};
    };

    struct StoreRecord {
        std::string              request_id;
        std::vector<BlockRecord> blocks;
    };

    void store(const std::shared_ptr<RequestBlockBuffer>& buffer, CacheStoreStoreDoneCallback callback) override {
        StoreRecord record;
        record.request_id = buffer->getRequestId();
        for (const auto& [key, block] : buffer->getBlocks()) {
            record.blocks.push_back({key, reinterpret_cast<uintptr_t>(block->addr.get()), block->len});
        }
        std::sort(record.blocks.begin(), record.blocks.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.key < rhs.key;
        });
        {
            std::lock_guard<std::mutex> lock(mutex_);
            records_.push_back(std::move(record));
        }
        if (callback) {
            callback(true, CacheStoreErrorCode::None);
        }
    }

    void load(const std::shared_ptr<RequestBlockBuffer>&,
              CacheStoreLoadDoneCallback callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        callback(true, CacheStoreErrorCode::None);
    }

    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                             const std::string&,
                                             uint32_t,
                                             uint32_t,
                                             int64_t,
                                             LoadContext::CheckCancelFunc,
                                             int,
                                             int) override {
        return nullptr;
    }

    std::shared_ptr<StoreContext> storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                               int64_t) override {
        return nullptr;
    }

    std::shared_ptr<RemoteStoreTask>
    submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&,
                          const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>&,
                          RemoteStoreTask::CheckCancelFunc) override {
        return nullptr;
    }

    void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>&) override {}

    bool regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>&) override {
        return true;
    }

    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string&) override {
        return nullptr;
    }

    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override {
        return null_memory_util_;
    }

    void debugInfo() override {}

    std::vector<StoreRecord> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_;
    }

private:
    mutable std::mutex          mutex_;
    std::vector<StoreRecord>    records_;
    std::shared_ptr<MemoryUtil> null_memory_util_;
};

torch::Tensor pinnedTensor(const std::vector<int32_t>& values, at::IntArrayRef shape) {
    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<int32_t>(), values.data(), values.size() * sizeof(int32_t));
    }
    return tensor;
}

torch::Tensor pinnedLongTensor(const std::vector<int64_t>& values, at::IntArrayRef shape) {
    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<int64_t>(), values.data(), values.size() * sizeof(int64_t));
    }
    return tensor;
}

torch::Tensor pinnedBoolTensor(size_t size, bool value) {
    auto tensor =
        torch::empty({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
    std::fill_n(tensor.data_ptr<bool>(), size, value);
    return tensor;
}

GptModelInputs makeInputs(const std::vector<int32_t>& input_lengths,
                          const std::vector<int64_t>& request_ids,
                          const std::vector<int64_t>& cache_keys,
                          size_t                      cache_keys_width,
                          const std::vector<int32_t>& block_ids,
                          size_t                      group_count,
                          size_t                      block_table_width,
                          size_t                      global_tokens_per_block,
                          size_t                      global_stride_bytes) {
    const size_t batch_size = input_lengths.size();
    const size_t token_count =
        static_cast<size_t>(std::accumulate(input_lengths.begin(), input_lengths.end(), int32_t{0}));

    std::vector<int32_t> tokens(token_count);
    std::iota(tokens.begin(), tokens.end(), int32_t{1});
    std::vector<int32_t> output_lengths(batch_size, 1);
    std::vector<int32_t> output_indexes;
    output_indexes.reserve(batch_size);
    int32_t token_offset = 0;
    for (const int32_t length : input_lengths) {
        token_offset += length;
        output_indexes.push_back(token_offset - 1);
    }

    GptModelInputs inputs;
    inputs.combo_tokens      = pinnedTensor(tokens, {static_cast<int64_t>(token_count)});
    inputs.input_lengths     = pinnedTensor(input_lengths, {static_cast<int64_t>(batch_size)});
    inputs.sequence_lengths  = pinnedTensor({}, {0});
    inputs.lm_output_lengths = pinnedTensor(output_lengths, {static_cast<int64_t>(batch_size)});
    inputs.lm_output_indexes = pinnedTensor(output_indexes, {static_cast<int64_t>(batch_size)});
    inputs.prefix_lengths    = pinnedTensor(std::vector<int32_t>(batch_size, 0), {static_cast<int64_t>(batch_size)});
    inputs.kv_cache_block_id = pinnedTensor(
        block_ids,
        {static_cast<int64_t>(group_count), static_cast<int64_t>(batch_size), static_cast<int64_t>(block_table_width)});
    inputs.kv_cache_kernel_block_id = inputs.kv_cache_block_id.clone().pin_memory();
    inputs.request_id               = pinnedLongTensor(request_ids, {static_cast<int64_t>(batch_size)});
    inputs.request_pd_separation    = pinnedBoolTensor(batch_size, true);
    inputs.cache_keys =
        pinnedLongTensor(cache_keys, {static_cast<int64_t>(batch_size), static_cast<int64_t>(cache_keys_width)});
    inputs.seq_size_per_block        = global_tokens_per_block;
    inputs.kernel_seq_size_per_block = global_tokens_per_block;
    inputs.kv_block_stride_bytes     = global_stride_bytes;
    inputs.kv_scale_stride_bytes     = 0;
    inputs.pd_separation             = true;
    inputs.use_opaque_kv_cache_store = true;
    return inputs;
}

class TestContextParallelProcessor: public IContextParallelProcessor {
public:
    explicit TestContextParallelProcessor(const ParallelismConfig& config,
                                          bool                     report_invalid_num_valid_tokens = false,
                                          bool                     corrupt_restored_width          = false):
        IContextParallelProcessor(config, /*split_hidden_states=*/true),
        report_invalid_num_valid_tokens_(report_invalid_num_valid_tokens),
        corrupt_restored_width_(corrupt_restored_width) {}

    size_t handleOutputs(torch::Tensor& hidden_states,
                         const GptModelInputs&,
                         const torch_ext::PyContextParallelParams& cp_params) override {
        ++handle_outputs_call_count_;
        const auto expected_tokens = cp_params.prefill_actual_input_lengths_cpu.sum().item<int64_t>();
        const auto local_tokens    = hidden_states.size(0);
        RTP_LLM_CHECK_WITH_INFO(local_tokens > 0, "test CP restore requires at least one local hidden-state row");
        RTP_LLM_CHECK_WITH_INFO(expected_tokens <= local_tokens * parallelism_config_.tp_size,
                                "test CP restore cannot recover %ld tokens from %ld rows across %ld ranks",
                                expected_tokens,
                                local_tokens,
                                parallelism_config_.tp_size);
        RTP_LLM_CHECK_WITH_INFO(cp_params.prefill_qkv_restore_indice.numel() >= expected_tokens,
                                "test CP restore has only %ld indices for %ld tokens",
                                cp_params.prefill_qkv_restore_indice.numel(),
                                expected_tokens);

        // Simulate deterministic remote rank chunks for the position-aware test model,
        // then apply the same restore indices as the real CP processor. Each remote
        // rank advances the encoded input position by one local chunk.
        std::vector<torch::Tensor> rank_chunks;
        rank_chunks.reserve(static_cast<size_t>(parallelism_config_.tp_size));
        for (int64_t rank = 0; rank < parallelism_config_.tp_size; ++rank) {
            rank_chunks.push_back(hidden_states + rank * local_tokens);
        }
        auto all_hidden      = torch::cat(rank_chunks, 0);
        auto restore_indices = cp_params.prefill_qkv_restore_indice.narrow(0, 0, expected_tokens);
        hidden_states        = all_hidden.index_select(0, restore_indices);

        const auto restored_tokens = static_cast<size_t>(hidden_states.size(0));
        if (corrupt_restored_width_) {
            hidden_states = hidden_states.narrow(1, 0, hidden_states.size(1) - 1);
        }
        return report_invalid_num_valid_tokens_ ? restored_tokens - 1 : restored_tokens;
    }

    void handleOutputsLastHidden(torch::Tensor&,
                                 const GptModelInputs&,
                                 const torch_ext::PyContextParallelParams&) override {}

    size_t handleOutputsCallCount() const {
        return handle_outputs_call_count_;
    }

protected:
    bool plan(const std::vector<int>& total_input_tokens,
              std::vector<int>&       input_tokens,
              std::vector<int>&       shuffle_indices,
              int,
              int,
              int cp_chunk_size,
              int) override {
        for (int i = 0; i < cp_chunk_size; ++i) {
            if (i < static_cast<int>(total_input_tokens.size())) {
                input_tokens[static_cast<size_t>(i)]    = total_input_tokens[static_cast<size_t>(i)];
                shuffle_indices[static_cast<size_t>(i)] = i;
            } else {
                input_tokens[static_cast<size_t>(i)]    = 0;
                shuffle_indices[static_cast<size_t>(i)] = -1;
            }
        }
        return true;
    }

    torch::Tensor generateQKVRestoreIndices(const torch::Tensor& chunk_lengths, int cp_size) override {
        const auto count = chunk_lengths.sum().item<int64_t>() * cp_size;
        return torch::arange(count, torch::TensorOptions().dtype(torch::kInt32));
    }

    torch::Tensor
    generateQKVPaddingMask(const torch::Tensor& chunk_lengths, const torch::Tensor&, int cp_size) override {
        const auto count = chunk_lengths.sum().item<int64_t>() * cp_size;
        return torch::ones({count}, torch::TensorOptions().dtype(torch::kBool));
    }

private:
    bool   report_invalid_num_valid_tokens_{false};
    bool   corrupt_restored_width_{false};
    size_t handle_outputs_call_count_{0};
};

struct Scenario {
    CacheConfig                      manager_config;
    GroupedCacheLayerLayout          layout;
    std::map<std::string, uintptr_t> base_addresses;
    GptModelInputs                   inputs;
    ParallelismConfig                parallelism;
    DeviceResourceConfig             device_resources;
    size_t                           model_id{0};
    std::optional<int>               mtp_cache_config_index;
    bool                             replace_cp_processor{false};
    bool                             report_invalid_num_valid_tokens{false};
    bool                             corrupt_restored_width{false};
    std::vector<int64_t>             capture_layer_ids;
    HiddenStateCaptureDtype          capture_dtype{HiddenStateCaptureDtype::BF16};
    int64_t                          hidden_size{1};
    bool                             install_final_layernorm{false};
    bool                             install_lm_head{false};
    bool                             increment_request_ids_each_forward{false};
};

Scenario makeMultiTagScenario() {
    // Keep topology order different from std::map order so the test catches
    // accidental group-index routing in place of stable tag routing.
    auto     config = makeCacheConfig({{"linear", 1, 24}, {"full", 2, 16}});
    auto     layout = makeLayout(config);
    auto     inputs = makeInputs(/*input_lengths=*/{4},
                             /*request_ids=*/{101},
                             /*cache_keys=*/{1001, 1002, 1003, 1004},
                             /*cache_keys_width=*/4,
                             /*block_ids=*/{3, 4, 5, 6, 1, 2, -1, -1},
                             /*group_count=*/2,
                             /*block_table_width=*/4,
                             /*global_tokens_per_block=*/2,
                             /*global_stride_bytes=*/24);
    Scenario scenario{std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
    return scenario;
}

Scenario makeMicroBatchScenario() {
    auto     config = makeCacheConfig({{"default", 2, 16}});
    auto     layout = makeLayout(config);
    auto     inputs = makeInputs(/*input_lengths=*/{2, 4, 2},
                             /*request_ids=*/{201, 202, 203},
                             /*cache_keys=*/{2101, 0, 2201, 2202, 2301, 0},
                             /*cache_keys_width=*/2,
                             /*block_ids=*/{1, -1, 2, 3, 4, -1},
                             /*group_count=*/1,
                             /*block_table_width=*/2,
                             /*global_tokens_per_block=*/2,
                             /*global_stride_bytes=*/16);
    Scenario scenario{std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
    scenario.device_resources.enable_layer_micro_batch = static_cast<int>(MicroBatchType::DS_PREFILL);
    return scenario;
}

Scenario makeContextParallelScenario() {
    auto     config = makeCacheConfig({{"default", 2, 16}});
    auto     layout = makeLayout(config);
    auto     inputs = makeInputs(/*input_lengths=*/{6},
                             /*request_ids=*/{301},
                             /*cache_keys=*/{3101, 3102, 3103, 3104, 3105, 3106},
                             /*cache_keys_width=*/6,
                             /*block_ids=*/{1, 2, 3},
                             /*group_count=*/1,
                             /*block_table_width=*/3,
                             /*global_tokens_per_block=*/2,
                             /*global_stride_bytes=*/16);
    Scenario scenario{std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
    scenario.parallelism.tp_size                            = 2;
    scenario.parallelism.tp_rank                            = 1;
    scenario.parallelism.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    scenario.parallelism.prefill_cp_config.kv_cache_sharded = true;
    scenario.replace_cp_processor                           = true;
    return scenario;
}

Scenario makeCaptureScenario(bool micro_batch, bool context_parallel, bool publisher_owner, bool fp8) {
    auto config = makeCacheConfig({{"default", 2, 16}});
    auto layout = makeLayout(config);
    auto inputs =
        makeInputs(/*input_lengths=*/context_parallel ? std::vector<int32_t>{6} : std::vector<int32_t>{2, 3},
                   /*request_ids=*/context_parallel ? std::vector<int64_t>{601} : std::vector<int64_t>{501, 502},
                   /*cache_keys=*/context_parallel ? std::vector<int64_t>{0, 0, 0} : std::vector<int64_t>{0, 0, 0, 0},
                   /*cache_keys_width=*/context_parallel ? 3 : 2,
                   /*block_ids=*/context_parallel ? std::vector<int32_t>{1, 2, 3} : std::vector<int32_t>{1, -1, 2, 3},
                   /*group_count=*/1,
                   /*block_table_width=*/context_parallel ? 3 : 2,
                   /*global_tokens_per_block=*/2,
                   /*global_stride_bytes=*/16);
    Scenario scenario{std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
    scenario.inputs.skip_lm_head          = true;
    scenario.inputs.capture_hidden_states = true;
    scenario.inputs.pd_separation         = false;
    scenario.capture_layer_ids            = {0, 1};
    scenario.capture_dtype                = fp8 ? HiddenStateCaptureDtype::FP8_E4M3 : HiddenStateCaptureDtype::BF16;
    if (micro_batch) {
        scenario.device_resources.enable_layer_micro_batch = static_cast<int>(MicroBatchType::DS_PREFILL);
    }
    if (context_parallel) {
        scenario.parallelism.tp_size                  = 2;
        scenario.parallelism.tp_rank                  = 0;
        scenario.parallelism.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
        scenario.replace_cp_processor                 = true;
    } else if (!publisher_owner) {
        scenario.parallelism.tp_size = 2;
        scenario.parallelism.tp_rank = 1;
    }
    return scenario;
}

Scenario makeMicroBatchFinalNormParityScenario(bool capture_hidden_states) {
    auto scenario                    = makeCaptureScenario(true, false, true, false);
    scenario.hidden_size             = 2;
    scenario.install_final_layernorm = true;
    if (!capture_hidden_states) {
        scenario.inputs.capture_hidden_states = false;
        scenario.capture_layer_ids.clear();
    }
    return scenario;
}

Scenario makeDisabledMicroBatchScenario(int32_t token_count, bool capture_hidden_states) {
    auto scenario                         = makeCaptureScenario(true, false, true, false);
    scenario.inputs                       = makeInputs(/*input_lengths=*/{token_count},
                                 /*request_ids=*/{701},
                                 /*cache_keys=*/{0, 0},
                                 /*cache_keys_width=*/2,
                                 /*block_ids=*/{1, -1},
                                 /*group_count=*/1,
                                 /*block_table_width=*/2,
                                 /*global_tokens_per_block=*/2,
                                 /*global_stride_bytes=*/16);
    scenario.inputs.skip_lm_head          = true;
    scenario.inputs.capture_hidden_states = capture_hidden_states;
    scenario.inputs.pd_separation         = false;
    if (!capture_hidden_states) {
        scenario.capture_layer_ids.clear();
    }
    return scenario;
}

Scenario makeMtpScenario() {
    auto main_config  = makeCacheConfig({{"main", 4, 16}});
    auto draft_config = std::make_shared<CacheConfig>(makeCacheConfig({{"draft", 2, 32}}));
    auto layout       = makeLayout(*draft_config);
    main_config.mtp_sub_configs.push_back(draft_config);
    auto     inputs = makeInputs(/*input_lengths=*/{4},
                             /*request_ids=*/{401},
                             /*cache_keys=*/{4101, 4102},
                             /*cache_keys_width=*/2,
                             /*block_ids=*/{1, 2},
                             /*group_count=*/1,
                             /*block_table_width=*/2,
                             /*global_tokens_per_block=*/2,
                             /*global_stride_bytes=*/32);
    Scenario scenario{
        std::move(main_config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
    scenario.model_id               = 7;
    scenario.mtp_cache_config_index = 0;
    return scenario;
}

Scenario makeScenario(const std::string& name) {
    if (name == "multi_tag") {
        return makeMultiTagScenario();
    }
    if (name == "micro_batch") {
        return makeMicroBatchScenario();
    }
    if (name == "cp_actual_lengths") {
        return makeContextParallelScenario();
    }
    if (name == "mtp_sub_config") {
        return makeMtpScenario();
    }
    if (name == "prefill_only_pd") {
        auto scenario                = makeMultiTagScenario();
        scenario.inputs.skip_lm_head = true;
        return scenario;
    }
    if (name == "capture_bf16") {
        return makeCaptureScenario(false, false, true, false);
    }
    if (name == "capture_explicit_off") {
        auto scenario                         = makeCaptureScenario(false, false, true, false);
        scenario.inputs.capture_hidden_states = false;
        return scenario;
    }
    if (name == "capture_with_lm_head") {
        auto scenario                = makeCaptureScenario(false, false, true, false);
        scenario.inputs.skip_lm_head = false;
        scenario.install_lm_head     = true;
        return scenario;
    }
    if (name == "capture_async_history") {
        auto scenario                               = makeCaptureScenario(false, false, true, false);
        scenario.increment_request_ids_each_forward = true;
        return scenario;
    }

    if (name == "capture_non_4096_hidden") {
        auto scenario        = makeCaptureScenario(false, false, true, false);
        scenario.hidden_size = 2048;
        return scenario;
    }
    if (name == "capture_bf16_tp") {
        auto scenario                = makeCaptureScenario(false, false, true, false);
        scenario.parallelism.tp_size = 2;
        return scenario;
    }

    if (name == "capture_non_owner") {
        return makeCaptureScenario(false, false, false, false);
    }
    if (name == "capture_ffn_service") {
        auto scenario = makeCaptureScenario(false, false, true, false);
        scenario.parallelism.ffn_disaggregate_config.enable_ffn_disaggregate = true;
        scenario.parallelism.ffn_disaggregate_config.is_ffn_rank             = true;
        return scenario;
    }
    if (name == "capture_fp8") {
        return makeCaptureScenario(false, false, true, true);
    }
    if (name == "capture_fp8_tp") {
        auto scenario                = makeCaptureScenario(false, false, true, true);
        scenario.parallelism.tp_size = 2;
        return scenario;
    }
    if (name == "capture_micro_batch") {
        return makeCaptureScenario(true, false, true, false);
    }
    if (name == "micro_batch_final_norm_parity_capture_off") {
        return makeMicroBatchFinalNormParityScenario(false);
    }
    if (name == "micro_batch_final_norm_parity_capture_on") {
        return makeMicroBatchFinalNormParityScenario(true);
    }
    if (name == "capture_disabled_micro_batch_fake_lane") {
        return makeDisabledMicroBatchScenario(/*token_count=*/2, /*capture_hidden_states=*/true);
    }
    if (name == "disabled_micro_batch_zero_tokens") {
        return makeDisabledMicroBatchScenario(/*token_count=*/0, /*capture_hidden_states=*/false);
    }
    if (name == "capture_micro_batch_tp") {
        auto scenario                = makeCaptureScenario(true, false, true, false);
        scenario.parallelism.tp_size = 2;
        return scenario;
    }
    if (name == "capture_cp") {
        return makeCaptureScenario(false, true, true, false);
    }
    if (name == "capture_cp_micro_batch") {
        return makeCaptureScenario(true, true, true, false);
    }
    if (name == "capture_cp_invalid_num_valid_tokens") {
        auto scenario                            = makeCaptureScenario(false, true, true, false);
        scenario.report_invalid_num_valid_tokens = true;
        return scenario;
    }
    if (name == "capture_cp_malformed_layout") {
        return makeCaptureScenario(false, true, true, false);
    }
    if (name == "capture_cp_malformed_restored_width") {
        auto scenario                   = makeCaptureScenario(false, true, true, false);
        scenario.corrupt_restored_width = true;
        return scenario;
    }
    if (name == "capture_duplicate_layer_id") {
        auto scenario              = makeCaptureScenario(false, false, true, false);
        scenario.capture_layer_ids = {0, 0};
        return scenario;
    }
    if (name == "capture_invalid_dtype") {
        auto scenario          = makeCaptureScenario(false, false, true, false);
        scenario.capture_dtype = static_cast<HiddenStateCaptureDtype>(99);
        return scenario;
    }
    if (name == "capture_negative_request_id") {
        auto scenario              = makeCaptureScenario(false, false, true, false);
        scenario.inputs.request_id = pinnedLongTensor({-1, 502}, {2});
        return scenario;
    }
    if (name == "capture_duplicate_request_id") {
        auto scenario              = makeCaptureScenario(false, false, true, false);
        scenario.inputs.request_id = pinnedLongTensor({501, 501}, {2});
        return scenario;
    }
    if (name == "capture_prepare_failure_tp") {
        auto scenario                = makeCaptureScenario(false, false, true, false);
        scenario.parallelism.tp_size = 2;
        auto indices                 = torch::tensor({0, 1}, torch::kInt64).reshape({1, 2});
        auto values                  = torch::tensor({501, 502}, torch::kInt64);
        scenario.inputs.request_id   = torch::sparse_coo_tensor(indices, values, {2}).coalesce();
        return scenario;
    }
    if (name == "capture_prepare_failure_non_owner") {
        auto scenario              = makeCaptureScenario(false, false, false, false);
        auto indices               = torch::tensor({0, 1}, torch::kInt64).reshape({1, 2});
        auto values                = torch::tensor({501, 502}, torch::kInt64);
        scenario.inputs.request_id = torch::sparse_coo_tensor(indices, values, {2}).coalesce();
        return scenario;
    }

    throw std::invalid_argument("unknown PyWrappedModel cache-store integration scenario: " + name);
}

py::dict serializeResult(const RecordingCacheStore&              store,
                         const std::map<std::string, uintptr_t>& base_addresses,
                         const GptModelOutputs&                  output) {
    py::list records;
    auto     snapshot = store.snapshot();
    std::sort(snapshot.begin(), snapshot.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.request_id != rhs.request_id) {
            return lhs.request_id < rhs.request_id;
        }
        const auto lhs_key = lhs.blocks.empty() ? std::string() : lhs.blocks.front().key;
        const auto rhs_key = rhs.blocks.empty() ? std::string() : rhs.blocks.front().key;
        return lhs_key < rhs_key;
    });
    for (const auto& record : snapshot) {
        py::dict serialized_record;
        serialized_record["request_id"] = record.request_id;
        py::list blocks;
        for (const auto& block : record.blocks) {
            py::dict serialized_block;
            serialized_block["key"]     = block.key;
            serialized_block["address"] = py::int_(block.address);
            serialized_block["length"]  = block.length;
            blocks.append(std::move(serialized_block));
        }
        serialized_record["blocks"] = std::move(blocks);
        records.append(std::move(serialized_record));
    }

    py::dict bases;
    for (const auto& [tag, address] : base_addresses) {
        bases[py::str(tag)] = py::int_(address);
    }
    py::dict result;
    result["records"]        = std::move(records);
    result["base_addresses"] = std::move(bases);
    result["logits_defined"] = output.logits.defined();
    result["hidden_width"]   = output.all_hidden_states.defined() ? output.all_hidden_states.size(1) : 0;
    return result;
}

py::dict runPyWrappedModelCacheStoreScenario(py::object         py_model,
                                             const std::string& scenario_name,
                                             size_t             forward_count,
                                             bool               ignore_deferred_errors,
                                             bool               hidden_state_capture_fail_open,
                                             bool               capture_only_first_forward,
                                             bool               skip_first_deferred_error_take) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    });

    RTP_LLM_CHECK_WITH_INFO(forward_count > 0, "forward_count must be positive");

    auto                scenario    = makeScenario(scenario_name);
    auto                cache_store = std::make_shared<RecordingCacheStore>();
    FakeMetricsReporter fake_metrics_reporter;
    const auto&         metrics_reporter = fake_metrics_reporter.reporter();
    auto                manager          = std::make_shared<KVCacheManager>(scenario.manager_config,
                                                    /*warmup=*/true,
                                                    /*metrics_reporter=*/nullptr,
                                                    KVCacheConfig{},
                                                    scenario.parallelism);
    manager->setCacheStore(cache_store);

    Weights weights;
    weights.layers.resize(scenario.capture_layer_ids.empty() ? 1 : 2);
    if (scenario.install_final_layernorm) {
        auto final_layernorm = std::make_shared<LayerNormWeights>();
        final_layernorm->gamma =
            torch::tensor({0.5f, 1.5f}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        weights.final_layernorm = std::move(final_layernorm);
    }
    if (scenario.install_lm_head) {
        auto lm_head = std::make_shared<DenseWeights>();
        lm_head->kernel =
            torch::ones({2, scenario.hidden_size}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        weights.lm_head = std::move(lm_head);
    }
    GptModelDescription description;
    description.data_type                    = DataType::TYPE_FP16;
    description.norm_type                    = NormType::rmsnorm;
    description.attention_conf.head_num      = 1;
    description.attention_conf.kv_head_num   = 1;
    description.attention_conf.size_per_head = scenario.hidden_size;

    const auto&        active_config = scenario.mtp_cache_config_index.has_value() ?
                                           manager->getMTPModuleCacheConfig(*scenario.mtp_cache_config_index) :
                                           manager->cacheConfig();
    GptModelInitParams params{weights,
                              description,
                              scenario.layout,
                              scenario.model_id,
                              scenario.parallelism,
                              HWKernelConfig{},
                              ProfilingDebugLoggingConfig{},
                              RuntimeConfig{},
                              ConcurrencyConfig{},
                              SpeculativeExecutionConfig{},
                              scenario.device_resources,
                              MlaOpsType::AUTO,
                              /*max_seq_len=*/64,
                              scenario.hidden_size,
                              active_config.seq_size_per_block,
                              active_config.kernel_seq_size_per_block,
                              manager,
                              scenario.mtp_cache_config_index,
                              /*hc_mult=*/1,
                              scenario.capture_layer_ids,
                              scenario.capture_dtype,
                              hidden_state_capture_fail_open,
                              metrics_reporter};

    GptModelOutputs            output;
    HiddenStateCaptureStats    capture_stats;
    size_t                     cp_handle_outputs_calls{0};
    std::vector<torch::Tensor> forward_hidden_states;
    std::vector<std::string>   deferred_capture_errors;
    forward_hidden_states.reserve(forward_count);
    {
        PyWrappedModel                model(params, std::move(py_model));
        TestContextParallelProcessor* test_cp_processor = nullptr;
        if (scenario.replace_cp_processor) {
            auto processor = std::make_unique<TestContextParallelProcessor>(
                scenario.parallelism, scenario.report_invalid_num_valid_tokens, scenario.corrupt_restored_width);
            test_cp_processor                 = processor.get();
            model.context_parallel_processor_ = std::move(processor);
        }
        for (size_t i = 0; i < forward_count; ++i) {
            auto forward_inputs = scenario.inputs;
            if (capture_only_first_forward && i > 0) {
                forward_inputs.capture_hidden_states = false;
            }
            if (scenario.replace_cp_processor) {
                // CP planning updates host input lengths in place; keep repeated test forwards isolated.
                forward_inputs.input_lengths = scenario.inputs.input_lengths.clone().pin_memory();
            }
            if (scenario.increment_request_ids_each_forward) {
                forward_inputs.request_id = scenario.inputs.request_id.clone().pin_memory();
                forward_inputs.request_id.add_(static_cast<int64_t>(i) * 1000);
            }
            output = model.forward(forward_inputs);
            if (!(skip_first_deferred_error_take && i == 0)) {
                if (auto capture_error = model.takeDeferredHiddenStateCaptureError(); capture_error.has_value()) {
                    if (!ignore_deferred_errors) {
                        throw std::runtime_error(*capture_error);
                    }
                    deferred_capture_errors.push_back(*capture_error);
                }
            }
            if (output.all_hidden_states.defined()) {
                forward_hidden_states.push_back(output.all_hidden_states.detach().cpu());
            } else {
                forward_hidden_states.emplace_back();
            }
        }
        capture_stats = model.hiddenStateCaptureStats();
        if (test_cp_processor != nullptr) {
            cp_handle_outputs_calls = test_cp_processor->handleOutputsCallCount();
        }
    }

    auto result                    = serializeResult(*cache_store, scenario.base_addresses, output);
    auto capture_metrics           = fake_metrics_reporter.snapshotHiddenStateCaptureMetrics();
    result["capture_metrics"]      = capture_metrics[0];
    result["capture_metric_types"] = capture_metrics[1];
    py::list serialized_forward_hidden_states;
    for (const auto& hidden_states : forward_hidden_states) {
        if (hidden_states.defined()) {
            serialized_forward_hidden_states.append(hidden_states);
        } else {
            serialized_forward_hidden_states.append(py::none());
        }
    }
    result["forward_hidden_states"]          = std::move(serialized_forward_hidden_states);
    result["deferred_capture_errors"]        = std::move(deferred_capture_errors);
    result["capture_failure_count"]          = py::int_(capture_stats.failure_count);
    result["capture_broken_rejection_count"] = py::int_(capture_stats.broken_rejection_count);
    result["cp_handle_outputs_calls"]        = py::int_(cp_handle_outputs_calls);
    return result;
}

}  // namespace
}  // namespace rtp_llm::test

PYBIND11_MODULE(libth_pywrapped_model_cache_store_integration_test, m) {
    torch_ext::registerPyOpDefs(m);
    rtp_llm::registerExecCtxOps(m);
    m.def("run_scenario",
          &rtp_llm::test::runPyWrappedModelCacheStoreScenario,
          py::arg("py_model"),
          py::arg("scenario_name"),
          py::arg("forward_count")                  = 1,
          py::arg("ignore_deferred_errors")         = false,
          py::arg("hidden_state_capture_fail_open") = false,
          py::arg("capture_only_first_forward")     = false,
          py::arg("skip_first_deferred_error_take") = false);
}
