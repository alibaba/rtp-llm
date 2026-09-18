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

#include <c10/core/InferenceMode.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;

namespace rtp_llm::test {
namespace {

constexpr int    kLayerId        = 0;
constexpr size_t kPhysicalBlocks = 8;

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
    explicit TestContextParallelProcessor(const ParallelismConfig& config):
        IContextParallelProcessor(config, /*split_hidden_states=*/true) {}

    size_t handleOutputs(torch::Tensor& hidden_states,
                         const GptModelInputs&,
                         const torch_ext::PyContextParallelParams&) override {
        return static_cast<size_t>(hidden_states.size(0));
    }

    void handleOutputsLastHidden(torch::Tensor&,
                                 const GptModelInputs&,
                                 const torch_ext::PyContextParallelParams&) override {}

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
};

Scenario makeMultiTagScenario() {
    // Keep topology order different from std::map order so the test catches
    // accidental group-index routing in place of stable tag routing.
    auto config = makeCacheConfig({{"linear", 1, 24}, {"full", 2, 16}});
    auto layout = makeLayout(config);
    auto inputs = makeInputs(/*input_lengths=*/{4},
                             /*request_ids=*/{101},
                             /*cache_keys=*/{1001, 1002, 1003, 1004},
                             /*cache_keys_width=*/4,
                             /*block_ids=*/{3, 4, 5, 6, 1, 2, -1, -1},
                             /*group_count=*/2,
                             /*block_table_width=*/4,
                             /*global_tokens_per_block=*/2,
                             /*global_stride_bytes=*/24);
    return {std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
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
    auto     config = makeCacheConfig({{"linear", 1, 24}, {"full", 2, 16}});
    auto     layout = makeLayout(config);
    auto     inputs = makeInputs(/*input_lengths=*/{6},
                             /*request_ids=*/{301},
                             /*cache_keys=*/{3101, 3102, 3103, 3104, 3105, 3106},
                             /*cache_keys_width=*/6,
                             /*block_ids=*/{3, 4, 5, 6, 7, 8, 1, 2, 3, -1, -1, -1},
                             /*group_count=*/2,
                             /*block_table_width=*/6,
                             /*global_tokens_per_block=*/1,
                             /*global_stride_bytes=*/24);
    Scenario scenario{std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
    scenario.parallelism.tp_size                            = 2;
    scenario.parallelism.tp_rank                            = 1;
    scenario.parallelism.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    scenario.parallelism.prefill_cp_config.kv_cache_sharded = false;
    scenario.replace_cp_processor                           = true;
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

Scenario makeMixedBatchScenario() {
    auto config = makeCacheConfig({{"full", 2, 16}, {"linear", 2, 24}});
    auto layout = makeLayout(config);
    auto inputs = makeInputs(/*input_lengths=*/{1, 1, 3},
                             /*request_ids=*/{0, 0, 501},
                             /*cache_keys=*/{0, 0, 0, 0, 5101, 5102},
                             /*cache_keys_width=*/2,
                             /*block_ids=*/{1, -1, 2, -1, 3, 4, 5, -1, 6, -1, 7, 0},
                             /*group_count=*/2,
                             /*block_table_width=*/2,
                             /*global_tokens_per_block=*/2,
                             /*global_stride_bytes=*/16);

    inputs.combo_tokens          = pinnedTensor({11, 12, 21, 22, 23}, {5});
    inputs.input_lengths         = pinnedTensor({8, 9, 3}, {3});
    inputs.sequence_lengths      = pinnedTensor({7, 8}, {2});
    inputs.prefix_lengths        = pinnedTensor({0}, {1});
    inputs.lm_output_indexes     = pinnedTensor({0, 1, 4}, {3});
    inputs.request_id            = pinnedLongTensor({501}, {1});
    inputs.request_pd_separation = pinnedBoolTensor(1, true);
    inputs.cache_keys            = pinnedLongTensor({5101, 5102}, {1, 2});
    inputs.combo_position_ids    = pinnedTensor(
        {100, 101, 102, 103, 104, 105, 106, 107, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211}, {20});
    inputs.text_tokens_mask = pinnedTensor({1, 1, 0, 0, 1}, {5});
    inputs.mm_features_locs = pinnedTensor({3}, {1});
    inputs.multimodal_features =
        std::vector<torch::Tensor>{torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat16))};
    inputs.trace_ids = {"decode-0", "decode-1", "context-0"};

    return {std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
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
    if (name == "mixed_batch") {
        return makeMixedBatchScenario();
    }
    throw std::invalid_argument("unknown PyWrappedModel cache-store integration scenario: " + name);
}

py::dict serializeResult(const RecordingCacheStore& store, const std::map<std::string, uintptr_t>& base_addresses) {
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
    return result;
}

void ensureTestRuntime() {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    });
}

py::dict runPyWrappedModelCacheStoreScenario(py::object          py_model,
                                             const std::string& scenario_name,
                                             bool               need_all_logits,
                                             bool               need_all_hidden_states) {
    ensureTestRuntime();
    auto scenario    = makeScenario(scenario_name);
    scenario.inputs.need_all_logits        = need_all_logits;
    scenario.inputs.need_all_hidden_states = need_all_hidden_states;
    auto cache_store = std::make_shared<RecordingCacheStore>();
    auto manager     = std::make_shared<KVCacheManager>(scenario.manager_config,
                                                    /*warmup=*/true,
                                                    /*metrics_reporter=*/nullptr,
                                                    KVCacheConfig{},
                                                    scenario.parallelism);
    manager->setCacheStore(cache_store);

    Weights weights;
    weights.layers.resize(1);
    if (scenario_name == "mixed_batch") {
        auto lm_head    = std::make_shared<DenseWeights>();
        lm_head->kernel = torch::ones(
            {1, 1}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        weights.lm_head = std::move(lm_head);
    }
    GptModelDescription description;
    description.data_type                    = DataType::TYPE_FP16;
    description.norm_type                    = NormType::rmsnorm;
    description.attention_conf.head_num      = 1;
    description.attention_conf.kv_head_num   = 1;
    description.attention_conf.size_per_head = 1;

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
                              /*hidden_size=*/1,
                              active_config.seq_size_per_block,
                              active_config.kernel_seq_size_per_block,
                              manager,
                              scenario.mtp_cache_config_index};

    GptModelOutputs outputs;
    bool            mixed_split_host_tensors_pinned = false;
    {
        PyWrappedModel model(params, std::move(py_model));
        if (scenario.replace_cp_processor) {
            model.context_parallel_processor_ = std::make_unique<TestContextParallelProcessor>(scenario.parallelism);
        }
        if (scenario_name == "mixed_batch") {
            const auto [decode, context]     = model.splitMixedInputs(scenario.inputs);
            const auto is_pinned_host_tensor = [](const torch::Tensor& tensor) {
                return !tensor.defined() || !tensor.device().is_cpu() || tensor.numel() == 0 || tensor.is_pinned();
            };
            mixed_split_host_tensors_pinned = is_pinned_host_tensor(decode.kv_cache_block_id)
                                              && is_pinned_host_tensor(decode.kv_cache_kernel_block_id)
                                              && is_pinned_host_tensor(context.kv_cache_block_id)
                                              && is_pinned_host_tensor(context.kv_cache_kernel_block_id)
                                              && is_pinned_host_tensor(context.lm_output_indexes)
                                              && is_pinned_host_tensor(context.mm_features_locs);
        }
        outputs = model.forward(scenario.inputs);
    }
    auto result = serializeResult(*cache_store, scenario.base_addresses);
    if (outputs.logits.defined()) {
        result["logits"] = outputs.logits.cpu();
    }
    if (outputs.hidden_states.defined()) {
        result["hidden_states"] = outputs.hidden_states.cpu();
    }
    if (outputs.all_hidden_states.defined()) {
        result["all_hidden_states"] = outputs.all_hidden_states.cpu();
    }
    if (outputs.all_logits.defined()) {
        result["all_logits"] = outputs.all_logits.cpu();
    }
    if (scenario_name == "mixed_batch") {
        result["mixed_split_host_tensors_pinned"] = mixed_split_host_tensors_pinned;
    }
    return result;
}

CacheConfig makeMropeGraphCacheConfig(const AttentionConfigs& attention) {
    ParallelismConfig parallelism;
    KVCacheSpecDesc desc;
    desc.tag        = "full";
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = DataType::TYPE_FP32;
    SpecBuildContext context;
    context.dtype                   = desc.dtype;
    context.seq_size_per_block       = 64;
    context.kernel_tokens_per_block  = 64;
    context.attn_config              = &attention;
    context.parallelism_config      = &parallelism;
    auto spec = MHAKVCacheSpec::build(desc, context);

    CacheConfig config;
    config.dtype                          = desc.dtype;
    config.layer_num                      = 1;
    config.layer_all_num                  = 1;
    config.block_num                      = kPhysicalBlocks;
    config.seq_size_per_block             = 64;
    config.kernel_seq_size_per_block      = 64;
    config.kv_block_stride_bytes          = spec->block_size_bytes();
    config.kv_block_size_bytes            = config.kv_block_stride_bytes;
    config.block_size_bytes               = config.kv_block_stride_bytes;
    config.use_independent_block_pools    = true;
    config.group_block_layout_initialized = true;

    GroupBase group;
    group.tag                       = desc.tag;
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.policy.explicit_block_num = kPhysicalBlocks;
    group.layer_ids                 = {kLayerId};
    group.block_num                 = kPhysicalBlocks;
    group.local_kv_head_num         = 1;
    group.seq_size_per_block        = 64;
    group.kernel_seq_size_per_block = 64;
    group.kv_block_stride_bytes     = config.kv_block_stride_bytes;
    LayerBase layer;
    layer.layer_id   = kLayerId;
    layer.group_tags = {desc.tag};
    config.setTopology({std::move(group)}, {std::move(layer)});
    return config;
}

py::list runMropeGraphPreparationScenario(py::object py_model) {
    // Match NormalEngine::loop/preRun across capture, preparation and replay.
    // Capture creates inference tensors that prepare updates in place.
    c10::InferenceMode inference_guard(true);
    ensureTestRuntime();
    Weights weights;
    weights.layers.resize(1);
    GptModelDescription description;
    description.data_type                        = DataType::TYPE_FP32;
    description.norm_type                        = NormType::rmsnorm;
    description.attention_conf.head_num          = 1;
    description.attention_conf.kv_head_num       = 1;
    description.attention_conf.size_per_head    = 4;
    description.attention_conf.rope_config.style = RopeStyle::Mrope;
    description.attention_conf.rope_config.index_factor = 3;
    auto config = makeMropeGraphCacheConfig(description.attention_conf);
    // Real MHA elements and dtype: [physical block, K/V, head, token, dim].
    // The opaque cache-store fixtures above intentionally use byte buffers;
    // those buffers do not satisfy KVCache::getLayerCache's MHA view contract.
    auto storage = torch::zeros({static_cast<int64_t>(kPhysicalBlocks), 2, 1, 64, 4},
                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    GroupedCacheLayerLayout::GroupLayouts group_layouts;
    group_layouts.emplace("full", CacheLayerLayout(std::vector<BlockBufferPtrInfo>{{storage, torch::Tensor()}}));
    GroupedCacheLayerLayout layout(config.topologyPtr(), std::move(group_layouts));
    auto manager = std::make_shared<KVCacheManager>(config, /*warmup=*/true);
    HWKernelConfig hw_config;
    hw_config.enable_cuda_graph = true;
    // MTP prefill captures multiples of gamma+1, i.e. [5,10] for B<=2.
    ConcurrencyConfig concurrency;
    concurrency.concurrency_limit = 2;
    RuntimeConfig runtime;
    runtime.fifo_scheduler_config.max_context_batch_size = 2;
    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_EAGLE;
    sp_config.model_type        = "qwen35_moe_mtp";
    sp_config.gen_num_per_cycle = 4;
    GptModelInitParams params{weights,
                              description,
                              layout,
                              /*model_id=*/1,
                              ParallelismConfig{},
                              hw_config,
                              ProfilingDebugLoggingConfig{},
                              runtime,
                              concurrency,
                              sp_config,
                              DeviceResourceConfig{},
                              MlaOpsType::AUTO,
                              /*max_seq_len=*/256,
                              /*hidden_size=*/4,
                              /*tokens_per_block=*/64,
                              /*kernel_tokens_per_block=*/64,
                              manager};
    PyWrappedModel model(params, py_model, /*is_prefill_cuda_graph_mode=*/true);
    auto* runner = dynamic_cast<CudaGraphRunner*>(model.graph_runner_);
    if (!runner) {
        throw std::runtime_error("mRoPE preparation regression requires the real CUDA graph runner");
    }

    ModelConfig model_config;
    model_config.max_seq_len                           = 256;
    model_config.vocab_size                            = 32;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = 2;
    model_config.attn_config.rope_config.index_factor  = 3;
    MtpBatchStreamProcessor processor(
        model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config, sp_config, /*warm_up=*/false);
    py::list results;
    const std::vector<std::vector<int32_t>> accept_lengths{{1, 3}, {3, 1}};
    for (size_t round = 0; round < accept_lengths.size(); ++round) {
        auto inputs = makeInputs({5, 5},
                                 {701, 702},
                                 {11, 12, 13, 14, 21, 22, 23, 24},
                                 4,
                                 {1, 2, 3, 4, 2, 3, 4, 5},
                                 1,
                                 4,
                                 64,
                                 config.kv_block_stride_bytes);
        inputs.pd_separation          = false;
        inputs.use_opaque_kv_cache_store = false;
        inputs.request_pd_separation = pinnedBoolTensor(2, false);
        inputs.prefix_lengths        = pinnedTensor({63, 127}, {2});
        std::vector<int32_t> positions;
        for (int request = 0; request < 2; ++request) {
            for (int step = 0; step < 5; ++step) {
                for (int axis = 0; axis < 3; ++axis) {
                    positions.push_back(static_cast<int32_t>(round * 1000) + request * 300 + axis * 70 + step);
                }
            }
        }
        inputs.combo_position_ids = pinnedTensor(positions, {30});
        GptModelOutputs target_output;
        target_output.all_hidden_states =
            torch::arange(40, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({10, 4});
        speculative::SpeculativeSamplerOutput rejection;
        rejection.accept_len    = pinnedTensor(accept_lengths[round], {2});
        rejection.accept_tokens = pinnedTensor({10, 11, 12, 13, 14, 20, 21, 22, 23, 24}, {2, 5});
        torch::Tensor compact_hidden;
        TensorHolder host_holder;
        processor.updateDecodePostDraftModelInput(inputs, target_output, rejection, 2, compact_hidden, host_holder);

        // The executor test covers when preparation is allowed. This fixture
        // follows its host-state contract: prepare only after real rejection.
        model.prepareAttentionInputs(inputs);
        py::dict result;
        result["prepared_before_forward"] = runner->prepared_attention_inputs_.load(std::memory_order_acquire);
        result["graph_key"]               = model.graph_state_.current_real_graph_seq_len;
        result["token_count"]             = inputs.combo_tokens.numel();
        auto& captured = runner->graph_instances_.at(model.graph_state_.current_real_graph_seq_len).mem_hold_.py_model_inputs_;
        // Single-group graph capture and setupKVCacheForAttentionInputs both
        // intentionally expose the direct PyAttentionInputs fast path.
        result["attention_group_count"] = layout.topology().groups().size();
        result["legacy_attention_inputs"] = captured.attention_inputs_by_tag.empty();
        result["capture_token_capacity"] = captured.input_ids.numel();
        result["capture_position_capacity"] = captured.combo_position_ids.numel();
        result["positions_before_forward"] = captured.combo_position_ids.narrow(0, 0, 12).reshape({4, 3}).cpu().clone();
        result["lengths_before_forward"] = captured.attention_inputs.input_lengths_device.cpu().clone();
        result["prefixes_before_forward"] = captured.attention_inputs.prefix_lengths_device.cpu().clone();

        // Exercise the second PyWrappedModel graph view separately: only a
        // block-table refresh occurs between preparation and actual replay.
        inputs.kv_cache_kernel_block_id = inputs.kv_cache_kernel_block_id.clone().pin_memory();
        inputs.kv_cache_kernel_block_id[0][0][0] = static_cast<int32_t>(round + 3);
        model.updateKVCacheKernelBlockId(inputs);
        result["block_before_forward"] =
            captured.attention_inputs.kv_cache_kernel_block_id_device[0][0].cpu().item<int32_t>();
        const int calls_before = py_model.attr("forward_calls").cast<int>();
        result["output"] = model.forward(inputs).all_hidden_states.cpu().clone();
        result["python_forward_delta"] = py_model.attr("forward_calls").cast<int>() - calls_before;
        result["prepared_after_forward"] = runner->prepared_attention_inputs_.load(std::memory_order_acquire);
        results.append(std::move(result));
    }
    return results;
}

}  // namespace
}  // namespace rtp_llm::test

PYBIND11_MODULE(libth_pywrapped_model_cache_store_integration_test, m) {
    torch_ext::registerPyOpDefs(m);
    m.def("run_scenario",
          &rtp_llm::test::runPyWrappedModelCacheStoreScenario,
          py::arg("py_model"),
          py::arg("scenario_name"),
          py::arg("need_all_logits") = false,
          py::arg("need_all_hidden_states") = false);
    m.def("run_mrope_graph_preparation", &rtp_llm::test::runMropeGraphPreparationScenario, py::arg("py_model"));
}
