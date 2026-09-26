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
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;

namespace rtp_llm::test {

struct PyWrappedModelTestPeer {
    static auto splitInputsIntoMicroBatches(PyWrappedModel& model, const GptModelInputs& inputs) {
        return model.splitInputsIntoMicroBatches(inputs, model.planMicroBatches(inputs));
    }

    static void replaceContextParallelProcessor(PyWrappedModel&                            model,
                                                std::unique_ptr<IContextParallelProcessor> processor) {
        model.context_parallel_processor_ = std::move(processor);
    }

    static bool generationPrefillCudaGraphReady(const PyWrappedModel& model) {
        return model.generation_prefill_graph_runner_ != nullptr;
    }
};

namespace {

constexpr int    kLayerId        = 0;
constexpr size_t kPhysicalBlocks = 8;

struct TestCacheSpec: public KVCacheSpec {
    TestCacheSpec(std::string cache_tag, size_t tokens_per_block, size_t bytes):
        KVCacheSpec(
            std::move(cache_tag), static_cast<uint32_t>(tokens_per_block), static_cast<uint32_t>(tokens_per_block), 1),
        bytes_(bytes) {
        type = KVCacheSpecType::OpaqueState;
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
    config.dtype                     = DataType::TYPE_INT8;
    config.layer_num                 = 1;
    config.seq_size_per_block        = groups.front().tokens_per_block;
    config.use_opaque_kv_cache_store = true;

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
        group.block_num                 = kPhysicalBlocks;
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
        if (config.topology().layerIdsForGroup(group.tag).empty()) {
            layouts.emplace(group.tag, CacheLayerLayout(std::vector<BlockBufferPtrInfo>(config.layer_num)));
            continue;
        }
        auto storage =
            torch::zeros({static_cast<int64_t>(kPhysicalBlocks), static_cast<int64_t>(group.kvBlockStrideBytes())},
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

GptModelInputs makeInputs(const std::vector<int32_t>&     input_lengths,
                          const std::vector<int64_t>&     request_ids,
                          const std::vector<int64_t>&     cache_keys,
                          size_t                          cache_keys_width,
                          const std::vector<int32_t>&     block_ids,
                          const std::vector<std::string>& group_tags,
                          size_t                          block_table_width,
                          size_t                          global_tokens_per_block,
                          size_t                          global_stride_bytes) {
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
    inputs.kv_cache_group_tags = group_tags;
    inputs.combo_tokens        = pinnedTensor(tokens, {static_cast<int64_t>(token_count)});
    inputs.input_lengths       = pinnedTensor(input_lengths, {static_cast<int64_t>(batch_size)});
    inputs.sequence_lengths    = pinnedTensor({}, {0});
    inputs.lm_output_lengths   = pinnedTensor(output_lengths, {static_cast<int64_t>(batch_size)});
    inputs.lm_output_indexes   = pinnedTensor(output_indexes, {static_cast<int64_t>(batch_size)});
    inputs.prefix_lengths      = pinnedTensor(std::vector<int32_t>(batch_size, 0), {static_cast<int64_t>(batch_size)});
    inputs.kv_cache_block_id   = pinnedTensor(block_ids,
                                            {static_cast<int64_t>(group_tags.size()),
                                             static_cast<int64_t>(batch_size),
                                             static_cast<int64_t>(block_table_width)});
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
    explicit TestContextParallelProcessor(const ParallelismConfig& config): IContextParallelProcessor(config) {}

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
    // Payload rows deliberately differ from topology order.
    auto config = makeCacheConfig({{"linear", 1, 24}, {"full", 2, 16}});
    auto layout = makeLayout(config);
    auto inputs = makeInputs(/*input_lengths=*/{4},
                             /*request_ids=*/{101},
                             /*cache_keys=*/{1001, 1002, 1003, 1004},
                             /*cache_keys_width=*/4,
                             /*block_ids=*/{1, 2, -1, -1, 3, 4, 5, 6},
                             /*group_tags=*/{"full", "linear"},
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
                             /*group_tags=*/{"default"},
                             /*block_table_width=*/2,
                             /*global_tokens_per_block=*/2,
                             /*global_stride_bytes=*/16);
    Scenario scenario{std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
    scenario.device_resources.enable_layer_micro_batch = static_cast<int>(MicroBatchType::DS_PREFILL);
    return scenario;
}

Scenario makeContextParallelScenario(size_t tp_rank) {
    auto     config = makeCacheConfig({{"linear", 1, 24}, {"full", 2, 16}});
    auto     layout = makeLayout(config);
    auto     inputs = makeInputs(/*input_lengths=*/{6},
                             /*request_ids=*/{301},
                             /*cache_keys=*/{3101, 3102, 3103, 3104, 3105, 3106},
                             /*cache_keys_width=*/6,
                             /*block_ids=*/{3, 4, 5, 6, 7, 8, 1, 2, 3, -1, -1, -1},
                             /*group_tags=*/{"linear", "full"},
                             /*block_table_width=*/6,
                             /*global_tokens_per_block=*/1,
                             /*global_stride_bytes=*/24);
    Scenario scenario{std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
    scenario.parallelism.tp_size                            = 2;
    scenario.parallelism.tp_rank                            = tp_rank;
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
                             /*group_tags=*/{"draft"},
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
    if (name == "mtp_placeholders" || name == "mtp_placeholders_extra_payload" || name == "mtp_missing_placeholder"
        || name == "single_model_extra_payload") {
        const bool single_group = name == "single_model_extra_payload";
        auto       main_config  = makeCacheConfig({{"unused", 2, 16}, {"draft", 2, 32}, {"other", 2, 24}});
        auto       draft_config = std::make_shared<CacheConfig>(makeCacheConfig({{"draft", 2, 32}}));
        if (!single_group) {
            draft_config->setTopology(main_config.topology().groups(), {LayerBase{kLayerId, {"draft"}}});
        }
        auto layout = makeLayout(*draft_config);
        main_config.mtp_sub_configs.push_back(draft_config);
        auto inputs = makeInputs(/*input_lengths=*/{4},
                                 /*request_ids=*/{401},
                                 /*cache_keys=*/{4101, 4102},
                                 /*cache_keys_width=*/2,
                                 /*block_ids=*/{5, 6, 1, 2, 3, 4},
                                 /*group_tags=*/{"other", "draft", "unused"},
                                 /*block_table_width=*/2,
                                 /*global_tokens_per_block=*/2,
                                 /*global_stride_bytes=*/16);
        if (name == "mtp_placeholders_extra_payload") {
            inputs.kv_cache_group_tags.push_back("target_only");
            inputs.kv_cache_block_id =
                torch::cat({inputs.kv_cache_block_id, pinnedTensor({7, 0}, {1, 1, 2})}, 0).pin_memory();
        } else if (name == "mtp_missing_placeholder") {
            inputs.kv_cache_group_tags = {"other", "draft"};
            inputs.kv_cache_block_id   = inputs.kv_cache_block_id.narrow(0, 0, 2);
        }
        inputs.kv_cache_kernel_block_id = inputs.kv_cache_block_id.clone().pin_memory();
        Scenario scenario{
            std::move(main_config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
        scenario.model_id               = 7;
        scenario.mtp_cache_config_index = 0;
        return scenario;
    }
    if (name == "micro_batch_multi_tag") {
        auto config = makeCacheConfig({{"linear", 1, 24}, {"full", 2, 16}});
        auto layout = makeLayout(config);
        auto inputs = makeInputs(
            /*input_lengths=*/{2, 4, 2},
            /*request_ids=*/{201, 202, 203},
            /*cache_keys=*/{2101, 2102, 0, 0, 2201, 2202, 2203, 2204, 2301, 2302, 0, 0},
            /*cache_keys_width=*/4,
            /*block_ids=*/{1, -1, -1, -1, 2, 3, -1, -1, 4, -1, -1, -1, 2, 3, -1, -1, 4, 5, 6, 7, 1, 2, -1, -1},
            /*group_tags=*/{"full", "linear"},
            /*block_table_width=*/4,
            /*global_tokens_per_block=*/1,
            /*global_stride_bytes=*/24);
        Scenario scenario{
            std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
        scenario.device_resources.enable_layer_micro_batch = static_cast<int>(MicroBatchType::DS_PREFILL);
        return scenario;
    }
    if (name == "fake_micro_batch") {
        return makeMultiTagScenario();
    }
    if (name == "duplicate_tags" || name == "empty_tag" || name == "missing_tags" || name == "unknown_tag"
        || name == "physical_group_mismatch" || name == "type_group_mismatch" || name == "multi_group_2d") {
        auto scenario = makeMultiTagScenario();
        if (name == "duplicate_tags") {
            scenario.inputs.kv_cache_group_tags = {"full", "full"};
        } else if (name == "empty_tag") {
            scenario.inputs.kv_cache_group_tags = {"full", ""};
        } else if (name == "missing_tags") {
            scenario.inputs.kv_cache_group_tags.clear();
        } else if (name == "unknown_tag") {
            scenario.inputs.kv_cache_group_tags = {"full", "unknown"};
        } else if (name == "physical_group_mismatch") {
            scenario.inputs.kv_cache_block_id = scenario.inputs.kv_cache_block_id.narrow(0, 0, 1);
        } else if (name == "type_group_mismatch") {
            scenario.inputs.kv_cache_group_types = torch::zeros({1}, torch::kInt32);
        } else {
            scenario.inputs.kv_cache_block_id        = scenario.inputs.kv_cache_block_id[0];
            scenario.inputs.kv_cache_kernel_block_id = scenario.inputs.kv_cache_kernel_block_id[0];
        }
        return scenario;
    }
    if (name == "single_group_2d" || name == "single_group_2d_no_tags" || name == "single_group_2d_unknown_tag") {
        auto scenario                            = makeMtpScenario();
        scenario.inputs.kv_cache_block_id        = scenario.inputs.kv_cache_block_id.squeeze(0);
        scenario.inputs.kv_cache_kernel_block_id = scenario.inputs.kv_cache_kernel_block_id.squeeze(0);
        if (name == "single_group_2d_no_tags") {
            scenario.inputs.kv_cache_group_tags.clear();
        } else if (name == "single_group_2d_unknown_tag") {
            scenario.inputs.kv_cache_group_tags = {"unknown"};
        }
        return scenario;
    }
    if (name == "multi_tag") {
        return makeMultiTagScenario();
    }
    if (name == "micro_batch") {
        return makeMicroBatchScenario();
    }
    if (name == "cp_actual_lengths_rank0") {
        return makeContextParallelScenario(/*tp_rank=*/0);
    }
    if (name == "cp_actual_lengths_rank1") {
        return makeContextParallelScenario(/*tp_rank=*/1);
    }
    if (name == "mtp_sub_config") {
        return makeMtpScenario();
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

py::dict runPyWrappedModelCacheStoreScenario(py::object py_model, const std::string& scenario_name, bool enable_graph) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    });

    const bool cacheless_warmup = scenario_name == "cacheless_warmup";
    const bool inspect_split = scenario_name == "micro_batch_split_pinned" || scenario_name == "micro_batch_split_cuda"
                               || scenario_name == "micro_batch_split_single_group"
                               || scenario_name == "micro_batch_split_2d";
    const bool single_group_split =
        scenario_name == "micro_batch_split_single_group" || scenario_name == "micro_batch_split_2d";
    const auto input_scenario = inspect_split ? (single_group_split ? "micro_batch" : "micro_batch_multi_tag") :
                                                (cacheless_warmup ? "multi_tag" : scenario_name);
    auto       scenario       = makeScenario(input_scenario);
    if (inspect_split) {
        scenario.inputs.kv_cache_kernel_block_id.add_(100);
        scenario.inputs.kv_cache_kernel_block_id.masked_fill_(scenario.inputs.kv_cache_block_id.lt(0), -1);
        if (scenario_name == "micro_batch_split_cuda") {
            scenario.inputs.kv_cache_block_id        = scenario.inputs.kv_cache_block_id.cuda();
            scenario.inputs.kv_cache_kernel_block_id = scenario.inputs.kv_cache_kernel_block_id.cuda();
        } else if (scenario_name == "micro_batch_split_2d") {
            scenario.inputs.kv_cache_block_id        = scenario.inputs.kv_cache_block_id.squeeze(0);
            scenario.inputs.kv_cache_kernel_block_id = scenario.inputs.kv_cache_kernel_block_id.squeeze(0);
        }
    }
    auto cache_store = std::make_shared<RecordingCacheStore>();
    auto manager     = std::make_shared<KVCacheManager>(scenario.manager_config,
                                                    /*warmup=*/true,
                                                    /*metrics_reporter=*/nullptr,
                                                    KVCacheConfig{},
                                                    scenario.parallelism);
    manager->setCacheStore(cache_store);

    Weights weights;
    weights.layers.resize(1);
    GptModelDescription description;
    description.data_type                    = DataType::TYPE_FP16;
    description.norm_type                    = NormType::rmsnorm;
    description.attention_conf.head_num      = 1;
    description.attention_conf.kv_head_num   = 1;
    description.attention_conf.size_per_head = 1;

    GptModelInitParams params{weights,
                              description,
                              cacheless_warmup ? std::nullopt : std::make_optional(scenario.layout),
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
                              manager,
                              scenario.mtp_cache_config_index};

    if (cacheless_warmup) {
        params.cache_manager                     = nullptr;
        scenario.inputs.warmup                   = true;
        scenario.inputs.kv_cache_block_id        = torch::Tensor();
        scenario.inputs.kv_cache_kernel_block_id = torch::Tensor();
    }

    params.hw_kernel_config.enable_cuda_graph = enable_graph;
    params.hw_kernel_config.decode_capture_batch_sizes = {1};
    if (enable_graph && params.kv_cache_layer_layout.has_value()) {
        // This fixture bypasses Executor, which normally supplies the capture width.
        params.kernel_block_table_width = CudaGraphRunner::captureKernelBlockTableWidth(
            scenario.layout.topology(), params.max_seq_len, params.sp_config.speculativeReserveStep());
    }
    {
        PyWrappedModel model(params, std::move(py_model));
        if (inspect_split) {
            const auto split = PyWrappedModelTestPeer::splitInputsIntoMicroBatches(model, scenario.inputs);
            py::list   batches;
            for (const auto& inputs : split.first) {
                py::dict batch;
                batch["tags"]          = inputs.kv_cache_group_tags;
                batch["physical"]      = inputs.kv_cache_block_id;
                batch["kernel"]        = inputs.kv_cache_kernel_block_id;
                batch["input_lengths"] = inputs.input_lengths;
                batches.append(std::move(batch));
            }
            py::dict result;
            result["source_tags"]     = scenario.inputs.kv_cache_group_tags;
            result["source_physical"] = scenario.inputs.kv_cache_block_id;
            result["source_kernel"]   = scenario.inputs.kv_cache_kernel_block_id;
            result["batches"]         = std::move(batches);
            return result;
        }
        if (scenario_name == "fake_micro_batch") {
            const auto split = PyWrappedModelTestPeer::splitInputsIntoMicroBatches(model, scenario.inputs);
            py::dict   result;
            result["real_tags"]             = split.first.at(0).kv_cache_group_tags;
            result["fake_tags"]             = split.first.at(1).kv_cache_group_tags;
            result["fake_physical_defined"] = split.first.at(1).kv_cache_block_id.defined();
            result["fake_kernel_defined"]   = split.first.at(1).kv_cache_kernel_block_id.defined();
            return result;
        }
        if (scenario.replace_cp_processor) {
            PyWrappedModelTestPeer::replaceContextParallelProcessor(
                model, std::make_unique<TestContextParallelProcessor>(scenario.parallelism));
        }
        (void)model.forward(scenario.inputs);
    }
    return serializeResult(*cache_store, scenario.base_addresses);
}

py::dict runDirtyGenerationPrefillCaptureScenario(py::object py_model) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    });

    bool                          saw_dirty_capture_error = false;
    size_t                        available_before        = 0;
    size_t                        available_after         = 0;
    std::weak_ptr<KVCacheManager> manager_weak;
    {
        auto config  = makeCacheConfig({{"full", 4, 64}});
        auto layout  = makeLayout(config);
        auto manager = std::make_shared<KVCacheManager>(config,
                                                        /*warmup=*/true,
                                                        /*metrics_reporter=*/nullptr,
                                                        KVCacheConfig{},
                                                        ParallelismConfig{});
        RTP_LLM_CHECK_WITH_INFO(manager->init(), "dirty prefill capture test cache manager init failed");
        manager_weak     = manager;
        available_before = manager->availableBlocksNum();

        Weights weights;
        weights.layers.resize(1);
        GptModelDescription description;
        description.data_type                    = DataType::TYPE_BF16;
        description.norm_type                    = NormType::rmsnorm;
        description.attention_conf.head_num      = 1;
        description.attention_conf.kv_head_num   = 1;
        description.attention_conf.size_per_head = 4;

        HWKernelConfig hw_kernel_config;
        hw_kernel_config.enable_cuda_graph                          = true;
        hw_kernel_config.generation_prefill_cuda_graph_max_requests = 1;
        hw_kernel_config.generation_prefill_capture_token_buckets   = {4};
        hw_kernel_config.decode_capture_batch_sizes                 = {1};

        GptModelInitParams params{weights,
                                  description,
                                  layout.layout,
                                  /*model_id=*/0,
                                  ParallelismConfig{},
                                  hw_kernel_config,
                                  ProfilingDebugLoggingConfig{},
                                  RuntimeConfig{},
                                  ConcurrencyConfig{},
                                  SpeculativeExecutionConfig{},
                                  DeviceResourceConfig{},
                                  MlaOpsType::AUTO,
                                  /*max_seq_len=*/4,
                                  /*hidden_size=*/4,
                                  manager,
                                  /*mtp_cache_config_index=*/std::nullopt};
        params.kernel_block_table_width = 1;
        try {
            PyWrappedModel model(params, py_model);
        } catch (const DirtyCudaGraphCaptureError&) {
            saw_dirty_capture_error = true;
            available_after         = manager->availableBlocksNum();
        }
    }

    py::dict result;
    result["saw_dirty_capture_error"] = saw_dirty_capture_error;
    result["available_before"]        = available_before;
    result["available_after"]         = available_after;
    // A dirty graph runner remains process-local by design, but it no longer
    // owns an allocator request or the KVCacheManager itself.
    result["manager_retained"] = !manager_weak.expired();
    return result;
}

py::dict runGenerationPrefillCaptureScenario(py::object py_model, const std::string& expected_error_substring) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    });

    auto config  = makeCacheConfig({{"full", 4, 64}});
    auto layout  = makeLayout(config);
    auto manager = std::make_shared<KVCacheManager>(config,
                                                    /*warmup=*/true,
                                                    /*metrics_reporter=*/nullptr,
                                                    KVCacheConfig{},
                                                    ParallelismConfig{});
    RTP_LLM_CHECK_WITH_INFO(manager->init(), "clean prefill capture test cache manager init failed");
    const size_t available_before = manager->availableBlocksNum();

    Weights weights;
    weights.layers.resize(1);
    GptModelDescription description;
    description.data_type                    = DataType::TYPE_BF16;
    description.norm_type                    = NormType::rmsnorm;
    description.attention_conf.head_num      = 1;
    description.attention_conf.kv_head_num   = 1;
    description.attention_conf.size_per_head = 4;

    HWKernelConfig hw_kernel_config;
    hw_kernel_config.enable_cuda_graph                          = true;
    hw_kernel_config.generation_prefill_cuda_graph_max_requests = 1;
    hw_kernel_config.generation_prefill_capture_token_buckets   = {4};
    hw_kernel_config.decode_capture_batch_sizes                 = {1};

    GptModelInitParams params{weights,
                              description,
                              layout.layout,
                              /*model_id=*/0,
                              ParallelismConfig{},
                              hw_kernel_config,
                              ProfilingDebugLoggingConfig{},
                              RuntimeConfig{},
                              ConcurrencyConfig{},
                              SpeculativeExecutionConfig{},
                              DeviceResourceConfig{},
                              MlaOpsType::AUTO,
                              /*max_seq_len=*/4,
                              /*hidden_size=*/4,
                              manager,
                              /*mtp_cache_config_index=*/std::nullopt};
    params.kernel_block_table_width = 1;

    size_t      available_during  = 0;
    bool        graph_enabled     = false;
    bool        saw_capture_error = false;
    std::string capture_error_message;
    try {
        PyWrappedModel model(params, std::move(py_model));
        available_during = manager->availableBlocksNum();
        graph_enabled    = PyWrappedModelTestPeer::generationPrefillCudaGraphReady(model);
    } catch (const std::exception& e) {
        // Failure scenarios name the exact injected stage they expect. Any
        // other constructor exception must escape and fail the Python test
        // instead of being reduced to a generic boolean and producing a false
        // green cleanup result. The success scenario passes an empty string and
        // therefore accepts no exception at all.
        if (expected_error_substring.empty()
            || std::string(e.what()).find(expected_error_substring) == std::string::npos) {
            throw;
        }
        saw_capture_error     = true;
        capture_error_message = e.what();
        available_during      = manager->availableBlocksNum();
    }

    py::dict result;
    result["graph_enabled"]         = graph_enabled;
    result["saw_capture_error"]     = saw_capture_error;
    result["capture_error_message"] = capture_error_message;
    result["available_before"]      = available_before;
    result["available_during"]      = available_during;
    result["available_after"]       = manager->availableBlocksNum();
    return result;
}

// Exercise input preparation, Python forward, graphs and custom output together.
// Only decoder math is replaced; all routing and post-layers code is production.
py::dict runCustomOutput(py::object py_model, py::object handler, torch::Tensor indexes, bool python_norm) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() { initRuntime(0, false, false, MlaOpsType::AUTO); });
    constexpr int     width             = 4;
    constexpr int64_t block_table_width = 2;
    const bool        enable_graph      = python_norm;
    Weights           weights;
    weights.layers.resize(1);
    auto lm_head            = std::make_shared<DenseWeights>();
    lm_head->kernel         = torch::eye(width, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
    weights.lm_head         = lm_head;
    auto norm               = std::make_shared<LayerNormWeights>();
    norm->gamma             = torch::ones({width}, lm_head->kernel.options());
    weights.final_layernorm = norm;
    GptModelDescription description;
    description.data_type                    = DataType::TYPE_BF16;
    description.norm_type                    = NormType::rmsnorm;
    description.attention_conf.head_num      = 1;
    description.attention_conf.kv_head_num   = 1;
    description.attention_conf.size_per_head = width;
    auto               layout = std::make_optional(makeLayout(makeCacheConfig({{"full", 4, 64}})).layout);
    GptModelInitParams params{weights, description, layout};
    if (enable_graph) {
        params.cache_manager = std::make_shared<KVCacheManager>(makeCacheConfig({{"full", 4, 64}}),
                                                                /*warmup=*/true,
                                                                /*metrics_reporter=*/nullptr,
                                                                KVCacheConfig{},
                                                                ParallelismConfig{});
        TORCH_CHECK(params.cache_manager->init(), "custom output test cache manager init failed");
        params.max_seq_len                                                 = 8;
        params.hidden_size                                                 = width;
        params.kernel_block_table_width                                    = block_table_width;
        params.hw_kernel_config.enable_cuda_graph                          = true;
        params.hw_kernel_config.generation_prefill_cuda_graph_max_requests = 2;
        params.hw_kernel_config.generation_prefill_capture_token_buckets   = {8};
        params.hw_kernel_config.decode_capture_batch_sizes                 = {1, 2};
        params.runtime_config.fifo_scheduler_config.max_context_batch_size = 2;
    }
    params.device_resource_config.enable_layer_micro_batch = python_norm ? 0 : 1;
    if (!handler.is_none()) {
        py_model.attr("custom_output_handler") = std::move(handler);
    }
    PyWrappedModel model(params, std::move(py_model));

    auto inputs         = makeInputs({3, 3}, {1, 2}, {1, 2, 3, 4}, 2, {1, 2, 3, 4}, {"full"}, block_table_width, 4, 64);
    inputs.combo_tokens = torch::arange(6, torch::kInt32).pin_memory();
    inputs.pd_separation            = false;
    inputs.kv_cache_block_id        = inputs.kv_cache_block_id.squeeze(0);
    inputs.kv_cache_kernel_block_id = inputs.kv_cache_kernel_block_id.squeeze(0);
    inputs.custom_output_indexes    = std::move(indexes);
    inputs.need_all_logits          = !python_norm;
    py::dict result;
    if (enable_graph) {
        model.prepareAttentionInputs(inputs);
        model.updateKVCacheKernelBlockId(inputs);
    }
    const auto outputs      = model.forward(inputs);
    result["graph_status"]  = generationPrefillCudaGraphStatusString(outputs.generation_prefill_cuda_graph_status);
    result["custom_output"] = outputs.custom_output;
    result["custom_output_error"] = outputs.custom_output_error;
    result["logits"]              = outputs.logits;
    if (enable_graph) {
        auto       decode              = inputs;
        const auto batch               = inputs.input_lengths.size(0);
        decode.combo_tokens            = torch::arange(batch, torch::kInt32).pin_memory();
        decode.sequence_lengths        = inputs.input_lengths;
        decode.prefix_lengths          = torch::empty({0}, torch::kInt32).pin_memory();
        decode.lm_output_indexes       = torch::arange(batch, torch::kInt32).pin_memory();
        decode.custom_output_indexes   = torch::Tensor();
        result["decode_custom_output"] = model.forward(decode).custom_output;
        // Same graph shape, different tokens and scoring positions: no stale replay data.
        inputs.combo_tokens          = inputs.combo_tokens.flip({0}).pin_memory();
        inputs.custom_output_indexes = (inputs.custom_output_indexes - 1).pin_memory();
        model.prepareAttentionInputs(inputs);
        model.updateKVCacheKernelBlockId(inputs);
        auto next                    = model.forward(inputs);
        result["next_custom_output"] = next.custom_output;
        result["next_logits"]        = next.logits;
    }
    return result;
}

}  // namespace
}  // namespace rtp_llm::test

PYBIND11_MODULE(libth_pywrapped_model_cache_store_integration_test, m) {
    torch_ext::registerPyOpDefs(m);
    m.def("run_post_layers", &rtp_llm::test::runCustomOutput);
    m.def("run_scenario",
          &rtp_llm::test::runPyWrappedModelCacheStoreScenario,
          py::arg("py_model"),
          py::arg("scenario_name"),
          py::arg("enable_graph") = false);
    m.def("run_dirty_generation_prefill_capture_scenario",
          &rtp_llm::test::runDirtyGenerationPrefillCaptureScenario,
          py::arg("py_model"));
    m.def("run_generation_prefill_capture_scenario",
          &rtp_llm::test::runGenerationPrefillCaptureScenario,
          py::arg("py_model"),
          py::arg("expected_error_substring") = "");
}
