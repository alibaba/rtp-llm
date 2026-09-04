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

struct RecordingGraphState {
    std::vector<std::string> calls;
    std::vector<int64_t>     position_id_numels;
    std::vector<int64_t>     token_type_id_numels;
    bool                     can_run_result{true};
};

class RecordingGraphBase: public GraphBase {
public:
    explicit RecordingGraphBase(std::shared_ptr<RecordingGraphState> recording):
        GraphBase(py::none()), recording_(std::move(recording)) {}

    void initCapture() override {}

    PyModelOutputs forward(const PyModelInputs&, CudaGraphState&) override {
        return {};
    }

    void setPositionEncoding(torch::Tensor) override {}
    void setTokenTypeEmbedding(torch::Tensor) override {}
    void setInputEmbeddingScalar(float) override {}

    bool canRun(const PyModelInputs& inputs, CudaGraphState&) override {
        record("canRun", inputs);
        return recording_->can_run_result;
    }

    void prepareAttentionInputs(const PyModelInputs& inputs, CudaGraphState&, bool) override {
        record("prepareAttentionInputs", inputs);
    }

    void updateKVCacheKernelBlockId(const PyModelInputs& inputs, CudaGraphState&) override {
        record("updateKVCacheKernelBlockId", inputs);
    }

private:
    void record(const std::string& call, const PyModelInputs& inputs) {
        const auto numel_or_missing = [](const torch::Tensor& tensor) {
            return tensor.defined() ? tensor.numel() : -1;
        };
        recording_->calls.push_back(call);
        recording_->position_id_numels.push_back(numel_or_missing(inputs.bert_embedding_inputs.combo_position_ids));
        recording_->token_type_id_numels.push_back(
            numel_or_missing(inputs.bert_embedding_inputs.combo_tokens_type_ids));
    }

    std::shared_ptr<RecordingGraphState> recording_;
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
    bool                             bert_embedding_weights{false};
    bool                             plan_micro_batches_only{false};
    bool                             graph_wiring_only{false};
    bool                             graph_can_run_result{true};
    size_t                           layer_num{1};
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

void addUnsupportedMicroBatchSignal(Scenario& scenario, const std::string& signal) {
    const auto token_count          = scenario.inputs.combo_tokens.numel();
    scenario.bert_embedding_weights = true;
    if (signal == "text_mask") {
        scenario.inputs.text_tokens_mask = pinnedTensor(std::vector<int32_t>(token_count, 1), {token_count});
    } else if (signal == "features") {
        scenario.inputs.multimodal_features =
            std::vector<torch::Tensor>{torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kFloat16))};
    } else if (signal == "locs") {
        scenario.inputs.mm_features_locs = pinnedTensor({0}, {1});
    } else if (signal == "extra") {
        scenario.inputs.mm_extra_input =
            std::vector<torch::Tensor>{torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kFloat16))};
    } else {
        throw std::invalid_argument("unknown micro-batch fallback signal: " + signal);
    }
}

Scenario makeUnsupportedMicroBatchScenario(const std::string& signal) {
    auto scenario = makeMicroBatchScenario();
    addUnsupportedMicroBatchSignal(scenario, signal);
    return scenario;
}

Scenario makeBertWithoutRequestOwnedMultimodalInputsScenario() {
    auto scenario                    = makeMicroBatchScenario();
    scenario.bert_embedding_weights  = true;
    scenario.plan_micro_batches_only = true;
    return scenario;
}

Scenario makeDegenerateUnsupportedMicroBatchScenario(const std::string& kind, const std::string& signal) {
    auto config = makeCacheConfig({{"default", 2, 16}});
    auto layout = makeLayout(config);

    std::vector<int32_t> input_lengths;
    if (kind == "single_request") {
        input_lengths = {2};
    } else if (kind == "mixed_multilayer") {
        input_lengths = {1, 2};
    } else {
        throw std::invalid_argument("unknown degenerate micro-batch kind: " + kind);
    }

    const size_t batch_size = input_lengths.size();
    auto         inputs     = makeInputs(input_lengths,
                             /*request_ids=*/std::vector<int64_t>(batch_size, 501),
                             /*cache_keys=*/std::vector<int64_t>(batch_size, 5101),
                             /*cache_keys_width=*/1,
                             /*block_ids=*/std::vector<int32_t>(batch_size, 1),
                             /*group_count=*/1,
                             /*block_table_width=*/1,
                             /*global_tokens_per_block=*/2,
                             /*global_stride_bytes=*/16);
    Scenario scenario{std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
    scenario.device_resources.enable_layer_micro_batch = static_cast<int>(MicroBatchType::DS_PREFILL);
    if (kind == "mixed_multilayer") {
        scenario.inputs.sequence_lengths = pinnedTensor({0}, {1});
        scenario.plan_micro_batches_only = true;
        scenario.layer_num               = 2;
    }
    addUnsupportedMicroBatchSignal(scenario, signal);
    return scenario;
}

Scenario makeBertWiringScenario() {
    auto                 scenario    = makeMultiTagScenario();
    const auto           token_count = scenario.inputs.combo_tokens.numel();
    std::vector<int32_t> position_ids(token_count);
    std::iota(position_ids.begin(), position_ids.end(), int32_t{0});
    scenario.inputs.combo_position_ids    = pinnedTensor(position_ids, {token_count});
    scenario.inputs.combo_tokens_type_ids = pinnedTensor(std::vector<int32_t>(token_count, 1), {token_count});
    scenario.bert_embedding_weights       = true;
    return scenario;
}

Scenario makeBertGraphWiringScenario(bool request_owned_mask, bool can_run_result = true) {
    auto scenario                 = makeBertWiringScenario();
    scenario.graph_wiring_only    = true;
    scenario.graph_can_run_result = can_run_result;
    if (request_owned_mask) {
        const auto token_count           = scenario.inputs.combo_tokens.numel();
        scenario.inputs.text_tokens_mask = pinnedTensor(std::vector<int32_t>(token_count, 1), {token_count});
    }
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
    if (name == "bert_wiring") {
        return makeBertWiringScenario();
    }
    if (name == "bert_graph_wiring") {
        return makeBertGraphWiringScenario(false);
    }
    if (name == "bert_graph_request_owned_mask") {
        return makeBertGraphWiringScenario(true);
    }
    if (name == "bert_graph_can_run_false") {
        return makeBertGraphWiringScenario(false, false);
    }
    if (name == "bert_without_request_owned_multimodal_inputs") {
        return makeBertWithoutRequestOwnedMultimodalInputsScenario();
    }
    const std::string unsupported_micro_batch_prefix = "unsupported_micro_batch_";
    if (name.rfind(unsupported_micro_batch_prefix, 0) == 0) {
        return makeUnsupportedMicroBatchScenario(name.substr(unsupported_micro_batch_prefix.size()));
    }
    const std::string degenerate_micro_batch_prefix = "degenerate_micro_batch_";
    if (name.rfind(degenerate_micro_batch_prefix, 0) == 0) {
        const auto remainder = name.substr(degenerate_micro_batch_prefix.size());
        for (const std::string kind : {"single_request", "mixed_multilayer"}) {
            const auto kind_prefix = kind + "_";
            if (remainder.rfind(kind_prefix, 0) == 0) {
                return makeDegenerateUnsupportedMicroBatchScenario(kind, remainder.substr(kind_prefix.size()));
            }
        }
    }
    if (name == "cp_actual_lengths") {
        return makeContextParallelScenario();
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

py::dict runPyWrappedModelCacheStoreScenario(py::object py_model, const std::string& scenario_name) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    });

    auto scenario    = makeScenario(scenario_name);
    auto cache_store = std::make_shared<RecordingCacheStore>();
    auto manager     = std::make_shared<KVCacheManager>(scenario.manager_config,
                                                    /*warmup=*/true,
                                                    /*metrics_reporter=*/nullptr,
                                                    KVCacheConfig{},
                                                    scenario.parallelism);
    manager->setCacheStore(cache_store);

    Weights weights;
    weights.layers.resize(scenario.layer_num);
    if (scenario.bert_embedding_weights) {
        auto position_encoding = std::make_shared<DenseWeights>();
        position_encoding->kernel =
            torch::zeros({64, 1}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        weights.position_encoding = std::move(position_encoding);

        auto token_type_embedding = std::make_shared<DenseWeights>();
        token_type_embedding->kernel =
            torch::zeros({2, 1}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        weights.token_type_embedding = std::move(token_type_embedding);
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

    std::optional<bool>                  micro_batch_plan_enabled;
    std::shared_ptr<RecordingGraphState> graph_recording;
    {
        PyWrappedModel model(params, std::move(py_model));
        if (scenario.replace_cp_processor) {
            model.context_parallel_processor_ = std::make_unique<TestContextParallelProcessor>(scenario.parallelism);
        }
        if (scenario.graph_wiring_only) {
            graph_recording                 = std::make_shared<RecordingGraphState>();
            graph_recording->can_run_result = scenario.graph_can_run_result;
            model.graph_runner_             = new RecordingGraphBase(graph_recording);
            model.enable_cuda_graph_        = true;
            model.prepareAttentionInputs(scenario.inputs, /*skip_forward_event_sync=*/true);
            model.updateKVCacheKernelBlockId(scenario.inputs);
        } else if (scenario.plan_micro_batches_only) {
            micro_batch_plan_enabled = model.planMicroBatches(scenario.inputs).enable;
        } else {
            (void)model.forward(scenario.inputs);
        }
    }
    auto result = serializeResult(*cache_store, scenario.base_addresses);
    if (micro_batch_plan_enabled.has_value()) {
        result["micro_batch_plan_enabled"] = *micro_batch_plan_enabled;
    }
    if (graph_recording) {
        result["graph_calls"]                = graph_recording->calls;
        result["graph_position_id_numels"]   = graph_recording->position_id_numels;
        result["graph_token_type_id_numels"] = graph_recording->token_type_id_numels;
    }
    return result;
}

}  // namespace
}  // namespace rtp_llm::test

PYBIND11_MODULE(libth_pywrapped_model_cache_store_integration_test, m) {
    torch_ext::registerPyOpDefs(m);
    m.def("run_scenario",
          &rtp_llm::test::runPyWrappedModelCacheStoreScenario,
          py::arg("py_model"),
          py::arg("scenario_name"));
}
