#include <algorithm>
#include <cstdint>
#include <cstring>
#include <list>
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
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "kmonitor/client/MetricType.h"
#include "kmonitor/client/core/MetricsRecord.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/HiddenStateCapturePublisher.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);
}

namespace rtp_llm::test {

struct PyWrappedModelTestPeer {
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
        auto scenario = makeMultiTagScenario();
        scenario.inputs.cache_save_positions = pinnedLongTensor({4}, {1, 1});
        scenario.inputs.cache_save_block_ids = pinnedTensor({7, 8}, {2, 1, 1});
        return scenario;
    }
    if (name == "sparse_save_split") {
        auto config = makeCacheConfig({{"default", 2, 16}});
        auto layout = makeLayout(config);
        auto inputs = makeInputs(/*input_lengths=*/{1, 1, 2, 4},
                                 /*request_ids=*/{201, 202, 203, 204},
                                 /*cache_keys=*/{11, 0, 12, 0, 13, 0, 14, 15},
                                 /*cache_keys_width=*/2,
                                 /*block_ids=*/{1, -1, 2, -1, 3, -1, 4, 5},
                                 /*group_tags=*/{"default"},
                                 /*block_table_width=*/2,
                                 /*global_tokens_per_block=*/2,
                                 /*global_stride_bytes=*/16);
        inputs.sequence_lengths = pinnedTensor({4, 5}, {2});
        inputs.prefix_lengths = pinnedTensor({0, 0}, {2});
        inputs.cache_save_positions = pinnedLongTensor({3, -1, 5, 6}, {2, 2});
        inputs.cache_save_block_ids = pinnedTensor({31, -1, 41, 42}, {1, 2, 2});
        return {std::move(config), std::move(layout.layout), std::move(layout.base_addresses), std::move(inputs)};
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
    if (name == "cp_actual_lengths_rank0") {
        return makeContextParallelScenario(/*tp_rank=*/0);
    }
    if (name == "cp_actual_lengths_rank1") {
        return makeContextParallelScenario(/*tp_rank=*/1);
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
        return makeCaptureScenario(false, true, false);
    }
    if (name == "capture_explicit_off") {
        auto scenario                         = makeCaptureScenario(false, true, false);
        scenario.inputs.capture_hidden_states = false;
        return scenario;
    }
    if (name == "capture_with_lm_head") {
        auto scenario                = makeCaptureScenario(false, true, false);
        scenario.inputs.skip_lm_head = false;
        scenario.install_lm_head     = true;
        return scenario;
    }
    if (name == "capture_async_history") {
        auto scenario                               = makeCaptureScenario(false, true, false);
        scenario.increment_request_ids_each_forward = true;
        return scenario;
    }

    if (name == "capture_non_4096_hidden") {
        auto scenario        = makeCaptureScenario(false, true, false);
        scenario.hidden_size = 2048;
        return scenario;
    }
    if (name == "capture_bf16_tp") {
        auto scenario                = makeCaptureScenario(false, true, false);
        scenario.parallelism.tp_size = 2;
        return scenario;
    }

    if (name == "capture_non_owner") {
        return makeCaptureScenario(false, false, false);
    }
    if (name == "capture_fp8") {
        return makeCaptureScenario(false, true, true);
    }
    if (name == "capture_fp8_tp") {
        auto scenario                = makeCaptureScenario(false, true, true);
        scenario.parallelism.tp_size = 2;
        return scenario;
    }
    if (name == "capture_cp") {
        return makeCaptureScenario(true, true, false);
    }
    if (name == "capture_cp_invalid_num_valid_tokens") {
        auto scenario                            = makeCaptureScenario(true, true, false);
        scenario.report_invalid_num_valid_tokens = true;
        return scenario;
    }
    if (name == "capture_cp_malformed_layout") {
        return makeCaptureScenario(true, true, false);
    }
    if (name == "capture_cp_malformed_restored_width") {
        auto scenario                   = makeCaptureScenario(true, true, false);
        scenario.corrupt_restored_width = true;
        return scenario;
    }
    if (name == "capture_duplicate_layer_id") {
        auto scenario              = makeCaptureScenario(false, true, false);
        scenario.capture_layer_ids = {0, 0};
        return scenario;
    }
    if (name == "capture_invalid_dtype") {
        auto scenario          = makeCaptureScenario(false, true, false);
        scenario.capture_dtype = static_cast<HiddenStateCaptureDtype>(99);
        return scenario;
    }
    if (name == "capture_negative_request_id") {
        auto scenario              = makeCaptureScenario(false, true, false);
        scenario.inputs.request_id = pinnedLongTensor({-1, 502}, {2});
        return scenario;
    }
    if (name == "capture_duplicate_request_id") {
        auto scenario              = makeCaptureScenario(false, true, false);
        scenario.inputs.request_id = pinnedLongTensor({501, 501}, {2});
        return scenario;
    }
    if (name == "capture_prepare_failure_tp") {
        auto scenario                = makeCaptureScenario(false, true, false);
        scenario.parallelism.tp_size = 2;
        auto indices                 = torch::tensor({0, 1}, torch::kInt64).reshape({1, 2});
        auto values                  = torch::tensor({501, 502}, torch::kInt64);
        scenario.inputs.request_id   = torch::sparse_coo_tensor(indices, values, {2}).coalesce();
        return scenario;
    }
    if (name == "capture_prepare_failure_non_owner") {
        auto scenario              = makeCaptureScenario(false, false, false);
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

// Exercise startup without a CUDA runtime; the Python fixture supplies the store.
py::dict flushCapturePublisherWithoutClose() {
    FakeMetricsReporter         fake_metrics_reporter;
    HiddenStateCapturePublisher publisher(/*layer_count=*/2,
                                          /*hidden_size=*/1,
                                          HiddenStateCaptureDtype::BF16,
                                          torch::kFloat16,
                                          /*owner=*/true,
                                          /*local_rank=*/0,
                                          fake_metrics_reporter.reporter());
    publisher.flushPending();
    py::dict result;
    result["flushed"] = true;
    return result;
}

py::dict closeCapturePublisherWithoutGil() {
    FakeMetricsReporter         fake_metrics_reporter;
    HiddenStateCapturePublisher publisher(/*layer_count=*/2,
                                          /*hidden_size=*/1,
                                          HiddenStateCaptureDtype::BF16,
                                          torch::kFloat16,
                                          /*owner=*/true,
                                          /*local_rank=*/0,
                                          fake_metrics_reporter.reporter());
    {
        py::gil_scoped_release release;
        publisher.flushAndClose();
        publisher.flushAndClose();
    }
    py::dict result;
    result["closed_without_gil"] = true;
    return result;
}

py::dict initializeCapturePublisher() {
    FakeMetricsReporter fake_metrics_reporter;
    py::dict            result;
    result["error"] = "";
    try {
        HiddenStateCapturePublisher publisher(/*layer_count=*/2,
                                              /*hidden_size=*/1,
                                              HiddenStateCaptureDtype::BF16,
                                              torch::kFloat16,
                                              /*owner=*/true,
                                              /*local_rank=*/0,
                                              fake_metrics_reporter.reporter());
        publisher.flushAndClose();
    } catch (const std::exception& error) {
        result["error"] = error.what();
    }
    auto metrics                   = fake_metrics_reporter.snapshotHiddenStateCaptureMetrics();
    result["capture_metrics"]      = metrics[0];
    result["capture_metric_types"] = metrics[1];
    return result;
}

py::list runTpSyncModelInputsScenario(int64_t                                   tp_rank,
                                      int64_t                                   tp_size,
                                      const std::vector<std::pair<bool, bool>>& flags,
                                      bool                                      skip_run,
                                      bool                                      initial_flags,
                                      bool                                      grouped_cache) {
    ParallelismConfig parallelism;
    parallelism.tp_rank = tp_rank;
    parallelism.tp_size = tp_size;
    GptModelInputs inputs;
    inputs.skip_lm_head          = initial_flags;
    inputs.capture_hidden_states = initial_flags;
    if (grouped_cache && !skip_run) {
        // Tags are rank-local configuration; only the payload uses canonical order on the wire.
        inputs.kv_cache_group_tags =
            tp_rank % 2 == 0 ? std::vector<std::string>{"linear", "full"} : std::vector<std::string>{"full", "linear"};
        if (tp_rank == 0) {
            inputs.kv_cache_block_id        = pinnedTensor({31, 32, 11, 12}, {2, 1, 2});
            inputs.kv_cache_kernel_block_id = pinnedTensor({131, 132, 111, 112}, {2, 1, 2});
            inputs.kv_cache_group_types     = pinnedTensor({0, 1}, {2});
            inputs.kv_cache_update_mapping  = pinnedTensor({0, 3, 4, 1, 1, 2}, {2, 3});
            inputs.cache_save_positions     = pinnedLongTensor({1536, 2000}, {1, 2});
            inputs.cache_save_block_ids     = pinnedTensor({41, -1, -1, 51}, {2, 1, 2});
        }
    }
    if (tp_rank == 0 && !skip_run) {
        inputs.combo_tokens          = pinnedTensor({7, 8, 9}, {3});
        inputs.input_lengths         = pinnedTensor({3}, {1});
        inputs.sequence_lengths      = pinnedTensor({}, {0});
        inputs.prefix_lengths        = pinnedTensor({0}, {1});
        inputs.lm_output_indexes     = pinnedTensor({2}, {1});
        inputs.request_id            = pinnedLongTensor({42}, {1});
        inputs.request_pd_separation = torch::zeros({1}, torch::TensorOptions(torch::kBool).pinned_memory(true));
    }

    py::list results;
    for (const auto& [skip_lm_head, capture_hidden_states] : flags) {
        // Non-root starts with no stream input and retains the previous round's execution flags.
        inputs.skip_run = tp_rank == 0 ? skip_run : true;
        if (tp_rank == 0) {
            inputs.skip_lm_head          = skip_lm_head;
            inputs.capture_hidden_states = capture_hidden_states;
        }
        tpSyncModelInputs(inputs, parallelism);
        py::dict result;
        result["skip_lm_head"]          = inputs.skip_lm_head;
        result["capture_hidden_states"] = inputs.capture_hidden_states;
        result["skip_run"]              = inputs.skip_run;
        result["combo_tokens"]  = inputs.combo_tokens.defined() ? py::cast(inputs.combo_tokens.clone()) : py::none();
        result["input_lengths"] = inputs.input_lengths.defined() ? py::cast(inputs.input_lengths.clone()) : py::none();
        result["request_id"]    = inputs.request_id.defined() ? py::cast(inputs.request_id.clone()) : py::none();
        if (grouped_cache) {
            result["group_tags"] = inputs.kv_cache_group_tags;
            result["physical"] =
                inputs.kv_cache_block_id.defined() ? py::cast(inputs.kv_cache_block_id.clone()) : py::none();
            result["kernel"] = inputs.kv_cache_kernel_block_id.defined() ?
                                   py::cast(inputs.kv_cache_kernel_block_id.clone()) :
                                   py::none();
            result["group_types"] =
                inputs.kv_cache_group_types.defined() ? py::cast(inputs.kv_cache_group_types.clone()) : py::none();
            result["copy_mapping"] = inputs.kv_cache_update_mapping.defined() ?
                                         py::cast(inputs.kv_cache_update_mapping.clone()) :
                                         py::none();
            result["cache_save_positions"] = inputs.cache_save_positions.defined() ?
                                                  py::cast(inputs.cache_save_positions.clone()) :
                                                  py::none();
            result["cache_save_block_ids"] = inputs.cache_save_block_ids.defined() ?
                                                  py::cast(inputs.cache_save_block_ids.clone()) :
                                                  py::none();
        }
        results.append(std::move(result));
    }
    return results;
}

py::dict runPyWrappedModelCacheStoreScenario(py::object         py_model,
                                             const std::string& scenario_name,
                                             size_t             forward_count,
                                             bool               ignore_deferred_errors,
                                             bool               capture_only_first_forward,
                                             bool               skip_first_deferred_error_take) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() {
        initRuntime(/*device_id=*/0);
    });

    RTP_LLM_CHECK_WITH_INFO(forward_count > 0, "forward_count must be positive");

    const bool cacheless_warmup = scenario_name == "cacheless_warmup";
    const bool inspect_split = scenario_name == "sparse_save_split"
                               || scenario_name == "micro_batch_split_pinned" || scenario_name == "micro_batch_split_cuda"
                               || scenario_name == "micro_batch_split_single_group"
                               || scenario_name == "micro_batch_split_2d";
    const bool single_group_split =
        scenario_name == "micro_batch_split_single_group" || scenario_name == "micro_batch_split_2d";
    const auto input_scenario = scenario_name == "sparse_save_split" ? scenario_name :
                                inspect_split ? (single_group_split ? "micro_batch" : "micro_batch_multi_tag") :
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
                              /*max_seq_len=*/64,
                              scenario.hidden_size,
                              manager,
                              scenario.mtp_cache_config_index,
                              /*hc_mult=*/1,
                              metrics_reporter,
                              /*kernel_block_table_width=*/0,
                              scenario.capture_layer_ids,
                              scenario.capture_dtype};
    if (cacheless_warmup) {
        params.cache_manager                     = nullptr;
        scenario.inputs.warmup                   = true;
        scenario.inputs.kv_cache_block_id        = torch::Tensor();
        scenario.inputs.kv_cache_kernel_block_id = torch::Tensor();
    }

    GptModelOutputs            output;
    HiddenStateCaptureStats    capture_stats;
    size_t                     cp_handle_outputs_calls{0};
    std::vector<torch::Tensor> forward_hidden_states;
    std::vector<std::string>   deferred_capture_errors;
    forward_hidden_states.reserve(forward_count);
    {
        PyWrappedModel                model(params, std::move(py_model));
        TestContextParallelProcessor* test_cp_processor = nullptr;
        if (inspect_split) {
            const auto split = scenario_name == "sparse_save_split" ?
                                   PyWrappedModelTestPeer::splitInputsIntoMicroBatches(
                                       model, scenario.inputs, MicroBatchPlan{true, {{0, 1}, {1, 1}, {1, 0}}}) :
                                   PyWrappedModelTestPeer::splitInputsIntoMicroBatches(model, scenario.inputs);
            py::list   batches;
            for (const auto& inputs : split.first) {
                py::dict batch;
                batch["tags"]          = inputs.kv_cache_group_tags;
                batch["physical"]      = inputs.kv_cache_block_id;
                batch["kernel"]        = inputs.kv_cache_kernel_block_id;
                batch["input_lengths"] = inputs.input_lengths;
                batch["cache_save_positions"] = inputs.cache_save_positions.defined() ?
                                                     py::cast(inputs.cache_save_positions.clone()) : py::none();
                batch["cache_save_block_ids"] = inputs.cache_save_block_ids.defined() ?
                                                     py::cast(inputs.cache_save_block_ids.clone()) : py::none();
                batches.append(std::move(batch));
            }
            py::dict result;
            result["source_tags"]     = scenario.inputs.kv_cache_group_tags;
            result["source_physical"] = scenario.inputs.kv_cache_block_id;
            result["source_kernel"]   = scenario.inputs.kv_cache_kernel_block_id;
            result["source_cache_save_positions"] = scenario.inputs.cache_save_positions.defined() ?
                                                        py::cast(scenario.inputs.cache_save_positions.clone()) : py::none();
            result["source_cache_save_block_ids"] = scenario.inputs.cache_save_block_ids.defined() ?
                                                        py::cast(scenario.inputs.cache_save_block_ids.clone()) : py::none();
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
            result["real_cache_save_defined"] = split.first.at(0).cache_save_positions.defined()
                                                  && split.first.at(0).cache_save_block_ids.defined();
            result["fake_cache_save_defined"] = split.first.at(1).cache_save_positions.defined()
                                                  || split.first.at(1).cache_save_block_ids.defined();
            return result;
        }
        if (scenario.replace_cp_processor) {
            auto processor = std::make_unique<TestContextParallelProcessor>(
                scenario.parallelism, scenario.report_invalid_num_valid_tokens, scenario.corrupt_restored_width);
            test_cp_processor = processor.get();
            PyWrappedModelTestPeer::replaceContextParallelProcessor(model, std::move(processor));
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

py::dict runDirtyGenerationPrefillCaptureScenario(py::object py_model) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() {
        initRuntime(/*device_id=*/0);
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

py::dict runGenerationPrefillCaptureScenario(py::object                                py_model,
                                             const std::string&                        expected_error_substring,
                                             const std::vector<std::pair<bool, bool>>& forward_flags) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() {
        initRuntime(/*device_id=*/0);
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
    weights.layers.resize(forward_flags.empty() ? 1 : 2);
    if (!forward_flags.empty()) {
        auto lm_head    = std::make_shared<DenseWeights>();
        lm_head->kernel = torch::ones({2, 4}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        weights.lm_head = std::move(lm_head);
    }
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
                              /*max_seq_len=*/4,
                              /*hidden_size=*/4,
                              manager,
                              /*mtp_cache_config_index=*/std::nullopt};
    params.kernel_block_table_width = 1;

    if (!forward_flags.empty()) {
        params.hidden_state_capture_layer_ids = {0, 1};
    }

    size_t      available_during  = 0;
    bool        graph_enabled     = false;
    bool        saw_capture_error = false;
    std::string capture_error_message;
    py::list    forwards;
    try {
        PyWrappedModel model(params, py_model);
        available_during = manager->availableBlocksNum();
        graph_enabled    = PyWrappedModelTestPeer::generationPrefillCudaGraphReady(model);
        for (size_t i = 0; i < forward_flags.size(); ++i) {
            // Reuse the captured FULL-group topology and reserved block 0; the
            // test model replaces attention math and never writes request KV.
            auto inputs = makeInputs(/*input_lengths=*/{3},
                                     /*request_ids=*/{801 + static_cast<int64_t>(i)},
                                     /*cache_keys=*/{0},
                                     /*cache_keys_width=*/1,
                                     /*block_ids=*/{0},
                                     /*group_tags=*/{"full"},
                                     /*block_table_width=*/1,
                                     config.group("full").seqSizePerBlock(),
                                     config.group("full").kvBlockStrideBytes());
            inputs.combo_tokens.add_(static_cast<int32_t>(i));
            inputs.pd_separation         = false;
            inputs.skip_lm_head          = forward_flags[i].first;
            inputs.capture_hidden_states = forward_flags[i].second;
            const auto calls_before      = py_model.attr("forward_calls").cast<int64_t>();
            auto       output            = model.forward(inputs);
            if (auto error = model.takeDeferredHiddenStateCaptureError(); error.has_value()) {
                throw std::runtime_error(*error);
            }
            py::dict forward;
            forward["python_forward_calls"] = py_model.attr("forward_calls").cast<int64_t>() - calls_before;
            forward["replayed"] =
                output.generation_prefill_cuda_graph_status == GenerationPrefillCudaGraphStatus::REPLAYED;
            forward["not_requested"] =
                output.generation_prefill_cuda_graph_status == GenerationPrefillCudaGraphStatus::NOT_REQUESTED;
            forward["logits"] = output.logits.defined() ? py::cast(output.logits.cpu()) : py::none();
            // Retain the main output across subsequent replays to catch loss
            // of the wrapper clone. Auxiliary graph output is a per-replay view.
            forward["hidden_states"]            = output.all_hidden_states;
            forward["mtp_target_hidden_states"] = output.mtp_target_hidden_states.cpu();
            forwards.append(std::move(forward));
            model.releaseBuffers();
        }
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
    result["forwards"]              = std::move(forwards);
    return result;
}

// Exercise input preparation, Python forward, graphs and custom output together.
// Only decoder math is replaced; all routing and post-layers code is production.
py::dict runCustomOutput(py::object py_model, py::object handler, torch::Tensor indexes, bool python_norm) {
    static std::once_flag runtime_once;
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
    rtp_llm::registerExecCtxOps(m);
    m.def("run_exact_prefix_pd_gather", &rtp_llm::test::runExactPrefixPdGatherScenario);
    m.def("run_post_layers", &rtp_llm::test::runCustomOutput);
    m.def("flush_capture_publisher_without_close", &rtp_llm::test::flushCapturePublisherWithoutClose);
    m.def("close_capture_publisher_without_gil", &rtp_llm::test::closeCapturePublisherWithoutGil);
    m.def("initialize_capture_publisher", &rtp_llm::test::initializeCapturePublisher);
    m.def("run_tp_sync_model_inputs_scenario",
          &rtp_llm::test::runTpSyncModelInputsScenario,
          py::arg("tp_rank"),
          py::arg("tp_size"),
          py::arg("flags"),
          py::arg("skip_run"),
          py::arg("initial_flags") = false,
          py::arg("grouped_cache") = false);
    m.def("run_scenario",
          &rtp_llm::test::runPyWrappedModelCacheStoreScenario,
          py::arg("py_model"),
          py::arg("scenario_name"),
          py::arg("forward_count")                  = 1,
          py::arg("ignore_deferred_errors")         = false,
          py::arg("capture_only_first_forward")     = false,
          py::arg("skip_first_deferred_error_take") = false);
    m.def("run_dirty_generation_prefill_capture_scenario",
          &rtp_llm::test::runDirtyGenerationPrefillCaptureScenario,
          py::arg("py_model"));
    m.def("run_generation_prefill_capture_scenario",
          &rtp_llm::test::runGenerationPrefillCaptureScenario,
          py::arg("py_model"),
          py::arg("expected_error_substring") = "",
          py::arg("forward_flags")            = std::vector<std::pair<bool, bool>>{});
}
