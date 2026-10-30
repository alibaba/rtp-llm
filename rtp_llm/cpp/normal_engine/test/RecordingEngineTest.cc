#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <pybind11/embed.h>
#include "alog/Configurator.h"
#include "autil/TimeUtility.h"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

PYBIND11_EMBEDDED_MODULE(recording_engine_test_ops, m) {
    // OpDefs binds Tensor-valued defaults; initialize their Python caster first.
    pybind11::module_::import("torch");
    torch_ext::registerPyOpDefs(m);
}

namespace rtp_llm {
class RecordingEngineTest: public DeviceTestBase {
public:
    void SetUp() override {}  // Initialize CUDA only in the fresh child process.
    void runLifecycleAndSnapshot(bool               tensor_capture = false,
                                 RoleType           server_role    = RoleType::PDFUSION,
                                 const std::string& phases         = "prefill");
    void runCaptureEligibility();
    void runCaptureContextPreparation();
    void runCaptureWithoutRecorder(bool enabled = true);
    void runCaptureConstructorFailure();
};

class CaptureContextTestModel: public MockModel {
public:
    CaptureContextTestModel(size_t vocab_size, std::vector<std::string>& snapshots, bool prefill, bool decode):
        MockModel(vocab_size), snapshots_(snapshots), prefill_(prefill), decode_(decode) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        const bool selected = inputs.sequence_lengths.numel() == 0 ? prefill_ : decode_;
        if (!selected || inputs.warmup || inputs.is_fake_stream || inputs.is_target_verify || inputs.is_draft_extend) {
            EXPECT_TRUE(inputs.tensor_capture_context_json.empty());
        } else {
            EXPECT_FALSE(inputs.tensor_capture_context_json.empty());
            snapshots_.push_back(inputs.tensor_capture_context_json);
        }
        return MockModel::forward(inputs);
    }

private:
    std::vector<std::string>& snapshots_;
    bool                      prefill_;
    bool                      decode_;
};

static std::string
writeCaptureConfig(const std::string& directory, bool enabled, const std::string& phases = "prefill") {
    const auto    path = directory + "/capture.json";
    std::ofstream file(path);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    if (enabled) {
        file << R"({"enabled":true,"authorized_capture":true,"layers":[0],"ops":["indexer.topk"],"phases":[)"
             << (phases == "both" ? "\"prefill\",\"decode\"" : "\"" + phases + "\"")
             << R"(],"rank_filter":"record_owner","max_calls":1,
                    "max_bundle_bytes":1024,"max_total_bytes":1024,"max_inflight_bytes":4096})";
    } else {
        file << R"({"enabled":false})";
    }
    file.close();
    return path;
}

void RecordingEngineTest::runLifecycleAndSnapshot(bool               tensor_capture,
                                                  RoleType           server_role,
                                                  const std::string& phases) {
    if ((tensor_capture || server_role == RoleType::DECODE) && !Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    DeviceTestBase::SetUp();
    char directory[] = "/tmp/rtp-recording-engine-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    setenv("RTP_LLM_RECORD_DIR", directory, 1);
    setenv("RTP_LLM_RECORD_SESSION", "engine-test", 1);
    const bool               capture_prefill = tensor_capture && phases != "decode" && server_role != RoleType::DECODE;
    const bool               capture_decode = tensor_capture && phases != "prefill" && server_role != RoleType::PREFILL;
    std::vector<std::string> snapshots;
    std::shared_ptr<NormalEngine> engine;
    if (tensor_capture || server_role == RoleType::DECODE) {
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
        params.profiling_debug_logging_config.tensor_capture_config =
            writeCaptureConfig(directory, tensor_capture, phases);
        params.pd_sep_config.role_type      = server_role;
        params.parallelism_config.role_type = server_role;
        NormalExecutor::test_model_factory  = [&](const GptModelInitParams&) {
            return std::make_unique<CaptureContextTestModel>(
                model_config.vocab_size, snapshots, capture_prefill, capture_decode);
        };
        engine                             = std::make_shared<NormalEngine>(params, nullptr);
        NormalExecutor::test_model_factory = nullptr;
    } else {
        engine = createMockEngine(CustomConfig{});
    }
    const auto batch_config =
        std::string("alog.max_msg_len=0\nalog.logger.batch_schedule=INFO, batchScheduleAppender\n")
        + "inherit.batch_schedule=false\nalog.appender.batchScheduleAppender=FileAppender\n"
        + "alog.appender.batchScheduleAppender.fileName=" + directory + "/batch_schedule.log\n"
        + "alog.appender.batchScheduleAppender.layout=PatternLayout\n"
        + "alog.appender.batchScheduleAppender.layout.LogPattern=%%m\n"
        + "alog.appender.batchScheduleAppender.async_flush=true\n";
    alog::Configurator::configureLoggerFromString(batch_config.c_str());
    auto input                             = std::make_shared<GenerateInput>();
    input->begin_time_us                   = autil::TimeUtility::currentTimeInMicroSeconds();
    input->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
    input->generate_config                 = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 4;
    input->generate_config->is_streaming   = true;
    input->generate_config->timeout_ms     = 10000;
    auto stream                            = engine->makeStream(input);
    if (server_role == RoleType::DECODE) {
        // Model the RPC allocation boundary without RPC or KV transfer: allocate
        // prompt blocks before publishing an ordinary decode stream to the scheduler.
        stream->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(stream->moveToNext(), StreamState::WAITING);
        ASSERT_FALSE(stream->hasError());
        stream->setIsContextStream(false);
    }
    engine->enqueue(stream);
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_FALSE(stream->nextOutput().ok());
    ASSERT_TRUE(engine->stop().ok());
    ExecutionRecorder::instance().close();
    std::ifstream file(std::string(directory) + "/batch_schedule.log");
    ASSERT_TRUE(file.good());
    const std::string batches((std::istreambuf_iterator<char>(file)), {});
    // Exercise the production snapshot path: nanosecond-sized IDs must remain
    // distinct decimal strings after a JSON round trip, even above 2^53.
    std::istringstream    lines(batches);
    std::set<std::string> execution_ids;
    for (std::string line; std::getline(lines, line);) {
        if (line.empty())
            continue;
        const auto record = autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(line));
        const auto id     = autil::legacy::AnyCast<std::string>(record.at("exec_id"));
        EXPECT_GT(std::stoll(id), 9007199254740991LL);
        EXPECT_EQ(std::to_string(std::stoll(id)), id);
        EXPECT_TRUE(execution_ids.insert(id).second);
        EXPECT_EQ(autil::legacy::AnyCast<std::string>(
                      autil::legacy::json::ParseJson(ExecutionRecorder::toJson(record.at("exec_id")))),
                  id);
    }
    const auto* decode_env        = std::getenv("RTP_LLM_RECORD_DECODE");
    const bool  record_decode_env = decode_env && std::string(decode_env) == "1";
    const bool  record_decode     = record_decode_env || capture_decode;
    EXPECT_EQ(execution_ids.empty(), server_role == RoleType::DECODE && !record_decode);
    if (tensor_capture) {
        const bool has_capture = capture_decode || (capture_prefill && server_role != RoleType::DECODE);
        EXPECT_EQ(snapshots.empty(), !has_capture);
        EXPECT_EQ(snapshots.size(),
                  (capture_prefill ? 1u : 0u) + (capture_decode ? (server_role == RoleType::DECODE ? 4u : 3u) : 0u));
        std::set<std::string> request_ids;
        int                   expected_decode_kv = server_role == RoleType::DECODE ? 2 : 3;
        for (const auto& snapshot : snapshots) {
            const auto context =
                autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(snapshot));
            const auto id = autil::legacy::AnyCast<std::string>(context.at("exec_id"));
            EXPECT_EQ(execution_ids.count(id), 1);
            EXPECT_EQ(autil::legacy::AnyCast<std::string>(context.at("session_id")), "engine-test");
            EXPECT_EQ(autil::legacy::AnyCast<std::string>(context.at("owner_id")),
                      ExecutionRecorder::instance().owner());
            EXPECT_EQ(autil::legacy::AnyCast<std::string>(context.at("model_role")), "target");
            EXPECT_EQ(autil::legacy::AnyCast<std::string>(context.at("server_role")), roleTypeToString(server_role));
            const auto requests = autil::legacy::AnyCast<ExecutionRecorder::JsonArray>(context.at("requests"));
            ASSERT_EQ(requests.size(), 1);
            const auto row   = autil::legacy::AnyCast<ExecutionRecorder::Json>(requests[0]);
            const auto phase = autil::legacy::AnyCast<std::string>(row.at("phase"));
            if (phase == "prefill") {
                EXPECT_TRUE(capture_prefill);
                EXPECT_EQ(ExecutionRecorder::toJson(row.at("q_len")), "3");
                EXPECT_EQ(ExecutionRecorder::toJson(row.at("kv_len")), "0");
            } else {
                EXPECT_EQ(phase, "decode");
                EXPECT_TRUE(capture_decode);
                EXPECT_EQ(ExecutionRecorder::toJson(row.at("q_len")), "1");
                EXPECT_EQ(std::stoi(ExecutionRecorder::toJson(row.at("kv_len"))), expected_decode_kv++);
            }
            EXPECT_EQ(row.count("request_id"), 1);
            request_ids.insert(autil::legacy::AnyCast<std::string>(row.at("request_id")));
            EXPECT_EQ(row.count("ttft_us"), 0);
        }
        if (has_capture) {
            EXPECT_EQ(request_ids.size(), 1);
        }
    }
    // Current optional fields are omitted for a single real sequence.
    for (const auto* field : {"sequence_id", "is_fake"}) {
        EXPECT_EQ(batches.find(std::string("\"") + field + "\":"), std::string::npos) << field << batches;
    }
    if (server_role != RoleType::DECODE || record_decode) {
        EXPECT_NE(batches.find("\"schema_version\":1"), std::string::npos) << batches;
        EXPECT_NE(batches.find("\"owner_id\":"), std::string::npos) << batches;
        EXPECT_NE(batches.find("\"exec_id\":"), std::string::npos) << batches;
        EXPECT_NE(batches.find("\"request_id\":"), std::string::npos) << batches;
        EXPECT_NE(batches.find("\"input_len\":3"), std::string::npos) << batches;
        EXPECT_NE(batches.find("\"reuse_len\":0"), std::string::npos) << batches;
        if (server_role != RoleType::DECODE) {
            EXPECT_NE(batches.find("\"q_len\":3"), std::string::npos) << batches;
            EXPECT_NE(batches.find("\"kv_len\":0"), std::string::npos) << batches;
        }
        EXPECT_NE(batches.find("\"ttft_us\":"), std::string::npos) << batches;
        EXPECT_EQ(batches.find("\"ttft_us\":null"), std::string::npos) << batches;
    }
    std::ifstream manifest_file(std::string(directory) + "/owner-" + ExecutionRecorder::instance().owner()
                                + "/manifest.json");
    ASSERT_TRUE(manifest_file.good());
    const std::string manifest_text((std::istreambuf_iterator<char>(manifest_file)), {});
    const auto        manifest =
        autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(manifest_text));
    const auto metadata = autil::legacy::AnyCast<ExecutionRecorder::Json>(manifest.at("engine"));
    EXPECT_EQ(autil::legacy::AnyCast<bool>(metadata.at("record_decode")), record_decode);
    EXPECT_EQ(autil::legacy::AnyCast<bool>(metadata.at("record_decode_env")), record_decode_env);
    EXPECT_EQ(autil::legacy::AnyCast<bool>(metadata.at("record_decode_capture")), capture_decode);
    EXPECT_EQ(autil::legacy::AnyCast<std::string>(metadata.at("tensor_capture_budget_scope")), "per_recorder_instance");
    EXPECT_EQ(autil::legacy::AnyCast<std::string>(metadata.at("decode_filter_policy")),
              record_decode_env ? "record_all_batches" :
                                  (capture_decode ? "record_pure_decode_for_capture_skip_mixed" :
                                                    "skip_entire_batch_containing_decode"));
    if (record_decode) {
        EXPECT_NE(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
        EXPECT_NE(batches.find("\"q_len\":1"), std::string::npos) << batches;
    } else {
        EXPECT_EQ(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
        EXPECT_EQ(batches.find("\"q_len\":1"), std::string::npos) << batches;
    }
    unsetenv("RTP_LLM_RECORD_DIR");
    unsetenv("RTP_LLM_RECORD_SESSION");
}

void RecordingEngineTest::runCaptureEligibility() {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    py::gil_scoped_acquire gil;
    struct CaptureInputs {
        GptModelInputs               model;
        torch_ext::PyAttentionInputs attention;
        ExecProperties               device;
        RoleType                     server_role   = RoleType::PDFUSION;
        bool                         graph_warmup  = false;
        bool                         graph_capture = false;
    };
    struct Case {
        const char*                         reason;
        std::function<void(CaptureInputs&)> apply;
    };
    const std::vector<Case> cases{
        {nullptr, [](CaptureInputs&) {}},
        {"unsupported_server_phase",
         [](CaptureInputs& x) { x.server_role = x.attention.is_prefill ? RoleType::DECODE : RoleType::PREFILL; }},
        {"unsupported_server_phase", [](CaptureInputs& x) { x.server_role = RoleType::VIT; }},
        {"unsupported_server_phase", [](CaptureInputs& x) { x.server_role = RoleType::FRONTEND; }},
        {"warmup", [](CaptureInputs& x) { x.model.warmup = true; }},
        {"warmup", [](CaptureInputs& x) { x.graph_warmup = true; }},
        {"graph", [](CaptureInputs& x) { x.graph_capture = true; }},
        {"graph", [](CaptureInputs& x) { x.attention.is_cuda_graph = true; }},
        {"graph", [](CaptureInputs& x) { x.attention.prefill_cuda_graph_copy_params.emplace(); }},
        {"microbatch", [](CaptureInputs& x) { x.device.enable_layer_micro_batch = MicroBatchType::DS_PREFILL; }},
        {"verify", [](CaptureInputs& x) { x.model.is_target_verify = true; }},
        {"verify", [](CaptureInputs& x) { x.attention.is_target_verify = true; }},
        {"draft", [](CaptureInputs& x) { x.model.is_draft_extend = true; }},
        {"draft", [](CaptureInputs& x) { x.attention.is_draft_extend = true; }},
        {"fake", [](CaptureInputs& x) { x.model.is_fake_stream = true; }},
        {"phase_mismatch", [](CaptureInputs& x) { x.attention.is_prefill = !x.attention.is_prefill; }},
        {"mixed_batch",
         [](CaptureInputs& x) {
             x.model.prefix_lengths   = torch::tensor({0}, torch::kInt32);
             x.model.sequence_lengths = torch::tensor({3}, torch::kInt32);
         }},
        {"unknown_lengths", [](CaptureInputs& x) { x.model.record_lengths.reset(); }},
        {"unknown_lengths", [](CaptureInputs& x) { x.model.record_lengths->kv_tokens_before.back() = -1; }},
        {"unknown_lengths", [](CaptureInputs& x) { x.model.record_lengths->q_tokens.back() = 0; }},
        {"unknown_lengths", [](CaptureInputs& x) { x.model.record_lengths->q_tokens.pop_back(); }},
        {"unknown_lengths", [](CaptureInputs& x) { x.model.record_lengths->kv_tokens_before.pop_back(); }},
        {"non_owner", [](CaptureInputs& x) { x.device.tp_rank = 1; }},
        {"empty", [](CaptureInputs& x) { x.model.input_lengths = torch::empty({0}, torch::kInt32); }},
        {"empty", [](CaptureInputs& x) { x.model.combo_tokens = torch::empty({0}, torch::kInt32); }},
        {"missing_context", [](CaptureInputs& x) { x.model.tensor_capture_context_json.clear(); }},
    };
    for (const auto role : {RoleType::PREFILL, RoleType::PDFUSION}) {
        for (bool cp_enabled : {false, true}) {
            for (bool cp_metadata : {false, true}) {
                for (const auto& test : cases) {
                    SCOPED_TRACE(roleTypeToString(role));
                    SCOPED_TRACE(cp_enabled);
                    SCOPED_TRACE(cp_metadata);
                    SCOPED_TRACE(test.reason ? test.reason : "eligible");
                    CaptureInputs inputs;
                    inputs.model.input_lengths               = torch::tensor({3}, torch::kInt32);
                    inputs.model.sequence_lengths            = torch::empty({0}, torch::kInt32);
                    inputs.model.combo_tokens                = torch::tensor({1, 2, 3}, torch::kInt32);
                    inputs.model.prefix_lengths              = torch::tensor({0}, torch::kInt32);
                    inputs.model.record_lengths              = GptModelInputs::RecordedLengths{{3}, {0}};
                    inputs.model.tensor_capture_context_json = "frozen-owner-context";
                    inputs.attention.is_prefill              = true;
                    inputs.server_role                       = role;
                    inputs.device.enable_prefill_cp          = cp_enabled;
                    inputs.device.tp_size                    = cp_enabled ? 8 : 1;
                    if (cp_metadata) {
                        inputs.attention.context_parallel_info.emplace();
                    }
                    test.apply(inputs);
                    const auto* reason = PyWrappedModel::tensorCaptureSkipReason(inputs.model,
                                                                                 inputs.attention,
                                                                                 inputs.device,
                                                                                 inputs.server_role,
                                                                                 inputs.graph_warmup,
                                                                                 inputs.graph_capture);
                    EXPECT_STREQ(reason, test.reason);
                }
            }
        }
    }
    auto decode_cases = cases;
    decode_cases.push_back(
        {"unsupported_decode_q_len", [](CaptureInputs& x) { x.model.record_lengths->q_tokens.back() = 2; }});
    decode_cases.push_back({"unsupported_decode_q_len",
                            [](CaptureInputs& x) { x.model.combo_tokens = torch::tensor({1, 2, 3}, torch::kInt32); }});
    decode_cases.push_back({"cp_decode_unsupported", [](CaptureInputs& x) { x.device.enable_prefill_cp = true; }});
    decode_cases.push_back(
        {"cp_decode_unsupported", [](CaptureInputs& x) { x.attention.context_parallel_info.emplace(); }});
    for (const auto role : {RoleType::DECODE, RoleType::PDFUSION}) {
        for (int dp_rank = 0; dp_rank < 8; ++dp_rank) {
            for (const auto& test : decode_cases) {
                SCOPED_TRACE(roleTypeToString(role));
                SCOPED_TRACE(dp_rank);
                SCOPED_TRACE(test.reason ? test.reason : "eligible");
                CaptureInputs inputs;
                inputs.model.input_lengths               = torch::tensor({3, 4}, torch::kInt32);
                inputs.model.sequence_lengths            = torch::tensor({3, 5}, torch::kInt32);
                inputs.model.prefix_lengths              = torch::empty({0}, torch::kInt32);
                inputs.model.combo_tokens                = torch::tensor({1, 2}, torch::kInt32);
                inputs.model.record_lengths              = GptModelInputs::RecordedLengths{{1, 1}, {3, 5}};
                inputs.model.tensor_capture_context_json = "frozen-owner-decode-context";
                inputs.attention.is_prefill              = false;
                inputs.server_role                       = role;
                ParallelismConfig parallel;
                parallel.role_type = role;
                parallel.dp_size = parallel.world_size = 8;
                parallel.dp_rank = parallel.world_rank     = dp_rank;
                parallel.prefill_cp_config.method          = CPRotateMethod::PREFILL_CP;
                parallel.prefill_cp_config.prefill_cp_size = 8;
                inputs.device                              = buildExecProperties(parallel, DeviceResourceConfig{});
                EXPECT_FALSE(inputs.device.enable_prefill_cp);
                EXPECT_EQ(inputs.device.tp_rank, 0);
                test.apply(inputs);
                EXPECT_STREQ(PyWrappedModel::tensorCaptureSkipReason(inputs.model,
                                                                     inputs.attention,
                                                                     inputs.device,
                                                                     inputs.server_role,
                                                                     inputs.graph_warmup,
                                                                     inputs.graph_capture),
                             test.reason);
            }
        }
    }
}

void RecordingEngineTest::runCaptureContextPreparation() {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    DeviceTestBase::SetUp();
    py::gil_scoped_acquire gil;
    py::module_::import("recording_engine_test_ops");
    auto     module            = py::module_::import("rtp_llm.utils.tensor_capture");
    auto     original_recorder = module.attr("TensorCaptureRecorder").cast<py::object>();
    py::dict scope;
    py::exec(R"(
contexts = []
phases = []
skips = []
class Recorder:
    def __init__(self, config_path, *, tp_rank, world_rank):
        pass
    def for_execution(self, context, *, phase):
        assert phase in ('prefill', 'decode')
        contexts.append(context)
        phases.append(phase)
        return {'captured': context}
    def note_skip(self, reason):
        skips.append(reason)
    def close(self, timeout=5.0):
        return True
class Model:
    def initialize(self, resources):
        return True
model = Model()
)",
             scope);
    module.attr("TensorCaptureRecorder") = scope["Recorder"];
    try {
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
        params.hw_kernel_config.enable_cuda_graph = false;
        char directory[]                          = "/tmp/rtp-capture-context-XXXXXX";
        if (!mkdtemp(directory)) {
            throw std::runtime_error("failed to create context test directory");
        }
        params.profiling_debug_logging_config.tensor_capture_config = writeCaptureConfig(directory, true);
        for (const auto role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
            params.pd_sep_config.role_type      = role;
            params.parallelism_config.role_type = role;
            NormalExecutor::test_model_factory  = [&](const GptModelInitParams& model_params) {
                auto model = std::make_unique<PyWrappedModel>(model_params, scope["model"].cast<py::object>());
                for (bool decode : {false, true}) {
                    if ((decode && role == RoleType::PREFILL) || (!decode && role == RoleType::DECODE)) {
                        continue;
                    }
                    for (bool graph : {false, true}) {
                        for (bool known_lengths : {false, true}) {
                            SCOPED_TRACE(roleTypeToString(role));
                            SCOPED_TRACE(decode);
                            SCOPED_TRACE(graph);
                            SCOPED_TRACE(known_lengths);
                            scope["contexts"].attr("clear")();
                            scope["phases"].attr("clear")();
                            scope["skips"].attr("clear")();
                            GptModelInputs inputs;
                            inputs.input_lengths    = torch::tensor({3}, torch::kInt32);
                            inputs.sequence_lengths = torch::empty({decode ? 1 : 0}, torch::kInt32);
                            inputs.prefix_lengths   = torch::empty({decode ? 0 : 1}, torch::kInt32);
                            inputs.combo_tokens     = torch::empty({decode ? 1 : 3}, torch::kInt32);
                            inputs.record_lengths   = GptModelInputs::RecordedLengths{
                                   {decode ? 1 : 3}, {known_lengths ? (decode ? 3 : 0) : -1}};
                            inputs.tensor_capture_context_json = "frozen-before-forward";
                            torch_ext::PyAttentionInputs attention;
                            attention.is_prefill    = !decode;
                            attention.is_cuda_graph = graph;
                            if (!decode) {
                                attention.context_parallel_info.emplace();
                            }
                            model->prepareTensorCaptureContext(inputs, attention);
                            const auto contexts = scope["contexts"].cast<std::vector<std::string>>();
                            const auto phases   = scope["phases"].cast<std::vector<std::string>>();
                            const auto skips    = scope["skips"].cast<std::vector<std::string>>();
                            if (graph || !known_lengths) {
                                EXPECT_TRUE(attention.tensor_capture_context.is_none());
                                EXPECT_TRUE(contexts.empty());
                                EXPECT_TRUE(phases.empty());
                                EXPECT_EQ(skips, (std::vector<std::string>{graph ? "graph" : "unknown_lengths"}));
                            } else {
                                EXPECT_FALSE(attention.tensor_capture_context.is_none());
                                EXPECT_EQ(contexts, (std::vector<std::string>{inputs.tensor_capture_context_json}));
                                EXPECT_EQ(phases, (std::vector<std::string>{decode ? "decode" : "prefill"}));
                                EXPECT_TRUE(skips.empty());
                            }
                            EXPECT_TRUE(model->attention_inputs_.tensor_capture_context.is_none());
                        }
                    }
                }
                return model;
            };
            NormalExecutor executor(params, nullptr, /*warm_up=*/true);
            NormalExecutor::test_model_factory = nullptr;
        }
    } catch (...) {
        NormalExecutor::test_model_factory   = nullptr;
        module.attr("TensorCaptureRecorder") = original_recorder;
        throw;
    }
    module.attr("TensorCaptureRecorder") = original_recorder;
}

void RecordingEngineTest::runCaptureWithoutRecorder(bool enabled) {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    DeviceTestBase::SetUp();
    unsetenv("RTP_LLM_RECORD_DIR");
    unsetenv("RTP_LLM_RECORD_SESSION");
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params      = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    char          directory[] = "/tmp/rtp-capture-config-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    params.profiling_debug_logging_config.tensor_capture_config = writeCaptureConfig(directory, enabled);
    EXPECT_EQ(PyWrappedModel::tensorCaptureConfigEnabled(params.profiling_debug_logging_config.tensor_capture_config),
              enabled);
    EXPECT_FALSE(PyWrappedModel::tensorCaptureConfigEnabled(""));
    EXPECT_FALSE(PyWrappedModel::tensorCaptureConfigEnabled("", "decode"));
    EXPECT_EQ(PyWrappedModel::tensorCaptureConfigEnabled(params.profiling_debug_logging_config.tensor_capture_config,
                                                         "prefill"),
              enabled);
    EXPECT_FALSE(PyWrappedModel::tensorCaptureConfigEnabled(params.profiling_debug_logging_config.tensor_capture_config,
                                                            "decode"));
    if (enabled) {
        EXPECT_THROW(NormalExecutor(params, nullptr), std::runtime_error);
    } else {
        EXPECT_NO_THROW(NormalExecutor(params, nullptr));
    }
}

void RecordingEngineTest::runCaptureConstructorFailure() {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    DeviceTestBase::SetUp();
    py::gil_scoped_acquire gil;
    // Use the production registrations before C++ passes init_resources to Python.
    auto     test_ops          = py::module_::import("recording_engine_test_ops");
    auto     module            = py::module_::import("rtp_llm.utils.tensor_capture");
    auto     original_recorder = module.attr("TensorCaptureRecorder").cast<py::object>();
    py::dict scope;
    scope["PyModelInitResources"] = test_ops.attr("PyModelInitResources");
    py::exec(R"(
import ctypes
events = []
fail_close = False
class Recorder:
    def __init__(self, config_path, *, tp_rank, world_rank):
        events.append('created')
    def close(self, timeout=5.0):
        assert ctypes.pythonapi.PyGILState_Check() == 1
        assert timeout == 5.0
        events.append('closed')
        if fail_close:
            raise RuntimeError('close-failure')
        return True
    def __del__(self):
        events.append('destroyed')
class Model:
    def initialize(self, resources):
        assert isinstance(resources, PyModelInitResources)
        events.append('initialized')
        raise RuntimeError('expected-model-init-failure')
model = Model()
)",
             scope);
    module.attr("TensorCaptureRecorder") = scope["Recorder"];
    try {
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
        char          directory[] = "/tmp/rtp-capture-constructor-XXXXXX";
        if (!mkdtemp(directory)) {
            throw std::runtime_error("failed to create constructor test directory");
        }
        params.profiling_debug_logging_config.tensor_capture_config = writeCaptureConfig(directory, false);
        params.py_model                                             = scope["model"].cast<py::object>();
        for (bool fail_close : {false, true}) {
            scope["fail_close"] = fail_close;
            scope["events"].attr("clear")();
            try {
                py::gil_scoped_release release;
                NormalExecutor         executor(params, nullptr, /*warm_up=*/true);
                ADD_FAILURE() << "model initialization should fail";
            } catch (const py::error_already_set& e) {
                EXPECT_NE(std::string(e.what()).find("expected-model-init-failure"), std::string::npos);
            }
            EXPECT_EQ(scope["events"].cast<std::vector<std::string>>(),
                      (std::vector<std::string>{"created", "initialized", "closed", "destroyed"}));
        }
    } catch (...) {
        module.attr("TensorCaptureRecorder") = original_recorder;
        throw;
    }
    module.attr("TensorCaptureRecorder") = original_recorder;
}

TEST_F(RecordingEngineTest, CaptureConstructorFailureClosesUnderGil) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runCaptureConstructorFailure();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, DisabledCaptureDoesNotRequireRecorder) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runCaptureWithoutRecorder(false);
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, CaptureRequiresReadyRecorder) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runCaptureWithoutRecorder();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, CaptureContextUsesRecordedBatch) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            // Decode batch recording must not enable decode tensor capture.
            setenv("RTP_LLM_RECORD_DECODE", "1", 1);
            runLifecycleAndSnapshot(true);
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, PrefillServerCaptureContextUsesRecordedBatch) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runLifecycleAndSnapshot(true, RoleType::PREFILL);
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, DecodeRecordingAndCapturePhaseMatrix) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    for (const auto role : {RoleType::DECODE, RoleType::PDFUSION}) {
        for (const auto* record_decode : {"0", "1"}) {
            for (const auto* phases : {"disabled", "prefill", "decode", "both"}) {
                SCOPED_TRACE(roleTypeToString(role));
                SCOPED_TRACE(record_decode);
                SCOPED_TRACE(phases);
                // Each cell re-execs: recordDecodeEnabled() and the recorder singleton
                // deliberately cache configuration for the lifetime of the process.
                ASSERT_EXIT(
                    {
                        setenv("RTP_LLM_RECORD_DECODE", record_decode, 1);
                        runLifecycleAndSnapshot(std::string(phases) != "disabled", role, phases);
                        _exit(::testing::Test::HasFailure() ? 1 : 0);
                    },
                    ::testing::ExitedWithCode(0),
                    "");
            }
        }
    }
}

TEST_F(RecordingEngineTest, CaptureEligibilityPreservesSafetyGatesWithCp) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runCaptureEligibility();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, CaptureContextPreparationAllowsCpAndOrdinaryDecode) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runCaptureContextPreparation();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, LifecycleAndSnapshot) {
    // Re-exec rather than fork with an initialized CUDA runtime. Each repeat
    // gets fresh recorder and alog state; failures in the child reach the parent.
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runLifecycleAndSnapshot();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

}  // namespace rtp_llm
