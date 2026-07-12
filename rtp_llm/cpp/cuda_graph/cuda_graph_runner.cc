#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/DecodeProbeTrigger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
using namespace torch_ext;

namespace rtp_llm {

namespace {

std::atomic<uint64_t>                    g_decode_checksum_record_id{0};
std::mutex                               g_decode_checksum_mutex;
std::mutex                               g_decode_checksum_trace_progress_mutex;
std::unordered_map<std::string, int64_t> g_decode_checksum_trace_start_prefix;
constexpr size_t                         kMaxDecodeChecksumTraceProgressEntries = 65536;

bool envFlag(const char* name, bool default_value = false) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return !(value.empty() || value == "0" || value == "false" || value == "off" || value == "no");
}

int64_t envInt64(const char* name, int64_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    auto  v   = std::strtoll(raw, &end, 10);
    return end == raw ? default_value : v;
}

std::string envString(const char* name, const std::string& default_value = "") {
    const char* raw = std::getenv(name);
    return raw == nullptr ? default_value : std::string(raw);
}

std::string filenameComponent(std::string value) {
    for (char& c : value) {
        const auto ch = static_cast<unsigned char>(c);
        if (!std::isalnum(ch) && c != '-' && c != '_' && c != '.') {
            c = '_';
        }
    }
    return value.empty() ? "unknown" : value;
}

std::vector<std::string> splitCsv(const std::string& value) {
    std::vector<std::string> out;
    std::stringstream        ss(value);
    std::string              item;
    while (std::getline(ss, item, ',')) {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char c) {
                       return !std::isspace(c);
                   }));
        item.erase(std::find_if(item.rbegin(),
                                item.rend(),
                                [](unsigned char c) {
                                    return !std::isspace(c);
                                })
                       .base(),
                   item.end());
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

void ensureDir(const std::string& dir) {
    if (dir.empty()) {
        return;
    }
    std::string current;
    current.reserve(dir.size());
    for (char c : dir) {
        current.push_back(c);
        if (c != '/') {
            continue;
        }
        if (current.size() > 1) {
            mkdir(current.c_str(), 0755);
        }
    }
    mkdir(dir.c_str(), 0755);
}

std::string parentDir(const std::string& path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return "";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream os;
    for (char c : value) {
        switch (c) {
            case '\\':
                os << "\\\\";
                break;
            case '"':
                os << "\\\"";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\r':
                os << "\\r";
                break;
            case '\t':
                os << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    os << c;
                }
        }
    }
    return os.str();
}

uint64_t fnv1a(const void* data, size_t bytes) {
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime  = 1099511628211ull;
    uint64_t           hash    = kOffset;
    const auto*        ptr     = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= ptr[i];
        hash *= kPrime;
    }
    return hash;
}

std::string hex64(uint64_t value) {
    std::ostringstream os;
    os << std::hex << std::setw(16) << std::setfill('0') << value;
    return os.str();
}

struct DecodeChecksumConfig {
    bool                     enabled{false};
    bool                     sync_device{true};
    int64_t                  every{1};
    int64_t                  max_records{0};
    int64_t                  max_lanes{8};
    int64_t                  max_output_steps_per_trace{0};
    std::string              file;
    std::vector<std::string> trace_filters;
    bool                     graph_probe_enabled{false};
};

const DecodeChecksumConfig& decodeChecksumConfig() {
    static DecodeChecksumConfig config = [] {
        DecodeChecksumConfig c;
        c.enabled       = envFlag("RTPLLM_DECODE_CHECKSUM_DEBUG", false);
        c.sync_device   = envFlag("RTPLLM_DECODE_CHECKSUM_SYNC_DEVICE", true);
        c.every         = std::max<int64_t>(1, envInt64("RTPLLM_DECODE_CHECKSUM_EVERY", 1));
        c.max_records   = envInt64("RTPLLM_DECODE_CHECKSUM_MAX_RECORDS", 0);
        c.max_lanes     = envInt64("RTPLLM_DECODE_CHECKSUM_MAX_LANES", 8);
        c.max_output_steps_per_trace =
            envInt64("RTPLLM_DECODE_CHECKSUM_MAX_OUTPUT_STEPS_PER_TRACE", 0);
        c.trace_filters = splitCsv(envString("RTPLLM_DECODE_CHECKSUM_TRACE_FILTER"));
        c.graph_probe_enabled = c.enabled && envFlag("RTPLLM_QWEN3_NEXT_GRAPH_PROBE", false);

        if (!c.enabled) {
            return c;
        }

        c.file = envString("RTPLLM_DECODE_CHECKSUM_FILE");
        if (c.file.empty()) {
            auto dir = envString("RTPLLM_DECODE_CHECKSUM_DIR", "/tmp/rtpllm_decode_checksum");
            ensureDir(dir);
            auto rank = envString("WORLD_RANK", "unknown");
            c.file    = dir + "/decode_checksum_rank" + rank + "_pid" + std::to_string(getpid()) + ".jsonl";
        } else {
            ensureDir(parentDir(c.file));
        }
        return c;
    }();
    return config;
}

bool traceSelected(const std::string& trace_id, const std::vector<std::string>& filters) {
    if (filters.empty()) {
        return true;
    }
    return std::any_of(filters.begin(), filters.end(), [&trace_id](const std::string& filter) {
        return trace_id.find(filter) != std::string::npos;
    });
}

bool hostScalarAt(const torch::Tensor& tensor, int64_t lane, int64_t& value) {
    if (!tensor.defined() || tensor.is_cuda() || tensor.dim() != 1 || lane < 0 || lane >= tensor.size(0)) {
        return false;
    }
    value = tensor[lane].item<int64_t>();
    return true;
}

bool outputStepAt(const PyModelInputs& inputs,
                  int64_t              lane,
                  const std::string&   trace_id,
                  int64_t&             output_step) {
    int64_t input_length    = 0;
    int64_t sequence_length = 0;
    if (hostScalarAt(inputs.attention_inputs.input_lengths, lane, input_length)
        && hostScalarAt(inputs.attention_inputs.sequence_lengths, lane, sequence_length)) {
        output_step = sequence_length - input_length;
        return true;
    }

    int64_t prefix_length = 0;
    if (trace_id.empty() || !hostScalarAt(inputs.attention_inputs.prefix_lengths, lane, prefix_length)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_decode_checksum_trace_progress_mutex);
    auto                        it = g_decode_checksum_trace_start_prefix.find(trace_id);
    if (it == g_decode_checksum_trace_start_prefix.end()) {
        if (g_decode_checksum_trace_start_prefix.size() >= kMaxDecodeChecksumTraceProgressEntries) {
            return false;
        }
        it = g_decode_checksum_trace_start_prefix.emplace(trace_id, prefix_length).first;
    }
    if (prefix_length < it->second) {
        it->second = prefix_length;
    }
    output_step = prefix_length - it->second;
    return true;
}

std::vector<int64_t> selectedLanes(const PyModelInputs&         inputs,
                                   int64_t                      current_bs,
                                   const DecodeChecksumConfig& config) {
    std::vector<int64_t> lanes;
    const int64_t        limit = config.max_lanes <= 0 ? current_bs : std::min<int64_t>(current_bs, config.max_lanes);
    for (int64_t lane = 0; lane < current_bs; ++lane) {
        const std::string trace_id =
            lane < static_cast<int64_t>(inputs.trace_ids.size()) ? inputs.trace_ids[lane] : "";
        if (!traceSelected(trace_id, config.trace_filters)) {
            continue;
        }
        if (config.max_output_steps_per_trace > 0) {
            int64_t output_step = 0;
            if (!outputStepAt(inputs, lane, trace_id, output_step)
                || output_step >= config.max_output_steps_per_trace) {
                continue;
            }
        }
        if (static_cast<int64_t>(lanes.size()) < limit) {
            lanes.push_back(lane);
        }
    }
    return lanes;
}

torch::Tensor selectLaneTensor(const torch::Tensor& tensor, int64_t lane) {
    if (!tensor.defined() || tensor.numel() <= 0) {
        return torch::Tensor();
    }
    if (tensor.dim() == 0) {
        return tensor;
    }
    if (lane < 0 || lane >= tensor.size(0)) {
        return torch::Tensor();
    }
    if (tensor.dim() == 1) {
        return tensor.slice(0, lane, lane + 1);
    }
    return tensor.select(0, lane);
}

torch::Tensor sliceTensorRange(const torch::Tensor& tensor, int64_t begin, int64_t end) {
    if (!tensor.defined() || tensor.numel() <= 0 || tensor.dim() == 0) {
        return torch::Tensor();
    }
    begin = std::max<int64_t>(0, std::min<int64_t>(begin, tensor.size(0)));
    end   = std::max<int64_t>(begin, std::min<int64_t>(end, tensor.size(0)));
    if (begin == end) {
        return torch::Tensor();
    }
    return tensor.slice(0, begin, end);
}

void appendShape(std::ostringstream& os, const torch::Tensor& tensor) {
    os << "[";
    for (int i = 0; i < tensor.dim(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << tensor.size(i);
    }
    os << "]";
}

void appendJsonFloat(std::ostringstream& os, double value) {
    if (std::isfinite(value)) {
        os << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    } else {
        os << "null";
    }
}

void appendTensorStats(std::ostringstream& os, const std::string& name, const torch::Tensor& tensor) {
    os << "\"" << name << "\":";
    if (!tensor.defined()) {
        os << "null";
        return;
    }

    try {
        torch::NoGradGuard no_grad;
        auto cpu = tensor.detach().contiguous();
        if (cpu.is_cuda()) {
            cpu = cpu.to(torch::kCPU);
        }

        os << "{\"shape\":";
        appendShape(os, tensor);
        os << ",\"dtype\":\"" << jsonEscape(c10::toString(tensor.scalar_type())) << "\"";
        os << ",\"device\":\"" << (tensor.is_cuda() ? "cuda" : "cpu") << "\"";
        os << ",\"numel\":" << tensor.numel();
        os << ",\"nbytes\":" << cpu.nbytes();
        os << ",\"byte_hash\":\"" << hex64(fnv1a(cpu.data_ptr(), cpu.nbytes())) << "\"";

        if (cpu.numel() > 0) {
            auto flat = cpu.reshape({cpu.numel()}).to(torch::kFloat32);
            os << ",\"sum\":";
            appendJsonFloat(os, flat.sum().item<double>());
            os << ",\"abs_sum\":";
            appendJsonFloat(os, flat.abs().sum().item<double>());
            os << ",\"min\":";
            appendJsonFloat(os, flat.min().item<double>());
            os << ",\"max\":";
            appendJsonFloat(os, flat.max().item<double>());
            os << ",\"sample\":[";
            const auto sample_count = std::min<int64_t>(flat.numel(), 8);
            const auto* data        = flat.data_ptr<float>();
            for (int64_t i = 0; i < sample_count; ++i) {
                if (i != 0) {
                    os << ",";
                }
                appendJsonFloat(os, data[i]);
            }
            os << "]";
        }
        os << "}";
    } catch (const std::exception& e) {
        os << "{\"error\":\"" << jsonEscape(e.what()) << "\"}";
    }
}

struct CudaGraphProbeDebugStatus {
    struct RecordDebug {
        int64_t attempts{0};
        int64_t recorded{0};
        int64_t skipped_not_cuda_graph{0};
        int64_t skipped_invalid_tensor{0};
        int64_t skipped_invalid_layout{0};
        int64_t last_layer_idx{-1};
        int64_t last_graph_bs{-1};
        int64_t last_token_rows{-1};
        int64_t last_residual_rows{-1};
        int64_t last_is_cuda_graph{-1};
    };

    struct PythonStatus {
        bool                 available{false};
        bool                 module_env_enabled{false};
        bool                 probe_created{false};
        bool                 buffer_available{false};
        std::vector<int64_t> layers;
        std::vector<int64_t> buffer_bucket_bs;
        RecordDebug          record_debug;
    };

    bool                 cpp_enabled{false};
    bool                 has_buffer_getter{false};
    bool                 has_status_getter{false};
    std::string          reason{"not_collected"};
    std::string          python_type;
    std::string          error;
    PythonStatus         python_status;
};

struct CudaGraphProbeCapture {
    torch::Tensor                              buffer;
    std::vector<int64_t>                       layers;
    std::unique_ptr<CudaGraphProbeDebugStatus> status;
};

struct DecodeChecksumRecord {
    bool                 selected{false};
    uint64_t             id{0};
    std::vector<int64_t> lanes;
};

DecodeChecksumRecord nextDecodeChecksumRecord(const PyModelInputs& inputs, int64_t current_bs) {
    const auto& config = decodeChecksumConfig();
    if (!config.enabled) {
        return {};
    }

    const uint64_t record_id = g_decode_checksum_record_id.fetch_add(1, std::memory_order_relaxed);
    if (config.max_records > 0 && record_id >= static_cast<uint64_t>(config.max_records)) {
        return {};
    }
    if (record_id % static_cast<uint64_t>(config.every) != 0) {
        return {};
    }
    auto lanes = selectedLanes(inputs, current_bs, config);
    if (lanes.empty()) {
        return {};
    }
    return {true, record_id, std::move(lanes)};
}

void appendGraphProbe(std::ostringstream&            os,
                      const CudaGraphProbeCapture&   graph_probe,
                      int64_t                        lane) {
    os << "\"graph_probe\":";
    if (!graph_probe.buffer.defined() || graph_probe.buffer.dim() != 3 || lane < 0
        || lane >= graph_probe.buffer.size(1)) {
        os << "null";
        return;
    }

    try {
        auto cpu = graph_probe.buffer.select(1, lane).detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        static const std::vector<std::string> fields = {
            "sum",
            "abs_sum",
            "square_sum",
            "min",
            "max",
            "nonfinite_count",
            "residual_sum",
            "residual_abs_sum",
            "residual_square_sum",
            "residual_min",
            "residual_max",
            "residual_nonfinite_count",
        };

        os << "{\"shape\":";
        appendShape(os, cpu);
        os << ",\"layers\":[";
        for (int64_t slot = 0; slot < cpu.size(0); ++slot) {
            if (slot != 0) {
                os << ",";
            }
            os << graph_probe.layers[slot];
        }
        os << "],\"fields\":[";
        for (size_t i = 0; i < fields.size(); ++i) {
            if (i != 0) {
                os << ",";
            }
            os << "\"" << fields[i] << "\"";
        }
        os << "],\"values\":[";
        const auto* data = cpu.data_ptr<float>();
        for (int64_t row = 0; row < cpu.size(0); ++row) {
            if (row != 0) {
                os << ",";
            }
            os << "[";
            for (int64_t col = 0; col < cpu.size(1); ++col) {
                if (col != 0) {
                    os << ",";
                }
                appendJsonFloat(os, data[row * cpu.size(1) + col]);
            }
            os << "]";
        }
        os << "]}";
    } catch (const std::exception& e) {
        os << "{\"error\":\"" << jsonEscape(e.what()) << "\"}";
    }
}

void appendInt64List(std::ostringstream& os, const std::vector<int64_t>& values) {
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << values[i];
    }
    os << "]";
}

void appendGraphProbeStatus(std::ostringstream& os, const CudaGraphProbeDebugStatus& status) {
    os << "{\"reason\":\"" << jsonEscape(status.reason) << "\"";
    os << ",\"cpp_enabled\":" << (status.cpp_enabled ? "true" : "false");
    os << ",\"has_buffer_getter\":" << (status.has_buffer_getter ? "true" : "false");
    os << ",\"has_status_getter\":" << (status.has_status_getter ? "true" : "false");
    os << ",\"python_type\":\"" << jsonEscape(status.python_type) << "\"";
    os << ",\"error\":";
    if (status.error.empty()) {
        os << "null";
    } else {
        os << "\"" << jsonEscape(status.error) << "\"";
    }
    os << ",\"python_status\":";
    if (!status.python_status.available) {
        os << "null";
    } else {
        os << "{\"module_env_enabled\":"
           << (status.python_status.module_env_enabled ? "true" : "false");
        os << ",\"probe_created\":" << (status.python_status.probe_created ? "true" : "false");
        os << ",\"buffer_available\":"
           << (status.python_status.buffer_available ? "true" : "false");
        os << ",\"layers\":";
        appendInt64List(os, status.python_status.layers);
        os << ",\"buffer_bucket_bs\":";
        appendInt64List(os, status.python_status.buffer_bucket_bs);
        const auto& debug = status.python_status.record_debug;
        os << ",\"record_debug\":{\"attempts\":" << debug.attempts;
        os << ",\"recorded\":" << debug.recorded;
        os << ",\"skipped_not_cuda_graph\":" << debug.skipped_not_cuda_graph;
        os << ",\"skipped_invalid_tensor\":" << debug.skipped_invalid_tensor;
        os << ",\"skipped_invalid_layout\":" << debug.skipped_invalid_layout;
        os << ",\"last_layer_idx\":" << debug.last_layer_idx;
        os << ",\"last_graph_bs\":" << debug.last_graph_bs;
        os << ",\"last_token_rows\":" << debug.last_token_rows;
        os << ",\"last_residual_rows\":" << debug.last_residual_rows;
        os << ",\"last_is_cuda_graph\":" << debug.last_is_cuda_graph << "}";
        os << "}";
    }
    os << "}";
}

CudaGraphProbeCapture getCudaGraphProbeBuffer(const py::object& py_instance, int64_t graph_bs) {
    const auto& config = decodeChecksumConfig();
    CudaGraphProbeCapture graph_probe;
    if (!config.enabled) {
        return graph_probe;
    }
    graph_probe.status = std::make_unique<CudaGraphProbeDebugStatus>();
    auto& status       = *graph_probe.status;
    status.cpp_enabled = config.graph_probe_enabled;
    if (!config.graph_probe_enabled) {
        status.reason = "cpp_disabled";
        return graph_probe;
    }

    try {
        status.python_type       = py::str(py::type::of(py_instance));
        status.has_buffer_getter = py::hasattr(py_instance, "get_cuda_graph_probe_buffer");
        status.has_status_getter = py::hasattr(py_instance, "get_cuda_graph_probe_debug_status");
    } catch (const std::exception& e) {
        status.reason = "python_introspection_error";
        status.error  = e.what();
        return graph_probe;
    }

    if (status.has_status_getter) {
        try {
            auto status_obj    = py_instance.attr("get_cuda_graph_probe_debug_status")(graph_bs);
            auto python_status = status_obj.cast<py::dict>();
            auto record_debug = python_status["record_debug"].cast<py::dict>();
            status.python_status.available          = true;
            status.python_status.module_env_enabled = python_status["module_env_enabled"].cast<bool>();
            status.python_status.probe_created      = python_status["probe_created"].cast<bool>();
            status.python_status.buffer_available   = python_status["buffer_available"].cast<bool>();
            status.python_status.layers             = python_status["layers"].cast<std::vector<int64_t>>();
            status.python_status.buffer_bucket_bs =
                python_status["buffer_bucket_bs"].cast<std::vector<int64_t>>();
            auto& debug                    = status.python_status.record_debug;
            debug.attempts                 = record_debug["attempts"].cast<int64_t>();
            debug.recorded                 = record_debug["recorded"].cast<int64_t>();
            debug.skipped_not_cuda_graph   = record_debug["skipped_not_cuda_graph"].cast<int64_t>();
            debug.skipped_invalid_tensor   = record_debug["skipped_invalid_tensor"].cast<int64_t>();
            debug.skipped_invalid_layout   = record_debug["skipped_invalid_layout"].cast<int64_t>();
            debug.last_layer_idx           = record_debug["last_layer_idx"].cast<int64_t>();
            debug.last_graph_bs            = record_debug["last_graph_bs"].cast<int64_t>();
            debug.last_token_rows          = record_debug["last_token_rows"].cast<int64_t>();
            debug.last_residual_rows       = record_debug["last_residual_rows"].cast<int64_t>();
            debug.last_is_cuda_graph       = record_debug["last_is_cuda_graph"].cast<int64_t>();
        } catch (const std::exception& e) {
            status.error = std::string("status_getter: ") + e.what();
        }
    }

    if (!status.has_buffer_getter) {
        status.reason = "buffer_getter_missing";
        return graph_probe;
    }
    try {
        auto result = py_instance.attr("get_cuda_graph_probe_buffer")(graph_bs);
        if (result.is_none()) {
            status.reason = "buffer_unavailable";
            return graph_probe;
        }
        auto capture = result.cast<py::tuple>();
        if (capture.size() != 2) {
            throw std::runtime_error("CUDA graph probe getter must return (buffer, layers)");
        }
        graph_probe.buffer = capture[0].cast<torch::Tensor>();
        graph_probe.layers = capture[1].cast<std::vector<int64_t>>();
        if (!graph_probe.buffer.defined() || graph_probe.buffer.dim() != 3
            || graph_probe.buffer.size(0) != static_cast<int64_t>(graph_probe.layers.size())) {
            throw std::runtime_error("CUDA graph probe buffer and layer metadata disagree");
        }
        status.reason = "ready";
        return graph_probe;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to get CUDA graph probe buffer: %s", e.what());
        status.reason = "buffer_getter_error";
        if (!status.error.empty()) {
            status.error += "; ";
        }
        status.error += e.what();
        graph_probe.buffer = torch::Tensor();
        graph_probe.layers.clear();
        return graph_probe;
    }
}

struct CudaGraphProbeRingConfig {
    bool        enabled{false};
    int64_t     max_records{0};
    int64_t     max_graph_bs{0};
    int64_t     max_bytes{0};
    int64_t     trigger_check_every{16};
    std::string rank;
    std::string trigger_file;
    std::string output_stem;
};

const CudaGraphProbeRingConfig& cudaGraphProbeRingConfig() {
    static const CudaGraphProbeRingConfig config = [] {
        CudaGraphProbeRingConfig c;
        c.enabled = envFlag("RTPLLM_CUDA_GRAPH_PROBE_RING_DEBUG", false)
                    && envFlag("RTPLLM_QWEN3_NEXT_GRAPH_PROBE", false);
        if (!c.enabled) {
            return c;
        }
        c.max_records = std::max<int64_t>(1, envInt64("RTPLLM_CUDA_GRAPH_PROBE_RING_MAX_RECORDS", 50000));
        c.max_graph_bs = std::max<int64_t>(1, envInt64("RTPLLM_CUDA_GRAPH_PROBE_RING_MAX_GRAPH_BS", 32));
        c.max_bytes = std::max<int64_t>(1, envInt64("RTPLLM_CUDA_GRAPH_PROBE_RING_MAX_BYTES", 2LL << 30));
        c.trigger_check_every =
            std::max<int64_t>(1, envInt64("RTPLLM_CUDA_GRAPH_PROBE_RING_TRIGGER_CHECK_EVERY", 16));
        const auto dir = envString("RTPLLM_CUDA_GRAPH_PROBE_RING_DIR", "/tmp/rtpllm_cuda_graph_probe_ring");
        ensureDir(dir);
        c.rank = filenameComponent(envString("WORLD_RANK", "unknown"));
        const auto generation = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
        c.trigger_file = envString("RTPLLM_CUDA_GRAPH_PROBE_RING_TRIGGER_FILE", dir + "/dump.trigger");
        c.output_stem = dir + "/cuda_graph_probe_ring_rank" + c.rank + "_pid" + std::to_string(getpid()) + "_gen"
                        + std::to_string(generation);
        return c;
    }();
    return config;
}

struct CudaGraphProbeRingMetadata {
    uint64_t                 record_id{0};
    int64_t                  current_bs{0};
    int64_t                  graph_bs{0};
    std::vector<std::string> trace_ids;
    std::vector<int64_t>     input_lengths;
    std::vector<int64_t>     sequence_lengths;
};

struct CudaGraphProbeBufferRef {
    torch::Tensor        buffer;
    std::vector<int64_t> layers;
};

struct CudaGraphProbeRingState {
    torch::Tensor                           buffer;
    std::vector<int64_t>                    layers;
    std::vector<CudaGraphProbeRingMetadata> metadata;
    std::unordered_map<int64_t, CudaGraphProbeBufferRef> graph_probes;
    int64_t                                 ring_id{-1};
    std::string                             tensor_file;
    std::string                             metadata_file;
    std::string                             completion_file;
    uint64_t                                total_records{0};
    bool                                    dumped{false};
    bool                                    disabled{false};
};

std::unordered_map<const void*, CudaGraphProbeRingState> g_cuda_graph_probe_rings;
std::atomic<int64_t>                                    g_cuda_graph_probe_ring_id{0};
std::mutex                                              g_cuda_graph_probe_ring_mutex;

struct CudaGraphRetrospectiveConfig {
    bool        enabled{false};
    bool        dual_graph{false};
    bool        eager_step{false};
    std::string rank;
    std::string output_dir;
};

const CudaGraphRetrospectiveConfig& cudaGraphRetrospectiveConfig() noexcept {
    static const CudaGraphRetrospectiveConfig config = [] {
        CudaGraphRetrospectiveConfig c;
        try {
            c.enabled = envFlag("RTPLLM_RETROSPECTIVE_PROBE_DEBUG", false)
                        && envFlag("RTPLLM_QWEN3_NEXT_GRAPH_PROBE", false);
            if (!c.enabled) {
                return c;
            }
            c.dual_graph = envFlag("RTPLLM_RETROSPECTIVE_DUAL_GRAPH_DEBUG", false);
            c.eager_step = envFlag("RTPLLM_RETROSPECTIVE_EAGER_STEP_DEBUG", false);
            c.rank = filenameComponent(envString("WORLD_RANK", "unknown"));
            c.output_dir =
                envString("RTPLLM_RETROSPECTIVE_PROBE_DIR", "/tmp/rtpllm_retrospective_probe");
            ensureDir(c.output_dir);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("Failed to initialize retrospective CUDA graph probe config: %s", e.what());
            c = {};
        } catch (...) {
            RTP_LLM_LOG_WARNING("Failed to initialize retrospective CUDA graph probe config");
            c = {};
        }
        return c;
    }();
    return config;
}

std::mutex g_cuda_graph_retrospective_mutex;

CudaGraphProbeCapture getCudaGraphProbeBufferLight(const py::object& py_instance, int64_t graph_bs) {
    CudaGraphProbeCapture graph_probe;
    try {
        if (!py::hasattr(py_instance, "get_cuda_graph_probe_buffer")) {
            return graph_probe;
        }
        auto result = py_instance.attr("get_cuda_graph_probe_buffer")(graph_bs);
        if (result.is_none()) {
            return graph_probe;
        }
        auto capture = result.cast<py::tuple>();
        if (capture.size() != 2) {
            return graph_probe;
        }
        graph_probe.buffer = capture[0].cast<torch::Tensor>();
        graph_probe.layers = capture[1].cast<std::vector<int64_t>>();
        if (!graph_probe.buffer.defined() || graph_probe.buffer.dim() != 3
            || graph_probe.buffer.size(0) != static_cast<int64_t>(graph_probe.layers.size())) {
            return {};
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to get lightweight CUDA graph probe buffer: %s", e.what());
        return {};
    }
    return graph_probe;
}

std::vector<int64_t> hostTensorPrefix(const torch::Tensor& tensor, int64_t count) {
    std::vector<int64_t> values;
    values.reserve(count);
    for (int64_t i = 0; i < count; ++i) {
        int64_t value = 0;
        values.push_back(hostScalarAt(tensor, i, value) ? value : -1);
    }
    return values;
}

void dumpCudaGraphProbeRing(CudaGraphProbeRingState& state) {
    const auto valid_records = std::min<uint64_t>(state.total_records, state.buffer.size(0));
    if (state.dumped || !state.buffer.defined() || valid_records == 0) {
        return;
    }
    const auto tensor_tmp     = state.tensor_file + ".tmp";
    const auto metadata_tmp   = state.metadata_file + ".tmp";
    const auto completion_tmp = state.completion_file + ".tmp";
    try {
        cuda_graph::graphDeviceSynchronize();
        const auto first_slot = state.total_records >= static_cast<uint64_t>(state.buffer.size(0))
                                    ? state.total_records % state.buffer.size(0)
                                    : 0;
        torch::Tensor cpu;
        if (first_slot == 0) {
            cpu = state.buffer.narrow(0, 0, valid_records).to(torch::kCPU).contiguous();
        } else {
            cpu = torch::empty({static_cast<int64_t>(valid_records),
                                state.buffer.size(1),
                                state.buffer.size(2),
                                state.buffer.size(3)},
                               state.buffer.options().device(torch::kCPU));
            const auto tail_records = state.buffer.size(0) - first_slot;
            cpu.narrow(0, 0, tail_records).copy_(state.buffer.narrow(0, first_slot, tail_records));
            cpu.narrow(0, tail_records, first_slot).copy_(state.buffer.narrow(0, 0, first_slot));
        }
        for (uint64_t i = 0; i < valid_records; ++i) {
            const auto& record = state.metadata[(first_slot + i) % state.buffer.size(0)];
            if (record.graph_bs < cpu.size(2)) {
                cpu[i].narrow(1, record.graph_bs, cpu.size(2) - record.graph_bs).zero_();
            }
        }
        std::ofstream tensor_out(tensor_tmp, std::ios::binary | std::ios::trunc);
        tensor_out.write(static_cast<const char*>(cpu.data_ptr()), cpu.nbytes());
        tensor_out.flush();
        if (!tensor_out.good()) {
            throw std::runtime_error("failed to write probe ring tensor file");
        }
        tensor_out.close();
        if (tensor_out.fail()) {
            throw std::runtime_error("failed to close probe ring tensor file");
        }

        std::ofstream out(metadata_tmp, std::ios::out | std::ios::trunc);
        for (uint64_t i = 0; i < valid_records; ++i) {
            const auto& record = state.metadata[(first_slot + i) % state.buffer.size(0)];
            out << "{\"rank\":\"" << jsonEscape(cudaGraphProbeRingConfig().rank) << "\",\"pid\":" << getpid()
                << ",\"runner_id\":" << state.ring_id << ",\"record_id\":" << record.record_id
                << ",\"current_bs\":" << record.current_bs
                << ",\"graph_bs\":" << record.graph_bs
                << ",\"ring_max_graph_bs\":" << state.buffer.size(2)
                << ",\"field_count\":" << state.buffer.size(3)
                << ",\"dtype\":\"float32\",\"layers\":[";
            for (size_t i = 0; i < state.layers.size(); ++i) {
                out << (i == 0 ? "" : ",") << state.layers[i];
            }
            out << "],\"trace_ids\":[";
            for (size_t i = 0; i < record.trace_ids.size(); ++i) {
                out << (i == 0 ? "" : ",") << "\"" << jsonEscape(record.trace_ids[i]) << "\"";
            }
            out << "],\"input_lengths\":[";
            for (size_t i = 0; i < record.input_lengths.size(); ++i) {
                out << (i == 0 ? "" : ",") << record.input_lengths[i];
            }
            out << "],\"sequence_lengths\":[";
            for (size_t i = 0; i < record.sequence_lengths.size(); ++i) {
                out << (i == 0 ? "" : ",") << record.sequence_lengths[i];
            }
            out << "]}\n";
        }
        if (!out.good()) {
            throw std::runtime_error("failed to write probe ring metadata file");
        }
        out.close();
        if (out.fail()) {
            throw std::runtime_error("failed to close probe ring metadata file");
        }
        if (rename(tensor_tmp.c_str(), state.tensor_file.c_str()) != 0
            || rename(metadata_tmp.c_str(), state.metadata_file.c_str()) != 0) {
            throw std::runtime_error("failed to publish probe ring data files");
        }
        std::ofstream completion_out(completion_tmp, std::ios::out | std::ios::trunc);
        completion_out << "{\"records\":" << valid_records << ",\"tensor\":\""
                       << jsonEscape(state.tensor_file) << "\",\"metadata\":\""
                       << jsonEscape(state.metadata_file) << "\"}\n";
        completion_out.flush();
        if (!completion_out.good()) {
            throw std::runtime_error("failed to write probe ring completion file");
        }
        completion_out.close();
        if (completion_out.fail() || rename(completion_tmp.c_str(), state.completion_file.c_str()) != 0) {
            throw std::runtime_error("failed to publish probe ring completion file");
        }
        state.dumped = true;
        RTP_LLM_LOG_WARNING("Dumped CUDA graph probe ring records=%lu tensor=%s metadata=%s completion=%s",
                            valid_records,
                            state.tensor_file.c_str(),
                            state.metadata_file.c_str(),
                            state.completion_file.c_str());
    } catch (const std::exception& e) {
        unlink(tensor_tmp.c_str());
        unlink(metadata_tmp.c_str());
        unlink(completion_tmp.c_str());
        RTP_LLM_LOG_WARNING("Failed to dump CUDA graph probe ring: %s", e.what());
        throw;
    }
}

void disableCudaGraphProbeRing(CudaGraphProbeRingState& state, const std::string& reason) {
    state.buffer = torch::Tensor();
    state.metadata.clear();
    state.graph_probes.clear();
    state.disabled = true;
    RTP_LLM_LOG_WARNING("Disabled CUDA graph probe ring after capture failure: %s", reason.c_str());
}

bool validCudaGraphProbeCapture(const CudaGraphProbeCapture& graph_probe) {
    return graph_probe.buffer.defined() && graph_probe.buffer.dim() == 3
           && graph_probe.buffer.size(0) == static_cast<int64_t>(graph_probe.layers.size());
}

void dumpCudaGraphPreviousReplay(const CudaGraphPreviousReplay& replay, const DecodeProbeTriggerEvent& event) {
    const auto& config = cudaGraphRetrospectiveConfig();
    const auto  stem = config.output_dir + "/cuda_graph_probe_retrospective_rank" + config.rank + "_pid"
                      + std::to_string(getpid()) + "_gen" + std::to_string(event.generation);
    const auto tensor_file     = stem + ".bin";
    const auto metadata_file   = stem + ".jsonl";
    const auto completion_file = stem + ".complete";
    const auto tensor_tmp      = tensor_file + ".tmp";
    const auto metadata_tmp    = metadata_file + ".tmp";
    const auto completion_tmp  = completion_file + ".tmp";
    try {
        cuda_graph::graphDeviceSynchronize();
        auto cpu = replay.probe_buffer.detach().to(torch::kCPU).contiguous();
        if (cpu.dim() >= 2) {
            const auto padded_begin = std::max<int64_t>(0, std::min<int64_t>(replay.current_bs, cpu.size(1)));
            if (padded_begin < cpu.size(1)) {
                cpu.slice(1, padded_begin, cpu.size(1)).zero_();
            }
        }

        std::ofstream tensor_out(tensor_tmp, std::ios::binary | std::ios::trunc);
        tensor_out.write(static_cast<const char*>(cpu.data_ptr()), cpu.nbytes());
        tensor_out.flush();
        if (!tensor_out.good()) {
            throw std::runtime_error("failed to write retrospective probe tensor file");
        }
        tensor_out.close();
        if (tensor_out.fail()) {
            throw std::runtime_error("failed to close retrospective probe tensor file");
        }

        std::ofstream out(metadata_tmp, std::ios::out | std::ios::trunc);
        out << "{\"rank\":\"" << jsonEscape(config.rank) << "\",\"pid\":" << getpid()
            << ",\"event_generation\":" << event.generation << ",\"event_trace_id\":\""
            << jsonEscape(event.trace_id) << "\",\"event_reason\":\"" << jsonEscape(event.reason)
            << "\",\"event_observed_sequence_length\":" << event.observed_sequence_length
            << ",\"replay_id\":" << replay.replay_id << ",\"graph_bs\":" << replay.graph_bs
            << ",\"current_bs\":" << replay.current_bs << ",\"nbytes\":" << cpu.nbytes() << ",\"dtype\":\""
            << jsonEscape(c10::toString(cpu.scalar_type())) << "\",\"shape\":[";
        for (int64_t i = 0; i < cpu.dim(); ++i) {
            out << (i == 0 ? "" : ",") << cpu.size(i);
        }
        out << "],\"layers\":[";
        for (size_t i = 0; i < replay.layers.size(); ++i) {
            out << (i == 0 ? "" : ",") << replay.layers[i];
        }
        out << "],\"lanes\":[";
        for (int64_t lane = 0; lane < replay.current_bs; ++lane) {
            const auto trace_id = lane < static_cast<int64_t>(replay.trace_ids.size()) ? replay.trace_ids[lane] : "";
            const auto input_length =
                lane < static_cast<int64_t>(replay.input_lengths.size()) ? replay.input_lengths[lane] : -1;
            const auto sequence_length =
                lane < static_cast<int64_t>(replay.sequence_lengths.size()) ? replay.sequence_lengths[lane] : -1;
            out << (lane == 0 ? "" : ",") << "{\"lane\":" << lane << ",\"trace_id\":\""
                << jsonEscape(trace_id) << "\",\"input_length\":" << input_length
                << ",\"sequence_length\":" << sequence_length << "}";
        }
        out << "]}\n";
        out.flush();
        if (!out.good()) {
            throw std::runtime_error("failed to write retrospective probe metadata file");
        }
        out.close();
        if (out.fail()) {
            throw std::runtime_error("failed to close retrospective probe metadata file");
        }

        if (rename(tensor_tmp.c_str(), tensor_file.c_str()) != 0
            || rename(metadata_tmp.c_str(), metadata_file.c_str()) != 0) {
            throw std::runtime_error("failed to publish retrospective probe data files");
        }
        std::ofstream completion_out(completion_tmp, std::ios::out | std::ios::trunc);
        completion_out << "{\"event_generation\":" << event.generation << ",\"tensor\":\""
                       << jsonEscape(tensor_file) << "\",\"metadata\":\"" << jsonEscape(metadata_file)
                       << "\"}\n";
        completion_out.flush();
        if (!completion_out.good()) {
            throw std::runtime_error("failed to write retrospective probe completion file");
        }
        completion_out.close();
        if (completion_out.fail() || rename(completion_tmp.c_str(), completion_file.c_str()) != 0) {
            throw std::runtime_error("failed to publish retrospective probe completion file");
        }
        RTP_LLM_LOG_WARNING("Dumped retrospective CUDA graph probe replay=%lu graph_bs=%ld trace=%s tensor=%s",
                            replay.replay_id,
                            replay.graph_bs,
                            event.trace_id.c_str(),
                            tensor_file.c_str());
    } catch (...) {
        unlink(tensor_tmp.c_str());
        unlink(metadata_tmp.c_str());
        unlink(completion_tmp.c_str());
        unlink(tensor_file.c_str());
        unlink(metadata_file.c_str());
        unlink(completion_file.c_str());
        throw;
    }
}

void releaseCudaGraphProbeRing(const void* runner_key) {
    if (!cudaGraphProbeRingConfig().enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_cuda_graph_probe_ring_mutex);
    g_cuda_graph_probe_rings.erase(runner_key);
}

void captureCudaGraphProbeRing(const void*                  runner_key,
                               const py::object&             py_instance,
                               const CudaGraphProbeCapture& supplied_graph_probe,
                               const PyModelInputs&         inputs,
                               const CudaGraphState&        graph_state) {
    const auto& config = cudaGraphProbeRingConfig();
    if (!config.enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_cuda_graph_probe_ring_mutex);
    try {
        auto& state = g_cuda_graph_probe_rings[runner_key];
        if (state.dumped || state.disabled) {
            return;
        }
        CudaGraphProbeCapture graph_probe;
        graph_probe.buffer = supplied_graph_probe.buffer;
        graph_probe.layers = supplied_graph_probe.layers;
        if (validCudaGraphProbeCapture(graph_probe)) {
            state.graph_probes[graph_state.current_real_graph_bs] = {graph_probe.buffer, graph_probe.layers};
        } else {
            if (graph_probe.buffer.defined()) {
                RTP_LLM_LOG_WARNING("Ignoring malformed supplied CUDA graph probe buffer");
                graph_probe.buffer = torch::Tensor();
                graph_probe.layers.clear();
            }
            const auto iter = state.graph_probes.find(graph_state.current_real_graph_bs);
            if (iter != state.graph_probes.end()) {
                graph_probe.buffer = iter->second.buffer;
                graph_probe.layers = iter->second.layers;
            } else {
                graph_probe = getCudaGraphProbeBufferLight(py_instance, graph_state.current_real_graph_bs);
                if (validCudaGraphProbeCapture(graph_probe)) {
                    state.graph_probes.emplace(graph_state.current_real_graph_bs,
                                               CudaGraphProbeBufferRef{graph_probe.buffer, graph_probe.layers});
                }
            }
        }
        if (!validCudaGraphProbeCapture(graph_probe)) {
            return;
        }
        if (!state.buffer.defined()) {
            if (graph_probe.buffer.size(1) > config.max_graph_bs) {
                RTP_LLM_LOG_WARNING("CUDA graph probe ring graph_bs=%ld exceeds configured max=%ld",
                                    graph_probe.buffer.size(1),
                                    config.max_graph_bs);
                return;
            }
            const __int128 requested_bytes = static_cast<__int128>(config.max_records)
                                               * graph_probe.buffer.size(0) * config.max_graph_bs
                                               * graph_probe.buffer.size(2) * graph_probe.buffer.element_size();
            if (requested_bytes > config.max_bytes) {
                disableCudaGraphProbeRing(
                    state,
                    "requested GPU bytes exceed RTPLLM_CUDA_GRAPH_PROBE_RING_MAX_BYTES="
                        + std::to_string(config.max_bytes));
                return;
            }
            state.ring_id = g_cuda_graph_probe_ring_id.fetch_add(1, std::memory_order_relaxed);
            const auto runner_suffix = "_runner" + std::to_string(state.ring_id);
            state.tensor_file         = config.output_stem + runner_suffix + ".bin";
            state.metadata_file       = config.output_stem + runner_suffix + ".jsonl";
            state.completion_file     = config.output_stem + runner_suffix + ".complete";
            state.layers = graph_probe.layers;
            state.buffer = torch::empty({config.max_records,
                                         graph_probe.buffer.size(0),
                                         config.max_graph_bs,
                                         graph_probe.buffer.size(2)},
                                        graph_probe.buffer.options());
            state.metadata.resize(config.max_records);
        }
        if (graph_probe.layers != state.layers || graph_probe.buffer.size(0) != state.buffer.size(1)
            || graph_probe.buffer.size(2) != state.buffer.size(3)
            || graph_probe.buffer.size(1) > state.buffer.size(2)) {
            RTP_LLM_LOG_WARNING("CUDA graph probe ring layout changed; skipping record");
            return;
        }

        const auto slot = state.total_records % config.max_records;
        state.buffer[slot].narrow(1, 0, graph_probe.buffer.size(1)).copy_(graph_probe.buffer, true);
        CudaGraphProbeRingMetadata metadata;
        metadata.record_id        = state.total_records;
        metadata.current_bs       = graph_state.current_batch_size;
        metadata.graph_bs         = graph_state.current_real_graph_bs;
        metadata.trace_ids.assign(inputs.trace_ids.begin(),
                                  inputs.trace_ids.begin()
                                      + std::min<int64_t>(inputs.trace_ids.size(), graph_state.current_batch_size));
        metadata.input_lengths =
            hostTensorPrefix(inputs.attention_inputs.input_lengths, graph_state.current_batch_size);
        metadata.sequence_lengths =
            hostTensorPrefix(inputs.attention_inputs.sequence_lengths, graph_state.current_batch_size);
        state.metadata[slot] = std::move(metadata);
        ++state.total_records;

        if (state.total_records % config.trigger_check_every == 0
            && access(config.trigger_file.c_str(), F_OK) == 0) {
            dumpCudaGraphProbeRing(state);
        }
    } catch (const std::exception& e) {
        auto iter = g_cuda_graph_probe_rings.find(runner_key);
        if (iter != g_cuda_graph_probe_rings.end() && !iter->second.disabled) {
            try {
                cuda_graph::graphDeviceSynchronize();
            } catch (...) {
            }
            disableCudaGraphProbeRing(iter->second, e.what());
        } else {
            RTP_LLM_LOG_WARNING("Failed to initialize CUDA graph probe ring: %s", e.what());
        }
    }
}

void appendLaneBlockMaps(std::ostringstream& os, const PyAttentionInputs& attention_inputs, int64_t lane) {
    os << "\"block_maps\":[";
    const auto group_count = attention_inputs.kv_cache_kernel_block_id_host_by_group.size();
    for (size_t g = 0; g < group_count; ++g) {
        if (g != 0) {
            os << ",";
        }
        os << "{\"group\":" << g << ",";
        appendTensorStats(os,
                          "host",
                          selectLaneTensor(attention_inputs.kv_cache_kernel_block_id_host_by_group[g], lane));
        os << ",";
        appendTensorStats(os,
                          "device",
                          selectLaneTensor(attention_inputs.kv_cache_kernel_block_id_device_by_group[g], lane));
        os << "}";
    }
    os << "]";
}

void writeDecodeChecksumRecord(const DecodeChecksumRecord& record,
                               const char*                  stage,
                               const PyModelInputs&         original_inputs,
                               const PyModelInputs&         graph_inputs,
                               const torch::Tensor&         hidden_states,
                               const CudaGraphProbeCapture& graph_probe,
                               const CudaGraphState&        state,
                               int32_t                      full_kv_cache_group_id) {
    const auto& config = decodeChecksumConfig();
    if (!record.selected) {
        return;
    }

    if (config.sync_device) {
        cuda_graph::graphDeviceSynchronize();
    }

    const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();

    std::ostringstream os;
    os << "{\"record_id\":" << record.id;
    os << ",\"ts_us\":" << now_us;
    os << ",\"pid\":" << getpid();
    os << ",\"rank\":\"" << jsonEscape(envString("WORLD_RANK", "")) << "\"";
    os << ",\"stage\":\"" << stage << "\"";
    os << ",\"path\":\"cuda_graph_decode\"";
    os << ",\"current_bs\":" << state.current_batch_size;
    os << ",\"graph_bs\":" << state.current_real_graph_bs;
    os << ",\"seq_len_sum\":" << state.seq_len_sum;
    os << ",\"full_kv_cache_group_id\":" << full_kv_cache_group_id;
    if (graph_probe.status) {
        os << ",\"graph_probe_status\":";
        appendGraphProbeStatus(os, *graph_probe.status);
    }
    os << ",\"lane_count\":" << record.lanes.size();
    os << ",\"lanes\":[";
    for (size_t i = 0; i < record.lanes.size(); ++i) {
        const int64_t     lane     = record.lanes[i];
        const std::string trace_id = lane < static_cast<int64_t>(original_inputs.trace_ids.size()) ?
                                         original_inputs.trace_ids[lane] :
                                         "";
        if (i != 0) {
            os << ",";
        }
        os << "{\"lane\":" << lane;
        os << ",\"trace_id\":\"" << jsonEscape(trace_id) << "\"";
        os << ",";
        appendTensorStats(os, "input_id", selectLaneTensor(graph_inputs.input_ids, lane));
        os << ",";
        appendTensorStats(os, "input_length", selectLaneTensor(graph_inputs.attention_inputs.input_lengths, lane));
        os << ",";
        appendTensorStats(os, "input_length_d", selectLaneTensor(graph_inputs.attention_inputs.input_lengths_d, lane));
        os << ",";
        appendTensorStats(os, "sequence_length", selectLaneTensor(graph_inputs.attention_inputs.sequence_lengths, lane));
        os << ",";
        appendTensorStats(
            os, "sequence_length_plus_1_d", selectLaneTensor(graph_inputs.attention_inputs.sequence_lengths_plus_1_d, lane));
        os << ",";
        appendTensorStats(
            os, "decode_cu_seqlens_host", sliceTensorRange(graph_inputs.attention_inputs.decode_cu_seqlens_host, lane, lane + 2));
        os << ",";
        appendTensorStats(
            os, "decode_cu_seqlens_d", sliceTensorRange(graph_inputs.attention_inputs.decode_cu_seqlens_d, lane, lane + 2));
        os << ",";
        appendLaneBlockMaps(os, graph_inputs.attention_inputs, lane);
        if (hidden_states.defined()) {
            os << ",";
            appendTensorStats(os, "hidden_state", selectLaneTensor(hidden_states, lane));
        }
        if (graph_probe.buffer.defined()) {
            os << ",";
            appendGraphProbe(os, graph_probe, lane);
        }
        os << "}";
    }
    os << "]}\n";

    std::lock_guard<std::mutex> lock(g_decode_checksum_mutex);
    std::ofstream               out(config.file, std::ios::app);
    out << os.str();
}

}  // namespace

CudaGraphRunner::~CudaGraphRunner() {
    retrospective_replays_.clear();
    releaseCudaGraphProbeRing(this);
    RTP_LLM_LOG_INFO("Release CudaGraphRunner .....");
    py::gil_scoped_acquire gil;
    py_instance_.release();
    RTP_LLM_LOG_INFO("Release CudaGraphRunner Successfully");
}

// clang-format off
// CUDA Graph Mode Configuration Table:
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// | Model Type                     | is_prefill_cuda_graph_mode_ | num_tokens_per_bs_                   | 是否已经支持   |
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// | Draft Model (prefill)          | true                        | gen_num_per_cycle + 1                | yes          |
// | Target Model (score, prefill)  | false                       | gen_num_per_cycle + 1                | yes          |
// | Draft Model (decode)           | false                       | 1                                    | yes          |
// | Embedding Model (prefill)      | true                        | max_seq_len                          | yes          |
// | Normal Model (decode)          | false                       | 1                                    | yes          |
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// Notes:
// - Speculative sampling: model_id == 0 (target), model_id == 1 (draft)
// clang-format on

// Helper function for optimized tensor copy using async operations with current CUDA stream
void optimizedCopyAsync(const torch::Tensor& src, torch::Tensor& dst, size_t size) {
    if (!src.defined() || !dst.defined() || src.numel() <= 0) {
        return;
    }

    RTP_LLM_PROFILE_SCOPE("optimizedCopyAsync");

    void* stream = reinterpret_cast<void*>(cuda_graph::graphGetCurrentStream().stream());
    if (src.is_cuda() && dst.is_cuda()) {
        cuda_graph::graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cuda_graph::GraphMemcpyKind::D2D, stream);
    } else if (!src.is_cuda() && !dst.is_cuda()) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), size);
    } else if (src.is_cuda() && !dst.is_cuda()) {
        cuda_graph::graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cuda_graph::GraphMemcpyKind::D2H, stream);
    } else {
        cuda_graph::graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cuda_graph::GraphMemcpyKind::H2D, stream);
    }
}

void fillHostI32Range(torch::Tensor& tensor, int64_t begin, int64_t end, int32_t value) {
    if (!tensor.defined() || tensor.is_cuda() || tensor.scalar_type() != torch::kInt32 || tensor.dim() != 1) {
        return;
    }
    const int64_t size = tensor.size(0);
    begin              = std::max<int64_t>(0, std::min<int64_t>(begin, size));
    end                = std::max<int64_t>(begin, std::min<int64_t>(end, size));
    auto* data         = tensor.data_ptr<int32_t>();
    for (int64_t i = begin; i < end; ++i) {
        data[i] = value;
    }
}

void copyHostI32RangeToDevice(const torch::Tensor& host, torch::Tensor& device, int64_t begin, int64_t end) {
    if (!host.defined() || !device.defined() || host.is_cuda() || !device.is_cuda()
        || host.scalar_type() != torch::kInt32 || device.scalar_type() != torch::kInt32 || host.dim() != 1
        || device.dim() != 1) {
        return;
    }
    const int64_t size = std::min<int64_t>(host.size(0), device.size(0));
    begin              = std::max<int64_t>(0, std::min<int64_t>(begin, size));
    end                = std::max<int64_t>(begin, std::min<int64_t>(end, size));
    if (begin == end) {
        return;
    }
    auto src = host.slice(0, begin, end);
    auto dst = device.slice(0, begin, end);
    optimizedCopyAsync(src, dst, static_cast<size_t>(end - begin) * sizeof(int32_t));
}

bool hasHybridBlockMaps(const PyAttentionInputs& attn_inputs) {
    return !attn_inputs.kv_cache_kernel_block_id_device_by_group.empty()
           && !attn_inputs.kv_cache_kernel_block_id_host_by_group.empty();
}

void selectActiveHybridBlockMapForGroup(PyAttentionInputs& attn_inputs, int32_t group_id) {
    if (group_id < 0 || !hasHybridBlockMaps(attn_inputs)) {
        return;
    }

    const auto group = static_cast<size_t>(group_id);
    RTP_LLM_CHECK_WITH_INFO(group < attn_inputs.kv_cache_kernel_block_id_device_by_group.size(),
                            "full kv cache group id out of range for device block map");
    RTP_LLM_CHECK_WITH_INFO(group < attn_inputs.kv_cache_kernel_block_id_host_by_group.size(),
                            "full kv cache group id out of range for host block map");

    attn_inputs.kv_cache_kernel_block_id_device = attn_inputs.kv_cache_kernel_block_id_device_by_group[group];
    attn_inputs.kv_cache_kernel_block_id_host   = attn_inputs.kv_cache_kernel_block_id_host_by_group[group];
}

void CudaGraphRunner::prepareInputs(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareInputs");
    // 1. non spec cuda graph:
    // is_prefill_cuda_graph_mode_ is set true only when use embedding model
    // 2. spec cuda graph:
    // 2.1 spec hold target model and draft model. when the user prompt first comes in, the target model
    // adn draft model will do real "prefill forward". And for this phase, we don't support cuda graph
    // 2.2 after real "prefill forward", it is consisted of three parts:
    // 2.2.1 target model score(verfiy)
    // 2.2.2 draft model do first forward (input is from 2.2.1)
    // 2.2.3 draft model do auto-agressive forward
    // for now we only support 2.2.1 and 2.2.3 in deocode cuda graph, and 2.2.2 will be support in prefill cuda graph.

    // should wait last forward done before prepare inputs
    forward_event_.synchronize();

    const size_t graph_idx =
        is_prefill_cuda_graph_mode_ ? state.current_real_graph_seq_len : state.current_real_graph_bs;
    auto& py_model_inputs_ = graph_instances_[graph_idx].mem_hold_.py_model_inputs_;
    auto  attn_pyobj       = graph_instances_[graph_idx].mem_hold_.attn_pyobj_;

    // Per-launch capacity contract: see fuse_copy_util.h sizing rationale.
    // Worst case here is ~8 contiguous + (1 + group_count) strided copies,
    // batched into one launch each. If new copies are added below — or if the
    // hybrid KV-cache group_count grows materially — re-check MAX_FUSED_*_COPIES.
    FusedD2DCopyParams     d2d_copies;
    FusedStridedCopyParams strided_d2d_copies;

    auto tryAddD2DCopy = [&d2d_copies](const torch::Tensor& src, torch::Tensor& dst, size_t bytes) {
        if (src.defined() && src.numel() > 0) {
            d2d_copies.add(src.data_ptr(), dst.data_ptr(), bytes);
        }
    };

    // Collect a strided 2D D2D copy: copies src[0..rows, 0..cols] into dst[0..rows, 0..cols]
    // where src and dst may have different column strides (copySmallerIntoLarger semantics).
    // For 1D tensors, falls back to a contiguous D2D copy to avoid silent data loss.
    auto tryAddStridedD2DCopy = [&strided_d2d_copies, &d2d_copies](const torch::Tensor& src, torch::Tensor& dst) {
        if (!src.defined() || src.numel() <= 0)
            return;
        if (src.dim() < 2) {
            d2d_copies.add(src.data_ptr(), dst.data_ptr(), src.numel() * src.element_size());
            return;
        }
        strided_d2d_copies.add(src.data_ptr(),
                               dst.data_ptr(),
                               src.size(0),
                               src.size(1) * src.element_size(),
                               src.stride(0) * src.element_size(),
                               dst.stride(0) * dst.element_size());
    };

    // H2H strided 2D copy via row-by-row memcpy (cannot use GPU kernel for host memory).
    // For 1D tensors, falls back to a contiguous memcpy.
    auto stridedCopyHost = [](const torch::Tensor& src, torch::Tensor& dst) {
        if (!src.defined() || src.numel() <= 0)
            return;
        RTP_LLM_PROFILE_SCOPE("stridedCopyHost");
        if (src.dim() < 2) {
            memcpy(dst.data_ptr(), src.data_ptr(), src.numel() * src.element_size());
            return;
        }
        const size_t nrows      = src.size(0);
        const size_t row_bytes  = src.size(1) * src.element_size();
        const size_t src_stride = src.stride(0) * src.element_size();
        const size_t dst_stride = dst.stride(0) * dst.element_size();
        const char*  src_ptr    = reinterpret_cast<const char*>(src.data_ptr());
        char*        dst_ptr    = reinterpret_cast<char*>(dst.data_ptr());
        for (size_t r = 0; r < nrows; ++r) {
            memcpy(dst_ptr + r * dst_stride, src_ptr + r * src_stride, row_bytes);
        }
    };

    // Hybrid cache: collect per-group D2D strided copies.
    const bool has_hybrid_cache = !inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
                                  && !inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.empty()
                                  && !py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
                                  && !py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group.empty();
    size_t hybrid_cache_group = 0;

    // clear kv_cache_kernel_block_id_device, otherwise it will cause the cache block pollution.
    // In hybrid mode the legacy 2-D field may alias a per-group tensor after Python
    // select_block_map_for_layer() mutates it during capture; per-group copies below
    // are the source of truth.
    if (!has_hybrid_cache) {
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.fill_(0);
    }

    // NOTE: kv_cache_block_id_{host,device} are physical block IDs dedicated for cache store
    // (see OpDefs.h). They are NOT consumed by any GPU attention kernel during CUDA graph replay;
    // attention kernels only use kv_cache_kernel_block_id_{host,device}. Cache store operations
    // run outside the CUDA graph and read from the original (non-graph) inputs directly.

    // Common device copy
    int token_num = is_prefill_cuda_graph_mode_ ? state.current_seq_len : inputs.input_ids.size(0);

    tryAddD2DCopy(inputs.input_ids, py_model_inputs_.input_ids, token_num * sizeof(int));
    tryAddD2DCopy(inputs.input_hiddens,
                  py_model_inputs_.input_hiddens,
                  inputs.input_hiddens.numel() * inputs.input_hiddens.element_size());
    tryAddD2DCopy(inputs.attention_inputs.cu_seqlens,
                  py_model_inputs_.attention_inputs.cu_seqlens,
                  (state.current_batch_size + 1) * sizeof(int));
    tryAddD2DCopy(inputs.attention_inputs.cu_kv_seqlens,
                  py_model_inputs_.attention_inputs.cu_kv_seqlens,
                  (state.current_batch_size + 1) * sizeof(int));
    tryAddD2DCopy(inputs.attention_inputs.input_lengths_d,
                  py_model_inputs_.attention_inputs.input_lengths_d,
                  state.current_batch_size * sizeof(int));
    // Strided 2D D2D copy for flat kv_cache_block_id. Skip it for hybrid KV
    // cache: the legacy field can alias one group tensor, so copying it in the
    // same fused kernel as per-group block maps can race on the same destination.
    if (!has_hybrid_cache) {
        tryAddStridedD2DCopy(inputs.attention_inputs.kv_cache_kernel_block_id_device,
                             py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device);
    }

    if (!is_prefill_cuda_graph_mode_) {
        // D2D copies — collected for single batched kernel launch
        tryAddD2DCopy(inputs.attention_inputs.prefix_lengths_d,
                      py_model_inputs_.attention_inputs.prefix_lengths_d,
                      state.current_batch_size * sizeof(int));
        tryAddD2DCopy(inputs.attention_inputs.sequence_lengths_plus_1_d,
                      py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d,
                      state.current_batch_size * sizeof(int));
        tryAddD2DCopy(inputs.attention_inputs.decode_cu_seqlens_d,
                      py_model_inputs_.attention_inputs.decode_cu_seqlens_d,
                      (state.current_batch_size + 1) * sizeof(int));
    } else {
        // D2D copy
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            tryAddD2DCopy(inputs.bert_embedding_inputs.combo_position_ids,
                          py_model_inputs_.bert_embedding_inputs.combo_position_ids,
                          state.current_seq_len * sizeof(int));
            tryAddD2DCopy(inputs.bert_embedding_inputs.combo_tokens_type_ids,
                          py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids,
                          state.current_seq_len * sizeof(int));
        }
    }

    if (has_hybrid_cache) {
        RTP_LLM_CHECK_WITH_INFO(
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size()
                == py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group.size(),
            "kv_cache_kernel_block_id_device_by_group size mismatch");
        hybrid_cache_group = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        RTP_LLM_CHECK_WITH_INFO(inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.size()
                                        == hybrid_cache_group
                                    && py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group.size()
                                           == hybrid_cache_group,
                                "kv_cache_kernel_block_id_host_by_group size mismatch");
        for (size_t g = 0; g < hybrid_cache_group; ++g) {
            // Non-full groups are consumed by linear/conv state kernels where -1 is the invalid block id.
            const int32_t fill_value =
                full_kv_cache_group_id_ >= 0 && static_cast<int32_t>(g) != full_kv_cache_group_id_ ? -1 : 0;
            py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group[g].fill_(fill_value);
            py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group[g].fill_(fill_value);
            tryAddStridedD2DCopy(inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[g],
                                 py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group[g]);
        }
    }

    // Launch ALL D2D copies (contiguous + strided) in two fused kernels
    fusedCopy(d2d_copies);
    fusedStridedCopy(strided_d2d_copies);

    // NOTE: we do H2H after D2D copies to let GPU finish the D2D copies as soon as possible,
    // so that the GPU can start the kernel launch as soon as possible.

    // H2H copies (common to both modes)
    optimizedCopyAsync(inputs.attention_inputs.cu_seqlens_host,
                       py_model_inputs_.attention_inputs.cu_seqlens_host,
                       (state.current_batch_size + 1) * sizeof(int));

    optimizedCopyAsync(inputs.attention_inputs.input_lengths,
                       py_model_inputs_.attention_inputs.input_lengths,
                       state.current_batch_size * sizeof(int));

    optimizedCopyAsync(inputs.attention_inputs.prefix_lengths,
                       py_model_inputs_.attention_inputs.prefix_lengths,
                       state.current_batch_size * sizeof(int));

    // Common H2H strided copies for kv_cache block tables (both decode & prefill).
    // Same aliasing rule as the device copy above.
    if (!has_hybrid_cache) {
        stridedCopyHost(inputs.attention_inputs.kv_cache_kernel_block_id_host,
                        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host);
    }

    optimizedCopyAsync(inputs.attention_inputs.kv_cache_layer_to_group,
                       py_model_inputs_.attention_inputs.kv_cache_layer_to_group,
                       inputs.attention_inputs.kv_cache_layer_to_group.numel() * sizeof(int32_t));

    if (!is_prefill_cuda_graph_mode_) {
        optimizedCopyAsync(inputs.attention_inputs.sequence_lengths,
                           py_model_inputs_.attention_inputs.sequence_lengths,
                           state.current_batch_size * sizeof(int));

        const int64_t graph_bs   = py_model_inputs_.attention_inputs.input_lengths.defined() ?
                                     py_model_inputs_.attention_inputs.input_lengths.size(0) :
                                     state.current_real_graph_bs;
        const int64_t current_bs = state.current_batch_size;
        if (current_bs < graph_bs) {
            // Decode graphs replay with graph_bs lanes. Mark inactive lanes as 0-token lanes before graph replay.
            fillHostI32Range(py_model_inputs_.attention_inputs.input_lengths, current_bs, graph_bs, 0);
            copyHostI32RangeToDevice(py_model_inputs_.attention_inputs.input_lengths,
                                     py_model_inputs_.attention_inputs.input_lengths_d,
                                     current_bs,
                                     graph_bs);

            fillHostI32Range(py_model_inputs_.attention_inputs.prefix_lengths, current_bs, graph_bs, 0);
            copyHostI32RangeToDevice(py_model_inputs_.attention_inputs.prefix_lengths,
                                     py_model_inputs_.attention_inputs.prefix_lengths_d,
                                     current_bs,
                                     graph_bs);

            fillHostI32Range(py_model_inputs_.attention_inputs.sequence_lengths, current_bs, graph_bs, 0);
            copyHostI32RangeToDevice(py_model_inputs_.attention_inputs.sequence_lengths,
                                     py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d,
                                     current_bs,
                                     graph_bs);

            fillHostI32Range(py_model_inputs_.attention_inputs.decode_cu_seqlens_host,
                             current_bs + 1,
                             graph_bs + 1,
                             static_cast<int32_t>(current_bs));
            copyHostI32RangeToDevice(py_model_inputs_.attention_inputs.decode_cu_seqlens_host,
                                     py_model_inputs_.attention_inputs.decode_cu_seqlens_d,
                                     current_bs + 1,
                                     graph_bs + 1);
        }
    } else {
        optimizedCopyAsync(inputs.attention_inputs.padding_offset,
                           py_model_inputs_.attention_inputs.padding_offset,
                           state.current_seq_len * sizeof(int));

        if (py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params) {
            auto* batch_size_ptr = py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params
                                       ->cuda_graph_prefill_batch_size.data_ptr<int>();
            *batch_size_ptr = state.current_batch_size;
        }
    }

    // Hybrid cache: H2H strided copies for per-group block tables
    if (has_hybrid_cache) {
        for (size_t g = 0; g < hybrid_cache_group; ++g) {
            stridedCopyHost(inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group[g],
                            py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group[g]);
        }
    }

    // Reset unused batch portions to prevent stale data (prefill only)
    if (is_prefill_cuda_graph_mode_) {
        if (state.current_batch_size < max_bs_) {
            py_model_inputs_.attention_inputs.prefix_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
            py_model_inputs_.attention_inputs.input_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
        }

        int last_valid_q = state.current_seq_len;
        int last_valid_kv =
            last_valid_q
            + inputs.attention_inputs.prefix_lengths.slice(0, 0, state.current_batch_size).sum().item<int>();
        py_model_inputs_.attention_inputs.cu_seqlens_host.slice(0, state.current_batch_size + 1, max_bs_ + 1)
            .fill_(last_valid_q);
        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, state.current_batch_size + 1, max_bs_ + 1)
            .fill_(last_valid_q);
        py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, state.current_batch_size + 1, max_bs_ + 1)
            .fill_(last_valid_kv);
    }

    // launch prepare_cuda_graph when attention inputs are ready
    {
        RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareInputs(prepare_cuda_graph)");
        selectActiveHybridBlockMapForGroup(py_model_inputs_.attention_inputs, full_kv_cache_group_id_);
        attn_pyobj.attr("prepare_cuda_graph")(py_model_inputs_.attention_inputs);
    }
}

PyModelOutputs CudaGraphRunner::forward(const PyModelInputs& inputs, CudaGraphState& state) {
    PyModelOutputs outputs;

    // decode or embedding model only
    RTP_LLM_LOG_DEBUG("Replay Start");
    prepareInputs(inputs, state);
    if (is_prefill_cuda_graph_mode_) {
        {
            RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayPrefill)");
            replayPrefill(state.current_real_graph_seq_len);
        }
        outputs.hidden_states =
            graph_instances_[state.current_real_graph_seq_len].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state.current_seq_len);
    } else {
        auto& graph_inputs = graph_instances_[state.current_real_graph_bs].mem_hold_.py_model_inputs_;
        dumpRetrospectiveProbeBeforeReplay();
        DecodeProbeTriggerEvent retrospective_event;
        const bool replay_debug = shouldReplayRetrospectiveDebug(inputs, state, retrospective_event);
        const bool run_eager_step =
            !replay_debug && shouldRunRetrospectiveEagerStep(inputs, state, retrospective_event);
        if (replay_debug || run_eager_step) {
            waitForRetrospectiveRanksReady(retrospective_event);
        }
        const auto checksum_record = nextDecodeChecksumRecord(inputs, state.current_batch_size);
        writeDecodeChecksumRecord(
            checksum_record, "before_replay", inputs, graph_inputs, torch::Tensor(), {}, state, full_kv_cache_group_id_);
        {
            RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayDecode)");
            if (replay_debug) {
                retrospective_debug_graph_instances_[state.current_real_graph_bs].graph_.replay();
            } else if (run_eager_step) {
                runRetrospectiveEagerStep(state.current_real_graph_bs, retrospective_event);
            } else {
                replayDecode(state.current_real_graph_bs);
            }
        }
        outputs.hidden_states =
            graph_instances_[state.current_real_graph_bs].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state.seq_len_sum);
        CudaGraphProbeCapture graph_probe;
        if (checksum_record.selected) {
            graph_probe = getCudaGraphProbeBuffer(py_instance_, state.current_real_graph_bs);
        }
        captureCudaGraphProbeRing(this, py_instance_, graph_probe, inputs, state);
        if (replay_debug || run_eager_step) {
            dumpRetrospectiveDebugReplay(inputs, state, retrospective_event);
        } else {
            retainRetrospectiveReplay(inputs, state);
        }
        writeDecodeChecksumRecord(checksum_record,
                                  "after_replay",
                                  inputs,
                                  graph_inputs,
                                  outputs.hidden_states,
                                  graph_probe,
                                  state,
                                  full_kv_cache_group_id_);
    }
    // record forward done event
    forward_event_.record(cuda_graph::graphGetCurrentStream());
    RTP_LLM_LOG_DEBUG("Replay End");
    return outputs;
}

bool CudaGraphRunner::tryGetRealGraphPrefillSeqLen(const PyModelInputs& inputs, CudaGraphState& state) {
    state.current_seq_len = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("prefill cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state.current_seq_len);
    // No captured graph for seq_len >= current (all captures smaller than requested)
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("prefill seq_len %d exceeds max captured %d, fallback to normal run",
                            state.current_seq_len,
                            capture_range_.back());
        return false;
    }
    state.current_real_graph_seq_len = *it;
    state.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
    return true;
}

bool CudaGraphRunner::tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs, CudaGraphState& state) {
    int cuda_graph_bs        = inputs.attention_inputs.input_lengths.size(0);
    state.current_batch_size = cuda_graph_bs;
    RTP_LLM_LOG_DEBUG("canRun judge for batch size: %d", cuda_graph_bs);
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("decode cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state.current_batch_size);
    // No captured graph for batch >= current (all captures smaller)
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("decode batch size %d exceeds max captured %d, fallback to normal run",
                            state.current_batch_size,
                            capture_range_.back());
        return false;
    }
    state.current_real_graph_bs = *it;
    RTP_LLM_LOG_DEBUG(
        "batch size used in replay: %d (graph key %d)", state.current_batch_size, state.current_real_graph_bs);

    if (inputs.attention_inputs.is_prefill) {
        state.seq_len_sum = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    } else {
        state.seq_len_sum = cuda_graph_bs;
    }
    RTP_LLM_LOG_DEBUG("can run cuda graph for decode");
    return true;
}

bool CudaGraphRunner::canRun(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.canRun");
    // Check if this is speculative sampling:
    // 1. prefix_lengths is not empty
    // 2. all values in input_lengths are the same
    // this is for 2.2.1
    if (is_target_verify_) {
        if (inputs.attention_inputs.is_target_verify) {
            // Target-verify must also respect captured decode range.
            // Otherwise we may replay an uncaptured graph key.
            return tryGetRealGraphDecodeBatchSize(inputs, state);
        }
        return false;
    }

    if (!enable_cuda_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_cuda_graph_mode_)) {
        return false;
    }

    if (!inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()) {
        const size_t group = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        if (kv_cache_group_num_ <= 0) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache detected but kv_cache_group_num_ is not set, fallback to normal run.");
            return false;
        }
        if (group != static_cast<size_t>(kv_cache_group_num_)) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache group size mismatch: inputs=%zu, captured=%d, fallback to normal run.",
                                group,
                                kv_cache_group_num_);
            return false;
        }
    }

    if (is_prefill_cuda_graph_mode_) {
        if (!tryGetRealGraphPrefillSeqLen(inputs, state)) {
            return false;
        }
        // current_real_graph_seq_len is always *it from lower_bound within capture_range_
        RTP_LLM_LOG_DEBUG("prefill cuda graph replay seq_len key %d", state.current_real_graph_seq_len);
    } else {
        if (!tryGetRealGraphDecodeBatchSize(inputs, state)) {
            return false;
        }
    }
    return true;
}

void CudaGraphRunner::initKernelInternalMemory() {
    torch::Tensor cu_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU)).pin_memory();
    torch::Tensor cu_kv_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    auto input_lengths  = capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths;
    auto prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;

    cu_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.cumsum(0);
    if (prefix_lengths.defined() && prefix_lengths.size(0) > 0) {
        cu_kv_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.add(prefix_lengths).cumsum(0);
    }
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens_host = cu_seqlens;
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens      = cu_seqlens.cuda();
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens   = cu_kv_seqlens.cuda();
}

int CudaGraphRunner::getCurrentRealGraphBs(const CudaGraphState& state) const {
    return state.current_real_graph_bs;
}

void CudaGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.attention_inputs.is_target_verify = is_target_verify_;
    inputs.attention_inputs.is_prefill       = is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1;

    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32_);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths   = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_);
    inputs.attention_inputs.input_lengths   = inputs.attention_inputs.input_lengths.pin_memory();
    inputs.attention_inputs.input_lengths_d = inputs.attention_inputs.input_lengths.cuda();
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.sequence_lengths.fill_(max_seq_len_ - num_tokens_per_bs - 1);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();

    const int64_t max_kv_blocks =
        static_cast<int64_t>(((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_) + sp_steps_);
    const int64_t max_blocks = max_kv_blocks * seq_size_per_block_ / kernel_seq_size_per_block_;
    // kv_cache_kernel_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        torch::zeros({int(max_bs_), max_blocks}, options_cuda_int32_);

    inputs.attention_inputs.kv_cache_kernel_block_id_host =
        torch::zeros({int(max_bs_), max_blocks}, options_cpu_int32_).pin_memory();

    auto layer_num = kv_cache_layer_to_group_.size();
    if (layer_num > 0) {
        auto kv_cache_layer_to_group_capture_ =
            torch::empty({static_cast<int64_t>(layer_num)}, options_cpu_int32_).pin_memory();
        auto* dst = kv_cache_layer_to_group_capture_.data_ptr<int32_t>();
        for (size_t i = 0; i < layer_num; ++i) {
            dst[i] = static_cast<int32_t>(kv_cache_layer_to_group_[i]);
        }

        // [layer_num] int32, pinned host tensor. Keep empty when not provided.
        inputs.attention_inputs.kv_cache_layer_to_group = kv_cache_layer_to_group_capture_;
    }

    // Hybrid cache: per-group block tables.
    inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.clear();
    if (kv_cache_group_num_ > 1) {
        inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.reserve(kv_cache_group_num_);
        inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.reserve(kv_cache_group_num_);
        for (int g = 0; g < kv_cache_group_num_; ++g) {
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.push_back(
                torch::zeros({int(max_bs_), max_blocks}, options_cuda_int32_));
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.push_back(
                torch::zeros({int(max_bs_), max_blocks}, options_cpu_int32_).pin_memory());
        }
        selectActiveHybridBlockMapForGroup(inputs.attention_inputs, full_kv_cache_group_id_);
    }

    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    if (num_tokens_per_bs_ > 1 && !is_prefill_cuda_graph_mode_) {
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, max_seq_len_ - num_tokens_per_bs_, options_cpu_int32_).pin_memory();
        inputs.attention_inputs.prefix_lengths_d = inputs.attention_inputs.prefix_lengths.cuda();
    } else if (is_prefill_cuda_graph_mode_) {
        // ROCm needs prefix>0 here for AiterPrefillImplPaged.support(); CUDA keeps prefix=0.
#if USING_ROCM
        const int prefix_init = isMtpDraftPrefillCudaGraph() ? max_seq_len_ : 0;
#else
        const int prefix_init = 0;
#endif
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, prefix_init, options_cpu_int32_).pin_memory();
        inputs.attention_inputs.prefix_lengths_d = inputs.attention_inputs.prefix_lengths.cuda();
    } else {
        // Decode CUDA graph mode: prefix_lengths should be empty tensor
        inputs.attention_inputs.prefix_lengths = torch::empty({0}, options_cpu_int32_).pin_memory();
    }
    // padding_offset [max_num_token_, int32] (for attention padding)
    inputs.attention_inputs.padding_offset            = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.padding_offset            = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype                     = model_data_type_;
    inputs.attention_inputs.is_s_padded               = true;
    inputs.attention_inputs.sequence_lengths_plus_1_d = torch::zeros({int(max_bs_)}, options_cuda_int32_);
    inputs.attention_inputs.decode_cu_seqlens_host =
        torch::arange(0,
                      max_bs_ + 1,
                      1,
                      torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    inputs.attention_inputs.decode_cu_seqlens_d =
        torch::arange(0, max_bs_ + 1, 1, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
}

void CudaGraphRunner::initCaptureAttentionInputsPost() {
    auto&         inputs                        = capture_mem_hold_.py_model_inputs_;
    torch::Tensor cuda_graph_prefill_batch_size = torch::zeros({1}, options_cpu_int32_).pin_memory();
    // as one batch to capture
    cuda_graph_prefill_batch_size.fill_(1);
    RTP_LLM_CHECK_WITH_INFO(cuda_graph_prefill_batch_size.is_pinned(),
                            "capture_mem_hold_ cuda_graph_prefill_batch_size is not pinned memory");

    // draft model prefill but not embedding model
    if (num_tokens_per_bs_ > 1 && num_tokens_per_bs_ != max_seq_len_) {
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            PyPrefillCudaGaphCopyParams{cuda_graph_prefill_batch_size, num_tokens_per_bs_, int(max_bs_)};
    } else {
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            PyPrefillCudaGaphCopyParams{cuda_graph_prefill_batch_size, max_seq_len_, int(max_bs_)};
    }
}

void CudaGraphRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void CudaGraphRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void CudaGraphRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

void CudaGraphRunner::initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token) {
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    // Initialize BertEmbeddingInputs for capture
    // combo_position_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_position_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // position_encoding: from weights
    inputs.bert_embedding_inputs.position_encoding = position_encoding_;

    // combo_tokens_type_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_tokens_type_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // token_type_embedding: from weights
    inputs.bert_embedding_inputs.token_type_embedding = token_type_embedding_;

    // input_embedding_scalar: fixed value
    inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar_;
}

void CudaGraphRunner::logCudaGraphPoolMemory(const char* phase) {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    cuda_graph::graphMemGetInfo(&free_bytes, &total_bytes);
    const size_t used_bytes        = total_bytes - free_bytes;
    const size_t pytorch_allocated = cuda_graph::graphAllocatedBytes();
    const size_t pytorch_reserved  = cuda_graph::graphReservedBytes();
    const size_t pool_overhead     = pytorch_reserved > pytorch_allocated ? pytorch_reserved - pytorch_allocated : 0;

    RTP_LLM_LOG_INFO("[CudaGraph Memory][%s] cudaMemGetInfo: used=%zu MiB, free=%zu MiB, total=%zu MiB | "
                     "PyTorch: allocated=%zu MiB, reserved=%zu MiB, pool_overhead=%zu MiB",
                     phase,
                     used_bytes / 1024 / 1024,
                     free_bytes / 1024 / 1024,
                     total_bytes / 1024 / 1024,
                     pytorch_allocated / 1024 / 1024,
                     pytorch_reserved / 1024 / 1024,
                     pool_overhead / 1024 / 1024);
}

void CudaGraphRunner::initCapture() {
    if (enable_cuda_graph_) {
        RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
        shared_graph_pool_ = cuda_graph::graphPoolHandle();
        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_LOG_INFO("CUDA graph capture for prefill, num_tokens_per_bs_: %d", num_tokens_per_bs_);
        }
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        if (is_prefill_cuda_graph_mode_) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
        } else {
            capture_range_ = getDecodeBatchSizesToCapture();
        }

        PyModelInputs inputs;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids     = torch::zeros({max_num_token_}, options_cuda_int32_);
        inputs.input_hiddens = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
        // Setup attention inputs using the extracted function
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);

        // Setup BertEmbedding inputs using the extracted function
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, is_prefill_cuda_graph_mode_);
        initKernelInternalMemory();

        if (!is_prefill_cuda_graph_mode_ && retrospectiveProbeToggleEnabled()) {
            RTP_LLM_CHECK_WITH_INFO(setPythonGraphProbeEnabled(false),
                                    "post-trigger graph probe requires an enabled Qwen CUDA graph probe");
        }

        // get real output data type (params already prepared in attn impl __init__/create_params)
        auto attn_pyobj = py_attn_pyobj_method_(capture_mem_hold_.py_model_inputs_, true);
        RTP_LLM_LOG_INFO("initCapture forward for output datatype start");
        py_forward_method_(capture_mem_hold_.py_model_inputs_, attn_pyobj);
        RTP_LLM_LOG_INFO("initCapture forward for output datatype end");
        output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
        capture_mem_hold_.setHiddenStates(output);
        initCaptureAttentionInputsPost();
        logCudaGraphPoolMemory("before_capture");

        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_CHECK_WITH_INFO(isEmbeddingStylePrefillCudaGraph() || isMtpDraftPrefillCudaGraph(),
                                    "prefill cuda graph: expected embedding-style or MTP draft layout");
            capturePrefill();
        } else {
            captureDecode();
        }
        logCudaGraphPoolMemory("after_capture");
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

void CudaGraphRunner::replayGraph(int key) {
    graph_instances_[key].graph_.replay();
}

void CudaGraphRunner::captureOneGraphInstance(int key, const char* key_type) {
    captureOneGraphInstance(key, key_type, graph_instances_[key].graph_);
}

void CudaGraphRunner::captureOneGraphInstance(int key, const char* key_type, at::cuda::CUDAGraph& graph) {
    auto inputs = graph_instances_[key].mem_hold_.py_model_inputs_;

    size_t pre_capture_reserved = cuda_graph::graphReservedBytes();

    // WarmUp twice (params already prepared in attn impl __init__/create_params when instance was created)
    RTP_LLM_LOG_INFO("WarmUp for %s %d start.", key_type, key);
    auto attn_pyobj = graph_instances_[key].mem_hold_.attn_pyobj_;
    try {
        py_forward_method_(inputs, attn_pyobj);
        py_forward_method_(inputs, attn_pyobj);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("WarmUp forward failed for %s %d: %s", key_type, key, e.what());
        throw;
    }
    RTP_LLM_LOG_INFO("WarmUp for %s %d successfully.", key_type, key);

    {
        // sync before capture
        cuda_graph::graphDeviceSynchronize();

        CudaGraphStreamLife stream_life(capture_stream_);
        std::string         output_dot_filename = "";
        if (enable_cuda_graph_debug_mode_) {
            graph.enable_debug_mode();
            std::string key_type_str = std::string(key_type);
            std::replace(key_type_str.begin(), key_type_str.end(), ' ', '_');
            output_dot_filename = "cuda_graph_tokens" + std::to_string(num_tokens_per_bs_) + "_" + key_type_str + "_"
                                  + std::to_string(key) + "_visualization.dot";
            RTP_LLM_LOG_INFO("CUDA Graph debug mode enabled, output file: %s", output_dot_filename.c_str());
        }
        RTP_LLM_LOG_INFO("Capture for %s %d begin.", key_type, key);
        PyModelOutputs outputs;
        {
            cuda_graph::graphCaptureBegin(graph, shared_graph_pool_);
            cuda_graph::GraphNcclCaptureContext capture_ctx;
            CudaGraphCaptureGuard               capture_guard(&capture_ctx);
            try {
                auto py_outputs_obj = py_forward_method_(inputs, attn_pyobj);
                outputs             = py_outputs_obj.cast<PyModelOutputs>();
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_ERROR("Capture forward failed for %s %d: %s", key_type, key, e.what());
                throw;
            }
            graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
            graph.capture_end();
        }
        if (!is_prefill_cuda_graph_mode_) {
            cacheRetrospectiveProbeHandle(key);
        }

        if (enable_cuda_graph_debug_mode_) {
            RTP_LLM_LOG_INFO("Calling debug_dump to generate: %s", output_dot_filename.c_str());
            graph.debug_dump(output_dot_filename.c_str());
            RTP_LLM_LOG_INFO("debug_dump completed for: %s", output_dot_filename.c_str());
        }

        size_t post_capture_reserved = cuda_graph::graphReservedBytes();
        size_t graph_pool_delta =
            post_capture_reserved > pre_capture_reserved ? post_capture_reserved - pre_capture_reserved : 0;
        RTP_LLM_LOG_INFO("[CudaGraph Memory] captured %s %d: pool_delta=%zu MiB, total_reserved=%zu MiB",
                         key_type,
                         key,
                         graph_pool_delta / 1024 / 1024,
                         post_capture_reserved / 1024 / 1024);
    }
}

bool CudaGraphRunner::dualGraphDebugEnabled() const noexcept {
    return !is_prefill_cuda_graph_mode_ && cudaGraphRetrospectiveConfig().enabled
           && cudaGraphRetrospectiveConfig().dual_graph;
}

bool CudaGraphRunner::eagerStepDebugEnabled() const noexcept {
    return !is_prefill_cuda_graph_mode_ && cudaGraphRetrospectiveConfig().enabled
           && cudaGraphRetrospectiveConfig().eager_step && !cudaGraphRetrospectiveConfig().dual_graph;
}

bool CudaGraphRunner::retrospectiveProbeToggleEnabled() const noexcept {
    return dualGraphDebugEnabled() || eagerStepDebugEnabled();
}

bool CudaGraphRunner::setPythonGraphProbeEnabled(bool enabled) noexcept {
    if (!retrospectiveProbeToggleEnabled()) {
        return false;
    }
    try {
        py::gil_scoped_acquire gil;
        if (!py::hasattr(py_instance_, "set_cuda_graph_probe_enabled")
            || !py::hasattr(py_instance_, "get_cuda_graph_probe_enabled")) {
            return false;
        }
        py_instance_.attr("set_cuda_graph_probe_enabled")(enabled);
        return py_instance_.attr("get_cuda_graph_probe_enabled")().cast<bool>() == enabled;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to toggle Qwen CUDA graph probe: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("Failed to toggle Qwen CUDA graph probe");
    }
    return false;
}

bool CudaGraphRunner::retrospectiveEventPendingForRank(DecodeProbeTriggerEvent& event) noexcept {
    if (!DecodeProbeTrigger::peek(event)) {
        return false;
    }
    const auto rank      = envInt64("WORLD_RANK", 0);
    const auto rank_mask = rank >= 0 && rank < 64 ? uint64_t{1} << rank : uint64_t{0};
    if (rank_mask == 0 || (event.required_rank_mask & rank_mask) == 0 || (event.ack_rank_mask & rank_mask) != 0) {
        return false;
    }
    return true;
}

void CudaGraphRunner::waitForRetrospectiveRanksReady(const DecodeProbeTriggerEvent& event) {
    if (!DecodeProbeTrigger::arrive(event.generation)) {
        DecodeProbeTrigger::acknowledge(event.generation, true);
        throw std::runtime_error("failed to join retrospective TP-rank barrier");
    }

    const auto timeout_ms = std::max<int64_t>(1, envInt64("RTPLLM_RETROSPECTIVE_READY_TIMEOUT_MS", 30000));
    const auto deadline   = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        DecodeProbeTriggerEvent current;
        if (!DecodeProbeTrigger::peek(current) || current.generation != event.generation) {
            DecodeProbeTrigger::acknowledge(event.generation, true);
            throw std::runtime_error("retrospective TP-rank barrier event changed");
        }
        if ((current.ready_rank_mask & current.required_rank_mask) == current.required_rank_mask) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    DecodeProbeTrigger::acknowledge(event.generation, true);
    throw std::runtime_error("timed out waiting for retrospective TP ranks");
}

bool CudaGraphRunner::shouldReplayRetrospectiveDebug(const PyModelInputs&  /* inputs */,
                                                     const CudaGraphState& state,
                                                     DecodeProbeTriggerEvent& event) noexcept {
    if (!dualGraphDebugEnabled() || retrospective_debug_graph_instances_.count(state.current_real_graph_bs) == 0
        || !retrospectiveEventPendingForRank(event)) {
        return false;
    }
    return true;
}

bool CudaGraphRunner::shouldRunRetrospectiveEagerStep(const PyModelInputs&  /* inputs */,
                                                      const CudaGraphState& /* state */,
                                                      DecodeProbeTriggerEvent& event) noexcept {
    return eagerStepDebugEnabled() && retrospectiveEventPendingForRank(event);
}

void CudaGraphRunner::runRetrospectiveEagerStep(int graph_bs, const DecodeProbeTriggerEvent& event) {
    if (!setPythonGraphProbeEnabled(true)) {
        DecodeProbeTrigger::acknowledge(event.generation, true);
        throw std::runtime_error("Qwen CUDA graph probe could not be enabled for eager step");
    }
    try {
        auto& graph_instance = graph_instances_.at(graph_bs);
        auto  py_outputs_obj = py_forward_method_(graph_instance.mem_hold_.py_model_inputs_,
                                                 graph_instance.mem_hold_.attn_pyobj_);
        auto  eager_outputs  = py_outputs_obj.cast<PyModelOutputs>();
        graph_instance.mem_hold_.decoder_layer_hidden_states_.copy_(eager_outputs.hidden_states);
    } catch (...) {
        setPythonGraphProbeEnabled(false);
        DecodeProbeTrigger::acknowledge(event.generation, true);
        throw;
    }
    if (!setPythonGraphProbeEnabled(false)) {
        DecodeProbeTrigger::acknowledge(event.generation, true);
        throw std::runtime_error("Qwen CUDA graph probe could not be disabled after eager step");
    }
}

void CudaGraphRunner::dumpRetrospectiveDebugReplay(const PyModelInputs&             inputs,
                                                   const CudaGraphState&            state,
                                                   const DecodeProbeTriggerEvent& event) noexcept {
    try {
        auto graph_probe = getCudaGraphProbeBufferLight(py_instance_, state.current_real_graph_bs);
        if (!validCudaGraphProbeCapture(graph_probe)) {
            throw std::runtime_error("debug graph replay produced no probe buffer");
        }
        CudaGraphPreviousReplay replay;
        replay.replay_id    = retrospective_replay_id_++;
        replay.graph_bs     = state.current_real_graph_bs;
        replay.current_bs   = std::max<int64_t>(0, std::min<int64_t>(state.current_batch_size, replay.graph_bs));
        replay.probe_buffer = graph_probe.buffer;
        replay.layers       = std::move(graph_probe.layers);
        replay.trace_ids.resize(replay.graph_bs);
        replay.input_lengths.resize(replay.graph_bs, -1);
        replay.sequence_lengths.resize(replay.graph_bs, -1);
        for (int64_t lane = 0; lane < replay.current_bs; ++lane) {
            if (lane < static_cast<int64_t>(inputs.trace_ids.size())) {
                replay.trace_ids[lane] = inputs.trace_ids[lane];
            }
            hostScalarAt(inputs.attention_inputs.input_lengths, lane, replay.input_lengths[lane]);
            hostScalarAt(inputs.attention_inputs.sequence_lengths, lane, replay.sequence_lengths[lane]);
        }
        dumpCudaGraphPreviousReplay(replay, event);
        DecodeProbeTrigger::acknowledge(event.generation);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to dump retrospective debug graph replay: %s", e.what());
        DecodeProbeTrigger::acknowledge(event.generation, true);
    } catch (...) {
        RTP_LLM_LOG_WARNING("Failed to dump retrospective debug graph replay");
        DecodeProbeTrigger::acknowledge(event.generation, true);
    }
}

void CudaGraphRunner::cacheRetrospectiveProbeHandle(int graph_bs) noexcept {
    if (!cudaGraphRetrospectiveConfig().enabled || dualGraphDebugEnabled() || eagerStepDebugEnabled()) {
        return;
    }
    try {
        auto graph_probe = getCudaGraphProbeBufferLight(py_instance_, graph_bs);
        if (!validCudaGraphProbeCapture(graph_probe)) {
            return;
        }
        std::lock_guard<std::mutex> lock(g_cuda_graph_retrospective_mutex);
        auto&                       replay = retrospective_replays_[graph_bs];
        replay.graph_bs                    = graph_bs;
        replay.probe_buffer                = graph_probe.buffer;
        replay.layers                      = std::move(graph_probe.layers);
        replay.trace_ids.resize(graph_bs);
        for (auto& trace_id : replay.trace_ids) {
            trace_id.reserve(sizeof(detail::DecodeProbeTriggerSharedRecord{}.trace_id) - 1);
        }
        replay.input_lengths.resize(graph_bs, -1);
        replay.sequence_lengths.resize(graph_bs, -1);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to cache retrospective CUDA graph probe buffer: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("Failed to cache retrospective CUDA graph probe buffer");
    }
}

void CudaGraphRunner::dumpRetrospectiveProbeBeforeReplay() noexcept {
    if (!cudaGraphRetrospectiveConfig().enabled || dualGraphDebugEnabled() || eagerStepDebugEnabled()) {
        return;
    }
    DecodeProbeTriggerEvent event;
    bool                    replay_selected = false;
    try {
        if (!DecodeProbeTrigger::peek(event)) {
            return;
        }
        const auto rank = envInt64("WORLD_RANK", 0);
        const auto rank_mask = rank >= 0 && rank < 64 ? uint64_t{1} << rank : uint64_t{0};
        if (rank_mask == 0 || (event.required_rank_mask & rank_mask) == 0
            || (event.ack_rank_mask & rank_mask) != 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(g_cuda_graph_retrospective_mutex);
        DecodeProbeTriggerEvent     current;
        if (!DecodeProbeTrigger::peek(current) || current.generation != event.generation
            || (current.required_rank_mask & rank_mask) == 0 || (current.ack_rank_mask & rank_mask) != 0) {
            return;
        }
        CudaGraphPreviousReplay* highest_replay       = nullptr;
        CudaGraphPreviousReplay* highest_exact_replay = nullptr;
        bool matching_trace_has_sequence_length       = false;
        const bool event_has_sequence_length          = current.observed_sequence_length >= 0;
        for (auto& item : retrospective_replays_) {
            auto&      replay = item.second;
            const auto active_lanes = std::max<int64_t>(
                0, std::min<int64_t>(replay.current_bs, static_cast<int64_t>(replay.trace_ids.size())));
            bool trace_matches        = false;
            bool exact_length_matches = false;
            for (int64_t lane = 0; lane < active_lanes; ++lane) {
                if (replay.trace_ids[lane] != current.trace_id) {
                    continue;
                }
                trace_matches = true;
                if (lane < static_cast<int64_t>(replay.sequence_lengths.size())
                    && replay.sequence_lengths[lane] >= 0) {
                    matching_trace_has_sequence_length = true;
                    if (event_has_sequence_length
                        && replay.sequence_lengths[lane] == current.observed_sequence_length - 1) {
                        exact_length_matches = true;
                    }
                }
            }
            if (!trace_matches) {
                continue;
            }
            if (highest_replay == nullptr || replay.replay_id > highest_replay->replay_id) {
                highest_replay = &replay;
            }
            if (exact_length_matches
                && (highest_exact_replay == nullptr || replay.replay_id > highest_exact_replay->replay_id)) {
                highest_exact_replay = &replay;
            }
        }
        auto* replay = event_has_sequence_length && matching_trace_has_sequence_length ? highest_exact_replay :
                                                                                         highest_replay;
        if (replay == nullptr) {
            return;
        }
        replay_selected = true;
        dumpCudaGraphPreviousReplay(*replay, current);
        DecodeProbeTrigger::acknowledge(current.generation);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to process retrospective CUDA graph trigger: %s", e.what());
        if (replay_selected && event.generation != 0) {
            DecodeProbeTrigger::acknowledge(event.generation, true);
        }
    } catch (...) {
        RTP_LLM_LOG_WARNING("Failed to process retrospective CUDA graph trigger");
        if (replay_selected && event.generation != 0) {
            DecodeProbeTrigger::acknowledge(event.generation, true);
        }
    }
}

void CudaGraphRunner::retainRetrospectiveReplay(const PyModelInputs& inputs, const CudaGraphState& state) noexcept {
    if (!cudaGraphRetrospectiveConfig().enabled || dualGraphDebugEnabled()) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(g_cuda_graph_retrospective_mutex);
        const auto                  replay = retrospective_replays_.find(state.current_real_graph_bs);
        if (replay == retrospective_replays_.end() || !replay->second.probe_buffer.defined()) {
            return;
        }
        auto& record      = replay->second;
        record.replay_id  = retrospective_replay_id_++;
        record.graph_bs   = state.current_real_graph_bs;
        record.current_bs = std::max<int64_t>(0, std::min<int64_t>(state.current_batch_size, record.graph_bs));
        constexpr size_t kTraceIdMaxBytes = sizeof(detail::DecodeProbeTriggerSharedRecord{}.trace_id) - 1;
        for (int64_t lane = 0; lane < record.graph_bs; ++lane) {
            if (lane >= record.current_bs) {
                record.trace_ids[lane].clear();
                record.input_lengths[lane]    = -1;
                record.sequence_lengths[lane] = -1;
                continue;
            }
            if (lane < static_cast<int64_t>(inputs.trace_ids.size())) {
                const auto& trace_id = inputs.trace_ids[lane];
                record.trace_ids[lane].assign(trace_id.data(), std::min(trace_id.size(), kTraceIdMaxBytes));
            } else {
                record.trace_ids[lane].clear();
            }
            if (!hostScalarAt(inputs.attention_inputs.input_lengths, lane, record.input_lengths[lane])) {
                record.input_lengths[lane] = -1;
            }
            if (!hostScalarAt(inputs.attention_inputs.sequence_lengths, lane, record.sequence_lengths[lane])) {
                record.sequence_lengths[lane] = -1;
            }
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to retain retrospective CUDA graph replay metadata: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("Failed to retain retrospective CUDA graph replay metadata");
    }
}

void CudaGraphRunner::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraph(key);
    cuda_graph::graphDeviceSynchronize();
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void CudaGraphRunner::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
    // Common slice operations for input_ids and padding_offset
    inputs.attention_inputs.is_prefill       = is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1;
    inputs.attention_inputs.is_target_verify = is_target_verify_;
    inputs.input_ids     = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, seq_len_or_tokens);
    inputs.input_hiddens = capture_mem_hold_.py_model_inputs_.input_hiddens.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.input_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.input_lengths_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths_d.slice(0, 0, batch_size);
    inputs.attention_inputs.padding_offset =
        capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, seq_len_or_tokens);

    // Common slice operations for attention inputs
    if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.defined()) {
        if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.size(0) > 0) {
            inputs.attention_inputs.prefix_lengths =
                capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.slice(0, 0, batch_size);
            inputs.attention_inputs.prefix_lengths_d =
                capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths_d.slice(0, 0, batch_size);
        } else {
            // For decode CUDA graph mode: prefix_lengths is empty tensor
            inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
        }
    }
    inputs.attention_inputs.sequence_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, batch_size);

    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_kernel_block_id_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.defined() ?
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, batch_size) :
            torch::Tensor();
    inputs.attention_inputs.kv_cache_block_id_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.defined() ?
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, batch_size) :
            torch::Tensor();
    inputs.attention_inputs.cu_seqlens_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens_host.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_kv_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens_d.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens_host.defined() ?
            capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens_host.slice(0, 0, batch_size + 1) :
            torch::Tensor();
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d.slice(0, 0, batch_size);

    const auto& cap_attn = capture_mem_hold_.py_model_inputs_.attention_inputs;
    inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.clear();
    if (!cap_attn.kv_cache_kernel_block_id_device_by_group.empty()
        && !cap_attn.kv_cache_kernel_block_id_host_by_group.empty()) {
        const size_t group = cap_attn.kv_cache_kernel_block_id_device_by_group.size();
        inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.reserve(group);
        inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.reserve(group);
        for (size_t g = 0; g < group; ++g) {
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.push_back(
                cap_attn.kv_cache_kernel_block_id_device_by_group[g].slice(0, 0, batch_size));
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.push_back(
                cap_attn.kv_cache_kernel_block_id_host_by_group[g].slice(0, 0, batch_size));
        }
        selectActiveHybridBlockMapForGroup(inputs.attention_inputs, full_kv_cache_group_id_);
    }

    // Common direct assignments (no slice needed)
    inputs.attention_inputs.dtype = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
    inputs.attention_inputs.kv_cache_layer_to_group =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_layer_to_group;
    inputs.bert_embedding_inputs        = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded = true;
}

CaptureMemoryHold CudaGraphRunner::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    // only when prefill or target model score phase, the num_tokens_per_bs_ > 1
    return CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, tokens_count),
                             inputs,
                             is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1);
}

CudaGraphRunner* CudaGraphRunner::createForPrefill(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph = true;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = params.max_seq_len;
    }
    CudaGraphRunner* runner = new CudaGraphRunner(params, std::move(py_instance));
    runner->initCapture();
    return runner;
}

CudaGraphRunner* CudaGraphRunner::createForDecode(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph = true;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = 1;
    }
    CudaGraphRunner* runner = new CudaGraphRunner(params, std::move(py_instance));
    runner->initCapture();
    return runner;
}

}  // namespace rtp_llm
