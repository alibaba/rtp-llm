#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
using namespace torch_ext;

namespace rtp_llm {

namespace {

std::atomic<uint64_t> g_decode_checksum_record_id{0};
std::mutex            g_decode_checksum_mutex;

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
    std::string              file;
    std::vector<std::string> trace_filters;
};

const DecodeChecksumConfig& decodeChecksumConfig() {
    static DecodeChecksumConfig config = [] {
        DecodeChecksumConfig c;
        c.enabled       = envFlag("RTPLLM_DECODE_CHECKSUM_DEBUG", false);
        c.sync_device   = envFlag("RTPLLM_DECODE_CHECKSUM_SYNC_DEVICE", true);
        c.every         = std::max<int64_t>(1, envInt64("RTPLLM_DECODE_CHECKSUM_EVERY", 1));
        c.max_records   = envInt64("RTPLLM_DECODE_CHECKSUM_MAX_RECORDS", 0);
        c.max_lanes     = envInt64("RTPLLM_DECODE_CHECKSUM_MAX_LANES", 8);
        c.trace_filters = splitCsv(envString("RTPLLM_DECODE_CHECKSUM_TRACE_FILTER"));

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

std::vector<int64_t> selectedLanes(const std::vector<std::string>& trace_ids,
                                   int64_t                         current_bs,
                                   const DecodeChecksumConfig&     config) {
    std::vector<int64_t> lanes;
    const int64_t        limit = config.max_lanes <= 0 ? current_bs : std::min<int64_t>(current_bs, config.max_lanes);
    for (int64_t lane = 0; lane < current_bs; ++lane) {
        const std::string trace_id = lane < static_cast<int64_t>(trace_ids.size()) ? trace_ids[lane] : "";
        if (!traceSelected(trace_id, config.trace_filters)) {
            continue;
        }
        lanes.push_back(lane);
        if (static_cast<int64_t>(lanes.size()) >= limit) {
            break;
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
            os << ",\"sum\":" << flat.sum().item<double>();
            os << ",\"abs_sum\":" << flat.abs().sum().item<double>();
            os << ",\"min\":" << flat.min().item<double>();
            os << ",\"max\":" << flat.max().item<double>();
            os << ",\"sample\":[";
            const auto sample_count = std::min<int64_t>(flat.numel(), 8);
            const auto* data        = flat.data_ptr<float>();
            for (int64_t i = 0; i < sample_count; ++i) {
                if (i != 0) {
                    os << ",";
                }
                os << data[i];
            }
            os << "]";
        }
        os << "}";
    } catch (const std::exception& e) {
        os << "{\"error\":\"" << jsonEscape(e.what()) << "\"}";
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

void writeDecodeChecksumRecord(const char*            stage,
                               const PyModelInputs&  original_inputs,
                               const PyModelInputs&  graph_inputs,
                               const torch::Tensor&  hidden_states,
                               const CudaGraphState& state,
                               int32_t               full_kv_cache_group_id) {
    const auto& config = decodeChecksumConfig();
    if (!config.enabled) {
        return;
    }

    const uint64_t record_id = g_decode_checksum_record_id.fetch_add(1, std::memory_order_relaxed);
    if (config.max_records > 0 && record_id >= static_cast<uint64_t>(config.max_records)) {
        return;
    }
    if (record_id % static_cast<uint64_t>(config.every) != 0) {
        return;
    }

    const auto lanes = selectedLanes(original_inputs.trace_ids, state.current_batch_size, config);
    if (lanes.empty()) {
        return;
    }
    if (config.sync_device) {
        cuda_graph::graphDeviceSynchronize();
    }

    const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();

    std::ostringstream os;
    os << "{\"record_id\":" << record_id;
    os << ",\"ts_us\":" << now_us;
    os << ",\"pid\":" << getpid();
    os << ",\"rank\":\"" << jsonEscape(envString("WORLD_RANK", "")) << "\"";
    os << ",\"stage\":\"" << stage << "\"";
    os << ",\"path\":\"cuda_graph_decode\"";
    os << ",\"current_bs\":" << state.current_batch_size;
    os << ",\"graph_bs\":" << state.current_real_graph_bs;
    os << ",\"seq_len_sum\":" << state.seq_len_sum;
    os << ",\"full_kv_cache_group_id\":" << full_kv_cache_group_id;
    os << ",\"lane_count\":" << lanes.size();
    os << ",\"lanes\":[";
    for (size_t i = 0; i < lanes.size(); ++i) {
        const int64_t     lane     = lanes[i];
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
        os << "}";
    }
    os << "]}\n";

    std::lock_guard<std::mutex> lock(g_decode_checksum_mutex);
    std::ofstream               out(config.file, std::ios::app);
    out << os.str();
}

}  // namespace

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
        writeDecodeChecksumRecord("before_replay", inputs, graph_inputs, torch::Tensor(), state, full_kv_cache_group_id_);
        {
            RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayDecode)");
            replayDecode(state.current_real_graph_bs);
        }
        outputs.hidden_states =
            graph_instances_[state.current_real_graph_bs].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state.seq_len_sum);
        writeDecodeChecksumRecord(
            "after_replay", inputs, graph_inputs, outputs.hidden_states, state, full_kv_cache_group_id_);
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
        auto&               graph               = graph_instances_[key].graph_;
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
