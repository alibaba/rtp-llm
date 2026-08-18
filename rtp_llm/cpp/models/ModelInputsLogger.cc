#include "rtp_llm/cpp/models/ModelInputsLogger.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <utility>
#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"

namespace rtp_llm {

namespace {

constexpr size_t  kModelInputsLogMaxBytes        = 100ULL * 1024ULL * 1024ULL;
constexpr size_t  kModelInputsLogFlushThreshold  = 100;
constexpr int64_t kModelInputsLogFlushIntervalUs = 100 * 1000;
constexpr size_t  kMaxTensorLogItems             = 64;

std::string formatLogTime() {
    const int64_t      now_us = autil::TimeUtility::currentTimeInMicroSeconds();
    std::ostringstream ms;
    ms << std::setfill('0') << std::setw(3) << (now_us / 1000 % 1000);
    return autil::TimeUtility::usFormat(now_us, "%Y-%m-%d %H:%M:%S") + "." + ms.str();
}

std::string jsonEscape(const std::string& input) {
    std::ostringstream os;
    for (unsigned char c : input) {
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
                os << static_cast<char>(c);
                break;
        }
    }
    return os.str();
}

std::string combineStringsForLog(const std::vector<std::string>& vec) {
    std::string result = "\" ";
    for (const auto& s : vec) {
        result += s + ", ";
    }
    result += "\"";
    return result;
}

std::string tensorDataSuffix(const std::string& tensor_with_data) {
    const auto pos = tensor_with_data.find(", Data(");
    return pos == std::string::npos ? tensor_with_data : tensor_with_data.substr(pos + 2);
}

std::string tensorLogString(const torch::Tensor& tensor, const torch::Tensor& host_snapshot = {}) {
    const torch::Tensor& meta_tensor  = tensor.defined() ? tensor : host_snapshot;
    const torch::Tensor& value_tensor = host_snapshot.defined() ? host_snapshot : tensor;
    if (!meta_tensor.defined()) {
        return tensorDebugString(meta_tensor);
    }
    const auto meta = tensorDebugString(meta_tensor);
    if (!value_tensor.defined() || value_tensor.is_cuda()) {
        return meta;
    }

    std::string data;
    switch (value_tensor.scalar_type()) {
        case torch::kInt32:
            data = tensorDebugStringWithData<int32_t>(value_tensor, kMaxTensorLogItems);
            break;
        case torch::kInt64:
            data = tensorDebugStringWithData<int64_t>(value_tensor, kMaxTensorLogItems);
            break;
        case torch::kBool:
            data = tensorDebugStringWithData<bool>(value_tensor, kMaxTensorLogItems);
            break;
        default:
            return meta;
    }
    return host_snapshot.defined() && tensor.defined() ? meta + ", " + tensorDataSuffix(data) : data;
}

int64_t firstRequestId(const GptModelInputs& inputs) {
    const auto& tensor = inputs.request_id;
    if (!tensor.defined() || tensor.is_cuda() || tensor.numel() <= 0) {
        return -1;
    }
    if (tensor.scalar_type() == torch::kInt64) {
        return tensor.contiguous().data_ptr<int64_t>()[0];
    }
    if (tensor.scalar_type() == torch::kInt32) {
        return tensor.contiguous().data_ptr<int32_t>()[0];
    }
    return -1;
}

std::string formatModelInputs(const GptModelInputs& inputs) {
    std::ostringstream os;
    os << "GptModelInputs { ";
    bool first  = true;
    auto append = [&](const char* name, const std::string& value) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << name << ": " << value;
    };
    auto appendTensor = [&](const char* name, const torch::Tensor& tensor, const torch::Tensor& host_snapshot) {
        append(name, tensorLogString(tensor, host_snapshot));
    };
    auto appendSize = [&](const char* name, size_t value) { append(name, std::to_string(value)); };
    auto appendBool = [&](const char* name, bool value) { append(name, value ? "true" : "false"); };

    append("trace_ids", combineStringsForLog(inputs.trace_ids));
    appendTensor("combo_tokens", inputs.combo_tokens, inputs.combo_tokens_host_for_log);
    appendTensor("input_lengths", inputs.input_lengths, inputs.input_lengths_host_for_log);
    appendTensor("sequence_lengths", inputs.sequence_lengths, inputs.sequence_lengths_host_for_log);
    appendTensor("lm_output_indexes", inputs.lm_output_indexes, torch::Tensor());
    appendTensor("prefix_lengths", inputs.prefix_lengths, inputs.prefix_lengths_host_for_log);
    appendTensor("sequence_lengths_plus_1", inputs.sequence_lengths_plus_1, torch::Tensor());
    appendTensor("combo_tokens_type_ids", inputs.combo_tokens_type_ids, torch::Tensor());
    appendTensor("combo_position_ids", inputs.combo_position_ids, torch::Tensor());
    appendTensor("last_hidden_states", inputs.last_hidden_states, torch::Tensor());
    appendTensor("attention_mask", inputs.attention_mask, torch::Tensor());
    appendTensor("kv_cache_block_id", inputs.kv_cache_block_id, torch::Tensor());
    appendTensor("kv_cache_layer_to_group", inputs.kv_cache_layer_to_group, torch::Tensor());
    appendTensor("kv_cache_group_types", inputs.kv_cache_group_types, torch::Tensor());
    appendTensor("kv_cache_update_mapping", inputs.kv_cache_update_mapping, torch::Tensor());
    appendTensor("request_id", inputs.request_id, torch::Tensor());
    appendTensor("request_pd_separation", inputs.request_pd_separation, torch::Tensor());
    appendSize("kv_block_stride_bytes", inputs.kv_block_stride_bytes);
    appendSize("kv_scale_stride_bytes", inputs.kv_scale_stride_bytes);
    appendSize("seq_size_per_block", inputs.seq_size_per_block);
    appendSize("kernel_seq_size_per_block", inputs.kernel_seq_size_per_block);
    appendBool("pd_separation", inputs.pd_separation);
    appendBool("decode_entrance", inputs.decode_entrance);
    appendBool("use_opaque_kv_cache_store", inputs.use_opaque_kv_cache_store);
    appendBool("need_all_logits", inputs.need_all_logits);
    appendBool("need_all_hidden_states", inputs.need_all_hidden_states);
    appendBool("need_moe_gating", inputs.need_moe_gating);
    appendBool("warmup", inputs.warmup);
    appendBool("skip_run", inputs.skip_run);
    appendBool("is_fake_stream", inputs.is_fake_stream);
    appendBool("is_target_verify", inputs.is_target_verify);
    os << "}";
    return os.str();
}

}  // namespace

class ModelInputsLogger::Writer {
public:
    Writer(int64_t rank_id, int backup_count): backup_count_(std::max(backup_count, 0)) {
        const auto      log_dir = std::filesystem::path(autil::EnvUtil::getEnv("LOG_PATH", std::string("logs")));
        std::error_code ec;
        std::filesystem::create_directories(log_dir, ec);
        if (ec) {
            RTP_LLM_LOG_WARNING(
                "Failed to create model inputs log directory %s: %s", log_dir.string().c_str(), ec.message().c_str());
            return;
        }

        const auto server_id = autil::EnvUtil::getEnv("FRONTEND_SERVER_ID", 0);
        file_path_ = log_dir / ("model_inputs_r" + std::to_string(rank_id) + "_s" + std::to_string(server_id) + ".log");
        bytes_     = std::filesystem::exists(file_path_, ec) ? std::filesystem::file_size(file_path_, ec) : 0;
        output_.open(file_path_, std::ios::out | std::ios::app);
        if (!output_.is_open()) {
            RTP_LLM_LOG_WARNING("Failed to open model inputs log file %s", file_path_.c_str());
            return;
        }
        valid_ = true;
    }

    void write(const std::string& line) {
        if (!valid_) {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        rotateIfNeeded(line.size() + 1);
        output_ << line << '\n';
        bytes_ += line.size() + 1;
        pending_lines_++;
        const auto now_us = autil::TimeUtility::currentTimeInMicroSeconds();
        if (pending_lines_ >= kModelInputsLogFlushThreshold
            || now_us - last_flush_us_ >= kModelInputsLogFlushIntervalUs) {
            output_.flush();
            pending_lines_ = 0;
            last_flush_us_ = now_us;
        }
    }

private:
    void rotateIfNeeded(size_t next_bytes) {
        if (bytes_ + next_bytes <= kModelInputsLogMaxBytes) {
            return;
        }
        output_.close();
        std::error_code ec;
        if (backup_count_ > 0) {
            std::filesystem::remove(file_path_.string() + "." + std::to_string(backup_count_), ec);
            for (int i = backup_count_ - 1; i >= 1; --i) {
                const auto src = file_path_.string() + "." + std::to_string(i);
                const auto dst = file_path_.string() + "." + std::to_string(i + 1);
                if (std::filesystem::exists(src, ec)) {
                    std::filesystem::rename(src, dst, ec);
                }
            }
            if (std::filesystem::exists(file_path_, ec)) {
                std::filesystem::rename(file_path_, file_path_.string() + ".1", ec);
            }
        } else {
            std::filesystem::remove(file_path_, ec);
        }
        output_.open(file_path_, std::ios::out | std::ios::trunc);
        bytes_ = 0;
    }

    std::mutex            mutex_;
    std::filesystem::path file_path_;
    std::ofstream         output_;
    int                   backup_count_  = 0;
    size_t                bytes_         = 0;
    size_t                pending_lines_ = 0;
    int64_t               last_flush_us_ = 0;
    bool                  valid_         = false;
};

ModelInputsLogger::ModelInputsLogger(int64_t rank_id, int backup_count, kmonitor::MetricsReporterPtr metrics_reporter):
    rank_id_(rank_id), backup_count_(backup_count), metrics_reporter_(std::move(metrics_reporter)) {}

ModelInputsLogger::~ModelInputsLogger() = default;

void ModelInputsLogger::log(const GptModelInputs& inputs) {
    if (inputs.is_fake_stream) {
        return;
    }

    const auto total_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    std::call_once(init_once_,
                   [&]() { writer_ = std::make_unique<ModelInputsLogger::Writer>(rank_id_, backup_count_); });
    if (!writer_) {
        return;
    }

    const auto model_inputs = formatModelInputs(inputs);
    const auto line         = "{\"id\":" + std::to_string(firstRequestId(inputs)) + ",\"log_time\":\"" + formatLogTime()
                      + "\",\"model_inputs\":\"" + jsonEscape(model_inputs) + "\"}";
    writer_->write(line);

    const auto total_us = autil::TimeUtility::currentTimeInMicroSeconds() - total_start_us;
    if (metrics_reporter_) {
        metrics_reporter_->report(total_us, "rtp_llm_model_inputs_log_us", kmonitor::MetricType::GAUGE, nullptr, true);
    }
}

}  // namespace rtp_llm
