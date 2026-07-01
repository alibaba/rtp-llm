#include "rtp_llm/cpp/models/ModelInputsLogger.h"

#include <ATen/core/Dict.h>
#include <ATen/core/jit_type.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>
#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <torch/serialize.h>

namespace rtp_llm {

namespace {

constexpr size_t  kModelInputsDumpMaxBytes        = 100ULL * 1024ULL * 1024ULL;
constexpr size_t  kModelInputsDumpFlushThreshold  = 100;
constexpr int64_t kModelInputsDumpFlushIntervalUs = 100 * 1000;
constexpr size_t  kRecordLengthBytes              = sizeof(uint64_t);

std::string timestampForFile(int64_t now_us) {
    std::ostringstream us;
    us << std::setfill('0') << std::setw(6) << (now_us % 1000000);
    return autil::TimeUtility::usFormat(now_us, "%Y%m%d_%H%M%S") + "_" + us.str();
}

torch::Tensor tensorForDump(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return {};
    }
    auto out = tensor.detach();
    if (out.scalar_type() == torch::kFloat8_e4m3fn) {
        out = out.view(torch::kChar);
    }
    return out.contiguous();
}

void addTensor(c10::impl::GenericDict& payload,
               const std::string&      name,
               const torch::Tensor&    tensor,
               const torch::Tensor&    host_snapshot = {}) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("model_inputs.dump.tensor(%s)", name.c_str());
    const auto& source = host_snapshot.defined() ? host_snapshot : tensor;
    if (!source.defined()) {
        payload.insert(name, c10::IValue());
        return;
    }
    if (source.is_cuda()) {
        payload.insert(name, c10::IValue());
        return;
    }
    payload.insert(name, tensorForDump(source));
}

void addSize(c10::impl::GenericDict& payload, const std::string& name, size_t value) {
    payload.insert(name, static_cast<int64_t>(value));
}

c10::impl::GenericDict buildModelInputsPayload(const GptModelInputs& inputs) {
    c10::impl::GenericDict payload(c10::StringType::get(), c10::AnyType::get());
    payload.reserve(96);

    payload.insert("trace_ids", inputs.trace_ids);

    addTensor(payload, "combo_tokens", inputs.combo_tokens, inputs.combo_tokens_host_for_log);
    addTensor(payload, "input_lengths", inputs.input_lengths, inputs.input_lengths_host_for_log);
    addTensor(payload, "sequence_lengths", inputs.sequence_lengths, inputs.sequence_lengths_host_for_log);
    addTensor(payload, "lm_output_indexes", inputs.lm_output_indexes);
    addTensor(payload, "prefix_lengths", inputs.prefix_lengths, inputs.prefix_lengths_host_for_log);
    addTensor(payload, "combo_tokens_type_ids", inputs.combo_tokens_type_ids);
    addTensor(payload, "combo_position_ids", inputs.combo_position_ids);
    addTensor(payload, "last_hidden_states", inputs.last_hidden_states);
    addTensor(payload, "attention_mask", inputs.attention_mask);
    addTensor(payload, "kv_cache_block_id", inputs.kv_cache_block_id);
    addTensor(payload, "kv_cache_layer_to_group", inputs.kv_cache_layer_to_group);
    addTensor(payload, "kv_cache_group_types", inputs.kv_cache_group_types);
    addTensor(payload, "kv_cache_update_mapping", inputs.kv_cache_update_mapping);
    addTensor(payload, "request_id", inputs.request_id);
    addTensor(payload, "request_pd_separation", inputs.request_pd_separation);

    addSize(payload, "kv_block_stride_bytes", inputs.kv_block_stride_bytes);
    addSize(payload, "kv_scale_stride_bytes", inputs.kv_scale_stride_bytes);
    addSize(payload, "seq_size_per_block", inputs.seq_size_per_block);
    addSize(payload, "kernel_seq_size_per_block", inputs.kernel_seq_size_per_block);
    payload.insert("pd_separation", inputs.pd_separation);
    payload.insert("decode_entrance", inputs.decode_entrance);
    payload.insert("need_all_logits", inputs.need_all_logits);
    payload.insert("need_moe_gating", inputs.need_moe_gating);
    payload.insert("warmup", inputs.warmup);
    payload.insert("skip_run", inputs.skip_run);
    payload.insert("is_fake_stream", inputs.is_fake_stream);
    payload.insert("is_target_verify", inputs.is_target_verify);

    return payload;
}

}  // namespace

class ModelInputsLogger::Writer {
public:
    Writer(int64_t rank_id, int backup_count): backup_count_(std::max(backup_count, 0)) {
        const auto log_path  = autil::EnvUtil::getEnv("LOG_PATH", std::string("logs"));
        const auto server_id = autil::EnvUtil::getEnv("FRONTEND_SERVER_ID", 0);
        prefix_              = "model_inputs_r" + std::to_string(rank_id) + "_s" + std::to_string(server_id);
        output_dir_          = std::filesystem::path(log_path);

        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec);
        if (ec) {
            RTP_LLM_LOG_WARNING("Failed to create model inputs dump directory %s: %s",
                                output_dir_.string().c_str(),
                                ec.message().c_str());
            return;
        }

        file_path_ = output_dir_ / (prefix_ + ".pt");
        bytes_     = std::filesystem::exists(file_path_, ec) ? std::filesystem::file_size(file_path_, ec) : 0;
        open(std::ios::out | std::ios::binary | std::ios::app);
    }

    void write(const std::vector<char>& record) {
        if (!valid_) {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        rotateIfNeeded(kRecordLengthBytes + record.size());

        const uint64_t record_size = record.size();
        output_.write(reinterpret_cast<const char*>(&record_size), sizeof(record_size));
        output_.write(record.data(), record.size());
        bytes_ += kRecordLengthBytes + record.size();

        pending_records_++;
        const auto now_us = autil::TimeUtility::currentTimeInMicroSeconds();
        if (pending_records_ >= kModelInputsDumpFlushThreshold
            || now_us - last_flush_us_ >= kModelInputsDumpFlushIntervalUs) {
            output_.flush();
            pending_records_ = 0;
            last_flush_us_   = now_us;
        }
    }

private:
    void open(std::ios_base::openmode mode) {
        output_.open(file_path_, mode);
        if (!output_.is_open()) {
            RTP_LLM_LOG_WARNING("Failed to open model inputs dump file %s", file_path_.c_str());
            valid_ = false;
            return;
        }
        valid_ = true;
    }

    void rotateIfNeeded(size_t next_bytes) {
        if (bytes_ == 0 || bytes_ + next_bytes <= kModelInputsDumpMaxBytes) {
            return;
        }

        output_.close();
        const auto rotated_path =
            output_dir_ / (prefix_ + "_" + timestampForFile(autil::TimeUtility::currentTimeInMicroSeconds()) + ".pt");
        std::error_code ec;
        std::filesystem::rename(file_path_, rotated_path, ec);
        if (ec) {
            RTP_LLM_LOG_WARNING("Failed to rotate model inputs dump file from %s to %s: %s",
                                file_path_.c_str(),
                                rotated_path.c_str(),
                                ec.message().c_str());
        }
        cleanupOldFiles();
        bytes_           = 0;
        pending_records_ = 0;
        open(std::ios::out | std::ios::binary | std::ios::trunc);
    }

    void cleanupOldFiles() {
        if (backup_count_ <= 0) {
            return;
        }

        std::vector<std::filesystem::directory_entry> rotated_files;
        std::error_code                               ec;
        for (const auto& entry : std::filesystem::directory_iterator(output_dir_, ec)) {
            if (ec || !entry.is_regular_file()) {
                continue;
            }
            const auto name = entry.path().filename().string();
            if (name.rfind(prefix_ + "_", 0) == 0 && entry.path().extension() == ".pt") {
                rotated_files.push_back(entry);
            }
        }
        if (rotated_files.size() <= static_cast<size_t>(backup_count_)) {
            return;
        }
        std::sort(rotated_files.begin(), rotated_files.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.path().filename().string() < rhs.path().filename().string();
        });
        const auto remove_count = rotated_files.size() - static_cast<size_t>(backup_count_);
        for (size_t i = 0; i < remove_count; ++i) {
            std::filesystem::remove(rotated_files[i], ec);
        }
    }

    std::mutex            mutex_;
    std::filesystem::path output_dir_;
    std::filesystem::path file_path_;
    std::ofstream         output_;
    std::string           prefix_;
    int                   backup_count_    = 0;
    size_t                bytes_           = 0;
    size_t                pending_records_ = 0;
    int64_t               last_flush_us_   = 0;
    bool                  valid_           = false;
};

ModelInputsLogger::ModelInputsLogger(int64_t rank_id, int backup_count, kmonitor::MetricsReporterPtr metrics_reporter):
    metrics_reporter_(std::move(metrics_reporter)), writer_(std::make_unique<Writer>(rank_id, backup_count)) {}

ModelInputsLogger::~ModelInputsLogger() = default;

void ModelInputsLogger::log(const GptModelInputs& inputs) {
    if (inputs.is_fake_stream) {
        return;
    }

    RTP_LLM_PROFILE_SCOPE("model_inputs.dump.total");
    const auto total_start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    try {
        c10::impl::GenericDict payload(c10::StringType::get(), c10::AnyType::get());
        {
            RTP_LLM_PROFILE_SCOPE("model_inputs.dump.build_payload");
            payload = buildModelInputsPayload(inputs);
        }
        std::vector<char> pickled;
        {
            RTP_LLM_PROFILE_SCOPE("model_inputs.dump.pickle_save");
            pickled = torch::pickle_save(c10::IValue(std::move(payload)));
        }
        {
            RTP_LLM_PROFILE_SCOPE("model_inputs.dump.write_file");
            writer_->write(pickled);
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to dump model inputs: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("Failed to dump model inputs: unknown exception");
    }

    const auto total_us = autil::TimeUtility::currentTimeInMicroSeconds() - total_start_us;
    try {
        if (metrics_reporter_) {
            RTP_LLM_PROFILE_SCOPE("model_inputs.dump.report_metric");
            metrics_reporter_->report(
                total_us, "rtp_llm_model_inputs_log_us", kmonitor::MetricType::GAUGE, nullptr, true);
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to report model inputs dump metric: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("Failed to report model inputs dump metric: unknown exception");
    }
}

}  // namespace rtp_llm
