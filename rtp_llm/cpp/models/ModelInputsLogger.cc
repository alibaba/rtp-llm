#include "rtp_llm/cpp/models/ModelInputsLogger.h"
#include <ATen/core/Dict.h>
#include <ATen/core/jit_type.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <c10/core/Event.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/serialize.h>
#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
constexpr int64_t     kSchemaVersion   = 1;
constexpr size_t      kChunkMaxBytes   = 64ULL * 1024ULL * 1024ULL;
constexpr size_t      kChunkMaxRecords = 256;
constexpr size_t      kQueueMaxBytes   = 256ULL * 1024ULL * 1024ULL;
std::atomic<uint64_t> g_file_sequence{0};
// clang-format off
#define MODEL_INPUT_TENSORS(X)                                                                                       \
    X(combo_tokens) X(input_lengths) X(sequence_lengths) X(lm_output_indexes) X(lm_output_lengths) X(prefix_lengths) \
    X(combo_tokens_type_ids) X(combo_position_ids) X(last_hidden_states) X(attention_mask)                           \
    X(kv_cache_block_id) X(kv_cache_kernel_block_id) X(kv_cache_group_types) X(kv_cache_update_mapping)             \
    X(text_tokens_mask) X(mm_features_locs) X(input_embeddings_locs)                                                 \
    X(request_id) X(request_pd_separation) X(cache_keys)
// clang-format on
const char* roleName(ModelInputsModelRole role) {
    static constexpr const char* names[] = {"normal", "target", "draft", "draft_prefill"};
    return names[static_cast<size_t>(role)];
}
const char* executionStage(const GptModelInputs& inputs) {
    if (inputs.is_target_verify) {
        return "target_verify";
    }
    const bool prefill = inputs.prefix_lengths.defined() && inputs.prefix_lengths.numel() > 0;
    const bool decode  = inputs.sequence_lengths.defined() && inputs.sequence_lengths.numel() > 0;
    return prefill && decode ? "mixed" : prefill ? "prefill" : decode ? "decode" : "unknown";
}
void addBytes(size_t& total, size_t bytes) {
    if (total <= kChunkMaxBytes) {
        total = bytes > kChunkMaxBytes - total ? kChunkMaxBytes + 1 : total + bytes;
    }
}
void addTensorBytes(size_t& total, const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return;
    }
    const auto element_size = tensor.element_size();
    const auto numel        = static_cast<size_t>(tensor.numel());
    addBytes(total, element_size && numel > kChunkMaxBytes / element_size ? kChunkMaxBytes + 1 : numel * element_size);
}
void addTensorListBytes(size_t& total, const std::optional<std::vector<torch::Tensor>>& tensors) {
    if (tensors) {
        for (const auto& tensor : *tensors) {
            addBytes(total, 256);
            addTensorBytes(total, tensor);
        }
    }
}
size_t estimateBytes(const GptModelInputs& inputs) {
    size_t bytes = 16 * 1024;
    for (const auto& trace_id : inputs.trace_ids) {
        addBytes(bytes, sizeof(std::string) + sizeof(uint64_t) + trace_id.size());
    }
#define ADD_TENSOR_BYTES(field) addTensorBytes(bytes, inputs.field);
    MODEL_INPUT_TENSORS(ADD_TENSOR_BYTES)
#undef ADD_TENSOR_BYTES
    addTensorListBytes(bytes, inputs.multimodal_features);
    addTensorListBytes(bytes, inputs.mm_extra_input);
    addTensorListBytes(bytes, inputs.input_embeddings);
    return bytes;
}
torch::Tensor snapshotTensor(const torch::Tensor& tensor, std::vector<c10::Device>& devices) {
    if (!tensor.defined()) {
        return {};
    }
    if (!tensor.device().is_cpu() && std::find(devices.begin(), devices.end(), tensor.device()) == devices.end()) {
        devices.push_back(tensor.device());
    }
    auto snapshot = tensor.detach();
    return snapshot.is_contiguous() ? snapshot.clone() : snapshot.contiguous();
}
void addTensor(c10::impl::GenericDict&   payload,
               const char*               name,
               const torch::Tensor&      tensor,
               std::vector<c10::Device>& devices,
               c10::impl::GenericDict&   float8_dtypes) {
    if (!tensor.defined()) {
        payload.insert(name, c10::IValue());
        return;
    }
    if (c10::isFloat8Type(tensor.scalar_type())) {
        float8_dtypes.insert(name, c10::toString(tensor.scalar_type()));
    }
    payload.insert(name, snapshotTensor(tensor, devices));
}
void addTensorList(c10::impl::GenericDict&                          payload,
                   const char*                                      name,
                   const std::optional<std::vector<torch::Tensor>>& tensors,
                   std::vector<c10::Device>&                        devices,
                   c10::impl::GenericDict&                          float8_dtypes) {
    if (!tensors) {
        payload.insert(name, c10::IValue());
        return;
    }
    c10::List<torch::Tensor> values;
    for (size_t i = 0; i < tensors->size(); ++i) {
        const auto& tensor = (*tensors)[i];
        if (tensor.defined() && c10::isFloat8Type(tensor.scalar_type())) {
            float8_dtypes.insert(std::string(name) + "[" + std::to_string(i) + "]",
                                 c10::toString(tensor.scalar_type()));
        }
        values.push_back(snapshotTensor(tensor, devices));
    }
    payload.insert(name, std::move(values));
}
c10::impl::GenericDict snapshotPayload(const GptModelInputs&     inputs,
                                       ModelInputsModelRole      role,
                                       int64_t                   model_id,
                                       int64_t                   sequence,
                                       int64_t                   dropped_before,
                                       std::vector<c10::Device>& devices) {
    c10::impl::GenericDict payload(c10::StringType::get(), c10::AnyType::get());
    payload.reserve(64);
    payload.insert("schema_version", kSchemaVersion);
    payload.insert("record_type", "gpt_model_inputs_snapshot");
    payload.insert("record_sequence", sequence);
    payload.insert("dropped_before", dropped_before);
    payload.insert("model_role", roleName(role));
    payload.insert("execution_stage", executionStage(inputs));
    payload.insert("model_id", model_id);
    payload.insert("trace_ids", inputs.trace_ids);
    c10::impl::GenericDict float8_dtypes(c10::StringType::get(), c10::StringType::get());
#define ADD_TENSOR(field) addTensor(payload, #field, inputs.field, devices, float8_dtypes);
    MODEL_INPUT_TENSORS(ADD_TENSOR)
#undef ADD_TENSOR
    addTensorList(payload, "multimodal_features", inputs.multimodal_features, devices, float8_dtypes);
    addTensorList(payload, "mm_extra_input", inputs.mm_extra_input, devices, float8_dtypes);
    addTensorList(payload, "input_embeddings", inputs.input_embeddings, devices, float8_dtypes);
    payload.insert("float8_dtypes", std::move(float8_dtypes));
    payload.insert("kv_block_stride_bytes", static_cast<int64_t>(inputs.kv_block_stride_bytes));
    payload.insert("kv_scale_stride_bytes", static_cast<int64_t>(inputs.kv_scale_stride_bytes));
    payload.insert("seq_size_per_block", static_cast<int64_t>(inputs.seq_size_per_block));
    payload.insert("kernel_seq_size_per_block", static_cast<int64_t>(inputs.kernel_seq_size_per_block));
    payload.insert("pd_separation", inputs.pd_separation);
    payload.insert("decode_entrance", inputs.decode_entrance);
    payload.insert("use_opaque_kv_cache_store", inputs.use_opaque_kv_cache_store);
    payload.insert("need_all_logits", inputs.need_all_logits);
    payload.insert("need_moe_gating", inputs.need_moe_gating);
    payload.insert("warmup", inputs.warmup);
    payload.insert("skip_run", inputs.skip_run);
    payload.insert("is_fake_stream", inputs.is_fake_stream);
    payload.insert("is_target_verify", inputs.is_target_verify);
    return payload;
}
std::vector<std::shared_ptr<c10::Event>> readyEvents(const std::vector<c10::Device>& devices) {
    std::vector<std::shared_ptr<c10::Event>> events;
    for (const auto& device : devices) {
        c10::impl::VirtualGuardImpl guard(device.type());
        auto                        event = std::make_shared<c10::Event>(device.type());
        event->record(guard.getStream(device));
        events.push_back(std::move(event));
    }
    return events;
}
torch::Tensor tensorForDump(const torch::Tensor& tensor) {
    auto output = tensor.detach();
    if (c10::isFloat8Type(output.scalar_type())) {
        output = output.view(torch::kChar);
    }
    return (output.device().is_cpu() ? output : output.cpu()).contiguous();
}
c10::IValue valueForDump(const c10::IValue& value) {
    if (value.isTensor()) {
        return tensorForDump(value.toTensor());
    }
    if (value.isTensorList()) {
        c10::List<torch::Tensor> tensors;
        for (const auto& tensor : value.toTensorList()) {
            tensors.push_back(tensorForDump(tensor));
        }
        return tensors;
    }
    return value;
}
class DumpWriter {
public:
    DumpWriter(int64_t rank_id, int backup_count): backup_count_(std::max(backup_count, 1)) {
        const auto log_path  = std::filesystem::path(autil::EnvUtil::getEnv("LOG_PATH", std::string("logs")));
        const auto server_id = autil::EnvUtil::getEnv("FRONTEND_SERVER_ID", 0);
        prefix_              = "model_inputs_r" + std::to_string(rank_id) + "_s" + std::to_string(server_id) + "_p"
                  + std::to_string(::getpid()) + "_";
        output_dir_ = log_path / "model_inputs";

        std::error_code ec;
        std::filesystem::create_directories(log_path, ec);
        if (ec || (::mkdir(output_dir_.c_str(), S_IRWXU) != 0 && errno != EEXIST)) {
            RTP_LLM_LOG_WARNING("Failed to create model inputs dump directory %s", output_dir_.c_str());
            return;
        }
        dir_fd_ = ::open(output_dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        struct stat st{};
        if (dir_fd_ < 0 || ::fstat(dir_fd_, &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != ::geteuid()
            || ::fchmod(dir_fd_, S_IRWXU) != 0) {
            RTP_LLM_LOG_WARNING("Model inputs dump directory %s is not owner-only", output_dir_.c_str());
            return;
        }
        valid_ = true;
        cleanup();
    }
    ~DumpWriter() {
        flush();
        if (dir_fd_ >= 0) {
            ::close(dir_fd_);
        }
    }
    bool valid() const {
        return valid_;
    }
    bool write(c10::impl::GenericDict payload, size_t bytes) {
        if (!records_.empty() && (records_.size() >= kChunkMaxRecords || bytes > kChunkMaxBytes - pending_bytes_)
            && !flush()) {
            return false;
        }
        pending_bytes_ += bytes;
        records_.emplace_back(std::move(payload));
        return (records_.size() < kChunkMaxRecords && pending_bytes_ < kChunkMaxBytes) || flush();
    }
    bool flush() {
        if (records_.empty()) {
            return true;
        }
        c10::List<c10::IValue> record_list(c10::AnyType::get());
        for (const auto& record : records_) {
            record_list.push_back(record);
        }
        c10::impl::GenericDict chunk(c10::StringType::get(), c10::AnyType::get());
        chunk.insert("schema_version", kSchemaVersion);
        chunk.insert("record_type", "model_inputs_chunk");
        chunk.insert("records", std::move(record_list));
        const auto         bytes = torch::pickle_save(c10::IValue(chunk));
        std::ostringstream id;
        id << std::setw(20) << std::setfill('0') << g_file_sequence.fetch_add(1, std::memory_order_relaxed);
        const auto final_name = prefix_ + id.str() + ".pt";
        const auto temp_name  = "." + final_name + ".tmp." + std::to_string(::getpid());
        const int fd = ::openat(dir_fd_, temp_name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        bool      ok = fd >= 0 && writeAll(fd, bytes) && ::fchmod(fd, 0600) == 0;
        if (fd >= 0 && ::close(fd) != 0) {
            ok = false;
        }
        if (!ok || ::renameat(dir_fd_, temp_name.c_str(), dir_fd_, final_name.c_str()) != 0) {
            if (fd >= 0) {
                ::unlinkat(dir_fd_, temp_name.c_str(), 0);
            }
            RTP_LLM_LOG_WARNING("Failed to atomically write model inputs dump %s", final_name.c_str());
            return false;
        }
        records_.clear();
        pending_bytes_ = 0;
        cleanup();
        return true;
    }

private:
    bool writeAll(int fd, const std::vector<char>& bytes) {
        size_t offset = 0;
        while (offset < bytes.size()) {
            const auto written = ::write(fd, bytes.data() + offset, bytes.size() - offset);
            if (written < 0 && errno == EINTR) {
                continue;
            }
            if (written <= 0) {
                return false;
            }
            offset += static_cast<size_t>(written);
        }
        return true;
    }
    void cleanup() {
        std::vector<std::filesystem::path> dumps;
        std::error_code                    ec;
        for (const auto& entry : std::filesystem::directory_iterator(output_dir_, ec)) {
            const auto name = entry.path().filename().string();
            if (name.rfind(prefix_, 0) == 0 && entry.path().extension() == ".pt") {
                dumps.push_back(entry.path());
            }
        }
        std::sort(dumps.begin(), dumps.end());
        while (dumps.size() > static_cast<size_t>(backup_count_)) {
            std::filesystem::remove(dumps.front(), ec);
            dumps.erase(dumps.begin());
        }
    }

    std::filesystem::path               output_dir_;
    std::string                         prefix_;
    int                                 backup_count_  = 1;
    int                                 dir_fd_        = -1;
    bool                                valid_         = false;
    size_t                              pending_bytes_ = 0;
    std::vector<c10::impl::GenericDict> records_;
};
struct PendingRecord {
    c10::impl::GenericDict                   payload;
    std::vector<std::shared_ptr<c10::Event>> events;
    int64_t                                  dropped_before;
    size_t                                   bytes;
};
}  // namespace
class ModelInputsLogger::Worker {
public:
    Worker(int64_t rank_id, int backup_count): writer_(rank_id, backup_count), disabled_(!writer_.valid()) {
        thread_ = std::thread([this]() { run(); });
    }
    ~Worker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_deadline_ = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            stopping_          = true;
        }
        ready_cv_.notify_one();
        thread_.join();
    }
    void enqueue(const GptModelInputs& inputs, ModelInputsModelRole role, int64_t model_id) {
        if (inputs.is_fake_stream || disabled_) {
            return;
        }
        const auto sequence = next_sequence_.fetch_add(1, std::memory_order_relaxed);
        const auto bytes    = estimateBytes(inputs);
        if (disabled_ || bytes > kChunkMaxBytes || !reserve(bytes)) {
            drop("record exceeds limits or writer is unavailable");
            return;
        }
        const auto dropped_before = dropped_since_last_.exchange(0, std::memory_order_relaxed);
        try {
            std::vector<c10::Device> devices;
            auto payload = snapshotPayload(inputs, role, model_id, sequence, dropped_before, devices);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push_back(PendingRecord{std::move(payload), readyEvents(devices), dropped_before, bytes});
            }
            ready_cv_.notify_one();
        } catch (const std::exception& e) {
            dropped_since_last_.fetch_add(dropped_before, std::memory_order_relaxed);
            release(bytes);
            drop(e.what());
        }
    }

private:
    bool reserve(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || bytes > kQueueMaxBytes - pending_bytes_) {
            return false;
        }
        pending_bytes_ += bytes;
        return true;
    }
    void release(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_bytes_ -= bytes;
    }
    void run() {
        while (true) {
            std::optional<PendingRecord> record;
            bool                         finish = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_cv_.wait_for(lock, std::chrono::seconds(1), [this]() { return stopping_ || !queue_.empty(); });
                if (!queue_.empty()) {
                    record.emplace(std::move(queue_.front()));
                    queue_.pop_front();
                } else {
                    finish = stopping_ && pending_bytes_ == 0;
                }
            }
            if (record) {
                process(*record);
                release(record->bytes);
                continue;
            }
            writeGap();
            if (finish) {
                return;
            }
        }
    }
    void process(PendingRecord& record) {
        try {
            if (disabled_ || !waitForEvents(record.events)) {
                dropped_since_last_.fetch_add(record.dropped_before, std::memory_order_relaxed);
                drop("CUDA snapshot did not become ready");
                return;
            }
            for (const auto& item : record.payload) {
                item.setValue(valueForDump(item.value()));
            }
            if (!writer_.write(std::move(record.payload), record.bytes)) {
                disabled_ = true;
            }
        } catch (const std::exception& e) {
            dropped_since_last_.fetch_add(record.dropped_before, std::memory_order_relaxed);
            drop(e.what());
        }
    }
    bool waitForEvents(const std::vector<std::shared_ptr<c10::Event>>& events) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (const auto& event : events) {
            while (!event->query()) {
                const auto wait_deadline = stopping_.load(std::memory_order_acquire) ? shutdown_deadline_ : deadline;
                if (std::chrono::steady_clock::now() >= wait_deadline) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        return true;
    }
    void writeGap() {
        try {
            if (!disabled_) {
                const auto dropped = dropped_since_last_.exchange(0, std::memory_order_relaxed);
                if (dropped > 0) {
                    c10::impl::GenericDict payload(c10::StringType::get(), c10::AnyType::get());
                    payload.insert("schema_version", kSchemaVersion);
                    payload.insert("record_type", "model_inputs_gap");
                    payload.insert("dropped_count", dropped);
                    payload.insert("next_attempt_sequence", next_sequence_.load(std::memory_order_relaxed));
                    disabled_ = !writer_.write(std::move(payload), 16 * 1024);
                }
                disabled_ = disabled_ || !writer_.flush();
            }
        } catch (const std::exception& e) {
            disabled_ = true;
            RTP_LLM_LOG_WARNING("Failed to write model inputs gap: %s", e.what());
        }
    }
    void drop(const char* reason) {
        dropped_since_last_.fetch_add(1, std::memory_order_relaxed);
        const auto count = dropped_records_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count == 1 || count % 1000 == 0) {
            RTP_LLM_LOG_WARNING("Dropped model inputs dump record: %s; dropped=%zu", reason, count);
        }
        ready_cv_.notify_one();
    }

    DumpWriter                            writer_;
    std::thread                           thread_;
    std::mutex                            mutex_;
    std::condition_variable               ready_cv_;
    std::deque<PendingRecord>             queue_;
    size_t                                pending_bytes_ = 0;
    std::atomic<bool>                     stopping_{false};
    std::chrono::steady_clock::time_point shutdown_deadline_;
    std::atomic<bool>                     disabled_{false};
    std::atomic<size_t>                   dropped_records_{0};
    std::atomic<int64_t>                  dropped_since_last_{0};
    std::atomic<int64_t>                  next_sequence_{0};
};
ModelInputsLogger::ModelInputsLogger(int64_t rank_id, int backup_count, kmonitor::MetricsReporterPtr metrics_reporter):
    metrics_reporter_(std::move(metrics_reporter)), worker_(std::make_unique<Worker>(rank_id, backup_count)) {}
ModelInputsLogger::~ModelInputsLogger() = default;
void ModelInputsLogger::log(const GptModelInputs& inputs, ModelInputsModelRole role, int64_t model_id) {
    if (inputs.is_fake_stream) {
        return;
    }
    const auto start_us = autil::TimeUtility::currentTimeInMicroSeconds();
    worker_->enqueue(inputs, role, model_id);
    if (metrics_reporter_) {
        const auto elapsed_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_us;
        metrics_reporter_->report(
            elapsed_us, "rtp_llm_model_inputs_log_us", kmonitor::MetricType::GAUGE, nullptr, true);
    }
}
}  // namespace rtp_llm
