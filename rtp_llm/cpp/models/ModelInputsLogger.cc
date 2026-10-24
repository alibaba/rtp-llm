#include "rtp_llm/cpp/models/ModelInputsLogger.h"
#include <ATen/core/Dict.h>
#include <ATen/core/jit_type.h>
#include <algorithm>
#include <array>
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
#include "rtp_llm/cpp/utils/ExecutionStageRecorder.h"
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
    X(combo_tokens) X(input_lengths) X(sequence_lengths) X(lm_output_indexes) X(custom_output_indexes) X(lm_output_lengths) X(prefix_lengths) \
    X(sequence_lengths_plus_1)                                                                                      \
    X(combo_tokens_type_ids) X(combo_position_ids) X(last_hidden_states) X(attention_mask)                           \
    X(kv_cache_block_id) X(kv_cache_kernel_block_id) X(kv_cache_group_types) X(kv_cache_update_mapping)             \
    X(text_tokens_mask) X(mm_features_locs) X(input_embeddings_locs)                                                 \
    X(request_id) X(request_pd_separation) X(cache_keys)
// clang-format on
const char* roleName(ModelInputsModelRole role) {
    static constexpr const char* names[] = {"normal", "target", "draft", "draft_prefill"};
    return names[static_cast<size_t>(role)];
}
struct LoggerTraceContext {
    int64_t              rank_id;
    int64_t              model_id;
    int64_t              record_sequence;
    ModelInputsModelRole role;
    std::string          trace_ids_details;
    uint64_t             enqueue_id    = 0;
    uint64_t             invocation_id = 0;
    std::string          producer_events_details{};
};
// Diagnostic formatting must neither touch CUDA nor change logger error handling.
template<typename Formatter>
std::string loggerTraceDetails(Formatter&& format) noexcept {
    try {
        if (!executionStageRecorder().enabled()) {
            return {};
        }
        std::ostringstream out;
        format(out);
        return out.str();
    } catch (...) {
        return {};
    }
}
void appendTraceContext(std::ostream& out, const LoggerTraceContext& context) {
    out << "rank=" << context.rank_id << " model_id=" << context.model_id << " model_role=" << roleName(context.role)
        << " record_sequence=" << context.record_sequence << " enqueue_id=" << context.enqueue_id
        << " invocation_id=" << context.invocation_id << context.trace_ids_details << context.producer_events_details;
}
std::string recordTraceDetails(const LoggerTraceContext& context) noexcept {
    return loggerTraceDetails([&](std::ostream& out) { appendTraceContext(out, context); });
}
std::string tensorTraceDetails(const LoggerTraceContext& context,
                               const char*               name,
                               int64_t                   index,
                               const torch::Tensor&      tensor,
                               const c10::IValue*        key               = nullptr,
                               const char*               operation_details = nullptr) noexcept {
    return loggerTraceDetails([&](std::ostream& out) {
        appendTraceContext(out, context);
        out << " tensor=";
        if (key) {
            out << key->toStringRef();
        } else {
            out << name;
        }
        if (index >= 0) {
            out << " index=" << index;
        }
        if (tensor.defined()) {
            out << " device=" << tensor.device() << " dtype=" << tensor.scalar_type() << " shape=" << tensor.sizes()
                << " bytes=" << tensor.nbytes();
        } else {
            out << " defined=false";
        }
        if (operation_details) {
            out << ' ' << operation_details;
        }
    });
}
std::string eventTraceDetails(const LoggerTraceContext& context,
                              const c10::Event*         event,
                              size_t                    index,
                              const c10::Device*        device = nullptr,
                              const c10::Stream*        stream = nullptr) noexcept {
    return loggerTraceDetails([&](std::ostream& out) {
        appendTraceContext(out, context);
        // Object identity links the existing producer event to worker queries.
        // Do not query the native CUDA event handle for diagnostics.
        out << " event_index=" << index << " event_object=" << static_cast<const void*>(event);
        if (device) {
            out << " device=" << *device;
        }
        if (stream) {
            out << " producer_stream_id=" << stream->id();
        }
    });
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
    for (const auto& tag : inputs.kv_cache_group_tags) {
        addBytes(bytes, sizeof(std::string) + sizeof(uint64_t) + tag.size());
    }
#define ADD_TENSOR_BYTES(field) addTensorBytes(bytes, inputs.field);
    MODEL_INPUT_TENSORS(ADD_TENSOR_BYTES)
#undef ADD_TENSOR_BYTES
    addTensorListBytes(bytes, inputs.multimodal_features);
    addTensorListBytes(bytes, inputs.mm_extra_input);
    addTensorListBytes(bytes, inputs.input_embeddings);
    return bytes;
}
torch::Tensor snapshotTensor(const torch::Tensor&      tensor,
                             std::vector<c10::Device>& devices,
                             const LoggerTraceContext& context,
                             const char*               name,
                             int64_t                   index = -1) {
    if (!tensor.defined()) {
        return {};
    }
    if (!tensor.device().is_cpu() && std::find(devices.begin(), devices.end(), tensor.device()) == devices.end()) {
        devices.push_back(tensor.device());
    }
    auto                snapshot   = tensor.detach();
    const bool          contiguous = snapshot.is_contiguous();
    ExecutionStageScope trace(contiguous ? "model_inputs.snapshot.clone" : "model_inputs.snapshot.contiguous",
                              tensorTraceDetails(context, name, index, tensor));
    return contiguous ? snapshot.clone() : snapshot.contiguous();
}
void addTensor(c10::impl::GenericDict&   payload,
               const char*               name,
               const torch::Tensor&      tensor,
               std::vector<c10::Device>& devices,
               c10::impl::GenericDict&   float8_dtypes,
               const LoggerTraceContext& context) {
    if (!tensor.defined()) {
        payload.insert(name, c10::IValue());
        return;
    }
    if (c10::isFloat8Type(tensor.scalar_type())) {
        float8_dtypes.insert(name, c10::toString(tensor.scalar_type()));
    }
    payload.insert(name, snapshotTensor(tensor, devices, context, name));
}
void addTensorList(c10::impl::GenericDict&                          payload,
                   const char*                                      name,
                   const std::optional<std::vector<torch::Tensor>>& tensors,
                   std::vector<c10::Device>&                        devices,
                   c10::impl::GenericDict&                          float8_dtypes,
                   const LoggerTraceContext&                        context) {
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
        values.push_back(snapshotTensor(tensor, devices, context, name, static_cast<int64_t>(i)));
    }
    payload.insert(name, std::move(values));
}
c10::impl::GenericDict snapshotPayload(const GptModelInputs&     inputs,
                                       ModelInputsModelRole      role,
                                       int64_t                   model_id,
                                       int64_t                   sequence,
                                       int64_t                   dropped_before,
                                       std::vector<c10::Device>& devices,
                                       const LoggerTraceContext& context) {
    ExecutionStageScope    trace("model_inputs.snapshot", recordTraceDetails(context));
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
    payload.insert("kv_cache_group_tags", inputs.kv_cache_group_tags);
    c10::impl::GenericDict float8_dtypes(c10::StringType::get(), c10::StringType::get());
#define ADD_TENSOR(field) addTensor(payload, #field, inputs.field, devices, float8_dtypes, context);
    MODEL_INPUT_TENSORS(ADD_TENSOR)
#undef ADD_TENSOR
    addTensorList(payload, "multimodal_features", inputs.multimodal_features, devices, float8_dtypes, context);
    addTensorList(payload, "mm_extra_input", inputs.mm_extra_input, devices, float8_dtypes, context);
    addTensorList(payload, "input_embeddings", inputs.input_embeddings, devices, float8_dtypes, context);
    payload.insert("float8_dtypes", std::move(float8_dtypes));
    payload.insert("kv_block_stride_bytes", static_cast<int64_t>(inputs.kv_block_stride_bytes));
    payload.insert("kv_scale_stride_bytes", static_cast<int64_t>(inputs.kv_scale_stride_bytes));
    payload.insert("seq_size_per_block", static_cast<int64_t>(inputs.seq_size_per_block));
    payload.insert("kernel_seq_size_per_block", static_cast<int64_t>(inputs.kernel_seq_size_per_block));
    payload.insert("pd_separation", inputs.pd_separation);
    payload.insert("decode_entrance", inputs.decode_entrance);
    payload.insert("use_opaque_kv_cache_store", inputs.use_opaque_kv_cache_store);
    payload.insert("need_all_logits", inputs.need_all_logits);
    payload.insert("need_all_hidden_states", inputs.need_all_hidden_states);
    payload.insert("need_moe_gating", inputs.need_moe_gating);
    payload.insert("warmup", inputs.warmup);
    payload.insert("skip_run", inputs.skip_run);
    payload.insert("is_fake_stream", inputs.is_fake_stream);
    payload.insert("is_target_verify", inputs.is_target_verify);
    return payload;
}
std::vector<std::shared_ptr<c10::Event>> readyEvents(const std::vector<c10::Device>& devices,
                                                     LoggerTraceContext&             context) {
    ExecutionStageScope trace("model_inputs.producer.ready_events", recordTraceDetails(context));
    struct ProducerEventMetadata {
        size_t            event_index;
        const c10::Event* event;
        c10::DeviceType   device_type;
        c10::DeviceIndex  device_index;
        int64_t           stream_id;
    };
    constexpr size_t                                    max_trace_events = 8;
    std::array<ProducerEventMetadata, max_trace_events> producer_events;
    size_t                                              recorded_events = 0;
    std::vector<std::shared_ptr<c10::Event>> events;
    for (const auto& device : devices) {
        c10::impl::VirtualGuardImpl guard(device.type());
        auto                        event = std::make_shared<c10::Event>(device.type());
        const auto                  stream = [&] {
            ExecutionStageScope get_stream("model_inputs.producer.get_stream",
                                           eventTraceDetails(context, event.get(), events.size(), &device));
            // Preserve the single original getStream call; stream.id() is CPU metadata.
            auto stream = guard.getStream(device);
            get_stream.annotate(eventTraceDetails(context, event.get(), events.size(), &device, &stream));
            return stream;
        }();
        if (trace.id() && recorded_events < producer_events.size()) {
            try {
                producer_events[recorded_events] = {
                    events.size(), event.get(), device.type(), device.index(), stream.id()};
                ++recorded_events;
            } catch (...) {
                // A diagnostic metadata failure must not change event recording.
            }
        }
        {
            ExecutionStageScope record("model_inputs.producer.event_record",
                                       eventTraceDetails(context, event.get(), events.size(), &device, &stream));
            event->record(stream);
            record.finish("record_returned");
        }
        events.push_back(std::move(event));
    }
    context.producer_events_details = loggerTraceDetails([&](std::ostream& out) {
        out << " producer_event_count=" << events.size() << " producer_events_shown=" << recorded_events
            << " producer_events_truncated=" << (recorded_events < events.size() ? "true" : "false")
            << " producer_events=[";
        for (size_t index = 0; index < recorded_events; ++index) {
            const auto& metadata = producer_events[index];
            out << (index ? "," : "") << "{index=" << metadata.event_index
                << " event_object=" << static_cast<const void*>(metadata.event)
                << " device=" << c10::Device(metadata.device_type, metadata.device_index)
                << " producer_stream_id=" << metadata.stream_id << '}';
        }
        out << ']';
    });
    trace.annotate(recordTraceDetails(context));
    return events;
}
torch::Tensor tensorForDump(const torch::Tensor&      tensor,
                            const LoggerTraceContext& context,
                            const c10::IValue&        key,
                            int64_t                   index = -1) {
    ExecutionStageScope trace("model_inputs.dump.tensor", tensorTraceDetails(context, "", index, tensor, &key));
    auto output = tensor.detach();
    if (c10::isFloat8Type(output.scalar_type())) {
        output = output.view(torch::kChar);
    }
    auto cpu_output = [&] {
        if (output.device().is_cpu()) {
            return output;
        }
        ExecutionStageScope copy(
            "model_inputs.dump.cpu",
            tensorTraceDetails(context, "", index, output, &key, "copy_stream=implicit_unobserved"));
        auto result = output.cpu();
        copy.finish("cpu_copy_returned");
        return result;
    }();
    ExecutionStageScope contiguous("model_inputs.dump.contiguous",
                                   tensorTraceDetails(context, "", index, cpu_output, &key));
    return cpu_output.contiguous();
}
c10::IValue valueForDump(const c10::IValue& value, const LoggerTraceContext& context, const c10::IValue& key) {
    if (value.isTensor()) {
        return tensorForDump(value.toTensor(), context, key);
    }
    if (value.isTensorList()) {
        c10::List<torch::Tensor> tensors;
        int64_t                  index = 0;
        for (const auto& tensor : value.toTensorList()) {
            tensors.push_back(tensorForDump(tensor, context, key, index++));
        }
        return tensors;
    }
    return value;
}
void appendPayloadTraceIdentity(std::ostream& out, const c10::impl::GenericDict& payload) {
    for (const auto& item : payload) {
        const auto& key = item.key().toStringRef();
        if (key == "record_sequence" || key == "model_id" || key == "next_attempt_sequence") {
            out << ' ' << key << '=' << item.value().toInt();
        } else if (key == "model_role" || key == "record_type") {
            out << ' ' << key << '=' << item.value().toStringRef();
        }
    }
}
class DumpWriter {
public:
    DumpWriter(int64_t rank_id, int backup_count): rank_id_(rank_id), backup_count_(std::max(backup_count, 1)) {
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
        ExecutionStageScope trace("model_inputs.writer.write", loggerTraceDetails([&](std::ostream& out) {
                                      out << "rank=" << rank_id_ << " estimated_bytes=" << bytes;
                                      appendPayloadTraceIdentity(out, payload);
                                  }));
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
        const auto             details = loggerTraceDetails([&](std::ostream& out) {
            out << "rank=" << rank_id_ << " records=" << records_.size() << " estimated_bytes=" << pending_bytes_
                << " first_record={";
            appendPayloadTraceIdentity(out, records_.front());
            out << " } last_record={";
            appendPayloadTraceIdentity(out, records_.back());
            out << " }";
        });
        ExecutionStageScope    flush_trace("model_inputs.writer.flush", details);
        c10::List<c10::IValue> record_list(c10::AnyType::get());
        for (const auto& record : records_) {
            record_list.push_back(record);
        }
        c10::impl::GenericDict chunk(c10::StringType::get(), c10::AnyType::get());
        chunk.insert("schema_version", kSchemaVersion);
        chunk.insert("record_type", "model_inputs_chunk");
        chunk.insert("records", std::move(record_list));
        const auto bytes = [&] {
            ExecutionStageScope serialize("model_inputs.writer.serialize", details);
            return torch::pickle_save(c10::IValue(chunk));
        }();
        std::ostringstream id;
        id << std::setw(20) << std::setfill('0') << g_file_sequence.fetch_add(1, std::memory_order_relaxed);
        const auto final_name = prefix_ + id.str() + ".pt";
        const auto temp_name  = "." + final_name + ".tmp." + std::to_string(::getpid());
        ExecutionStageScope write_trace("model_inputs.writer.file_write", loggerTraceDetails([&](std::ostream& out) {
                                            out << details << " serialized_bytes=" << bytes.size()
                                                << " file=" << final_name;
                                        }));
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
            write_trace.finish("write_failed");
            flush_trace.finish("write_failed");
            return false;
        }
        write_trace.finish("write_returned");
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

    int64_t                             rank_id_;
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
    LoggerTraceContext                       trace_context;
};
}  // namespace
class ModelInputsLogger::Worker {
public:
    Worker(int64_t rank_id, int backup_count):
        rank_id_(rank_id), writer_(rank_id, backup_count), disabled_(!writer_.valid()) {
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
        LoggerTraceContext  context{rank_id_, model_id, sequence, role, loggerTraceDetails([&](std::ostream& out) {
                                       // trace_ids are already CPU strings. Never read request_id's tensor for tracing.
                                       const auto shown = std::min<size_t>(inputs.trace_ids.size(), 8);
                                       out << " trace_id_count=" << inputs.trace_ids.size()
                                           << " trace_ids_shown=" << shown << " trace_ids=[";
                                       bool length_truncated = false;
                                       for (size_t i = 0; i < shown; ++i) {
                                           length_truncated |= inputs.trace_ids[i].size() > 256;
                                           out << (i ? "," : "") << std::quoted(inputs.trace_ids[i].substr(0, 256));
                                       }
                                       out << "] trace_ids_truncated="
                                           << (shown < inputs.trace_ids.size() || length_truncated ? "true" : "false");
                                   })};
        ExecutionStageScope trace("model_inputs.enqueue", recordTraceDetails(context));
        context.enqueue_id    = trace.id();
        context.invocation_id = trace.parentId();
        trace.annotate(recordTraceDetails(context));
        const auto bytes = estimateBytes(inputs);
        if (disabled_ || bytes > kChunkMaxBytes || !reserve(bytes)) {
            drop("record exceeds limits or writer is unavailable");
            trace.finish("dropped");
            return;
        }
        const auto dropped_before = dropped_since_last_.exchange(0, std::memory_order_relaxed);
        try {
            std::vector<c10::Device> devices;
            auto payload = snapshotPayload(inputs, role, model_id, sequence, dropped_before, devices, context);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push_back(PendingRecord{
                    std::move(payload), readyEvents(devices, context), dropped_before, bytes, std::move(context)});
            }
            ready_cv_.notify_one();
        } catch (const std::exception& e) {
            dropped_since_last_.fetch_add(dropped_before, std::memory_order_relaxed);
            release(bytes);
            drop(e.what());
            trace.finish("exception");
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
        ExecutionStageScope trace(
            "model_inputs.worker.record", recordTraceDetails(record.trace_context), 0, record.trace_context.enqueue_id);
        try {
            if (disabled_ || !waitForEvents(record.events, record.trace_context)) {
                dropped_since_last_.fetch_add(record.dropped_before, std::memory_order_relaxed);
                drop("CUDA snapshot did not become ready");
                trace.finish("events_not_ready");
                return;
            }
            for (const auto& item : record.payload) {
                item.setValue(valueForDump(item.value(), record.trace_context, item.key()));
            }
            if (!writer_.write(std::move(record.payload), record.bytes)) {
                disabled_ = true;
                trace.finish("write_failed");
            }
        } catch (const std::exception& e) {
            dropped_since_last_.fetch_add(record.dropped_before, std::memory_order_relaxed);
            drop(e.what());
            trace.finish("exception");
        }
    }
    bool waitForEvents(const std::vector<std::shared_ptr<c10::Event>>& events, const LoggerTraceContext& context) {
        ExecutionStageScope wait("model_inputs.worker.event_wait", recordTraceDetails(context));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        size_t              index    = 0;
        for (const auto& event : events) {
            const auto          event_details = eventTraceDetails(context, event.get(), index);
            ExecutionStageScope event_wait("model_inputs.worker.event", event_details);
            while (true) {
                const bool ready = [&] {
                    ExecutionStageScope query("model_inputs.worker.event_query", event_details);
                    const bool          ready = event->query();
                    query.finish(ready ? "completed" : "not_ready");
                    return ready;
                }();
                if (ready) {
                    event_wait.finish("completed");
                    break;
                }
                const auto wait_deadline = stopping_.load(std::memory_order_acquire) ? shutdown_deadline_ : deadline;
                if (std::chrono::steady_clock::now() >= wait_deadline) {
                    event_wait.finish("timeout");
                    wait.finish("timeout");
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            ++index;
        }
        wait.finish("all_events_ready");
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

    int64_t                               rank_id_;
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
