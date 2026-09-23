#include "rtp_llm/cpp/utils/CudacoreDiagnostics.h"
#include "rtp_llm/cpp/utils/CudacoreFlightRecorder.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#include <utility>

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

// ---------------------------------------------------------------------------
// Driver ABI mirrors
//
// The module deliberately does not include <cuda.h>: it must build on CPU and
// ROCm configurations and stays usable when the CUDA toolkit disagrees with the
// installed driver. The coredump attribute ids and CUresult codes below are the
// stable values from cuda.h (CUDA 13.x).
// ---------------------------------------------------------------------------

constexpr int kCuSuccess                 = 0;
constexpr int kCuErrorInvalidValue       = 1;
constexpr int kCuErrorNotInitialized     = 3;
constexpr int kCuErrorDeinitialized      = 4;
constexpr int kCuErrorInvalidContext     = 201;
constexpr int kCuErrorContextIsDestroyed = 709;
constexpr int kCuErrorNotSupported       = 801;

constexpr int kAttribEnableOnException = 1;
constexpr int kAttribTriggerHost       = 2;
constexpr int kAttribLightweight       = 3;
constexpr int kAttribEnableUserTrigger = 4;
constexpr int kAttribFile              = 5;
constexpr int kAttribPipe              = 6;
constexpr int kAttribGenerationFlags   = 7;

// Runtime cudaError_t / driver CUresult values of device-fault errors that must
// enter the fatal collection path. Numeric values are identical in both enums.
constexpr int kFatalIllegalAddress   = 700;
constexpr int kFatalLaunchTimeout    = 702;
constexpr int kFatalAssert           = 710;
constexpr int kFatalEccUncorrectable = 214;
constexpr int kFatalIllegalInsn      = 715;
constexpr int kFatalMisalignedAddr   = 716;
constexpr int kFatalBadAddressSpace  = 717;
constexpr int kFatalInvalidPc        = 718;
constexpr int kFatalLaunchFailed     = 719;

constexpr const char* kSchemaVersion = "rtp_llm.cudacore_manifest.v1";
constexpr const char* kLeaseSchemaVersion = "rtp_llm.cudacore_lease.v1";
// Minimum spacing between evidence (manifest) rewrites while a window is open.
constexpr int64_t     kEvidenceWriteIntervalMs = 500;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

int64_t nowWallMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t nowMonoMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

const std::string& hostnameCached() {
    static const std::string hostname = [] {
        char buffer[256] = {};
        if (::gethostname(buffer, sizeof(buffer) - 1) != 0) {
            return std::string("unknown");
        }
        return std::string(buffer);
    }();
    return hostname;
}

std::string pidNamespaceId() {
    char    buffer[128] = {};
    ssize_t size        = ::readlink("/proc/self/ns/pid", buffer, sizeof(buffer) - 1);
    if (size <= 0) {
        return {};
    }
    return std::string(buffer, static_cast<size_t>(size));
}

std::string threadIdString() {
    return "tid=" + std::to_string(static_cast<long>(::syscall(SYS_gettid)));
}

std::string processStartId() {
    // pid alone is reused; pid + process start ticks is stable inside one process
    // and distinguishes a reused pid from the original worker.
    static const std::string start_id = [] {
        std::string value;
        FILE*       file = ::fopen("/proc/self/stat", "r");
        if (file != nullptr) {
            char line[4096] = {};
            if (::fgets(line, sizeof(line), file) != nullptr) {
                // comm may contain spaces and parentheses; fields after the last ')' start at field 3.
                const char* cursor = ::strrchr(line, ')');
                if (cursor != nullptr) {
                    ++cursor;
                    int         index = 3;  // field number of the first token after comm
                    const char* token = ::strtok(const_cast<char*>(cursor), " ");
                    while (token != nullptr && index < 22) {
                        token = ::strtok(nullptr, " ");
                        ++index;
                    }
                    if (token != nullptr) {
                        value = token;
                    }
                }
            }
            ::fclose(file);
        }
        if (value.empty()) {
            value = "0";
        }
        return std::to_string(static_cast<long>(::getpid())) + "-" + value;
    }();
    return start_id;
}

std::string startupCwd() {
    static const std::string cwd = [] {
        char buffer[4096] = {};
        if (::getcwd(buffer, sizeof(buffer) - 1) == nullptr) {
            return std::string(".");
        }
        return std::string(buffer);
    }();
    return cwd;
}

std::string truncateText(const std::string& text, size_t limit = CudacoreDiagConstants::kMaxTextBytes) {
    if (text.size() <= limit) {
        return text;
    }
    return text.substr(0, limit) + "...<truncated>";
}

std::string sanitizeComponent(const std::string& value) {
    std::string sanitized;
    sanitized.reserve(std::min<size_t>(value.size(), 96));
    for (char character : value) {
        const bool safe = (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z')
                          || (character >= '0' && character <= '9') || character == '.' || character == '_'
                          || character == '-';
        sanitized.push_back(safe ? character : '_');
        if (sanitized.size() >= 96) {
            break;
        }
    }
    return sanitized.empty() ? std::string("unknown") : sanitized;
}

std::string escapeJson(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() + 8);
    for (char character : text) {
        switch (character) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(character) < 0x20) {
                    char buffer[8] = {};
                    ::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned char>(character));
                    escaped += buffer;
                } else {
                    escaped.push_back(character);
                }
        }
    }
    return escaped;
}

std::string jsonString(const std::string& value) {
    return "\"" + escapeJson(truncateText(value)) + "\"";
}

bool ensureDirectory(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    std::string partial;
    size_t      start = path.front() == '/' ? 1 : 0;
    for (size_t index = start; index <= path.size(); ++index) {
        if (index == path.size() || path[index] == '/') {
            partial   = path.substr(0, index);
            if (partial.empty() || partial == "/") {
                continue;
            }
            if (::mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
    }
    struct stat info = {};
    return ::stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

std::string parentDirectory(const std::string& path) {
    const size_t slash = path.rfind('/');
    if (slash == std::string::npos) {
        return {};
    }
    if (slash == 0) {
        return "/";
    }
    return path.substr(0, slash);
}

// Write file content with an exclusive create; never overwrites existing files.
bool writeExclusive(const std::string& path, const std::string& content, bool* existed = nullptr) {
    int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) {
        if (existed != nullptr && errno == EEXIST) {
            *existed = true;
        }
        return false;
    }
    size_t written = 0;
    while (written < content.size()) {
        ssize_t chunk = ::write(descriptor, content.data() + written, content.size() - written);
        if (chunk <= 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        written += static_cast<size_t>(chunk);
    }
    ::close(descriptor);
    return written == content.size();
}

// Atomic replace through a temporary file in the same directory.
bool writeAtomicReplace(const std::string& path, const std::string& content) {
    const std::string temporary = path + ".tmp." + std::to_string(static_cast<long>(::getpid()));
    int               descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (descriptor < 0) {
        return false;
    }
    size_t written = 0;
    while (written < content.size()) {
        ssize_t chunk = ::write(descriptor, content.data() + written, content.size() - written);
        if (chunk <= 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        written += static_cast<size_t>(chunk);
    }
    const bool complete = written == content.size();
    ::close(descriptor);
    if (!complete) {
        (void)::unlink(temporary.c_str());
        return false;
    }
    if (::rename(temporary.c_str(), path.c_str()) != 0) {
        (void)::unlink(temporary.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Driver symbol resolution (no link-time CUDA dependency)
// ---------------------------------------------------------------------------

using AttributeGetFn = int (*)(int attrib, void* value, size_t* size);
using VersionGetFn   = int (*)(int* version);
using DeviceGetFn    = int (*)(int* device, int ordinal);
using UuidFn         = int (*)(void* uuid, int device);

struct DriverSymbols {
    void*         handle{nullptr};
    AttributeGetFn get_attribute_global{nullptr};
    AttributeGetFn get_attribute{nullptr};
    VersionGetFn   driver_get_version{nullptr};
    DeviceGetFn    device_get{nullptr};
    UuidFn         device_get_uuid{nullptr};
};

void* openDriverLibrary() {
    for (const char* name : {"libcuda.so.1", "libcuda.so"}) {
        void* handle = ::dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            return handle;
        }
    }
    return nullptr;
}

const DriverSymbols& driverSymbols() {
    static const DriverSymbols symbols = [] {
        DriverSymbols resolved;
        resolved.handle = openDriverLibrary();
        if (resolved.handle == nullptr) {
            return resolved;
        }
        resolved.get_attribute_global =
            reinterpret_cast<AttributeGetFn>(::dlsym(resolved.handle, "cuCoredumpGetAttributeGlobal"));
        resolved.get_attribute = reinterpret_cast<AttributeGetFn>(::dlsym(resolved.handle, "cuCoredumpGetAttribute"));
        resolved.driver_get_version = reinterpret_cast<VersionGetFn>(::dlsym(resolved.handle, "cuDriverGetVersion"));
        resolved.device_get        = reinterpret_cast<DeviceGetFn>(::dlsym(resolved.handle, "cuDeviceGet"));
        resolved.device_get_uuid   = reinterpret_cast<UuidFn>(::dlsym(resolved.handle, "cuDeviceGetUuid"));
        return resolved;
    }();
    return symbols;
}

int runtimeApiVersion() {
    using RuntimeVersionFn = int (*)(int* version);
    static const RuntimeVersionFn runtime_get_version = [] {
        for (const char* name : {"libcudart.so.13", "libcudart.so.12", "libcudart.so"}) {
            void* handle = ::dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (handle != nullptr) {
                auto* symbol = reinterpret_cast<RuntimeVersionFn>(::dlsym(handle, "cudaRuntimeGetVersion"));
                if (symbol != nullptr) {
                    return symbol;
                }
            }
        }
        return static_cast<RuntimeVersionFn>(nullptr);
    }();
    int version = 0;
    if (runtime_get_version != nullptr && runtime_get_version(&version) == 0) {
        return version;
    }
    return 0;
}

// Test seam: when installed, it replaces the driver symbols entirely.
cudacore_test::AttributeApi& testAttributeApi() {
    static cudacore_test::AttributeApi api;
    return api;
}

bool usingTestAttributeApi() {
    return testAttributeApi().installed;
}

// ---------------------------------------------------------------------------
// Attribute queries
// ---------------------------------------------------------------------------

CudacoreAttributeStatus statusFromApiCode(int code) {
    if (code == kCuSuccess) {
        return CudacoreAttributeStatus::Ok;
    }
    if (code == kCuErrorNotSupported || code == kCuErrorInvalidValue) {
        // INVALID_VALUE is reported for attributes the driver does not know.
        return CudacoreAttributeStatus::Unsupported;
    }
    if (code == kCuErrorInvalidContext || code == kCuErrorContextIsDestroyed || code == kCuErrorNotInitialized
        || code == kCuErrorDeinitialized) {
        return CudacoreAttributeStatus::Unavailable;
    }
    return CudacoreAttributeStatus::Error;
}

CudacoreBoolAttribute queryBool(bool global_scope, int attrib) {
    CudacoreBoolAttribute attribute;
    const AttributeGetFn fn = global_scope ? driverSymbols().get_attribute_global : driverSymbols().get_attribute;
    if (usingTestAttributeApi()) {
        const AttributeGetFn test_fn = global_scope ? testAttributeApi().get_global : testAttributeApi().get;
        if (test_fn == nullptr) {
            attribute.status = CudacoreAttributeStatus::Unsupported;
            return attribute;
        }
        bool   value = false;
        size_t size  = sizeof(value);
        const int code = test_fn(attrib, &value, &size);
        attribute.api_code = code;
        attribute.status   = statusFromApiCode(code);
        attribute.value    = value;
        return attribute;
    }
    if (fn == nullptr) {
        attribute.status = CudacoreAttributeStatus::Unsupported;
        return attribute;
    }
    bool   value = false;
    size_t size  = sizeof(value);
    const int code = fn(attrib, &value, &size);
    attribute.api_code = code;
    attribute.status   = statusFromApiCode(code);
    attribute.value    = value;
    return attribute;
}

CudacoreStringAttribute queryString(bool global_scope, int attrib) {
    CudacoreStringAttribute attribute;
    const AttributeGetFn    fn = global_scope ? driverSymbols().get_attribute_global : driverSymbols().get_attribute;
    const AttributeGetFn    test_fn = global_scope ? testAttributeApi().get_global : testAttributeApi().get;
    const AttributeGetFn    active  = usingTestAttributeApi() ? test_fn : fn;
    if (active == nullptr) {
        attribute.status = CudacoreAttributeStatus::Unsupported;
        return attribute;
    }

    // Ask for the required size first; drivers that do not answer the probe fall
    // back to one bounded read.
    size_t required = 0;
    int    code     = active(attrib, nullptr, &required);
    attribute.api_code = code;
    if (code != kCuSuccess || required == 0) {
        required = 0;
    }
    const size_t capacity = std::min<size_t>(std::max<size_t>(required, 1), CudacoreDiagConstants::kMaxAttributeBytes);
    std::string  buffer(capacity, '\0');
    size_t       size    = capacity;
    const int    read_code = active(attrib, buffer.data(), &size);
    attribute.api_code     = read_code;
    attribute.status       = statusFromApiCode(read_code);
    if (attribute.status != CudacoreAttributeStatus::Ok) {
        return attribute;
    }
    const size_t length = std::min<size_t>(size, capacity);
    attribute.value.assign(buffer.data(), ::strnlen(buffer.data(), length));
    attribute.truncated = required > capacity;
    return attribute;
}

CudacoreFlagsAttribute queryFlags(bool global_scope, int attrib) {
    CudacoreFlagsAttribute attribute;
    const AttributeGetFn   fn =
        global_scope ? driverSymbols().get_attribute_global : driverSymbols().get_attribute;
    const AttributeGetFn test_fn = global_scope ? testAttributeApi().get_global : testAttributeApi().get;
    const AttributeGetFn active  = usingTestAttributeApi() ? test_fn : fn;
    if (active == nullptr) {
        attribute.status = CudacoreAttributeStatus::Unsupported;
        return attribute;
    }
    uint64_t value = 0;
    size_t   size  = sizeof(value);
    const int code = active(attrib, &value, &size);
    attribute.api_code = code;
    attribute.status   = statusFromApiCode(code);
    attribute.value    = value;
    return attribute;
}

const char* statusName(CudacoreAttributeStatus status) {
    switch (status) {
        case CudacoreAttributeStatus::NotQueried:
            return "not_queried";
        case CudacoreAttributeStatus::Ok:
            return "ok";
        case CudacoreAttributeStatus::Unsupported:
            return "unsupported";
        case CudacoreAttributeStatus::Unavailable:
            return "unavailable";
        case CudacoreAttributeStatus::Error:
            return "error";
        case CudacoreAttributeStatus::Skipped:
            return "skipped";
    }
    return "unknown";
}

std::string boolAttributeJson(const char* name, const CudacoreBoolAttribute& attribute) {
    std::string json = std::string("\"") + name + "\":{\"status\":" + jsonString(statusName(attribute.status))
                       + ",\"api_code\":" + std::to_string(attribute.api_code);
    if (attribute.status == CudacoreAttributeStatus::Ok) {
        json += std::string(",\"value\":") + (attribute.value ? "true" : "false");
    }
    json += "}";
    return json;
}

std::string stringAttributeJson(const char* name, const CudacoreStringAttribute& attribute) {
    std::string json = std::string("\"") + name + "\":{\"status\":" + jsonString(statusName(attribute.status))
                       + ",\"api_code\":" + std::to_string(attribute.api_code);
    if (attribute.status == CudacoreAttributeStatus::Ok) {
        json += ",\"value\":" + jsonString(attribute.value)
                + ",\"truncated\":" + (attribute.truncated ? "true" : "false");
    }
    json += "}";
    return json;
}

std::string flagsAttributeJson(const char* name, const CudacoreFlagsAttribute& attribute) {
    std::string json = std::string("\"") + name + "\":{\"status\":" + jsonString(statusName(attribute.status))
                       + ",\"api_code\":" + std::to_string(attribute.api_code);
    if (attribute.status == CudacoreAttributeStatus::Ok) {
        json += ",\"value\":" + std::to_string(attribute.value);
    }
    json += "}";
    return json;
}

std::string attributesJson(const CudacoreAttributeSnapshot& snapshot) {
    std::string json = "{";
    json += "\"symbols_source\":" + jsonString(snapshot.symbols_source);
    json += ",\"driver_version\":" + std::to_string(snapshot.driver_version);
    json += ",\"runtime_version\":" + std::to_string(snapshot.runtime_version);
    json += ",\"global\":{";
    json += boolAttributeJson("enable_on_exception", snapshot.global.enable_on_exception);
    json += "," + boolAttributeJson("enable_user_trigger", snapshot.global.enable_user_trigger);
    json += "," + boolAttributeJson("trigger_host", snapshot.global.trigger_host);
    json += "," + boolAttributeJson("lightweight", snapshot.global.lightweight);
    json += "," + stringAttributeJson("file", snapshot.global.file);
    json += "," + stringAttributeJson("pipe", snapshot.global.pipe);
    json += "," + flagsAttributeJson("generation_flags", snapshot.global.generation_flags);
    json += "},\"context\":{";
    json += std::string("\"valid\":") + (snapshot.context.context_valid ? "true" : "false");
    json += "," + boolAttributeJson("enable_on_exception", snapshot.context.enable_on_exception);
    json += "," + stringAttributeJson("file", snapshot.context.file);
    json += "," + flagsAttributeJson("generation_flags", snapshot.context.generation_flags);
    json += "}}";
    return json;
}

// Compact aggregate: which attributes could not be read, without inventing values.
std::string attributeQueryStatusJson(const CudacoreAttributeSnapshot& snapshot) {
    std::string not_ok;
    auto        note = [&not_ok](const char* name, CudacoreAttributeStatus status) {
        if (status == CudacoreAttributeStatus::Ok || status == CudacoreAttributeStatus::Skipped) {
            return;
        }
        if (!not_ok.empty()) {
            not_ok += ",";
        }
        not_ok += jsonString(std::string(name) + "=" + statusName(status));
    };
    note("global.enable_on_exception", snapshot.global.enable_on_exception.status);
    note("global.enable_user_trigger", snapshot.global.enable_user_trigger.status);
    note("global.trigger_host", snapshot.global.trigger_host.status);
    note("global.lightweight", snapshot.global.lightweight.status);
    note("global.file", snapshot.global.file.status);
    note("global.pipe", snapshot.global.pipe.status);
    note("global.generation_flags", snapshot.global.generation_flags.status);
    note("context.enable_on_exception", snapshot.context.enable_on_exception.status);
    note("context.file", snapshot.context.file.status);
    note("context.generation_flags", snapshot.context.generation_flags.status);
    return "{\"symbols_source\":" + jsonString(snapshot.symbols_source) + ",\"not_ok\":[" + not_ok + "]}";
}

// ---------------------------------------------------------------------------
// Diagnostics directory and process identity
// ---------------------------------------------------------------------------

bool isPlainFileTemplate(const std::string& template_path) {
    if (template_path.empty()) {
        return false;
    }
    // A '|' prefix or suffix makes the template a shell pipe / existing FIFO
    // form: those must not be interpreted as a directory.
    if (template_path.find('|') != std::string::npos) {
        return false;
    }
    const std::string directory = parentDirectory(template_path);
    const std::string name      = directory.empty() ? template_path
                                                    : template_path.substr(directory.size() + 1);
    const size_t      percent   = name.find('%');
    const std::string static_prefix = percent == std::string::npos ? name : name.substr(0, percent);
    if (static_prefix.empty()) {
        return false;
    }
    std::string probe = directory.empty() ? static_prefix : directory + "/" + static_prefix;
    struct stat info  = {};
    if (::stat(probe.c_str(), &info) == 0 && !S_ISREG(info.st_mode)) {
        return false;  // an existing FIFO/pipe endpoint
    }
    return true;
}

std::string effectiveDumpFileTemplate() {
    const char* env_value = ::getenv("CUDA_COREDUMP_FILE");
    if (env_value != nullptr && env_value[0] != '\0') {
        return env_value;
    }
    return {};
}

struct DiagnosticsConfig {
    std::string dir;
    std::string reason;
};

struct SharedState {
    std::mutex                mutex;
    std::string               dir_override;
    std::string               resolved_dir;
    bool                      has_snapshot{false};
    CudacoreAttributeSnapshot snapshot;

    CudacoreProcessIdentity identity;

    // Lock-free mirrors for the per-step / per-transfer gates: the steady-state
    // path must not pay for a mutex on every call.
    std::atomic<bool>        incident_flag{false};
    std::atomic<bool>        terminal_flag{false};
    std::atomic<int64_t>     deadline_mono_atomic{0};

    bool                   has_record{false};
    FatalCudaErrorRecord   record;
    int64_t                deadline_mono_ms{0};
    int64_t                deadline_wall_ms{0};
    int64_t                incident_wall_ms{0};
    int64_t                incident_mono_ms{0};
    std::string            incident_id;
    bool                   collector_started{false};
    bool                   collector_stop{false};
    bool                   inline_claimed{false};
    uint64_t               incident_generation{0};
    int64_t                collection_window_ms{0};  // 0 => fixed default (test seam)
    bool                   terminal{false};
    CudacoreCollectionOutcome outcome;
    std::condition_variable   collector_wakeup;
    std::condition_variable   finished;
};

SharedState& sharedState() {
    // Leaked on purpose: a fault may abort the process while other threads use it.
    static SharedState* state = new SharedState();
    return *state;
}

std::atomic<std::terminate_handler> previous_terminate_handler{nullptr};

void cudacoreTerminateHandler() noexcept {
    // The watchdog and destructor paths can terminate outside the engine catch.
    // This handler only reads atomics and sleeps; the collector does the I/O.
    SharedState& state = sharedState();
    if (state.incident_flag.load(std::memory_order_acquire)) {
        static constexpr char waiting[] = "[CudacoreDiag] terminate guard waiting for collection\n";
        (void)::write(STDERR_FILENO, waiting, sizeof(waiting) - 1);
        const int64_t deadline = state.deadline_mono_atomic.load(std::memory_order_acquire);
        while (!state.terminal_flag.load(std::memory_order_acquire) && nowMonoMs() < deadline) {
            const struct timespec pause = {0, 20 * 1000 * 1000};
            (void)::nanosleep(&pause, nullptr);
        }
        static constexpr char done[] = "[CudacoreDiag] terminate guard collection window closed\n";
        (void)::write(STDERR_FILENO, done, sizeof(done) - 1);
    }
    const auto previous = previous_terminate_handler.load(std::memory_order_acquire);
    if (previous != nullptr && previous != &cudacoreTerminateHandler) {
        previous();
    }
    std::abort();
}

// Caller holds state.mutex (or is the only writer) when reading the override.
int64_t effectiveWindowMs(const SharedState& state) {
    return state.collection_window_ms > 0 ? state.collection_window_ms : CudacoreDiagConstants::kCollectionDeadlineMs;
}

DiagnosticsConfig resolveDiagnosticsConfig() {
    SharedState&         state = sharedState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.dir_override.empty()) {
        return {state.dir_override, "test_override"};
    }
    if (!state.resolved_dir.empty()) {
        return {state.resolved_dir, "cached"};
    }

    std::string template_path = effectiveDumpFileTemplate();
    if (template_path.empty() && state.has_snapshot
        && state.snapshot.global.file.status == CudacoreAttributeStatus::Ok) {
        template_path = state.snapshot.global.file.value;
    }
    std::string dir;
    std::string reason;
    if (!template_path.empty()) {
        if (isPlainFileTemplate(template_path)) {
            const std::string parent = parentDirectory(template_path);
            dir                      = parent.empty() ? startupCwd() : parent;
            reason                   = "coredump_file_parent";
        } else {
            reason = "coredump_target_is_pipe_or_fifo";
        }
    } else {
        reason = "coredump_file_not_configured";
    }
    if (dir.empty()) {
        dir = startupCwd() + "/" + CudacoreDiagConstants::kDiagnosticsDirName;
    }
    state.resolved_dir = dir;
    return {dir, reason};
}

std::string resolveDiagnosticsDir() {
    return resolveDiagnosticsConfig().dir;
}

std::string manifestBaseName(const SharedState& state) {
    const std::string host = sanitizeComponent(hostnameCached());
    const std::string pid  = std::to_string(static_cast<long>(::getpid()));
    const std::string start = sanitizeComponent(state.identity.worker_start_id.empty()
                                                    ? processStartId()
                                                    : state.identity.worker_start_id);
    return host + "." + pid + "." + start;
}

// ---------------------------------------------------------------------------
// Manifest content
// ---------------------------------------------------------------------------

const char* domainName(FatalCudaErrorDomain domain) {
    switch (domain) {
        case FatalCudaErrorDomain::CudaRuntime:
            return "cuda_runtime";
        case FatalCudaErrorDomain::CudaDriver:
            return "cuda_driver";
        case FatalCudaErrorDomain::Cublas:
            return "cublas";
        case FatalCudaErrorDomain::TorchException:
            return "torch_exception";
        case FatalCudaErrorDomain::Unknown:
            return "unknown";
    }
    return "unknown";
}

const char* siteName(FatalCudaErrorSite site) {
    switch (site) {
        case FatalCudaErrorSite::BatchedCopySubmit:
            return "batched_copy_submit";
        case FatalCudaErrorSite::BatchedCopyCompletion:
            return "batched_copy_completion";
        case FatalCudaErrorSite::StagedCopy:
            return "staged_copy";
        case FatalCudaErrorSite::GenericCheck:
            return "generic_check";
        case FatalCudaErrorSite::DeviceHostCopy:
            return "device_host_copy";
        case FatalCudaErrorSite::EngineStep:
            return "engine_step";
        case FatalCudaErrorSite::Unknown:
            return "unknown";
    }
    return "unknown";
}

const char* triggerName(CudacoreTriggerStatus status) {
    switch (status) {
        case CudacoreTriggerStatus::NotAttempted:
            return "NOT_ATTEMPTED";
        case CudacoreTriggerStatus::Sent:
            return "TRIGGER_SENT";
        case CudacoreTriggerStatus::AlreadyObservedProgress:
            return "ALREADY_OBSERVED_PROGRESS";
        case CudacoreTriggerStatus::AlreadyAttemptedByPeer:
            return "ALREADY_ATTEMPTED_BY_PEER";
        case CudacoreTriggerStatus::PipeNotConfigured:
            return "PIPE_NOT_CONFIGURED";
        case CudacoreTriggerStatus::PipeMissing:
            return "PIPE_MISSING";
        case CudacoreTriggerStatus::NoReader:
            return "NO_READER";
        case CudacoreTriggerStatus::PermissionDenied:
            return "PERMISSION_DENIED";
        case CudacoreTriggerStatus::BrokenPipe:
            return "EPIPE";
        case CudacoreTriggerStatus::PipeFull:
            return "EAGAIN_PIPE_FULL";
        case CudacoreTriggerStatus::IoTimeout:
            return "IO_TIMEOUT";
        case CudacoreTriggerStatus::WriteFailed:
            return "WRITE_FAILED";
    }
    return "UNKNOWN";
}

const char* terminalName(CudacoreTerminalState state) {
    switch (state) {
        case CudacoreTerminalState::NotStarted:
            return "NOT_STARTED";
        case CudacoreTerminalState::CollectionObservedDone:
            return "COLLECTION_OBSERVED_DONE";
        case CudacoreTerminalState::NoMechanismAvailable:
            return "NO_MECHANISM_AVAILABLE";
        case CudacoreTerminalState::DeadlineExceeded:
            return "DEADLINE_EXCEEDED";
    }
    return "UNKNOWN";
}

std::string firstErrorJson(const FatalCudaErrorRecord& record) {
    std::string json = "{";
    json += "\"domain\":" + jsonString(domainName(record.domain));
    json += ",\"code\":" + std::to_string(record.code);
    json += ",\"code_name\":" + jsonString(record.code_name);
    json += ",\"message\":" + jsonString(record.message);
    json += ",\"site\":" + jsonString(siteName(record.site));
    json += ",\"source_file\":" + jsonString(record.source_file);
    json += ",\"source_line\":" + std::to_string(record.source_line);
    json += ",\"time_wall_ms\":" + std::to_string(record.wall_ms);
    json += ",\"time_mono_ms\":" + std::to_string(record.mono_ms);
    json += ",\"thread\":" + jsonString(record.thread_id);
    json += ",\"device_index\":" + std::to_string(record.device_index);
    json += ",\"rank\":" + std::to_string(record.rank);
    json += ",\"stream\":" + jsonString(record.stream);
    json += ",\"low_confidence\":" + std::string(record.low_confidence ? "true" : "false");
    json += ",\"transfer\":{";
    json += std::string("\"known\":") + (record.has_group_set_id ? "true" : "false");
    json += ",\"group_set_id\":" + std::to_string(record.group_set_id);
    json += std::string(",\"host_known\":") + (record.has_host_span ? "true" : "false");
    json += ",\"host_base\":" + std::to_string(record.host_base);
    json += ",\"host_bytes\":" + std::to_string(record.host_bytes);
    json += ",\"direction\":" + jsonString(record.direction == nullptr ? "none" : record.direction);
    json += "}";
    json += ",\"tile_total\":" + std::to_string(record.tile_total);
    json += ",\"bytes_total\":" + std::to_string(record.bytes_total);
    json += ",\"tiles_truncated\":" + std::string(record.tiles_truncated ? "true" : "false");
    json += ",\"metadata_truncated\":" + std::string(record.metadata_truncated ? "true" : "false");
    json += ",\"tiles\":[";
    for (size_t index = 0; index < record.tiles.size(); ++index) {
        if (index != 0) {
            json += ",";
        }
        json += "{\"dst\":" + std::to_string(record.tiles[index].dst) + ",\"src\":"
                + std::to_string(record.tiles[index].src) + ",\"bytes\":"
                + std::to_string(record.tiles[index].bytes) + "}";
    }
    json += "]}";
    return json;
}

struct ManifestInputs {
    FatalCudaErrorRecord      record;
    bool                      has_snapshot{false};
    CudacoreAttributeSnapshot snapshot;
    CudacoreProcessIdentity   identity;
    std::string               incident_id;
    std::string               base_name;
    int64_t                   window_ms{0};
    int64_t                   deadline_epoch_ms{0};
};

std::string buildManifest(const ManifestInputs&            inputs,
                          const CudacoreCollectionOutcome& outcome,
                          const std::string&               phase) {
    const FatalCudaErrorRecord& record = inputs.record;
    std::string                 json   = "{";
    json += "\"schema_version\":" + jsonString(kSchemaVersion);
    json += ",\"phase\":" + jsonString(phase);
    json += ",\"incident_id\":" + jsonString(inputs.incident_id);
    json += ",\"worker_start_id\":" + jsonString(inputs.identity.worker_start_id);
    json += ",\"host\":" + jsonString(hostnameCached());
    json += ",\"pid\":" + std::to_string(static_cast<long>(::getpid()));
    json += ",\"pid_namespace\":" + jsonString(inputs.identity.pid_namespace);
    json += ",\"rank\":" + std::to_string(record.rank);
    json += ",\"gpu_uuid\":" + jsonString(inputs.identity.gpu_uuid);
    json += ",\"device_ordinal\":" + std::to_string(inputs.identity.device_ordinal);
    json += ",\"image_version\":" + jsonString(inputs.identity.image_version);
    json += ",\"driver_version\":" + std::to_string(inputs.identity.driver_version);
    json += ",\"runtime_version\":" + std::to_string(inputs.identity.runtime_version);
    if (inputs.has_snapshot) {
        json += ",\"startup_coredump_attributes\":" + attributesJson(inputs.snapshot);
        json += ",\"attribute_query_status\":" + attributeQueryStatusJson(inputs.snapshot);
    } else {
        json += ",\"startup_coredump_attributes\":null";
        json += ",\"attribute_query_status\":null";
    }
    json += ",\"first_error\":" + firstErrorJson(record);
    json += ",\"recent_cuda_submissions\":" + snapshotCudacoreFlightRecorderJson();
    json += ",\"window_ms\":" + std::to_string(inputs.window_ms);
    json += ",\"deadline_epoch_ms\":" + std::to_string(inputs.deadline_epoch_ms);
    json += ",\"lease_path\":" + jsonString(outcome.lease_path);
    json += ",\"trigger\":{";
    json += "\"status\":" + jsonString(triggerName(outcome.trigger));
    json += ",\"attempted\":" + std::string(outcome.trigger_attempted ? "true" : "false");
    json += ",\"sent\":" + std::string(outcome.trigger_sent ? "true" : "false");
    json += ",\"errno\":" + std::to_string(outcome.trigger_errno);
    json += ",\"dump_file_template\":" + jsonString(outcome.dump_file_template);
    json += "}";
    json += ",\"file_events\":{";
    json += "\"observed_file\":" + jsonString(outcome.observed_file);
    json += ",\"bytes\":" + std::to_string(outcome.observed_bytes);
    json += ",\"file_seen\":" + std::string(outcome.file_seen ? "true" : "false");
    json += ",\"size_stable\":" + std::string(outcome.file_size_stable ? "true" : "false");
    json += ",\"driver_progress_seen\":" + std::string(outcome.driver_progress_seen ? "true" : "false");
    json += ",\"driver_progress_source\":" + jsonString("stderr(captured by parent)");
    json += "}";
    json += ",\"collector_terminal_state\":" + jsonString(terminalName(outcome.terminal));
    json += ",\"waited_ms\":" + std::to_string(outcome.waited_ms);
    json += ",\"target_exit\":{\"observed\":false,\"note\":" +
            jsonString("worker exit is observed by the parent process, not from inside the worker") + "}";
    json += ",\"note\":" + jsonString(outcome.note);
    json += ",\"offline_validation_result\":" + jsonString("pending");
    json += "}";
    return json;
}

std::string writeManifest(const std::string&               dir,
                          SharedState&                     state,
                          const CudacoreCollectionOutcome& outcome,
                          const std::string&               phase) {
    if (dir.empty() || !ensureDirectory(dir)) {
        return {};
    }
    ManifestInputs inputs;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        inputs.record            = state.record;
        inputs.has_snapshot      = state.has_snapshot;
        inputs.snapshot          = state.snapshot;
        inputs.identity          = state.identity;
        inputs.incident_id       = state.incident_id;
        inputs.base_name         = manifestBaseName(state);
        inputs.window_ms         = effectiveWindowMs(state);
        inputs.deadline_epoch_ms = state.deadline_wall_ms;
    }
    const std::string base    = dir + "/cudacore_incident." + inputs.base_name;
    std::string       path    = base + ".json";
    const std::string content = buildManifest(inputs, outcome, phase);
    if (phase == "fault") {
        bool existed = false;
        if (!writeExclusive(path, content, &existed)) {
            if (!existed) {
                return {};
            }
            path = base + "." + std::to_string(nowMonoMs()) + ".json";
            if (!writeExclusive(path, content)) {
                return {};
            }
        }
        return path;
    }
    // Completion phase updates the file written at fault time; if a peer module
    // copy owns it, keep both by writing a suffixed copy.
    if (!writeAtomicReplace(path, content)) {
        const std::string alternate = base + "." + std::to_string(nowMonoMs()) + ".complete.json";
        if (!writeExclusive(alternate, content)) {
            return {};
        }
        return alternate;
    }
    return path;
}

// Pre-fault readiness check: the diagnostics directory must exist/writable and a
// configured user-trigger pipe must have the expected type *before* a fault, so
// a later failure cannot be blamed on missing permissions or a wrong path.
CudacoreReadiness checkCudacoreReadinessImpl() {
    CudacoreReadiness readiness;
    const DiagnosticsConfig config = resolveDiagnosticsConfig();
    readiness.dir         = config.dir;
    readiness.dir_reason  = config.reason;
    readiness.dir_ok      = ensureDirectory(readiness.dir);
    if (readiness.dir_ok) {
        const std::string probe =
            readiness.dir + "/.cudacore_write_probe." + std::to_string(static_cast<long>(::getpid()));
        const int         fd    = ::open(probe.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            (void)::close(fd);
            (void)::unlink(probe.c_str());
            readiness.dir_writable = true;
        }
    }

    const char* env_pipe = ::getenv("CUDA_COREDUMP_PIPE");
    if (env_pipe != nullptr && env_pipe[0] != '\0') {
        readiness.pipe_configured = true;
        readiness.pipe_path       = env_pipe;
    } else {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.has_snapshot && state.snapshot.global.pipe.status == CudacoreAttributeStatus::Ok
            && !state.snapshot.global.pipe.value.empty()) {
            readiness.pipe_configured = true;
            readiness.pipe_path       = state.snapshot.global.pipe.value;
        }
    }
    if (readiness.pipe_configured) {
        struct stat info = {};
        if (::stat(readiness.pipe_path.c_str(), &info) == 0) {
            readiness.pipe_exists  = true;
            readiness.pipe_is_fifo = S_ISFIFO(info.st_mode) != 0;
        }
    }
    return readiness;
}

std::string readinessJson(const CudacoreReadiness& readiness) {
    std::string json = "{";
    json += "\"diagnostics_dir\":" + jsonString(readiness.dir);
    json += ",\"diagnostics_dir_source\":" + jsonString(readiness.dir_reason);
    json += ",\"diagnostics_dir_ok\":" + std::string(readiness.dir_ok ? "true" : "false");
    json += ",\"diagnostics_dir_writable\":" + std::string(readiness.dir_writable ? "true" : "false");
    json += ",\"user_trigger_pipe_configured\":" + std::string(readiness.pipe_configured ? "true" : "false");
    json += ",\"user_trigger_pipe\":" + jsonString(readiness.pipe_path);
    json += ",\"user_trigger_pipe_exists\":" + std::string(readiness.pipe_exists ? "true" : "false");
    json += ",\"user_trigger_pipe_is_fifo\":" + std::string(readiness.pipe_is_fifo ? "true" : "false");
    json += "}";
    return json;
}

// Collection lease for the parent process: it carries the worker start identity
// and the *original* wall-clock deadline, so a parent cleanup never restarts a
// fresh window and never waits on a reused PID. Written on the collector thread
// before the trigger, so it exists even when the engine thread is stuck.
std::string writeLeaseFile(const std::string& dir, SharedState& state) {
    if (dir.empty() || !ensureDirectory(dir)) {
        return {};
    }
    std::string            base;
    std::string            incident_id;
    CudacoreProcessIdentity identity;
    int                    rank          = -1;
    int64_t                created_wall  = 0;
    int64_t                deadline_wall = 0;
    int64_t                created_mono  = 0;
    int64_t                deadline_mono = 0;
    int64_t                window_ms     = 0;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        base          = manifestBaseName(state);
        incident_id   = state.incident_id;
        identity      = state.identity;
        rank          = state.record.rank;
        created_wall  = state.incident_wall_ms;
        deadline_wall = state.deadline_wall_ms;
        created_mono  = state.incident_mono_ms;
        deadline_mono = state.deadline_mono_ms;
        window_ms     = effectiveWindowMs(state);
    }
    std::string json = "{";
    json += "\"schema_version\":" + jsonString(kLeaseSchemaVersion);
    json += ",\"host\":" + jsonString(hostnameCached());
    json += ",\"pid\":" + std::to_string(static_cast<long>(::getpid()));
    json += ",\"worker_start_id\":" + jsonString(identity.worker_start_id);
    json += ",\"rank\":" + std::to_string(rank);
    json += ",\"device_ordinal\":" + std::to_string(identity.device_ordinal);
    json += ",\"incident_id\":" + jsonString(incident_id);
    json += ",\"created_epoch_ms\":" + std::to_string(created_wall);
    json += ",\"deadline_epoch_ms\":" + std::to_string(deadline_wall);
    json += ",\"created_mono_ms\":" + std::to_string(created_mono);
    json += ",\"deadline_mono_ms\":" + std::to_string(deadline_mono);
    json += ",\"window_ms\":" + std::to_string(window_ms);
    json += "}";
    const std::string path = dir + "/cudacore_lease." + base;
    if (!writeAtomicReplace(path, json)) {
        return {};
    }
    return path;
}

std::string writeStartupManifest(const CudacoreAttributeSnapshot& snapshot,
                                 const CudacoreProcessIdentity&   identity,
                                 const CudacoreReadiness&         readiness) {
    const std::string dir = resolveDiagnosticsDir();
    if (dir.empty() || !ensureDirectory(dir)) {
        return {};
    }
    SharedState& state = sharedState();
    std::string  base;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.identity.worker_start_id = identity.worker_start_id;
        base                           = "cudacore_attributes." + manifestBaseName(state);
    }
    std::string json = "{";
    json += "\"schema_version\":" + jsonString(kSchemaVersion);
    json += ",\"phase\":" + jsonString("startup_audit");
    json += ",\"host\":" + jsonString(hostnameCached());
    json += ",\"pid\":" + std::to_string(static_cast<long>(::getpid()));
    json += ",\"pid_namespace\":" + jsonString(identity.pid_namespace);
    json += ",\"rank\":" + std::to_string(identity.rank);
    json += ",\"device_ordinal\":" + std::to_string(identity.device_ordinal);
    json += ",\"gpu_uuid\":" + jsonString(identity.gpu_uuid);
    json += ",\"image_version\":" + jsonString(identity.image_version);
    json += ",\"worker_start_id\":" + jsonString(identity.worker_start_id);
    json += ",\"startup_coredump_attributes\":" + attributesJson(snapshot);
    json += ",\"readiness\":" + readinessJson(readiness);
    json += "}";

    const std::string path = dir + "/" + base + ".json";
    bool              existed = false;
    if (writeExclusive(path, json, &existed) || existed) {
        return path;
    }
    const std::string alternate = dir + "/" + base + "." + std::to_string(nowMonoMs()) + ".json";
    if (writeExclusive(alternate, json)) {
        return alternate;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Process identity
// ---------------------------------------------------------------------------

CudacoreProcessIdentity buildIdentity(int rank, int device_ordinal) {
    CudacoreProcessIdentity identity;
    identity.hostname       = hostnameCached();
    identity.pid            = static_cast<long>(::getpid());
    identity.pid_namespace  = pidNamespaceId();
    identity.rank           = rank;
    identity.device_ordinal = device_ordinal;
    identity.worker_start_id = processStartId();

    const char* image = ::getenv("RTP_LLM_IMAGE_VERSION");
    if (image != nullptr && image[0] != '\0') {
        identity.image_version = image;
    }

    const DriverSymbols& symbols = driverSymbols();
    int                  driver_version = 0;
    if (symbols.driver_get_version != nullptr) {
        (void)symbols.driver_get_version(&driver_version);
    }
    identity.driver_version  = driver_version;
    identity.runtime_version = runtimeApiVersion();

    if (symbols.device_get != nullptr && symbols.device_get_uuid != nullptr && device_ordinal >= 0) {
        int    device = 0;
        int    code   = symbols.device_get(&device, device_ordinal);
        char   uuid[16] = {};
        if (code == kCuSuccess) {
            code = symbols.device_get_uuid(uuid, device);
        }
        if (code == kCuSuccess) {
            char formatted[64] = {};
            ::snprintf(formatted,
                       sizeof(formatted),
                       "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                       static_cast<unsigned char>(uuid[0]),
                       static_cast<unsigned char>(uuid[1]),
                       static_cast<unsigned char>(uuid[2]),
                       static_cast<unsigned char>(uuid[3]),
                       static_cast<unsigned char>(uuid[4]),
                       static_cast<unsigned char>(uuid[5]),
                       static_cast<unsigned char>(uuid[6]),
                       static_cast<unsigned char>(uuid[7]),
                       static_cast<unsigned char>(uuid[8]),
                       static_cast<unsigned char>(uuid[9]),
                       static_cast<unsigned char>(uuid[10]),
                       static_cast<unsigned char>(uuid[11]),
                       static_cast<unsigned char>(uuid[12]),
                       static_cast<unsigned char>(uuid[13]),
                       static_cast<unsigned char>(uuid[14]),
                       static_cast<unsigned char>(uuid[15]));
            identity.gpu_uuid = formatted;
        }
    }
    return identity;
}

// ---------------------------------------------------------------------------
// Trigger, observation and bounded wait
// ---------------------------------------------------------------------------

std::string resolveUserTriggerPipe() {
    const char* env_value = ::getenv("CUDA_COREDUMP_PIPE");
    if (env_value != nullptr && env_value[0] != '\0') {
        return env_value;
    }
    SharedState& state = sharedState();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.has_snapshot && state.snapshot.global.pipe.status == CudacoreAttributeStatus::Ok
            && !state.snapshot.global.pipe.value.empty()) {
            return state.snapshot.global.pipe.value;
        }
    }
    // Driver default when CUDA_COREDUMP_PIPE is unset.
    return "corepipe.cuda." + hostnameCached() + "." + std::to_string(static_cast<long>(::getpid()));
}

struct TriggerResult {
    CudacoreTriggerStatus status{CudacoreTriggerStatus::NotAttempted};
    int                   error_number{0};
    bool                  attempted{false};
    bool                  sent{false};
};

ssize_t writeIgnoringSigpipe(int descriptor, const void* data, size_t bytes, int* error_number) {
    sigset_t blocked = {};
    sigset_t previous = {};
    ::sigemptyset(&blocked);
    ::sigaddset(&blocked, SIGPIPE);
    ::pthread_sigmask(SIG_BLOCK, &blocked, &previous);
    const ssize_t written = ::write(descriptor, data, bytes);
    *error_number         = errno;
    if (written < 0 && *error_number == EPIPE) {
        // Consume the pending SIGPIPE instead of letting it kill the collector.
        struct timespec zero = {0, 0};
        (void)::sigtimedwait(&blocked, nullptr, &zero);
    }
    ::pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    return written;
}

// Raw FIFO write: nonblocking open, retry for a reader and one bounded write.
TriggerResult triggerUserCoredumpRaw(const std::string& pipe_path, int64_t budget_ms) {
    TriggerResult result;
    result.attempted = true;
    if (pipe_path.empty()) {
        result.status = CudacoreTriggerStatus::PipeNotConfigured;
        return result;
    }

    const int64_t deadline  = nowMonoMs() + std::max<int64_t>(budget_ms, 1);
    while (true) {
        const int descriptor = ::open(pipe_path.c_str(), O_WRONLY | O_NONBLOCK);
        if (descriptor >= 0) {
            const char    byte         = '\n';
            int           error_number = 0;
            const ssize_t written      = writeIgnoringSigpipe(descriptor, &byte, 1, &error_number);
            ::close(descriptor);
            if (written == 1) {
                result.status = CudacoreTriggerStatus::Sent;
                result.sent   = true;
                // A successful write is only TRIGGER_SENT, never "dump succeeded".
                return result;
            }
            result.error_number = error_number;
            if (error_number == EPIPE) {
                result.status = CudacoreTriggerStatus::BrokenPipe;
            } else if (error_number == EAGAIN || error_number == EWOULDBLOCK) {
                result.status = CudacoreTriggerStatus::PipeFull;
            } else {
                result.status = CudacoreTriggerStatus::WriteFailed;
            }
            return result;
        }
        const int last_errno = errno;
        if (last_errno == ENOENT) {
            result.status       = CudacoreTriggerStatus::PipeMissing;
            result.error_number = last_errno;
            return result;
        }
        if (last_errno == EACCES || last_errno == EPERM) {
            result.status       = CudacoreTriggerStatus::PermissionDenied;
            result.error_number = last_errno;
            return result;
        }
        if (last_errno != ENXIO) {
            result.status       = CudacoreTriggerStatus::WriteFailed;
            result.error_number = last_errno;
            return result;
        }
        if (nowMonoMs() >= deadline) {
            result.status       = CudacoreTriggerStatus::NoReader;
            result.error_number = last_errno;
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

// Adds the once-per-incident claim on top of the raw write: static linking may
// keep one module state per DSO, so the right to trigger is claimed through an
// exclusive file that is shared by every copy inside the process.
TriggerResult triggerUserCoredump(const std::string& pipe_path, int64_t budget_ms) {
    const std::string dir = resolveDiagnosticsDir();
    SharedState&      state = sharedState();
    if (ensureDirectory(dir)) {
        std::string base;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            base = manifestBaseName(state) + "." + std::to_string(state.incident_mono_ms);
        }
        const std::string claim = dir + "/.cudacore_trigger_claim." + base;
        const int         fd    = ::open(claim.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd < 0 && errno == EEXIST) {
            TriggerResult result;
            result.status    = CudacoreTriggerStatus::AlreadyAttemptedByPeer;
            result.attempted = true;
            return result;
        }
        if (fd >= 0) {
            (void)::close(fd);
        }
    }
    return triggerUserCoredumpRaw(pipe_path, budget_ms);
}

struct DumpWatch {
    bool        watchable{false};
    bool        exact_path{false};
    std::string exact;
    std::string directory;
    std::string prefix;
};

DumpWatch buildDumpWatch(const std::string& template_path) {
    DumpWatch watch;
    if (template_path.empty() || template_path.find('|') != std::string::npos || !isPlainFileTemplate(template_path)) {
        return watch;
    }
    std::string expanded = template_path;
    for (const auto& [token, value] :
         {std::pair<const char*, std::string>{"%h", hostnameCached()},
          std::pair<const char*, std::string>{"%p", std::to_string(static_cast<long>(::getpid()))}}) {
        size_t position = 0;
        while ((position = expanded.find(token, position)) != std::string::npos) {
            expanded.replace(position, 2, value);
            position += value.size();
        }
    }
    const size_t      percent = expanded.find('%');
    const std::string directory = parentDirectory(expanded);
    watch.directory             = directory.empty() ? startupCwd() : directory;
    if (percent == std::string::npos) {
        watch.watchable  = true;
        watch.exact_path = true;
        watch.exact      = expanded;
        return watch;
    }
    const std::string name   = directory.empty() ? expanded : expanded.substr(directory.size() + 1);
    const std::string prefix = name.substr(0, name.find('%'));
    if (prefix.empty()) {
        return watch;
    }
    watch.watchable = true;
    watch.prefix    = prefix;
    return watch;
}

struct ObservedFile {
    bool        seen{false};
    std::string path;
    uint64_t    bytes{0};
};

ObservedFile scanForDumpFile(const DumpWatch& watch, int64_t not_before_wall_ms) {
    ObservedFile observed;
    if (!watch.watchable) {
        return observed;
    }
    if (watch.exact_path) {
        struct stat info = {};
        if (::stat(watch.exact.c_str(), &info) == 0 && S_ISREG(info.st_mode)) {
            observed.seen  = true;
            observed.path  = watch.exact;
            observed.bytes = static_cast<uint64_t>(info.st_size);
        }
        return observed;
    }
    DIR* directory = ::opendir(watch.directory.c_str());
    if (directory == nullptr) {
        return observed;
    }
    int64_t best_wall = 0;
    while (struct dirent* entry = ::readdir(directory)) {
        const std::string name = entry->d_name;
        if (name.rfind(watch.prefix, 0) != 0) {
            continue;
        }
        const std::string path = watch.directory + "/" + name;
        struct stat       info = {};
        if (::stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode)) {
            continue;
        }
        const int64_t modified_ms = static_cast<int64_t>(info.st_mtime) * 1000;
        if (modified_ms + 1000 < not_before_wall_ms) {
            continue;  // stale file from an earlier run
        }
        if (modified_ms >= best_wall) {
            best_wall      = modified_ms;
            observed.seen  = true;
            observed.path  = path;
            observed.bytes = static_cast<uint64_t>(info.st_size);
        }
    }
    ::closedir(directory);
    return observed;
}

bool startupDumpDefinitelyDisabled() {
    SharedState& state = sharedState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.has_snapshot) {
        return false;
    }
    const bool global_off = state.snapshot.global.enable_on_exception.status == CudacoreAttributeStatus::Ok
                            && !state.snapshot.global.enable_on_exception.value;
    const bool context_off = !state.snapshot.context.context_valid
                             || (state.snapshot.context.enable_on_exception.status == CudacoreAttributeStatus::Ok
                                 && !state.snapshot.context.enable_on_exception.value);
    return global_off && context_off;
}

bool userTriggerDefinitelyDisabled() {
    SharedState& state = sharedState();
    std::lock_guard<std::mutex> lock(state.mutex);
    const bool attribute_off = state.has_snapshot && state.snapshot.global.enable_user_trigger.status
                                                     == CudacoreAttributeStatus::Ok
                               && !state.snapshot.global.enable_user_trigger.value;
    const char* env_value = ::getenv("CUDA_ENABLE_USER_TRIGGERED_COREDUMP");
    const bool  env_on    = env_value != nullptr && env_value[0] == '1';
    return attribute_off && !env_on;
}

// Fills trigger/observation fields into outcome; lease and manifest paths set by
// the caller stay untouched.
void runCollectionWindow(CudacoreCollectionOutcome& outcome) noexcept {
    SharedState& state = sharedState();

    std::string template_path;
    std::string pipe_path;
    std::string diagnostics_dir;
    int64_t     deadline         = 0;
    int64_t     incident_wall_ms = 0;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        template_path = effectiveDumpFileTemplate();
        if (template_path.empty() && state.has_snapshot
            && state.snapshot.global.file.status == CudacoreAttributeStatus::Ok) {
            template_path = state.snapshot.global.file.value;
        }
        // The deadline is fixed by the first error; a later caller never extends it.
        deadline         = state.deadline_mono_ms;
        incident_wall_ms = state.incident_wall_ms;
    }
    pipe_path                  = resolveUserTriggerPipe();
    diagnostics_dir            = resolveDiagnosticsDir();
    outcome.dump_file_template = template_path;
    outcome.diagnostics_dir    = diagnostics_dir;

    const DumpWatch watch      = buildDumpWatch(template_path);
    const int64_t   start_mono = nowMonoMs();

    const bool dump_disabled  = startupDumpDefinitelyDisabled();
    const bool trigger_off    = userTriggerDefinitelyDisabled();
    const bool nothing_to_wait = dump_disabled && trigger_off && !watch.watchable;
    if (nothing_to_wait) {
        outcome.terminal = CudacoreTerminalState::NoMechanismAvailable;
        outcome.note =
            "exception dump disabled, user trigger disabled and no dump file template: nothing can be collected";
        outcome.waited_ms = nowMonoMs() - start_mono;
        return;
    }

    // The window is about to open: publish the lease before the trigger, so the
    // parent can respect the window even if this thread dies right afterwards.
    outcome.lease_path = writeLeaseFile(diagnostics_dir, state);

    // Skip the extra trigger when an automatic dump already started.
    const ObservedFile existing = scanForDumpFile(watch, incident_wall_ms);
    if (existing.seen) {
        outcome.trigger        = CudacoreTriggerStatus::AlreadyObservedProgress;
        outcome.file_seen      = true;
        outcome.observed_file  = existing.path;
        outcome.observed_bytes = existing.bytes;
    } else if (!trigger_off && nowMonoMs() < deadline) {
        const int64_t trigger_budget =
            std::min<int64_t>(CudacoreDiagConstants::kTriggerIoBudgetMs, deadline - nowMonoMs());
        const TriggerResult trigger = trigger_budget > 0 ? triggerUserCoredump(pipe_path, trigger_budget)
                                                         : TriggerResult{};
        outcome.trigger           = trigger.status;
        outcome.trigger_errno     = trigger.error_number;
        outcome.trigger_attempted = trigger.attempted;
        outcome.trigger_sent      = trigger.sent;
    } else {
        outcome.note = "user triggered coredump is disabled; automatic exception dump only";
    }

    // The window always runs to the shared deadline. A stable file size is only
    // an observation: segmented dumps pause for longer than the poll interval,
    // and releasing the process early would truncate the dump.
    uint64_t    last_size          = 0;
    std::string last_path;
    bool        dirty              = true;  // Persist trigger/lease even before a file appears.
    int64_t     last_evidence_mono = 0;
    auto observe = [&](const ObservedFile& observed) {
        const bool stable = observed.seen && observed.bytes > 0 && observed.path == last_path
                            && observed.bytes == last_size;
        dirty = dirty || outcome.file_size_stable != stable;
        outcome.file_size_stable = stable;
        if (observed.seen) {
            dirty = dirty || !outcome.file_seen || observed.path != outcome.observed_file
                    || observed.bytes != outcome.observed_bytes;
            outcome.file_seen      = true;
            outcome.observed_file  = observed.path;
            outcome.observed_bytes = observed.bytes;
            last_path              = observed.path;
            last_size              = observed.bytes;
        } else {
            last_path.clear();
            last_size = 0;
        }
    };
    while (nowMonoMs() < deadline) {
        observe(scanForDumpFile(watch, incident_wall_ms));
        outcome.waited_ms = nowMonoMs() - start_mono;
        // Changes suppressed by the throttle remain dirty until persisted. The
        // driver may abort the process before this window returns.
        if (dirty && nowMonoMs() - last_evidence_mono >= kEvidenceWriteIntervalMs) {
            last_evidence_mono = nowMonoMs();
            const std::string progress = writeManifest(diagnostics_dir, state, outcome, "progress");
            if (!progress.empty()) {
                outcome.manifest_path = progress;
                dirty = false;
            }
        }
        const int64_t remaining = deadline - nowMonoMs();
        if (remaining <= 0) {
            break;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(std::min<int64_t>(CudacoreDiagConstants::kStatusPollIntervalMs, remaining)));
    }

    observe(scanForDumpFile(watch, incident_wall_ms));
    // A completed application wait is not a completed driver dump, even when
    // the output was temporarily stable. Offline validation remains necessary.
    outcome.terminal          = CudacoreTerminalState::DeadlineExceeded;
    outcome.deadline_exceeded = true;
    outcome.waited_ms = nowMonoMs() - start_mono;
    // Completeness is not proven by existence or size: offline cuda-gdb
    // validation stays pending and is recorded separately.
}

// ---------------------------------------------------------------------------
// Fatal error classification
// ---------------------------------------------------------------------------

std::string lowerCase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return result;
}

bool hasFatalMarker(const std::string& message) {
    static const char* markers[] = {
        "an illegal memory access was encountered",
        "an illegal instruction was encountered",
        "unspecified launch failure",
        "misaligned address",
        "invalid pc",
        "invalid address space",
        "device-side assert",
        "device side assert",
        "an exception occurred on the device while executing a kernel",
        "cudaerrorillegaladdress",
        "cudaerrorlaunchfailure",
        "cudaerrorillegalinstruction",
        "cudaerrormisalignedaddress",
        "cudaerrorinvalidpc",
        "cudaerrorassert",
        "cudaerroreccuncorrectable",
        "uncorrectable ecc error",
    };
    for (const char* marker : markers) {
        if (message.find(marker) != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API: attributes
// ---------------------------------------------------------------------------

void setCudacoreProcessIdentity(int rank, int device_ordinal) noexcept {
    try {
        SharedState& state = sharedState();
        const CudacoreProcessIdentity identity = buildIdentity(rank, device_ordinal);
        std::lock_guard<std::mutex> lock(state.mutex);
        state.identity = identity;
    } catch (...) {
    }
}

CudacoreProcessIdentity cudacoreProcessIdentity() noexcept {
    try {
        SharedState& state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.identity.worker_start_id.empty()) {
            state.identity = buildIdentity(state.identity.rank, state.identity.device_ordinal);
        }
        return state.identity;
    } catch (...) {
        return CudacoreProcessIdentity{};
    }
}

CudacoreAttributeSnapshot queryCudacoreAttributes(bool context_valid) noexcept {
    CudacoreAttributeSnapshot snapshot;
    try {
        if (usingTestAttributeApi()) {
            snapshot.symbols_source = "stub";
        } else if (driverSymbols().get_attribute_global != nullptr || driverSymbols().get_attribute != nullptr) {
            snapshot.symbols_source = "driver";
        } else {
            snapshot.symbols_source = "unavailable";
        }

        snapshot.global.enable_on_exception = queryBool(/*global_scope=*/true, kAttribEnableOnException);
        snapshot.global.enable_user_trigger = queryBool(/*global_scope=*/true, kAttribEnableUserTrigger);
        snapshot.global.trigger_host        = queryBool(/*global_scope=*/true, kAttribTriggerHost);
        snapshot.global.lightweight         = queryBool(/*global_scope=*/true, kAttribLightweight);
        snapshot.global.file                = queryString(/*global_scope=*/true, kAttribFile);
        snapshot.global.pipe                = queryString(/*global_scope=*/true, kAttribPipe);
        snapshot.global.generation_flags    = queryFlags(/*global_scope=*/true, kAttribGenerationFlags);

        snapshot.context.context_valid = context_valid;
        if (context_valid) {
            snapshot.context.enable_on_exception = queryBool(/*global_scope=*/false, kAttribEnableOnException);
            snapshot.context.file                = queryString(/*global_scope=*/false, kAttribFile);
            snapshot.context.generation_flags    = queryFlags(/*global_scope=*/false, kAttribGenerationFlags);
        } else {
            snapshot.context.enable_on_exception.status = CudacoreAttributeStatus::Skipped;
            snapshot.context.file.status                = CudacoreAttributeStatus::Skipped;
            snapshot.context.generation_flags.status    = CudacoreAttributeStatus::Skipped;
        }

        const DriverSymbols& symbols = driverSymbols();
        int                  driver_version = 0;
        if (symbols.driver_get_version != nullptr) {
            (void)symbols.driver_get_version(&driver_version);
        }
        snapshot.driver_version  = driver_version;
        snapshot.runtime_version = runtimeApiVersion();
    } catch (...) {
    }
    return snapshot;
}

std::string formatCudacoreAttributeAudit(const CudacoreAttributeSnapshot& snapshot,
                                         const CudacoreProcessIdentity&   identity) noexcept {
    try {
        std::string json = "{";
        json += "\"host\":" + jsonString(identity.hostname);
        json += ",\"pid\":" + std::to_string(identity.pid);
        json += ",\"pid_namespace\":" + jsonString(identity.pid_namespace);
        json += ",\"rank\":" + std::to_string(identity.rank);
        json += ",\"device_ordinal\":" + std::to_string(identity.device_ordinal);
        json += ",\"gpu_uuid\":" + jsonString(identity.gpu_uuid);
        json += ",\"worker_start_id\":" + jsonString(identity.worker_start_id);
        json += ",\"image_version\":" + jsonString(identity.image_version);
        json += ",\"startup_coredump_attributes\":" + attributesJson(snapshot);
        json += "}";
        return json;
    } catch (...) {
        return "{}";
    }
}

void auditCudacoreAttributesAtStartup() noexcept {
    try {
        const int64_t begin_mono = nowMonoMs();
        // The collection executor must exist before any fault: it performs the
        // trigger and the evidence writes on its own thread.
        startCudacoreCollector();
        CudacoreProcessIdentity identity = cudacoreProcessIdentity();
        const CudacoreAttributeSnapshot snapshot = queryCudacoreAttributes(/*context_valid=*/true);
        {
            SharedState& state = sharedState();
            std::lock_guard<std::mutex> lock(state.mutex);
            state.snapshot     = snapshot;
            state.has_snapshot = true;
            identity.rank      = state.identity.rank;
            identity.device_ordinal = state.identity.device_ordinal;
        }
        const std::string        audit_json = formatCudacoreAttributeAudit(snapshot, identity);
        const CudacoreReadiness  readiness  = checkCudacoreReadiness();
        const std::string        manifest   = writeStartupManifest(snapshot, identity, readiness);
        const int64_t            elapsed_ms = nowMonoMs() - begin_mono;
        RTP_LLM_LOG_INFO("[CudacoreDiag] startup audit elapsed_ms=%lld manifest=%s readiness=%s %s",
                         static_cast<long long>(elapsed_ms),
                         manifest.empty() ? "<none>" : manifest.c_str(),
                         readinessJson(readiness).c_str(),
                         audit_json.c_str());
        if (!readiness.dir_writable) {
            RTP_LLM_LOG_WARNING("[CudacoreDiag] diagnostics directory is not writable (%s); collection evidence "
                                "will only be logged",
                                readiness.dir.c_str());
        }
        if (readiness.pipe_configured && !readiness.pipe_is_fifo) {
            RTP_LLM_LOG_WARNING("[CudacoreDiag] configured user-trigger pipe %s is not an existing FIFO (exists=%d); "
                                "the user-trigger request may not reach the driver",
                                readiness.pipe_path.c_str(),
                                readiness.pipe_exists ? 1 : 0);
        }
        if (snapshot.global.enable_on_exception.status == CudacoreAttributeStatus::Ok
            && !snapshot.global.enable_on_exception.value) {
            RTP_LLM_LOG_WARNING("[CudacoreDiag] driver reports exception coredump disabled; GPU dumps will not be "
                                "generated on device faults");
        }
    } catch (...) {
        RTP_LLM_LOG_WARNING("[CudacoreDiag] startup audit failed; continuing without it");
    }
}

std::string cudacoreDiagnosticsDir() noexcept {
    try {
        return resolveDiagnosticsDir();
    } catch (...) {
        return {};
    }
}

CudacoreReadiness checkCudacoreReadiness() noexcept {
    try {
        return checkCudacoreReadinessImpl();
    } catch (...) {
        return CudacoreReadiness{};
    }
}

// ---------------------------------------------------------------------------
// Public API: fatal classification and first error record
// ---------------------------------------------------------------------------

bool isFatalCudaRuntimeError(int cuda_error) noexcept {
    switch (cuda_error) {
        case kFatalIllegalAddress:
        case kFatalLaunchTimeout:
        case kFatalAssert:
        case kFatalEccUncorrectable:
        case kFatalIllegalInsn:
        case kFatalMisalignedAddr:
        case kFatalBadAddressSpace:
        case kFatalInvalidPc:
        case kFatalLaunchFailed:
            return true;
        default:
            return false;
    }
}

bool isFatalCudaDriverError(int cu_result) noexcept {
    return isFatalCudaRuntimeError(cu_result);
}

bool isFatalCudaException(const std::exception& exception) noexcept {
    try {
        return hasFatalMarker(lowerCase(exception.what()));
    } catch (...) {
        return false;
    }
}

namespace {

void fillTimestamps(FatalCudaErrorRecord& record) {
    record.wall_ms  = nowWallMs();
    record.mono_ms  = nowMonoMs();
    record.thread_id = threadIdString();
    SharedState& state = sharedState();
    std::lock_guard<std::mutex> lock(state.mutex);
    record.rank = state.identity.rank;
}

// Fixed metadata budget: first identity fields, then as many tiles as fit in
// kMaxMetadataBytes. Totals and truncation flags are always kept.
void applyMetadataBudget(FatalCudaErrorRecord& record) {
    record.message      = truncateText(record.message);
    record.code_name    = truncateText(record.code_name);
    record.source_file  = truncateText(record.source_file, 256);
    record.thread_id    = truncateText(record.thread_id, 64);
    record.stream       = truncateText(record.stream, 128);

    if (record.tile_total == 0) {
        record.tile_total = record.tiles.size();
    }
    if (record.tiles.size() > CudacoreDiagConstants::kMaxTileMetadata) {
        record.tiles.resize(CudacoreDiagConstants::kMaxTileMetadata);
        record.tiles_truncated = true;
    }
    constexpr size_t kTileRecordBytes = sizeof(FatalCudaTileRecord);
    size_t           used             = record.message.size() + record.code_name.size() + record.source_file.size()
                            + record.thread_id.size() + record.stream.size();
    size_t kept = 0;
    while (kept < record.tiles.size() && used + kTileRecordBytes <= CudacoreDiagConstants::kMaxMetadataBytes) {
        used += kTileRecordBytes;
        ++kept;
    }
    if (kept < record.tiles.size()) {
        record.tiles.resize(kept);
        record.metadata_truncated = true;
    }
}

}  // namespace

FatalCudaErrorRecord buildCudaRuntimeErrorRecord(int                cuda_error,
                                                 FatalCudaErrorSite site,
                                                 const char*        file,
                                                 int                line,
                                                 int                device_index) noexcept {
    FatalCudaErrorRecord record;
    try {
        record.domain       = FatalCudaErrorDomain::CudaRuntime;
        record.code         = cuda_error;
        record.code_name    = "cudaError(" + std::to_string(cuda_error) + ")";
        record.message      = "CUDA runtime error " + std::to_string(cuda_error);
        record.site         = site;
        record.source_file  = file == nullptr ? "" : file;
        record.source_line  = line;
        record.device_index = device_index;
        fillTimestamps(record);
    } catch (...) {
    }
    return record;
}

FatalCudaErrorRecord buildCudaDriverErrorRecord(int                cu_result,
                                                FatalCudaErrorSite site,
                                                const char*        file,
                                                int                line,
                                                int                device_index) noexcept {
    FatalCudaErrorRecord record;
    try {
        record.domain       = FatalCudaErrorDomain::CudaDriver;
        record.code         = cu_result;
        record.code_name    = "CUresult(" + std::to_string(cu_result) + ")";
        record.message      = "CUDA driver error " + std::to_string(cu_result);
        record.site         = site;
        record.source_file  = file == nullptr ? "" : file;
        record.source_line  = line;
        record.device_index = device_index;
        fillTimestamps(record);
    } catch (...) {
    }
    return record;
}

FatalCudaErrorRecord buildCudaExceptionRecord(const std::exception& exception,
                                              FatalCudaErrorSite    site,
                                              const char*           file,
                                              int                   line) noexcept {
    FatalCudaErrorRecord record;
    try {
        record.domain         = FatalCudaErrorDomain::TorchException;
        record.code           = -1;
        record.message        = truncateText(exception.what() == nullptr ? "" : exception.what());
        record.site           = site;
        record.source_file    = file == nullptr ? "" : file;
        record.source_line    = line;
        record.low_confidence = true;  // classified from text, not from a numeric code
        fillTimestamps(record);
    } catch (...) {
    }
    return record;
}

bool recordFirstFatalCudaError(FatalCudaErrorRecord record) noexcept {
    installCudacoreTerminateGuard();
    try {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.has_record) {
            return false;  // first error wins; the deadline is never extended
        }
        if (record.wall_ms == 0) {
            record.wall_ms = nowWallMs();
        }
        if (record.mono_ms == 0) {
            record.mono_ms = nowMonoMs();
        }
        if (record.rank < 0) {
            record.rank = state.identity.rank;
        }
        if (record.device_index < 0) {
            record.device_index = state.identity.device_ordinal;
        }
        if (record.thread_id.empty()) {
            record.thread_id = threadIdString();
        }
        applyMetadataBudget(record);
        ++state.incident_generation;
        state.has_record        = true;
        state.terminal          = false;
        state.terminal_flag.store(false, std::memory_order_release);
        state.record            = std::move(record);
        state.incident_wall_ms  = state.record.wall_ms;
        state.incident_mono_ms  = state.record.mono_ms;
        state.deadline_mono_ms  = state.record.mono_ms + effectiveWindowMs(state);
        state.deadline_mono_atomic.store(state.deadline_mono_ms, std::memory_order_release);
        state.deadline_wall_ms  = state.record.wall_ms + effectiveWindowMs(state);
        state.incident_flag.store(true, std::memory_order_release);
        freezeCudacoreFlightRecorder();
        const std::string base  = manifestBaseName(state);
        state.incident_id       = base + "." + std::to_string(state.record.mono_ms);
        const FatalCudaErrorRecord& stored = state.record;
        RTP_LLM_LOG_ERROR(
            "[CudacoreDiag] first fatal CUDA error: domain=%s code=%d site=%s device=%d rank=%d file=%s:%d "
            "tiles=%llu bytes=%llu low_confidence=%d incident_id=%s",
            domainName(stored.domain),
            stored.code,
            siteName(stored.site),
            stored.device_index,
            stored.rank,
            stored.source_file.c_str(),
            stored.source_line,
            static_cast<unsigned long long>(stored.tile_total),
            static_cast<unsigned long long>(stored.bytes_total),
            stored.low_confidence ? 1 : 0,
            state.incident_id.c_str());
        // Wake the collection executor immediately: the trigger and the evidence
        // must not depend on the engine thread ever reaching a wait call.
        state.collector_wakeup.notify_all();
        return true;
    } catch (...) {
        return false;
    }
}

void annotateFatalCudaTransferContext(uint64_t  group_set_id,
                                      bool      device_to_host,
                                      uintptr_t host_base,
                                      uint64_t  host_bytes) noexcept {
    try {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        // Multiple copy streams can report the same device fault. Only the
        // thread that claimed the first error may attach its transfer context.
        if (!state.has_record || state.record.thread_id != threadIdString()) {
            return;
        }
        if (!state.record.has_group_set_id) {
            state.record.has_group_set_id = true;
            state.record.group_set_id     = group_set_id;
        }
        if (!state.record.has_host_span) {
            state.record.has_host_span = true;
            state.record.host_base     = host_base;
            state.record.host_bytes    = host_bytes;
            state.record.direction     = device_to_host ? "D2H" : "H2D";
        }
    } catch (...) {
    }
}

bool fatalCudacoreIncidentActive() noexcept {
    try {
        return sharedState().incident_flag.load(std::memory_order_acquire);
    } catch (...) {
        return false;
    }
}

bool fatalCudacoreCollectionInProgress() noexcept {
    try {
        SharedState& state = sharedState();
        return state.incident_flag.load(std::memory_order_acquire)
               && !state.terminal_flag.load(std::memory_order_acquire);
    } catch (...) {
        return false;
    }
}

void installCudacoreTerminateGuard() noexcept {
    try {
        static std::mutex           install_mutex;
        std::lock_guard<std::mutex> lock(install_mutex);
        (void)sharedState();
        // PyTorch or another library may have replaced the handler since warmup.
        // Reinstall only on a fatal error or the first copy entry point.
        if (std::get_terminate() != &cudacoreTerminateHandler) {
            const auto previous = std::set_terminate(&cudacoreTerminateHandler);
            previous_terminate_handler.store(previous, std::memory_order_release);
        }
    } catch (...) {}
}

void recordCudacorePoolLifetime(
    const char* pool_name, uintptr_t base, uint64_t bytes, int device_index, bool initialized) noexcept {
    try {
        if (base == 0 || bytes == 0) {
            return;
        }
        const std::string dir = resolveDiagnosticsDir();
        if (!ensureDirectory(dir)) {
            return;
        }
        const std::string path = dir + "/cudacore_allocations." + sanitizeComponent(hostnameCached()) + "."
                                 + std::to_string(static_cast<long>(::getpid())) + "."
                                 + sanitizeComponent(processStartId()) + ".jsonl";
        std::string entry = "{\"event\":" + jsonString(initialized ? "pool_initialized" : "pool_destroyed");
        entry += ",\"time_mono_ms\":" + std::to_string(nowMonoMs());
        entry += ",\"pool\":" + jsonString(pool_name == nullptr ? "" : pool_name);
        entry += ",\"base\":" + std::to_string(base);
        entry += ",\"bytes\":" + std::to_string(bytes);
        entry += ",\"device_index\":" + std::to_string(device_index);
        entry += "}\n";
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (fd < 0) {
            return;
        }
        (void)::write(fd, entry.data(), entry.size());
        (void)::close(fd);
    } catch (...) {}
}

bool fatalCudacoreErrorRecord(FatalCudaErrorRecord& out) noexcept {
    try {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.has_record) {
            return false;
        }
        out = state.record;
        return true;
    } catch (...) {
        return false;
    }
}

// ---------------------------------------------------------------------------
// Collection executor and bounded collection window
// ---------------------------------------------------------------------------

namespace {

// Runs the whole collection once and publishes the outcome. Executed either by
// the collector thread or, when no collector was started, by the first waiter.
void runCollectionAndPublish() {
    SharedState& state = sharedState();
    uint64_t     generation = 0;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.has_record) {
            return;
        }
        generation = state.incident_generation;
    }

    CudacoreCollectionOutcome outcome;
    try {
        const std::string dir = resolveDiagnosticsDir();
        outcome.diagnostics_dir = dir;
        outcome.manifest_path = writeManifest(dir, state, outcome, "fault");
        runCollectionWindow(outcome);
        const std::string completed = writeManifest(dir, state, outcome, "complete");
        if (!completed.empty()) {
            outcome.manifest_path = completed;
        }
    } catch (...) {
        outcome.note += " collection window raised an internal error";
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.incident_generation != generation) {
        return;  // a reset or a newer incident superseded this run
    }
    state.outcome  = outcome;
    state.terminal = true;
    state.terminal_flag.store(true, std::memory_order_release);
    state.finished.notify_all();
}

void collectorLoop() {
    SharedState& state = sharedState();
    while (true) {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.collector_wakeup.wait(lock, [] {
            SharedState& current = sharedState();
            return (current.has_record && !current.terminal) || current.collector_stop;
        });
        if (state.collector_stop) {
            return;
        }
        lock.unlock();
        runCollectionAndPublish();
    }
}

}  // namespace

void startCudacoreCollector() noexcept {
    try {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.collector_started) {
            return;
        }
        // Detached: the executor runs while the engine waits for collection;
        // it does not survive process exit. It performs file
        // I/O only - no CUDA calls, no Python, no allocation of unbounded data.
        std::thread(collectorLoop).detach();
        // Publish only after creation succeeds, so failure preserves inline fallback.
        state.collector_started = true;
    } catch (...) {
        RTP_LLM_LOG_WARNING("[CudacoreDiag] collector thread unavailable; retaining inline fallback");
    }
}

CudacoreCollectionOutcome waitForCudacoreCollection() noexcept {
    SharedState* state = nullptr;
    try {
        state = &sharedState();
    } catch (...) {
        return {};
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    if (!state->has_record) {
        lock.unlock();
        CudacoreCollectionOutcome empty;
        empty.note = "no fatal CUDA incident recorded";
        return empty;
    }
    if (state->terminal) {
        CudacoreCollectionOutcome outcome = state->outcome;
        lock.unlock();
        return outcome;
    }

    if (!state->collector_started && !state->inline_claimed) {
        // No collection executor was started: this caller performs the window.
        state->inline_claimed = true;
        lock.unlock();
        runCollectionAndPublish();
        lock.lock();
        return state->outcome;
    }

    // The collection executor owns the window; waiters only share the deadline.
    const auto deadline_point =
        std::chrono::steady_clock::time_point(std::chrono::milliseconds(state->deadline_mono_ms))
        + std::chrono::seconds(5);
    state->finished.wait_until(lock, deadline_point, [&] { return state->terminal; });
    CudacoreCollectionOutcome outcome = state->outcome;
    lock.unlock();
    return outcome;
}

// ---------------------------------------------------------------------------
// Test seams
// ---------------------------------------------------------------------------

namespace cudacore_test {

void installAttributeApi(const AttributeApi& api) noexcept {
    testAttributeApi() = api;
    testAttributeApi().installed = true;
}

void resetAttributeApi() noexcept {
    testAttributeApi() = AttributeApi{};
}

void setDiagnosticsDirOverride(const std::string& dir) noexcept {
    try {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.dir_override = dir;
        state.resolved_dir.clear();
    } catch (...) {
    }
}

void resetDiagnosticsDirOverride() noexcept {
    try {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.dir_override.clear();
        state.resolved_dir.clear();
    } catch (...) {
    }
}

void resetIncidentState() noexcept {
    try {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        ++state.incident_generation;
        state.has_record      = false;
        state.record          = FatalCudaErrorRecord{};
        state.deadline_mono_ms = 0;
        state.deadline_mono_atomic.store(0, std::memory_order_release);
        state.deadline_wall_ms = 0;
        state.incident_wall_ms = 0;
        state.incident_mono_ms = 0;
        state.incident_id.clear();
        state.inline_claimed  = false;
        state.terminal        = false;
        state.outcome         = CudacoreCollectionOutcome{};
        state.incident_flag.store(false, std::memory_order_release);
        cudacore_test::resetFlightRecorder();
        state.terminal_flag.store(false, std::memory_order_release);
    } catch (...) {
    }
}

void setCollectionWindowMsForTest(int64_t window_ms) noexcept {
    try {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.collection_window_ms = window_ms > 0 ? window_ms : 0;
    } catch (...) {
    }
}

void setStartupAttributeSnapshot(const CudacoreAttributeSnapshot& snapshot) noexcept {
    try {
        SharedState&                state = sharedState();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.snapshot     = snapshot;
        state.has_snapshot = true;
    } catch (...) {
    }
}

CudacoreTriggerStatus triggerUserCoredumpForTest(const std::string& pipe_path,
                                                 int64_t            budget_ms,
                                                 int*               error_number) noexcept {
    try {
        const TriggerResult result = triggerUserCoredumpRaw(pipe_path, budget_ms);
        if (error_number != nullptr) {
            *error_number = result.error_number;
        }
        return result.status;
    } catch (...) {
        if (error_number != nullptr) {
            *error_number = -1;
        }
        return CudacoreTriggerStatus::WriteFailed;
    }
}

}  // namespace cudacore_test

}  // namespace rtp_llm
