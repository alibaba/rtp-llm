#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

// Temporary GPU coredump diagnostics for the 2026-09-16 prefill CUDA crash.
// See internal_source/docs/prefill_cuda_crash_20260916_2152/cudacore_remediation_todo.md
//
// P0: read-only audit of the coredump configuration the driver actually adopted.
// P1: structured first fatal error record, at most one bounded user-trigger
//     attempt, a process-wide bounded collection window and an on-disk manifest.
//
// Diagnostic API entry points are noexcept and do not create a CUDA context or
// call a coredump setter. The terminate guard delegates to the previous handler
// after its bounded wait, or aborts if none is installed. All numbers below are
// constants on purpose; rollout is controlled by the image version.

namespace rtp_llm {

struct CudacoreDiagConstants {
    // One process-wide monotonic deadline per incident, measured from the first
    // fatal error. Not an NVIDIA guarantee; adjust the constant and re-verify.
    static constexpr int64_t     kCollectionDeadlineMs   = 30000;
    // Open + wait-for-reader + write retries share this budget (no extra time).
    static constexpr int64_t     kTriggerIoBudgetMs      = 1000;
    static constexpr int64_t     kStatusPollIntervalMs   = 100;
    // Upper bound on bounded first-error tile metadata.
    static constexpr std::size_t kMaxTileMetadata        = 1024;
    static constexpr std::size_t kMaxMetadataBytes       = 256 * 1024;
    // Caps for driver attribute strings / manifest text fields.
    static constexpr std::size_t kMaxAttributeBytes      = 4096;
    static constexpr std::size_t kMaxTextBytes           = 1024;
    static constexpr const char* kDiagnosticsDirName     = "cudacore_diagnostics";
};

// ---------------------------------------------------------------------------
// P0: read-only coredump attribute audit
// ---------------------------------------------------------------------------

enum class CudacoreAttributeStatus {
    NotQueried,
    Ok,
    // Attribute or symbol not provided by this driver/CUDA version. Never
    // reported as a valid value: unknown must not masquerade as false.
    Unsupported,
    // No usable context (not initialized / invalid / destroyed context).
    Unavailable,
    Error,
    Skipped,
};

struct CudacoreBoolAttribute {
    CudacoreAttributeStatus status{CudacoreAttributeStatus::NotQueried};
    int                     api_code{0};
    bool                    value{false};
};

struct CudacoreStringAttribute {
    CudacoreAttributeStatus status{CudacoreAttributeStatus::NotQueried};
    int                     api_code{0};
    std::string             value;
    bool                    truncated{false};
};

struct CudacoreFlagsAttribute {
    CudacoreAttributeStatus status{CudacoreAttributeStatus::NotQueried};
    int                     api_code{0};
    uint64_t                value{0};
};

struct CudacoreGlobalAttributes {
    CudacoreBoolAttribute   enable_on_exception;
    CudacoreBoolAttribute   enable_user_trigger;
    CudacoreBoolAttribute   trigger_host;  // deprecated since CUDA 12.5, recorded as reported
    CudacoreBoolAttribute   lightweight;   // deprecated since CUDA 12.5, recorded as reported
    CudacoreStringAttribute file;
    CudacoreStringAttribute pipe;
    CudacoreFlagsAttribute  generation_flags;
};

struct CudacoreContextAttributes {
    bool                    context_valid{false};
    CudacoreBoolAttribute   enable_on_exception;
    CudacoreStringAttribute file;
    CudacoreFlagsAttribute  generation_flags;
};

struct CudacoreAttributeSnapshot {
    CudacoreGlobalAttributes  global;
    CudacoreContextAttributes context;
    int                       driver_version{0};
    int                       runtime_version{0};
    // "driver" when the real driver symbols were resolved, "unavailable" when
    // no CUDA driver was reachable, "stub" when a test API was installed.
    std::string               symbols_source{"unavailable"};
};

struct CudacoreProcessIdentity {
    std::string hostname;
    long        pid{0};
    std::string pid_namespace;   // best effort, empty when unavailable
    int         rank{-1};
    int         device_ordinal{-1};
    std::string gpu_uuid;        // empty when the driver query failed
    std::string image_version;   // from the image/version env, empty when unset
    int         driver_version{0};
    int         runtime_version{0};
    // pid + process start time: stable inside one process, distinct after PID reuse.
    std::string worker_start_id;
};

// Process-wide identity used by the audit and the manifest. Safe to call once
// per process after CUDA initialization; later calls update rank/device only.
void setCudacoreProcessIdentity(int rank, int device_ordinal) noexcept;
CudacoreProcessIdentity cudacoreProcessIdentity() noexcept;

// Query the effective configuration. context_valid must reflect whether the
// caller has already established a current context; on false the context scope
// is marked Skipped and no context is created implicitly.
CudacoreAttributeSnapshot queryCudacoreAttributes(bool context_valid) noexcept;

// Audit entry point: queries once, logs a structured summary, writes the
// startup manifest into the diagnostics directory and remembers the effective
// dump file/pipe settings for the collection window. Never blocks startup.
void auditCudacoreAttributesAtStartup() noexcept;

// Pre-fault readiness of the collection plumbing: diagnostics directory
// existence/permissions and the user-trigger pipe type/path. Read-only apart
// from creating the diagnostics directory and a short-lived write probe.
struct CudacoreReadiness {
    bool        dir_ok{false};
    bool        dir_writable{false};
    std::string dir;
    std::string dir_reason;
    bool        pipe_configured{false};
    std::string pipe_path;
    bool        pipe_exists{false};
    bool        pipe_is_fifo{false};
};

CudacoreReadiness checkCudacoreReadiness() noexcept;

// Single-line JSON summary, also used by tests.
std::string formatCudacoreAttributeAudit(const CudacoreAttributeSnapshot& snapshot,
                                         const CudacoreProcessIdentity&   identity) noexcept;

// ---------------------------------------------------------------------------
// P1: first fatal error record, trigger and bounded collection window
// ---------------------------------------------------------------------------

enum class FatalCudaErrorDomain {
    CudaRuntime,
    CudaDriver,
    Cublas,
    TorchException,
    Unknown,
};

enum class FatalCudaErrorSite {
    BatchedCopySubmit,
    BatchedCopyCompletion,
    StagedCopy,
    GenericCheck,
    DeviceHostCopy,
    EngineStep,
    Unknown,
};

struct FatalCudaTileRecord {
    uintptr_t dst{0};
    uintptr_t src{0};
    uint64_t  bytes{0};
};

// Bounded by CudacoreDiagConstants: at most kMaxTileMetadata tiles and
// kMaxMetadataBytes of text/tile payload. Metadata is copied on the faulting
// thread before the objects can die, and never by reading GPU memory.
struct FatalCudaErrorRecord {
    FatalCudaErrorDomain domain{FatalCudaErrorDomain::Unknown};
    int                  code{-1};  // numeric code, -1 when only text was available
    std::string          code_name;
    std::string          message;
    FatalCudaErrorSite   site{FatalCudaErrorSite::Unknown};
    std::string          source_file;
    int                  source_line{0};
    int                  device_index{-1};
    int                  rank{-1};
    std::string          stream;
    bool                 has_group_set_id{false};
    uint64_t             group_set_id{0};
    bool                 has_host_span{false};
    uintptr_t            host_base{0};
    uint64_t             host_bytes{0};
    const char*          direction{"none"};
    std::string          thread_id;
    std::vector<FatalCudaTileRecord> tiles;
    uint64_t             tile_total{0};
    uint64_t             bytes_total{0};
    bool                 tiles_truncated{false};
    bool                 metadata_truncated{false};
    // True when the classification relies on message text instead of a numeric
    // code (e.g. a PyTorch wrapper that lost the CUDA error code).
    bool                        low_confidence{false};
    int64_t                     wall_ms{0};
    int64_t                     mono_ms{0};
};

// Fatal whitelists. Ordinary OOM, invalid arguments and other recoverable
// errors must not enter the fatal collection path.
bool isFatalCudaRuntimeError(int cuda_error) noexcept;
bool isFatalCudaDriverError(int cu_result) noexcept;
// Structured reason first, message text second (low confidence when matched).
bool isFatalCudaException(const std::exception& exception) noexcept;

FatalCudaErrorRecord buildCudaRuntimeErrorRecord(int                  cuda_error,
                                                 FatalCudaErrorSite   site,
                                                 const char*          file,
                                                 int                  line,
                                                 int                  device_index) noexcept;
FatalCudaErrorRecord buildCudaDriverErrorRecord(int                cu_result,
                                                FatalCudaErrorSite site,
                                                const char*        file,
                                                int                line,
                                                int                device_index) noexcept;
FatalCudaErrorRecord buildCudaExceptionRecord(const std::exception& exception,
                                              FatalCudaErrorSite    site,
                                              const char*           file,
                                              int                   line) noexcept;

// Process-wide, first-wins. Returns true when this call claimed the incident.
// Later errors are dropped: the first record and the first deadline are kept.
bool recordFirstFatalCudaError(FatalCudaErrorRecord record) noexcept;

// Attach transfer identity to the stored record without overwriting an
// existing annotation. Used by the copy strategies that know the block ids.
void annotateFatalCudaTransferContext(uint64_t             group_set_id,
                                      bool                 device_to_host,
                                      uintptr_t            host_base,
                                      uint64_t             host_bytes) noexcept;

// True once an incident was recorded; the instance must not accept new work.
bool fatalCudacoreIncidentActive() noexcept;
// True while the bounded collection window is still open.
bool fatalCudacoreCollectionInProgress() noexcept;

// Snapshot of the stored record, for logging/manifest/tests.
bool fatalCudacoreErrorRecord(FatalCudaErrorRecord& out) noexcept;

// Install once in the CUDA copy module before faults occur. On an uncaught
// exception after a fatal CUDA error, give the collector its remaining bounded
// window, then delegate to the original terminate handler. No CUDA or locks.
void installCudacoreTerminateGuard() noexcept;

// Cold-path allocation provenance for checking first-error pointers against
// pool ranges without querying a poisoned CUDA context.
void recordCudacorePoolLifetime(
    const char* pool_name, uintptr_t base, uint64_t bytes, int device_index, bool initialized) noexcept;

enum class CudacoreTriggerStatus {
    NotAttempted,
    Sent,                    // write succeeded; NOT a dump-succeeded signal
    AlreadyObservedProgress, // an automatic dump had already started
    AlreadyAttemptedByPeer,  // another copy of this module already triggered
    PipeNotConfigured,
    PipeMissing,
    NoReader,
    PermissionDenied,
    BrokenPipe,
    PipeFull,
    IoTimeout,
    WriteFailed,
};

enum class CudacoreTerminalState {
    NotStarted,
    // The window ran to its end and a stable non-zero dump file was observed.
    // Observation is not validation: offline cuda-gdb loading stays pending.
    CollectionObservedDone,
    NoMechanismAvailable,  // dump disabled and no trigger/watch target
    DeadlineExceeded,
};

struct CudacoreCollectionOutcome {
    CudacoreTriggerStatus trigger{CudacoreTriggerStatus::NotAttempted};
    int                   trigger_errno{0};
    bool                  trigger_attempted{false};
    bool                  trigger_sent{false};
    std::string           dump_file_template;
    std::string           observed_file;
    uint64_t              observed_bytes{0};
    bool                  file_seen{false};
    bool                  file_size_stable{false};
    bool                  driver_progress_seen{false};
    CudacoreTerminalState terminal{CudacoreTerminalState::NotStarted};
    bool                  deadline_exceeded{false};
    int64_t               waited_ms{0};
    std::string           manifest_path;
    std::string           lease_path;
    std::string           diagnostics_dir;
    std::string           note;
};

// Starts the collection executor: a thread created before any fault that wakes up
// on the first fatal error and performs the trigger, the bounded observation and
// the evidence writes on its own. Collection therefore does not depend on the
// engine thread reaching a wait call. Idempotent, never throws. Production calls
// it from the startup audit; unit tests may call it directly.
void startCudacoreCollector() noexcept;

// Blocking entry point for every thread that must survive until the collection
// window closes. When the collector executor is not running, the first caller
// performs the collection itself. All callers share one deadline: it starts at
// the first fatal error and a later caller can never extend it.
CudacoreCollectionOutcome waitForCudacoreCollection() noexcept;

// Diagnostics directory: parent of CUDA_COREDUMP_FILE when that is a plain file
// template, otherwise <startup cwd>/cudacore_diagnostics. Empty on failure.
std::string cudacoreDiagnosticsDir() noexcept;

// Test-only seams. Production code must not call these.
namespace cudacore_test {
struct AttributeApi {
    // Explicit flag: an installed API with null getters means "symbols absent"
    // and must report Unsupported instead of falling back to the real driver.
    bool installed{false};
    int (*get_global)(int attrib, void* value, size_t* size){nullptr};
    int (*get)(int attrib, void* value, size_t* size){nullptr};
};
void installAttributeApi(const AttributeApi& api) noexcept;
void resetAttributeApi() noexcept;
void setDiagnosticsDirOverride(const std::string& dir) noexcept;
void resetDiagnosticsDirOverride() noexcept;
// Clears the first-error record, the deadline and the cross-process claims.
void resetIncidentState() noexcept;
void setStartupAttributeSnapshot(const CudacoreAttributeSnapshot& snapshot) noexcept;
// Shrinks the collection window (0 = fixed default) so tests do not wait 30s.
void setCollectionWindowMsForTest(int64_t window_ms) noexcept;
// Runs the FIFO user-trigger path directly, without the once-per-incident claim.
CudacoreTriggerStatus
triggerUserCoredumpForTest(const std::string& pipe_path, int64_t budget_ms, int* error_number) noexcept;
}  // namespace cudacore_test

}  // namespace rtp_llm
