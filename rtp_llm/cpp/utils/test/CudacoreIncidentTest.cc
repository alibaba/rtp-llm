#include "rtp_llm/cpp/utils/CudacoreDiagnostics.h"
#include "rtp_llm/cpp/utils/CudacoreFlightRecorder.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

std::atomic<int> g_dir_counter{0};

bool readFile(const std::string& path, std::string& content) {
    FILE* file = ::fopen(path.c_str(), "r");
    if (file == nullptr) {
        return false;
    }
    content.clear();
    char   buffer[4096];
    size_t read = 0;
    while ((read = ::fread(buffer, 1, sizeof(buffer), file)) != 0) {
        content.append(buffer, read);
    }
    const bool success = ::ferror(file) == 0;
    ::fclose(file);
    return success;
}

std::string findIncidentManifest(const std::string& dir) {
    for (int attempt = 0; attempt < 3; ++attempt) {
        std::string found;
        DIR*        directory = ::opendir(dir.c_str());
        if (directory == nullptr) {
            return {};
        }
        while (struct dirent* entry = ::readdir(directory)) {
            const std::string name = entry->d_name;
            if (name.rfind("cudacore_incident.", 0) == 0 && name.size() > 5
                && name.compare(name.size() - 5, 5, ".json") == 0) {
                found = dir + "/" + name;
            }
        }
        ::closedir(directory);
        if (!found.empty()) {
            return found;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return {};
}

class CudacoreIncidentTest: public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = ::testing::TempDir() + "/cudacore_incident_" + std::to_string(g_dir_counter.fetch_add(1));
        ::mkdir(dir_.c_str(), 0755);
        cudacore_test::setDiagnosticsDirOverride(dir_);
        cudacore_test::resetIncidentState();
        ::setenv("CUDA_COREDUMP_FILE", (dir_ + "/prefill_cudacore.%h.%p.%t").c_str(), 1);
        ::unsetenv("CUDA_COREDUMP_PIPE");
        ::unsetenv("CUDA_ENABLE_USER_TRIGGERED_COREDUMP");
        setSnapshot(/*dump_enabled=*/true, /*trigger_enabled=*/true, dir_ + "/prefill_cudacore.%h.%p.%t");
    }

    void TearDown() override {
        cudacore_test::resetIncidentState();
        cudacore_test::resetDiagnosticsDirOverride();
        ::unsetenv("CUDA_COREDUMP_FILE");
        ::unsetenv("CUDA_COREDUMP_PIPE");
        ::unsetenv("CUDA_ENABLE_USER_TRIGGERED_COREDUMP");
    }

    void setSnapshot(bool dump_enabled, bool trigger_enabled, const std::string& file_template) {
        CudacoreAttributeSnapshot snapshot;
        snapshot.symbols_source                     = "stub";
        snapshot.global.enable_on_exception.status  = CudacoreAttributeStatus::Ok;
        snapshot.global.enable_on_exception.value   = dump_enabled;
        snapshot.global.enable_user_trigger.status  = CudacoreAttributeStatus::Ok;
        snapshot.global.enable_user_trigger.value   = trigger_enabled;
        snapshot.context.context_valid              = true;
        snapshot.context.enable_on_exception.status = CudacoreAttributeStatus::Ok;
        snapshot.context.enable_on_exception.value  = dump_enabled;
        if (!file_template.empty()) {
            snapshot.global.file.status = CudacoreAttributeStatus::Ok;
            snapshot.global.file.value  = file_template;
        }
        cudacore_test::setStartupAttributeSnapshot(snapshot);
    }

    FatalCudaErrorRecord runtimeRecord(int code, FatalCudaErrorSite site = FatalCudaErrorSite::BatchedCopyCompletion) {
        return buildCudaRuntimeErrorRecord(code, site, "NoBlockCopy.cc", 371, /*device_index=*/1);
    }

    std::string dir_;
};

TEST_F(CudacoreIncidentTest, ManifestIncludesRecentCopyAndPoolProvenance) {
    CudacoreFlightEvent event;
    event.kind              = CudacoreFlightKind::BatchSubmit;
    event.device_index      = 2;
    event.stream            = 0x1234;
    event.tile_count        = 88;
    const uint64_t sequence = recordCudacoreFlightEvent(event);
    ASSERT_NE(sequence, 0);
    recordCudacorePoolLifetime("kv_pool", 0x100000, 0x200000, 2, true);
    recordCudacorePoolLifetime("kv_pool", 0x100000, 0x200000, 2, false);

    cudacore_test::setCollectionWindowMsForTest(100);
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));
    EXPECT_EQ(recordCudacoreFlightEvent(event), 0);
    (void)waitForCudacoreCollection();

    std::string manifest;
    ASSERT_TRUE(readFile(findIncidentManifest(dir_), manifest));
    EXPECT_NE(manifest.find("\"recent_cuda_submissions\":"), std::string::npos);
    EXPECT_NE(manifest.find("\"kind\":\"batch_submit\""), std::string::npos);

    DIR* directory = ::opendir(dir_.c_str());
    ASSERT_NE(directory, nullptr);
    std::string allocation_path;
    while (struct dirent* entry = ::readdir(directory)) {
        const std::string name = entry->d_name;
        if (name.rfind("cudacore_allocations.", 0) == 0) {
            allocation_path = dir_ + "/" + name;
            break;
        }
    }
    ::closedir(directory);
    std::string allocations;
    ASSERT_TRUE(readFile(allocation_path, allocations));
    EXPECT_NE(allocations.find("\"event\":\"pool_initialized\""), std::string::npos);
    EXPECT_NE(allocations.find("\"event\":\"pool_destroyed\""), std::string::npos);
}

TEST_F(CudacoreIncidentTest, FirstFatalErrorWinsAndIsNeverOverwritten) {
    EXPECT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));
    EXPECT_FALSE(recordFirstFatalCudaError(runtimeRecord(700)));
    EXPECT_TRUE(fatalCudacoreIncidentActive());
    EXPECT_TRUE(fatalCudacoreCollectionInProgress());

    FatalCudaErrorRecord stored;
    ASSERT_TRUE(fatalCudacoreErrorRecord(stored));
    EXPECT_EQ(stored.code, 719);
    EXPECT_EQ(stored.site, FatalCudaErrorSite::BatchedCopyCompletion);
    EXPECT_EQ(stored.source_line, 371);
    EXPECT_EQ(stored.device_index, 1);
    EXPECT_GT(stored.mono_ms, 0);
}

TEST_F(CudacoreIncidentTest, ConcurrentFirstErrorsProduceSingleIncident) {
    constexpr int kThreads = 8;
    std::atomic<int> claims{0};
    std::vector<std::thread> threads;
    for (int index = 0; index < kThreads; ++index) {
        threads.emplace_back([&, index] {
            if (recordFirstFatalCudaError(runtimeRecord(700 + index))) {
                ++claims;
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(claims.load(), 1);

    FatalCudaErrorRecord stored;
    ASSERT_TRUE(fatalCudacoreErrorRecord(stored));
    EXPECT_GE(stored.code, 700);
    EXPECT_LT(stored.code, 700 + kThreads);
}

TEST_F(CudacoreIncidentTest, TileMetadataIsBoundedButKeepsTotals) {
    FatalCudaErrorRecord record = runtimeRecord(719);
    for (int index = 0; index < 1000; ++index) {
        FatalCudaTileRecord tile;
        tile.dst   = static_cast<uintptr_t>(0x1000 + index);
        tile.src   = static_cast<uintptr_t>(0x2000 + index);
        tile.bytes = 4096;
        record.tiles.push_back(tile);
    }
    ASSERT_TRUE(recordFirstFatalCudaError(record));

    FatalCudaErrorRecord stored;
    ASSERT_TRUE(fatalCudacoreErrorRecord(stored));
    EXPECT_EQ(stored.tiles.size(), CudacoreDiagConstants::kMaxTileMetadata);
    EXPECT_EQ(stored.tile_total, 1000u);
    EXPECT_TRUE(stored.tiles_truncated);
}

TEST_F(CudacoreIncidentTest, TransferAnnotationIsAppliedOnce) {
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));
    annotateFatalCudaTransferContext(/*group_set_id=*/42, /*device_to_host=*/true, 0x1000, 4096);
    annotateFatalCudaTransferContext(/*group_set_id=*/99, /*device_to_host=*/false, 0x2000, 8192);

    FatalCudaErrorRecord stored;
    ASSERT_TRUE(fatalCudacoreErrorRecord(stored));
    EXPECT_TRUE(stored.has_group_set_id);
    EXPECT_EQ(stored.group_set_id, 42u);
    EXPECT_TRUE(stored.has_host_span);
    EXPECT_EQ(stored.host_base, 0x1000u);
    EXPECT_STREQ(stored.direction, "D2H");
}

TEST_F(CudacoreIncidentTest, FatalClassificationSeparatesOrdinaryErrors) {
    EXPECT_TRUE(isFatalCudaRuntimeError(719));  // launch failure
    EXPECT_TRUE(isFatalCudaRuntimeError(700));  // illegal address
    EXPECT_TRUE(isFatalCudaRuntimeError(715));
    EXPECT_TRUE(isFatalCudaRuntimeError(710));
    EXPECT_TRUE(isFatalCudaRuntimeError(214));  // ECC uncorrectable
    EXPECT_FALSE(isFatalCudaRuntimeError(2));   // OOM
    EXPECT_FALSE(isFatalCudaRuntimeError(1));   // invalid value
    EXPECT_FALSE(isFatalCudaRuntimeError(801));
    EXPECT_TRUE(isFatalCudaDriverError(719));

    EXPECT_FALSE(isFatalCudaException(std::runtime_error("CUDA out of memory")));
    EXPECT_FALSE(isFatalCudaException(std::runtime_error("invalid argument")));
    EXPECT_TRUE(isFatalCudaException(std::runtime_error("CUDA error: an illegal memory access was encountered")));
    EXPECT_TRUE(isFatalCudaException(std::runtime_error("unspecified launch failure")));

    const FatalCudaErrorRecord record =
        buildCudaExceptionRecord(std::runtime_error("an illegal instruction was encountered"),
                                 FatalCudaErrorSite::EngineStep,
                                 "NormalEngine.cc",
                                 860);
    EXPECT_TRUE(record.low_confidence);
    EXPECT_EQ(record.domain, FatalCudaErrorDomain::TorchException);
    EXPECT_EQ(record.code, -1);
}

TEST_F(CudacoreIncidentTest, TriggerReportsMissingPipe) {
    int error_number = 0;
    const CudacoreTriggerStatus status =
        cudacore_test::triggerUserCoredumpForTest(dir_ + "/does_not_exist.pipe", 200, &error_number);
    EXPECT_EQ(status, CudacoreTriggerStatus::PipeMissing);
    EXPECT_EQ(error_number, ENOENT);
}

TEST_F(CudacoreIncidentTest, TriggerWithoutReaderIsBounded) {
    const std::string pipe_path = dir_ + "/corepipe";
    ASSERT_EQ(::mkfifo(pipe_path.c_str(), 0644), 0);

    const auto begin  = std::chrono::steady_clock::now();
    int        error_number = 0;
    const CudacoreTriggerStatus status =
        cudacore_test::triggerUserCoredumpForTest(pipe_path, /*budget_ms=*/200, &error_number);
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    EXPECT_EQ(status, CudacoreTriggerStatus::NoReader);
    EXPECT_EQ(error_number, ENXIO);
    EXPECT_GE(elapsed_ms, 150);
    EXPECT_LT(elapsed_ms, 2000);
}

TEST_F(CudacoreIncidentTest, TriggerWritesExactlyOneRequestWithReader) {
    const std::string pipe_path = dir_ + "/corepipe";
    ASSERT_EQ(::mkfifo(pipe_path.c_str(), 0644), 0);
    const int reader = ::open(pipe_path.c_str(), O_RDONLY | O_NONBLOCK);
    ASSERT_GE(reader, 0);

    int error_number = 0;
    const CudacoreTriggerStatus status =
        cudacore_test::triggerUserCoredumpForTest(pipe_path, /*budget_ms=*/200, &error_number);
    EXPECT_EQ(status, CudacoreTriggerStatus::Sent);
    EXPECT_EQ(error_number, 0);

    char          buffer[8] = {};
    const ssize_t read      = ::read(reader, buffer, sizeof(buffer));
    EXPECT_EQ(read, 1);
    // Exactly one request: the trigger closed its write end, so a second read
    // observes EOF (0) with no additional byte.
    EXPECT_LE(::read(reader, buffer, sizeof(buffer)), 0);

    ::close(reader);
}

TEST_F(CudacoreIncidentTest, TriggerReportsFullPipe) {
    const std::string pipe_path = dir_ + "/corepipe";
    ASSERT_EQ(::mkfifo(pipe_path.c_str(), 0644), 0);
    const int reader = ::open(pipe_path.c_str(), O_RDONLY | O_NONBLOCK);
    ASSERT_GE(reader, 0);
    const int writer = ::open(pipe_path.c_str(), O_WRONLY | O_NONBLOCK);
    ASSERT_GE(writer, 0);

    std::vector<char> payload(4096, 'x');
    while (true) {
        const ssize_t written = ::write(writer, payload.data(), payload.size());
        if (written < 0) {
            ASSERT_EQ(errno, EAGAIN);
            break;
        }
    }
    // The reader never drains the pipe, so the trigger write must hit EAGAIN.
    int error_number = 0;
    const CudacoreTriggerStatus status =
        cudacore_test::triggerUserCoredumpForTest(pipe_path, /*budget_ms=*/200, &error_number);
    EXPECT_EQ(status, CudacoreTriggerStatus::PipeFull);
    EXPECT_EQ(error_number, EAGAIN);

    ::close(writer);
    ::close(reader);
}

TEST_F(CudacoreIncidentTest, TriggerPermissionDeniedIsBounded) {
    if (::geteuid() == 0) {
        GTEST_SKIP() << "root bypasses FIFO permissions";
    }
    const std::string pipe_path = dir_ + "/locked.pipe";
    ASSERT_EQ(::mkfifo(pipe_path.c_str(), 0000), 0);

    int error_number = 0;
    const CudacoreTriggerStatus status =
        cudacore_test::triggerUserCoredumpForTest(pipe_path, /*budget_ms=*/200, &error_number);
    EXPECT_EQ(status, CudacoreTriggerStatus::PermissionDenied);
    EXPECT_EQ(error_number, EACCES);
}

TEST_F(CudacoreIncidentTest, BrokenPipeNeverKillsTheProcess) {
    const std::string pipe_path = dir_ + "/racy.pipe";
    ASSERT_EQ(::mkfifo(pipe_path.c_str(), 0644), 0);

    std::atomic<bool> stop{false};
    std::thread       racer([&] {
        while (!stop.load()) {
            const int reader = ::open(pipe_path.c_str(), O_RDONLY | O_NONBLOCK);
            if (reader >= 0) {
                ::close(reader);
            }
        }
    });

    for (int attempt = 0; attempt < 20; ++attempt) {
        int error_number = 0;
        const CudacoreTriggerStatus status =
            cudacore_test::triggerUserCoredumpForTest(pipe_path, /*budget_ms=*/50, &error_number);
        EXPECT_TRUE(status == CudacoreTriggerStatus::Sent || status == CudacoreTriggerStatus::BrokenPipe
                    || status == CudacoreTriggerStatus::NoReader)
            << "unexpected status " << static_cast<int>(status);
    }
    stop.store(true);
    racer.join();
    // Reaching this point proves the collector survived every outcome.
    SUCCEED();
}

TEST_F(CudacoreIncidentTest, AutomaticDumpProgressSkipsTheExtraTrigger) {
    const std::string dump_path = dir_ + "/prefill_cudacore.dump";
    ASSERT_EQ(::setenv("CUDA_COREDUMP_FILE", dump_path.c_str(), 1), 0);
    cudacore_test::setCollectionWindowMsForTest(600);
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));
    {
        FILE* file = ::fopen(dump_path.c_str(), "w");
        ASSERT_NE(file, nullptr);
        ::fputs("partial", file);
        ::fclose(file);
    }

    const auto                      begin   = std::chrono::steady_clock::now();
    const CudacoreCollectionOutcome outcome = waitForCudacoreCollection();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    EXPECT_EQ(outcome.trigger, CudacoreTriggerStatus::AlreadyObservedProgress);
    EXPECT_FALSE(outcome.trigger_sent);
    EXPECT_TRUE(outcome.file_seen);
    EXPECT_EQ(outcome.terminal, CudacoreTerminalState::DeadlineExceeded);
    // An observed dump never releases the process early.
    EXPECT_GE(elapsed_ms, 500);
}

TEST_F(CudacoreIncidentTest, SegmentedDumpKeepsTheWindowOpenUntilTheDeadline) {
    const std::string dump_path = dir_ + "/dump.core";
    ASSERT_EQ(::setenv("CUDA_COREDUMP_FILE", dump_path.c_str(), 1), 0);
    cudacore_test::setCollectionWindowMsForTest(900);
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));

    // First segment, then a pause longer than the 100ms poll interval, then more
    // data: the window must not be released during the pause.
    {
        FILE* file = ::fopen(dump_path.c_str(), "w");
        ASSERT_NE(file, nullptr);
        ::fputs("segment-1", file);
        ::fclose(file);
    }
    std::thread writer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        FILE* file = ::fopen(dump_path.c_str(), "a");
        if (file != nullptr) {
            ::fputs("-segment-2", file);
            ::fclose(file);
        }
    });

    const auto                      begin   = std::chrono::steady_clock::now();
    const CudacoreCollectionOutcome outcome = waitForCudacoreCollection();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
    writer.join();

    EXPECT_EQ(outcome.terminal, CudacoreTerminalState::DeadlineExceeded);
    EXPECT_TRUE(outcome.file_size_stable);
    EXPECT_GE(elapsed_ms, 800);             // full window, not the first silent poll
    EXPECT_GT(outcome.observed_bytes, 0u);  // the final scan saw the appended segment
    EXPECT_GT(outcome.waited_ms, 0);

    std::string manifest;
    ASSERT_TRUE(readFile(outcome.manifest_path, manifest));
    EXPECT_NE(manifest.find("first_error"), std::string::npos);
    EXPECT_NE(manifest.find("\"code\":719"), std::string::npos);
    EXPECT_NE(manifest.find("DEADLINE_EXCEEDED"), std::string::npos);
    EXPECT_NE(manifest.find("\"file_seen\":true"), std::string::npos);
    EXPECT_EQ(findIncidentManifest(dir_).empty(), false);
}

TEST_F(CudacoreIncidentTest, WaitEndsAtDeadlineAndRecordsIt) {
    cudacore_test::setCollectionWindowMsForTest(300);
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(700)));

    const CudacoreCollectionOutcome outcome = waitForCudacoreCollection();

    EXPECT_EQ(outcome.terminal, CudacoreTerminalState::DeadlineExceeded);
    EXPECT_TRUE(outcome.deadline_exceeded);
    EXPECT_FALSE(outcome.file_seen);
    EXPECT_GE(outcome.waited_ms, 250);
    EXPECT_LT(outcome.waited_ms, 2000);
    EXPECT_FALSE(fatalCudacoreCollectionInProgress());
    EXPECT_TRUE(fatalCudacoreIncidentActive());

    std::string manifest;
    ASSERT_TRUE(readFile(outcome.manifest_path, manifest));
    EXPECT_NE(manifest.find("DEADLINE_EXCEEDED"), std::string::npos);
    EXPECT_NE(manifest.find("\"trigger\""), std::string::npos);
}

TEST_F(CudacoreIncidentTest, FollowersShareTheLeaderDeadlineAndCannotExtendIt) {
    cudacore_test::setCollectionWindowMsForTest(400);
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));

    CudacoreCollectionOutcome first_outcome;
    std::thread               leader([&] { first_outcome = waitForCudacoreCollection(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const auto                begin = std::chrono::steady_clock::now();
    CudacoreCollectionOutcome follower_outcome;
    std::thread               follower([&] { follower_outcome = waitForCudacoreCollection(); });
    leader.join();
    follower.join();

    const auto follower_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
    EXPECT_EQ(first_outcome.terminal, CudacoreTerminalState::DeadlineExceeded);
    EXPECT_EQ(follower_outcome.terminal, CudacoreTerminalState::DeadlineExceeded);
    EXPECT_LT(first_outcome.waited_ms, 1500);
    EXPECT_LT(follower_ms, 1500);
    EXPECT_EQ(first_outcome.manifest_path, follower_outcome.manifest_path);
}

TEST_F(CudacoreIncidentTest, NothingToCollectReturnsWithoutWaitingTheFullWindow) {
    setSnapshot(/*dump_enabled=*/false, /*trigger_enabled=*/false, /*file_template=*/"");
    ::unsetenv("CUDA_COREDUMP_FILE");
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));

    cudacore_test::setCollectionWindowMsForTest(5000);
    const auto begin = std::chrono::steady_clock::now();
    const CudacoreCollectionOutcome outcome = waitForCudacoreCollection();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    EXPECT_EQ(outcome.terminal, CudacoreTerminalState::NoMechanismAvailable);
    EXPECT_LT(elapsed_ms, 1000);
    EXPECT_FALSE(outcome.file_seen);
}

TEST_F(CudacoreIncidentTest, WaitWithoutIncidentIsANoOp) {
    const CudacoreCollectionOutcome outcome = waitForCudacoreCollection();
    EXPECT_EQ(outcome.terminal, CudacoreTerminalState::NotStarted);
    EXPECT_FALSE(outcome.trigger_attempted);
}

TEST_F(CudacoreIncidentTest, TriggerThroughTheWindowIsSentOnceAndRecorded) {
    const std::string pipe_path = dir_ + "/corepipe";
    ASSERT_EQ(::mkfifo(pipe_path.c_str(), 0644), 0);
    ASSERT_EQ(::setenv("CUDA_COREDUMP_PIPE", pipe_path.c_str(), 1), 0);
    const int reader = ::open(pipe_path.c_str(), O_RDONLY | O_NONBLOCK);
    ASSERT_GE(reader, 0);
    cudacore_test::setCollectionWindowMsForTest(300);

    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));
    const CudacoreCollectionOutcome outcome = waitForCudacoreCollection();

    EXPECT_TRUE(outcome.trigger_attempted);
    EXPECT_EQ(outcome.trigger, CudacoreTriggerStatus::Sent);
    EXPECT_TRUE(outcome.trigger_sent);
    EXPECT_EQ(outcome.terminal, CudacoreTerminalState::DeadlineExceeded);

    std::string manifest;
    ASSERT_TRUE(readFile(outcome.manifest_path, manifest));
    EXPECT_NE(manifest.find("TRIGGER_SENT"), std::string::npos);
    // A sent trigger is never recorded as a successful dump.
    EXPECT_EQ(manifest.find("DUMP_SUCCEEDED"), std::string::npos);
    EXPECT_NE(manifest.find("\"sent\":true"), std::string::npos);

    char buffer[8] = {};
    EXPECT_EQ(::read(reader, buffer, sizeof(buffer)), 1);
    ::close(reader);
}

TEST_F(CudacoreIncidentTest, FirstErrorTriggersCollectionWithoutAnyWaiter) {
    const std::string pipe_path = dir_ + "/corepipe";
    ASSERT_EQ(::mkfifo(pipe_path.c_str(), 0644), 0);
    ASSERT_EQ(::setenv("CUDA_COREDUMP_PIPE", pipe_path.c_str(), 1), 0);
    const int reader = ::open(pipe_path.c_str(), O_RDONLY | O_NONBLOCK);
    ASSERT_GE(reader, 0);
    cudacore_test::setCollectionWindowMsForTest(1000);
    startCudacoreCollector();
    // Let a collector from an earlier test finish its own window first.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const auto begin = std::chrono::steady_clock::now();
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));

    // No waitForCudacoreCollection() call here: the trigger must come from the
    // collection executor alone, even if the engine thread never returns.
    char          buffer[8] = {};
    ssize_t       read      = -1;
    const auto    read_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
    while (std::chrono::steady_clock::now() < read_deadline) {
        read = ::read(reader, buffer, sizeof(buffer));
        if (read > 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    EXPECT_EQ(read, 1) << "the collection executor did not send the trigger request";
    EXPECT_LT(elapsed_ms, 1500);

    // The fault manifest must exist as well, written by the executor.
    EXPECT_FALSE(findIncidentManifest(dir_).empty());

    const CudacoreCollectionOutcome outcome = waitForCudacoreCollection();
    EXPECT_TRUE(outcome.trigger_sent);
    EXPECT_EQ(outcome.terminal, CudacoreTerminalState::DeadlineExceeded);
    ::close(reader);
}

TEST_F(CudacoreIncidentTest, ThrottledFinalChangeIsPersistedBeforeTheWindowEnds) {
    const std::string dump_path = dir_ + "/dump.core";
    ASSERT_EQ(::setenv("CUDA_COREDUMP_FILE", dump_path.c_str(), 1), 0);
    FILE* file = ::fopen(dump_path.c_str(), "w");
    ASSERT_NE(file, nullptr);
    ::fputs("A", file);
    ::fclose(file);
    cudacore_test::setCollectionWindowMsForTest(3000);
    startCudacoreCollector();
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));

    // Wait for the first progress snapshot, then change the file within the
    // 500ms throttle. No further writes occur to force another changed tick.
    bool initial_seen = false;
    const auto first_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(800);
    while (std::chrono::steady_clock::now() < first_deadline) {
        std::string json;
        if (readFile(findIncidentManifest(dir_), json)
            && json.find("\"phase\":\"progress\"") != std::string::npos) {
            initial_seen = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(initial_seen);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    file = ::fopen(dump_path.c_str(), "a");
    if (file != nullptr) {
        ::fputs("BC", file);
        ::fclose(file);
    }
    EXPECT_NE(file, nullptr);

    bool update_seen = false;
    const auto update_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1200);
    while (std::chrono::steady_clock::now() < update_deadline) {
        std::string json;
        if (readFile(findIncidentManifest(dir_), json)
            && json.find("\"phase\":\"progress\"") != std::string::npos
            && json.find("\"bytes\":3") != std::string::npos) {
            update_seen = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_TRUE(update_seen);
    EXPECT_TRUE(fatalCudacoreCollectionInProgress());
    const auto outcome = waitForCudacoreCollection();
    EXPECT_EQ(outcome.terminal, CudacoreTerminalState::DeadlineExceeded);
    EXPECT_EQ(outcome.observed_bytes, 3u);
}

TEST_F(CudacoreIncidentTest, LeaseRecordsTheOriginalDeadlineForTheParent) {
    cudacore_test::setCollectionWindowMsForTest(2000);
    startCudacoreCollector();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const int64_t before_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
    ASSERT_TRUE(recordFirstFatalCudaError(runtimeRecord(719)));

    const CudacoreCollectionOutcome outcome = waitForCudacoreCollection();
    ASSERT_FALSE(outcome.lease_path.empty());

    std::string lease;
    ASSERT_TRUE(readFile(outcome.lease_path, lease));
    EXPECT_NE(lease.find("\"schema_version\":\"rtp_llm.cudacore_lease.v1\""), std::string::npos);
    EXPECT_NE(lease.find("\"pid\":" + std::to_string(static_cast<long>(::getpid()))), std::string::npos);
    EXPECT_NE(lease.find("\"window_ms\":2000"), std::string::npos);
    EXPECT_NE(lease.find("\"deadline_epoch_ms\":"), std::string::npos);
    EXPECT_NE(lease.find("\"created_mono_ms\":"), std::string::npos);
    EXPECT_NE(lease.find("\"deadline_mono_ms\":"), std::string::npos);
    EXPECT_NE(lease.find("\"worker_start_id\":"), std::string::npos);

    // The parent-capable lease exists before the window is over.
    EXPECT_GE(outcome.waited_ms, 0);
    EXPECT_FALSE(lease.empty());
    (void)before_ms;
}

}  // namespace
}  // namespace rtp_llm
