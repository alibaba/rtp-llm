#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ftw.h>
#include <future>
#include <thread>
#include <unistd.h>
#include <gtest/gtest.h>

#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"
#include "autil/legacy/json.h"
#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/core/MetricsCollector.h"

namespace rtp_llm::test {
namespace {
namespace fs   = std::filesystem;
using Manifest = std::map<std::string, std::string>;
using Stage    = CrcBlockCopyResult::FailureStage;

uint32_t referenceCrc(const uint8_t* data, size_t bytes) {
    uint32_t crc = ~uint32_t(0);
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ ((crc & 1) ? 0x82f63b78U : 0);
    }
    return ~crc;
}

TEST(KVCacheCrcMetricsTest, RankOneWithoutOrdinaryReporterEmitsTaggedCrcMetrics) {
    // Manual snapshots prevent the background sampler from consuming these QPS values.
    autil::EnvGuard manual("kmonitorManuallyMode", "true");
    autil::EnvGuard log_sink("kmonitorEnableLogFileSink", "false");
    ASSERT_FALSE(kmonitor::KMonitorFactory::IsStarted());
    // Reporters must die before Shutdown(), which deletes their KMonitor objects.
    auto        shutdown = std::shared_ptr<void>(nullptr, [](void*) { kmonitor::KMonitorFactory::Shutdown(); });
    CacheConfig cache;
    cache.layer_all_num = 0;
    KVCacheConfig     kv_cache_config;
    ParallelismConfig parallel;
    parallel.world_rank = 1;
    parallel.tp_rank    = 1;
    parallel.dp_rank    = 7;
    auto connector      = std::make_unique<KVCacheMemoryConnector>(
        cache, kv_cache_config, parallel, nullptr, std::vector<std::string>{}, nullptr);

    // Exercise worker initialization directly; no GPU/workspace initialization is needed.
    connector->initCrcMetrics();
    ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    ASSERT_NE(connector->crc_metrics_reporter_, nullptr);
    EXPECT_EQ(connector->metrics_reporter_, nullptr);
    EXPECT_EQ(connector->metrics_reporter_thread_, nullptr);
    connector->reportCrcMetrics(true, 4096, 10, 1, 0);
    connector->reportCrcMetrics(true, 0, 0, 0, 0, 1);
    connector->reportReadMetrics(true, 10, 1, 1);

    kmonitor::MetricsCollector snapshot;
    connector->crc_metrics_reporter_->_monitor->GetMetrics(
        &snapshot, {kmonitor::NORMAL}, autil::TimeUtility::currentTimeInMilliSeconds());
    int failed_series  = 0;
    int written_series = 0;
    for (const auto* record : snapshot.GetRecords().getRecords()) {
        for (const auto* value : record->Values()) {
            EXPECT_EQ(value->Name().find("rtp_llm_kv_cache_memory_cache_read_"), std::string::npos);
            if (value->Name().find("rtp_llm_kv_cache_crc_") == std::string::npos)
                continue;
            EXPECT_EQ(record->Tags()->FindTag("rank"), "1");
            EXPECT_EQ(record->Tags()->FindTag("dp_rank"), "7");
            EXPECT_EQ(record->Tags()->FindTag("copy_direction"), "TO_GPU");
            if (value->Name().find("rtp_llm_kv_cache_crc_failed_qps") != std::string::npos) {
                ++failed_series;
                EXPECT_GT(std::stod(value->Value()), 0);
            }
            if (value->Name().find("rtp_llm_kv_cache_crc_dump_written_qps") != std::string::npos) {
                ++written_series;
                EXPECT_GT(std::stod(value->Value()), 0);
            }
        }
    }
    EXPECT_EQ(failed_series, 1);
    EXPECT_EQ(written_series, 1);

    // TP0 keeps using its supplied reporter rather than creating another monitor.
    kmonitor::MetricsTags tags("dp_rank", "7");
    auto                  ordinary = std::make_shared<kmonitor::MetricsReporter>("", "", tags);
    parallel.world_rank            = 0;
    parallel.tp_rank               = 0;
    KVCacheMemoryConnector leader(cache, kv_cache_config, parallel, nullptr, {}, ordinary);
    leader.initCrcMetrics();
    EXPECT_EQ(leader.metrics_reporter_, ordinary);
    EXPECT_EQ(leader.crc_metrics_reporter_, ordinary);

    // A handled CRC-route disk failure must retain the ordinary disk copy metrics.
    kv_cache_config.enable_memory_cache_disk = true;
    leader.crc_enabled_                      = true;
    for (auto direction : {MemoryOperationRequestPB::H2D, MemoryOperationRequestPB::D2H}) {
        MemoryOperationRequestPB request;
        request.set_copy_direction(direction);
        request.add_copy_items()->set_backing_type(MemoryOperationRequestPB::DISK);
        MemoryOperationResponsePB response;
        EXPECT_TRUE(leader.copyCache(request, response));
        EXPECT_FALSE(response.success());  // No workspace/backing is initialized in this metrics fixture.
    }
    kmonitor::MetricsCollector disk_snapshot;
    ordinary->_monitor->GetMetrics(&disk_snapshot, {kmonitor::NORMAL}, autil::TimeUtility::currentTimeInMilliSeconds());
    int disk_series = 0;
    for (const auto* record : disk_snapshot.GetRecords().getRecords()) {
        for (const auto* value : record->Values()) {
            if (value->Name().find("rtp_llm_kv_cache_disk_cache_copy_") == std::string::npos)
                continue;
            const auto direction = record->Tags()->FindTag("copy_direction");
            EXPECT_TRUE(direction == "TO_GPU" || direction == "FROM_GPU");
            if (value->Name().find("qps") != std::string::npos)
                EXPECT_GT(std::stod(value->Value()), 0);
            else
                EXPECT_GE(std::stod(value->Value()), 0);
            ++disk_series;
        }
    }
    EXPECT_EQ(disk_series, 6);  // QPS, failures and latency for each direction.
}

class KVCacheCrcDiagnosticsTest: public ::testing::Test {
protected:
    void SetUp() override {
        char path[] = "/tmp/kv_crc_diagnostic_XXXXXX";
        ASSERT_NE(::mkdtemp(path), nullptr);
        path_                = path;
        cache_.layer_all_num = 0;
        ParallelismConfig parallel;
        parallel.world_rank = 3;
        connector_          = std::make_unique<KVCacheMemoryConnector>(
            cache_, kv_cache_config_, parallel, nullptr, std::vector<std::string>{}, nullptr);
        connector_->crc_dump_path_ = path_;
        connector_->crc_dump_pool_ = std::make_shared<autil::LockFreeThreadPool>(1, 2, nullptr, "CrcDiagnosticTest");
        ASSERT_TRUE(connector_->crc_dump_pool_->start());
        connector_->crc_layout_ = 47;
        request_.set_copy_direction(MemoryOperationRequestPB::H2D);
        request_.set_trace_id("diagnostic-test");
        auto* item = request_.add_copy_items();
        item->set_cache_key(17);
        request_.set_request_id(31);
    }

    bool waitForDump() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lock(connector_->crc_mutex_);
                if (connector_->crc_dump_pending_ == 0)
                    return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    }

    void TearDown() override {
        if (connector_) {
            EXPECT_TRUE(waitForDump());
            connector_.reset();
        }
        if (!path_.empty()) {
            EXPECT_EQ(::nftw(
                          path_.c_str(),
                          [](const char* path, const struct stat*, int type, struct FTW*) {
                              return type == FTW_DP ? ::rmdir(path) : ::unlink(path);
                          },
                          16,
                          FTW_DEPTH | FTW_PHYS),
                      0);
        }
    }

    std::vector<uint8_t> block(const std::vector<uint8_t>& payload, int64_t writer = 23) {
        CrcBlockFooter footer;
        footer.crc32c = referenceCrc(payload.data(), payload.size());
        std::vector<uint8_t> result(CrcBlockCopy::storageBytes(payload.size()), 0xa5);
        std::memcpy(result.data(), payload.data(), payload.size());
        std::memcpy(result.data() + CrcBlockCopy::footerOffset(payload.size()), &footer, sizeof(footer));
        std::memcpy(result.data() + CrcBlockCopy::transferBytes(payload.size()), &writer, sizeof(writer));
        return result;
    }

    CrcBlockCopyResult sourceFailure(const std::vector<uint8_t>& cpu, const std::vector<uint8_t>& gpu) {
        CrcBlockCopyResult result;
        result.failure_stage    = Stage::SOURCE_CRC;
        result.gpu_crc_observed = true;
        CrcBlockFooter footer;
        std::memcpy(&footer, cpu.data() + CrcBlockCopy::footerOffset(gpu.size()), sizeof(footer));
        result.checked_footer   = footer;
        result.expected_crc     = footer.crc32c;
        result.actual_crc       = referenceCrc(gpu.data(), gpu.size());
        result.staging_snapshot = gpu;
        result.staging_footer   = footer;
        return result;
    }

    void dump(CrcBlockCopyResult          result,
              const std::vector<uint8_t>& cpu,
              size_t                      bytes,
              const std::vector<uint8_t>* inherited = nullptr) {
        ASSERT_TRUE(connector_->reserveCrcDump());
        connector_->dumpCrcFailure(request_,
                                   0,
                                   CacheBlockKind::COMPLETE,
                                   std::move(result),
                                   cpu.data(),
                                   bytes,
                                   inherited ? inherited->data() : nullptr);
    }

    fs::path dumpDirectory() {
        for (const auto& entry : fs::directory_iterator(path_))
            if (fs::exists(entry.path() / "manifest.json"))
                return entry.path();
        ADD_FAILURE() << "no completed dump";
        return path_;
    }

    std::vector<uint8_t> read(const char* name) {
        std::ifstream input(dumpDirectory() / name, std::ios::binary);
        EXPECT_TRUE(input.good()) << name;
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }

    Manifest manifest() {
        const auto json = read("manifest.json");
        Manifest   result;
        autil::legacy::FromJsonString(result, std::string(json.begin(), json.end()));
        return result;
    }

    const std::vector<uint8_t>              payload_{'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    std::string                             path_;
    CacheConfig                             cache_;
    KVCacheConfig                           kv_cache_config_;
    std::unique_ptr<KVCacheMemoryConnector> connector_;
    MemoryOperationRequestPB                request_;
};

TEST_F(KVCacheCrcDiagnosticsTest, OwnedFullCpuSnapshotSurvivesBackingReuse) {
    auto cpu = block(payload_);
    cpu[0] ^= 1;
    const auto           original = cpu;
    std::vector<uint8_t> gpu(cpu.begin(), cpu.begin() + payload_.size());
    auto                 result = sourceFailure(cpu, gpu);
    // Hold the worker until the live backing has been reused. Release on early assertion failure too.
    auto gate    = std::make_shared<std::promise<void>>();
    auto ready   = gate->get_future().share();
    auto release = std::shared_ptr<void>(nullptr, [gate](void*) { gate->set_value(); });
    ASSERT_EQ(connector_->crc_dump_pool_->pushTask([ready] { ready.wait(); }), autil::ThreadPoolBase::ERROR_NONE);
    dump(std::move(result), cpu, payload_.size());
    std::fill(cpu.begin(), cpu.end(), 0);
    release.reset();
    ASSERT_TRUE(waitForDump());
    EXPECT_EQ(read("cpu.bin"), original);
    EXPECT_EQ(read("gpu_staging.bin"), gpu);
    auto info = manifest();
    EXPECT_EQ(info["checked_footer_crc32c"], std::to_string(0xe3069283U));
    EXPECT_EQ(info["cpu_crc32c"], std::to_string(referenceCrc(gpu.data(), gpu.size())));
    EXPECT_EQ(info["diagnosis"], "cpu_gpu_payload_agree_checked_crc_mismatch");
    EXPECT_EQ(info["rank"], "3");
    EXPECT_EQ(info["storage_bytes"], std::to_string(original.size()));
    EXPECT_EQ(info["truncated"], "false");
    EXPECT_FALSE(info["temporal_limit"].empty());
}

TEST_F(KVCacheCrcDiagnosticsTest, ValidCpuAndDifferentGpuDoesNotClaimTransferCausality) {
    auto cpu = block(payload_);
    auto gpu = payload_;
    gpu.back() ^= 1;
    dump(sourceFailure(cpu, gpu), cpu, payload_.size());
    ASSERT_TRUE(waitForDump());
    auto info = manifest();
    EXPECT_EQ(info["cpu_crc32c"], std::to_string(0xe3069283U));
    EXPECT_EQ(info["diagnosis"], "cpu_snapshot_matches_checked_crc_gpu_differs");
    EXPECT_EQ(info["payload_different_bytes"], "1");
    EXPECT_EQ(info["payload_first_difference"], "8");
    EXPECT_EQ(info["gpu_snapshot_role"], "checked_source_before_overlay");
}

TEST_F(KVCacheCrcDiagnosticsTest, StoredCrcFailureKeepsHostIdentityAndWriterProvenance) {
    cache_.layer_all_num         = 1;
    cache_.kv_block_stride_bytes = payload_.size();
    request_.mutable_copy_items(0)->set_is_complete(true);
    request_.mutable_copy_items(0)->add_gpu_blocks(11);
    auto cpu = block(payload_);
    cpu[CrcBlockCopy::footerOffset(payload_.size())] ^= 1;
    dump(sourceFailure(cpu, payload_), cpu, payload_.size());
    ASSERT_TRUE(waitForDump());
    auto info = manifest();
    EXPECT_EQ(info["diagnosis"], "cpu_gpu_payload_agree_checked_crc_mismatch");
    EXPECT_EQ(info["cache_key"], "17");
    EXPECT_EQ(info["writer_request_id"], "23");
    EXPECT_EQ(info["reader_request_id"], "31");
    EXPECT_EQ(info["layout"], "47");
    EXPECT_EQ(info["snapshot_format_version"], "3");
    EXPECT_EQ(info["slot_0_layer"], "0");
    EXPECT_EQ(info["slot_0_payload_offset"], "0");
    EXPECT_EQ(info["slot_0_stride_bytes"], "9");
    EXPECT_EQ(read("cpu.bin"), cpu);
}

TEST_F(KVCacheCrcDiagnosticsTest, FailedGpuCrcStatusIsNotClassifiedAsCpuOrTransferCorruption) {
    auto cpu              = block(payload_);
    auto result           = sourceFailure(cpu, payload_);
    result.gpu_crc_status = 7;
    dump(std::move(result), cpu, payload_.size());
    ASSERT_TRUE(waitForDump());
    auto info = manifest();
    EXPECT_EQ(info["diagnosis"], "gpu_crc_status_failed");
    EXPECT_EQ(info["gpu_crc_status"], "7");
    EXPECT_EQ(info["payload_different_bytes"], "0");
}

TEST_F(KVCacheCrcDiagnosticsTest, GpuCrcObservationDisagreementWithSnapshotRemainsExplicit) {
    auto cpu    = block(payload_);
    auto result = sourceFailure(cpu, payload_);
    result.actual_crc ^= 1;
    dump(std::move(result), cpu, payload_.size());
    ASSERT_TRUE(waitForDump());
    auto info = manifest();
    EXPECT_EQ(info["diagnosis"], "gpu_crc_observation_disagrees_with_snapshot");
    EXPECT_EQ(info["gpu_snapshot_cpu_crc32c"], std::to_string(0xe3069283U));
    EXPECT_EQ(info["payload_different_bytes"], "0");
}

TEST_F(KVCacheCrcDiagnosticsTest, ChangedCpuFooterIsReportedAlongsideEarlierCheckedFooter) {
    auto cpu = block(payload_);
    auto gpu = payload_;
    gpu.back() ^= 1;
    auto result = sourceFailure(cpu, gpu);
    cpu[CrcBlockCopy::footerOffset(payload_.size())] ^= 1;
    dump(std::move(result), cpu, payload_.size());
    ASSERT_TRUE(waitForDump());
    auto info = manifest();
    EXPECT_EQ(info["cpu_footer_differs_from_gpu_checked_footer"], "true");
    EXPECT_NE(info["checked_footer_crc32c"], info["cpu_footer_crc32c"]);
    EXPECT_EQ(info["diagnosis"], "cpu_snapshot_matches_checked_crc_gpu_differs");
}

TEST_F(KVCacheCrcDiagnosticsTest, DifferentGpuFooterPreservesHealthyCpuEvidence) {
    const auto cpu    = block(payload_);
    auto       result = sourceFailure(cpu, payload_);
    result.checked_footer->crc32c ^= 1;
    result.expected_crc   = result.checked_footer->crc32c;
    result.staging_footer = result.checked_footer;
    dump(std::move(result), cpu, payload_.size());
    ASSERT_TRUE(waitForDump());
    auto info = manifest();
    EXPECT_EQ(info["cpu_matches_snapshot_footer_crc"], "true");
    EXPECT_EQ(info["cpu_matches_checked_crc"], "false");
    EXPECT_EQ(info["cpu_footer_differs_from_gpu_checked_footer"], "true");
    EXPECT_EQ(info["payload_different_bytes"], "0");
    EXPECT_NE(info["expected_crc32c"], info["cpu_footer_crc32c"]);
    EXPECT_EQ(read("cpu.bin"), cpu);
}

TEST_F(KVCacheCrcDiagnosticsTest, FreshD2hGpuCrcStatusFailureHasOutputRole) {
    request_.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto output             = block(payload_);
    auto result             = sourceFailure(output, payload_);
    result.failure_stage    = Stage::CRC_COMPUTE;
    result.output_written   = true;
    result.gpu_crc_observed = false;
    result.checked_footer.reset();
    result.gpu_crc_status = 7;
    dump(std::move(result), output, payload_.size());
    ASSERT_TRUE(waitForDump());
    auto info = manifest();
    EXPECT_EQ(info["diagnosis"], "gpu_crc_status_failed");
    EXPECT_EQ(info["gpu_crc_role"], "output");
    EXPECT_EQ(info["expected_crc32c"], "unavailable");
    EXPECT_EQ(info["actual_crc32c"], "unavailable");
    EXPECT_EQ(info["compared_cpu_role"], "candidate_output");
}

TEST_F(KVCacheCrcDiagnosticsTest, InheritedFailureCapturesSourceBeforeCandidateIsWritten) {
    request_.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto source = block(payload_, 19);
    source[0] ^= 1;
    std::vector<uint8_t> gpu(source.begin(), source.begin() + payload_.size());
    auto                 output = block(payload_, 23);
    dump(sourceFailure(source, gpu), output, payload_.size(), &source);
    ASSERT_TRUE(waitForDump());
    EXPECT_EQ(read("cpu.bin"), source);
    EXPECT_EQ(read("gpu_staging.bin"), gpu);
    EXPECT_FALSE(fs::exists(dumpDirectory() / "cpu_output.bin"));
    auto info = manifest();
    EXPECT_EQ(info["writer_request_id"], "19");
    EXPECT_EQ(info["output_written"], "false");
    EXPECT_EQ(info["gpu_snapshot_role"], "checked_source_before_overlay");
    EXPECT_EQ(info["gpu_footer_role"], "source");
    EXPECT_EQ(info["compared_cpu_role"], "source");
}

TEST_F(KVCacheCrcDiagnosticsTest, OutputFailureDoesNotPresentNewPayloadAsInheritedSource) {
    request_.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto source = block(payload_, 19);
    auto gpu    = payload_;
    gpu[0] ^= 1;
    auto output             = block(gpu);
    auto result             = sourceFailure(source, payload_);
    result.failure_stage    = Stage::CRC_COMPUTE;
    result.output_written   = true;
    result.staging_snapshot = gpu;
    result.gpu_crc_status   = 7;
    result.gpu_crc_observed = false;
    result.checked_footer.reset();
    dump(std::move(result), output, payload_.size(), &source);
    ASSERT_TRUE(waitForDump());
    auto info = manifest();
    EXPECT_EQ(info["diagnosis"], "gpu_crc_status_failed");
    EXPECT_EQ(info["gpu_snapshot_role"], "candidate_output_after_overlay");
    EXPECT_EQ(info["gpu_crc_role"], "output");
    EXPECT_EQ(info["compared_cpu_role"], "candidate_output");
    EXPECT_EQ(info["original_gpu_source_available"], "false");
    EXPECT_EQ(info["payload_different_bytes"], "0");
    EXPECT_EQ(read("cpu.bin"), source);
    EXPECT_EQ(read("cpu_output.bin"), output);
    EXPECT_EQ(info["writer_request_id"], "19");
    EXPECT_EQ(info["output_writer_request_id"], "23");
}

TEST_F(KVCacheCrcDiagnosticsTest, FullBlockBeyondEightMiBIncludesTailAndFooter) {
    std::vector<uint8_t> payload(8 * 1024 * 1024 + 31, 0x5a);
    auto                 cpu = block(payload);
    payload.back() ^= 1;
    dump(sourceFailure(cpu, payload), cpu, payload.size());
    ASSERT_TRUE(waitForDump());
    EXPECT_EQ(read("cpu.bin"), cpu);
    EXPECT_EQ(read("gpu_staging.bin"), payload);
    auto info = manifest();
    EXPECT_EQ(info["payload_different_bytes"], "1");
    EXPECT_EQ(info["payload_first_difference"], std::to_string(payload.size() - 1));
    EXPECT_EQ(info["captured_bytes"], std::to_string(cpu.size()));
    EXPECT_EQ(info["cpu_crc32c"], std::to_string(referenceCrc(cpu.data(), payload.size())));
}

TEST_F(KVCacheCrcDiagnosticsTest, OversizedFullDumpRejectedBeforeDereferencingBacking) {
    EXPECT_THROW(connector_->dumpCrcFailure(
                     request_, 0, CacheBlockKind::COMPLETE, {}, nullptr, 2ULL * 1024 * 1024 * 1024, nullptr),
                 std::length_error);
    EXPECT_TRUE(fs::is_empty(path_));
}
}  // namespace
}  // namespace rtp_llm::test
