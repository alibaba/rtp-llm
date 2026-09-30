#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"

#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rtp_llm {
namespace {
namespace fs = std::filesystem;

std::vector<uint8_t> readBytes(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
std::string readText(const fs::path& path) {
    const auto bytes = readBytes(path);
    return {bytes.begin(), bytes.end()};
}

class CrcDumpWriterTest: public ::testing::Test {
protected:
    void SetUp() override {
        const char* test_tmp = std::getenv("TEST_TMPDIR");
        std::string pattern  = std::string(test_tmp ? test_tmp : "/tmp") + "/crc_dump_test_XXXXXX";
        ASSERT_NE(mkdtemp(pattern.data()), nullptr);
        root_ = pattern;
    }
    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(root_, ignored);
    }
    static CrcCopyFailure failure() {
        CrcCopyFailure result{0, 0x12345678, 0xe3069283, ""};
        result.cpu_snapshot.resize(16);
        std::memcpy(result.cpu_snapshot.data(), "123456789", 9);
        std::memcpy(result.cpu_snapshot.data() + 12, &result.expected, 4);
        result.staging_snapshot = result.cpu_snapshot;
        return result;
    }
    static CrcCopyItem item() {
        return {nullptr, 9, 16, {{reinterpret_cast<void*>(0x1234), 0, 9}}};
    }
    fs::path root_;
};

TEST_F(CrcDumpWriterTest, SavesCheckedGpuBytesAndLaterCpuMutationSeparately) {
    CrcDumpWriter writer(7, root_.string());
    auto          observed = failure();
    observed.cpu_snapshot[0] ^= 1;
    observed.cpu_snapshot[12] ^= 1;
    auto reservation = writer.reserve(16);
    ASSERT_NE(reservation, nullptr);
    const auto path = writer.write(*reservation,
                                   item(),
                                   observed,
                                   {{"kind", "SWA"},
                                    {"member_0_tag", "state\"swa\n"},
                                    {"layout_tile_0_layer_id", "21"},
                                    {"direction", "DISK->DEVICE"}});
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(readBytes(fs::path(path) / "cpu.bin"), observed.cpu_snapshot);
    EXPECT_EQ(readBytes(fs::path(path) / "gpu_staging.bin"),
              std::vector<uint8_t>(observed.staging_snapshot.begin(), observed.staging_snapshot.begin() + 9));
    EXPECT_EQ(readBytes(fs::path(path) / "checked_footer.bin"),
              std::vector<uint8_t>(observed.staging_snapshot.begin() + 12, observed.staging_snapshot.end()));
    EXPECT_EQ(readBytes(fs::path(path) / "gpu_staging_footer.bin"), readBytes(fs::path(path) / "checked_footer.bin"));
    EXPECT_NE(readBytes(fs::path(path) / "cpu_footer.bin"), readBytes(fs::path(path) / "checked_footer.bin"));
    const auto manifest = readText(fs::path(path) / "manifest.json");
    for (const auto* field : {"\"rank\": \"7\"",
                              "\"kind\": \"SWA\"",
                              "\"layout_tile_0_layer_id\": \"21\"",
                              "\"member_0_tag\": \"state\\\"swa\\u000a\"",
                              "\"gpu_snapshot_cpu_crc32c\": \"3808858755\"",
                              "\"cpu_gpu_payload_differing_bytes\": \"1\"",
                              "\"cpu_gpu_payload_first_difference\": \"0\"",
                              "\"cpu_footer_differs_from_gpu_checked_footer\": \"true\"",
                              "\"tile_0_bytes\": \"9\"",
                              "\"truncated\": \"false\""})
        EXPECT_NE(manifest.find(field), std::string::npos) << field << "\n" << manifest;
}

TEST_F(CrcDumpWriterTest, ComputeFailureKeepsEvidenceWithoutInventingAValidChecksumOrCheckedFooter) {
    CrcDumpWriter writer(2, root_.string());
    auto          observed            = failure();
    observed.status                   = CrcCopyStatus::CRC_COMPUTE_ERROR;
    observed.nvcomp_status            = 7;
    observed.checked_footer_available = false;  // Failed store seals an invalid candidate; it checks no stored footer.
    auto reservation                  = writer.reserve(16);
    ASSERT_NE(reservation, nullptr);
    const auto path = writer.write(*reservation, item(), observed, {{"direction", "DEVICE->HOST"}});
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(readBytes(fs::path(path) / "cpu.bin"), observed.cpu_snapshot);
    EXPECT_FALSE(fs::exists(fs::path(path) / "checked_footer.bin"));
    const auto manifest = readText(fs::path(path) / "manifest.json");
    EXPECT_NE(manifest.find("\"failure_stage\": \"crc_compute\""), std::string::npos);
    EXPECT_NE(manifest.find("\"actual_crc32c\": \"unavailable\""), std::string::npos);
    EXPECT_NE(manifest.find("\"expected_crc32c\": \"unavailable\""), std::string::npos);
    EXPECT_NE(manifest.find("\"gpu_crc_status\": \"7\""), std::string::npos);
}

TEST_F(CrcDumpWriterTest, CaptureFailureStillWritesVerdictAndExplicitMissingSnapshot) {
    CrcDumpWriter writer(0, root_.string());
    auto          observed = failure();
    observed.staging_snapshot.clear();
    observed.capture_error = "GPU snapshot unavailable";
    auto reservation       = writer.reserve(16);
    ASSERT_NE(reservation, nullptr);
    const auto path = writer.write(*reservation, item(), observed, {});
    ASSERT_FALSE(path.empty());
    EXPECT_FALSE(fs::exists(fs::path(path) / "gpu_staging.bin"));
    EXPECT_TRUE(fs::exists(fs::path(path) / "cpu.bin"));
    EXPECT_NE(readText(fs::path(path) / "manifest.json").find("\"staging_captured\": \"false\""), std::string::npos);
}

TEST_F(CrcDumpWriterTest, ConcurrentWorkersShareTwoAdmissionsPerMinute) {
    CrcDumpWriter            writer(0, root_.string());
    const auto               now = CrcDumpWriter::Clock::now();
    std::atomic<unsigned>    admitted{0};
    std::vector<std::thread> workers;
    for (size_t i = 0; i < 32; ++i)
        workers.emplace_back([&] {
            if (writer.reserve(16, now))
                ++admitted;
        });
    for (auto& worker : workers)
        worker.join();
    EXPECT_EQ(admitted, 2u);
    EXPECT_EQ(writer.reserve(16, now), nullptr);
    EXPECT_NE(writer.reserve(16, now + std::chrono::minutes(1)), nullptr);
    EXPECT_NE(writer.reserve(16, now + std::chrono::minutes(1)), nullptr);
    EXPECT_EQ(writer.reserve(16, now + std::chrono::minutes(1)), nullptr);
}

TEST_F(CrcDumpWriterTest, ReservationsCoverConcurrentSnapshotsAndReleaseAfterFailure) {
    CrcDumpWriter writer(0, root_.string(), CrcDumpWriter::kMetadataBytes + 1024);
    auto          reservation = writer.reserve(16);
    ASSERT_NE(reservation, nullptr);
    EXPECT_EQ(writer.reserve(16), nullptr);
    // Remove the admitted output directory and put a file in its place: force
    // a real filesystem error after snapshot admission, without permission tricks.
    fs::remove_all(root_ / "rank_0");
    std::ofstream(root_ / "rank_0") << "blocked";
    EXPECT_TRUE(writer.write(*reservation, item(), failure(), {}).empty());
    fs::remove(root_ / "rank_0");
    auto retry = writer.reserve(16);
    ASSERT_NE(retry, nullptr) << "failed write leaked its pending byte reservation";
    EXPECT_FALSE(writer.write(*retry, item(), failure(), {}).empty());
}

TEST_F(CrcDumpWriterTest, RotatesOldCompletedDumpsBeforeReservingFullEvidence) {
    CrcDumpWriter writer(0, root_.string(), 2 * CrcDumpWriter::kMetadataBytes);
    const auto    old = root_ / "rank_0" / "crc_old";
    fs::create_directories(old);
    {
        std::ofstream file(old / "cpu.bin");
        file.seekp(1536 * 1024);
        file.put('x');
    }
    const auto unrelated = root_ / "rank_0" / "keep_me";
    fs::create_directories(unrelated);
    std::ofstream(unrelated / "metadata") << "not a CRC dump";
    auto reservation = writer.reserve(16);
    ASSERT_NE(reservation, nullptr);
    EXPECT_FALSE(fs::exists(old));
    EXPECT_TRUE(fs::exists(unrelated));
    EXPECT_FALSE(writer.write(*reservation, item(), failure(), {}).empty());
}

TEST_F(CrcDumpWriterTest, RejectsOversizedRecordsBeforeFilesystemOrSnapshotAllocation) {
    CrcDumpWriter writer(0, root_.string());
    EXPECT_EQ(writer.reserve(CrcDumpWriter::kQuotaBytes), nullptr);
    EXPECT_FALSE(fs::exists(root_ / "rank_0"));
    EXPECT_EQ(CrcDumpWriter::forRank(9181), CrcDumpWriter::forRank(9181));
    EXPECT_NE(CrcDumpWriter::forRank(9181), CrcDumpWriter::forRank(9182));
}
}  // namespace
}  // namespace rtp_llm
