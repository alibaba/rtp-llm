#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"

#include <gtest/gtest.h>
#include <array>
#include <chrono>
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
        CrcCopyFailure result{0, 0x12345678, 0xe3069283};
        result.cpu_snapshot.resize(16);
        std::memcpy(result.cpu_snapshot.data(), "123456789", 9);
        std::memcpy(result.cpu_snapshot.data() + 12, &result.expected, 4);
        result.staging_snapshot = result.cpu_snapshot;
        return result;
    }
    static CrcCopyItem item() {
        return {nullptr, 9, 16, {{reinterpret_cast<void*>(0x1234), 0, 9}}};
    }
    static std::string
    dump(CrcDumpWriter& writer, const CrcCopyFailure& observed, CrcDumpWriter::Metadata metadata = {}) {
        auto source          = item();
        source.dump_metadata = [&] { return metadata; };
        return writer.write(source, observed, [&] { return observed; });
    }
    fs::path root_;
};

TEST_F(CrcDumpWriterTest, SavesCheckedGpuBytesAndLaterCpuMutationSeparately) {
    CrcDumpWriter writer(7, root_.string());
    auto          observed = failure();
    observed.cpu_snapshot[0] ^= 1;
    observed.cpu_snapshot[12] ^= 1;
    const auto path = dump(writer,
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
    const auto path                   = dump(writer, observed, {{"direction", "DEVICE->HOST"}});
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
    const auto path        = dump(writer, observed);
    ASSERT_FALSE(path.empty());
    EXPECT_FALSE(fs::exists(fs::path(path) / "gpu_staging.bin"));
    EXPECT_TRUE(fs::exists(fs::path(path) / "cpu.bin"));
    EXPECT_NE(readText(fs::path(path) / "manifest.json").find("\"staging_captured\": \"false\""), std::string::npos);
}

TEST_F(CrcDumpWriterTest, ConcurrentWritesSerializeCaptureAndKeepNewestTwenty) {
    CrcDumpWriter            writer(0, root_.string());
    std::atomic<unsigned>    active{0}, completed{0};
    std::atomic<bool>        overlap{false};
    std::vector<std::thread> workers;
    for (size_t i = 0; i < 32; ++i) {
        workers.emplace_back([&, i] {
            auto observed       = failure();
            observed.item_index = i;
            const auto path     = writer.write(item(), observed, [&] {
                if (active.fetch_add(1) != 0)
                    overlap = true;
                std::this_thread::yield();
                auto snapshot = observed;
                active.fetch_sub(1);
                return snapshot;
            });
            if (!path.empty())
                ++completed;
        });
    }
    for (auto& worker : workers)
        worker.join();
    EXPECT_EQ(completed.load(), 32u);
    EXPECT_EQ(active.load(), 0u);
    EXPECT_FALSE(overlap.load());
    size_t count = 0;
    for (const auto& entry : fs::directory_iterator(root_ / "rank_0")) {
        ++count;
        EXPECT_TRUE(fs::exists(entry.path() / "manifest.json"));
        EXPECT_EQ(readBytes(entry.path() / "cpu.bin"), failure().cpu_snapshot);
    }
    EXPECT_EQ(count, 20u);
}

TEST_F(CrcDumpWriterTest, FailedWriteKeepsPreviousDumpAndAllowsRetry) {
    CrcDumpWriter writer(0, root_.string(), 1);
    const auto    first = dump(writer, failure());
    ASSERT_FALSE(first.empty());
    EXPECT_TRUE(
        writer.write(item(), failure(), []() -> CrcCopyFailure { throw std::runtime_error("snapshot unavailable"); })
            .empty());
    EXPECT_TRUE(fs::exists(fs::path(first) / "manifest.json"));
    const auto retry = dump(writer, failure());
    ASSERT_FALSE(retry.empty());
    EXPECT_FALSE(fs::exists(first));
    EXPECT_TRUE(fs::exists(fs::path(retry) / "manifest.json"));
}

TEST_F(CrcDumpWriterTest, RotatesOnlyOldCompletedDumpsAndPreservesUnrelatedEntries) {
    CrcDumpWriter writer(0, root_.string(), 2);
    const auto    rank_dir = root_ / "rank_0";
    const auto    old      = rank_dir / "crc_old";
    const auto    recent   = rank_dir / "crc_recent";
    for (const auto& path : {old, recent}) {
        fs::create_directories(path);
        std::ofstream(path / "manifest.json") << "{}";
        std::ofstream(path / "cpu.bin") << "evidence";
    }
    fs::last_write_time(old / "manifest.json", fs::file_time_type::clock::now() - std::chrono::hours(2));
    fs::last_write_time(recent / "manifest.json", fs::file_time_type::clock::now() - std::chrono::hours(1));
    const auto unrelated  = rank_dir / "keep_me";
    const auto incomplete = rank_dir / "crc_incomplete";
    fs::create_directory(unrelated);
    fs::create_directory(incomplete);
    const auto outside = root_ / "outside";
    fs::create_directory(outside);
    std::ofstream(outside / "keep.bin") << "keep";
    fs::create_directory_symlink(outside, rank_dir / "crc_link");
    ASSERT_FALSE(dump(writer, failure()).empty());
    EXPECT_FALSE(fs::exists(old));
    EXPECT_TRUE(fs::exists(recent));
    EXPECT_TRUE(fs::exists(unrelated));
    EXPECT_TRUE(fs::exists(incomplete));
    EXPECT_TRUE(fs::is_symlink(rank_dir / "crc_link"));
    EXPECT_EQ(fs::file_size(outside / "keep.bin"), 4u);
}

TEST_F(CrcDumpWriterTest, RetainsNewestDumpsWhenManifestTimestampsTie) {
    CrcDumpWriter         writer(0, root_.string(), 8);
    std::vector<fs::path> paths;
    const auto            same_time = fs::file_time_type::clock::now() - std::chrono::hours(1);
    for (size_t i = 0; i < 8; ++i) {
        const auto path = dump(writer, failure());
        ASSERT_FALSE(path.empty());
        paths.push_back(path);
        fs::last_write_time(paths.back() / "manifest.json", same_time);
    }
    CrcDumpWriter next_writer(0, root_.string(), 5);
    const auto    latest = dump(next_writer, failure());
    ASSERT_FALSE(latest.empty());
    EXPECT_TRUE(fs::exists(fs::path(latest) / "manifest.json"));
    for (size_t i = 0; i < paths.size(); ++i)
        EXPECT_EQ(fs::exists(paths[i]), i >= 4) << "dump index=" << i;
}

TEST_F(CrcDumpWriterTest, UnavailableDirectorySkipsCaptureAndAllowsRetry) {
    std::ofstream(root_ / "rank_0") << "blocked";
    CrcDumpWriter writer(0, root_.string());
    unsigned      captures = 0;
    auto          capture  = [&] {
        ++captures;
        return failure();
    };
    EXPECT_TRUE(writer.write(item(), failure(), capture).empty());
    EXPECT_EQ(captures, 0u);
    fs::remove(root_ / "rank_0");
    EXPECT_FALSE(writer.write(item(), failure(), capture).empty());
    EXPECT_EQ(captures, 1u);
}

TEST_F(CrcDumpWriterTest, WorkersShareWriterPerRank) {
    EXPECT_EQ(CrcDumpWriter::forRank(9181), CrcDumpWriter::forRank(9181));
    EXPECT_NE(CrcDumpWriter::forRank(9181), CrcDumpWriter::forRank(9182));
}
}  // namespace
}  // namespace rtp_llm
