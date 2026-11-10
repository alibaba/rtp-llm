#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/util/json_util.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/PrefillRequestLogger.h"

namespace rtp_llm {
namespace {
class RequestLogDirectory {
public:
    RequestLogDirectory():
        root(
            std::filesystem::temp_directory_path()
            / ("prefill_request_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        std::filesystem::create_directory(root);
        ::chmod(root.c_str(), 0755);
    }
    ~RequestLogDirectory() {
        std::filesystem::remove_all(root);
    }
    std::filesystem::path output() const {
        return root / "prefill_requests";
    }
    std::filesystem::path file() const {
        return output() / ("prefill_request_r3_p" + std::to_string(::getpid()) + ".log");
    }
    std::filesystem::path root;
};

class ScopedUmask {
public:
    ScopedUmask(): previous_(::umask(0022)) {}
    ~ScopedUmask() {
        ::umask(previous_);
    }

private:
    mode_t previous_;
};

bool waitForRequest(const std::filesystem::path& path, int64_t request_id) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
        std::ifstream input(path);
        std::string   line;
        while (std::getline(input, line)) {
            google::protobuf::Struct record;
            if (google::protobuf::util::JsonStringToMessage(line, &record).ok()
                && record.fields().at("request").struct_value().fields().at("request_id").string_value()
                       == std::to_string(request_id)) {
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
}  // namespace

TEST(PrefillRequestLoggerTest, RoundTripsLongRequestWithoutTruncation) {
    RequestLogDirectory directory;
    GenerateInputPB     request;
    request.set_request_id(9223372036854770000LL);
    request.mutable_request_info()->set_request_id("dash-request");
    request.mutable_request_info()->set_trace_id("trace\nwith\"quotes");
    request.mutable_generate_config()->set_max_new_tokens(128);
    request.mutable_generate_config()->mutable_random_seed()->set_value(1234);
    for (int i = 0; i < 400000; ++i) {
        request.add_token_ids(100000 + i);
    }
    {
        PrefillRequestLogger logger(directory.root.string(), 3, 1);
        logger.log(request);
    }
    std::ifstream input(directory.file());
    std::string   line;
    ASSERT_TRUE(std::getline(input, line));
    EXPECT_GT(line.size(), 2048000);
    google::protobuf::Struct record;
    ASSERT_TRUE(google::protobuf::util::JsonStringToMessage(line, &record).ok());
    EXPECT_EQ(record.fields().at("record_type").string_value(), "prefill_request");
    EXPECT_EQ(record.fields().at("rank_id").number_value(), 3);
    std::string request_json;
    ASSERT_TRUE(
        google::protobuf::util::MessageToJsonString(record.fields().at("request").struct_value(), &request_json).ok());
    GenerateInputPB restored;
    ASSERT_TRUE(google::protobuf::util::JsonStringToMessage(request_json, &restored).ok());
    EXPECT_EQ(restored.SerializeAsString(), request.SerializeAsString());
    EXPECT_FALSE(std::getline(input, line));
}

TEST(PrefillRequestLoggerTest, WritesWhileAliveAndKeepsRotatedFilesPrivate) {
    ScopedUmask          mask;
    RequestLogDirectory  directory;
    PrefillRequestLogger logger(directory.root.string(), 3, 2, 1);
    for (int id = 1; id <= 4; ++id) {
        GenerateInputPB request;
        request.set_request_id(id);
        request.add_token_ids(id);
        logger.log(request);
        ASSERT_TRUE(waitForRequest(directory.file(), id));
        struct stat st{};
        ASSERT_EQ(::stat(directory.output().c_str(), &st), 0);
        EXPECT_EQ(st.st_mode & 0777, 0700);
        for (const auto& entry : std::filesystem::directory_iterator(directory.output())) {
            ASSERT_EQ(::lstat(entry.path().c_str(), &st), 0);
            EXPECT_TRUE(S_ISREG(st.st_mode));
            EXPECT_EQ(st.st_mode & 0777, 0600);
        }
    }
    EXPECT_TRUE(waitForRequest(directory.file().string() + ".1", 3));
    EXPECT_TRUE(waitForRequest(directory.file().string() + ".2", 2));
    EXPECT_FALSE(std::filesystem::exists(directory.file().string() + ".3"));
}

TEST(PrefillRequestLoggerTest, RejectsSymlinkDirectoryAndFile) {
    RequestLogDirectory directory;
    const auto          target = directory.root / "target";
    std::filesystem::create_directory(target);
    ASSERT_EQ(::symlink(target.c_str(), directory.output().c_str()), 0);
    EXPECT_THROW(PrefillRequestLogger(directory.root.string(), 3, 1), std::exception);
    ASSERT_EQ(::unlink(directory.output().c_str()), 0);
    std::filesystem::create_directory(directory.output());
    const auto target_file = target / "untouched";
    std::ofstream(target_file) << "unchanged";
    ASSERT_EQ(::symlink(target_file.c_str(), directory.file().c_str()), 0);
    EXPECT_THROW(PrefillRequestLogger(directory.root.string(), 3, 1), std::exception);
    std::ifstream input(target_file);
    std::string   content;
    ASSERT_TRUE(std::getline(input, content));
    EXPECT_EQ(content, "unchanged");
}

}  // namespace rtp_llm
