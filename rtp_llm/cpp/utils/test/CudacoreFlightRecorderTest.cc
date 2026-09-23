#include "rtp_llm/cpp/utils/CudacoreFlightRecorder.h"

#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>
#include <string>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(CudacoreFlightRecorderTest, RetainsOnlyRecentSubmissionsAndFreezesAtFault) {
    ASSERT_EQ(::setenv("RTP_LLM_CUDACORE_FLIGHT_RECORDER", "1", 1), 0);
    cudacore_test::resetFlightRecorder();
    prepareCudacoreFlightRecorder();

    CudacoreFlightEvent event;
    event.kind         = CudacoreFlightKind::BatchSubmit;
    event.device_index = 2;
    event.stream       = 0x1234;
    event.tile_count   = 88;
    event.total_bytes  = 4096;
    for (uint64_t i = 1; i <= kCudacoreFlightCapacity + 45; ++i) {
        EXPECT_EQ(recordCudacoreFlightEvent(event), i);
    }

    const std::string snapshot = snapshotCudacoreFlightRecorderJson();
    EXPECT_EQ(snapshot.find("\"sequence\":45,"), std::string::npos);
    EXPECT_NE(snapshot.find("\"sequence\":46,"), std::string::npos);
    EXPECT_NE(snapshot.find("\"overwritten\":45,"), std::string::npos);
    EXPECT_NE(snapshot.find("\"device_index\":2"), std::string::npos);

    freezeCudacoreFlightRecorder();
    const std::string frozen = snapshotCudacoreFlightRecorderJson();
    event.kind               = CudacoreFlightKind::BatchCompletion;
    event.cuda_error         = 719;
    for (size_t i = 0; i < kCudacorePostFaultCapacity + 10; ++i) {
        ASSERT_NE(recordCudacoreFlightEvent(event), 0);
    }
    const std::string after = snapshotCudacoreFlightRecorderJson();
    const auto        begin = frozen.find("\"events\":[");
    const auto        end   = frozen.find(",\"post_fault_events\":");
    EXPECT_NE(after.find(frozen.substr(begin, end - begin)), std::string::npos);
    EXPECT_NE(after.find("\"post_fault_overwritten\":10"), std::string::npos);
    EXPECT_NE(after.find("\"cuda_error\":719"), std::string::npos);
}

TEST(CudacoreFlightRecorderTest, ConcurrentWritersAndSnapshotsDoNotDropEvents) {
    cudacore_test::resetFlightRecorder();
    std::atomic<bool>        start{false};
    std::atomic<size_t>      lost{0};
    std::vector<std::thread> writers;
    for (int t = 0; t < 8; ++t) {
        writers.emplace_back([&, t] {
            while (!start.load()) {
                std::this_thread::yield();
            }
            CudacoreFlightEvent event;
            event.copy_id = t + 1;
            for (size_t i = 0; i < 256; ++i) {
                if (recordCudacoreFlightEvent(event) == 0) {
                    ++lost;
                }
            }
        });
    }
    start.store(true);
    for (int i = 0; i < 4; ++i) {
        (void)snapshotCudacoreFlightRecorderJson();
    }
    for (auto& writer : writers) {
        writer.join();
    }
    EXPECT_EQ(lost.load(), 0);
    const auto json = snapshotCudacoreFlightRecorderJson();
    for (size_t sequence = 1; sequence <= 2048; ++sequence) {
        ASSERT_NE(json.find("\"sequence\":" + std::to_string(sequence) + ","), std::string::npos);
    }
    EXPECT_NE(json.find("\"dropped\":0"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
