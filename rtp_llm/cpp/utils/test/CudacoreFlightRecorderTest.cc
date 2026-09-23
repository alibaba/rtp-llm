#include "rtp_llm/cpp/utils/CudacoreFlightRecorder.h"

#include <cstdlib>
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
    for (uint64_t i = 1; i <= 301; ++i) {
        EXPECT_EQ(recordCudacoreFlightEvent(event), i);
    }

    const std::string snapshot = snapshotCudacoreFlightRecorderJson();
    EXPECT_EQ(snapshot.find("\"sequence\":45,"), std::string::npos);
    EXPECT_NE(snapshot.find("\"sequence\":46,"), std::string::npos);
    EXPECT_NE(snapshot.find("\"sequence\":301,"), std::string::npos);
    EXPECT_NE(snapshot.find("\"device_index\":2"), std::string::npos);

    freezeCudacoreFlightRecorder();
    EXPECT_EQ(recordCudacoreFlightEvent(event), 0);
    EXPECT_EQ(snapshotCudacoreFlightRecorderJson(), snapshot);
}

}  // namespace
}  // namespace rtp_llm
