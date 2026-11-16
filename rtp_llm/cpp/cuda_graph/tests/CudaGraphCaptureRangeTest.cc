#include "gtest/gtest.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_capture_range.h"

namespace rtp_llm {
namespace {

TEST(CudaGraphCaptureRangeTest, DraftPrefillUsesDecodeBatchBuckets) {
    const std::vector<int> decode_batch_sizes{1, 4, 8, 16, 24, 32, 64};
    const std::vector<int> expected_seq_lens{4, 16, 32, 64, 96, 128, 256};

    EXPECT_EQ(draftPrefillCaptureSeqLens(decode_batch_sizes, 4), expected_seq_lens);
}

TEST(CudaGraphCaptureRangeTest, EmptyDecodeRangeStaysEmpty) {
    EXPECT_TRUE(draftPrefillCaptureSeqLens({}, 4).empty());
}

TEST(CudaGraphCaptureRangeTest, AdaptiveSparseBucketsCoverEveryLiveRow) {
    for (int concurrency : {1, 2, 7, 48}) {
        for (int width : {2, 3, 6}) {
            const auto rows = adaptiveVerifyCaptureRows(concurrency, width);
            ASSERT_EQ(rows.back(), concurrency * width);
            ASSERT_TRUE(std::is_sorted(rows.begin(), rows.end()));
            EXPECT_EQ(std::adjacent_find(rows.begin(), rows.end()), rows.end());
            for (int r : rows) {
                auto shape = adaptiveVerifyGraphShape(concurrency, width, r);
                EXPECT_EQ(shape.rows, r);
                EXPECT_EQ(shape.batch_capacity, std::min(r, concurrency));
                auto lengths = adaptiveVerifyCaptureLengths(shape);
                int  sum     = 0;
                for (int length : lengths) {
                    EXPECT_GE(length, 1);
                    EXPECT_LE(length, width);
                    sum += length;
                }
                EXPECT_EQ(sum, r);
            }
            // Every feasible combination, including cap=0 and heterogeneous caps.
            for (int b = 1; b <= concurrency; ++b) {
                for (int r = b; r <= b * width; ++r) {
                    auto shape = adaptiveVerifyReplayShape(concurrency, width, b, r);
                    EXPECT_EQ(shape.rows, r);
                    EXPECT_GE(shape.batch_capacity, b);
                    const int  bucket   = *std::lower_bound(rows.begin(), rows.end(), r);
                    const auto captured = adaptiveVerifyGraphShape(concurrency, width, bucket);
                    EXPECT_GE(captured.rows, r);
                    EXPECT_GE(captured.batch_capacity, b);
                }
            }
        }
    }
}

TEST(CudaGraphCaptureRangeTest, AdaptiveUsesVllmGridAndClosesFinalBucket) {
    const std::vector<int> expected{1,  2,  4,  8,  10, 12,  14,  16,  20,  24,  28,  32,  40,
                                    48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384};
    EXPECT_EQ(adaptiveVerifyCaptureRows(64, 6), expected);
    for (int maximum : {3, 7, 65, 1024, 1025, 4095, 4096, 4097, 8192}) {
        const auto rows = adaptiveVerifyCaptureRows(1, maximum);
        EXPECT_EQ(rows.back(), maximum);
        EXPECT_EQ(rows.front(), 1);
        EXPECT_TRUE(std::is_sorted(rows.begin(), rows.end()));
        EXPECT_EQ(std::adjacent_find(rows.begin(), rows.end()), rows.end());
        for (int n = 1; n <= maximum; ++n)
            EXPECT_NE(std::lower_bound(rows.begin(), rows.end(), n), rows.end());
    }
}

TEST(CudaGraphCaptureRangeTest, CustomRowsNormalizeAndCoverAllLegalShapes) {
    EXPECT_EQ(configuredAdaptiveVerifyCaptureRows(64, 6, ""), adaptiveVerifyCaptureRows(64, 6));
    for (const auto& config : {"24", " 19, 3, 11, 3 ", "1,2,7,23,24"}) {
        const auto rows = configuredAdaptiveVerifyCaptureRows(4, 6, config);
        EXPECT_EQ(rows.back(), 24);
        EXPECT_TRUE(std::is_sorted(rows.begin(), rows.end()));
        EXPECT_EQ(std::adjacent_find(rows.begin(), rows.end()), rows.end());
        for (int b = 1; b <= 4; ++b) {
            for (int r = b; r <= b * 6; ++r) {
                auto shape = adaptiveVerifyGraphShape(4, 6, *std::lower_bound(rows.begin(), rows.end(), r));
                EXPECT_GE(shape.batch_capacity, b);
                const auto lengths = adaptiveVerifyCaptureLengths(shape);
                int        sum     = 0;
                for (int n : lengths) {
                    EXPECT_GE(n, 1);
                    EXPECT_LE(n, 6);
                    sum += n;
                }
                EXPECT_EQ(sum, shape.rows);
            }
        }
    }
    EXPECT_EQ(configuredAdaptiveVerifyCaptureRows(4, 6, " 19,3,11,3 "), (std::vector<int>{3, 11, 19, 24}));
    EXPECT_EQ(configuredAdaptiveVerifyCaptureRows(1, 2, "2"), (std::vector<int>{2}));
    EXPECT_EQ(configuredAdaptiveVerifyCaptureRows(1, std::numeric_limits<int>::max(), "2147483647"),
              (std::vector<int>{std::numeric_limits<int>::max()}));
}

TEST(CudaGraphCaptureRangeTest, CustomRowsRejectMalformedAndOutOfRangeValues) {
    for (const auto& config : {" ",
                               ",1",
                               "1,",
                               "1,,2",
                               "0",
                               "-1",
                               "+1",
                               "1.5",
                               "1x",
                               "1 2",
                               "25",
                               "2147483648",
                               "9999999999999999999999999",
                               "1;2"}) {
        EXPECT_THROW(configuredAdaptiveVerifyCaptureRows(4, 6, config), std::invalid_argument) << config;
    }
    EXPECT_THROW(configuredAdaptiveVerifyCaptureRows(1, 2, "9"), std::invalid_argument);
}

TEST(CudaGraphCaptureRangeTest, AdaptiveRejectsInvalidDimensions) {
    EXPECT_THROW(adaptiveVerifyCaptureRows(0, 6), std::invalid_argument);
    EXPECT_THROW(adaptiveVerifyCaptureRows(4, 1), std::invalid_argument);
    EXPECT_THROW(adaptiveVerifyCaptureRows(std::numeric_limits<int>::max(), 6), std::invalid_argument);
    for (const auto& dims : std::vector<std::pair<int, int>>{{0, 1}, {5, 5}, {2, 1}, {2, 13}, {1, 0}, {4, 25}}) {
        EXPECT_THROW(adaptiveVerifyReplayShape(4, 6, dims.first, dims.second), std::invalid_argument);
    }
}

}  // namespace
}  // namespace rtp_llm
