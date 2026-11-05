#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/resource/testdata/main_projection/MainProjectionReference.h"
#include "aios/navi_sched/resource/testdata/legacy_v76/LegacyV76NonLinearPrefillModelR.h"
#include "unittest/unittest.h"
#include <cmath>
#include <cstdlib>
using namespace navi_sched;
class LegacyV76FitParityTest : public TESTBASE {};

TEST_F(LegacyV76FitParityTest,
       testChronologicalPredictionsAndEveryAdamUpdateMatchMainProjection)
{
    current_projection_reference::MainProjectionReference archived;
    NonLinearPrefillModelR adapted;
    archived._pdSeparated = adapted._pdSeparated = true;
    archived._useCapacitySplitFeature = adapted._useCapacitySplitFeature = false;
    archived.setExecutionGeometryEnabled(true);
    adapted.setExecutionGeometryEnabled(true);
    archived.setProjectedExecutionGeometryEnabled(true);
    adapted.setProjectedExecutionGeometryEnabled(true);
    // Offline mutation hook: never used by serving code.
    adapted._errorClipMs = std::getenv("NAVI_TEST_LEGACY_ORACLE_FORCE_CLIP") ? 5000 : 0;
    auto oldWeights = archived.getInitialParameters();
    auto newWeights = adapted.getInitialParameters();
    ASSERT_EQ(oldWeights, newWeights);
    EXPECT_EQ(29, archived.getBinaryVersion());
    EXPECT_EQ(33, adapted.getBinaryVersion());
    std::vector<NaviSchedLatencySample> batch;
    for (int seq = 0; seq < 1024; ++seq) {
        NaviSchedLatencySample sample;
        const auto count = 1 + seq % 64;
        for (int request = 0; request < count; ++request) {
            const int hit = (seq * 13 + request * 5) % 128 * 1152;
            const int miss = 1 + (seq * 17 + request * 11) % 512 * 96;
            sample.features.appendRequest(hit, miss);
        }
        ASSERT_TRUE(sample.features.appendBatchCount(1));
        ASSERT_TRUE(
            sample.features.setExecutionGeometry(1 + seq % 24, 6144 * (1 + seq % 96)));
        // Always predict before this sample becomes visible to either fitter.
        const auto oldPrediction =
            archived.predictConfigured(sample.features, oldWeights);
        const auto newPrediction = adapted.predictConfigured(sample.features, newWeights);
        ASSERT_DOUBLE_EQ(oldPrediction, newPrediction) << seq;
        sample.latencyMs = 10 + (seq * 71) % 18000;
        batch.push_back(sample);
        if (batch.size() > 32) batch.erase(batch.begin());
        std::vector<double> oldNext, newNext;
        ASSERT_TRUE(archived.fit(batch, oldWeights, oldNext, 0.125, 4.0));
        ASSERT_TRUE(adapted.fit(batch, newWeights, newNext, 0.125, 4.0));
        ASSERT_EQ(oldNext, newNext) << seq;
        const auto oldAdam = archived.getOptimizerState();
        const auto newAdam = adapted.getOptimizerState();
        ASSERT_EQ(oldAdam.step, newAdam.step);
        for (size_t i = 0; i < oldWeights.size(); ++i) {
            ASSERT_DOUBLE_EQ(oldAdam.adamMoment1[i], newAdam.adamMoment1[i])
                << seq << ":" << i;
            ASSERT_DOUBLE_EQ(oldAdam.adamMoment2[i], newAdam.adamMoment2[i])
                << seq << ":" << i;
        }
        oldWeights.swap(oldNext);
        newWeights.swap(newNext);
    }
}

// The retained upstream projection fix intentionally changes update 4. Keep an
// explicit historical check so that removing the fix cannot look like parity.
TEST_F(LegacyV76FitParityTest, testMainProjectionDiffersFromArchivedRollbackAtUpdateFour)
{
    LegacyV76NonLinearPrefillModelR legacy;
    current_projection_reference::MainProjectionReference current;
    legacy._pdSeparated = current._pdSeparated = true;
    legacy._useCapacitySplitFeature = current._useCapacitySplitFeature = false;
    legacy.setExecutionGeometryEnabled(true);
    current.setExecutionGeometryEnabled(true);
    legacy.setProjectedExecutionGeometryEnabled(true);
    current.setProjectedExecutionGeometryEnabled(true);
    auto legacyWeights = legacy.getInitialParameters();
    auto currentWeights = current.getInitialParameters();
    ASSERT_EQ(legacyWeights, currentWeights);
    std::vector<NaviSchedLatencySample> batch;
    for (int seq = 0; seq <= 4; ++seq) {
        NaviSchedLatencySample sample;
        for (int request = 0; request < 1 + seq % 64; ++request) {
            const int hit = (seq * 13 + request * 5) % 128 * 1152;
            const int miss = 1 + (seq * 17 + request * 11) % 512 * 96;
            sample.features.appendRequest(hit, miss);
        }
        ASSERT_TRUE(sample.features.appendBatchCount(1));
        ASSERT_TRUE(sample.features.setExecutionGeometry(
            1 + seq % 24, 6144 * (1 + seq % 96)));
        ASSERT_DOUBLE_EQ(legacy.predictConfigured(sample.features, legacyWeights),
                         current.predictConfigured(sample.features, currentWeights)) << seq;
        sample.latencyMs = 10 + (seq * 71) % 18000;
        batch.push_back(sample);
        std::vector<double> legacyNext, currentNext;
        ASSERT_TRUE(legacy.fit(batch, legacyWeights, legacyNext, 0.125, 4.0));
        ASSERT_TRUE(current.fit(batch, currentWeights, currentNext, 0.125, 4.0));
        if (seq < 4) {
            ASSERT_EQ(legacyNext, currentNext) << seq;
        } else {
            constexpr size_t curvature = 8;
            ASSERT_DOUBLE_EQ(legacyWeights[curvature], legacyNext[curvature]);
            ASSERT_LT(currentNext[curvature], legacyNext[curvature]);
            ASSERT_NE(legacyNext, currentNext);
        }
        legacyWeights.swap(legacyNext);
        currentWeights.swap(currentNext);
    }
}
