package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Guards the shipped master configs against FlexlbConfig schema drift:
 * every master config and performance model pair must parse against the
 * current FlexlbConfig schema. MockPerformanceModel.load runs
 * ConfigService.parse with FAIL_ON_UNKNOWN_PROPERTIES, so any renamed or
 * removed field (e.g. the base switch renamed prefill outlierRejection
 * maxWaitVsAverageMultiplier to maxProjectedDrainVsAverageMultiplier)
 * surfaces here before a remote run wastes a full build.
 */
class ConfigSchemaGuardTest {

    private static final String[] MASTERS = {
            "../tools/online_eval/data/config/master_fixed_window.json",
            "../tools/online_eval/data/config/master_fixed_window_4g.json",
            "../tools/online_eval/data/config/master_fixed_window_slo500_wait160.json",
    };

    private static final String[] PERFORMANCES = {
            "../tools/online_eval/data/performance/dsv4_flash_performance.fast_ab.json",
            "../tools/online_eval/data/performance/dsv4_flash_performance.realistic.json",
    };

    @Test
    void shippedConfigPairsParseAgainstCurrentSchema() throws Exception {
        for (String master : MASTERS) {
            for (String performance : PERFORMANCES) {
                assertNotNull(MockPerformanceModel.load(performance, master),
                        master + " + " + performance + " must load");
            }
        }
    }
}
