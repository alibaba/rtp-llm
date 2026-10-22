package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/** Shipped master/performance fixtures must parse against the strict current schema. */
class ConfigSchemaGuardTest {

    private static final String[] MASTERS = {
            "../tools/online_eval/data/config/master_fixed_window.json",
    };

    private static final String[] PERFORMANCES = {
            "../tools/online_eval/data/performance/dsv4_flash_performance.fast_ab.json",
            "../tools/online_eval/data/performance/dsv4_flash_performance.sm100_dev.json",
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
