package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

/**
 * Tests that WorkerBatcher selects the correct algorithm based on config switches.
 *
 * <p>Switch matrix:
 * <ul>
 *   <li>autoTpmEnabled=false, autoTpmQueueYieldEnabled=false → FixedWindow</li>
 *   <li>autoTpmEnabled=true,  autoTpmQueueYieldEnabled=false → FixedWindow</li>
 *   <li>autoTpmEnabled=true,  autoTpmQueueYieldEnabled=true  → PriorityYield</li>
 * </ul>
 */
class WorkerBatcherAlgorithmSelectionTest {

    @Test
    void bothDisabled_selectsFixedWindow() throws Exception {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setAutoTpmEnabled(false);
        cfg.setAutoTpmQueueYieldEnabled(false);

        WorkerBatcher batcher = createBatcher(cfg);
        assertInstanceOf(FixedWindowBatcherAlgorithm.class, getAlgorithm(batcher));
    }

    @Test
    void enabledButYieldDisabled_selectsFixedWindow() throws Exception {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(false);

        WorkerBatcher batcher = createBatcher(cfg);
        assertInstanceOf(FixedWindowBatcherAlgorithm.class, getAlgorithm(batcher));
    }

    @Test
    void enabledAndYieldEnabled_selectsPriorityYield() throws Exception {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);

        WorkerBatcher batcher = createBatcher(cfg);
        assertInstanceOf(PriorityYieldBatcherAlgorithm.class, getAlgorithm(batcher));
    }

    // ---- helpers ----

    private static WorkerBatcher createBatcher(FlexlbConfig cfg) {
        PrefillEndpoint ep = mock(PrefillEndpoint.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        return new WorkerBatcher("test-worker", ep, cfg, reporter);
    }

    private static Object getAlgorithm(WorkerBatcher batcher) throws Exception {
        Field field = WorkerBatcher.class.getDeclaredField("algorithm");
        field.setAccessible(true);
        return field.get(batcher);
    }
}
