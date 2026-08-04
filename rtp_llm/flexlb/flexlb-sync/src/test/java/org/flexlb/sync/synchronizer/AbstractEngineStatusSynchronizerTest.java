package org.flexlb.sync.synchronizer;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadPoolExecutor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AbstractEngineStatusSynchronizerTest {

    @AfterEach
    void shutdownExecutors() {
        if (AbstractEngineStatusSynchronizer.engineSyncExecutor != null) {
            AbstractEngineStatusSynchronizer.engineSyncExecutor.shutdownNow();
        }
        if (AbstractEngineStatusSynchronizer.statusCheckExecutor != null) {
            AbstractEngineStatusSynchronizer.statusCheckExecutor.shutdownNow();
        }
        AbstractEngineStatusSynchronizer.engineSyncExecutor = null;
        AbstractEngineStatusSynchronizer.statusCheckExecutor = null;
    }

    @Test
    void usesConfiguredSynchronizationQueueCapacities() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEngineSyncExecutorQueueCapacity(7);
        config.setStatusCheckExecutorQueueCapacity(11);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);

        new TestSynchronizer(configService);

        ThreadPoolExecutor engineSyncExecutor = (ThreadPoolExecutor) AbstractEngineStatusSynchronizer.engineSyncExecutor;
        ThreadPoolExecutor statusCheckExecutor = (ThreadPoolExecutor) AbstractEngineStatusSynchronizer.statusCheckExecutor;
        assertEquals(7, engineSyncExecutor.getQueue().remainingCapacity());
        assertEquals(11, statusCheckExecutor.getQueue().remainingCapacity());
    }

    @Test
    void rejectsNonPositiveSynchronizationExecutorConfiguration() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEngineSyncExecutorThreads(0);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);

        assertThrows(IllegalArgumentException.class, () -> new TestSynchronizer(configService));
    }

    private static final class TestSynchronizer extends AbstractEngineStatusSynchronizer {

        private TestSynchronizer(ConfigService configService) {
            super(null,
                    null,
                    null,
                    null,
                    configService);
        }

        @Override
        protected void syncEngineStatus() {
        }
    }
}
