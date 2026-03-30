package org.flexlb.balance.resource;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DynamicWorkerManagerTest {

    @Mock
    private ConfigService configService;
    @Mock
    private ResourceMeasureFactory resourceMeasureFactory;
    @Mock
    private ResourceMeasure resourceMeasure;

    private DynamicWorkerManager manager;
    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setScheduleWorkerSize(8);
        config.setResourceCheckIntervalMs(100);
        when(configService.loadBalanceConfig()).thenReturn(config);
        manager = new DynamicWorkerManager(configService, resourceMeasureFactory);
    }

    @AfterEach
    void tearDown() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    @Test
    void initialPermits_shouldEqualMaxWorkers() {
        assertEquals(8, manager.getTotalPermits());
    }

    @Test
    void recalculate_withZeroWaterLevel_shouldMaxPermits() {
        setupWorkers(0.0);
        manager.recalculateWorkerCapacity();

        // Water level 0% → capacity 100% → 8 workers
        // Already at 8, so no change
        assertEquals(8, manager.getTotalPermits());
    }

    @Test
    void recalculate_withHighWaterLevel_shouldReducePermits() {
        setupWorkers(75.0);
        manager.recalculateWorkerCapacity();

        // Water level 75% → capacity 25% → 2 workers
        // From 8 to 7 (step=1)
        assertEquals(7, manager.getTotalPermits());
    }

    @Test
    void recalculate_multipleIterations_shouldConverge() {
        setupWorkers(50.0);

        // Water level 50% → capacity 50% → target 4 workers
        // Starting from 8, each iteration reduces by 1
        for (int i = 0; i < 10; i++) {
            manager.recalculateWorkerCapacity();
        }

        assertEquals(4, manager.getTotalPermits());
    }

    @Test
    void recalculate_withFullWaterLevel_shouldReduceToZero() {
        setupWorkers(100.0);

        // Water level 100% → capacity 0% → target 0 workers
        for (int i = 0; i < 10; i++) {
            manager.recalculateWorkerCapacity();
        }

        assertEquals(0, manager.getTotalPermits());
    }

    @Test
    void totalPermits_shouldNeverGoNegative() {
        setupWorkers(100.0);

        // Run many iterations to ensure no underflow
        for (int i = 0; i < 20; i++) {
            manager.recalculateWorkerCapacity();
        }

        assertTrue(manager.getTotalPermits() >= 0);
    }

    private void setupWorkers(double waterLevel) {
        WorkerStatus ws = new WorkerStatus();
        ws.setAlive(true);
        ws.setAvailableKvCacheTokens(new AtomicLong(100));
        ws.setUsedKvCacheTokens(new AtomicLong(0));
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.1:8080", ws);

        lenient().when(resourceMeasureFactory.getMeasure(any())).thenReturn(resourceMeasure);
        lenient().when(resourceMeasure.calculateAverageWaterLevel(any())).thenReturn(waterLevel);
    }
}
