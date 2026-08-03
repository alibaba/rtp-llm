package org.flexlb.balance.resource;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ResourceWaterLevelTest {

    @Test
    void prefillWaterLevelUsesWorkerWaitingQueue() {
        FlexlbConfig config = new FlexlbConfig();
        config.setMaxPrefillQueueSize(20);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setWaitingTaskList(Map.of(
                "request-1", new TaskInfo(),
                "request-2", new TaskInfo(),
                "request-3", new TaskInfo(),
                "request-4", new TaskInfo(),
                "request-5", new TaskInfo()));

        assertEquals(25.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    @Test
    void decodeWaterLevelUsesConfiguredKvCacheRange() {
        FlexlbConfig config = new FlexlbConfig();
        config.setDecodeFullSpeedThreshold(40);
        config.setDecodeStopThreshold(80);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setUsedKvCacheTokens(new AtomicLong(60));
        workerStatus.setAvailableKvCacheTokens(new AtomicLong(40));

        assertEquals(50.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    private static ConfigService configService(FlexlbConfig config) {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        return configService;
    }
}
