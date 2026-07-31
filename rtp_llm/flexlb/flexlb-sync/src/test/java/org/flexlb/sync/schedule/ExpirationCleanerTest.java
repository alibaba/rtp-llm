package org.flexlb.sync.schedule;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ExpirationCleanerTest {

    @Test
    void keepsTaskBeforeConfirmationTimeout() {
        WorkerStatus workerStatus = workerStatusWithLocalTask();
        ExpirationCleaner cleaner = expirationCleaner(new FlexlbConfig());

        workerStatus.getLocalTaskMap().get("request-1")
                .setLastActiveTimeUs(System.nanoTime() / 1000 - TimeUnit.SECONDS.toMicros(299));
        cleaner.doClean(workerStatusMap(workerStatus), RoleType.PREFILL);

        assertTrue(workerStatus.getLocalTaskMap().containsKey("request-1"));
        assertEquals(860, workerStatus.getRunningQueueTime().get());
    }

    @Test
    void removesTaskAfterConfirmationTimeout() {
        WorkerStatus workerStatus = workerStatusWithLocalTask();
        TaskInfo task = workerStatus.getLocalTaskMap().get("request-1");
        task.setLastActiveTimeUs(System.nanoTime() / 1000 - TimeUnit.SECONDS.toMicros(301));
        ExpirationCleaner cleaner = expirationCleaner(new FlexlbConfig());
        ch.qos.logback.classic.Logger pvLogger =
                (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger");
        ListAppender<ILoggingEvent> pvEvents = new ListAppender<>();
        pvEvents.start();
        pvLogger.addAppender(pvEvents);

        try {
            cleaner.doClean(workerStatusMap(workerStatus), RoleType.PREFILL);
        } finally {
            pvLogger.detachAppender(pvEvents);
            pvEvents.stop();
        }

        assertFalse(workerStatus.getLocalTaskMap().containsKey("request-1"));
        assertEquals(TaskStateEnum.CLEANED, task.getTaskState());
        assertEquals(0, workerStatus.getRunningQueueTime().get());
        assertEquals(1, pvEvents.list.size());
        String pvEvent = pvEvents.list.getFirst().getFormattedMessage();
        assertTrue(pvEvent.contains("\"eventType\":\"task_confirmation_timeout\""));
        assertTrue(pvEvent.contains("\"requestId\":\"request-1\""));
        assertTrue(pvEvent.contains("\"confirmationTimeoutMs\":300000"));
    }

    @Test
    void usesConfiguredConfirmationTimeout() {
        WorkerStatus workerStatus = workerStatusWithLocalTask();
        workerStatus.getLocalTaskMap().get("request-1")
                .setLastActiveTimeUs(System.nanoTime() / 1000 - TimeUnit.SECONDS.toMicros(11));
        FlexlbConfig config = new FlexlbConfig();
        config.setTaskConfirmTimeoutMs(10_000);

        expirationCleaner(config).doClean(workerStatusMap(workerStatus), RoleType.PREFILL);

        assertFalse(workerStatus.getLocalTaskMap().containsKey("request-1"));
    }

    private ExpirationCleaner expirationCleaner(FlexlbConfig config) {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        return new ExpirationCleaner(mock(FlexMonitor.class), configService);
    }

    private WorkerStatus workerStatusWithLocalTask() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        workerStatus.setRole(RoleType.PREFILL.getCode());
        workerStatus.getStatusLastUpdateTime().set(System.nanoTime() / 1000);

        TaskInfo task = new TaskInfo();
        task.setRequestId("request-1");
        task.setInputLength(1_000);
        task.setPrefixLength(200);
        task.setPredictedPrefixLength(200);
        task.setCacheMatchSource("KVCM");
        workerStatus.putLocalTask(task.getRequestId(), task);
        return workerStatus;
    }

    private Map<String, WorkerStatus> workerStatusMap(WorkerStatus workerStatus) {
        Map<String, WorkerStatus> workerStatuses = new HashMap<>();
        workerStatuses.put(workerStatus.getIpPort(), workerStatus);
        return workerStatuses;
    }
}
