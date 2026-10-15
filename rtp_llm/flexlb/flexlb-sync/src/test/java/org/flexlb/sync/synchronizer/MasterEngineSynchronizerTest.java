package org.flexlb.sync.synchronizer;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class MasterEngineSynchronizerTest {
    @Test
    @SuppressWarnings("unchecked")
    void reportsEachSlowRunningCycleWithThreadOccupancyAndPoolState() throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.getWorkerRegistry().getHealth().setStatusPollIntervalMs(1L);
        config.getWorkerRegistry().getHealth().setStatusStaleAfterMs(200L);
        var configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        var model = mock(ModelMetaConfig.class);
        when(model.modelName()).thenReturn("trace-model");
        when(model.requiredRoles()).thenReturn(List.of());
        var directory = new WorkerDirectory(mock(EndpointRegistry.class));
        var synchronizer = new MasterEngineSynchronizer(mock(WorkerAddressService.class),
                mock(EngineHealthReporter.class), directory,
                mock(EngineGrpcService.class), model, mock(CacheAwareService.class),
                mock(DynamicCacheIntervalService.class), configService);
        // Exercise the reporting method without live scheduling changing the active set.
        synchronizer.destroy();
        Map<EndpointTrace.Poll, RoleType> traces = (Map<EndpointTrace.Poll, RoleType>)
                ReflectionTestUtils.getField(synchronizer, "activeSyncs");
        assertNotNull(traces);
        var queued = new EndpointTrace.Poll("SYNC", 1L);
        queued.record(Stage.RUNNER_SUBMITTED);
        traces.put(queued, RoleType.PREFILL);
        traces.put(running(2L, Stage.SYNC_RETIREMENT), RoleType.PREFILL);
        traces.put(running(3L, Stage.SYNC_RETIREMENT), RoleType.PREFILL);
        traces.put(running(4L, Stage.SYNC_DISCOVERY), RoleType.DECODE);
        var ended = running(5L, Stage.SYNC_DISPATCH);
        ended.record(Stage.RUNNER_ENDED);
        traces.put(ended, RoleType.PREFILL);
        var delayed = new EndpointTrace.Poll("SYNC", 6L);
        delayed.record(Stage.RUNNER_SUBMITTED);
        TimeUnit.MILLISECONDS.sleep(150L);

        try (var logs = new SyncTraceLogs()) {
            delayed.record(Stage.RUNNER_STARTED);
            delayed.record(Stage.SYNC_DISCOVERY);
            traces.put(delayed, RoleType.DECODE);
            synchronizer.reportSlowTraces();

            assertEquals(4, logs.lines.size(), "report both slow execution and long queue waits");
            assertCycle(logs, 2L, "PREFILL", Stage.SYNC_RETIREMENT);
            assertCycle(logs, 3L, "PREFILL", Stage.SYNC_RETIREMENT);
            assertCycle(logs, 4L, "DECODE", Stage.SYNC_DISCOVERY);
            assertCycle(logs, 6L, "DECODE", Stage.SYNC_DISCOVERY);
            assertFalse(logs.lines.stream().anyMatch(line -> line.contains("cycle:1,")
                    || line.contains("cycle:5,")), "queued and completed cycles do not occupy a running thread");
            assertTrue(logs.lines.stream().allMatch(line -> line.contains("engine={active:0")
                    && line.contains("status={active:0") && line.contains("queued:0")
                    && line.contains("completed:0")));
            assertTrue(logs.lines.stream().allMatch(line -> line.contains("thread=" + Thread.currentThread().getName())
                    && line.contains("queueUs=") && line.contains("runUs=")
                    && line.contains(" tasksPerCycle=0 ")));
        }
    }

    private static EndpointTrace.Poll running(long cycle, Stage stage) {
        var poll = new EndpointTrace.Poll("SYNC", cycle);
        poll.record(Stage.RUNNER_SUBMITTED);
        poll.record(Stage.RUNNER_STARTED);
        poll.record(Stage.SYNC_DISCOVERY);
        poll.record(stage);
        return poll;
    }

    private static void assertCycle(SyncTraceLogs logs, long cycle, String role, Stage stage) {
        assertTrue(logs.lines.stream().anyMatch(line -> line.contains("cycle:" + cycle + ",")
                && line.contains(" role=" + role + " ")
                && line.contains("stage:" + stage.name() + ",")),
                "missing running cycle " + cycle + " for " + role + "/" + stage + ": " + logs.lines);
    }

    private static final class SyncTraceLogs extends AppenderBase<ILoggingEvent> implements AutoCloseable {
        private final ch.qos.logback.classic.Logger logger = (ch.qos.logback.classic.Logger)
                LoggerFactory.getLogger("syncLogger");
        private final ConcurrentLinkedQueue<String> lines = new ConcurrentLinkedQueue<>();

        private SyncTraceLogs() {
            start();
            logger.addAppender(this);
        }

        @Override
        protected void append(ILoggingEvent event) {
            if (event.getMessage().startsWith("event=worker_sync_trace ")) {
                lines.add(event.getFormattedMessage());
            }
        }

        @Override
        public void close() {
            logger.detachAppender(this);
            stop();
        }
    }
}
