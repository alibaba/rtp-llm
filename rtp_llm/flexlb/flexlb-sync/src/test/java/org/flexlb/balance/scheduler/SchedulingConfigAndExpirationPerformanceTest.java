package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.InflightRegistrar;
import org.flexlb.balance.scheduler.priority.PriorityAdmissionScheduler;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigValidationException;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.FifoOrderingConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.config.QueueSchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Regression and low-jitter performance coverage for the public scheduling
 * mode matrix and its single absolute request-expiration timer.
 */
@Tag("performance-regression")
class SchedulingConfigAndExpirationPerformanceTest {

    private static final int TIMER_REQUEST_COUNT = 512;
    private static final int EARLY_COMPLETION_COUNT = TIMER_REQUEST_COUNT / 2;

    @TestFactory
    Stream<DynamicTest> parsesSupportedSchedulingModeMatrix() {
        return List.of(
                new ModeCase("direct_non_batch", """
                        {
                          "schemaVersion": 1,
                          "scheduler": {"type": "DIRECT"},
                          "dispatcher": {"type": "NON_BATCH"}
                        }
                        """, true, false, false),
                new ModeCase("queue_fifo_batch", """
                        {
                          "schemaVersion": 1,
                          "scheduler": {
                            "type": "QUEUE",
                            "ordering": {"type": "FIFO"}
                          },
                          "dispatcher": {"type": "BATCH", "maxRequests": 11}
                        }
                        """, false, false, true),
                new ModeCase("queue_fifo_non_batch", """
                        {
                          "schemaVersion": 1,
                          "scheduler": {
                            "type": "QUEUE",
                            "ordering": {"type": "FIFO"}
                          },
                          "dispatcher": {
                            "type": "NON_BATCH",
                            "maxInflightRequestsPerPrefillWorker": 19
                          }
                        }
                        """, false, false, false),
                new ModeCase("queue_priority_batch", """
                        {
                          "schemaVersion": 1,
                          "scheduler": {
                            "type": "QUEUE",
                            "ordering": {"type": "PRIORITY", "defaultPriority": 73}
                          },
                          "dispatcher": {"type": "BATCH", "maxRequests": 11}
                        }
                        """, false, true, true),
                new ModeCase("queue_priority_non_batch", """
                        {
                          "schemaVersion": 1,
                          "scheduler": {
                            "type": "QUEUE",
                            "ordering": {"type": "PRIORITY", "defaultPriority": 73}
                          },
                          "dispatcher": {
                            "type": "NON_BATCH",
                            "maxInflightRequestsPerPrefillWorker": 19
                          }
                        }
                        """, false, true, false))
                .stream()
                .map(mode -> DynamicTest.dynamicTest(mode.name(), () -> assertMode(mode)));
    }

    @Test
    void rejectsInvalidModeCombinationsAndInactiveTaggedUnionFields() {
        List<String> invalidDocuments = List.of(
                """
                        {
                          "scheduler": {"type": "DIRECT"},
                          "dispatcher": {"type": "BATCH"}
                        }
                        """,
                """
                        {
                          "scheduler": {"type": "DIRECT"},
                          "dispatcher": {
                            "type": "NON_BATCH",
                            "maxInflightRequestsPerPrefillWorker": 19
                          }
                        }
                        """,
                """
                        {
                          "scheduler": {
                            "type": "DIRECT",
                            "ordering": {"type": "FIFO"}
                          },
                          "dispatcher": {"type": "NON_BATCH"}
                        }
                        """,
                """
                        {
                          "scheduler": {
                            "type": "QUEUE",
                            "ordering": {"type": "FIFO", "defaultPriority": 73}
                          },
                          "dispatcher": {"type": "BATCH"}
                        }
                        """,
                """
                        {
                          "scheduler": {
                            "type": "QUEUE",
                            "ordering": {"type": "PRIORITY"}
                          },
                          "dispatcher": {
                            "type": "NON_BATCH",
                            "maxRequests": 11
                          }
                        }
                        """);

        for (String document : invalidDocuments) {
            assertThrows(ConfigValidationException.class, () -> ConfigService.parse(document));
        }
    }

    @Test
    @Timeout(15)
    void sharedExpirationTimerExpiresPendingRequestsAndEagerlyRemovesCompletedOnes()
            throws Exception {
        FlexlbConfig priorityConfig = ConfigService.parse("""
                {
                  "scheduler": {
                    "type": "QUEUE",
                    "ordering": {"type": "PRIORITY", "defaultPriority": 50}
                  },
                  "dispatcher": {"type": "NON_BATCH"}
                }
                """);
        ConfigService configService = new StaticConfigService(priorityConfig);
        HoldingAdmissionScheduler admissionScheduler = new HoldingAdmissionScheduler();
        PriorityScheduler scheduler = new PriorityScheduler(
                configService,
                mock(Router.class),
                mock(EndpointRegistry.class),
                mock(BatchDispatcher.class),
                mock(BatchSchedulerReporter.class),
                admissionScheduler,
                null,
                mock(EngineCancelChannel.class));

        try {
            assertTrue(scheduler.removesCanceledRequestExpirations());

            // Prime the timer thread, while proving that a completed request is
            // not retained until its deliberately distant absolute deadline.
            long warmupExpiresAtMs = System.currentTimeMillis()
                    + TimeUnit.MINUTES.toMillis(1);
            List<CompletableFuture<Response>> warmup = new ArrayList<>(32);
            for (int index = 0; index < 32; index++) {
                warmup.add(scheduler.submit(context(10_000L + index, warmupExpiresAtMs)));
            }
            awaitCondition(() -> scheduler.requestExpirationQueueSize() == warmup.size(), 1_000);
            Response warmupSuccess = successResponse();
            warmup.forEach(future -> future.complete(warmupSuccess));
            awaitCondition(() -> scheduler.requestExpirationQueueSize() == 0, 1_000);

            long clockBaseMs = System.currentTimeMillis();
            long clockBaseNanos = System.nanoTime();
            long expirationDelayMs = 500;
            long expiresAtMs = clockBaseMs + expirationDelayMs;
            long expectedExpirationNanos = clockBaseNanos
                    + TimeUnit.MILLISECONDS.toNanos(expirationDelayMs);

            List<CompletableFuture<Response>> futures =
                    new ArrayList<>(TIMER_REQUEST_COUNT);
            AtomicLong firstExpirationNanos = new AtomicLong(Long.MAX_VALUE);
            AtomicLong lastExpirationNanos = new AtomicLong(Long.MIN_VALUE);
            for (int index = 0; index < TIMER_REQUEST_COUNT; index++) {
                CompletableFuture<Response> future = scheduler.submit(
                        context(20_000L + index, expiresAtMs));
                if (index >= EARLY_COMPLETION_COUNT) {
                    future.whenComplete((ignored, error) -> {
                        long completedAt = System.nanoTime();
                        firstExpirationNanos.accumulateAndGet(completedAt, Math::min);
                        lastExpirationNanos.accumulateAndGet(completedAt, Math::max);
                    });
                }
                futures.add(future);
            }

            Response earlySuccess = successResponse();
            for (int index = 0; index < EARLY_COMPLETION_COUNT; index++) {
                assertTrue(futures.get(index).complete(earlySuccess));
            }

            CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new))
                    .get(3, TimeUnit.SECONDS);

            for (int index = 0; index < EARLY_COMPLETION_COUNT; index++) {
                assertSame(earlySuccess, futures.get(index).join());
            }
            assertEquals(8511, StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                    "request expiration must preserve the public 8511 contract");
            for (int index = EARLY_COMPLETION_COUNT; index < TIMER_REQUEST_COUNT; index++) {
                Response timeout = futures.get(index).join();
                assertFalse(timeout.isSuccess());
                assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                        timeout.getCode());
            }

            awaitCondition(() -> scheduler.requestExpirationQueueSize() == 0, 1_000);
            assertEquals(32 + TIMER_REQUEST_COUNT, admissionScheduler.scheduledCount());

            long earlyToleranceNanos = TimeUnit.MILLISECONDS.toNanos(50);
            long maximumTailNanos = TimeUnit.MILLISECONDS.toNanos(1_500);
            assertTrue(firstExpirationNanos.get()
                            >= expectedExpirationNanos - earlyToleranceNanos,
                    "the absolute-expiration timer fired materially early");
            assertTrue(lastExpirationNanos.get()
                            <= expectedExpirationNanos + maximumTailNanos,
                    "the shared timer did not drain its bounded burst promptly");
            assertTrue(lastExpirationNanos.get() - firstExpirationNanos.get()
                            <= maximumTailNanos,
                    "same-deadline completions were spread across an unstable tail");
        } finally {
            scheduler.shutdown();
            admissionScheduler.shutdown();
        }
    }

    private static void assertMode(ModeCase mode) {
        FlexlbConfig config = ConfigService.parse(mode.document());

        assertEquals(mode.direct(), config.isDirect());
        assertEquals(!mode.direct(), config.isQueue());
        assertEquals(mode.priority(), config.isPriorityOrdering());
        assertEquals(mode.batch(), config.isBatchDispatch());

        if (mode.direct()) {
            assertInstanceOf(DirectSchedulerConfig.class, config.getScheduler());
        } else {
            QueueSchedulerConfig queue = assertInstanceOf(
                    QueueSchedulerConfig.class, config.getScheduler());
            if (mode.priority()) {
                PriorityOrderingConfig priority = assertInstanceOf(
                        PriorityOrderingConfig.class, queue.getOrdering());
                assertEquals(73, priority.getDefaultPriority());
            } else {
                assertInstanceOf(FifoOrderingConfig.class, queue.getOrdering());
            }
        }

        if (mode.batch()) {
            BatchDispatcherConfig batch = assertInstanceOf(
                    BatchDispatcherConfig.class, config.getDispatcher());
            assertEquals(11, batch.getMaxRequests());
            assertTrue(config.isBatchDispatch());
        } else {
            NonBatchDispatcherConfig nonBatch = assertInstanceOf(
                    NonBatchDispatcherConfig.class, config.getDispatcher());
            if (mode.direct()) {
                assertNull(nonBatch.getMaxInflightRequestsPerPrefillWorker());
            } else {
                assertEquals(19, nonBatch.getMaxInflightRequestsPerPrefillWorker());
            }
            assertFalse(config.isBatchDispatch());
        }
    }

    private static BalanceContext context(long requestId, long expiresAtMs) {
        Request request = new Request();
        request.setRequestId(requestId);

        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, expiresAtMs));
        return context;
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }

    private static void awaitCondition(BooleanSupplier condition, long timeoutMs)
            throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (!condition.getAsBoolean() && System.nanoTime() < deadlineNanos) {
            Thread.sleep(1);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true before timeout");
    }

    private record ModeCase(
            String name,
            String document,
            boolean direct,
            boolean priority,
            boolean batch) {
    }

    /** Avoid Mockito interception in the submit hot path measured by this regression. */
    private static final class StaticConfigService extends ConfigService {
        private final FlexlbConfig config;

        private StaticConfigService(FlexlbConfig config) {
            this.config = config;
        }

        @Override
        public FlexlbConfig loadBalanceConfig() {
            return config;
        }
    }

    /**
     * Deliberately holds requests in pre-registration admission so only the
     * shared absolute-expiration timer can complete them.
     */
    private static final class HoldingAdmissionScheduler extends PriorityAdmissionScheduler {
        private final AtomicLong scheduled = new AtomicLong();

        private HoldingAdmissionScheduler() {
            super(null, null, null, null, null, null, null, null);
        }

        @Override
        public void schedule(BalanceContext ctx,
                             CompletableFuture<Response> future,
                             InflightRegistrar registrar) {
            scheduled.incrementAndGet();
        }

        private long scheduledCount() {
            return scheduled.get();
        }
    }
}
