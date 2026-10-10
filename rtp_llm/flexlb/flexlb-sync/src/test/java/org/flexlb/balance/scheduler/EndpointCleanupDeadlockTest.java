package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillCleanupDeadlockFixture;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

/**
 * Cleanup must finish while a business operation holds the context monitor and needs
 * the endpoint lock. A child JVM contains any regression to a non-interruptible deadlock.
 */
class EndpointCleanupDeadlockTest {

    @TempDir Path outputDirectory;

    @Test
    void decodeCleanupAndDeliveryFinish() throws Exception {
        verifyCompletion("decode");
    }

    @Test
    void prefillCleanupAndReservationFinish() throws Exception {
        verifyCompletion("prefill");
    }

    @Test
    void decodeCleanupAndDispatchRejectionFinish() throws Exception {
        verifyCompletion("decode-rejection");
    }

    @Test
    void prefillIndividualCleanupAndReservationFinish() throws Exception {
        verifyCompletion("prefill-individual");
    }

    @Test
    void timerCloseDoesNotHoldRegistrationLockWhileWaitingForContext() throws Exception {
        verifyCompletion("timer-close");
    }

    private void verifyCompletion(String endpoint) throws Exception {
        Path output = outputDirectory.resolve(endpoint + ".txt");
        String classpath = System.getProperty("surefire.test.class.path",
                System.getProperty("java.class.path"));
        Process child = new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-cp", classpath, Reproducer.class.getName(), endpoint)
                .redirectErrorStream(true).redirectOutput(output.toFile()).start();
        try {
            assertTrue(child.waitFor(30, TimeUnit.SECONDS), "child timed out: " + output);
            String evidence = Files.readString(output);
            assertEquals(0, child.exitValue(), evidence);
            assertTrue(evidence.contains("COMPLETED " + endpoint), evidence);
        } finally {
            if (child.isAlive()) {
                child.destroyForcibly();
                assertTrue(child.waitFor(5, TimeUnit.SECONDS));
            }
        }
    }

    public static class Reproducer {

        public static void main(String[] args) {
            try {
                run(args[0]);
                System.exit(0);
            } catch (Throwable failure) {
                failure.printStackTrace();
                System.exit(1);
            }
        }

        private static void verifyTimerClose() throws Exception {
            var config = SchedulingTestConfig.batchConfig();
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            AbstractRequestScheduler registry = mock(AbstractRequestScheduler.class);
            ExpirationTimer timer = new ExpirationTimer(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry));
            RequestContext requestContext = RequestProtocolTestSupport.context(config, 992L);
            AbstractRequestScheduler requestOwner = RequestProtocolTestSupport.initialize(mock(ResponseCompletionExecutor.class), requestContext, timer);

            CountDownLatch contextHeld = new CountDownLatch(1);
            CountDownLatch closeReachesContexts = new CountDownLatch(1);
            AtomicReference<Throwable> failure = new AtomicReference<>();
            doAnswer(call -> {
                closeReachesContexts.countDown();
                return java.util.List.of(requestContext);
            }).when(SchedulerTestSupport.repository(registry)).snapshotActive();
            Thread holder = new Thread(() -> {
                try {
                    synchronized (requestContext) {
                        contextHeld.countDown();
                        assertTrue(closeReachesContexts.await(5, TimeUnit.SECONDS));
                        // close() is about to acquire Context. Registration must remain available
                        // to reject this late request; holding it while waiting for Context deadlocks.
                        assertThrows(java.util.concurrent.RejectedExecutionException.class, () -> timer.scheduleRequestDeadline(requestContext, Long.MAX_VALUE));
                    }
                } catch (Throwable error) {
                    failure.set(error);
                }
            }, "context-checks-timer-registration");
            Thread closer = new Thread(() -> {
                try {
                    assertTrue(contextHeld.await(5, TimeUnit.SECONDS));
                    timer.close();
                } catch (Throwable error) {
                    failure.set(error);
                }
            }, "timer-close-waits-for-context");
            holder.setDaemon(true);
            closer.setDaemon(true);
            holder.start();
            closer.start();
            holder.join(8_000);
            closer.join(8_000);
            if (failure.get() != null) {
                throw new AssertionError(failure.get());
            }
            assertFalse(holder.isAlive() || closer.isAlive(), "Context / timer registration lock cycle");
        }

        private static void run(String kind) throws Exception {
            if (kind.equals("timer-close")) {
                verifyTimerClose();
                System.out.println("COMPLETED " + kind);
                return;
            }
            var config = SchedulingTestConfig.batchConfig();
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            AbstractRequestScheduler registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
            long id = 991L;
            var context = RequestProtocolTestSupport.context(config, id);
            var future = RequestProtocolTestSupport.register(registry, context);
            RequestContext requestContext = registry.findRequestContext(id);
            CountDownLatch contextHeld = new CountDownLatch(1);
            CountDownLatch endpointHeld = new CountDownLatch(1);
            AtomicReference<Throwable> failure = new AtomicReference<>();
            LongPredicate ownership = requestId -> {
                // This callback is invoked by the real endpoint sweep under its real lock.
                endpointHeld.countDown();
                return org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).retainsIdentity(requestId);
            };
            Runnable sweep;
            Runnable endpointOperation;
            if (kind.startsWith("decode")) {
                var endpoint = spy(EndpointTestSupport.decode(WorkerStatus.createDiscovered(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(mock(AbstractRequestScheduler.class))));
                DecodeResources.ReservationHandle reservation;
                try (var pin = endpoint.tryPinGeneration()) {
                    assertNotNull(pin);
                    reservation = EndpointTestSupport.reserveUnqueuedDecode(endpoint, pin, id, 1L, 1L, 50);
                }
                assertNotNull(reservation);
                context.setFuture(future);
                var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new org.flexlb.dao.loadbalance.Response(), null, null, null, endpoint, reservation, System.currentTimeMillis());
                var registered = new RequestProtocolTestSupport.Registered(item, future);
                RequestProtocolTestSupport.bindRoute(registry, registered);
                var claim = RequestProtocolTestSupport.claimBatchWithoutPrediction(registry, item, 1L, () -> true);
                assertNotNull(claim);
                // Negative TTL makes the fixture eligible without a wall-clock sleep.
                sweep = () -> assertEquals(0, endpoint.evictExpiredRequests(-1L, ownership));
                if (kind.equals("decode-rejection")) {
                    doAnswer(invocation -> {
                        assertFalse(Thread.holdsLock(requestContext), "reservation release must not hold the context monitor");
                        contextHeld.countDown();
                        assertTrue(endpointHeld.await(5, TimeUnit.SECONDS));
                        return invocation.callRealMethod();
                    }).when(endpoint).release(reservation, DecodeResources.ReleaseReason.NOT_SENT);
                }
                endpointOperation = switch(kind) {
                    case "decode-rejection" ->
                        () -> claim.item.ctx().scheduler().completeDelivery(claim, org.flexlb.balance.delivery.DeliveryResult.notSent(new IllegalStateException("not sent")));
                    default ->
                        () -> registry.setDeliveryPrediction(claim, new org.flexlb.balance.projection.WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 30_000L);
                };
            } else {
                boolean batch = kind.equals("prefill");
                var fixture = new PrefillCleanupDeadlockFixture(id, batch);
                sweep = batch ? () -> fixture.sweepBatches(ownership) : () -> fixture.sweepIndividuals(ownership);
                endpointOperation = fixture::reserveNextBatch;
            }
            // Reproduce the production lock order, without mocking either lock or ownership check.
            // The context monitor is held explicitly to isolate the inversion from admission setup.
            Thread holder = new Thread(() -> {
                try {
                    if (kind.equals("decode-rejection")) {
                        // Reservation release and cleanup can contend for the endpoint without holding Context.
                        endpointOperation.run();
                    } else {
                        synchronized (requestContext) {
                            contextHeld.countDown();
                            assertTrue(endpointHeld.await(5, TimeUnit.SECONDS));
                            endpointOperation.run();
                        }
                    }
                } catch (Throwable e) {
                    failure.set(e);
                }
            }, "context-waits-for-endpoint");
            Thread cleaner = new Thread(() -> {
                try {
                    assertTrue(contextHeld.await(5, TimeUnit.SECONDS));
                    sweep.run();
                } catch (Throwable e) {
                    failure.set(e);
                }
            }, "cleanup-waits-for-context");
            holder.setDaemon(true);
            cleaner.setDaemon(true);
            holder.start();
            cleaner.start();
            holder.join(8_000);
            cleaner.join(8_000);
            if (failure.get() != null) {
                throw new AssertionError(failure.get());
            }
            if (holder.isAlive() || cleaner.isAlive()) {
                var bean = ManagementFactory.getThreadMXBean();
                for (var info : bean.getThreadInfo(new long[] { holder.threadId(), cleaner.threadId() }, true, true)) {
                    System.err.println(info);
                }
                throw new AssertionError("Cleanup and business operation did not finish");
            }
            System.out.println("COMPLETED " + kind);
        }
    }
}
