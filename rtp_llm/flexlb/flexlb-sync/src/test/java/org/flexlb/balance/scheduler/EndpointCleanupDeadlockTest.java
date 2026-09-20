package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillCleanupDeadlockFixture;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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
 * Cleanup must finish while a business operation holds the Slot monitor and needs
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
    void decodeCleanupAndWorkerTerminalFinish() throws Exception {
        verifyCompletion("decode-terminal");
    }

    @Test
    void timerCloseDoesNotHoldRegistrationLockWhileWaitingForSlot() throws Exception {
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
            RequestRegistry registry = mock(RequestRegistry.class);
            ExpirationTimer timer = new ExpirationTimer(registry, service);
            RequestSlot slot = new RequestSlot(mock(RequestCompletionPublisher.class), RequestLifecycleTestSupport.context(config, 992L),
                    timer, new RequestTerminalCleanup(timer), () -> { });
            CountDownLatch slotHeld = new CountDownLatch(1);
            CountDownLatch closeReachesSlots = new CountDownLatch(1);
            AtomicReference<Throwable> failure = new AtomicReference<>();
            doAnswer(call -> {
                closeReachesSlots.countDown();
                return java.util.List.of(slot);
            }).when(registry).snapshotSlots();
            Thread holder = new Thread(() -> {
                try {
                    synchronized (slot) {
                        slotHeld.countDown();
                        assertTrue(closeReachesSlots.await(5, TimeUnit.SECONDS));
                        // close() is about to acquire Slot. Registration must remain available
                        // to reject this late request; holding it while waiting for Slot deadlocks.
                        assertThrows(java.util.concurrent.RejectedExecutionException.class,
                                () -> timer.attachRequestDeadline(slot, Long.MAX_VALUE));
                    }
                } catch (Throwable error) { failure.set(error); }
            }, "slot-checks-timer-registration");
            Thread closer = new Thread(() -> {
                try {
                    assertTrue(slotHeld.await(5, TimeUnit.SECONDS));
                    timer.close();
                } catch (Throwable error) { failure.set(error); }
            }, "timer-close-waits-for-slot");
            holder.setDaemon(true);
            closer.setDaemon(true);
            holder.start();
            closer.start();
            holder.join(8_000);
            closer.join(8_000);
            if (failure.get() != null) { throw new AssertionError(failure.get()); }
            assertFalse(holder.isAlive() || closer.isAlive(), "Slot / timer registration lock cycle");
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
            RequestRegistry registry = new RequestRegistry(service,
                    mock(BatchSchedulerReporter.class), mock(RequestSchedulerReporter.class));
            long id = 991L;
            var context = RequestLifecycleTestSupport.context(config, id);
            var future = registry.register(context);
            RequestSlot slot = registry.requestSlot(id);
            CountDownLatch slotHeld = new CountDownLatch(1);
            CountDownLatch endpointHeld = new CountDownLatch(1);
            AtomicReference<Throwable> failure = new AtomicReference<>();
            LongPredicate ownership = requestId -> {
                // This callback is invoked by the real endpoint sweep under its real lock.
                endpointHeld.countDown();
                return registry.retainForSchedulerCleanup(requestId);
            };
            Runnable sweep;
            Runnable endpointOperation;
            if (kind.startsWith("decode")) {
                var endpoint = spy(new DecodeEndpoint(WorkerStatus.createDiscovered(
                        RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null),
                        mock(EndpointEventProjector.class)));
                DecodeEndpoint.ReservationHandle reservation;
                try (var pin = endpoint.tryPinGeneration()) {
                    assertNotNull(pin);
                    reservation = endpoint.reserveUnqueued(pin, id, 1L, 1L, 50);
                }
                assertNotNull(reservation);
                var item = new ScheduledRequest(context, future,
                        new org.flexlb.dao.loadbalance.Response(), null, null, null,
                        endpoint, reservation, System.currentTimeMillis());
                var registered = new RequestLifecycleTestSupport.Registered(item, future);
                RequestLifecycleTestSupport.bindRoute(registry, registered);
                var claim = RequestLifecycleTestSupport.claimBatchWithoutPrediction(
                        registry, item, 1L, () -> true);
                assertNotNull(claim);
                // Negative TTL makes the fixture eligible without a wall-clock sleep.
                sweep = () -> assertEquals(0, endpoint.evictExpiredRequests(-1L, ownership));
                if (kind.equals("decode-rejection")) {
                    doAnswer(invocation -> {
                        assertFalse(Thread.holdsLock(slot), "reservation release must not hold the Slot monitor");
                        slotHeld.countDown();
                        assertTrue(endpointHeld.await(5, TimeUnit.SECONDS));
                        return invocation.callRealMethod();
                    }).when(endpoint).release(reservation, DecodeEndpoint.ReleaseReason.NOT_SENT);
                }
                if (kind.equals("decode-terminal")) {
                    doAnswer(invocation -> {
                        assertFalse(Thread.holdsLock(slot), "Worker cleanup must not retain Slot");
                        slotHeld.countDown();
                        assertTrue(endpointHeld.await(5, TimeUnit.SECONDS));
                        return invocation.callRealMethod();
                    }).when(endpoint).release(reservation, DecodeEndpoint.ReleaseReason.COUNTERPART_FINISHED);
                }
                endpointOperation = switch (kind) {
                    case "decode-rejection" -> () -> claim.complete(
                            org.flexlb.balance.delivery.DeliveryResult.notSent(new IllegalStateException("not sent")));
                    case "decode-terminal" -> () -> slot.processDecodeStatus(endpoint,
                            DecodeEndpoint.WorkerStatusFact.terminal(reservation, 0L));
                    default -> () -> registry.setDeliveryPrediction(claim,
                            new org.flexlb.balance.projection.WorkSnapshot(
                                    System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 30_000L);
                };
            } else {
                boolean batch = kind.equals("prefill");
                var fixture = new PrefillCleanupDeadlockFixture(id, batch);
                sweep = batch ? () -> fixture.sweepBatches(ownership)
                        : () -> fixture.sweepIndividuals(ownership);
                endpointOperation = fixture::reserveNextBatch;
            }
            // Reproduce the production lock order, without mocking either lock or ownership check.
            // The slot monitor is held explicitly to isolate the inversion from admission setup.
            Thread holder = new Thread(() -> {
                try {
                    if (kind.equals("decode-rejection") || kind.equals("decode-terminal")) {
                        // Reservation release and cleanup can contend for the endpoint without holding Slot.
                        endpointOperation.run();
                    } else {
                        synchronized (slot) {
                            slotHeld.countDown();
                            assertTrue(endpointHeld.await(5, TimeUnit.SECONDS));
                            endpointOperation.run();
                        }
                    }
                } catch (Throwable e) { failure.set(e); }
            }, "slot-waits-for-endpoint");
            Thread cleaner = new Thread(() -> {
                try {
                    assertTrue(slotHeld.await(5, TimeUnit.SECONDS));
                    sweep.run();
                } catch (Throwable e) { failure.set(e); }
            }, "cleanup-waits-for-slot");
            holder.setDaemon(true);
            cleaner.setDaemon(true);
            holder.start();
            cleaner.start();
            holder.join(8_000);
            cleaner.join(8_000);
            if (failure.get() != null) { throw new AssertionError(failure.get()); }
            if (holder.isAlive() || cleaner.isAlive()) {
                var bean = ManagementFactory.getThreadMXBean();
                for (var info : bean.getThreadInfo(
                        new long[] {holder.threadId(), cleaner.threadId()}, true, true)) {
                    System.err.println(info);
                }
                throw new AssertionError("Cleanup and business operation did not finish");
            }
            System.out.println("COMPLETED " + kind);
        }
    }
}
