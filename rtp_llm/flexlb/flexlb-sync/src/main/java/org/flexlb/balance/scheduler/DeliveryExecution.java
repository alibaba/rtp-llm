package org.flexlb.balance.scheduler;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.config.ConfigService;
import org.flexlb.metric.ExecutorStats;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.LongSupplier;

import static com.google.common.base.Preconditions.checkState;

/**
 * Batch admission, transport execution and real RPC response drain. No request policy lives here.
 */
public final class DeliveryExecution implements ExecutorStats {
    private static final int TIMESTAMP_BITS = 30;
    private static final int PROCESS_INSTANCE_BITS = 12;
    private static final int SEQUENCE_BITS = 21;
    private static final int TIMESTAMP_SHIFT = PROCESS_INSTANCE_BITS + SEQUENCE_BITS;
    private static final long EPOCH_SECONDS = 1_577_836_800L;
    private static final long PROCESS_INSTANCE_MASK = (1L << PROCESS_INSTANCE_BITS) - 1L;
    private static final long SEQUENCE_LIMIT = 1L << SEQUENCE_BITS;

    private final Consumer<Throwable> failures;

    private Throwable executionFailure;

    private final ThreadPoolExecutor dispatchExecutor;
    private final ThreadPoolExecutor completionExecutor;
    private final Semaphore deliveryPermits;

    private final int deliveryAdmissionCapacity;
    private final long maxInflightBytes;

    private final AtomicLong inflightPayloadBytes = new AtomicLong();
    private final AtomicInteger pendingDeliveryResponses = new AtomicInteger();
    private final ReentrantReadWriteLock deliveryLifecycle = new ReentrantReadWriteLock(true);
    private final Object deliveryDrain = new Object();
    private final CopyOnWriteArraySet<Runnable> deliveryCapacityListeners =
            new CopyOnWriteArraySet<>();

    private volatile boolean acceptingDeliveries = true;

    final AdmissionCapacity decodeDeliveryCapacity;

    final int maxInflightBatches;

    // IDs use 30 timestamp bits, 12 process bits and 21 sequence bits. Process hashes may collide;
    // the timestamp and sequence guarantee uniqueness within this executor's lifetime.
    private final long processInstanceId =
            hashProcessInstance(
                    detectLocalIp()
                            + ":"
                            + ProcessHandle.current().pid()
                            + ":"
                            + ManagementFactory.getRuntimeMXBean().getStartTime());

    private final LongSupplier currentTimeMillis = System::currentTimeMillis;

    private long lastTimestampSeconds = -1L;
    private long nextSequence;

    DeliveryExecution(ConfigService config, Consumer<Throwable> failures) {
        this.failures =
                cause -> {
                    synchronized (deliveryDrain) {
                        executionFailure = Failures.append(executionFailure, cause);
                    }
                    failures.accept(cause);
                };
        var configuration = config.loadBalanceConfig();
        var sizing = configuration.getInternalRuntime();
        int threads = sizing.getBatchDispatchThreads();
        deliveryAdmissionCapacity = Math.addExact(threads, sizing.getBatchDispatchQueueCapacity());
        maxInflightBytes = Math.max(1L, sizing.getBatchDispatchMaxInflightBytes());
        deliveryPermits = new Semaphore(deliveryAdmissionCapacity);
        dispatchExecutor =
                new ThreadPoolExecutor(
                        threads,
                        threads,
                        60L,
                        TimeUnit.SECONDS,
                        new LinkedBlockingQueue<>(),
                        new NamedThreadFactory("flexlb-dispatch-executor"),
                        new ThreadPoolExecutor.AbortPolicy());
        int completionThreads = sizing.getBatchDispatchCompletionThreads();
        completionExecutor =
                new ThreadPoolExecutor(
                        completionThreads,
                        completionThreads,
                        60L,
                        TimeUnit.SECONDS,
                        new LinkedBlockingQueue<>(),
                        new NamedThreadFactory("flexlb-dispatch-completion"),
                        new ThreadPoolExecutor.AbortPolicy());
        var availability = configuration.getRouter().getRoles().getDecode().getAvailability();
        decodeDeliveryCapacity =
                new AdmissionCapacity(
                        availability.getMaxEngineRequests() == null
                                ? 0L
                                : availability.getMaxEngineRequests(),
                        availability.getMaxKvUsagePercent());
        maxInflightBatches = configuration.getDispatcher().getMaxInflightPerPrefillWorker();
    }

    @Override
    public int activeThreads() {
        return dispatchExecutor.getActiveCount();
    }

    @Override
    public int queuedTasks() {
        return dispatchExecutor.getQueue().size();
    }

    @Override
    public int poolSize() {
        return dispatchExecutor.getPoolSize();
    }

    @Override
    public long completedTasks() {
        return dispatchExecutor.getCompletedTaskCount();
    }

    public void awaitDeliveries() {
        stopDeliveryAdmission();
        boolean interrupted = false;
        synchronized (deliveryDrain) {
            while (deliveryPermits.availablePermits() != deliveryAdmissionCapacity
                    || pendingDeliveryResponses.get() != 0) {
                try {
                    deliveryDrain.wait();
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        }
        while (!dispatchExecutor.isTerminated() || !completionExecutor.isTerminated()) {
            try {
                dispatchExecutor.awaitTermination(1, TimeUnit.DAYS);
                completionExecutor.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException ignored) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        synchronized (deliveryDrain) {
            Failures.rethrow(executionFailure, "delivery execution failed");
        }
    }

    synchronized long nextBatchId() {
        long timestampSeconds =
                TimeUnit.MILLISECONDS.toSeconds(currentTimeMillis.getAsLong()) - EPOCH_SECONDS;
        if (timestampSeconds < 0L || (timestampSeconds >>> TIMESTAMP_BITS) != 0L) {
            throw new IllegalStateException(
                    timestampSeconds < 0L
                            ? "system clock is before the batch ID epoch"
                            : "batch ID timestamp overflow");
        }
        if (timestampSeconds < lastTimestampSeconds) {
            throw new IllegalStateException(
                    "system clock moved backwards while generating batch IDs");
        }
        long sequence = timestampSeconds == lastTimestampSeconds ? nextSequence : 0L;
        // Zero denotes no batch in scheduler APIs; the first epoch second must skip it.
        if (timestampSeconds == 0L && processInstanceId == 0L && sequence == 0L) {
            sequence = 1L;
        }
        if (sequence >= SEQUENCE_LIMIT) {
            throw new IllegalStateException("batch ID sequence exhausted for the current second");
        }
        lastTimestampSeconds = timestampSeconds;
        nextSequence = sequence + 1L;
        return (timestampSeconds << TIMESTAMP_SHIFT)
                | (processInstanceId << SEQUENCE_BITS)
                | sequence;
    }

    CapacityBoundary.Attempt<DeliveryPermit> tryReserveDelivery() {
        var lock = deliveryLifecycle.readLock();
        lock.lock();
        try {
            if (!acceptingDeliveries) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.failed(
                                new IllegalStateException("delivery execution is closed")));
            }
            if (!deliveryPermits.tryAcquire()) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.unavailable(CapacityBoundary.Resource.DISPATCH_EXECUTOR));
            }
            try {
                return CapacityBoundary.Attempt.accepted(new DeliveryPermit());
            } catch (Throwable failure) {
                deliveryPermits.release();
                throw failure;
            }
        } finally {
            lock.unlock();
        }
    }

    boolean capacityAvailable(RouteResult item, CapacityBoundary.Resource resource) {
        return switch (resource) {
            case DISPATCH_EXECUTOR -> deliveryCapacityAvailable();
            case PREFILL_REQUEST ->
                    item.prefillEndpoint()
                            .requestCapacityAvailable(item.prefillEndpoint().requestLimit, 1);
            case PREFILL_BATCH -> item.prefillEndpoint().batchCapacityAvailable(maxInflightBatches);
            case DECODE_ENGINE ->
                    item.decodeEndpoint()
                            .shouldRetryDispatch(item.decodeReservation(), decodeDeliveryCapacity);
        };
    }

    boolean deliveryCapacityAvailable() {
        return !acceptingDeliveries || deliveryPermits.availablePermits() > 0;
    }

    Runnable watchDeliveryCapacity(Runnable wake) {
        deliveryCapacityListeners.add(wake);
        return () -> deliveryCapacityListeners.remove(wake);
    }

    void stopDeliveryAdmission() {
        var lock = deliveryLifecycle.writeLock();
        lock.lock();
        try {
            acceptingDeliveries = false;
        } finally {
            lock.unlock();
        }
        for (Runnable listener : deliveryCapacityListeners) {
            try {
                listener.run();
            } catch (Throwable failure) {
                failures.accept(failure);
            }
        }
        finishDeliveryDrain();
    }

    private static long hashProcessInstance(String processIdentity) {
        try {
            byte[] hash =
                    MessageDigest.getInstance("SHA-256")
                            .digest(processIdentity.getBytes(StandardCharsets.UTF_8));
            long lowBytes =
                    Byte.toUnsignedInt(hash[hash.length - 1])
                            | (Byte.toUnsignedInt(hash[hash.length - 2]) << Byte.SIZE);
            return lowBytes & PROCESS_INSTANCE_MASK;
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 not available", impossible);
        }
    }

    private static String detectLocalIp() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException failure) {
            Logger.warn("Failed to detect local IP for batch IDs; using loopback", failure);
            return "127.0.0.1";
        }
    }

    private <T> void observeDeliveryResponse(
            CompletableFuture<T> response, BiConsumer<T, Throwable> handler) {
        pendingDeliveryResponses.incrementAndGet();
        var started = new AtomicBoolean();
        try {
            response.handleAsync(
                            (reply, failure) -> {
                                started.set(true);
                                handler.accept(reply, failure);
                                return null;
                            },
                            completionExecutor)
                    .whenComplete(
                            (ignored, failure) -> {
                                if (failure != null && !started.get()) {
                                    // The response executor never ran the handler: deliver the
                                    // uncertain fact once.
                                    try {
                                        handler.accept(null, failure);
                                    } catch (Throwable callbackFailure) {
                                        failures.accept(callbackFailure);
                                    }
                                } else if (failure != null) {
                                    failures.accept(failure);
                                }
                                finishDeliveryResponse();
                            });
        } catch (Throwable registrationFailure) {
            try {
                handler.accept(null, registrationFailure);
            } catch (Throwable callbackFailure) {
                failures.accept(callbackFailure);
            } finally {
                finishDeliveryResponse();
            }
        }
    }

    private void finishDeliveryResponse() {
        pendingDeliveryResponses.decrementAndGet();
        finishDeliveryDrain();
    }

    private void finishDeliveryDrain() {
        if (!acceptingDeliveries
                && deliveryPermits.availablePermits() == deliveryAdmissionCapacity
                && pendingDeliveryResponses.get() == 0) {
            dispatchExecutor.shutdown();
            completionExecutor.shutdown();
            synchronized (deliveryDrain) {
                deliveryDrain.notifyAll();
            }
        }
    }

    private void releaseDeliveryPermit() {
        deliveryPermits.release();
        for (Runnable listener : deliveryCapacityListeners) {
            try {
                listener.run();
            } catch (Throwable failure) {
                failures.accept(failure);
            }
        }
        finishDeliveryDrain();
    }

    /**
     * One infrastructure permit. No business state or callbacks run under an infrastructure lock.
     */
    final class DeliveryPermit implements AutoCloseable {
        private static final int RESERVED = 0;
        private static final int SUBMITTED = 1;
        private static final int RUNNING = 2;
        private static final int RELEASED = 3;
        private static final int AWAITING_RESPONSE = 4;

        // RESERVED -> SUBMITTED -> RUNNING -> AWAITING_RESPONSE -> RELEASED.
        // close releases RESERVED; rejection releases SUBMITTED; sender exit without a response
        // releases RUNNING.
        // Only the CAS winner entering RELEASED returns the charged bytes and infrastructure
        // permit.
        private final AtomicInteger state = new AtomicInteger(RESERVED);
        // Only the sender charges bytes; the state CAS publishes them to the one releaser.
        private long chargedBytes;

        @Override
        public void close() {
            if (state.compareAndSet(RESERVED, RELEASED)) {
                release();
            }
        }

        boolean tryChargeBytes(long bytes) {
            checkState(
                    state.get() == RUNNING && chargedBytes == 0L,
                    "delivery is not ready for payload admission");
            long current;
            do {
                current = inflightPayloadBytes.get();
                if (bytes > maxInflightBytes - current) {
                    return false;
                }
            } while (!inflightPayloadBytes.compareAndSet(current, current + bytes));
            chargedBytes = bytes;
            return true;
        }

        void submit(Runnable work, Consumer<Throwable> rejected) {
            Objects.requireNonNull(work, "work");
            Objects.requireNonNull(rejected, "rejected");
            checkState(
                    state.compareAndSet(RESERVED, SUBMITTED), "delivery permit already resolved");
            try {
                dispatchExecutor.execute(
                        () -> {
                            if (!state.compareAndSet(SUBMITTED, RUNNING)) {
                                return;
                            }
                            try {
                                work.run();
                            } catch (Throwable failure) {
                                failures.accept(failure);
                            } finally {
                                releaseIfNoResponse();
                            }
                        });
            } catch (RuntimeException | Error failure) {
                if (!state.compareAndSet(SUBMITTED, RELEASED)) {
                    failures.accept(failure);
                    return;
                }
                release();
                try {
                    rejected.accept(failure);
                } catch (Throwable cleanupFailure) {
                    failures.accept(Failures.append(failure, cleanupFailure));
                }
            }
        }

        void releaseIfNoResponse() {
            if (state.compareAndSet(RUNNING, RELEASED)) {
                release();
            }
        }

        <T> void observeResponse(CompletableFuture<T> response, BiConsumer<T, Throwable> handler) {
            checkState(
                    state.compareAndSet(RUNNING, AWAITING_RESPONSE),
                    "delivery response already observed");
            DeliveryExecution.this.observeDeliveryResponse(
                    response,
                    (reply, failure) -> {
                        // Return capacity before request callbacks can submit another batch.
                        if (state.compareAndSet(AWAITING_RESPONSE, RELEASED)) {
                            release();
                        }
                        handler.accept(reply, failure);
                    });
        }

        private void release() {
            inflightPayloadBytes.addAndGet(-chargedBytes);
            releaseDeliveryPermit();
        }
    }
}
