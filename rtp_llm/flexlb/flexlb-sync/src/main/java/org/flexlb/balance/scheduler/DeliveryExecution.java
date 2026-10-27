package org.flexlb.balance.scheduler;

import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.delivery.BatchIdGenerator;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.config.ConfigService;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.SafeFlexMonitor;
import org.flexlb.util.Failures;
import org.springframework.core.env.Environment;

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

import static com.google.common.base.Preconditions.checkState;

/**
 * Batch admission, transport execution and real RPC response drain. No request policy lives here.
 */
final class DeliveryExecution {
    private final Consumer<Throwable> failures;
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
    private BatchIdGenerator batchIds;
    private static final RequestTimeEstimation.AdmissionBlockSemantics DELIVERY_CAPACITY_BLOCK =
            new RequestTimeEstimation.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RequestTimeEstimation.AfterProbeAdmission.BLOCKED,
                    "DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RoleType.PREFILL);

    void configureBatchIds(Environment environment) {
        if (batchIds == null) {
            return;
        }
        try {
            batchIds =
                    BatchIdGenerator.forCurrentProcess(
                            Integer.parseInt(
                                    environment.getProperty(
                                            "server.port",
                                            System.getProperty("server.port", "7001"))));
        } catch (NumberFormatException ignored) {
        }
    }

    void configureDeliveryMetrics(MeterRegistry registry) {
        SafeFlexMonitor.run(
                MetricConstant.DISPATCH_EXECUTOR_ACTIVE_THREADS,
                () ->
                        Gauge.builder(
                                        "flexlb." + MetricConstant.DISPATCH_EXECUTOR_ACTIVE_THREADS,
                                        dispatchExecutor,
                                        ThreadPoolExecutor::getActiveCount)
                                .description("Dispatch executor active thread count")
                                .register(registry));
        SafeFlexMonitor.run(
                MetricConstant.DISPATCH_EXECUTOR_QUEUE_SIZE,
                () ->
                        Gauge.builder(
                                        "flexlb." + MetricConstant.DISPATCH_EXECUTOR_QUEUE_SIZE,
                                        dispatchExecutor,
                                        executor -> executor.getQueue().size())
                                .description("Dispatch executor pending task queue size")
                                .register(registry));
        SafeFlexMonitor.run(
                MetricConstant.DISPATCH_EXECUTOR_POOL_SIZE,
                () ->
                        Gauge.builder(
                                        "flexlb." + MetricConstant.DISPATCH_EXECUTOR_POOL_SIZE,
                                        dispatchExecutor,
                                        ThreadPoolExecutor::getPoolSize)
                                .description("Dispatch executor current pool size")
                                .register(registry));
        SafeFlexMonitor.run(
                MetricConstant.DISPATCH_EXECUTOR_COMPLETED_TASKS,
                () ->
                        FunctionCounter.builder(
                                        "flexlb."
                                                + MetricConstant.DISPATCH_EXECUTOR_COMPLETED_TASKS,
                                        dispatchExecutor,
                                        ThreadPoolExecutor::getCompletedTaskCount)
                                .description("Dispatch executor total completed tasks")
                                .register(registry));
    }

    void reportExecutorMetrics(org.flexlb.service.monitor.DeliveryMetricsReporter reporter) {
        reporter.reportDispatchExecutorMetrics(
                dispatchExecutor.getActiveCount(),
                dispatchExecutor.getQueue().size(),
                dispatchExecutor.getPoolSize());
    }

    long nextBatchId() {
        return batchIds.nextId();
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
                        CapacityBoundary.unavailable(
                                CapacityBoundary.Resource.DISPATCH_EXECUTOR,
                                DELIVERY_CAPACITY_BLOCK));
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

    boolean deliveryCapacityAvailable() {
        return !acceptingDeliveries || deliveryPermits.availablePermits() > 0;
    }

    Runnable watchDeliveryCapacity(Runnable wake) {
        deliveryCapacityListeners.add(wake);
        return () -> deliveryCapacityListeners.remove(wake);
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
        private final AtomicLong chargedBytes = new AtomicLong();

        boolean tryChargeBytes(long bytes) {
            checkState(
                    state.get() == RUNNING && chargedBytes.get() == 0L,
                    "delivery is not ready for payload admission");
            long current;
            do {
                current = inflightPayloadBytes.get();
                if (bytes > maxInflightBytes - current) {
                    return false;
                }
            } while (!inflightPayloadBytes.compareAndSet(current, current + bytes));
            chargedBytes.set(bytes);
            return true;
        }

        private void release() {
            inflightPayloadBytes.addAndGet(-chargedBytes.getAndSet(0L));
            releaseDeliveryPermit();
        }

        void releaseIfNoResponse() {
            if (state.compareAndSet(RUNNING, RELEASED)) {
                release();
            }
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
                try {
                    rejected.accept(failure);
                } catch (Throwable cleanupFailure) {
                    failures.accept(Failures.append(failure, cleanupFailure));
                } finally {
                    release();
                }
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

        @Override
        public void close() {
            if (state.compareAndSet(RESERVED, RELEASED)) {
                release();
            }
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

    void awaitDeliveries() {
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
    }

    DeliveryExecution(ConfigService config, Consumer<Throwable> failures) {
        this.failures = failures;
        var configuration = config.loadBalanceConfig();
        if (configuration.getDispatcher().getType()
                == org.flexlb.config.DispatcherConfig.Type.BATCH) {
            batchIds = BatchIdGenerator.forCurrentProcess(Integer.getInteger("server.port", 7001));
        }
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
}
