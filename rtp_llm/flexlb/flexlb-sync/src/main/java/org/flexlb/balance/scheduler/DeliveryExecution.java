package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.delivery.BatchIdGenerator;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.route.RoleType;
import org.flexlb.config.ConfigService;
import org.flexlb.constant.MetricConstant;
import org.flexlb.metric.SafeFlexMonitor;
import org.flexlb.util.Failures;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.springframework.core.env.Environment;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Consumer;
import java.util.function.BiConsumer;
import static com.google.common.base.Preconditions.checkState;

/** Batch admission, transport execution and real RPC response drain. No request policy lives here. */
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
    private final CopyOnWriteArraySet<Runnable> deliveryCapacityListeners = new CopyOnWriteArraySet<>();
    private volatile boolean acceptingDeliveries = true;
    final AdmissionCapacity decodeDeliveryCapacity;
    final int maxInflightBatches;
    private BatchIdGenerator batchIds;
    private static final RouteProjection.AdmissionBlockSemantics DELIVERY_CAPACITY_BLOCK =
            new RouteProjection.AdmissionBlockSemantics("DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RouteProjection.AfterProbeAdmission.BLOCKED, "DELIVERY_CAPACITY_BATCH_ADMISSION", RoleType.PREFILL);

    void configureBatchIds(Environment environment) {
        if (batchIds == null) { return; }
        try { batchIds = BatchIdGenerator.forCurrentProcess(Integer.parseInt(environment.getProperty("server.port",
                System.getProperty("server.port", "7001")))); }
        catch (NumberFormatException ignored) { }
    }

    void configureDeliveryMetrics(MeterRegistry registry) {
        SafeFlexMonitor.run(MetricConstant.DISPATCH_EXECUTOR_ACTIVE_THREADS, () ->
                Gauge.builder("flexlb." + MetricConstant.DISPATCH_EXECUTOR_ACTIVE_THREADS, dispatchExecutor,
                        ThreadPoolExecutor::getActiveCount).description("Dispatch executor active thread count").register(registry));
        SafeFlexMonitor.run(MetricConstant.DISPATCH_EXECUTOR_QUEUE_SIZE, () ->
                Gauge.builder("flexlb." + MetricConstant.DISPATCH_EXECUTOR_QUEUE_SIZE, dispatchExecutor,
                        executor -> executor.getQueue().size()).description("Dispatch executor pending task queue size").register(registry));
        SafeFlexMonitor.run(MetricConstant.DISPATCH_EXECUTOR_POOL_SIZE, () ->
                Gauge.builder("flexlb." + MetricConstant.DISPATCH_EXECUTOR_POOL_SIZE, dispatchExecutor,
                        ThreadPoolExecutor::getPoolSize).description("Dispatch executor current pool size").register(registry));
        SafeFlexMonitor.run(MetricConstant.DISPATCH_EXECUTOR_COMPLETED_TASKS, () ->
                FunctionCounter.builder("flexlb." + MetricConstant.DISPATCH_EXECUTOR_COMPLETED_TASKS, dispatchExecutor,
                        ThreadPoolExecutor::getCompletedTaskCount).description("Dispatch executor total completed tasks").register(registry));
    }

    long nextBatchId() { return batchIds.nextId(); }

    CapacityBoundary.Attempt<DeliveryPermit> tryReserveDelivery() {
        var lock = deliveryLifecycle.readLock();
        lock.lock();
        try {
            if (!acceptingDeliveries) { return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(
                    new IllegalStateException("delivery execution is closed"))); }
            if (!deliveryPermits.tryAcquire()) { return CapacityBoundary.Attempt.rejected(CapacityBoundary.unavailable(
                    CapacityBoundary.Resource.DISPATCH_EXECUTOR, DELIVERY_CAPACITY_BLOCK)); }
            try { return CapacityBoundary.Attempt.accepted(new DeliveryPermit()); }
            catch (Throwable failure) { deliveryPermits.release(); throw failure; }
        } finally { lock.unlock(); }
    }

    boolean deliveryCapacityAvailable() { return !acceptingDeliveries || deliveryPermits.availablePermits() > 0; }

    Runnable watchDeliveryCapacity(Runnable wake) {
        deliveryCapacityListeners.add(wake);
        return () -> deliveryCapacityListeners.remove(wake);
    }

    /** One infrastructure permit. No business state or callbacks run under an infrastructure lock. */
    final class DeliveryPermit implements AutoCloseable {
        // 0 reserved, 1 submitted, 2 running, 3 released, 4 awaiting RPC response.
        private final AtomicInteger state = new AtomicInteger();
        private final AtomicLong chargedBytes = new AtomicLong();

        boolean tryChargeBytes(long bytes) {
            checkState(state.get() == 2 && chargedBytes.get() == 0L, "delivery is not ready for payload admission");
            long current;
            do {
                current = inflightPayloadBytes.get();
                if (bytes > maxInflightBytes - current) { return false; }
            } while (!inflightPayloadBytes.compareAndSet(current, current + bytes));
            chargedBytes.set(bytes);
            return true;
        }

        private void release() {
            inflightPayloadBytes.addAndGet(-chargedBytes.getAndSet(0L));
            releaseDeliveryPermit();
        }

        void releaseIfNoResponse() {
            if (state.compareAndSet(2, 3)) { release(); }
        }

        void submit(Runnable work, Consumer<Throwable> rejected) {
            Objects.requireNonNull(work, "work");
            Objects.requireNonNull(rejected, "rejected");
            checkState(state.compareAndSet(0, 1), "delivery permit already resolved");
            try {
                dispatchExecutor.execute(() -> {
                    if (!state.compareAndSet(1, 2)) { return; }
                    try { work.run(); }
                    catch (Throwable failure) { failures.accept(failure); }
                    finally { releaseIfNoResponse(); }
                });
            } catch (RuntimeException | Error failure) {
                if (!state.compareAndSet(1, 3)) { failures.accept(failure); return; }
                try { rejected.accept(failure); }
                catch (Throwable cleanupFailure) { failures.accept(Failures.append(failure, cleanupFailure)); }
                finally { release(); }
            }
        }

        <T> void observeResponse(CompletableFuture<T> response, BiConsumer<T, Throwable> handler) {
            checkState(state.compareAndSet(2, 4), "delivery response already observed");
            DeliveryExecution.this.observeDeliveryResponse(response, (reply, failure) -> {
                // Return capacity before request callbacks can submit another batch.
                if (state.compareAndSet(4, 3)) { release(); }
                handler.accept(reply, failure);
            });
        }

        @Override public void close() {
            if (state.compareAndSet(0, 3)) { release(); }
        }
    }

    private <T> void observeDeliveryResponse(CompletableFuture<T> response, BiConsumer<T, Throwable> handler) {
        pendingDeliveryResponses.incrementAndGet();
        var started = new AtomicBoolean();
        try {
            response.handleAsync((reply, failure) -> {
                started.set(true);
                handler.accept(reply, failure);
                return null;
            }, completionExecutor).whenComplete((ignored, failure) -> {
                if (failure != null && !started.get()) {
                    // The response executor never ran the handler: deliver the uncertain fact once.
                    try { handler.accept(null, failure); }
                    catch (Throwable callbackFailure) { failures.accept(callbackFailure); }
                } else if (failure != null) { failures.accept(failure); }
                finishDeliveryResponse();
            });
        } catch (Throwable registrationFailure) {
            try { handler.accept(null, registrationFailure); }
            catch (Throwable callbackFailure) { failures.accept(callbackFailure); }
            finally { finishDeliveryResponse(); }
        }
    }

    private void releaseDeliveryPermit() {
        deliveryPermits.release();
        for (Runnable listener : deliveryCapacityListeners) {
            try { listener.run(); }
            catch (Throwable failure) { failures.accept(failure); }
        }
        finishDeliveryDrain();
    }

    private void finishDeliveryResponse() {
        pendingDeliveryResponses.decrementAndGet();
        finishDeliveryDrain();
    }

    private void finishDeliveryDrain() {
        if (!acceptingDeliveries && deliveryPermits.availablePermits() == deliveryAdmissionCapacity
                && pendingDeliveryResponses.get() == 0) {
            dispatchExecutor.shutdown();
            completionExecutor.shutdown();
            synchronized (deliveryDrain) { deliveryDrain.notifyAll(); }
        }
    }

    void stopDeliveryAdmission() {
        var lock = deliveryLifecycle.writeLock();
        lock.lock();
        try { acceptingDeliveries = false; }
        finally { lock.unlock(); }
        for (Runnable listener : deliveryCapacityListeners) {
            try { listener.run(); }
            catch (Throwable failure) { failures.accept(failure); }
        }
        finishDeliveryDrain();
    }

    void awaitDeliveries() {
        stopDeliveryAdmission();
        boolean interrupted = false;
        synchronized (deliveryDrain) {
            while (deliveryPermits.availablePermits() != deliveryAdmissionCapacity || pendingDeliveryResponses.get() != 0) {
                try { deliveryDrain.wait(); }
                catch (InterruptedException ignored) { interrupted = true; }
            }
        }
        while (!dispatchExecutor.isTerminated() || !completionExecutor.isTerminated()) {
            try {
                dispatchExecutor.awaitTermination(1, TimeUnit.DAYS);
                completionExecutor.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException ignored) { interrupted = true; }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
    }

    DeliveryExecution(ConfigService config, Consumer<Throwable> failures) {
        this.failures = failures;
        var configuration = config.loadBalanceConfig();
        if (configuration.getDispatcher().getType() == org.flexlb.config.DispatcherConfig.Type.BATCH) {
            batchIds = BatchIdGenerator.forCurrentProcess(Integer.getInteger("server.port", 7001));
        }
        var sizing = configuration.getInternalRuntime();
        int threads = sizing.getBatchDispatchThreads();
        deliveryAdmissionCapacity = Math.addExact(threads, sizing.getBatchDispatchQueueCapacity());
        maxInflightBytes = Math.max(1L, sizing.getBatchDispatchMaxInflightBytes());
        deliveryPermits = new Semaphore(deliveryAdmissionCapacity);
        dispatchExecutor = new ThreadPoolExecutor(threads, threads, 60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(), new NamedThreadFactory("flexlb-dispatch-executor"),
                new ThreadPoolExecutor.AbortPolicy());
        int completionThreads = sizing.getBatchDispatchCompletionThreads();
        completionExecutor = new ThreadPoolExecutor(completionThreads, completionThreads, 60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(), new NamedThreadFactory("flexlb-dispatch-completion"),
                new ThreadPoolExecutor.AbortPolicy());
        var availability = configuration.getRouter().getRoles().getDecode().getAvailability();
        decodeDeliveryCapacity = new AdmissionCapacity(availability.getMaxEngineRequests() == null ? 0L
                : availability.getMaxEngineRequests(), availability.getMaxKvUsagePercent());
        maxInflightBatches = configuration.getDispatcher().getMaxInflightPerPrefillWorker();
    }
}
