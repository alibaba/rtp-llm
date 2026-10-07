package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.BatchDeliveryStrategy.PreparedSubmission;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.BiConsumer;

/**
 * Default batch-submission execution adapter.
 * <p>
 * Owns its own thread pool for asynchronous gRPC dispatch.
 * Handles the full pipeline: build request → send → parse response → callback.
 * Does NOT manage inflight state; results are reported through the delivery
 * observer port.
 */
@Component
public class DefaultBatchDispatcher {

    private static final String METRIC_PREFIX = "flexlb.";
    private static final long EXECUTOR_KEEP_ALIVE_SECONDS = 60L;
    private static final RouteProjection.AdmissionBlockSemantics
            CAPACITY_BLOCK_SEMANTICS =
            new RouteProjection.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RouteProjection.AfterProbeAdmission.BLOCKED,
                    "DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RoleType.PREFILL);

    private final EngineGrpcClient grpcClient;
    private final ConfigService configService;
    private final ThreadPoolExecutor dispatchExecutor;
    private final ThreadPoolExecutor completionExecutor;
    private final int admissionCapacity;
    private final Semaphore admissionPermits;
    /**
     * Byte-based in-flight payload budget. The count semaphore bounds batch
     * COUNT; with 512MiB-max messages a count bound alone permits up to
     * 320 x 512MiB of serialized direct buffers pinned by grpc-netty
     * ChannelOutboundBuffer/WriteQueue until the engine ACK or the 5s
     * deadline. This budget bounds the total wire-size actually in flight
     * (case55: observed 8.58GB pinned = 320 permits x ~27MB avg).
     */
    private final long maxInflightBytes;
    private final AtomicLong inflightPayloadBytes;
    // Admission ends at RPC handoff; this separate count keeps the callback
    // executor alive while accepted RPCs are still awaiting completion.
    private final AtomicInteger pendingCompletions = new AtomicInteger();
    private final MeterRegistry meterRegistry;
    private final ReentrantReadWriteLock admissionLifecycle =
            new ReentrantReadWriteLock(true);
    private final Lock admissionReadLock = admissionLifecycle.readLock();
    private final Lock admissionWriteLock = admissionLifecycle.writeLock();
    private final CopyOnWriteArraySet<Runnable> capacityListeners =
            new CopyOnWriteArraySet<>();
    private final CapacityBoundary.Availability capacityAvailability =
            new CapacityBoundary.Availability() {
                @Override
                public boolean isAvailable() {
                    // "Available" means the rejecting condition changed and
                    // the active head must retry admission. Shutdown is such
                    // a change: the retry returns a typed terminal failure.
                    return !acceptingSubmissions
                            || admissionPermits.availablePermits() > 0;
                }

                @Override
                public void addListener(Runnable listener) {
                    capacityListeners.add(listener);
                }

                @Override
                public void removeListener(Runnable listener) {
                    capacityListeners.remove(listener);
                }
            };
    private volatile boolean acceptingSubmissions = true;

    @Autowired
    public DefaultBatchDispatcher(EngineGrpcClient grpcClient, ConfigService configService,
                                  @Autowired(required = false) MeterRegistry meterRegistry) {
        this(grpcClient, configService, meterRegistry,
                configService.loadBalanceConfig().getInternalRuntime()
                        .getBatchDispatchThreads(),
                configService.loadBalanceConfig().getInternalRuntime()
                        .getBatchDispatchQueueCapacity());
    }

    /**
     * Package-visible sizing injection keeps integration fixtures bounded and deterministic.
     * Byte budget defaults from config unless explicitly injected (tests).
     */
    DefaultBatchDispatcher(EngineGrpcClient grpcClient, ConfigService configService,
                           MeterRegistry meterRegistry, int poolSize, int queueSize) {
        this(grpcClient, configService, meterRegistry, poolSize, queueSize,
                configService.loadBalanceConfig().getInternalRuntime()
                        .getBatchDispatchMaxInflightBytes());
    }

    /**
     * Full injection: count-based admission permits plus the byte-based
     * in-flight payload budget. Both are held until the EnqueueBatch RPC
     * completes; the byte budget is charged at dispatch-task construction
     * (items frozen) and is what actually bounds the serialized
     * direct-buffer footprint the RPCs pin while awaiting the engine ACK.
     */
    DefaultBatchDispatcher(EngineGrpcClient grpcClient, ConfigService configService,
                           MeterRegistry meterRegistry, int poolSize, int queueSize,
                           long maxInflightBytes) {
        this.grpcClient = grpcClient;
        this.configService = configService;
        this.meterRegistry = meterRegistry;
        this.admissionCapacity = Math.addExact(poolSize, queueSize);
        this.admissionPermits = new Semaphore(admissionCapacity);
        this.maxInflightBytes = Math.max(1L, maxInflightBytes);
        this.inflightPayloadBytes = new AtomicLong(0L);
        Logger.info("FlexLB dispatch executor config: poolSize={}, logicalAdmissionCapacity={}, maxInflightBytes={}, threadFactory=flexlb-dispatch-executor, rejectionPolicy=AbortPolicy",
                poolSize, admissionCapacity, this.maxInflightBytes);
        // Permits bound accepted reservations through their RPC handoff. The
        // physical queue stays unbounded so an accepted batch cannot be
        // rejected after commit.
        this.dispatchExecutor = new ThreadPoolExecutor(
                poolSize, poolSize,
                EXECUTOR_KEEP_ALIVE_SECONDS, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(),
                new NamedThreadFactory("flexlb-dispatch-executor"),
                new ThreadPoolExecutor.AbortPolicy());
        int completionThreads = Math.max(1,
                configService.loadBalanceConfig().getInternalRuntime()
                        .getBatchDispatchCompletionThreads());
        this.completionExecutor = new ThreadPoolExecutor(
                completionThreads, completionThreads,
                EXECUTOR_KEEP_ALIVE_SECONDS, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(),
                new NamedThreadFactory("flexlb-dispatch-completion"),
                new ThreadPoolExecutor.AbortPolicy());
        registerMetrics();
    }

    /**
     * Register Micrometer gauges and function counters for the dispatch executor.
     *
     * <p>Metrics exposed:
     * <ul>
     *   <li>{@code flexlb_dispatch_executor_active_threads} — gauge: active thread count</li>
     *   <li>{@code flexlb_dispatch_executor_queue_size} — gauge: pending task queue length</li>
     *   <li>{@code flexlb_dispatch_executor_pool_size} — gauge: current thread pool size</li>
     *   <li>{@code flexlb_dispatch_executor_completed_tasks_total} — counter: completed task count</li>
     * </ul>
     *
     * <p>When {@link MeterRegistry} is not available, metric registration is silently skipped.
     */
    private void registerMetrics() {
        if (meterRegistry == null) {
            Logger.info("MeterRegistry not available, skipping dispatch executor metrics");
            return;
        }

        Gauge.builder(METRIC_PREFIX + MetricConstant.DISPATCH_EXECUTOR_ACTIVE_THREADS,
                        dispatchExecutor, ThreadPoolExecutor::getActiveCount)
                .description("Dispatch executor active thread count")
                .register(meterRegistry);

        Gauge.builder(METRIC_PREFIX + MetricConstant.DISPATCH_EXECUTOR_QUEUE_SIZE,
                        dispatchExecutor, exec -> exec.getQueue().size())
                .description("Dispatch executor pending task queue size")
                .register(meterRegistry);

        Gauge.builder(METRIC_PREFIX + MetricConstant.DISPATCH_EXECUTOR_POOL_SIZE,
                        dispatchExecutor, ThreadPoolExecutor::getPoolSize)
                .description("Dispatch executor current pool size")
                .register(meterRegistry);

        FunctionCounter.builder(METRIC_PREFIX + MetricConstant.DISPATCH_EXECUTOR_COMPLETED_TASKS,
                        dispatchExecutor, ThreadPoolExecutor::getCompletedTaskCount)
                .description("Dispatch executor total completed tasks")
                .register(meterRegistry);

        Logger.info("FlexLB dispatch executor metrics registered with MeterRegistry");
    }

    public CapacityBoundary.Attempt<PreparedSubmission>
            tryPrepareSubmission() {
        admissionReadLock.lock();
        try {
            if (!acceptingSubmissions) {
                return rejectedFailure(
                        new IllegalStateException(
                                "batch dispatcher is shut down"));
            }
            if (!admissionPermits.tryAcquire()) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.unavailable(
                                capacityAvailability,
                                CAPACITY_BLOCK_SEMANTICS));
            }
            return CapacityBoundary.Attempt.accepted(
                    new PermitReservation());
        } finally {
            admissionReadLock.unlock();
        }
    }

    private static CapacityBoundary.Attempt<PreparedSubmission> rejectedFailure(
            Throwable cause) {
        return CapacityBoundary.Attempt.rejected(
                CapacityBoundary.failed(cause));
    }

    @PreDestroy
    public void shutdown() {
        admissionWriteLock.lock();
        try {
            if (!acceptingSubmissions) {
                return;
            }
            acceptingSubmissions = false;
            tryShutdownExecutor();
        } finally {
            admissionWriteLock.unlock();
        }
        signalCapacityAvailable();
    }

    private void releasePermit() {
        admissionPermits.release();
        signalCapacityAvailable();
        tryShutdownExecutor();
    }

    /**
     * O(1) wire-size upper bound for one dispatch batch: the same accounting
     * the batch grouping uses (per-item envelope bound) plus outer framing.
     * Never parses or copies GenerateInput payloads.
     */
    private static long wireSizeUpperBound(List<ScheduledRequest> items) {
        long total = 16L; // batch_id + outer framing
        for (ScheduledRequest item : items) {
            total += item.batchPayloadSizeUpperBound();
        }
        return total;
    }

    private void finishCompletion() {
        pendingCompletions.decrementAndGet();
        tryShutdownExecutor();
    }

    private void tryShutdownExecutor() {
        if (!acceptingSubmissions
                && admissionPermits.availablePermits()
                == admissionCapacity
                && pendingCompletions.get() == 0) {
            dispatchExecutor.shutdown();
            completionExecutor.shutdown();
        }
    }

    private void signalCapacityAvailable() {
        for (Runnable listener : capacityListeners) {
            try {
                listener.run();
            } catch (Throwable listenerFailure) {
                Logger.error("Batch dispatcher capacity listener failed", listenerFailure);
            }
        }
    }

    /** One dispatch-task permit, acquired before the canonical commit. */
    private final class PermitReservation
            implements PreparedSubmission {

        private enum PermitPhase {
            PREPARED,
            SUBMITTED,
            /** doDispatch registered the RPC completion observer: only that
             *  observer may release the permit (when the RPC completes). */
            AWAITING_RPC,
            RELEASED
        }

        private final AtomicReference<PermitPhase> phase =
                new AtomicReference<>(PermitPhase.PREPARED);

        /** Wire-size charged against the in-flight byte budget. 0 until the
         *  dispatch task freezes its items; never negative. */
        private final AtomicLong chargedBytes = new AtomicLong(0L);

        @Override
        public void submit(BatchDeliveryStrategy.Delivery delivery) {
            Objects.requireNonNull(delivery, "delivery");
            if (!phase.compareAndSet(
                    PermitPhase.PREPARED, PermitPhase.SUBMITTED)) {
                throw new IllegalStateException(
                        "prepared batch submission cannot submit from "
                                + phase.get());
            }
            try {
                dispatchExecutor.execute(() -> {
                    try {
                        // Completion ownership: doDispatch settles the permit
                        // on EVERY terminal path — either by transferring to
                        // AWAITING_RPC (the registered completion observer
                        // releases when the EnqueueBatch RPC completes), or by
                        // releasing it directly when no RPC completion will
                        // ever arrive (build failure, oversize, pre-send
                        // throw, null future, registration failure). A
                        // Delivery that returns WITHOUT ever calling the
                        // sender (all requests expired/cancelled before
                        // handoff) also leaves no completion behind, so the
                        // permit must be settled after delivery.run returns.
                        delivery.run((items, batchId, predictedMs, reason, observer) -> {
                            // Byte-budget admission: charge the frozen
                            // wire-size before any serialization. A rejection
                            // here is pre-send by construction: nothing was
                            // serialized, so NOT_SENT is the honest outcome.
                            long bytes = wireSizeUpperBound(items);
                            if (!tryChargeBytes(bytes)) {
                                Logger.warn(
                                        "EnqueueBatch byte budget exhausted; failing batch {} ({}B > budget {}B, in-flight {}B) as NOT_SENT",
                                        batchId, bytes, maxInflightBytes,
                                        inflightPayloadBytes.get());
                                failItems(items, batchId,
                                        "in-flight payload byte budget exhausted",
                                        observer);
                                // No RPC, no observer: settle count and bytes.
                                settleIfNoRpcRegistered();
                                return;
                            }
                            doDispatch(dispatchTask(items, batchId, predictedMs,
                                    reason, observer, PermitReservation.this));
                        });
                    } catch (Throwable deliveryFailure) {
                        // Delivery owns admission cleanup; do not infer a
                        // second per-request outcome from task failure.
                        Logger.error("Batch delivery task failed", deliveryFailure);
                    } finally {
                        // Exactly-once settle: if doDispatch transferred to
                        // AWAITING_RPC this is a NO-OP (only the completion
                        // observer releases). Otherwise this returns the
                        // permit — without it, a sender-less delivery would
                        // leak admission permanently and shutdown could
                        // never complete.
                        settleIfNoRpcRegistered();
                    }
                });
            } catch (RuntimeException | Error submissionFailure) {
                finishSubmitted();
                throw submissionFailure;
            }
        }

        @Override
        public void close() {
            if (phase.compareAndSet(
                    PermitPhase.PREPARED, PermitPhase.RELEASED)) {
                releasePermit();
            }
        }

        private void finishSubmitted() {
            if (phase.compareAndSet(
                    PermitPhase.SUBMITTED, PermitPhase.RELEASED)) {
                releasePermit();
            }
        }

        /**
         * Charges this batch's wire-size upper bound against the in-flight
         * byte budget. Called once at dispatch-task construction (items
         * frozen). Rejecting here means the batch was NOT serialized: the
         * items are failed NOT_SENT and the permit settles, so admission
         * count and byte budget return together.
         */
        boolean tryChargeBytes(long bytes) {
            long budget = maxInflightBytes;
            long current;
            long next;
            do {
                current = inflightPayloadBytes.get();
                next = current + bytes;
                if (next > budget) {
                    return false;
                }
            } while (!inflightPayloadBytes.compareAndSet(current, next));
            chargedBytes.set(bytes);
            return true;
        }

        private void releaseChargedBytes() {
            long bytes = chargedBytes.getAndSet(0L);
            if (bytes > 0L) {
                inflightPayloadBytes.addAndGet(-bytes);
                signalCapacityAvailable();
            }
        }

        /**
         * Transfers completion ownership to the registered RPC completion
         * observer. Called by doDispatch after the observer is installed.
         * If the task-level settle already ran (task raced ahead), the
         * permit was already returned and the observer's later release is a
         * no-op — safe because the RPC future is already complete in that
         * interleaving or the release is simply idempotent.
         */
        void transferToRpcCompletion() {
            phase.compareAndSet(
                    PermitPhase.SUBMITTED, PermitPhase.AWAITING_RPC);
        }

        /**
         * Releases the admission permit when the dispatched EnqueueBatch RPC
         * has completed (success, failure, or uncertain), or when the task
         * wrapper settled because no RPC completion was registered.
         * Idempotent via the phase CAS.
         */
        void releaseOnRpcCompletion() {
            PermitPhase current = phase.get();
            if (current == PermitPhase.RELEASED) {
                return;
            }
            if (phase.compareAndSet(
                    PermitPhase.AWAITING_RPC, PermitPhase.RELEASED)
                    || phase.compareAndSet(
                    PermitPhase.PREPARED, PermitPhase.RELEASED)) {
                releaseChargedBytes();
                releasePermit();
            }
        }

        /**
         * Task-wrapper settle: releases the permit only when no RPC
         * completion observer took ownership. AWAITING_RPC (observer
         * registered) and RELEASED are left untouched; SUBMITTED (and
         * PREPARED, if the delivery threw before submit advanced the phase)
         * release now because nothing else ever will.
         */
        void settleIfNoRpcRegistered() {
            PermitPhase current = phase.get();
            if (current == PermitPhase.AWAITING_RPC
                    || current == PermitPhase.RELEASED) {
                return;
            }
            if (phase.compareAndSet(
                    PermitPhase.SUBMITTED, PermitPhase.RELEASED)) {
                releaseChargedBytes();
                releasePermit();
                return;
            }
            if (phase.compareAndSet(
                    PermitPhase.PREPARED, PermitPhase.RELEASED)) {
                releaseChargedBytes();
                releasePermit();
            }
        }
    }

    private static DispatchTask dispatchTask(
            List<ScheduledRequest> exactItems,
            long batchId,
            long predictedMs,
            String decisionReason,
            BiConsumer<ScheduledRequest, DeliveryResult> observer,
            PermitReservation permitReservation) {
        List<ScheduledRequest> frozenItems = List.copyOf(exactItems);
        if (frozenItems.isEmpty()) {
            throw new IllegalArgumentException("batch cannot be empty");
        }
        if (batchId <= 0L || predictedMs < 0L) {
            throw new IllegalArgumentException(
                    "batchId must be positive and predictedMs non-negative");
        }
        Objects.requireNonNull(permitReservation, "permitReservation");
        // Permit release is idempotent (PermitReservation phase CAS). The
        // registered completion observer releases when the EnqueueBatch RPC
        // completes, so the admission semaphore bounds IN-FLIGHT batches,
        // not merely dispatched ones.
        return new DispatchTask(
                frozenItems,
                frozenItems.getFirst().prefillEp(),
                batchId,
                predictedMs,
                Objects.requireNonNull(decisionReason, "decisionReason"),
                Objects.requireNonNull(observer, "observer"),
                permitReservation);
    }

    private record DispatchTask(List<ScheduledRequest> items,
                                PrefillEndpoint prefillEndpoint,
                                long batchId,
                                long predictedMs,
                                String reason,
                                BiConsumer<ScheduledRequest,
                                        DeliveryResult> observer,
                                PermitReservation permitReservation) {
    }

    // ==================== Internal: dispatch pipeline (runs on executor thread) ====================

    private void doDispatch(DispatchTask task) {
        DispatchAttempt attempt = new DispatchAttempt();
        try {
            doDispatchInternal(task, attempt);
        } catch (Throwable unexpectedFailure) {
            Logger.error("Unexpected dispatch failure batch_id={} rpc_invocation_started={}",
                    task.batchId(), attempt.rpcInvocationStarted, unexpectedFailure);
            if (attempt.rpcInvocationStarted) {
                // Once invocation starts, cleanup is unsafe even if the
                // exception escaped an otherwise defensive post-send path.
                markUncertain(task.items(), task.batchId(),
                        unexpectedFailure, task.observer());
            } else {
                failItems(task.items(), task.batchId(),
                        unexpectedFailure, task.observer());
            }
            // The RPC will never complete through the observer on this path;
            // settle the in-flight permit (idempotent).
            task.permitReservation().settleIfNoRpcRegistered();
        }
    }

    private void doDispatchInternal(DispatchTask task,
                                    DispatchAttempt attempt) {
        List<ScheduledRequest> items = task.items();
        PrefillEndpoint prefillEp = task.prefillEndpoint();
        long batchId = task.batchId();
        BiConsumer<ScheduledRequest, DeliveryResult> observer =
                task.observer();

        // 1. Build gRPC request
        EngineRpcService.EnqueueBatchRequestPB request;
        try {
            request = buildBatchRequest(batchId, items);
            int serializedSize = request.getSerializedSize();
            if (serializedSize < 0 || serializedSize > org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE) {
                throw new IllegalArgumentException(
                        "EnqueueBatch payload exceeds "
                                + (org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE / (1024 * 1024))
                                + " MiB gRPC limit: " + serializedSize + " bytes");
            }
        } catch (Exception e) {
            Logger.error("Failed to build FlexLB batch request batchId: {}", batchId, e);
            failItems(items, batchId,
                    "Batch request build failed: " + e.getMessage(), observer);
            // No RPC was invoked; no completion observer will ever run. The
            // task wrapper's settle also covers this, but settling here makes
            // the ownership explicit (exactly-once via phase CAS).
            task.permitReservation().settleIfNoRpcRegistered();
            return;
        }

        // 2. Log dispatch
        try {
            logDispatch(batchId, items, prefillEp,
                    task.predictedMs(), task.reason());
        } catch (Throwable loggingFailure) {
            // Reporting is not part of transport ownership. A logger failure
            // cannot turn an otherwise valid committed batch into a delivery
            // failure.
            Logger.warn("Batch dispatch logging failed batch_id={}",
                    batchId, loggingFailure);
        }

        // 3. Send gRPC (async)
        // Resolve every potentially fallible argument before entering the RPC
        // invocation block. A failure here is definitely pre-send and is
        // handled by doDispatch's outer guard.
        requireBatchDispatcher();
        String prefillIp = prefillEp.getIp();
        int prefillGrpcPort = prefillEp.getGrpcPort();
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpcFuture;
        try {
            long dispatchedNanos = System.nanoTime();
            for (ScheduledRequest item : items) {
                item.ctx().setBatchDispatchedNanos(dispatchedNanos);
            }
            attempt.rpcInvocationStarted = true;
            rpcFuture = grpcClient.batchEnqueueAsync(
                    prefillIp, prefillGrpcPort, request);
        } catch (Throwable invocationFailure) {
            // Once client invocation starts, a synchronous exception does not
            // prove that no bytes were written. Treat it as ambiguous. The
            // delivery state is uncertain post-invocation; do NOT relabel it
            // as NOT_SENT.
            markUncertain(items, batchId, invocationFailure, observer);
            // No observer was registered; settle the permit now (idempotent
            // with the task wrapper's finally).
            task.permitReservation().settleIfNoRpcRegistered();
            return;
        }
        if (rpcFuture == null) {
            RuntimeException missingFuture = new RuntimeException(
                    "EnqueueBatch client returned null future after invocation");
            markUncertain(items, batchId, missingFuture, observer);
            task.permitReservation().settleIfNoRpcRegistered();
            return;
        }
        // Increment while this dispatch still owns its admission permit. That
        // prevents shutdown from observing both zero pending completions and
        // all permits returned before the completion observer is registered.
        pendingCompletions.incrementAndGet();
        try {
            CompletableFuture<Void> completionObserver = rpcFuture.handleAsync(
                    (response, ex) -> {
                        // Release the in-flight admission permit FIRST: by
                        // the time request callbacks observe completion, the
                        // permit must already be back (otherwise a caller
                        // reacting to the callback could spuriously fail to
                        // re-admit). Exactly-once via phase CAS.
                        finishCompletion();
                        task.permitReservation().releaseOnRpcCompletion();
                        try {
                            if (ex != null) {
                                Throwable cause = unwrapCompletionFailure(ex);
                                Logger.debug("EnqueueBatch failed batchId: {}, entrypoint: {}:{}, err: {}",
                                        batchId, prefillIp, prefillGrpcPort, cause.getMessage());
                                // Once the asynchronous RPC is invoked, no
                                // transport status proves the server did not
                                // accept the request. Reconcile every transport
                                // failure through the Engine-side request-id fence.
                                markUncertain(items, batchId, cause, observer);
                            } else if (response == null) {
                                markUncertain(items, batchId, new RuntimeException(
                                        "EnqueueBatch returned null response"), observer);
                            } else {
                                handleResponse(batchId, items, response, observer);
                            }
                        } catch (Throwable completionFailure) {
                            // This callback is unconditionally post-invocation. Never
                            // let an unexpected response-processing failure fall back
                            // to definite failure/cleanup.
                            markUncertain(items, batchId, completionFailure, observer);
                        }
                        return null;
                    }, completionExecutor);
            // Ownership transfer: from here the registered completion
            // observer is the ONLY permit releaser (when the RPC completes).
            // The task wrapper's finally settle becomes a no-op.
            task.permitReservation().transferToRpcCompletion();
        } catch (Throwable registrationFailure) {
            finishCompletion();
            // Callback registration is post-invocation. The RPC may already
            // be in flight even though no completion observer was installed
            // (the delivery outcome stays UNCERTAIN, not NOT_SENT). No
            // observer owns the permit now, so settle it here (idempotent
            // with the task wrapper's finally).
            task.permitReservation().settleIfNoRpcRegistered();
            markUncertain(items, batchId, registrationFailure, observer);
        }
    }

    private static Throwable unwrapCompletionFailure(Throwable failure) {
        return failure instanceof CompletionException && failure.getCause() != null
                ? failure.getCause() : failure;
    }

    private void requireBatchDispatcher() {
        DispatcherConfig dispatcher =
                configService.loadBalanceConfig().getDispatcher();
        if (dispatcher.getType() == DispatcherConfig.Type.BATCH) {
            return;
        }
        throw new IllegalStateException(
                "batch submission requires BATCH dispatcher configuration");
    }

    private static void markUncertain(List<ScheduledRequest> items, long batchId,
                                      Throwable error,
                                      BiConsumer<ScheduledRequest,
                                              DeliveryResult> observer) {
        for (ScheduledRequest item : items) {
            try {
                observer.accept(
                        item, DeliveryResult.uncertain(error));
            } catch (Throwable callbackFailure) {
                Logger.error("Dispatch-uncertain callback failed request_id={} batch_id={}",
                        item.requestId(), batchId, callbackFailure);
            }
        }
    }

    private void failItems(List<ScheduledRequest> items,
                           long batchId, String message,
                           BiConsumer<ScheduledRequest,
                                   DeliveryResult> observer) {
        failItems(items, batchId, new RuntimeException(message), observer);
    }

    private void failItems(List<ScheduledRequest> items,
                           long batchId, Throwable error,
                           BiConsumer<ScheduledRequest,
                                   DeliveryResult> observer) {
        for (ScheduledRequest item : items) {
            try {
                observer.accept(
                        item, DeliveryResult.notSent(error));
            } catch (Throwable callbackFailure) {
                Logger.error("Dispatch-failure callback failed request_id={} batch_id={}",
                        item.requestId(), batchId, callbackFailure);
            }
        }
    }

    // ==================== Response parsing ====================

    private void handleResponse(long batchId, List<ScheduledRequest> items,
                                EngineRpcService.EnqueueBatchResponsePB response,
                                BiConsumer<ScheduledRequest,
                                        DeliveryResult> observer) {
        if (response.getBatchId() != batchId) {
            RuntimeException mismatch = new RuntimeException(
                    "EnqueueBatch batch_id mismatch: expected " + batchId
                            + " but got " + response.getBatchId());
            markUncertain(items, batchId, mismatch, observer);
            return;
        }
        Set<Long> expectedIds = new HashSet<>();
        List<String> protocolViolations = new ArrayList<>();
        for (ScheduledRequest item : items) {
            expectedIds.add(item.requestId());
        }

        Map<Long, EngineRpcService.EnqueueBatchErrorPB> errorByRequestId =
                new HashMap<>();
        for (EngineRpcService.EnqueueBatchErrorPB error : response.getErrorsList()) {
            long requestId = error.getRequestId();
            if (!expectedIds.contains(requestId)) {
                protocolViolations.add(
                        "error references unknown request_id=" + requestId);
            }
            if (errorByRequestId.putIfAbsent(requestId, error) != null) {
                protocolViolations.add(
                        "duplicate error for request_id=" + requestId);
            }
        }
        Set<Long> successIds = new HashSet<>();
        for (EngineRpcService.EnqueueBatchSuccessPB success : response.getSuccessesList()) {
            long requestId = success.getRequestId();
            if (!expectedIds.contains(requestId)) {
                protocolViolations.add(
                        "success references unknown request_id=" + requestId);
            }
            if (!successIds.add(requestId)) {
                protocolViolations.add(
                        "duplicate success for request_id=" + requestId);
            }
        }
        for (Long requestId : successIds) {
            if (errorByRequestId.containsKey(requestId)) {
                protocolViolations.add(
                        "request_id appears in both success and error: "
                                + requestId);
            }
        }
        for (Long requestId : expectedIds) {
            if (!successIds.contains(requestId)
                    && !errorByRequestId.containsKey(requestId)) {
                protocolViolations.add(
                        "response is missing request_id=" + requestId);
            }
        }
        if (!protocolViolations.isEmpty()) {
            markUncertain(
                    items,
                    batchId,
                    new RuntimeException(
                            "Malformed EnqueueBatch response: "
                                    + String.join("; ", protocolViolations)),
                    observer);
            return;
        }

        for (ScheduledRequest item : items) {
            try {
                if (successIds.contains(item.requestId())) {
                    observer.accept(
                            item,
                            DeliveryResult.delivered());
                } else if (errorByRequestId.containsKey(item.requestId())) {
                    EngineRpcService.EnqueueBatchErrorPB error = errorByRequestId.get(item.requestId());
                    long errorCode = error.hasErrorInfo()
                            ? error.getErrorInfo().getErrorCode()
                            : 0L;
                    String errorMessage = error.hasErrorInfo()
                            ? error.getErrorInfo().getErrorMessage()
                            : "missing error_info";
                    observer.accept(
                            item,
                            DeliveryResult.prefillRejected(
                                    new RuntimeException(
                                            "EnqueueBatch rejected request "
                                                    + item.requestId()
                                                    + " error_code=" + errorCode
                                                    + ": " + errorMessage)));
                } else {
                    observer.accept(
                            item,
                            DeliveryResult.uncertain(
                                    new RuntimeException(
                                            "EnqueueBatch missing ack for request "
                                                    + item.requestId())));
                }
            } catch (Throwable callbackFailure) {
                // The callback may already have committed this item's state
                // before throwing. Never issue a second, contradictory
                // callback for it, and never let it reclassify earlier items.
                Logger.error("EnqueueBatch item callback failed request_id={} batch_id={}",
                        item.requestId(), batchId, callbackFailure);
            }
        }
    }

    // ==================== gRPC request building ====================

    private EngineRpcService.EnqueueBatchRequestPB buildBatchRequest(long batchId, List<ScheduledRequest> items)
            throws InvalidProtocolBufferException {
        EngineRpcService.EnqueueBatchRequestPB.Builder builder =
                EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                        .setBatchId(batchId)
                        .setFetchAttachTimeoutMs(configService.loadBalanceConfig()
                                .getDispatcher().getFetchAttachTimeoutMs());
        BatchRoleAddressCache roleAddresses = new BatchRoleAddressCache();
        if (!items.isEmpty()) {
            long dpRank = items.get(0).prefill().getDpRank();
            boolean singleDpRank = true;
            for (int i = 1; i < items.size(); i++) {
                if (items.get(i).prefill().getDpRank() != dpRank) {
                    singleDpRank = false;
                    break;
                }
            }
            if (singleDpRank) {
                builder.addDpSlots(buildDpSlot(dpRank, items, roleAddresses));
                return builder.build();
            }
        }

        Map<Long, List<ScheduledRequest>> byDpRank = new HashMap<>();
        for (ScheduledRequest item : items) {
            byDpRank.computeIfAbsent(item.prefill().getDpRank(), ignored -> new ArrayList<>()).add(item);
        }
        List<Map.Entry<Long, List<ScheduledRequest>>> ranks =
                new ArrayList<>(byDpRank.entrySet());
        ranks.sort(Map.Entry.comparingByKey());
        for (Map.Entry<Long, List<ScheduledRequest>> entry : ranks) {
            builder.addDpSlots(buildDpSlot(
                    entry.getKey(), entry.getValue(), roleAddresses));
        }
        return builder.build();
    }

    private EngineRpcService.EnqueueBatchDpSlotPB buildDpSlot(
            long dpRank,
            List<ScheduledRequest> items,
            BatchRoleAddressCache roleAddresses)
            throws InvalidProtocolBufferException {
        EngineRpcService.EnqueueBatchDpSlotPB.Builder slot =
                EngineRpcService.EnqueueBatchDpSlotPB.newBuilder()
                        .setDpRank((int) dpRank);
        for (ScheduledRequest item : items) {
            slot.addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                    .setInput(buildInput(item, roleAddresses))
                    .build());
        }
        return slot.build();
    }

    private EngineRpcService.GenerateInputPB buildInput(
            ScheduledRequest item,
            BatchRoleAddressCache roleAddresses)
            throws InvalidProtocolBufferException {
        ByteString generateInput = item.ctx().getGenerateInputPb();
        if (generateInput == null || generateInput.isEmpty()) {
            throw new IllegalArgumentException("generateInputPb is missing for request " + item.requestId());
        }
        EngineRpcService.GenerateInputPB.Builder input =
                EngineRpcService.GenerateInputPB.newBuilder();
        input.mergeFrom(generateInput);
        if (input.getRequestId() != item.requestId()) {
            throw new IllegalArgumentException("request_id mismatch between schedule request and GenerateInputPB");
        }
        EngineRpcService.GenerateConfigPB.Builder config = input.getGenerateConfigBuilder();
        List<EngineRpcService.RoleAddrPB> visionAddrs = config.getRoleAddrsList().stream()
                .filter(addr -> RoleTypeProtoConverter.fromRoleAddr(addr) == RoleType.VIT)
                .toList();
        config.clearRoleAddrs();
        config.addAllRoleAddrs(visionAddrs);
        addRoleAddr(config, roleAddresses.prefill(item.prefill()));
        addRoleAddr(config, roleAddresses.decode(item.decode()));
        // Pass the normalized Auto-TPM priority through to the engine
        // (metrics tagging only). normalize() always sets 1-100, so every
        // dispatched request carries its priority into the proto field.
        Request request = item.ctx().getRequest();
        if (request != null) {
            input.setPriority(request.getPriority());
        }
        return input.build();
    }

    private static void addRoleAddr(
            EngineRpcService.GenerateConfigPB.Builder config,
            EngineRpcService.RoleAddrPB roleAddress) {
        if (roleAddress == null) {
            return;
        }
        config.addRoleAddrs(roleAddress);
    }

    private static EngineRpcService.RoleAddrPB buildRoleAddr(ServerStatus serverStatus) {
        RoleType role = serverStatus.getRole();
        return EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(RoleTypeProtoConverter.toLegacyProto(role))
                .setRoleStr(role.getCode())
                .setIp(serverStatus.getServerIp())
                .setHttpPort(serverStatus.getHttpPort())
                .setGrpcPort(serverStatus.getGrpcPort())
                .build();
    }

    private static boolean sameRoleAddr(
            EngineRpcService.RoleAddrPB cached,
            ServerStatus serverStatus) {
        RoleType role = serverStatus.getRole();
        return cached.getRole() == RoleTypeProtoConverter.toLegacyProto(role)
                && cached.getRoleStr().equals(role.getCode())
                && cached.getIp().equals(serverStatus.getServerIp())
                && cached.getHttpPort() == serverStatus.getHttpPort()
                && cached.getGrpcPort() == serverStatus.getGrpcPort();
    }

    /** Reuses immutable role addresses while building one batch payload. */
    private static final class BatchRoleAddressCache {
        private EngineRpcService.RoleAddrPB prefill;
        private EngineRpcService.RoleAddrPB decode;

        private EngineRpcService.RoleAddrPB prefill(ServerStatus serverStatus) {
            if (serverStatus == null) {
                return null;
            }
            if (prefill == null || !sameRoleAddr(prefill, serverStatus)) {
                prefill = buildRoleAddr(serverStatus);
            }
            return prefill;
        }

        private EngineRpcService.RoleAddrPB decode(ServerStatus serverStatus) {
            if (serverStatus == null) {
                return null;
            }
            if (decode == null || !sameRoleAddr(decode, serverStatus)) {
                decode = buildRoleAddr(serverStatus);
            }
            return decode;
        }
    }

    // ==================== Logging ====================

    private void logDispatch(long batchId, List<ScheduledRequest> items,
                             PrefillEndpoint prefillEp, long predMs, String reason) {
        if (!Logger.isDebugEnabled()) {
            return;
        }
        long totalTokens = 0;
        long totalHit = 0;
        StringBuilder itemDetail = new StringBuilder();
        for (int i = 0; i < items.size(); i++) {
            ScheduledRequest item = items.get(i);
            long seqLen = item.seqLen();
            long hitCache = item.hitCache();
            totalTokens += seqLen;
            totalHit += hitCache;
            if (i > 0) {
                itemDetail.append(", ");
            }
            itemDetail.append("{req_id=").append(item.requestId())
                    .append(" seq_len=").append(seqLen)
                    .append(" hit_cache=").append(hitCache).append('}');
        }

        ScheduledRequest head = items.get(0);
        long now = System.currentTimeMillis();
        long waitMs = now - head.enqueuedAtMs();
        long remainingMs = head.expiresAtMs() - now;

        Logger.debug("flexlb_batch_dispatch batch_id={} batch_size={} total_tokens={} total_hit={} "
                        + "pred_ms={} reason={} wait_ms={} request_remaining_ms={} "
                        + "prefill={}:{} items=[{}]",
                batchId, items.size(), totalTokens, totalHit, predMs, reason,
                waitMs, remainingMs,
                prefillEp.getIp(), prefillEp.getHttpPort(),
                itemDetail);
    }

    /** Per-dispatch phase marker used only by the executor thread. */
    private static final class DispatchAttempt {
        private boolean rpcInvocationStarted;
    }
}
