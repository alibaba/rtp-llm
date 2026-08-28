package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.ConfigService;
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
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Default batch-submission execution adapter.
 * <p>
 * Owns its own thread pool for asynchronous gRPC dispatch.
 * Handles the full pipeline: build request → send → parse response → callback.
 * Does NOT manage inflight state; results are reported through the delivery
 * observer port.
 */
@Component
public class DefaultBatchDispatcher implements BatchSubmissionPort {

    private static final String METRIC_PREFIX = "flexlb.";
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
    private final int admissionCapacity;
    private final Semaphore admissionPermits;
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

    /** Package-visible sizing injection keeps integration fixtures bounded and deterministic. */
    DefaultBatchDispatcher(EngineGrpcClient grpcClient, ConfigService configService,
                           MeterRegistry meterRegistry, int poolSize, int queueSize) {
        this.grpcClient = grpcClient;
        this.configService = configService;
        this.meterRegistry = meterRegistry;
        this.admissionCapacity = Math.addExact(poolSize, queueSize);
        this.admissionPermits = new Semaphore(admissionCapacity);
        Logger.info("FlexLB dispatch executor config: poolSize={}, logicalAdmissionCapacity={}, threadFactory=flexlb-dispatch-executor, rejectionPolicy=AbortPolicy",
                poolSize, admissionCapacity);
        // Permits bound accepted reservations through their RPC handoff. The
        // physical queue stays unbounded so an accepted batch cannot be
        // rejected after commit.
        this.dispatchExecutor = new ThreadPoolExecutor(
                poolSize, poolSize,
                60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(),
                new NamedThreadFactory("flexlb-dispatch-executor"),
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

    @Override
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
            implements BatchSubmissionPort.PreparedSubmission {

        private enum PermitPhase {
            PREPARED,
            SUBMITTED,
            RELEASED
        }

        private final AtomicReference<PermitPhase> phase =
                new AtomicReference<>(PermitPhase.PREPARED);

        @Override
        public void submitBatch(
                BatchSubmissionPort.Command command,
                BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> observer) {
            DispatchTask submittedTask = dispatchTask(command, observer);
            if (!phase.compareAndSet(
                    PermitPhase.PREPARED, PermitPhase.SUBMITTED)) {
                throw new IllegalStateException(
                        "prepared batch submission cannot submit from "
                                + phase.get());
            }
            try {
                dispatchExecutor.execute(() -> {
                    try {
                        doDispatch(submittedTask);
                    } finally {
                        finishSubmitted();
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
    }

    @SuppressWarnings("unchecked")
    private static DispatchTask dispatchTask(
            BatchSubmissionPort.Command command,
            BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> observer) {
        List<ScheduledRequest> items =
                (List<ScheduledRequest>) (List<?>) command.exactItems();
        return new DispatchTask(
                items,
                items.get(0).prefillEp(),
                command.batchId(),
                command.predictedMs(),
                command.metadata().decisionReason(),
                observer);
    }

    private record DispatchTask(List<ScheduledRequest> items,
                                PrefillEndpoint prefillEndpoint,
                                long batchId,
                                long predictedMs,
                                String reason,
                                BiConsumer<ScheduledRequest,
                                        SlotDeliveryPort.Completion> observer) {
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
        }
    }

    private void doDispatchInternal(DispatchTask task,
                                    DispatchAttempt attempt) {
        List<ScheduledRequest> items = task.items();
        PrefillEndpoint prefillEp = task.prefillEndpoint();
        long batchId = task.batchId();
        BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> observer =
                task.observer();

        // 1. Build gRPC request
        EngineRpcService.EnqueueBatchRequestPB request;
        try {
            request = buildBatchRequest(batchId, items);
        } catch (Exception e) {
            Logger.error("Failed to build FlexLB batch request batchId: {}", batchId, e);
            failItems(items, batchId,
                    "Batch request build failed: " + e.getMessage(), observer);
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
        long deadlineMs = activeBatchConfig().getEnqueueRpcTimeoutMs();
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
                    prefillIp, prefillGrpcPort, request, deadlineMs);
        } catch (Throwable invocationFailure) {
            // Once client invocation starts, a synchronous exception does not
            // prove that no bytes were written. Treat it as ambiguous.
            markUncertain(items, batchId, invocationFailure, observer);
            return;
        }
        if (rpcFuture == null) {
            RuntimeException missingFuture = new RuntimeException(
                    "EnqueueBatch client returned null future after invocation");
            markUncertain(items, batchId, missingFuture, observer);
            return;
        }
        // Increment while this dispatch still owns its admission permit. That
        // prevents shutdown from observing both zero pending completions and
        // all permits returned before the completion observer is registered.
        pendingCompletions.incrementAndGet();
        try {
            CompletableFuture<Void> completionObserver = rpcFuture.handleAsync(
                    (response, ex) -> {
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
                    }, dispatchExecutor);
            completionObserver.whenComplete((ignored, observerFailure) -> {
                try {
                    if (observerFailure != null) {
                        markUncertain(items, batchId,
                                unwrapCompletionFailure(observerFailure), observer);
                    }
                } finally {
                    finishCompletion();
                }
            });
        } catch (Throwable registrationFailure) {
            finishCompletion();
            // Callback registration is post-invocation. The RPC may already
            // be in flight even though no completion observer was installed.
            markUncertain(items, batchId, registrationFailure, observer);
        }
    }

    private static Throwable unwrapCompletionFailure(Throwable failure) {
        return failure instanceof CompletionException && failure.getCause() != null
                ? failure.getCause() : failure;
    }

    private DispatcherConfig activeBatchConfig() {
        DispatcherConfig dispatcher =
                configService.loadBalanceConfig().getDispatcher();
        if (dispatcher.getType() == DispatcherConfig.Type.BATCH) {
            return dispatcher;
        }
        throw new IllegalStateException(
                "batch submission requires BATCH dispatcher configuration");
    }

    private static void markUncertain(List<ScheduledRequest> items, long batchId,
                                      Throwable error,
                                      BiConsumer<ScheduledRequest,
                                              SlotDeliveryPort.Completion> observer) {
        for (ScheduledRequest item : items) {
            try {
                observer.accept(
                        item, SlotDeliveryPort.Completion.uncertain(error));
            } catch (Throwable callbackFailure) {
                Logger.error("Dispatch-uncertain callback failed request_id={} batch_id={}",
                        item.requestId(), batchId, callbackFailure);
            }
        }
    }

    private void failItems(List<ScheduledRequest> items,
                           long batchId, String message,
                           BiConsumer<ScheduledRequest,
                                   SlotDeliveryPort.Completion> observer) {
        failItems(items, batchId, new RuntimeException(message), observer);
    }

    private void failItems(List<ScheduledRequest> items,
                           long batchId, Throwable error,
                           BiConsumer<ScheduledRequest,
                                   SlotDeliveryPort.Completion> observer) {
        for (ScheduledRequest item : items) {
            try {
                observer.accept(
                        item, SlotDeliveryPort.Completion.failed(error));
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
                                        SlotDeliveryPort.Completion> observer) {
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
                            SlotDeliveryPort.Completion.delivered());
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
                            SlotDeliveryPort.Completion.failed(
                                    new EngineRejectedException(
                                            errorCode,
                                            "EnqueueBatch rejected request "
                                                    + item.requestId() + ": "
                                                    + errorMessage)));
                } else {
                    observer.accept(
                            item,
                            SlotDeliveryPort.Completion.uncertain(
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

    /** Domain error returned for one item in an otherwise successful batch RPC. */
    public static final class EngineRejectedException extends RuntimeException {
        private final long errorCode;

        public EngineRejectedException(long errorCode, String message) {
            super(message);
            this.errorCode = errorCode;
        }

        public long errorCode() {
            return errorCode;
        }
    }

    // ==================== gRPC request building ====================

    private EngineRpcService.EnqueueBatchRequestPB buildBatchRequest(long batchId, List<ScheduledRequest> items)
            throws InvalidProtocolBufferException {
        EngineRpcService.EnqueueBatchRequestPB.Builder builder =
                EngineRpcService.EnqueueBatchRequestPB.newBuilder().setBatchId(batchId);
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
        try {
            byDpRank.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .forEach(entry -> {
                        try {
                            builder.addDpSlots(buildDpSlot(
                                    entry.getKey(), entry.getValue(), roleAddresses));
                        } catch (InvalidProtocolBufferException e) {
                            throw new BatchRequestBuildException(e);
                        }
                    });
        } catch (BatchRequestBuildException e) {
            throw (InvalidProtocolBufferException) e.getCause();
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
        config.clearRoleAddrs();
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

    // ==================== Internal exception wrapper ====================

    /**
     * Wraps checked {@link InvalidProtocolBufferException} to propagate through
     * stream lambdas in {@link #buildBatchRequest}.
     */
    private static final class BatchRequestBuildException extends RuntimeException {
        private BatchRequestBuildException(Throwable cause) {
            super(cause);
        }
    }

    /** Per-dispatch phase marker used only by the executor thread. */
    private static final class DispatchAttempt {
        private boolean rpcInvocationStarted;
    }
}
