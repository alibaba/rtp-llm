package org.flexlb.balance.scheduler;

import com.google.protobuf.InvalidProtocolBufferException;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.BiConsumer;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

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

    /**
     * One executor-capacity permit prepared before commit. Successful submission
     * transfers the permit to the executor; close then becomes a no-op. The
     * strategy still owns the batch admission until Delivery runs and settles it.
     */
    public interface PreparedSubmission extends AutoCloseable {
        void submit(Delivery delivery);

        @Override
        void close();
    }

    /** Runs once on the dispatch executor, with no further queue before send. */
    @FunctionalInterface
    public interface Delivery {
        void run(BatchSender sender);
    }

    /** Transport consumes the strategy's final batch; it never selects members. */
    @FunctionalInterface
    public interface BatchSender {
        void sendBatch(
                List<RequestRoute> exactItems,
                long batchId,
                long predictedMs,
                String decisionReason,
                BiConsumer<RequestRoute, DeliveryResult> observer);
    }



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
    // Admission ends at RPC handoff; this separate count keeps the callback
    // executor alive while accepted RPCs are still awaiting completion.
    private final AtomicInteger pendingCompletions = new AtomicInteger();
    private final Object drain = new Object();
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
        int poolSize = configService.loadBalanceConfig().getInternalRuntime().getBatchDispatchThreads();
        int queueSize = configService.loadBalanceConfig().getInternalRuntime().getBatchDispatchQueueCapacity();
        this.grpcClient = grpcClient;
        this.configService = configService;
        this.admissionCapacity = Math.addExact(poolSize, queueSize);
        this.admissionPermits = new Semaphore(admissionCapacity);
        Logger.info("FlexLB dispatch executor config: poolSize={}, logicalAdmissionCapacity={}, threadFactory=flexlb-dispatch-executor, rejectionPolicy=AbortPolicy",
                poolSize, admissionCapacity);
        // Permits bound accepted reservations through their RPC handoff. The
        // physical queue stays unbounded so an accepted batch cannot be
        // rejected after commit.
        this.dispatchExecutor = new ThreadPoolExecutor(
                poolSize, poolSize,
                EXECUTOR_KEEP_ALIVE_SECONDS, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(),
                new NamedThreadFactory("flexlb-dispatch-executor"),
                new ThreadPoolExecutor.AbortPolicy());
        int completionThreads = configService.loadBalanceConfig().getInternalRuntime()
                .getBatchDispatchCompletionThreads();
        this.completionExecutor = new ThreadPoolExecutor(
                completionThreads, completionThreads,
                EXECUTOR_KEEP_ALIVE_SECONDS, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(),
                new NamedThreadFactory("flexlb-dispatch-completion"),
                new ThreadPoolExecutor.AbortPolicy());
        registerMetrics(meterRegistry);
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
    private void registerMetrics(MeterRegistry meterRegistry) {
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
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(
                        new IllegalStateException("batch dispatcher is shut down")));
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

    /** Stop admission and wait for submitted dispatches and RPC observers. */
    void shutdownAndAwait() {
        shutdown();
        boolean interrupted = false;
        synchronized (drain) {
            while (admissionPermits.availablePermits() != admissionCapacity
                    || pendingCompletions.get() != 0) {
                try { drain.wait(); }
                catch (InterruptedException ignored) { interrupted = true; }
            }
        }
        tryShutdownExecutor();
        while (!dispatchExecutor.isTerminated() || !completionExecutor.isTerminated()) {
            try {
                dispatchExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
                completionExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException ignored) { interrupted = true; }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
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
            completionExecutor.shutdown();
            synchronized (drain) { drain.notifyAll(); }
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
            RELEASED
        }

        private final AtomicReference<PermitPhase> phase =
                new AtomicReference<>(PermitPhase.PREPARED);

        @Override
        public void submit(DefaultBatchDispatcher.Delivery delivery) {
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
                        delivery.run(DefaultBatchDispatcher.this::dispatchBatch);
                    } catch (Throwable deliveryFailure) {
                        // Delivery owns admission cleanup; do not infer a
                        // second per-request outcome from task failure.
                        Logger.error("Batch delivery task failed", deliveryFailure);
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

    private void dispatchBatch(
            List<RequestRoute> exactItems,
            long batchId,
            long predictedMs,
            String decisionReason,
            BiConsumer<RequestRoute, DeliveryResult> observer) {
        List<RequestRoute> items = List.copyOf(exactItems);
        checkArgument(!items.isEmpty(), "batch cannot be empty");
        checkArgument(batchId > 0L && predictedMs >= 0L, "batchId must be positive and predictedMs non-negative");
        Objects.requireNonNull(decisionReason, "decisionReason");
        Objects.requireNonNull(observer, "observer");
        boolean invoked = false;
        try {
            PrefillEndpoint endpoint = items.getFirst().prefillEp();
            EngineRpcService.EnqueueBatchRequestPB request;
            try {
                request = buildBatchRequest(batchId, items);
            } catch (Exception failure) {
                throw new IllegalArgumentException("Batch request build failed: " + failure.getMessage(), failure);
            }
            requireBatchDispatcher();
            String ip = endpoint.getIp();
            int port = endpoint.getGrpcPort();
            // Recheck each exact claim at the RPC boundary, then remove refused members from the payload.
            List<RequestRoute> sending = null;
            for (int index = 0; index < items.size(); index++) {
                RequestRoute item = items.get(index);
                var claim = item.ctx().delivery();
                if (claim != null && item.ctx().scheduler().tryStartSend(claim)) {
                    if (sending != null) { sending.add(item); }
                } else {
                    if (sending == null) {
                        sending = new ArrayList<>(items.size());
                        sending.addAll(items.subList(0, index));
                    }
                    observer.accept(item, DeliveryResult.notSent(new java.util.concurrent.CancellationException("delivery abandoned before send")));
                }
            }
            if (sending != null && sending.isEmpty()) { return; }
            if (sending != null) {
                java.util.Set<Long> ids = new java.util.HashSet<>();
                sending.forEach(item -> ids.add(item.requestId()));
                var filtered = request.toBuilder().clearDpSlots();
                for (var slot : request.getDpSlotsList()) {
                    var selected = slot.toBuilder().clearRequests();
                    slot.getRequestsList().stream().filter(member -> ids.contains(member.getInput().getRequestId()))
                            .forEach(selected::addRequests);
                    if (selected.getRequestsCount() != 0) { filtered.addDpSlots(selected); }
                }
                request = filtered.build();
                items = List.copyOf(sending);
            }
            try {
                logDispatch(batchId, items, endpoint, predictedMs, decisionReason);
            } catch (Throwable loggingFailure) {
                logFailure("Batch dispatch logging failed", batchId, loggingFailure);
            }
            long dispatchedNanos = System.nanoTime();
            for (RequestRoute item : items) {
                item.ctx().setBatchDispatchedNanos(dispatchedNanos);
                FlexlbTrace.setScheduleAttribute(item.ctx().getTraceContext(), FlexlbTrace.BATCH_ID, batchId);
                FlexlbTrace.setScheduleAttribute(item.ctx().getTraceContext(), FlexlbTrace.BATCH_SIZE,
                        (long) items.size());
                FlexlbTrace.setScheduleAttribute(item.ctx().getTraceContext(), FlexlbTrace.DISPATCH_REASON, decisionReason);
            }
            invoked = true;
            var response = Objects.requireNonNull(grpcClient.batchEnqueueAsync(ip, port, request),
                    "EnqueueBatch client returned null future after invocation");
            registerEnqueueBatchCallback(items, batchId, observer, response);
        } catch (Throwable failure) {
            logFailure("Batch dispatch failed", batchId, failure);
            publishResult(items, batchId,
                    invoked ? DeliveryResult.uncertain(failure) : DeliveryResult.notSent(failure), observer);
        }
    }

    /** Register EnqueueBatch result publication and track callback completion for shutdown. */
    private void registerEnqueueBatchCallback(List<RequestRoute> items, long batchId,
                                   BiConsumer<RequestRoute, DeliveryResult> observer,
                                   CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> response) {
        // The dispatch permit is still held: shutdown cannot pass this observer registration.
        pendingCompletions.incrementAndGet();
        boolean registered = false;
        try {
            response.handleAsync((reply, failure) -> {
                if (failure != null) {
                    publishResult(items, batchId,
                            DeliveryResult.uncertain(unwrapCompletionFailure(failure)), observer);
                } else {
                    handleResponse(batchId, items,
                            Objects.requireNonNull(reply, "EnqueueBatch returned null response"), observer);
                }
                return null;
            }, completionExecutor).whenComplete((ignored, failure) -> {
                try {
                    if (failure != null) {
                        publishResult(items, batchId,
                                DeliveryResult.uncertain(unwrapCompletionFailure(failure)), observer);
                    }
                } finally {
                    finishCompletion();
                }
            });
            registered = true;
        } finally {
            if (!registered) {
                finishCompletion();
            }
        }
    }

    private static void logFailure(String operation, long batchId, Throwable failure) {
        try {
            Logger.error("{} batch_id={}", operation, batchId, failure);
        } catch (Throwable ignored) {
            // Logging cannot change transport ownership or interrupt remaining item callbacks.
        }
    }

    private static Throwable unwrapCompletionFailure(Throwable failure) {
        return failure instanceof CompletionException && failure.getCause() != null
                ? failure.getCause() : failure;
    }

    private void requireBatchDispatcher() {
        DispatcherConfig dispatcher =
                configService.loadBalanceConfig().getDispatcher();
        checkState(dispatcher.getType() == DispatcherConfig.Type.BATCH,
                "batch submission requires BATCH dispatcher configuration");
    }

    /** Publish the same transport fact to every member, isolating each observer. */
    private static void publishResult(List<RequestRoute> items, long batchId,
                                      DeliveryResult result,
                                      BiConsumer<RequestRoute, DeliveryResult> observer) {
        for (RequestRoute item : items) {
            try {
                observer.accept(item, result);
            } catch (Throwable callbackFailure) {
                logFailure("Dispatch callback failed request_id=" + item.requestId(), batchId, callbackFailure);
            }
        }
    }

    // ==================== Response parsing ====================

    private void handleResponse(long batchId, List<RequestRoute> items,
                                EngineRpcService.EnqueueBatchResponsePB response,
                                BiConsumer<RequestRoute,
                                        DeliveryResult> observer) {
        if (response.getBatchId() != batchId) {
            RuntimeException mismatch = new RuntimeException(
                    "EnqueueBatch batch_id mismatch: expected " + batchId
                            + " but got " + response.getBatchId());
            publishResult(items, batchId, DeliveryResult.uncertain(mismatch), observer);
            return;
        }
        Map<Long, DeliveryResult> acknowledgements = HashMap.newHashMap(items.size());
        List<String> protocolViolations = new ArrayList<>();
        for (RequestRoute item : items) {
            acknowledgements.put(item.requestId(), null);
        }
        for (var error : response.getErrorsList()) {
            long code = error.hasErrorInfo() ? error.getErrorInfo().getErrorCode() : 0L;
            String message = error.hasErrorInfo() ? error.getErrorInfo().getErrorMessage() : "missing error_info";
            acknowledge(acknowledgements, error.getRequestId(), DeliveryResult.prefillRejected(new RuntimeException(
                    "EnqueueBatch rejected request " + error.getRequestId() + " error_code=" + code + ": " + message)),
                    protocolViolations);
        }
        for (var success : response.getSuccessesList()) {
            acknowledge(acknowledgements, success.getRequestId(), DeliveryResult.delivered(), protocolViolations);
        }
        acknowledgements.forEach((requestId, result) -> {
            if (result == null) {
                protocolViolations.add("response is missing request_id=" + requestId);
            }
        });
        if (!protocolViolations.isEmpty()) {
            publishResult(
                    items,
                    batchId,
                    DeliveryResult.uncertain(new RuntimeException(
                            "Malformed EnqueueBatch response: "
                                    + String.join("; ", protocolViolations))),
                    observer);
            return;
        }

        // Record a validated RPC response before callbacks can publish a
        // Schedule response and end the request's SERVER span.
        long responseNanos = System.nanoTime();
        for (RequestRoute item : items) {
            FlexlbTrace.setScheduleDuration(item.ctx().getTraceContext(), FlexlbTrace.ENQUEUE_BATCH_MS,
                    item.ctx().getBatchDispatchedNanos(), responseNanos);
        }
        for (RequestRoute item : items) {
            try {
                observer.accept(item, acknowledgements.get(item.requestId()));
            } catch (Throwable callbackFailure) {
                // The callback may already have committed this item's state
                // before throwing. Never issue a second, contradictory
                // callback for it, and never let it reclassify earlier items.
                logFailure("EnqueueBatch callback failed request_id=" + item.requestId(), batchId, callbackFailure);
            }
        }
    }

    /** Null is an expected member awaiting ACK; a result can be installed exactly once. */
    private static void acknowledge(Map<Long, DeliveryResult> acknowledgements, long requestId,
                                    DeliveryResult result, List<String> violations) {
        String kind = result.status() == DeliveryResult.Status.DELIVERED ? "success" : "error";
        if (!acknowledgements.containsKey(requestId)) {
            violations.add(kind + " references unknown request_id=" + requestId);
        }
        DeliveryResult previous = acknowledgements.putIfAbsent(requestId, result);
        if (previous != null) {
            violations.add(previous.status() == result.status()
                    ? "duplicate " + kind + " for request_id=" + requestId
                    : "request_id appears in both success and error: " + requestId);
        }
    }

    // ==================== gRPC request building ====================

    private EngineRpcService.EnqueueBatchRequestPB buildBatchRequest(long batchId, List<RequestRoute> items)
            throws InvalidProtocolBufferException, InterruptedException {
        EngineRpcService.EnqueueBatchRequestPB.Builder builder =
                EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                        .setBatchId(batchId)
                        .setFetchAttachTimeoutMs(items.getFirst().ctx().getConfig().getDispatcher().getFetchAttachTimeoutMs());
        BatchRoleAddressCache roleAddresses = new BatchRoleAddressCache();
        long firstRank = items.getFirst().prefill().getDpRank();
        var firstSlot = EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank((int) firstRank);
        Map<Long, EngineRpcService.EnqueueBatchDpSlotPB.Builder> slots = null;
        for (RequestRoute item : items) {
            long rank = item.prefill().getDpRank();
            var slot = firstSlot;
            if (rank != firstRank) {
                if (slots == null) {
                    slots = new TreeMap<>();
                    slots.put(firstRank, firstSlot);
                }
                slot = slots.computeIfAbsent(rank, key ->
                        EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank(key.intValue()));
            }
            slot.addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                    .setInput(buildInput(item, roleAddresses)));
        }
        if (slots == null) {
            builder.addDpSlots(firstSlot);
        } else {
            slots.values().forEach(builder::addDpSlots);
        }
        return builder.build();
    }

    private EngineRpcService.GenerateInputPB buildInput(
            RequestRoute item,
            BatchRoleAddressCache roleAddresses)
            throws InvalidProtocolBufferException, InterruptedException {
        EngineRpcService.GenerateInputPB generateInput = item.ctx().getGenerateInput();
        if (generateInput == null) {
            throw new IllegalArgumentException("generateInputPb is missing for request " + item.requestId());
        }
        EngineRpcService.GenerateInputPB.Builder input = generateInput.toBuilder();
        checkArgument(input.getRequestId() == item.requestId(),
                "request_id mismatch between schedule request and GenerateInputPB");
        // This batch RPC carries independent requests. Propagate each Schedule
        // parent in its own payload, never in the shared RPC metadata.
        if (FlexlbTrace.isEnabled() && item.ctx().getTraceContext() != null) {
            try {
                var carrier = FlexlbTrace.inject(item.ctx().getTraceContext());
                String traceparent = carrier.get("traceparent");
                if (traceparent != null && !traceparent.isEmpty()) {
                    var traceContext = input.getRequestInfo().getTraceContext().toBuilder()
                            .setTraceparent(traceparent)
                            .setTracestate(carrier.getOrDefault("tracestate", ""))
                            .build();
                    input.getRequestInfoBuilder().setTraceContext(traceContext);
                }
            } catch (Throwable ignored) {
                // Tracing must not prevent dispatch or erase the incoming carrier.
            }
        }
        EngineRpcService.GenerateConfigPB.Builder config = input.getGenerateConfigBuilder();
        config.clearRoleAddrs();
        roleAddresses.prefill = addRoleAddr(config, item.prefill(), roleAddresses.prefill);
        roleAddresses.decode = addRoleAddr(config, item.decode(), roleAddresses.decode);
        // Preserve the priority frozen for Decode admission, including the zero sentinel.
        input.setPriority(item.priority());
        return input.build();
    }

    private static EngineRpcService.RoleAddrPB addRoleAddr(
            EngineRpcService.GenerateConfigPB.Builder config,
            ServerStatus status,
            EngineRpcService.RoleAddrPB cached) {
        if (status != null) {
            if (cached == null || !sameRoleAddr(cached, status)) {
                cached = buildRoleAddr(status);
            }
            config.addRoleAddrs(cached);
        }
        return cached;
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
        return cached.getRoleStr().equals(role.getCode())
                && cached.getIp().equals(serverStatus.getServerIp())
                && cached.getHttpPort() == serverStatus.getHttpPort()
                && cached.getGrpcPort() == serverStatus.getGrpcPort();
    }

    /** Reuses immutable role addresses while building one batch payload. */
    private static final class BatchRoleAddressCache {
        private EngineRpcService.RoleAddrPB prefill;
        private EngineRpcService.RoleAddrPB decode;
    }

    // ==================== Logging ====================

    private void logDispatch(long batchId, List<RequestRoute> items,
                             PrefillEndpoint prefillEp, long predMs, String reason) {
        if (!Logger.isDebugEnabled()) {
            return;
        }
        long totalTokens = 0;
        long totalHit = 0;
        StringBuilder itemDetail = new StringBuilder();
        for (int i = 0; i < items.size(); i++) {
            RequestRoute item = items.get(i);
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

        RequestRoute head = items.get(0);
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

}
