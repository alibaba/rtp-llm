package org.flexlb.balance.scheduler;

import com.google.common.collect.MapMaker;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.UnknownFieldSet;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Failures;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Builds and sends engine requests, validates batch replies, and runs bounded cancellation retries.
 * Late replies are applied to the request scheduler before shutdown drains the callbacks.
 */
public final class EngineDelivery {
    static final long CLEANUP_TIMEOUT_MS = 5_000L;
    private static final int MAX_ATTEMPTS = 8;

    private static final ThreadLocal<Boolean> executingStop = new ThreadLocal<>();

    private final EngineGrpcClient grpcClient;
    private final java.util.concurrent.Executor stopWorkers =
            task -> Thread.ofVirtual().name("delivery-stop-work").start(task);
    // MapMaker weak keys use identity: equal-valued claims must not share an operation.
    private final ConcurrentMap<Object, StopTask> sessions = new MapMaker().weakKeys().makeMap();
    private final java.util.Set<StopTask> activeStops =
            java.util.concurrent.ConcurrentHashMap.newKeySet();

    EngineDelivery(EngineGrpcClient grpcClient) {
        this.grpcClient = grpcClient;
    }

    CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> batchEnqueueAsync(
            String ip, int port, EngineRpcService.EnqueueBatchRequestPB request) {
        return grpcClient.batchEnqueueAsync(ip, port, request);
    }

    /**
     * Executes cancellation for the supplied operation identity and returns the engine reply.
     * The caller owns acknowledgement deadlines; a late RPC reply still completes this future.
     */
    CompletableFuture<CancelResult> cancel(
            RequestContext context,
            Object identity,
            String ip,
            int port,
            CancelReason reason,
            long timeoutMs) {
        var delivery = context.snapshot().delivery();
        var claim = delivery == null ? null : delivery.identity();
        var task = record(claim == null ? identity : claim, claim, context);
        CompletableFuture<CancelResult> response;
        synchronized (task) {
            if (task.closed
                    || task.stopRequested
                            && (task.cancelCall == null
                                    || task.cancelCall.identity() != identity)) {
                return CompletableFuture.failedFuture(
                        new java.util.concurrent.RejectedExecutionException(
                                "cancel operation closed"));
            }
            if (task.cancelCall == null || task.cancelCall.identity() != identity) {
                task.cancelCall =
                        new CancelCall(
                                identity,
                                ip,
                                port,
                                reason,
                                System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs),
                                new CompletableFuture<>());
            }
            response = task.cancelCall.response();
        }
        task.start();
        if (task.closed)
            response.completeExceptionally(
                    new java.util.concurrent.RejectedExecutionException("cancel operation closed"));
        return response;
    }

    boolean isStoppingOnCurrentThread() {
        return Boolean.TRUE.equals(executingStop.get());
    }

    void awaitStops() {
        Throwable problem = null;
        long deadline =
                System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CLEANUP_TIMEOUT_MS + 1_000L);
        for (var task : activeStops) {
            try {
                synchronized (task) {
                    while (task.taskScheduled
                            || task.applyingReplies != 0
                            || !task.closed && task.pending != 0) {
                        long remaining = deadline - System.nanoTime();
                        if (remaining <= 0) {
                            throw new java.util.concurrent.TimeoutException(
                                    "stop callbacks pending");
                        }
                        TimeUnit.NANOSECONDS.timedWait(task, remaining);
                    }
                }
            } catch (Exception failure) {
                if (failure instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                }
                task.finished.completeExceptionally(failure);
                task.close();
                problem = Failures.append(problem, failure);
            }
        }
        Failures.rethrow(problem, "delivery stop callbacks did not drain");
    }

    void deliveryChanged(Delivery.Member claim) {
        var task = sessions.get(claim);
        if (task != null) task.start();
    }

    /**
     * Accept the entire stop operation; callers do not drive retries or sender completion.
     * Completion ends this stop attempt; request resource cleanup has its own Member completion.
     */
    CompletableFuture<Void> stop(Delivery.Member claim) {
        for (; ; ) {
            if (claim.completion().toCompletableFuture().isDone()) {
                return CompletableFuture.completedFuture(null);
            }
            var delivery = claim.owner().snapshot().delivery();
            if (delivery == null || delivery.identity() != claim) {
                return CompletableFuture.completedFuture(null);
            }
            var task = record(claim, claim, claim.owner());
            synchronized (task) {
                if (task.closed) {
                    sessions.remove(claim, task);
                    continue;
                }
                if (task.stopRequested) {
                    return task.finished;
                }
                task.stopRequested = true;
            }
            task.start();
            return task.finished;
        }
    }

    /**
     * Validate input without allocating a decoded token array; Engine sees the same protobuf wire fields.
     */
    static GenerateInputPB parseInput(ByteString bytes) throws InvalidProtocolBufferException {
        try {
            CodedInputStream input = bytes.newCodedInput();
            var metadata = GenerateInputPB.newBuilder();
            var tokens = new Tokens();
            int retainedStart = 0;
            int tagStart = 0;
            for (int tag; (tag = input.readTag()) != 0; tagStart = input.getTotalBytesRead()) {
                if (tag == 16 || tag == 18) {
                    mergeMetadata(metadata, bytes, retainedStart, tagStart);
                    tokens.readField(tag, input, bytes);
                    retainedStart = input.getTotalBytesRead();
                    continue;
                }
                if (!input.skipField(tag)) {
                    input.checkLastTagWas(0);
                }
            }
            input.checkLastTagWas(0);
            mergeMetadata(metadata, bytes, retainedStart, bytes.size());
            var parsed = metadata.build();
            ByteString encoded = tokens.finish();
            if (!encoded.isEmpty() && bytes.asReadOnlyByteBufferList().size() > 1) {
                // A segmented input may use a copying stream decoder for metadata. Give
                // tokens their own compact storage so they cannot pin copied large fields.
                var compact = ByteString.newOutput(encoded.size());
                encoded.writeTo(compact);
                encoded = compact.toByteString();
            }
            if (!encoded.isEmpty()) {
                // Unknown fields use the standard protobuf writer. On the Engine this is the
                // known packed token_ids field again; no schema or RPC method changes.
                var field = UnknownFieldSet.Field.newBuilder().addLengthDelimited(encoded).build();
                parsed =
                        parsed.toBuilder()
                                .mergeUnknownFields(
                                        UnknownFieldSet.newBuilder().mergeField(2, field).build())
                                .build();
            }
            return parsed;
        } catch (InvalidProtocolBufferException malformed) {
            // Diagnostics can outlive dispatch. Never let a parse failure pin a partial
            // tensor/message.
            throw malformed.setUnfinishedMessage(null);
        } catch (IOException failure) {
            throw new InvalidProtocolBufferException(failure);
        }
    }

    /** O(1) wire-size bound using the route's exact, immutable endpoint addresses. */
    static long payloadSizeUpperBound(RouteResult route) {
        return route.ctx().data().getGenerateInputSize()
                + 64L
                + addressSizeUpperBound(route.prefillEndpoint().getIp())
                + (route.decodeEndpoint() == null
                        ? 0L
                        : addressSizeUpperBound(route.decodeEndpoint().getIp()));
    }

    static EngineRpcService.EnqueueBatchRequestPB prepare(
            long batchId, List<Delivery.Member> claims) {
        try {
            var first = claims.getFirst();
            var config = first.owner().data().getConfig();
            var builder =
                    EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                            .setBatchId(batchId)
                            .setFetchAttachTimeoutMs(
                                    config.getDispatcher().getFetchAttachTimeoutMs());
            var addresses = new RoleAddresses();
            long firstRank = first.route().prefillDpRank();
            var firstSlot =
                    EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank((int) firstRank);
            Map<Long, EngineRpcService.EnqueueBatchDpSlotPB.Builder> slots = null;
            for (Delivery.Member claim : claims) {
                var item = claim.route();
                long rank = item.prefillDpRank();
                var slot = firstSlot;
                if (rank != firstRank) {
                    if (slots == null) {
                        slots = new TreeMap<>();
                        slots.put(firstRank, firstSlot);
                    }
                    slot =
                            slots.computeIfAbsent(
                                    rank,
                                    key ->
                                            EngineRpcService.EnqueueBatchDpSlotPB.newBuilder()
                                                    .setDpRank(key.intValue()));
                }
                slot.addRequests(
                        EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                                .setInput(input(item, addresses)));
            }
            if (slots == null) {
                builder.addDpSlots(firstSlot);
            } else {
                slots.values().forEach(builder::addDpSlots);
            }
            return builder.build();
        } catch (Exception failure) {
            throw new IllegalArgumentException(
                    "Batch request build failed: " + failure.getMessage(), failure);
        }
    }

    /** Remove abandoned members from an already complete immutable payload without parsing again. */
    static EngineRpcService.EnqueueBatchRequestPB retain(
            EngineRpcService.EnqueueBatchRequestPB prepared, List<Delivery.Member> sending) {
        var ids = new HashSet<String>();
        for (Delivery.Member claim : sending) {
            ids.add(claim.route().requestId());
        }
        var filtered = prepared.toBuilder().clearDpSlots();
        for (var slot : prepared.getDpSlotsList()) {
            var selected = slot.toBuilder().clearRequests();
            for (var member : slot.getRequestsList()) {
                if (ids.contains(org.flexlb.engine.grpc.RequestId.parse(member.getInput()))) {
                    selected.addRequests(member);
                }
            }
            if (selected.getRequestsCount() > 0) {
                filtered.addDpSlots(selected);
            }
        }
        return filtered.build();
    }

    /** Results have the exact sending-claim order. One malformed member invalidates the entire ACK. */
    static List<Delivery.Result> validate(
            long batchId,
            List<Delivery.Member> claims,
            EngineRpcService.EnqueueBatchResponsePB response,
            Throwable failure) {
        if (failure != null || response == null || response.getBatchId() != batchId) {
            Throwable problem =
                    failure instanceof CompletionException && failure.getCause() != null
                            ? failure.getCause()
                            : failure;
            if (problem == null) {
                problem =
                        new IllegalStateException(
                                "Invalid EnqueueBatch response for batch " + batchId);
            }
            return Collections.nCopies(claims.size(), Delivery.Result.uncertain(problem));
        }
        var expected = HashMap.<String, Integer>newHashMap(claims.size());
        var results = new Delivery.Result[claims.size()];
        var violations = new ArrayList<String>();
        for (int i = 0; i < claims.size(); i++) {
            String id = claims.get(i).route().requestId();
            if (expected.put(id, i) != null) {
                violations.add("duplicate sending request_id=" + id);
            }
        }
        for (var error : response.getErrorsList()) {
            long code = error.hasErrorInfo() ? error.getErrorInfo().getErrorCode() : 0L;
            String message =
                    error.hasErrorInfo()
                            ? error.getErrorInfo().getErrorMessage()
                            : "missing error_info";
            acknowledge(
                    expected,
                    results,
                    org.flexlb.engine.grpc.RequestId.parse(error),
                    Delivery.Result.prefillRejected(
                            new RuntimeException(
                                    "EnqueueBatch rejected request "
                                            + org.flexlb.engine.grpc.RequestId.parse(error)
                                            + " error_code="
                                            + code
                                            + ": "
                                            + message)),
                    violations);
        }
        for (var success : response.getSuccessesList()) {
            acknowledge(
                    expected,
                    results,
                    org.flexlb.engine.grpc.RequestId.parse(success),
                    Delivery.Result.delivered(),
                    violations);
        }
        expected.forEach(
                (id, index) -> {
                    if (index >= 0 && results[index] == null) {
                        violations.add("response is missing request_id=" + id);
                    }
                });
        if (!violations.isEmpty()) {
            return Collections.nCopies(
                    claims.size(),
                    Delivery.Result.uncertain(
                            new IllegalStateException(
                                    "Malformed EnqueueBatch response: "
                                            + String.join("; ", violations))));
        }
        return List.of(results);
    }

    private static ContextResult.CancelEvidenceKind cancelEvidence(
            EngineGrpcClient.CancelReply reply) {
        return switch (reply) {
            case REQUEST_CLEANED -> ContextResult.CancelEvidenceKind.CLEANED;
            case REQUEST_FENCED -> ContextResult.CancelEvidenceKind.FENCED;
            case ACCEPTED -> ContextResult.CancelEvidenceKind.ACKNOWLEDGED;
            case NOT_FOUND -> ContextResult.CancelEvidenceKind.NOT_FOUND;
            case FAILED -> ContextResult.CancelEvidenceKind.UNKNOWN;
        };
    }

    private StopTask record(Object key, Delivery.Member claim, RequestContext context) {
        var existing = sessions.get(key);
        if (existing != null) {
            return existing;
        }
        var created = new StopTask(key, claim, context);
        var previous = sessions.putIfAbsent(key, created);
        if (previous != null) {
            activeStops.remove(created);
            return previous;
        }
        if (claim != null) {
            claim.completion().whenComplete((ignored, failure) -> created.close());
        }
        return created;
    }

    private static void mergeMetadata(
            GenerateInputPB.Builder metadata, ByteString bytes, int start, int end)
            throws IOException {
        if (start == end) {
            return;
        }
        var input = bytes.substring(start, end).newCodedInput();
        // ByteString is immutable: tensors and unknown byte fields can share its storage.
        input.enableAliasing(true);
        metadata.mergeFrom(input);
        input.checkLastTagWas(0);
    }

    private static long addressSizeUpperBound(String ip) {
        return 64L + (ip == null ? 0L : 3L * ip.length());
    }

    private static void acknowledge(
            Map<String, Integer> expected,
            Delivery.Result[] results,
            String id,
            Delivery.Result result,
            List<String> violations) {
        String kind = result.status() == Delivery.Result.Status.DELIVERED ? "success" : "error";
        Integer index = expected.get(id);
        if (index == null || index < 0) {
            violations.add(kind + " references unknown request_id=" + id);
            int unknownKind = result.status() == Delivery.Result.Status.DELIVERED ? -1 : -2;
            if (index == null) {
                expected.put(id, unknownKind);
            } else {
                violations.add(
                        index == unknownKind
                                ? "duplicate " + kind + " for request_id=" + id
                                : "request_id appears in both success and error: " + id);
            }
            return;
        }
        var previous = results[index];
        if (previous != null) {
            violations.add(
                    previous.status() == result.status()
                            ? "duplicate " + kind + " for request_id=" + id
                            : "request_id appears in both success and error: " + id);
        } else {
            results[index] = result;
        }
    }

    private static EngineRpcService.GenerateInputPB input(RouteResult item, RoleAddresses addresses)
            throws Exception {
        var data = item.ctx().data();
        var prepared = data.getGenerateInput();
        if (prepared == null) {
            throw new IllegalArgumentException(
                    "generateInputPb is missing for request " + item.requestId());
        }
        var input = prepared.toBuilder();
        checkArgument(
                java.util.Objects.equals(
                        org.flexlb.engine.grpc.RequestId.parse(input), item.requestId()),
                "request_id mismatch between schedule request and GenerateInputPB");
        if (FlexlbTrace.isEnabled() && data.getTraceContext() != null) {
            try {
                var carrier = FlexlbTrace.inject(data.getTraceContext());
                String parent = carrier.get("traceparent");
                if (parent != null && !parent.isEmpty()) {
                    input.getRequestInfoBuilder()
                            .setTraceContext(
                                    input.getRequestInfo().getTraceContext().toBuilder()
                                            .setTraceparent(parent)
                                            .setTracestate(carrier.getOrDefault("tracestate", "")));
                }
            } catch (Throwable ignored) {
            }
        }
        var config = input.getGenerateConfigBuilder().clearRoleAddrs();
        var prefill = item.role(RoleType.PREFILL);
        var decode = item.role(RoleType.DECODE);
        if (prefill != null) {
            addresses.prefill = role(prefill, addresses.prefill);
            config.addRoleAddrs(addresses.prefill);
        }
        if (decode != null) {
            addresses.decode = role(decode, addresses.decode);
            config.addRoleAddrs(addresses.decode);
        }
        return input.setPriority(item.priority()).build();
    }

    private static EngineRpcService.RoleAddrPB role(
            RoleSelection selected, EngineRpcService.RoleAddrPB cached) {
        var role = selected.role();
        if (cached != null
                && cached.getRoleStr().equals(role.getCode())
                && cached.getIp().equals(selected.ip())
                && cached.getHttpPort() == selected.httpPort()
                && cached.getGrpcPort() == selected.grpcPort()) {
            return cached;
        }
        return EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(RoleTypeProtoConverter.toRoleAddrProto(role))
                .setRoleStr(role.getCode())
                .setIp(selected.ip())
                .setHttpPort(selected.httpPort())
                .setGrpcPort(selected.grpcPort())
                .build();
    }

    /**
     * Engine cancellation reply and the exact delivery receipt, if a delivery existed.
     * The caller records the receipt together with its own request update.
     */
    public record CancelResult(
            EngineGrpcClient.CancelReply reply, ContextResult.CleanupAcknowledgement cleanup) {}

    private record CancelCall(
            Object identity,
            String ip,
            int port,
            CancelReason reason,
            long expiresAt,
            CompletableFuture<CancelResult> response) {}

    /**
     * Serializes cancellation attempts and applies their replies to the owning request before completing.
     */
    private final class StopTask implements Runnable {
        private final Object key;
        private final Delivery.Member claim;
        private final RequestContext context;
        private final AbstractRequestScheduler owner;
        private final CompletableFuture<Void> finished = new CompletableFuture<>();

        private volatile CancelCall cancelCall;

        private volatile boolean remoteCleanupConfirmed;
        private volatile boolean closed, stopRequested;

        private volatile Thread runner;

        // One scheduled task serializes retries; callbacks are counted separately for shutdown.
        private int pending, applyingReplies;
        private boolean taskScheduled;
        private boolean rerunRequested;

        // Only the running workflow changes its retry state.
        // Completion means the Scheduler has applied the receipt, including late receipts.
        private CompletableFuture<Long> rpc;
        private CancelCall attemptedCancel;

        private boolean postSend;
        private int attempts;
        private long deadline, rpcDeadline;

        StopTask(Object key, Delivery.Member claim, RequestContext context) {
            this.key = key;
            this.claim = claim;
            this.context = context;
            this.owner = context.data().getRequestScheduler();
            activeStops.add(this);
        }

        @Override
        public void run() {
            executingStop.set(true);
            runner = Thread.currentThread();
            var operation = cancelCall;
            try {
                if (shouldStopSendingCancel()) {
                    finished.complete(null);
                } else if (stopRequested) {
                    stopRemotely();
                } else if (operation != null && operation != attemptedCancel) {
                    attemptedCancel = operation;
                    if (System.nanoTime() < operation.expiresAt()) {
                        attempts = 0;
                        requestProof(operation);
                    }
                }
            } catch (RuntimeException | Error failure) {
                finished.completeExceptionally(failure);
                owner.recordRequestFailure(failure);
            } finally {
                boolean unused;
                boolean rerun;
                synchronized (this) {
                    executingStop.remove();
                    runner = null;
                    taskScheduled = false;
                    rerun = rerunRequested && !closed;
                    rerunRequested = false;
                    unused =
                            !stopRequested
                                    && attempts == 0
                                    && pending == 0
                                    && operation == cancelCall;
                    if (unused) {
                        closed = true;
                    }
                    notifyProgress();
                }
                if (unused) {
                    close();
                } else if (rerun) {
                    start();
                }
            }
        }

        private void start() {
            synchronized (this) {
                if (closed) {
                    return;
                }
                if (taskScheduled) {
                    rerunRequested = true;
                    signal();
                    return;
                }
                taskScheduled = true;
            }
            signal();
            try {
                stopWorkers.execute(this);
            } catch (RuntimeException | Error failure) {
                try {
                    finished.completeExceptionally(failure);
                    close();
                    owner.recordRequestFailure(failure);
                } finally {
                    synchronized (this) {
                        taskScheduled = false;
                        notifyProgress();
                    }
                }
            }
        }

        private void signal() {
            LockSupport.unpark(runner);
        }

        private void close() {
            boolean drained;
            synchronized (this) {
                closed = true;
                drained = pending == 0;
                sessions.remove(key, this);
                notifyProgress();
            }
            signal();
            finished.complete(null);
            if (drained) {
                var operation = cancelCall;
                if (operation != null) {
                    operation
                            .response()
                            .completeExceptionally(
                                    new java.util.concurrent.RejectedExecutionException(
                                            "cancel operation closed"));
                }
            }
        }

        /**
         * Called under this task's monitor after task or reply progress changes.
         */
        private void notifyProgress() {
            if (closed && !taskScheduled && applyingReplies == 0) activeStops.remove(this);
            notifyAll();
        }

        private void stopRemotely() {
            if (deadline == 0) {
                deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CLEANUP_TIMEOUT_MS);
            }
            while (!shouldStopSendingCancel()) {
                var delivery = context.snapshot().delivery();
                var send = delivery == null ? ContextResult.SendState.UNKNOWN : delivery.send();
                boolean sending =
                        send == ContextResult.SendState.NOT_STARTED
                                || send == ContextResult.SendState.SENDING;
                if (!postSend && !sending) {
                    postSend = true;
                    deadline =
                            System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CLEANUP_TIMEOUT_MS);
                }
                if (System.nanoTime() >= deadline) {
                    break;
                }
                if (send == ContextResult.SendState.NOT_STARTED || sending && attempts > 0) {
                    await(deadline, () -> context.snapshot().delivery().send() != send);
                    continue;
                }
                if (rpc != null) {
                    await(Math.min(deadline, rpcDeadline), rpc::isDone);
                    long completedAt = rpc.getNow(System.nanoTime());
                    rpc = null;
                    await(
                            Math.min(
                                    deadline,
                                    completedAt
                                            + TimeUnit.MILLISECONDS.toNanos(
                                                    Math.min(400L, 25L << (attempts - 1)))),
                            () -> false);
                }
                if (shouldStopSendingCancel()
                        || System.nanoTime() >= deadline
                        || attempts >= MAX_ATTEMPTS) {
                    break;
                }
                requestProof(cancelCall);
            }
            if (shouldStopSendingCancel()) finished.complete(null);
            else finishUnresolved();
        }

        private boolean shouldStopSendingCancel() {
            if (closed) {
                return true;
            }
            var delivery = context.snapshot().delivery();
            if (claim != null && (delivery == null || delivery.identity() != claim)) {
                close();
                return true;
            }
            if (delivery != null
                    && (delivery.send() == ContextResult.SendState.NOT_SENT
                            || delivery.send() == ContextResult.SendState.ROUTE_CLOSED)) {
                return true;
            }
            if (delivery != null && delivery.prefillProof() && delivery.decodeProof()) {
                return true;
            }
            return remoteCleanupConfirmed;
        }

        private void await(long until, BooleanSupplier ready) {
            while (!shouldStopSendingCancel() && !ready.getAsBoolean()) {
                long remaining = until - System.nanoTime();
                if (remaining <= 0) {
                    return;
                }
                LockSupport.parkNanos(this, remaining);
                if (Thread.currentThread().isInterrupted()) {
                    throw new IllegalStateException("delivery stop interrupted");
                }
            }
        }

        private void requestProof(CancelCall operation) {
            var state = context.snapshot();
            var reason =
                    state.cancellation() != null
                            ? state.cancellation().reason()
                            : state.delivery() != null && state.delivery().abandonment() != null
                                    ? state.delivery().abandonment()
                                    : operation.reason();
            // An expired unsent call must not replace a later cancellation.
            if (attempts == 0 && operation != null && System.nanoTime() >= operation.expiresAt()) {
                if (!stopRequested) {
                    return;
                }
                operation = null;
            }
            long timeout =
                    attempts++ == 0 && operation != null
                            ? Math.max(
                                    1L,
                                    TimeUnit.NANOSECONDS.toMillis(
                                            operation.expiresAt() - System.nanoTime()))
                            : 500L;
            synchronized (this) {
                pending++;
            }
            rpcDeadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeout);
            var exact = operation;
            try {
                var remote =
                        Objects.requireNonNull(
                                grpcClient.cancelAsync(
                                        operation == null
                                                ? claim.route().prefillEndpoint().getIp()
                                                : operation.ip(),
                                        operation == null
                                                ? claim.route().prefillEndpoint().getGrpcPort()
                                                : operation.port(),
                                        reason.toRequest(context.data().getRequestId()),
                                        timeout),
                                "cancel future");
                rpc =
                        remote.handle(
                                (reply, failure) ->
                                        applyReply(
                                                failure == null && reply != null
                                                        ? reply
                                                        : EngineGrpcClient.CancelReply.FAILED,
                                                exact));
                rpc.whenComplete((ignored, failure) -> signal());
            } catch (RuntimeException | Error failed) {
                if (failed instanceof Error) {
                    owner.recordRequestFailure(failed);
                }
                rpc =
                        CompletableFuture.completedFuture(
                                applyReply(EngineGrpcClient.CancelReply.FAILED, exact));
            }
        }

        private long applyReply(EngineGrpcClient.CancelReply reply, CancelCall operation) {
            synchronized (this) {
                applyingReplies++;
            }
            var previous = executingStop.get();
            executingStop.set(true);
            var evidence = EngineGrpcClient.CancelReply.FAILED;
            try {
                var cleanup =
                        claim == null
                                ? null
                                : new ContextResult.CleanupAcknowledgement(
                                        claim, cancelEvidence(reply), System.currentTimeMillis());
                boolean returned =
                        operation != null
                                && operation.response().complete(new CancelResult(reply, cleanup));
                if (!returned && cleanup != null) {
                    context.data().getRequestScheduler().recordRemoteCleanup(cleanup);
                }
                evidence = reply;
            } catch (RuntimeException | Error failed) {
                owner.recordRequestFailure(failed);
            } finally {
                if (evidence == EngineGrpcClient.CancelReply.REQUEST_CLEANED) {
                    remoteCleanupConfirmed = true;
                }
                if (operation != null) {
                    operation.response().complete(new CancelResult(evidence, null));
                }
                synchronized (this) {
                    pending--;
                    applyingReplies--;
                    notifyProgress();
                }
                if (claim == null) {
                    close();
                }
                if (previous == null) executingStop.remove();
                else executingStop.set(previous);
            }
            return System.nanoTime();
        }

        private void finishUnresolved() {
            if (finished.complete(null)) {
                owner.recordRequestFailure(
                        new IllegalStateException(
                                "delivery stop ended without release proof: request_id="
                                        + context.data().getRequestId()));
            }
        }
    }

    /**
     * Combines packed and unpacked token fields while preserving the original token sequence.
     */
    private static final class Tokens {
        private ByteString packed;
        private ByteString.Output combined;
        private CodedOutputStream writer;

        private void readField(int tag, CodedInputStream input, ByteString bytes)
                throws IOException {
            if (tag == 16) {
                writer().writeInt32NoTag(input.readInt32());
                return;
            }
            int length = input.readRawVarint32();
            int previousLimit = input.pushLimit(length);
            int start = input.getTotalBytesRead();
            while (input.getBytesUntilLimit() > 0) {
                input.readInt32();
            }
            input.popLimit(previousLimit);
            ByteString chunk = bytes.substring(start, input.getTotalBytesRead());
            if (packed == null && writer == null) {
                packed = chunk;
            } else {
                writer().writeRawBytes(chunk);
            }
        }

        private CodedOutputStream writer() throws IOException {
            if (writer == null) {
                combined = ByteString.newOutput(packed == null ? 32 : Math.max(32, packed.size()));
                writer = CodedOutputStream.newInstance(combined, 64);
                if (packed != null) {
                    writer.writeRawBytes(packed);
                    packed = null;
                }
            }
            return writer;
        }

        private ByteString finish() throws IOException {
            if (writer == null) {
                return packed == null ? ByteString.EMPTY : packed;
            }
            writer.flush();
            return combined.toByteString();
        }
    }

    private static final class RoleAddresses {
        private EngineRpcService.RoleAddrPB prefill;
        private EngineRpcService.RoleAddrPB decode;
    }
}
