package org.flexlb.mock.client;

import io.grpc.*;
import io.grpc.stub.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.*;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.mock.protocol.*;

/**
 * One native Schedule -> Fetch/Generate exchange. Frontend-style primary/backup scheduling; no Engine fallback.
 */
public final class NativeClient implements AutoCloseable {
    private final java.util.function.Supplier<List<String>> masters;
    private final java.util.function.Consumer<ObjectNode> events;
    private final VitClient vision;

    private final long cacheKeySeed;
    private final int prefillCpSize;

    private final Map<String, ManagedChannel> channels = new HashMap<>();
    private final Map<Long, Call> active = new HashMap<>();
    private final ScheduledThreadPoolExecutor timers = new ScheduledThreadPoolExecutor(1,
        Thread.ofPlatform().daemon().name("mock-client-deadline").factory());
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();

    private boolean closed, closing;

    public NativeClient(String master) {
        this(() -> List.of(master));
        if (master == null || master.isBlank()) throw new IllegalArgumentException("Master gRPC address required");
    }

    public NativeClient(java.util.function.Supplier<List<String>> masters) {
        this(masters, event -> {});
    }

    NativeClient(java.util.function.Supplier<List<String>> masters, java.util.function.Consumer<ObjectNode> events) {
        this(masters, events, MediaInput.IMAGE_TOKEN);
    }

    NativeClient(java.util.function.Supplier<List<String>> masters, java.util.function.Consumer<ObjectNode> events, int imageToken) {
        this(masters, events, imageToken, 0L);
    }

    NativeClient(java.util.function.Supplier<List<String>> masters, java.util.function.Consumer<ObjectNode> events, int imageToken, long seed) {
        this(masters, events, imageToken, seed, 1);
    }

    NativeClient(java.util.function.Supplier<List<String>> masters, java.util.function.Consumer<ObjectNode> events, int imageToken, long seed, int prefillCpSize) {
        cacheKeySeed = seed;
        this.prefillCpSize = prefillCpSize;
        vision = new VitClient(imageToken);
        this.masters = Objects.requireNonNull(masters); this.events = Objects.requireNonNull(events);
        timers.setRemoveOnCancelPolicy(true);
    }

    public Call send(FlexlbScheduleRequestPB request, Duration timeout) {
        Objects.requireNonNull(timeout);
        long budget = timeout.toNanos();
        if (budget <= 0) throw new IllegalArgumentException("Positive request timeout required");
        GenerateInputPB input;
        try { input = GenerateInputPB.parseFrom(request.getGenerateInput()); }
        catch (com.google.protobuf.InvalidProtocolBufferException error) { throw new IllegalArgumentException("Invalid native generate_input", error); }
        if (request.getRequestId() <= 0 || input.getRequestId() != request.getRequestId() || input.getTokenIdsCount() == 0
                || input.getTokenIdsCount() != request.getSeqLen() || input.getGenerateConfig().getMaxNewTokens() < 1
                || input.getGenerateConfig().getMaxNewTokens() != request.getMaxNewTokens() || request.getVitRouteOnly()
                || input.getGenerateConfig().getNumBeams() > 1 || input.getGenerateConfig().getNumReturnSequences() > 1)
            throw new IllegalArgumentException("Client currently requires matching single-sequence request metadata");
        if (input.getGenerateConfig().getTimeoutMs() > 0) budget = Math.min(budget, TimeUnit.MILLISECONDS.toNanos(input.getGenerateConfig().getTimeoutMs()));
        long baseBlockSize = request.getCacheKeyBlockSize();
        if (baseBlockSize > 0) {
            var routed = request.toBuilder().setCacheKeyBlockSize(Math.multiplyExact(Math.toIntExact(baseBlockSize), prefillCpSize));
            if (input.getMultimodalInputsCount() == 0) {
                routed.clearBlockCacheKeys().addAllBlockCacheKeys(routingKeys(MediaInput.text(input), baseBlockSize));
            }
            request = routed.build();
        }
        Call call;
        synchronized (this) {
            if (closed || closing) throw new IllegalStateException("Client closed");
            if (active.containsKey(request.getRequestId())) throw new IllegalArgumentException("Active request_id already exists");
            call = new Call(request, input, budget, baseBlockSize);
            events.accept(call.event("ISSUED").put("epoch_ms", Long.toString(call.sentAt.toEpochMilli())));
            active.put(request.getRequestId(), call);
        }
        synchronized (call) {
            if (call.terminal) return call;
            call.context.addListener(c -> {
                if (c.getDeadline().isExpired()) call.finish(Status.DEADLINE_EXCEEDED, 0, Outcome.timeout);
            }, Runnable::run);
            if (!call.terminal) call.context.run(() -> call.schedule(0));
        }
        return call;
    }

    @Override public void close() {
        List<Call> calls;
        synchronized (this) { if (closed || closing) return; closing = true; calls = List.copyOf(active.values()); }
        for (var call : calls) call.cancel();
        timers.shutdownNow(); vision.close(); workers.shutdown();
        try { workers.awaitTermination(2, TimeUnit.SECONDS); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        workers.shutdownNow();
        List<ManagedChannel> owned;
        synchronized (this) { closed = true; owned = List.copyOf(channels.values()); }
        owned.forEach(ManagedChannel::shutdownNow);
        for (var channel : owned) try { channel.awaitTermination(3, TimeUnit.SECONDS); }
        catch (InterruptedException error) { Thread.currentThread().interrupt(); break; }
    }

    private synchronized ManagedChannel channel(String target) {
        if (closed) throw new IllegalStateException("Client closed");
        return channels.computeIfAbsent(target, t -> ManagedChannelBuilder.forTarget(t).usePlaintext().build());
    }

    private List<Long> routingKeys(MediaInput.Prepared input, long baseBlockSize) {
        if (baseBlockSize <= 0) return List.of();
        return MediaInput.sequence(input, Math.toIntExact(baseBlockSize), cacheKeySeed, prefillCpSize).keys();
    }

    private static String message(Status status) {
        String text = status.getDescription() == null ? "" : status.getDescription();
        return text.length() <= 4096 ? text : text.substring(0, 4096) + " [truncated]";
    }

    private static String address(FlexlbServerStatusPB server) {
        if (server.getServerIp().isBlank() || server.getGrpcPort() <= 0 || server.getGrpcPort() > 65535)
            throw Status.DATA_LOSS.withDescription("Invalid native route address").asRuntimeException();
        return (server.getServerIp().contains(":") ? "[" + server.getServerIp() + "]" : server.getServerIp()) + ":" + server.getGrpcPort();
    }

    public enum Outcome { success, error, cancelled, timeout }

    public record Result(long requestId, Instant sentAt, Long ttftNanos, long elapsedNanos,
                         int outputTokens, Outcome outcome, String grpcCode, long errorCode, String message, Long tpotNanos) {}

    /**
     * Tracks one generation from Master routing and optional ViT preparation through streaming completion.
     */
    public final class Call {
        private FlexlbScheduleRequestPB request;
        private GenerateInputPB input;

        private final List<String> routes;

        private String scheduledMaster;

        private boolean visionPrepared, visionRerouted;

        private final Context.CancellableContext context;
        private final CompletableFuture<Result> result = new CompletableFuture<>();
        private final CompletableFuture<Long> firstToken = new CompletableFuture<>();
        private final Instant sentAt = Instant.now();

        private final long started = System.nanoTime();

        private Long ttft;

        private long lastTokenNanos;
        private long batchId;
        private int outputTokens, scheduleAttempt;
        private boolean pendingSchedule;
        private boolean terminal, finalFrame, scheduleStarted, routeRejected;

        private final long baseBlockSize;

        private Call(FlexlbScheduleRequestPB request, GenerateInputPB input, long budget, long baseBlockSize) {
            this.request = request; this.input = input;
            this.baseBlockSize = baseBlockSize;
            routes = List.copyOf(masters.get());
            if (routes.size() > 2 || new HashSet<>(routes).size() != routes.size() || routes.stream().anyMatch(String::isBlank))
                throw new IllegalArgumentException("Expected distinct primary and optional backup Master");
            context = Context.ROOT.withDeadlineAfter(budget, TimeUnit.NANOSECONDS, timers);
        }

        public CompletionStage<Result> completion() { return result.minimalCompletionStage(); }

        public CompletionStage<Long> firstToken() { return firstToken.minimalCompletionStage(); }

        public void cancel() { finish(Status.CANCELLED.withDescription("Client cancelled request"), 0, Outcome.cancelled); }

        private ObjectNode event(String kind) {
            return JsonNodeFactory.instance.objectNode().put("kind", kind)
                .put("request_id", Long.toString(request.getRequestId()))
                .put("epoch_ms", Long.toString(System.currentTimeMillis()))
                .put("elapsed_nanos", Long.toString(System.nanoTime() - started));
        }

        private void emit(String kind) { events.accept(event(kind)); }

        private void scheduleResult(Status status, long code, Boolean responseSuccess) {
            if (!pendingSchedule) return;
            pendingSchedule = false;
            var result = event("SCHEDULE_RESULT").put("attempt", scheduleAttempt).put("master", scheduledMaster)
                .put("grpc_code", status.getCode().name()).put("error_code", Long.toString(code)).put("message", message(status));
            if (responseSuccess != null) result.put("response_success", responseSuccess);
            events.accept(result);
        }

        private void schedule(int attempt) {
            if (routes.isEmpty()) { finish(Status.UNAVAILABLE.withDescription("No healthy Master"), 0, Outcome.error); return; }
            try {
                scheduledMaster = routes.get(attempt);
                scheduleAttempt++; pendingSchedule = true;
                events.accept(event("SCHEDULE_START").put("attempt", scheduleAttempt).put("master", scheduledMaster));
                boolean visionRoute = MediaInput.present(input) && !visionPrepared;
                if (!visionRoute) scheduleStarted = true;
                var query = visionRoute ? request.toBuilder().setVitRouteOnly(true).clearSelectedVit()
                    .clearCacheAffinityKeys().addAllCacheAffinityKeys(input.getMultimodalInputsList().stream().map(MediaInput::key).toList()).build() : request;
                FlexlbServiceGrpc.newStub(channel(scheduledMaster)).schedule(query, new StreamObserver<>() {
                    private boolean received;
                    @Override public void onNext(FlexlbScheduleResponsePB route) {
                        synchronized (Call.this) {
                            if (terminal) return;
                            if (received) { finish(Status.DATA_LOSS.withDescription("Multiple Schedule replies"), 0, Outcome.error); return; }
                            received = true;
                            scheduleResult(Status.OK.withDescription(route.getErrorMessage()), route.getCode(), route.getSuccess());
                            batchId = route.hasLifecycle() ? route.getLifecycle().getBatchId() : 0;
                            if (!route.getSuccess() && route.getCode() == 8408 && visionPrepared && !visionRerouted) {
                                // A new Master has its own worker generation; acquire a fresh V pin once.
                                visionPrepared = false; visionRerouted = true; request = request.toBuilder().clearSelectedVit().build();
                                context.run(() -> schedule(attempt)); return;
                            }
                            if (!route.getSuccess() || route.getCode() != 200) {
                                routeRejected = !route.hasLifecycle();
                                finish(Status.FAILED_PRECONDITION.withDescription(route.getErrorMessage()), route.getCode(), Outcome.error); return;
                            }
                            if (visionRoute) prepareVision(route);
                            else context.run(() -> consume(route));
                        }
                    }
                    @Override public void onError(Throwable error) {
                        synchronized (Call.this) {
                            if (terminal) return;
                            scheduleResult(Status.fromThrowable(error), 0, null);
                            var code = Status.fromThrowable(error).getCode();
                            // Real Frontend retries transport failures once; never replay a reply or deadline.
                            if (!received && !context.isCancelled() && code != Status.Code.DEADLINE_EXCEEDED
                                    && attempt + 1 < routes.size())
                                context.run(() -> schedule(attempt + 1));
                            else fail(error);
                        }
                    }
                    @Override public void onCompleted() {
                        synchronized (Call.this) { if (!received) finish(Status.DATA_LOSS.withDescription("Empty Schedule response"), 0, Outcome.error); }
                    }
                });
            } catch (RuntimeException error) { fail(error); }
        }

        private synchronized void finish(Status status, long code, Outcome outcome) {
            if (terminal) return;
            terminal = true;
            scheduleResult(status, code, null);
            if (!firstToken.isDone()) firstToken.completeExceptionally(status.asRuntimeException());
            context.cancel(null);
            synchronized (NativeClient.this) { active.remove(request.getRequestId(), this); }
            if (outcome != Outcome.success && scheduleStarted && !routeRejected) {
                // Cancel against this request's last attempted Master, not a newer discovery snapshot.
                try { workers.submit(() -> {
                    try {
                        FlexlbServiceGrpc.newBlockingStub(channel(scheduledMaster)).withDeadlineAfter(1, TimeUnit.SECONDS).cancel(
                            FlexlbCancelRequestPB.newBuilder().setRequestId(request.getRequestId()).setBatchId(batchId)
                                .setReason(outcome == Outcome.timeout ? CancelReasonPB.CANCEL_REASON_DEADLINE_EXCEEDED : CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED).build());
                    } catch (RuntimeException failure) {
                        System.getLogger(NativeClient.class.getName()).log(System.Logger.Level.WARNING,
                            "Best-effort Master Cancel failed for request " + request.getRequestId() + ": " + Status.fromThrowable(failure).getCode());
                    }
                }); } catch (RejectedExecutionException closing) { /* Client shutdown owns the remaining channels. */ }
            }
            var ended = event("TERMINAL").put("outcome", outcome.name()).put("grpc_code", status.getCode().name())
                .put("error_code", Long.toString(code)).put("message", message(status)).put("output_tokens", outputTokens);
            if (ttft != null) ended.put("ttft_nanos", Long.toString(ttft));
            events.accept(ended);
            Long tpot = ttft != null && outputTokens > 1 ? (lastTokenNanos - ttft) / (outputTokens - 1) : null;
            result.complete(new Result(request.getRequestId(), sentAt, ttft, ended.path("elapsed_nanos").asLong(), outputTokens,
                outcome, status.getCode().name(), code, status.getDescription() == null ? "" : status.getDescription(), tpot));
        }

        private void prepareVision(FlexlbScheduleResponsePB route) {
            try {
                var targets = route.getServerStatusList().stream().filter(s -> s.getRole().equals("VIT")).toList();
                if (targets.size() != 1 || route.getEnqueuedByMaster() || route.hasLifecycle())
                    throw Status.DATA_LOSS.withDescription("Invalid V-only route").asRuntimeException();
                var selected = targets.getFirst(); address(selected);
                var source = input.toBuilder();
                source.getGenerateConfigBuilder().clearRoleAddrs().addRoleAddrs(RoleAddrPB.newBuilder()
                    .setRole(RoleAddrPB.RoleType.VIT).setIp(selected.getServerIp()).setGrpcPort(selected.getGrpcPort()));
                var media = source.build();
                workers.execute(() -> context.run(() -> {
                    try {
                        var prepared = vision.routingInput(media,
                            System.nanoTime() + Math.max(0, context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS)));
                        synchronized (Call.this) {
                            if (terminal) return;
                            visionPrepared = true;
                            request = request.toBuilder().setGenerateInput(media.toByteString()).setSeqLen(prepared.input().getTokenIdsCount())
                                .setSelectedVit(selected).clearBlockCacheKeys()
                                .addAllBlockCacheKeys(routingKeys(prepared, baseBlockSize)).build();
                            schedule(0);
                        }
                    } catch (RuntimeException error) { fail(error); }
                }));
            } catch (RuntimeException error) { fail(error); }
        }

        private void fail(Throwable error) {
            long code = 0;
            var trailers = Status.trailersFromThrowable(error);
            if (trailers != null) {
                var raw = trailers.get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER));
                if (raw != null) try { code = ErrorDetailsPB.parseFrom(raw).getErrorCode(); }
                catch (com.google.protobuf.InvalidProtocolBufferException malformed) { /* Preserve the transport status. */ }
            }
            var status = Status.fromThrowable(error);
            finish(status, code, status.getCode() == Status.Code.DEADLINE_EXCEEDED ? Outcome.timeout : Outcome.error);
        }

        private void consume(FlexlbScheduleResponsePB route) {
            try {
                if (route.hasLifecycle() && route.getLifecycle().getRequestId() != request.getRequestId())
                    throw Status.DATA_LOSS.withDescription("Schedule lifecycle request_id mismatch").asRuntimeException();
                var prefills = route.getServerStatusList().stream().filter(s -> s.getRole().equals("PREFILL")).toList();
                if (prefills.size() != 1) throw Status.DATA_LOSS.withDescription("Exactly one Prefill route required").asRuntimeException();
                var p = prefills.getFirst();
                var stub = RpcServiceGrpc.newStub(channel(address(p)));
                var responses = new ClientResponseObserver<Object, GenerateOutputsPB>() {
                    private ClientCallStreamObserver<Object> stream;
                    @Override public void beforeStart(ClientCallStreamObserver<Object> stream) {
                        this.stream = stream; stream.disableAutoRequestWithInitial(1);
                    }
                    @Override public void onNext(GenerateOutputsPB output) {
                        synchronized (Call.this) {
                            if (terminal) return;
                            if (output.hasErrorInfo() && output.getErrorInfo().getErrorCodeValue() != 0) {
                                finish(Status.UNKNOWN.withDescription(output.getErrorInfo().getErrorMessage()), output.getErrorInfo().getErrorCodeValue(), Outcome.error); return;
                            }
                            var flat = output.getFlattenOutput();
                            int length = flat.getAuxInfoCount() == 1 ? flat.getAuxInfo(0).getOutputLen() : 0;
                            int width = flat.getOutputIds().getInt32Data().size() / 4;
                            // A context-bound terminal frame may carry no new tokens after P's first output.
                            boolean emptyFinal = flat.getFinishedCount() == 1 && flat.getFinished(0)
                                && outputTokens > 0 && length == outputTokens && width == 0;
                            if (finalFrame || output.getRequestId() != input.getRequestId() || !output.hasFlattenOutput()
                                    || flat.getFinishedCount() != 1 || flat.getAuxInfoCount() != 1 || !flat.hasOutputIds()
                                    || flat.getOutputIds().getDataType() != TensorPB.DataType.INT32
                                    || (!emptyFinal && flat.getOutputIds().getInt32Data().isEmpty()) || flat.getOutputIds().getInt32Data().size() % 4 != 0
                                    || flat.getOutputIds().getShapeCount() != 3 || flat.getOutputIds().getShape(0) != 1 || flat.getOutputIds().getShape(1) != 1
                                    || flat.getOutputIds().getShape(2) != width || (!emptyFinal && width != length - outputTokens)
                                    || (!emptyFinal && length <= outputTokens) || length > input.getGenerateConfig().getMaxNewTokens()) {
                                finish(Status.DATA_LOSS.withDescription("Invalid generation output stream"), 0, Outcome.error); return;
                            }
                            // Native Engine may finish at its context limit, even with ignore_eos or a larger minimum.
                            outputTokens = flat.getAuxInfo(0).getOutputLen(); finalFrame = flat.getFinished(0);
                            if (width > 0) {
                                lastTokenNanos = System.nanoTime() - started;
                                if (ttft == null) { ttft = lastTokenNanos; emit("FIRST_TOKEN"); firstToken.complete(ttft); }
                            }
                            if (!terminal) stream.request(1);
                        }
                    }
                    @Override public void onError(Throwable error) { fail(error); }
                    @Override public void onCompleted() {
                        synchronized (Call.this) {
                            finish(finalFrame ? Status.OK : Status.DATA_LOSS.withDescription("Stream ended without final output"), 0,
                                finalFrame ? Outcome.success : Outcome.error);
                        }
                    }
                };
                if (route.getEnqueuedByMaster()) stub.fetchResponse(FetchRequestPB.newBuilder().setRequestId(input.getRequestId()).build(), responses);
                else {
                    var routed = input.toBuilder(); var config = routed.getGenerateConfigBuilder().clearRoleAddrs();
                    for (var server : route.getServerStatusList()) {
                        var role = switch (server.getRole()) {
                            case "PREFILL" -> RoleAddrPB.RoleType.PREFILL;
                            case "DECODE" -> RoleAddrPB.RoleType.DECODE;
                            case "VIT" -> RoleAddrPB.RoleType.VIT;
                            default -> throw Status.UNIMPLEMENTED.withDescription("Unsupported native route role: " + server.getRole()).asRuntimeException();
                        };
                        address(server);
                        config.addRoleAddrs(RoleAddrPB.newBuilder().setRole(role).setIp(server.getServerIp()).setGrpcPort(server.getGrpcPort()));
                    }
                    stub.generateStreamCall(routed.build(), responses);
                }
            } catch (RuntimeException error) { fail(error); }
        }
    }
}
