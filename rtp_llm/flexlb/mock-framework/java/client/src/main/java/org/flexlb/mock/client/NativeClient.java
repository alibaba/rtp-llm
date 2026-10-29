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
import org.flexlb.mock.client.evidence.ClientEvidence.Event;

/** One native Schedule -> Fetch/Generate exchange. Frontend-style primary/backup scheduling; no Engine fallback. */
public final class NativeClient implements AutoCloseable {
    public enum Outcome { success, error, cancelled, timeout }
    public record Result(long requestId, Instant sentAt, Long ttftNanos, long elapsedNanos,
                         int outputTokens, Outcome outcome, String grpcCode, long errorCode, String message) {}
    private final java.util.function.Supplier<List<String>> masters;
    private final java.util.function.Consumer<Event.Builder> events;
    private final Map<String, ManagedChannel> channels = new HashMap<>();
    private final Map<Long, Call> active = new HashMap<>();
    private final ScheduledThreadPoolExecutor timers = new ScheduledThreadPoolExecutor(1,
        Thread.ofPlatform().daemon().name("mock-client-deadline").factory());
    private final ExecutorService cleanup = Executors.newVirtualThreadPerTaskExecutor();
    private boolean closed, closing;

    public NativeClient(String master) {
        this(() -> List.of(master));
        if (master == null || master.isBlank()) throw new IllegalArgumentException("Master gRPC address required");
    }
    public NativeClient(java.util.function.Supplier<List<String>> masters) {
        this(masters, event -> {});
    }
    NativeClient(java.util.function.Supplier<List<String>> masters, java.util.function.Consumer<Event.Builder> events) {
        this.masters = Objects.requireNonNull(masters); this.events = Objects.requireNonNull(events);
        timers.setRemoveOnCancelPolicy(true);
    }
    private synchronized ManagedChannel channel(String target) {
        if (closed) throw new IllegalStateException("Client closed");
        return channels.computeIfAbsent(target, t -> ManagedChannelBuilder.forTarget(t).usePlaintext().build());
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
                || input.getMultimodalInputsCount() != 0 || input.hasV41Inputs() || input.hasMultimodalTokenLayout()
                || input.getGenerateConfig().getNumBeams() > 1 || input.getGenerateConfig().getNumReturnSequences() > 1)
            throw new IllegalArgumentException("Client currently requires matching single-sequence text request metadata");
        if (input.getGenerateConfig().getTimeoutMs() > 0) budget = Math.min(budget, TimeUnit.MILLISECONDS.toNanos(input.getGenerateConfig().getTimeoutMs()));
        Call call;
        synchronized (this) {
            if (closed || closing) throw new IllegalStateException("Client closed");
            if (active.containsKey(request.getRequestId())) throw new IllegalArgumentException("Active request_id already exists");
            call = new Call(request, input, budget);
            events.accept(call.event(Event.Kind.ISSUED).setEpochMs(call.sentAt.toEpochMilli()));
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

    public final class Call {
        private final FlexlbScheduleRequestPB request;
        private final GenerateInputPB input;
        private final List<String> routes;
        private String scheduledMaster;
        private final Context.CancellableContext context;
        private final CompletableFuture<Result> result = new CompletableFuture<>();
        private final CompletableFuture<Long> firstToken = new CompletableFuture<>();
        private final Instant sentAt = Instant.now();
        private final long started = System.nanoTime();
        private Long ttft;
        private long batchId;
        private int outputTokens, scheduleAttempt;
        private boolean pendingSchedule;
        private boolean terminal, finalFrame, scheduleStarted, routeRejected;
        private Call(FlexlbScheduleRequestPB request, GenerateInputPB input, long budget) {
            this.request = request; this.input = input;
            routes = List.copyOf(masters.get());
            if (routes.size() > 2 || new HashSet<>(routes).size() != routes.size() || routes.stream().anyMatch(String::isBlank))
                throw new IllegalArgumentException("Expected distinct primary and optional backup Master");
            context = Context.ROOT.withDeadlineAfter(budget, TimeUnit.NANOSECONDS, timers);
        }
        public CompletionStage<Result> completion() { return result.minimalCompletionStage(); }
        public CompletionStage<Long> firstToken() { return firstToken.minimalCompletionStage(); }
        public void cancel() { finish(Status.CANCELLED.withDescription("Client cancelled request"), 0, Outcome.cancelled); }

        private Event.Builder event(Event.Kind kind) {
            return Event.newBuilder().setRequestId(request.getRequestId()).setEpochMs(System.currentTimeMillis())
                .setElapsedNanos(System.nanoTime() - started).setKind(kind);
        }
        private void emit(Event.Kind kind) { events.accept(event(kind)); }
        private void scheduleResult(Status status, long code, Boolean responseSuccess) {
            if (!pendingSchedule) return;
            pendingSchedule = false;
            var result = event(Event.Kind.SCHEDULE_RESULT).setAttempt(scheduleAttempt).setMaster(scheduledMaster)
                .setGrpcCode(status.getCode().name()).setErrorCode(code).setMessage(message(status));
            if (responseSuccess != null) result.setResponseSuccess(responseSuccess);
            events.accept(result);
        }
        private void schedule(int attempt) {
            if (routes.isEmpty()) { finish(Status.UNAVAILABLE.withDescription("No healthy Master"), 0, Outcome.error); return; }
            try {
                scheduledMaster = routes.get(attempt);
                scheduleStarted = true; scheduleAttempt = attempt + 1; pendingSchedule = true;
                events.accept(event(Event.Kind.SCHEDULE_START).setAttempt(scheduleAttempt).setMaster(scheduledMaster));
                FlexlbServiceGrpc.newStub(channel(scheduledMaster)).schedule(request, new StreamObserver<>() {
                    private boolean received;
                    @Override public void onNext(FlexlbScheduleResponsePB route) {
                        synchronized (Call.this) {
                            if (terminal) return;
                            if (received) { finish(Status.DATA_LOSS.withDescription("Multiple Schedule replies"), 0, Outcome.error); return; }
                            received = true;
                            scheduleResult(Status.OK.withDescription(route.getErrorMessage()), route.getCode(), route.getSuccess());
                            batchId = route.hasLifecycle() ? route.getLifecycle().getBatchId() : 0;
                            if (!route.getSuccess() || route.getCode() != 200) {
                                routeRejected = !route.hasLifecycle();
                                finish(Status.FAILED_PRECONDITION.withDescription(route.getErrorMessage()), route.getCode(), Outcome.error); return;
                            }
                            context.run(() -> consume(route));
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
                            if (finalFrame || output.getRequestId() != input.getRequestId() || !output.hasFlattenOutput()
                                    || flat.getFinishedCount() != 1 || flat.getAuxInfoCount() != 1 || !flat.hasOutputIds()
                                    || flat.getOutputIds().getDataType() != TensorPB.DataType.INT32
                                    || flat.getOutputIds().getInt32Data().isEmpty() || flat.getOutputIds().getInt32Data().size() % 4 != 0
                                    || flat.getOutputIds().getShapeCount() != 3 || flat.getOutputIds().getShape(0) != 1 || flat.getOutputIds().getShape(1) != 1
                                    || flat.getOutputIds().getShape(2) != width || width != (input.getGenerateConfig().getReturnIncremental() ? length - outputTokens : length)
                                    || length <= outputTokens || length > input.getGenerateConfig().getMaxNewTokens()) {
                                finish(Status.DATA_LOSS.withDescription("Invalid text output stream"), 0, Outcome.error); return;
                            }
                            outputTokens = flat.getAuxInfo(0).getOutputLen(); finalFrame = flat.getFinished(0);
                            if (ttft == null) { ttft = System.nanoTime() - started; emit(Event.Kind.FIRST_TOKEN); firstToken.complete(ttft); }
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
                            default -> throw Status.UNIMPLEMENTED.withDescription("Unsupported native route role: " + server.getRole()).asRuntimeException();
                        };
                        address(server);
                        config.addRoleAddrs(RoleAddrPB.newBuilder().setRole(role).setIp(server.getServerIp()).setGrpcPort(server.getGrpcPort()));
                    }
                    stub.generateStreamCall(routed.build(), responses);
                }
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
        private synchronized void finish(Status status, long code, Outcome outcome) {
            if (terminal) return;
            terminal = true;
            scheduleResult(status, code, null);
            if (!firstToken.isDone()) firstToken.completeExceptionally(status.asRuntimeException());
            context.cancel(null);
            synchronized (NativeClient.this) { active.remove(request.getRequestId(), this); }
            if (outcome != Outcome.success && scheduleStarted && !routeRejected) {
                // Cancel against this request's last attempted Master, not a newer discovery snapshot.
                try { cleanup.submit(() -> {
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
            var ended = event(Event.Kind.TERMINAL).setOutcome(outcome.name()).setGrpcCode(status.getCode().name())
                .setErrorCode(code).setMessage(message(status)).setOutputTokens(outputTokens);
            if (ttft != null) ended.setTtftNanos(ttft);
            events.accept(ended);
            result.complete(new Result(request.getRequestId(), sentAt, ttft, ended.getElapsedNanos(), outputTokens,
                outcome, status.getCode().name(), code, status.getDescription() == null ? "" : status.getDescription()));
        }
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
    @Override public void close() {
        List<Call> calls;
        synchronized (this) { if (closed || closing) return; closing = true; calls = List.copyOf(active.values()); }
        for (var call : calls) call.cancel();
        timers.shutdownNow(); cleanup.shutdown();
        try { cleanup.awaitTermination(2, TimeUnit.SECONDS); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        cleanup.shutdownNow();
        List<ManagedChannel> owned;
        synchronized (this) { closed = true; owned = List.copyOf(channels.values()); }
        owned.forEach(ManagedChannel::shutdownNow);
        for (var channel : owned) try { channel.awaitTermination(3, TimeUnit.SECONDS); }
        catch (InterruptedException error) { Thread.currentThread().interrupt(); break; }
    }
}
