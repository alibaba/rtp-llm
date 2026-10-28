package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;
import org.flexlb.mock.metrics.EngineMetrics;

/** Bounded text Prefill FIFO. Owns Forward, deadline and KV through local completion or transfer. */
final class LocalPrefill implements AutoCloseable {
    static final int LIMIT = TaskHistory.LIMIT;
    record Result(GenerateOutputsPB output, Status status, long engineCode) {}
    private final RtpConfig config;
    private final DeviceCache cache;
    private final EngineMetrics metrics;
    private final SimulatedCompute compute = new SimulatedCompute();
    private final ScheduledThreadPoolExecutor deadlines = new ScheduledThreadPoolExecutor(1,
        Thread.ofPlatform().daemon().name("rtp-request-deadline").factory());
    private final ExecutorService replies = Executors.newVirtualThreadPerTaskExecutor();
    private final Map<Long, Session> active = new HashMap<>();
    private final ArrayDeque<Session> waiting = new ArrayDeque<>();
    private final TaskHistory history = new TaskHistory();
    private Session running;
    private boolean closed;

    LocalPrefill(RtpConfig config, DeviceCache cache) {
        this.config = config; this.cache = cache;
        metrics = new EngineMetrics("rtp", config.role().name(), config.id(), this::metricState);
        deadlines.setRemoveOnCancelPolicy(true);
    }

    final class Session {
        final GenerateInputPB input;
        final long started = System.nanoTime();
        final long deadline;
        final CompletableFuture<Result> result = new CompletableFuture<>();
        final List<Long> keys;
        final boolean retained;
        final CompletableFuture<Result> termination = new CompletableFuture<>();
        boolean ready;
        long computedAt;
        DeviceCache.Lease lease;
        SimulatedCompute.Execution execution;
        ScheduledFuture<?> timeout;
        long runningAt;
        int reused;
        boolean terminal;
        Session(GenerateInputPB input, long rpcDeadline, boolean retained) {
            this.retained = retained;
            this.input = input;
            long requestDeadline = input.getGenerateConfig().getTimeoutMs() > 0 ? started + TimeUnit.MILLISECONDS.toNanos(input.getGenerateConfig().getTimeoutMs()) : Long.MAX_VALUE;
            deadline = Math.min(requestDeadline, rpcDeadline);
            keys = PrefixKeys.completeBlocks(input, config.blockSize());
        }
        CompletionStage<Result> completion() { return result.minimalCompletionStage(); }
        void cancel() { finish(this, Status.CANCELLED.withDescription("Client cancelled local generation"), 0); }
        void preempt() { finish(this, Status.RESOURCE_EXHAUSTED.withDescription("Preempted by a higher-priority request"), 8429); }
        void expire() { finish(this, Status.DEADLINE_EXCEEDED.withDescription("RPC deadline expired"), 603); }
    }

    Session submit(GenerateInputPB input) { return submit(input, Long.MAX_VALUE); }

    synchronized Session submit(GenerateInputPB input, long rpcDeadline) {
        return submit(input, rpcDeadline, false);
    }

    synchronized Session submit(GenerateInputPB input, long rpcDeadline, boolean retained) {
        validate(input, retained);
        if (closed) throw Status.UNAVAILABLE.withDescription("RTP node is closing").asRuntimeException();
        if (active.containsKey(input.getRequestId())) throw Status.ALREADY_EXISTS.withDescription("Active request id already exists").asRuntimeException();
        if (active.size() >= LIMIT) throw Status.RESOURCE_EXHAUSTED.withDescription("Mock local request limit reached").asRuntimeException();
        var session = new Session(input, rpcDeadline, retained);
        active.put(input.getRequestId(), session); waiting.add(session);
        metrics.started();
        if (session.deadline != Long.MAX_VALUE)
            session.timeout = deadlines.schedule(() -> finish(session, Status.DEADLINE_EXCEEDED.withDescription("Request budget expired"), 603),
                Math.max(0, session.deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
        advance();
        return session;
    }

    void validate(GenerateInputPB input, boolean retained) {
        if (config.role() != Role.prefill || (retained ? input.getGenerateConfig().getMaxNewTokens() < 1 : input.getGenerateConfig().getMaxNewTokens() != 1))
            throw Status.UNIMPLEMENTED.withDescription("Local generation supports one token; transfer prepares only the Prefill token").asRuntimeException();
        var generation = input.getGenerateConfig();
        RtpRequestValidation.supported(input);
        if (input.getTokenIdsCount() < 1 || (long) input.getTokenIdsCount() + 1 > config.maxSeqLen()
                || config.maxBatchTokens() > 0 && input.getTokenIdsCount() > config.maxBatchTokens()
                || generation.getTimeoutMs() < 0 || generation.getMinNewTokens() < 0 || generation.getMinNewTokens() > generation.getMaxNewTokens()
                || generation.getNumBeams() < 0 || generation.getNumReturnSequences() < 0)
            throw RtpErrors.failure(Status.INVALID_ARGUMENT.withDescription("Invalid input length or local generation settings"), 605);
    }

    /** All callers hold this monitor. Allocation is delayed until FIFO selection, not enqueue. */
    private void advance() {
        while (!closed && running == null && !waiting.isEmpty()) {
            var session = waiting.removeFirst(); running = session;
            if (System.nanoTime() >= session.deadline) {
                end(session, Status.DEADLINE_EXCEEDED.withDescription("Request expired in queue"), 603); continue;
            }
            try {
                boolean reuse = config.reuseCache() && config.deviceCache() && session.input.getGenerateConfig().getReuseCache();
                int matchCount = Math.min(session.keys.size(), (session.input.getTokenIdsCount() - 1) / config.blockSize());
                var keys = reuse ? session.keys.subList(0, matchCount) : List.<Long>of();
                int blocks = (int) (((long) session.input.getTokenIdsCount() + config.blockSize() - 1) / config.blockSize());
                var admission = cache.acquire(blocks, keys);
                if (admission.lease() == null) {
                    end(session, Status.RESOURCE_EXHAUSTED.withDescription("KV allocation failed: " + admission.failure()), 602); continue;
                }
                session.lease = admission.lease(); session.reused = session.lease.reusedBlocks() * config.blockSize();
                metrics.cacheLookup(session.input.getTokenIdsCount(), session.reused);
                session.runningAt = System.nanoTime();
                var request = new ForwardBatch.Request(Long.toString(session.input.getRequestId()), session.input.getTokenIdsCount(), session.reused,
                    1, 0, session.input.getTokenIdsCount() - session.reused, session.input.getPriority(), List.of());
                var batch = new ForwardBatch("local/" + session.input.getRequestId(), List.of(new ForwardBatch.Partition(0, List.of(request))));
                var plan = config.compute().plan(Role.prefill, config.id(), batch);
                session.execution = compute.submit(plan);
                session.execution.completion().thenAccept(outcome -> {
                    if (outcome.state() == SimulatedCompute.State.COMPLETED) computed(session, outcome.elapsedNanos());
                    else finish(session, Status.CANCELLED.withDescription("Computation cancelled"), 0);
                });
            } catch (RuntimeException error) { end(session, Status.INTERNAL.withDescription("Forward failed: " + error.getMessage()), 0); }
        }
    }

    private synchronized void computed(Session session, long elapsedNanos) {
        if (session.terminal) return;
        if (System.nanoTime() < session.deadline)
            metrics.forwardCompleted(elapsedNanos, session.input.getTokenIdsCount(), session.input.getTokenIdsCount() - session.reused, 1);
        if (session.retained && System.nanoTime() < session.deadline) {
            // Data exists only after Forward completed. Keep the lease, including its partial tail block.
            session.ready = true; session.computedAt = System.nanoTime();
            if (running == session) running = null;
            var result = new Result(LocalResponse.oneToken(session.input, session.reused, session.computedAt - session.started), Status.OK, 0);
            replies.execute(() -> session.result.complete(result));
            advance();
        } else finish(session, Status.OK, 0);
    }

    synchronized void requireReady(Session session) {
        if (session.terminal || !session.ready || System.nanoTime() >= session.deadline)
            throw Status.FAILED_PRECONDITION.withDescription("Prefill transfer data is no longer owned").asRuntimeException();
    }

    synchronized void consume(Session session, Runnable deliver) {
        requireReady(session);
        deliver.run();
        finish(session, Status.OK, 0);
    }

    private synchronized void finish(Session session, Status status, long code) {
        if (session.terminal) return;
        if (status.isOk() && System.nanoTime() >= session.deadline) {
            status = Status.DEADLINE_EXCEEDED.withDescription("Request budget expired before commit"); code = 603;
        }
        end(session, status, code); advance();
    }

    private void end(Session session, Status status, long code) {
        if (session.terminal) return;
        session.terminal = true;
        if (session.timeout != null) session.timeout.cancel(false);
        if (session.execution != null && !status.isOk()) session.execution.cancel();
        if (session.lease != null) {
            try {
                if (status.isOk()) session.lease.complete(config.reuseCache() && config.deviceCache()
                    && session.input.getGenerateConfig().getReuseCache() ? session.keys : List.of());
                else session.lease.close();
            } catch (RuntimeException error) {
                status = Status.INTERNAL.withDescription("KV finalization failed: " + error.getMessage()); code = 0;
                try { session.lease.close(); } catch (RuntimeException release) { status = status.augmentDescription("Release failed: " + release); }
            }
        }
        active.remove(session.input.getRequestId(), session); waiting.remove(session);
        metrics.finished(switch (status.getCode()) {
            case OK -> EngineMetrics.Outcome.success;
            case CANCELLED -> EngineMetrics.Outcome.cancelled;
            case DEADLINE_EXCEEDED -> EngineMetrics.Outcome.timeout;
            default -> EngineMetrics.Outcome.error;
        });
        if (running == session) running = null;
        var task = task(session).setEndTimeMs(System.currentTimeMillis()).setIterateCount(status.isOk() ? 1 : 0);
        if (!status.isOk()) task.setErrorInfo(ErrorDetailsPB.newBuilder().setErrorCode(code == 0 ? status.getCode().value() : code).setErrorMessage(status.getDescription()));
        history.add(task.build());
        var output = status.isOk() ? LocalResponse.oneToken(session.input, session.reused, System.nanoTime() - session.started) : null;
        var result = new Result(output, status, code);
        replies.execute(() -> { session.result.complete(result); session.termination.complete(result); });
    }

    synchronized TaskHistory.Snapshot snapshot(long latestFinished) {
        return history.snapshot(latestFinished, active.values().stream().map(s -> task(s).build()).toList(),
                                active.size() - (running == null ? 0 : 1), running == null ? 0 : 1);
    }

    String scrapeMetrics() { return metrics.scrape(); }

    private synchronized EngineMetrics.State metricState() {
        var kv = cache.snapshot(0, false);
        long size = config.blockSize();
        return new EngineMetrics.State(active.size() - (running == null ? 0 : 1), running == null ? 0 : 1,
            kv.totalBlocks() * size, kv.availableBlocks() * size, kv.freeBlocks() * size, kv.heldBlocks() * size);
    }
    private TaskInfoPB.Builder task(Session s) {
        boolean executing = s.runningAt != 0 && !s.ready;
        long nanos = System.nanoTime();
        return TaskInfoPB.newBuilder().setRequestId(s.input.getRequestId()).setInputLength(s.input.getTokenIdsCount())
            .setBatchId(s.input.hasGroupId() ? s.input.getGroupId().getValue() : 0)
            .setPrefixLength(s.reused).setPriority(s.input.getPriority()).setDpRank(0).setIsWaiting(!executing)
            .setPhase(s.ready ? TaskPhase.TASK_PHASE_KV_ALLOCATED : executing ? TaskPhase.TASK_PHASE_RUNNING : TaskPhase.TASK_PHASE_PENDING)
            .setWaitingTimeMs(TimeUnit.NANOSECONDS.toMillis((s.runningAt != 0 ? s.runningAt : nanos) - s.started))
            .setExecutionTimeMs(s.runningAt != 0 ? TimeUnit.NANOSECONDS.toMillis((s.ready ? s.computedAt : nanos) - s.runningAt) : 0);
    }

    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        for (var session : List.copyOf(active.values())) end(session, Status.UNAVAILABLE.withDescription("RTP node closed"), 0);
        compute.close(); deadlines.shutdownNow(); replies.shutdown();
    }
}
