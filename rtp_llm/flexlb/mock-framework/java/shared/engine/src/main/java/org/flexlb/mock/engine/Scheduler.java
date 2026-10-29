package org.flexlb.mock.engine;

import java.util.List;
import java.util.Objects;
import org.flexlb.mock.cache.DeviceCache;
import org.flexlb.mock.compute.ForwardBatch;
import org.flexlb.mock.compute.SimulatedCompute;

/** Called under the EngineLoop lock. Queues, reservations and result commits belong to the implementation. */
public interface Scheduler extends AutoCloseable {
    /** Create one fresh policy per Engine, sharing its exact domain lock and instance-owned cache. */
    @FunctionalInterface
    interface Factory {
        Scheduler create(Object lock, DeviceCache cache);
    }

    record Input(String identity, long requestId, long groupId, int tokens, int maxNewTokens, int priority,
                 long started, long deadline, List<Long> keys, boolean reuse, boolean retained, int minNewTokens, boolean ignoreEos) {
        public Input {
            if (identity == null || identity.isBlank() || tokens < 1 || maxNewTokens < 1 || minNewTokens < 0 || minNewTokens > maxNewTokens)
                throw new IllegalArgumentException("Invalid scheduling input");
            keys = List.copyOf(keys);
        }
    }

    /** Read-only handle. Late callbacks must keep this identity, not look up a reusable request ID. */
    interface Request {
        Input input();
        RequestState.Snapshot snapshot();
        int reused();
    }

    record Output(int generated, int iterations, int committed, boolean finished) {}

    interface Listener {
        default boolean writable() { return true; }
        default boolean cancelled() { return false; }
        default void allocated(Request request) {}
        void output(Request request, Output output);
        void terminal(Request request, RequestFailure failure);
    }

    interface Control {}
    interface TransferResult { RequestFailure failure(); }
    enum ReportedPhase { PENDING, RECEIVED, KV_ALLOCATED, RUNNING }
    record Task(long requestId, long batchId, int inputLength, int prefixLength, int priority,
                ReportedPhase phase, int iterations, long waitingMillis, long executionMillis,
                long endTimeMillis, long errorCode, String errorMessage) {}
    record Status(List<Task> tasks, List<Task> finished, long finishedVersion, int waiting, int running) {
        public Status { tasks = List.copyOf(tasks); finished = List.copyOf(finished); }
    }

    /** Owner is an opaque execution object, never a request ID or a mutable queue. */
    record Work(Object owner, ForwardBatch batch) {
        public Work { Objects.requireNonNull(owner); Objects.requireNonNull(batch); }
    }

    Request input(Input input, Listener listener);
    void control(Request request, Control event);
    void transferCompleted(Request request, TransferResult result);
    void finish(Request request, RequestFailure failure);
    RequestState.Use beginTransfer(Request request);
    Status status(long latestFinishedVersion);
    long nextDeadline();
    List<Request> requests();
    QueueSnapshot queues();
    String scrapeMetrics();

    /** Also runs while a Forward is pending: deadlines and I/O admission must not depend on compute completion. */
    void maintain();
    Work schedule();
    void computeCompleted(Work work, SimulatedCompute.Outcome outcome);
    void failed(Work work, RuntimeException error);
    boolean cancelled(Work work);

    /** Bounded, per-request ordered callbacks. EngineLoop invokes these outside the scheduling lock. */
    List<Runnable> takeNotifications();
    @Override void close();
}
