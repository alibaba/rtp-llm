package org.flexlb.mock.engine;

import java.util.List;
import java.util.Objects;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.cache.CacheSequence;
import org.flexlb.mock.compute.ForwardBatch;
import org.flexlb.mock.compute.SimulatedCompute;

/**
 * Called under the EngineLoop lock. Queues, reservations and result commits belong to the implementation.
 */
public interface Scheduler extends AutoCloseable {
    /**
     * Create one fresh policy per Engine, sharing its domain lock, cache and terminal log. Internal state transitions remain backend-owned.
     */
    @FunctionalInterface
    interface Factory {
        Scheduler create(Object lock, KvCache cache, TaskHistory finished);
    }

    record Input(String identity, long requestId, long groupId, int tokens, int maxNewTokens, int priority,
                 long started, long deadline, CacheSequence cacheSequence, boolean reuse, boolean retained, int minNewTokens, boolean ignoreEos, boolean streaming) {
        public Input {
            if (identity == null || identity.isBlank() || tokens < 1 || maxNewTokens < 1 || minNewTokens < 0 || minNewTokens > maxNewTokens)
                throw new IllegalArgumentException("Invalid scheduling input");
            Objects.requireNonNull(cacheSequence);
            if (cacheSequence.tokens().size() != tokens) throw new IllegalArgumentException("Cache identity length differs from input");
        }
        public List<Long> keys() { return cacheSequence.keys(); }
    }

    /**
     * Read-only handle. Late callbacks must keep this identity, not look up a reusable request ID.
     */
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
                long endTimeMillis, RequestFailure failure, boolean prefixLengthValid) {}
    record Status(List<Task> tasks, List<Task> finished, long finishedVersion, int waiting, int running) {
        public Status { tasks = List.copyOf(tasks); finished = List.copyOf(finished); }
    }

    /**
     * Owner is an opaque execution object, never a request ID or a mutable queue.
     */
    record Work(Object owner, ForwardBatch batch) {
        public Work { Objects.requireNonNull(owner); Objects.requireNonNull(batch); }
    }

    record InputItem(Input input, Listener listener) {
        public InputItem { Objects.requireNonNull(input); Objects.requireNonNull(listener); }
    }
    record InputResult(Request request, RequestFailure failure) {}

    Request input(Input input, Listener listener);
    /**
     * One complete explicit batch; accepted members retain this boundary until policy fallback.
     */
    default List<InputResult> input(List<InputItem> batch) {
        throw new RequestFailure(RequestFailure.Reason.INVALID, "Explicit groups are not supported by this scheduler");
    }
    void control(Request request, Control event);
    void transferCompleted(Request request, TransferResult result);
    void finish(Request request, RequestFailure failure);
    RequestState.Use beginTransfer(Request request);
    /**
     * Current execution facts; no protocol lifecycle or finished cursor. Called under the EngineLoop lock.
     */
    List<Task> tasks();
    long nextDeadline();
    List<Request> requests();
    QueueSnapshot queues();
    /**
     * Also runs while a Forward is pending: deadlines and I/O admission must not depend on compute completion.
     */
    void maintain();
    Work schedule();
    void computeCompleted(Work work, SimulatedCompute.Outcome outcome);
    void failed(Work work, RuntimeException error);
    boolean cancelled(Work work);

    /**
     * Bounded, per-request ordered callbacks. EngineLoop invokes these outside the scheduling lock.
     */
    List<Runnable> takeNotifications();
    @Override void close();
}
