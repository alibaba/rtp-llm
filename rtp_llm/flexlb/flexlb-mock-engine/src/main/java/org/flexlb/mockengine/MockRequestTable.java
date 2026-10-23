package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Supplier;

/** Per-engine request metadata. Output consumption, diagnostics and deadline
 * ownership have separate lifetimes but share one request identity. KV leases
 * and WorkerStatus completion delivery remain owned by their existing protocols.
 */
final class MockRequestTable {
    private static final class Request {
        volatile LinkedBlockingQueue<GenerateOutputsPB> output;
        volatile MockRequestDeadline deadline;
        volatile String state;

        boolean empty() { return output == null && deadline == null && state == null; }
    }

    private final ConcurrentHashMap<Long, Request> requests = new ConcurrentHashMap<>();
    private final AtomicInteger outputs = new AtomicInteger();

    private void update(long id, Consumer<Request> change) {
        requests.compute(id, (key, current) -> {
            var request = current == null ? new Request() : current;
            boolean hadOutput = request.output != null;
            change.accept(request);
            if (hadOutput != (request.output != null)) outputs.addAndGet(hadOutput ? -1 : 1);
            return request.empty() ? null : request;
        });
    }

    LinkedBlockingQueue<GenerateOutputsPB> output(long id) {
        var request = requests.get(id);
        return request == null ? null : request.output;
    }

    void output(long id, LinkedBlockingQueue<GenerateOutputsPB> queue) {
        update(id, request -> request.output = queue);
    }

    LinkedBlockingQueue<GenerateOutputsPB> ensureOutput(
            long id, Supplier<LinkedBlockingQueue<GenerateOutputsPB>> factory) {
        var result = new AtomicReference<LinkedBlockingQueue<GenerateOutputsPB>>();
        update(id, request -> {
            if (request.output == null) request.output = factory.get();
            result.set(request.output);
        });
        return result.get();
    }

    void removeOutput(long id) { update(id, request -> request.output = null); }

    void removeOutput(long id, LinkedBlockingQueue<GenerateOutputsPB> expected) {
        update(id, request -> {
            if (request.output == expected) request.output = null;
        });
    }

    String state(long id) {
        var request = requests.get(id);
        return request == null ? null : request.state;
    }

    void state(long id, String state) { update(id, request -> request.state = state); }

    Map<Long, String> states() {
        Map<Long, String> result = new LinkedHashMap<>();
        requests.forEach((id, request) -> {
            var state = request.state;
            if (state != null) result.put(id, state);
        });
        return result;
    }

    MockRequestDeadline deadline(long id) {
        var request = requests.get(id);
        return request == null ? null : request.deadline;
    }

    MockRequestDeadline inheritDeadline(long id, MockRequestDeadline candidate) {
        var previous = new AtomicReference<MockRequestDeadline>();
        update(id, request -> {
            previous.set(request.deadline);
            if (request.deadline == null) request.deadline = candidate;
        });
        return previous.get();
    }

    void removeDeadline(long id, MockRequestDeadline expected) {
        update(id, request -> {
            if (request.deadline == expected) request.deadline = null;
        });
    }

    void finish(long id) {
        var previous = new AtomicReference<MockRequestDeadline>();
        update(id, request -> {
            previous.set(request.deadline);
            request.deadline = null;
        });
        if (previous.get() != null) previous.get().close();
    }

    void removeOutputsIf(BiPredicate<Long, LinkedBlockingQueue<GenerateOutputsPB>> predicate) {
        requests.forEach((id, request) -> {
            var queue = request.output;
            // Predicates may acquire engine locks. Never invoke them while a
            // table update holds the ConcurrentHashMap bin lock.
            if (queue != null && predicate.test(id, queue)) removeOutput(id, queue);
        });
    }

    void removeStatesIf(BiPredicate<Long, String> predicate) {
        requests.forEach((id, request) -> {
            var state = request.state;
            if (state != null && predicate.test(id, state)) update(id, current -> {
                if (state.equals(current.state)) current.state = null;
            });
        });
    }

    void clearOutputs() { requests.keySet().forEach(this::removeOutput); }
    void clearStates() { requests.keySet().forEach(id -> state(id, null)); }
    void closeDeadlines() { requests.keySet().forEach(this::finish); }

    java.util.Set<Long> deadlineIds() {
        var ids = ConcurrentHashMap.<Long>newKeySet();
        requests.forEach((id, request) -> { if (request.deadline != null) ids.add(id); });
        return ids;
    }

    int outputCount() { return outputs.get(); }
    int size() { return requests.size(); }
}
