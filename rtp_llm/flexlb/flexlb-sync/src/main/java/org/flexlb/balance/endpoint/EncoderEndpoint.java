package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.dao.master.WorkerStatus;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Encoder endpoint with local decisions that have not appeared in WorkerStatus.
 * Its task IDs also identify which status facts belong to this exact endpoint.
 */
public final class EncoderEndpoint extends WorkerEndpoint {

    private record RequestObservation(boolean seenInWorker, long estimatedUncachedTokens) { }

    private final EndpointEventProjector endpointEvents;
    private final ConcurrentMap<String, RequestObservation> requestObservations = new ConcurrentHashMap<>();
    private final AtomicInteger pendingRequestCount = new AtomicInteger();

    /**
     * Bind an Encoder worker generation to its request lifecycle projector.
     */
    public EncoderEndpoint(WorkerStatus status, EndpointEventProjector endpointEvents) {
        super(status);
        this.endpointEvents = endpointEvents;
    }

    /**
     * Track a selected request until status reports it running or finished.
     * Returns false if the worker is retiring or already tracks this request.
     */
    public boolean trackSelectedRequest(String requestId, long estimatedUncachedTokens) {
        if (isGenerationRetiringOrRetired()) { return false; }
        AtomicBoolean added = new AtomicBoolean();
        requestObservations.compute(requestId, (id, current) -> {
            if (current != null) { return current; }
            pendingRequestCount.incrementAndGet();
            added.set(true);
            return new RequestObservation(false, Math.max(0L, estimatedUncachedTokens));
        });
        return added.get();
    }

    /**
     * Remove a request after its lifecycle ends, including local pending load.
     */
    public void forgetRequest(String requestId) {
        RequestObservation removed = requestObservations.remove(requestId);
        if (removed == null) {
            return;
        }
        if (!removed.seenInWorker()) {
            pendingRequestCount.decrementAndGet();
        }
        endpointEvents.onEncoderCapacityChanged();
    }

    /**
     * Count selected requests still awaiting their first WorkerStatus task.
     */
    public int pendingEncoderRequestCount() {
        return pendingRequestCount.get();
    }

    /**
     * Estimated Encoder work: client MM token estimates before status, then
     * each active task's uncached synthesized input length from WorkerStatus.
     */
    public long inflightUncachedTokenEstimate() {
        var activeTasks = getStatus().committedEngineObservation().runningTaskList();
        long total = 0L;
        for (WorkerStatus.TaskObservation task : activeTasks.values()) {
            total = addSaturated(total, Math.max(0L, task.inputLength()));
        }
        for (var entry : requestObservations.entrySet()) {
            if (!entry.getValue().seenInWorker() && !activeTasks.containsKey(entry.getKey())) {
                total = addSaturated(total, entry.getValue().estimatedUncachedTokens());
            }
        }
        return total;
    }

    private static long addSaturated(long total, long additional) {
        return Long.MAX_VALUE - total < additional ? Long.MAX_VALUE : total + additional;
    }

    @Override
    public Runnable applyPreparedStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        WorkerStatus.StatusObservation observation = prepared.observation();
        if (!observation.alive()) { beginRetirement(); }
        ws.publishPreparedStatus(prepared);
        return projectStatus(observation, true);
    }

    @Override
    public Runnable observeStatusHeartbeat(WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        if (observation.owner() != ws) {
            throw new IllegalArgumentException("Status observation belongs to another WorkerStatus generation");
        }
        return projectStatus(observation, false);
    }

    private Runnable projectStatus(WorkerStatus.StatusObservation observation, boolean includeFinished) {
        List<WorkerStatus.TaskObservation> running = new ArrayList<>();
        List<WorkerStatus.TaskObservation> finished = new ArrayList<>();
        for (WorkerStatus.TaskObservation task : observation.runningTasks().values()) {
            if (markObserved(task.requestId())) {
                running.add(task);
            }
        }
        if (includeFinished) {
            for (WorkerStatus.TaskObservation task : observation.finishedTasks().values()) {
                if (markObserved(task.requestId())) {
                    finished.add(task);
                }
            }
        }
        return () -> endpointEvents.onEncoderStatus(this, running, finished);
    }

    private boolean markObserved(String requestId) {
        return requestObservations.computeIfPresent(requestId, (id, current) -> {
            if (!current.seenInWorker()) {
                pendingRequestCount.decrementAndGet();
            }
            return new RequestObservation(true, current.estimatedUncachedTokens());
        }) != null;
    }

    @Override
    protected void closeEndpoint() {
        List<String> requestIds = List.copyOf(requestObservations.keySet());
        for (String requestId : requestIds) {
            forgetRequest(requestId);
        }
        endpointEvents.onEncoderGenerationRetired(this, requestIds);
        endpointEvents.onEncoderCapacityChanged();
    }
}
