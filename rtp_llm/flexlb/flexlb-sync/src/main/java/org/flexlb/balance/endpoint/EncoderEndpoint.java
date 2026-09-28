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

    private enum RequestObservation { WAITING_FOR_WORKER, SEEN_IN_WORKER }

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
    public boolean trackSelectedRequest(String requestId) {
        if (isGenerationRetiringOrRetired()) { return false; }
        AtomicBoolean added = new AtomicBoolean();
        requestObservations.compute(requestId, (id, current) -> {
            if (current != null) { return current; }
            pendingRequestCount.incrementAndGet();
            added.set(true);
            return RequestObservation.WAITING_FOR_WORKER;
        });
        return added.get();
    }

    /**
     * Remove a request after its lifecycle ends, including local pending load.
     */
    public void forgetRequest(String requestId) {
        requestObservations.computeIfPresent(requestId, (id, current) -> {
            if (current == RequestObservation.WAITING_FOR_WORKER) {
                pendingRequestCount.decrementAndGet();
            }
            return null;
        });
    }

    /**
     * Count selected requests still awaiting their first WorkerStatus task.
     */
    public int pendingEncoderRequestCount() {
        return pendingRequestCount.get();
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
            if (current == RequestObservation.WAITING_FOR_WORKER) {
                pendingRequestCount.decrementAndGet();
            }
            return RequestObservation.SEEN_IN_WORKER;
        }) != null;
    }

    @Override
    protected void closeEndpoint() {
        List<String> requestIds = List.copyOf(requestObservations.keySet());
        for (String requestId : requestIds) {
            forgetRequest(requestId);
        }
        endpointEvents.onEncoderGenerationRetired(this, requestIds);
    }
}
