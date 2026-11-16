package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

/**
 * Supplies idle preemption or externally completed resource results for scheduler concurrency tests.
 */
public final class RequestPreemptionTestSupport {
    private RequestPreemptionTestSupport() {}

    public static RequestPreemption mock() {
        var result = org.mockito.Mockito.mock(RequestPreemption.class);
        org.mockito.Mockito.when(result.isIdle()).thenReturn(true);
        return result;
    }

    public static Controlled controlled() {
        return new Controlled();
    }

    public record AcquiredResource(
            DecodeEndpoint.RequestRegistration reservation, boolean failed, String detail) {}

    /**
     * Supplies a controllable resource result for a waiting request and its endpoint.
     */
    public interface ResourceAcquisition {
        CompletableFuture<AcquiredResource> acquire(
                RequestContext request, WorkerEndpoint endpoint);
    }

    /**
     * Retains pending acquisitions and exposes completed results as scheduler work.
     */
    public static class Controlled implements RequestPreemption {
        private final Map<RequestContext.SelectionAttempt, Pending> pending = new HashMap<>();

        private Runnable wakeup = () -> {};

        private boolean stopped;

        public final ResourceAcquisition acquisition =
                org.mockito.Mockito.mock(ResourceAcquisition.class);

        @Override
        public void setWakeup(Runnable wakeup) {
            this.wakeup = wakeup;
        }

        @Override
        public synchronized boolean start(
                RequestContext.SelectionAttempt selection,
                WorkerEndpoint endpoint,
                Consumer<RoleSelection> handler) {
            if (stopped || !(endpoint instanceof DecodeEndpoint) || pending.size() >= 2)
                return false;
            var result = acquisition.acquire(selection.requestContext(), endpoint);
            if (result == null) return false;
            pending.put(selection, new Pending((DecodeEndpoint) endpoint, result, handler));
            result.whenComplete((value, failure) -> wakeup.run());
            return true;
        }

        @Override
        public synchronized List<Work> readyWork() {
            var work = new ArrayList<Work>();
            for (var entry : pending.entrySet()) {
                if (!entry.getValue().result.isDone()) continue;
                var exact = entry.getKey();
                var item = entry.getValue();
                work.add(
                        new Work(
                                exact,
                                () -> {
                                    try {
                                        var result = item.result.join();
                                        RoleSelection acquired = null;
                                        if (result != null && result.reservation() != null) {
                                            var status = new ServerStatus();
                                            status.setRole(RoleType.DECODE);
                                            status.setRequestId(
                                                    exact.requestContext().getRequestId());
                                            status.setSuccess(true);
                                            acquired =
                                                    org.flexlb.balance.schedule.SelectionTestSupport
                                                            .decode(
                                                                    item.endpoint,
                                                                    status,
                                                                    result.reservation());
                                        }
                                        item.handler.accept(acquired);
                                    } finally {
                                        synchronized (this) {
                                            pending.remove(exact);
                                        }
                                        wakeup.run();
                                    }
                                }));
            }
            return work;
        }

        @Override
        public synchronized Set<RequestContext> waitingRequests() {
            var result = new HashSet<RequestContext>();
            for (var exact : pending.keySet()) result.add(exact.requestContext());
            return result;
        }

        @Override
        public void requestCheck(WorkerEndpoint endpoint) {}

        @Override
        public void requestPeriodicCheck() {}

        @Override
        public void cancel(RequestContext request) {
            wakeup.run();
        }

        @Override
        public synchronized boolean isIdle() {
            return pending.isEmpty();
        }

        @Override
        public void close() {
            stopAdmission();
        }

        @Override
        public synchronized void stopAdmission() {
            stopped = true;
        }

        private record Pending(
                DecodeEndpoint endpoint,
                CompletableFuture<AcquiredResource> result,
                Consumer<RoleSelection> handler) {}
    }
}
