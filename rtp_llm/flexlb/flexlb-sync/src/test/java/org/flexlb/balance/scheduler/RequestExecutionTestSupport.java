package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.OrderedPrefillRequestQueue;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.QueuedRequestScheduler.PrefillQueueScheduler;
import org.flexlb.config.FlexlbConfig;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.locks.ReentrantLock;

/**
 * Builds registered routes and controlled queue dependencies while leaving resource mutation to endpoint APIs.
 */
public final class RequestExecutionTestSupport {
    private static final java.util.Map<PrefillQueueScheduler, Controller> controllers =
            java.util.Collections.synchronizedMap(new java.util.IdentityHashMap<>());

    private RequestExecutionTestSupport() {}

    /**
     * Queue and ledger unit tests supply registration separately from their mocked route input.
     * An existing Decode registration may remain as an immutable collaborator; Prefill is bound
     * when the queue fixture registers its work. Selection integration tests use real executions.
     */
    public static RouteResult mockProtocolRoute(RouteResult metadata) {
        org.junit.jupiter.api.Assertions.assertNull(
                metadata.prefillReservation(),
                "mock route input must not already own Prefill registration");
        return org.mockito.Mockito.mock(
                RouteResult.class,
                org.mockito.Mockito.withSettings()
                        .useConstructor(metadata.ctx(), metadata.roles())
                        .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));
    }

    /**
     * Register Prefill before constructing a complete execution for delivery integration tests.
     */
    public static RouteResult registerPrefillRoute(
            RequestContext context,
            PrefillEndpoint prefill,
            org.flexlb.balance.endpoint.DecodeEndpoint decode,
            org.flexlb.balance.endpoint.DecodeResources.RequestReservation decodeRegistration,
            long workMs,
            long enqueuedAtMs) {
        if (context.data().getSchedulingMetadata() == null)
            context.data()
                    .setSchedulingMetadata(
                            org.flexlb.dao.SchedulingMetadata.explicit(
                                    context.data().getRequest().getRawPriority(),
                                    context.data()
                                            .getConfig()
                                            .resolveExpiresAtMs(context.data().getStartTime())));
        context.data().initializeExecutionOrder(enqueuedAtMs);
        var input =
                new org.flexlb.balance.planner.GroupPlanner.Item(
                        context.data().getRequestId(),
                        context.data().getPriority(),
                        context.data().getWorkerEnqueueSequence(),
                        context.data().getFirstWorkerEnqueueTime(),
                        context.data().getRequestExpiresAtMs(),
                        context.data().getRequest().getHardKvTokens(),
                        0L);
        var ledger = org.flexlb.balance.endpoint.EndpointTestSupport.resources(prefill);
        var registered =
                ledger.tryRegisterSelection(
                        input,
                        ledger.captureRouteProjectionInputs(),
                        workMs,
                        context.data().getConfig().isQueue());
        org.junit.jupiter.api.Assertions.assertEquals(
                org.flexlb.balance.PlacementResult.Registration.REGISTERED, registered.status());
        var roles = new java.util.ArrayList<RoleSelection>();
        try {
            roles.add(
                    new RoleSelection.Prefill(
                            prefill,
                            metadata(
                                    prefill,
                                    context.data().getRequestId(),
                                    org.flexlb.dao.route.RoleType.PREFILL),
                            workMs,
                            registered.record()));
            if (decode != null)
                roles.add(
                        new RoleSelection.Decode(
                                decode,
                                metadata(
                                        decode,
                                        context.data().getRequestId(),
                                        org.flexlb.dao.route.RoleType.DECODE),
                                decodeRegistration));
            return new RouteResult(context, roles);
        } catch (RuntimeException | Error failure) {
            org.flexlb.util.Failures.append(
                    failure,
                    org.flexlb.util.Failures.run(null, () -> ledger.rollback(registered.record())));
            org.flexlb.util.Failures.append(
                    failure,
                    org.flexlb.util.Failures.run(
                            null,
                            () ->
                                    RoleSelection.Decode.release(
                                            decode,
                                            decodeRegistration,
                                            org.flexlb.balance.endpoint.DecodeResources
                                                    .ReleaseReason.LOCAL_ROLLBACK)));
            throw failure;
        }
    }

    public static PrefillQueueScheduler create(
            String key,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DeliveryDispatcher delivery,
            AbstractRequestScheduler scheduler) {
        return create(
                key,
                endpoint,
                config,
                delivery,
                scheduler,
                org.flexlb.balance.planner.BuiltinGroupPlanner.from(config));
    }

    public static PrefillQueueScheduler create(
            String key,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DeliveryDispatcher delivery,
            AbstractRequestScheduler scheduler,
            org.flexlb.balance.planner.GroupPlanner grouping) {
        return create(
                key,
                endpoint,
                config,
                delivery,
                scheduler,
                grouping,
                RequestExecutionTestSupport::schedule);
    }

    public static PrefillQueueScheduler create(
            String key,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DeliveryDispatcher delivery,
            AbstractRequestScheduler scheduler,
            org.flexlb.balance.planner.GroupPlanner grouping,
            java.util.function.Function<PrefillQueueScheduler, PrefillQueueScheduler.Decision>
                    scheduling) {
        var dependencies = SchedulerTestSupport.queueDependencies(scheduler, config);
        var index =
                OrderedPrefillRequestQueue.ordered(
                        16, QueueExecutionSettings.queueOrder(config.isPriorityOrdering()));
        var state =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        new ReentrantLock(), index, System::currentTimeMillis);
        org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(endpoint, state);
        ReflectionTestUtils.setField(delivery, "execution", dependencies);
        var worker =
                org.mockito.Mockito.mock(
                        PrefillQueueScheduler.class,
                        org.mockito.Mockito.withSettings()
                                .useConstructor(
                                        key,
                                        endpoint,
                                        QueueExecutionSettings.capture(config, grouping),
                                        delivery)
                                .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));
        org.flexlb.balance.endpoint.EndpointTestSupport.attachWork(endpoint, worker);
        org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(endpoint, state);
        DeliveryTestSupport.installQueueFixture(worker, delivery);
        attachDriver(worker, scheduler, scheduling);
        return worker;
    }

    public static void attachDriver(
            PrefillQueueScheduler worker,
            AbstractRequestScheduler owner,
            java.util.function.Function<PrefillQueueScheduler, PrefillQueueScheduler.Decision>
                    scheduling) {
        var controller = new Controller(worker, scheduling, owner);
        controllers.put(worker, controller);
        var availability =
                (PlacementAvailability)
                        ReflectionTestUtils.getField(
                                worker.prefillEndpoint(), "placementAvailability");
        if (availability != null)
            availability.addListener(
                    new PlacementAvailability.Listener() {
                        public void onAvailabilityChanged(PlacementKey key) {
                            controller.wake();
                        }

                        public void onEndpointChanged(
                                org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {
                            controller.wake();
                        }
                    });

        org.mockito.Mockito.doAnswer(
                        call -> {
                            var result = call.callRealMethod();
                            controller.wake();
                            return result;
                        })
                .when(worker)
                .signal(org.mockito.ArgumentMatchers.anyBoolean());
        org.mockito.Mockito.doAnswer(
                        call -> {
                            controller.stop();
                            controller.awaitStopped(Long.MAX_VALUE);
                            controllers.remove(worker);
                            return call.callRealMethod();
                        })
                .when(worker)
                .stopAndAwait();
    }

    public static void start(PrefillQueueScheduler worker) {
        var controller = controllers.get(worker);
        if (controller == null) return; // The real scheduler, or the test caller, owns timing.
        org.junit.jupiter.api.Assertions.assertFalse(controller.active || worker.stopping());
        controller.active = true;
    }

    public static boolean offer(PrefillQueueScheduler worker, RouteResult route) {
        if (worker.stopping() || !worker.prefillEndpoint().acceptsNewRequests()) return false;
        var controller = controllers.get(worker);
        if (!SelectionTestSupport.registerPrefill(
                        org.flexlb.balance.endpoint.EndpointTestSupport.resources(
                                worker.prefillEndpoint()),
                        route,
                        route.prefillWorkMs(),
                        true)
                .accepted()) return false;
        if (controller == null) return worker.prefillEndpoint().tryEnqueueRoute(route);
        if (route.ctx().data().getRequestScheduler() == null) {
            if (controller.owner == null)
                SchedulerTestSupport.bindEndpointOwner(worker.prefillEndpoint(), route);
            else SchedulerTestSupport.bindOwner(route.ctx(), controller.owner);
        }
        boolean accepted = controller.active && worker.prefillEndpoint().tryEnqueueRoute(route);
        controller.wake();
        return accepted;
    }

    public static void endpointChanged(PrefillEndpoint endpoint, PrefillEndpoint.Signal kind) {
        var work = org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(endpoint);
        if (work == null) return;
        if (kind != PrefillEndpoint.Signal.PLACEMENT)
            work.signal(kind == PrefillEndpoint.Signal.CAPACITY);
        var controller = controllers.get(work);
        if (controller != null) controller.wake();
    }

    public static PrefillQueueScheduler.Decision schedule(PrefillQueueScheduler work) {
        return QueuedRequestScheduler.decideQueueScheduling(work);
    }

    public static PrefillEndpoint.QueueSnapshot capture(PrefillQueueScheduler runtime) {
        return state(runtime).captureQueue(Integer.MAX_VALUE);
    }

    public static PrefillEndpoint state(PrefillQueueScheduler runtime) {
        return org.flexlb.balance.endpoint.EndpointTestSupport.resources(runtime.prefillEndpoint());
    }

    static CapacityBoundary boundaryOnly(RouteResult item, CapacityBoundary boundary) {
        return boundary;
    }

    static QueuedRequestScheduler.PrefillQueueScheduler schedulingState(
            PrefillQueueScheduler execution) {
        var controller = controllers.get(execution);
        return controller == null ? null : controller.work;
    }

    private static org.flexlb.dao.loadbalance.ServerStatus metadata(
            org.flexlb.balance.endpoint.WorkerEndpoint endpoint,
            String requestId,
            org.flexlb.dao.route.RoleType role) {
        var status = new org.flexlb.dao.loadbalance.ServerStatus();
        status.setSuccess(true);
        status.setRequestId(requestId);
        status.setRole(role);
        status.setServerIp(endpoint.getIp());
        status.setHttpPort(endpoint.getHttpPort());
        status.setGrpcPort(endpoint.getGrpcPort());
        return status;
    }

    private static final class Controller implements RequestScheduler {
        final PrefillQueueScheduler execution;
        final AbstractRequestScheduler owner;
        final QueuedRequestScheduler.PrefillQueueScheduler work;
        final java.util.function.Function<PrefillQueueScheduler, PrefillQueueScheduler.Decision>
                policy;
        final Thread thread;

        volatile boolean stopped;

        final java.util.concurrent.atomic.AtomicLong notifications =
                new java.util.concurrent.atomic.AtomicLong();

        volatile boolean active;

        final java.util.concurrent.atomic.AtomicBoolean started =
                new java.util.concurrent.atomic.AtomicBoolean();

        PrefillQueueScheduler.Decision waiting;

        long next = Long.MAX_VALUE;

        Controller(
                PrefillQueueScheduler execution,
                java.util.function.Function<PrefillQueueScheduler, PrefillQueueScheduler.Decision>
                        policy,
                AbstractRequestScheduler owner) {
            this.owner = owner;
            this.execution = execution;
            this.policy = policy;
            this.work = execution;
            this.thread = Thread.ofVirtual().unstarted(this::run);
        }

        public void start() {
            thread.start();
        }

        public void schedule() {
            next = scheduleAt(System.nanoTime());
        }

        public void onRequest(RequestContext request) {}

        public void onEndpointChanged(org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {
            wake();
        }

        public void close() {
            stop();
            awaitStopped(Long.MAX_VALUE);
        }

        public void stop() {
            stopped = true;
            wake();
        }

        public boolean awaitStopped(long deadline) {
            if (Thread.currentThread() == thread) return false;
            try {
                return deadline == Long.MAX_VALUE
                        ? join()
                        : thread.join(
                                java.time.Duration.ofNanos(
                                        Math.max(0L, deadline - System.nanoTime())));
            } catch (InterruptedException failure) {
                Thread.currentThread().interrupt();
                throw new AssertionError(failure);
            }
        }

        private void run() {
            while (!stopped) {
                long observed = notifications.get();
                schedule();
                if (stopped || notifications.get() != observed) continue;
                if (next == Long.MAX_VALUE) java.util.concurrent.locks.LockSupport.park(this);
                else
                    java.util.concurrent.locks.LockSupport.parkNanos(
                            this, Math.max(1L, next - System.nanoTime()));
            }
        }

        private void wake() {
            if (active && started.compareAndSet(false, true)) start();
            notifications.incrementAndGet();
            java.util.concurrent.locks.LockSupport.unpark(thread);
        }

        private boolean join() throws InterruptedException {
            thread.join();
            return true;
        }

        private long scheduleAt(long now) {
            if (execution.stopping()) return Long.MAX_VALUE;
            long ms = System.currentTimeMillis();
            try {
                if (waiting == null
                        ? execution.waitCurrent(null, ms)
                        : now < next && execution.waitCurrent(waiting, ms)) return next;
                execution.observeWait(null);
                waiting = policy.apply(execution);
                if (waiting.waitReason() != null)
                    execution.recordQueueWait(waiting.request(), waiting.waitReason());
                execution.observeWait(waiting);
                if (!execution.waitCurrent(waiting, ms)) return next = now;
                long at =
                        waiting.request() == null
                                ? Long.MAX_VALUE
                                : waiting.unavailable() == null
                                        ? waiting.wakeAtMs()
                                        : waiting.request().expiresAtMs();
                return next =
                        at == Long.MAX_VALUE
                                ? at
                                : now
                                        + java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(
                                                Math.max(0L, at - ms));
            } catch (Throwable failure) {
                execution.stopAndDrain(failure);
                stop();
                return Long.MAX_VALUE;
            }
        }
    }
}
