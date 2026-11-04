package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillActiveIndex;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork;
import org.flexlb.config.FlexlbConfig;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.locks.ReentrantLock;

public final class RequestExecutionTestSupport {

    private RequestExecutionTestSupport() {}

    static CapacityBoundary boundaryOnly(RequestRoute item, CapacityBoundary boundary) {
        return boundary;
    }

    public static EndpointWork create(
            String key,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DeliveryDispatcher delivery,
            RequestLifecycle scheduler) {
        return create(
                key,
                endpoint,
                config,
                delivery,
                scheduler,
                org.flexlb.balance.planner.BuiltinGroupPlanner.from(config));
    }

    public static EndpointWork create(
            String key,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DeliveryDispatcher delivery,
            RequestLifecycle scheduler,
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

    public static EndpointWork create(
            String key,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DeliveryDispatcher delivery,
            RequestLifecycle scheduler,
            org.flexlb.balance.planner.GroupPlanner grouping,
            java.util.function.Function<EndpointWork, EndpointWork.Decision> scheduling) {
        var dependencies = SchedulerTestSupport.queueDependencies(scheduler, config);
        var index =
                PrefillActiveIndex.ordered(
                        16, QueueExecutionSettings.queueOrder(config.isPriorityOrdering()));
        var state =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        new ReentrantLock(), index, System::currentTimeMillis);
        org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(endpoint, state);
        ReflectionTestUtils.setField(delivery, "execution", dependencies);
        var worker =
                org.mockito.Mockito.mock(
                        EndpointWork.class,
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
            EndpointWork worker,
            RequestLifecycle owner,
            java.util.function.Function<EndpointWork, EndpointWork.Decision> scheduling) {
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

    public static void start(EndpointWork worker) {
        var controller = controllers.get(worker);
        if (controller == null) return; // The real scheduler, or the test caller, owns timing.
        org.junit.jupiter.api.Assertions.assertFalse(controller.active || worker.stopping());
        controller.active = true;
    }

    public static boolean offer(EndpointWork worker, RequestRoute route) {
        if (worker.stopping()) return false;
        var controller = controllers.get(worker);
        if (controller == null) return worker.prefillEndpoint().offerQueued(route);
        if (route.ctx().data().getLifecycle() == null) {
            if (controller.owner == null)
                SchedulerTestSupport.bindEndpointOwner(worker.prefillEndpoint(), route);
            else SchedulerTestSupport.bindOwner(route.ctx(), controller.owner);
        }
        boolean accepted = controller.active && worker.prefillEndpoint().offerQueued(route);
        controller.wake();
        return accepted;
    }

    private static final java.util.Map<EndpointWork, Controller> controllers =
            java.util.Collections.synchronizedMap(new java.util.IdentityHashMap<>());

    public static void endpointChanged(PrefillEndpoint endpoint, PrefillEndpoint.Signal kind) {
        var work = org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(endpoint);
        if (work == null) return;
        if (kind != PrefillEndpoint.Signal.PLACEMENT)
            work.signal(kind == PrefillEndpoint.Signal.CAPACITY);
        var controller = controllers.get(work);
        if (controller != null) controller.wake();
    }

    static QueuedRequestScheduler.EndpointWork schedulingState(EndpointWork execution) {
        var controller = controllers.get(execution);
        return controller == null ? null : controller.work;
    }

    public static EndpointWork.Decision schedule(EndpointWork work) {
        return QueuedRequestScheduler.decideExecution(work);
    }

    private static final class Controller implements RequestScheduler {
        final EndpointWork execution;
        final RequestLifecycle owner;
        final QueuedRequestScheduler.EndpointWork work;
        final java.util.function.Function<EndpointWork, EndpointWork.Decision> policy;
        final Thread thread;
        volatile boolean stopped;
        final java.util.concurrent.atomic.AtomicLong notifications =
                new java.util.concurrent.atomic.AtomicLong();
        volatile boolean active;
        final java.util.concurrent.atomic.AtomicBoolean started =
                new java.util.concurrent.atomic.AtomicBoolean();
        EndpointWork.Decision waiting;
        long next = Long.MAX_VALUE;

        Controller(
                EndpointWork execution,
                java.util.function.Function<EndpointWork, EndpointWork.Decision> policy,
                RequestLifecycle owner) {
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

        public void onRequest(RequestContext request) {}

        public void onRequestChanged(RequestContext request) {
            wake();
        }

        public void onEndpointChanged(org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {
            wake();
        }

        public void close() {
            stop();
            awaitStopped(Long.MAX_VALUE);
        }
    }

    public static PrefillEndpoint.QueueSnapshot capture(EndpointWork runtime) {
        return state(runtime).captureQueue(Integer.MAX_VALUE);
    }

    public static PrefillEndpoint state(EndpointWork runtime) {
        return org.flexlb.balance.endpoint.EndpointTestSupport.resources(runtime.prefillEndpoint());
    }
}
